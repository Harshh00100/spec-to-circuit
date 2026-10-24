"""Repair loop: requirement text -> intent -> (design -> simulate -> verify) x up to N.

The one rule everything here serves: the final status comes ONLY from the verifier's report.
This module never sets COMPLETE itself, never reads the model's rationale as evidence, and
never counts a criterion it could not measure as met.

  * simulate and verify are passed in (simulate(netlist) -> SimulationResult,
    verify(criteria, SimulationResult) -> VerificationReport), so the loop is testable without
    ngspice and the real modules are wired in one place.
  * Criteria checked = the intent's criteria + one unverifiable criterion per stated condition
    that could not be honoured (see design.unapplied_criteria). Those can never pass, so the run
    cannot end COMPLETE for a problem that was not actually simulated.
  * The loop retries only when a redesign could help: some criterion FAILed, or the simulation
    produced no evidence (one retry for that). Criteria that are merely unverifiable stop the loop.
  * Feedback to the model is built by Python from measured numbers. It never repeats the model's
    own text, and it carries a rule-of-thumb sizing hint computed from the measured/target ratio.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .design import (
    DesignError,
    conditions_from_intent,
    propose_design,
    unapplied_criteria,
)
from .intent import compile_intent, is_supported
from .llm import LLMClient, LLMError, client_from_env
from .netlists import W_RANGE_UM, L_RANGE_UM, inverter_netlist
from .schemas import (
    Criterion,
    DesignProposal,
    IterationRecord,
    LLMCallRecord,
    OverallStatus,
    RunReport,
    SimulationResult,
    Verdict,
    VerificationReport,
)

MAX_ITERATIONS = 4
MAX_FEEDBACK_ATTEMPTS = 3
FEEDBACK_SAFETY_MARGIN = 1.25  # aim a little past the target so one step is enough
_TIME_METRICS = frozenset({"tpd", "tphl", "tplh"})
_LEVEL_METRICS = frozenset({"voh", "vol"})

Simulate = Callable[[str], SimulationResult]
Verify = Callable[[list[Criterion], SimulationResult], VerificationReport]


@dataclass
class Attempt:
    design: DesignProposal
    sim: SimulationResult
    report: VerificationReport


# --------------------------------------------------------------------------
# Feedback for the model (Python-written, from measurements only)
# --------------------------------------------------------------------------
def _hint(attempt: Attempt) -> str | None:
    if not attempt.report.evidence_ok:
        return "Use sizes inside the allowed range; widths of a few um are the safest."
    worst = 0.0
    level_fail = False
    for v in attempt.report.verdicts:
        c = v.criterion
        if v.verdict != Verdict.FAIL or not c.measurable:
            continue
        if c.metric in _LEVEL_METRICS:
            level_fail = True
        elif (c.metric in _TIME_METRICS and c.operator in ("<", "<=") and v.measured
              and c.threshold and v.measured > 0 and c.threshold > 0):
            worst = max(worst, v.measured / c.threshold)
    hints: list[str] = []
    if worst > 1.0:
        mn, mp = attempt.design.transistors["mn"], attempt.design.transistors["mp"]
        low, high = W_RANGE_UM
        if mn.w_um >= high and mp.w_um >= high:
            hints.append("Both widths are already at the allowed maximum; this target may not be reachable.")
        else:
            scale = worst * FEEDBACK_SAFETY_MARGIN
            new_mn = min(max(mn.w_um * scale, low), high)
            new_mp = min(max(mp.w_um * scale, low), high)
            hints.append(
                f"Hint: delay falls roughly in proportion to width. The measured delay is {worst:.2f}x "
                f"the target, so try widths about {scale:.2f}x larger (mn w_um about {new_mn:.3g}, "
                f"mp w_um about {new_mp:.3g}) with l_um {L_RANGE_UM[0]:g}."
            )
    if level_fail:
        hints.append("Output levels voh/vol are set by the supply and load, not by transistor sizes.")
    return " ".join(hints) or None


def build_feedback(attempts: list[Attempt], max_attempts: int = MAX_FEEDBACK_ATTEMPTS) -> str:
    """Compact text for the next design prompt. Contains only designs and measured results."""
    shown = attempts[-max_attempts:]
    first_number = len(attempts) - len(shown) + 1
    lines: list[str] = []
    for number, attempt in enumerate(shown, start=first_number):
        mn, mp = attempt.design.transistors["mn"], attempt.design.transistors["mp"]
        head = f"Attempt {number}: mn w={mn.w_um:g} l={mn.l_um:g}, mp w={mp.w_um:g} l={mp.l_um:g} (um)"
        if not attempt.report.evidence_ok:
            reason = (attempt.sim.error or "no measurements")[:120]
            lines.append(f"{head} -> the simulation gave no usable result ({reason})")
            continue
        parts: list[str] = []
        for v in attempt.report.verdicts:
            c = v.criterion
            if not c.measurable:
                continue
            if v.measured is None:
                parts.append(f"{c.metric} not measured")
            else:
                parts.append(
                    f"{c.metric} {v.measured:.3g} {c.unit} (needs {c.operator} {c.threshold:g} {c.unit}) "
                    f"{v.verdict.value}"
                )
        lines.append(f"{head} -> " + "; ".join(parts))
    hint = _hint(attempts[-1])
    if hint:
        lines.append(hint)
    return "\n".join(lines)[:1400]


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def _failed_call(purpose: str, llm: LLMClient | None, exc: Exception) -> LLMCallRecord:
    model = str(getattr(llm, "model", "") or "unknown")
    return LLMCallRecord(purpose=purpose, model=model, latency_s=0.0, ok=False, error=str(exc)[:300])


def _emitter(on_event: Callable[[dict[str, Any]], None] | None) -> Callable[[dict[str, Any]], None]:
    def emit(event: dict[str, Any]) -> None:
        if on_event is None:
            return
        try:
            on_event(event)
        except Exception:  # a broken UI callback must never change a verification run
            pass

    return emit


def run_spec(
    text: str,
    llm: LLMClient | None = None,
    *,
    simulate: Simulate,
    verify: Verify,
    max_iterations: int = MAX_ITERATIONS,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> RunReport:
    run = RunReport(run_id=uuid.uuid4().hex[:8], spec_text=text if isinstance(text, str) else "")
    emit = _emitter(on_event)

    def no_evidence_report(criteria: list[Criterion], why: str) -> VerificationReport:
        return verify(criteria, SimulationResult(ok=False, error=f"no simulation was run: {why}"))

    def finish() -> RunReport:
        if run.final_report is not None:
            run.status = run.final_report.overall
        emit({"type": "done", "status": run.status.value,
              "iterations": len(run.iterations), "run_id": run.run_id})
        return run

    # 1. intent
    try:
        client = llm if llm is not None else client_from_env()
        intent = compile_intent(text, client)
    except (LLMError, ValueError) as exc:
        run.llm_calls.append(_failed_call("intent", llm, exc))
        emit({"type": "error", "stage": "intent", "message": str(exc)})
        return finish()
    spec = intent.spec
    run.intent = spec
    run.llm_calls.append(intent.call)
    emit({"type": "intent", "spec": spec.model_dump(mode="json"), "notes": intent.notes})

    # 2. conditions (applied by Python; unhonoured ones become unverifiable criteria)
    cond = conditions_from_intent(spec)
    criteria = list(spec.criteria) + unapplied_criteria(cond)
    emit({"type": "conditions", "cload_ff": cond.cload_ff, "vdd": cond.vdd, "applied": cond.applied,
          "assumed": cond.assumed, "unapplied": cond.unapplied})

    # 3. nothing to design or simulate?
    reason = None
    if not is_supported(spec):
        reason = f"no design template for circuit type '{spec.circuit_type}'"
    elif not any(c.measurable for c in spec.criteria):
        reason = "no measurable criteria to design against"
    if reason:
        run.final_report = no_evidence_report(criteria, reason)
        emit({"type": "error", "stage": "design", "message": reason})
        return finish()

    # 4. design -> simulate -> verify, repeated while a redesign could help
    attempts: list[Attempt] = []
    feedback: str | None = None
    evidence_failures = 0
    for index in range(1, max_iterations + 1):
        try:
            proposal = propose_design(spec, cond, client, feedback)
            netlist = inverter_netlist(proposal.design, vdd=cond.vdd, cload_ff=cond.cload_ff)
        except (LLMError, DesignError, ValueError) as exc:
            run.llm_calls.append(_failed_call("design", client, exc))
            emit({"type": "error", "stage": "design", "index": index, "message": str(exc)})
            break
        run.llm_calls.append(proposal.call)
        shape = proposal.design.model_dump(mode="json", exclude={"rationale"})
        if any(shape == a.design.model_dump(mode="json", exclude={"rationale"}) for a in attempts):
            # the model ignored the feedback and repeated a design: simulating it again proves nothing new
            emit({"type": "error", "stage": "design", "index": index,
                  "message": "the model repeated an earlier design; stopping instead of re-simulating it"})
            break
        emit({"type": "design", "index": index, "design": proposal.design.model_dump(mode="json"),
              "notes": proposal.notes})

        try:
            sim = simulate(netlist)
        except Exception as exc:  # any simulator failure is "no evidence", never a pass
            sim = SimulationResult(ok=False, error=f"simulator raised {type(exc).__name__}: {exc}"[:300])
        report = verify(criteria, sim)
        run.iterations.append(IterationRecord(index=index, design=proposal.design, simulation=sim, report=report))
        attempts.append(Attempt(proposal.design, sim, report))
        emit({"type": "iteration", "index": index, "overall": report.overall.value,
              "report": report.model_dump(mode="json")})

        if report.overall == OverallStatus.COMPLETE:
            break
        has_fail = any(v.verdict == Verdict.FAIL for v in report.verdicts)
        # a measurable criterion left UNVERIFIED (or no simulation at all) may be caused by the design,
        # e.g. an edge that never crosses the threshold, so it earns one retry, not an endless loop
        no_evidence = (not report.evidence_ok) or any(
            v.verdict == Verdict.UNVERIFIED and v.criterion.measurable for v in report.verdicts
        )
        evidence_failures = evidence_failures + 1 if no_evidence else 0
        if not has_fail and not no_evidence:
            break  # only non-measurable criteria remain; a redesign cannot fix that
        if evidence_failures >= 2:
            break  # still no evidence; do not burn more model calls
        feedback = build_feedback(attempts)

    run.final_report = (
        run.iterations[-1].report if run.iterations
        else no_evidence_report(criteria, "the design step failed before any simulation")
    )
    return finish()
