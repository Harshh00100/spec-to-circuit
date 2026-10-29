"""Benchmark: a plain LLM-only answer versus this system, on the same model and the same specs.

    python -m benchmark.run_benchmark              # first 3 specs (slow on a CPU)
    python -m benchmark.run_benchmark --limit 5    # all 5
    python -m benchmark.run_benchmark --only fast40,impossible1ps

Needs NGSPICE_BIN and LLM_MODEL set, like `python -m app.run`.

What is compared
  Baseline  The model gets the requirement and is asked, in ONE reply, for transistor sizes, its own
            estimate of the delay, and whether the spec is met. No tools, no feedback, no checks.
            Its claim is then scored against a real ngspice run of the sizes it chose (the model never
            sees that result). Sizes outside the template ranges are NOT clamped: that would help it.
  System    The normal pipeline (`run_spec`): intent, design, ngspice, verifier, repair loop.
            Its verdict comes from the verifier. As a cross-check, the final simulated delay is
            compared with the limit written in this file, independent of the compiled criteria.

What the numbers mean
  This measures how often each side is RIGHT ABOUT whether the spec is met, and how often it
  actually produces a design that meets it. One run per spec on one small model is an anecdote,
  not statistics: report it as such.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.llm import LLMClient, LLMError, client_from_env
from app.loop import run_spec
from app.netlists import L_RANGE_UM, W_RANGE_UM, inverter_netlist
from app.schemas import DesignProposal, SimulationResult, TransistorSize
from app.simulator import run_ngspice
from app.verifier import verify


@dataclass(frozen=True)
class Spec:
    id: str
    text: str
    tpd_limit_s: float      # the real limit, used only to score both sides
    cload_ff: float
    vdd: float
    note: str


# Order matters: --limit 3 takes the first three.
SPECS: list[Spec] = [
    Spec("fast40", "CMOS inverter, propagation delay under 40 ps at 20 fF load", 40e-12, 20.0, 1.8,
         "tight but reachable; needs repair"),
    Spec("impossible1ps", "CMOS inverter, propagation delay under 1 ps at 20 fF load", 1e-12, 20.0, 1.8,
         "unreachable; the honest answer is FAILED"),
    Spec("load50", "CMOS inverter, propagation delay under 0.1 ns driving 50 fF", 100e-12, 50.0, 1.8,
         "larger load"),
    Spec("easy2ns", "CMOS inverter, propagation delay under 2 ns at 20 fF load", 2e-9, 20.0, 1.8,
         "easy"),
    Spec("big100ff", "CMOS inverter, propagation delay under 200 ps at 100 fF load, 3.3 V supply",
         200e-12, 100.0, 3.3, "large load and a stated supply"),
]


# --------------------------------------------------------------------------
# Baseline: one plain model answer
# --------------------------------------------------------------------------
class _Size(BaseModel):
    model_config = ConfigDict(extra="ignore")
    w_um: float = Field(allow_inf_nan=False)
    l_um: float = Field(allow_inf_nan=False)


class BaselineAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    mn: _Size
    mp: _Size
    expected_delay_ps: float | None = None
    meets_spec: bool
    rationale: str | None = ""


BASELINE_SYSTEM = (
    "You are a circuit design assistant. Design a CMOS inverter for the user's requirement by choosing "
    "the width and length of the NMOS (mn) and the PMOS (mp), in micrometres. "
    f"Allowed range: w_um {W_RANGE_UM[0]:g} to {W_RANGE_UM[1]:g}, l_um {L_RANGE_UM[0]:g} to {L_RANGE_UM[1]:g}. "
    "Also give your own estimate of the propagation delay in picoseconds and say whether your design "
    "meets the requirement. Reply as JSON: "
    '{"mn": {"w_um": number, "l_um": number}, "mp": {"w_um": number, "l_um": number}, '
    '"expected_delay_ps": number, "meets_spec": true or false, "rationale": short text}.'
)


def run_baseline(spec: Spec, llm: LLMClient, simulate: Callable[[str], SimulationResult]) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {"outcome": None, "claimed_met": None, "claimed_delay_ps": None,
                           "measured_ps": None, "actually_met": None, "model_calls": 1, "detail": ""}
    try:
        answer, call = llm.complete_json(
            "baseline", [{"role": "system", "content": BASELINE_SYSTEM}, {"role": "user", "content": spec.text}],
            BaselineAnswer)
    except LLMError as exc:
        row.update(outcome="no_answer", detail=str(exc)[:200])
        row["model_s"] = row["total_s"] = round(time.perf_counter() - started, 1)
        return row
    row["model_s"] = round(call.latency_s, 1)
    row["claimed_met"] = answer.meets_spec
    row["claimed_delay_ps"] = answer.expected_delay_ps
    row["sizes"] = {"mn": [answer.mn.w_um, answer.mn.l_um], "mp": [answer.mp.w_um, answer.mp.l_um]}
    try:
        design = DesignProposal(
            circuit_type="inverter", stage_count=1,
            transistors={"mn": TransistorSize(w_um=answer.mn.w_um, l_um=answer.mn.l_um),
                         "mp": TransistorSize(w_um=answer.mp.w_um, l_um=answer.mp.l_um)})
        netlist = inverter_netlist(design, vdd=spec.vdd, cload_ff=spec.cload_ff)   # raises on out-of-range
    except ValueError as exc:
        row.update(outcome="invalid_design", actually_met=False, detail=str(exc)[:200])
        row["total_s"] = round(time.perf_counter() - started, 1)
        return row
    sim = simulate(netlist)
    tpd = sim.measurements.get("tpd") if sim.ok else None
    if tpd is None:
        row.update(outcome="not_simulated", detail=(sim.error or "no tpd measurement")[:200])
    else:
        met = tpd < spec.tpd_limit_s
        row.update(measured_ps=round(tpd * 1e12, 2), actually_met=met)
        if answer.meets_spec == met:
            row["outcome"] = "claim_correct"
        elif answer.meets_spec:
            row["outcome"] = "false_claim"        # said "met", the simulation says it is not
        else:
            row["outcome"] = "missed_pass"        # said "not met", but it is
    row["total_s"] = round(time.perf_counter() - started, 1)
    return row


# --------------------------------------------------------------------------
# System: the normal pipeline
# --------------------------------------------------------------------------
def run_system(spec: Spec, llm: LLMClient, simulate: Callable[[str], SimulationResult],
               verify_fn: Callable = verify) -> dict[str, Any]:
    started = time.perf_counter()
    report = run_spec(spec.text, llm, simulate=simulate, verify=verify_fn)
    row: dict[str, Any] = {"status": report.status.value, "iterations": len(report.iterations),
                           "model_calls": len(report.llm_calls),
                           "model_s": round(sum(c.latency_s for c in report.llm_calls), 1),
                           "measured_ps": None, "actually_met": None, "outcome": None}
    last = report.iterations[-1].simulation if report.iterations else None
    tpd = last.measurements.get("tpd") if last is not None and last.ok else None
    if tpd is not None:
        row["measured_ps"] = round(tpd * 1e12, 2)
        row["actually_met"] = tpd < spec.tpd_limit_s
    if report.status.value == "COMPLETE":
        # cross-check against the limit in this file, not the compiled criteria
        row["outcome"] = "verified_and_met" if row["actually_met"] else "WRONG_COMPLETE"
    elif report.status.value == "FAILED":
        row["outcome"] = "failed_honestly" if not row["actually_met"] else "failed_but_met"
    else:
        row["outcome"] = "incomplete"
    row["total_s"] = round(time.perf_counter() - started, 1)
    return row


# --------------------------------------------------------------------------
# Scoring and output
# --------------------------------------------------------------------------
def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    b = [r["baseline"] for r in results]
    s = [r["system"] for r in results]
    return {
        "specs": n,
        "baseline_designs_that_meet_spec": sum(1 for r in b if r["actually_met"]),
        "baseline_false_claims": sum(1 for r in b if r["outcome"] == "false_claim"),
        "baseline_invalid_or_no_answer": sum(1 for r in b if r["outcome"] in ("invalid_design", "no_answer", "not_simulated")),
        "system_complete_and_met": sum(1 for r in s if r["outcome"] == "verified_and_met"),
        "system_wrong_complete": sum(1 for r in s if r["outcome"] == "WRONG_COMPLETE"),
        "system_failed_honestly": sum(1 for r in s if r["outcome"] == "failed_honestly"),
        "system_incomplete": sum(1 for r in s if r["outcome"] == "incomplete"),
    }


def _fmt(value: Any, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value}{suffix}"


def to_markdown(results: list[dict[str, Any]], model: str, when: str) -> str:
    lines = [f"# Benchmark: plain LLM answer vs Spec-to-Circuit", "",
             f"Date {when}. Model `{model}`. ngspice Level-1 inverter. One run per spec.", "",
             "| Spec | Baseline claimed | Baseline real delay | Baseline outcome | System verdict | System real delay | Attempts | System outcome |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        b, s = r["baseline"], r["system"]
        claimed = "met" if b["claimed_met"] else ("not met" if b["claimed_met"] is False else "n/a")
        lines.append(f"| {r['spec']['text']} | {claimed} | {_fmt(b['measured_ps'], ' ps')} | {b['outcome']} | "
                     f"{s['status']} | {_fmt(s['measured_ps'], ' ps')} | {s['iterations']} | {s['outcome']} |")
    t = summarize(results)
    lines += ["", "## Totals", "",
              f"- Specs run: {t['specs']}",
              f"- Baseline: {t['baseline_designs_that_meet_spec']} design(s) actually met the spec; "
              f"{t['baseline_false_claims']} time(s) it said the spec was met when the simulation says it was not; "
              f"{t['baseline_invalid_or_no_answer']} unusable answer(s).",
              f"- System: {t['system_complete_and_met']} COMPLETE with a measured pass; "
              f"{t['system_wrong_complete']} wrong COMPLETE; {t['system_failed_honestly']} FAILED correctly; "
              f"{t['system_incomplete']} INCOMPLETE.",
              "", "## How to read this", "",
              "- The baseline's claim is scored against a real ngspice run of the sizes it chose; it never saw that result.",
              "- The system's verdict comes from the verifier. \"Wrong COMPLETE\" means COMPLETE although the final simulated delay misses the limit; it should be 0.",
              "- One run per spec on a small local model is an anecdote, not a statistic. Results change from run to run.",
              "- The baseline gets the allowed size ranges but no feedback. Out-of-range sizes are counted as invalid, not clamped.",
              "- Both sides use the same simulator, the same Level-1 models, and the stated load and supply."]
    return "\n".join(lines) + "\n"


def run_all(specs: list[Spec], llm: LLMClient, simulate: Callable[[str], SimulationResult],
            verify_fn: Callable = verify, log: Callable[[str], None] = lambda m: None) -> list[dict[str, Any]]:
    results = []
    for i, spec in enumerate(specs, 1):
        log(f"[{i}/{len(specs)}] {spec.text}")
        log("   baseline ...")
        baseline = run_baseline(spec, llm, simulate)
        log(f"   baseline: {baseline['outcome']}  claimed_met={baseline['claimed_met']}  measured={baseline['measured_ps']} ps")
        log("   system ...")
        system = run_system(spec, llm, simulate, verify_fn)
        log(f"   system: {system['status']} after {system['iterations']} attempt(s)  "
            f"measured={system['measured_ps']} ps  ({system['total_s']} s)")
        results.append({"spec": {"id": spec.id, "text": spec.text, "limit_ps": spec.tpd_limit_s * 1e12,
                                 "cload_ff": spec.cload_ff, "vdd": spec.vdd, "note": spec.note},
                        "baseline": baseline, "system": system})
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=3, help="how many specs to run (default 3; max %d)" % len(SPECS))
    ap.add_argument("--only", default="", help="comma-separated spec ids: " + ", ".join(s.id for s in SPECS))
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"), help="folder for the results")
    args = ap.parse_args(argv)

    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        unknown = [w for w in wanted if w not in {s.id for s in SPECS}]
        if unknown:
            print(f"unknown spec id(s): {unknown}", file=sys.stderr)
            return 2
        specs = [s for s in SPECS if s.id in wanted]
    else:
        specs = SPECS[: max(1, args.limit)]
    if not os.environ.get("NGSPICE_BIN"):
        print("NGSPICE_BIN is not set; the simulator may not be found.", file=sys.stderr)

    llm = client_from_env()
    started = time.perf_counter()
    results = run_all(specs, llm, run_ngspice, verify, log=lambda m: print(m, flush=True))
    when = date.today().isoformat()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"benchmark-{when}.json").write_text(
        json.dumps({"date": when, "model": llm.model, "results": results, "summary": summarize(results)}, indent=2),
        encoding="utf-8")
    md = to_markdown(results, llm.model, when)
    (out / f"benchmark-{when}.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Saved to {out}  (total {time.perf_counter() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
