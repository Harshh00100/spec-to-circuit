"""Design proposer: the LLM chooses transistor sizes, Python does everything else.

What the model may decide: mn / mp width and length (inside the allowed ranges).
What it may NOT decide: the supply, the load and every other condition. Those come from
the user's requirement (IntentSpec.parameters) and are applied by Python
(conditions_from_intent). Otherwise a model could "meet" a delay target by quietly
choosing a tiny load.

If a stated condition cannot be honoured (unusable value, unknown knob), it is reported
in Conditions.unapplied and unapplied_criteria() turns each one into an UNVERIFIABLE
criterion, so the final status can never be COMPLETE for a problem we did not simulate.

Out-of-range sizes are clamped into range and the clamp is reported in `notes`; the
clamped design is the one that gets simulated, so the verdict is about what really ran.

Try it (needs Ollama; two model calls, so it is slow on CPU):
    python -m app.design "CMOS inverter, propagation delay under 2 ns at 20 fF load"
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .intent import LOAD_KEYS, SUPPLY_KEYS, SUPPORTED_CIRCUITS
from .llm import LLMClient, LLMError, client_from_env
from .netlists import (
    CLOAD_RANGE_FF,
    DEFAULT_CLOAD_FF,
    DEFAULT_VDD,
    L_RANGE_UM,
    VDD_RANGE_V,
    W_RANGE_UM,
    inverter_netlist,
)
from .schemas import Criterion, DesignProposal, IntentSpec, LLMCallRecord, TransistorSize
from .units import UnitError, unit_info

MAX_FEEDBACK_CHARS = 1500
MAX_RATIONALE_CHARS = 300


class DesignError(ValueError):
    """The design stage cannot proceed (nothing to design for, or no template)."""


# --------------------------------------------------------------------------
# Conditions (applied by Python, never chosen by the model)
# --------------------------------------------------------------------------
@dataclass
class Conditions:
    cload_ff: float = DEFAULT_CLOAD_FF
    vdd: float = DEFAULT_VDD
    applied: list[str] = field(default_factory=list)    # stated by the user and used
    assumed: list[str] = field(default_factory=list)    # not stated; defaults used
    unapplied: list[str] = field(default_factory=list)  # stated but NOT honoured


_QUANTITY = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-z\u00b5\u03bc]*)\s*$")


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _fix_case(unit: str, base: str) -> str:
    """'pf' -> 'pF', 'mv' -> 'mV' (models often get the case of the base unit wrong)."""
    return unit[: -len(base)] + base if unit.lower().endswith(base.lower()) else unit


def _quantity(value: Any, dimension: str, base: str, bare_means_base: bool) -> float | None:
    """Value in SI units, or None if it is not a usable quantity of the given dimension."""
    if isinstance(value, bool):
        return None
    unit = ""
    try:
        if isinstance(value, (int, float)):
            number = float(value)
        elif isinstance(value, str):
            match = _QUANTITY.match(value)
            if not match:
                return None
            number, unit = float(match.group(1)), match.group(2)
        else:
            return None
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if unit == "":
        return number if bare_means_base else None
    for candidate in (unit, _fix_case(unit, base)):
        try:
            found_dimension, factor = unit_info(candidate)
        except UnitError:
            continue
        if found_dimension == dimension:
            return number * factor
    return None


def conditions_from_intent(spec: IntentSpec) -> Conditions:
    cond = Conditions()
    seen_load = seen_supply = False
    for raw_key, value in spec.parameters.items():
        key, shown = _key(str(raw_key)), f"{raw_key} = {value}"
        if key in LOAD_KEYS:
            if seen_load:
                cond.unapplied.append(f"{shown}: a load was already given")
                continue
            seen_load = True
            farads = _quantity(value, "capacitance", "F", bare_means_base=False)
            femto = None if farads is None else farads * 1e15
            low, high = CLOAD_RANGE_FF
            if femto is None or not low <= femto <= high:
                cond.unapplied.append(
                    f"{shown}: needs a capacitance with a unit between {low:g} fF and "
                    f"{high:g} fF; simulated at the default {DEFAULT_CLOAD_FF:g} fF instead"
                )
            else:
                cond.cload_ff = femto
                cond.applied.append(f"load capacitance = {femto:g} fF")
        elif key in SUPPLY_KEYS:
            if seen_supply:
                cond.unapplied.append(f"{shown}: a supply was already given")
                continue
            seen_supply = True
            volts = _quantity(value, "voltage", "V", bare_means_base=True)
            low, high = VDD_RANGE_V
            if volts is None or not low <= volts <= high:
                cond.unapplied.append(
                    f"{shown}: needs a voltage between {low:g} V and {high:g} V; "
                    f"simulated at the default {DEFAULT_VDD:g} V instead"
                )
            else:
                cond.vdd = volts
                cond.applied.append(f"supply = {volts:g} V")
        else:
            cond.unapplied.append(f"{shown}: not modelled by the inverter template")
    if not seen_load:
        cond.assumed.append(f"load capacitance not stated; assumed {DEFAULT_CLOAD_FF:g} fF")
    if not seen_supply:
        cond.assumed.append(f"supply not stated; assumed {DEFAULT_VDD:g} V")
    return cond


def unapplied_criteria(cond: Conditions) -> list[Criterion]:
    """One UNVERIFIABLE criterion per stated condition that was not honoured (blocks COMPLETE)."""
    return [
        Criterion(
            name=f"condition applied: {text.split(':', 1)[0][:100]}",
            measurable=False,
            unverifiable_reason=text,
        )
        for text in cond.unapplied
    ]


# --------------------------------------------------------------------------
# What the model fills in
# --------------------------------------------------------------------------
class RawSize(BaseModel):
    model_config = ConfigDict(extra="ignore")
    w_um: float = Field(allow_inf_nan=False)
    l_um: float = Field(allow_inf_nan=False)


class RawDesign(BaseModel):
    model_config = ConfigDict(extra="ignore")
    mn: RawSize
    mp: RawSize
    rationale: str | None = ""


_EXAMPLE = {
    "mn": {"w_um": 0.6, "l_um": 0.18},
    "mp": {"w_um": 1.2, "l_um": 0.18},
    "rationale": "PMOS is about twice as wide as NMOS to balance rise and fall delay.",
}

SYSTEM_PROMPT = (
    "You size the two transistors of a CMOS inverter: mn (NMOS) and mp (PMOS). "
    'Reply as JSON: {"mn": {"w_um": number, "l_um": number}, '
    '"mp": {"w_um": number, "l_um": number}, "rationale": short text}.\n'
    f"Allowed range: w_um {W_RANGE_UM[0]:g} to {W_RANGE_UM[1]:g}, "
    f"l_um {L_RANGE_UM[0]:g} to {L_RANGE_UM[1]:g} (micrometres). "
    "Values outside the range are clamped.\n"
    "The supply and load are fixed by the user; you cannot change them.\n"
    "Hints: a larger w_um and a smaller l_um give more drive current, so less delay, but "
    "use more area. The PMOS is weaker than the NMOS, so mp is usually about 2x wider than mn.\n"
    "If feedback from earlier attempts is given, change the sizes to fix what failed.\n"
    f"Example JSON: {json.dumps(_EXAMPLE)}\n"
)


@dataclass
class DesignResult:
    design: DesignProposal
    call: LLMCallRecord
    notes: list[str] = field(default_factory=list)  # clamps applied (for the UI)
    raw: dict[str, Any] = field(default_factory=dict)  # what the model actually said


def _describe_target(c: Criterion) -> str:
    return f"{c.metric} {c.operator} {c.threshold:g} {c.unit}"


def _clamp(label: str, value: float, bounds: tuple[float, float], notes: list[str]) -> float:
    low, high = bounds
    clamped = min(max(value, low), high)
    if clamped != value:
        notes.append(f"{label} {value:g} clamped to {clamped:g} (allowed {low:g} to {high:g})")
    return clamped


def propose_design(
    spec: IntentSpec,
    conditions: Conditions,
    llm: LLMClient | None = None,
    feedback: str | None = None,
) -> DesignResult:
    """Ask the model for transistor sizes. Raises DesignError if there is nothing to design for,
    LLMError / LLMOutputError if the model cannot be reached or keeps returning bad JSON."""
    if spec.circuit_type not in SUPPORTED_CIRCUITS:
        raise DesignError(f"no design template for circuit type '{spec.circuit_type}'")
    targets = [_describe_target(c) for c in spec.criteria if c.measurable]
    if not targets:
        raise DesignError("no measurable criteria, so there is nothing to design against")

    user = (
        "Targets (all must be met):\n"
        + "\n".join(f"- {t}" for t in targets)
        + f"\nFixed conditions: load {conditions.cload_ff:g} fF, supply {conditions.vdd:g} V."
    )
    if feedback:
        user += "\nFeedback from earlier attempts:\n" + feedback.strip()[:MAX_FEEDBACK_CHARS]

    client = llm if llm is not None else client_from_env()
    raw, record = client.complete_json(
        "design",
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        RawDesign,
    )

    notes: list[str] = []
    sizes = {}
    for name, size in (("mn", raw.mn), ("mp", raw.mp)):
        sizes[name] = TransistorSize(
            w_um=_clamp(f"{name}.w_um", size.w_um, W_RANGE_UM, notes),
            l_um=_clamp(f"{name}.l_um", size.l_um, L_RANGE_UM, notes),
        )
    design = DesignProposal(
        circuit_type="inverter",
        stage_count=1,
        transistors=sizes,
        rationale=(raw.rationale or "").strip()[:MAX_RATIONALE_CHARS],
    )
    return DesignResult(design=design, call=record, notes=notes, raw=raw.model_dump())


# --------------------------------------------------------------------------
# CLI: requirement -> intent -> conditions -> design -> netlist (no simulation)
# --------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    from .intent import compile_intent, is_supported

    parser = argparse.ArgumentParser(prog="python -m app.design")
    parser.add_argument("text", help="the circuit requirement, in quotes")
    args = parser.parse_args(argv)
    try:
        client = client_from_env()
        intent = compile_intent(args.text, client)
        print(intent.spec.understood_summary)
        if not is_supported(intent.spec):
            print("error: no template for this circuit type", file=sys.stderr)
            return 1
        cond = conditions_from_intent(intent.spec)
        result = propose_design(intent.spec, cond, client)
    except (LLMError, DesignError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for label, items in (("applied", cond.applied), ("assumed", cond.assumed),
                         ("NOT applied", cond.unapplied), ("note", result.notes)):
        for item in items:
            print(f"{label}: {item}")
    print(result.design.model_dump_json(indent=2))
    print(inverter_netlist(result.design, vdd=cond.vdd, cload_ff=cond.cload_ff))
    print(result.call.model_dump_json())
    return 0


if __name__ == "__main__":
    sys.exit(_main())
