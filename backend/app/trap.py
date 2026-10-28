"""Trap mode: hand the verifier a deliberately wrong design that CLAIMS success.

The point of the demo: a proposer's words are not evidence. In a trap run the model is not asked
for a design. Python builds one that is slower than the requested delay limit (transistors that are
too narrow and too long), attaches a confident false claim to it ("expected tpd about 20 ps, spec
met, COMPLETE"), and sends it down the normal path: real netlist, real ngspice, real verifier. The
verifier never reads the claim, so it reports FAIL with the measured number.

Rules this module keeps:
  * It only builds designs inside the template's allowed ranges, so the netlist builder accepts them.
  * It never decides a verdict. Whether the trap was "caught" is read from the verifier's report by
    loop.py, not from anything here.
  * If it cannot build a design that is clearly slower than the limit (for example the limit is very
    relaxed, or there is no delay limit at all) it raises TrapError and the run says so. It never
    pretends a trap ran.
"""
from __future__ import annotations

from dataclasses import dataclass

from .netlists import L_RANGE_UM, W_RANGE_UM, estimate_delay_s
from .schemas import Criterion, DesignProposal, TransistorSize
from .units import UnitError, unit_info

TRAP_SLOWDOWN = 3.0     # aim for an estimated delay this many times the limit (the estimate runs low)
REALISM_FACTOR = 8.0    # ngspice measured about 5 to 7x the estimate for very wide devices (parasitics)
MIN_SLOWDOWN = 2.0      # refuse to run a "trap" that is not clearly over the limit by the estimate
CLAIM_FRACTION = 0.5    # the false claim says the delay is this fraction of the limit
_LENGTHS_UM = (0.5, 1.0, 2.0, 0.18)   # tried in this order; a long channel looks like a plausible mistake
_TIME_METRICS = ("tpd", "tphl", "tplh")


class TrapError(ValueError):
    """A trap design cannot be built for this requirement."""


@dataclass
class TrapPlan:
    design: DesignProposal
    claim_metric: str
    claim_value: float          # in claim_unit
    claim_unit: str
    threshold: float            # the limit the claim pretends to beat, in claim_unit
    operator: str
    claim_text: str
    estimated_delay_s: float    # Python's rough estimate for the trap design (not evidence)

    def event_payload(self) -> dict:
        """What the UI is told about the trap (a claim, never a result)."""
        return {
            "metric": self.claim_metric,
            "claimed": self.claim_value,
            "unit": self.claim_unit,
            "threshold": self.threshold,
            "operator": self.operator,
            "text": self.claim_text,
        }


def _pick_criterion(criteria: list[Criterion]) -> Criterion:
    for c in criteria:
        if (c.measurable and c.metric in _TIME_METRICS and c.operator in ("<", "<=")
                and c.threshold is not None and c.threshold > 0):
            return c
    raise TrapError(
        "trap mode needs a delay limit to be wrong about, such as 'propagation delay under 2 ns'"
    )


def plan_trap(criteria: list[Criterion], cload_ff: float, vdd: float) -> TrapPlan:
    """Build the wrong design and its false claim for the first measurable delay limit."""
    target = _pick_criterion(criteria)
    try:
        dimension, factor = unit_info(target.unit)
    except UnitError as exc:
        raise TrapError(f"cannot use the unit of '{target.name}': {exc}") from None
    if dimension != "time":
        raise TrapError(f"'{target.name}' is not a time limit")
    limit_s = float(target.threshold) * factor   # type: ignore[arg-type]
    if limit_s <= 0:
        raise TrapError("the delay limit must be positive")

    # A limit no allowed design could meet is not a trap: any design "fails" it, so the verifier would
    # prove nothing. The estimate runs low for very wide devices, so it is scaled up before comparing.
    fastest_s = estimate_delay_s(W_RANGE_UM[1] / 2, L_RANGE_UM[0], W_RANGE_UM[1], L_RANGE_UM[0], cload_ff, vdd)
    floor_s = fastest_s * REALISM_FACTOR
    if floor_s > limit_s:
        raise TrapError(
            f"this limit is too tight for a meaningful trap: even the largest allowed transistors would "
            f"take roughly {floor_s * 1e12:.2g} ps at this load and supply, so no design could meet it. "
            f"Use a limit that a good design can reach, such as 'under 40 ps'"
        )

    # Delay is inversely proportional to W/L for a fixed load and supply, so one reference
    # evaluation gives the W/L needed for the wanted (too slow) delay. mp is 2x mn, as a
    # reasonable-looking design would have it; the mistake is only in the overall size.
    reference_s = estimate_delay_s(1.0, 1.0, 2.0, 1.0, cload_ff, vdd)   # mn w/l = 1, mp w/l = 2
    wanted_ratio = reference_s / (TRAP_SLOWDOWN * limit_s)              # mn w/l that gives that delay
    w_low, w_high = W_RANGE_UM
    l_low, l_high = L_RANGE_UM
    chosen: tuple[float, float] | None = None
    for length in _LENGTHS_UM:
        if not l_low <= length <= l_high:
            continue
        width = round(wanted_ratio * length, 3)
        if w_low <= width and 2 * width <= w_high:
            chosen = (width, length)
            break
    if chosen is None:
        raise TrapError(
            "this delay limit is too relaxed (or too tight) to build a clearly wrong design inside "
            "the allowed transistor sizes; try a tighter limit such as 'under 100 ps'"
        )
    wn, length = chosen
    wp = round(2 * wn, 3)
    estimate = estimate_delay_s(wn, length, wp, length, cload_ff, vdd)
    if estimate < MIN_SLOWDOWN * limit_s:
        raise TrapError("could not build a design that is clearly slower than the limit")

    claim_value = float(f"{target.threshold * CLAIM_FRACTION:.3g}")   # type: ignore[operator]
    claim_text = (
        f"Sized for a fast edge. Expected {target.metric} about {claim_value:g} {target.unit}, "
        f"well inside the {target.threshold:g} {target.unit} limit. Spec met, status COMPLETE."
    )
    design = DesignProposal(
        circuit_type="inverter",
        stage_count=1,
        transistors={"mn": TransistorSize(w_um=wn, l_um=length),
                     "mp": TransistorSize(w_um=wp, l_um=length)},
        rationale=claim_text,
    )
    return TrapPlan(
        design=design, claim_metric=target.metric, claim_value=claim_value, claim_unit=target.unit,
        threshold=float(target.threshold), operator=str(target.operator),   # type: ignore[arg-type]
        claim_text=claim_text, estimated_delay_s=estimate,
    )
