"""Deterministic verifier. The LLM never decides pass/fail; this module does.

Rules (conservative on purpose):
  * A criterion is PASS or FAIL only if the simulator produced a finite
    measurement for exactly that metric and the units are consistent.
  * Anything else -- not measurable, simulator did not run, measurement
    failed or missing, unknown unit, unit/metric mismatch -- is UNVERIFIED.
  * The overall run is COMPLETE only if every criterion is PASS. A single
    FAIL makes it FAILED; otherwise any UNVERIFIED makes it INCOMPLETE.
"""
from __future__ import annotations

import math
import operator
from typing import Sequence

from .schemas import (
    Criterion,
    CriterionVerdict,
    OverallStatus,
    SimulationResult,
    Verdict,
    VerificationReport,
)
from .units import UnitError, metric_dimension, unit_info

_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def _unverified(criterion: Criterion, reason: str) -> CriterionVerdict:
    return CriterionVerdict(criterion=criterion, verdict=Verdict.UNVERIFIED, reason=reason)


def verify_criterion(criterion: Criterion, sim: SimulationResult) -> CriterionVerdict:
    if not criterion.measurable:
        return _unverified(
            criterion, criterion.unverifiable_reason or "criterion is not measurable"
        )
    if not sim.ok:
        return _unverified(
            criterion, f"no simulator evidence: {sim.error or 'simulation did not complete'}"
        )
    if criterion.operator is None or criterion.threshold is None:
        return _unverified(criterion, "criterion has no operator or threshold")

    metric = criterion.metric.strip().lower()
    failed = {m.lower() for m in sim.failed_measures}
    if metric in failed:
        return _unverified(criterion, f"simulator could not evaluate measurement '{metric}'")

    measurements = {k.lower(): v for k, v in sim.measurements.items()}
    raw = measurements.get(metric)
    if raw is None:
        return _unverified(criterion, f"simulator produced no measurement named '{metric}'")
    if not math.isfinite(raw):
        return _unverified(criterion, f"measurement '{metric}' is not a finite number")

    try:
        dimension, factor = unit_info(criterion.unit)
    except UnitError as exc:
        return _unverified(criterion, str(exc))
    expected = metric_dimension(metric)
    if expected is not None and expected != dimension:
        return _unverified(
            criterion,
            f"unit mismatch: metric '{metric}' is a {expected} but unit "
            f"'{criterion.unit}' is a {dimension}",
        )

    threshold_si = criterion.threshold * factor
    passed = _OPS[criterion.operator](raw, threshold_si)
    if criterion.operator in ("<", "<="):
        margin_si = threshold_si - raw
    else:
        margin_si = raw - threshold_si

    measured = raw / factor
    margin = margin_si / factor
    margin_pct = (margin_si / abs(threshold_si) * 100.0) if threshold_si != 0 else None
    unit = criterion.unit
    reason = (
        f"measured {measured:.4g} {unit}; requirement {criterion.operator} "
        f"{criterion.threshold:.4g} {unit}; margin {margin:+.4g} {unit}"
    ).replace("  ", " ")

    return CriterionVerdict(
        criterion=criterion,
        verdict=Verdict.PASS if passed else Verdict.FAIL,
        measured=measured,
        margin=margin,
        margin_pct=margin_pct,
        reason=reason,
    )


def overall_status(verdicts: Sequence[CriterionVerdict]) -> OverallStatus:
    if any(v.verdict == Verdict.FAIL for v in verdicts):
        return OverallStatus.FAILED
    if not verdicts or any(v.verdict == Verdict.UNVERIFIED for v in verdicts):
        return OverallStatus.INCOMPLETE
    return OverallStatus.COMPLETE


def verify(criteria: Sequence[Criterion], sim: SimulationResult) -> VerificationReport:
    verdicts = [verify_criterion(c, sim) for c in criteria]
    counts = {v.value: 0 for v in Verdict}
    for item in verdicts:
        counts[item.verdict.value] += 1
    return VerificationReport(
        verdicts=verdicts,
        overall=overall_status(verdicts),
        counts=counts,
        evidence_ok=sim.ok,
    )
