"""Pure verifier tests: no ngspice needed. Proves PASS / FAIL / UNVERIFIED logic."""
import math
import unittest

from pydantic import ValidationError

from app.schemas import Criterion, OverallStatus, SimulationResult, Verdict
from app.verifier import verify, verify_criterion


def crit(metric="tpd", op="<", threshold=150.0, unit="ps", name=None, **kw):
    return Criterion(
        name=name or f"{metric} {op} {threshold}{unit}",
        metric=metric,
        operator=op,
        threshold=threshold,
        unit=unit,
        measurable=True,
        **kw,
    )


def sim(**measurements):
    return SimulationResult(ok=True, measurements=measurements)


class VerdictLogic(unittest.TestCase):
    def test_pass_with_margin(self):
        v = verify_criterion(crit(threshold=150, unit="ps"), sim(tpd=105e-12))
        self.assertEqual(v.verdict, Verdict.PASS)
        self.assertAlmostEqual(v.measured, 105.0)
        self.assertAlmostEqual(v.margin, 45.0)
        self.assertAlmostEqual(v.margin_pct, 30.0)

    def test_fail_has_negative_margin(self):
        v = verify_criterion(crit(threshold=100, unit="ps"), sim(tpd=130e-12))
        self.assertEqual(v.verdict, Verdict.FAIL)
        self.assertAlmostEqual(v.margin, -30.0)

    def test_greater_than_direction(self):
        c = crit(metric="voh", op=">", threshold=1.7, unit="V")
        self.assertEqual(verify_criterion(c, sim(voh=1.79)).verdict, Verdict.PASS)
        self.assertEqual(verify_criterion(c, sim(voh=1.6)).verdict, Verdict.FAIL)
        self.assertAlmostEqual(verify_criterion(c, sim(voh=1.6)).margin, -0.1)

    def test_strict_vs_inclusive_boundary(self):
        self.assertEqual(
            verify_criterion(crit(op="<", threshold=100, unit="s"), sim(tpd=100.0)).verdict,
            Verdict.FAIL,
        )
        self.assertEqual(
            verify_criterion(crit(op="<=", threshold=100, unit="s"), sim(tpd=100.0)).verdict,
            Verdict.PASS,
        )

    def test_unit_conversion_ns_vs_seconds(self):
        # 0.2 ns requirement vs 150 ps measured -> PASS only if units are normalised
        v = verify_criterion(crit(threshold=0.2, unit="ns"), sim(tpd=150e-12))
        self.assertEqual(v.verdict, Verdict.PASS)
        v = verify_criterion(crit(threshold=0.1, unit="ns"), sim(tpd=150e-12))
        self.assertEqual(v.verdict, Verdict.FAIL)

    def test_metric_lookup_is_case_insensitive(self):
        v = verify_criterion(crit(metric="TPD"), SimulationResult(ok=True, measurements={"Tpd": 1e-10}))
        self.assertEqual(v.verdict, Verdict.PASS)


class UnverifiedLogic(unittest.TestCase):
    def test_non_measurable_is_unverified(self):
        c = Criterion(name="low noise", measurable=False)
        v = verify_criterion(c, sim(tpd=1e-10))
        self.assertEqual(v.verdict, Verdict.UNVERIFIED)
        self.assertTrue(v.reason)

    def test_missing_metric_is_unverified(self):
        v = verify_criterion(crit(metric="gain_db", unit="dB", op=">", threshold=20), sim(tpd=1e-10))
        self.assertEqual(v.verdict, Verdict.UNVERIFIED)

    def test_failed_measure_is_unverified_even_if_value_present(self):
        s = SimulationResult(ok=True, measurements={"tpd": 1e-10}, failed_measures=["tpd"])
        self.assertEqual(verify_criterion(crit(), s).verdict, Verdict.UNVERIFIED)

    def test_simulator_failure_means_no_verdicts_at_all(self):
        s = SimulationResult(ok=False, error="ngspice timed out", measurements={"tpd": 1e-12})
        report = verify([crit(), crit(metric="voh", op=">", threshold=1.0, unit="V")], s)
        self.assertTrue(all(v.verdict == Verdict.UNVERIFIED for v in report.verdicts))
        self.assertFalse(report.evidence_ok)
        self.assertEqual(report.overall, OverallStatus.INCOMPLETE)

    def test_nan_measurement_is_unverified(self):
        v = verify_criterion(crit(), sim(tpd=math.nan))
        self.assertEqual(v.verdict, Verdict.UNVERIFIED)

    def test_unknown_unit_is_unverified(self):
        v = verify_criterion(crit(unit="parsecs"), sim(tpd=1e-10))
        self.assertEqual(v.verdict, Verdict.UNVERIFIED)

    def test_unit_dimension_mismatch_is_unverified(self):
        v = verify_criterion(crit(metric="tpd", unit="V", threshold=1.0), sim(tpd=1e-10))
        self.assertEqual(v.verdict, Verdict.UNVERIFIED)
        self.assertIn("mismatch", v.reason)


class OverallStatusLogic(unittest.TestCase):
    def test_complete_only_when_all_pass(self):
        r = verify([crit(threshold=150), crit(metric="voh", op=">", threshold=1.7, unit="V")],
                   sim(tpd=100e-12, voh=1.8))
        self.assertEqual(r.overall, OverallStatus.COMPLETE)
        self.assertEqual(r.counts, {"PASS": 2, "FAIL": 0, "UNVERIFIED": 0})
        self.assertEqual(r.unresolved, [])

    def test_any_fail_means_failed(self):
        r = verify([crit(threshold=150), crit(metric="voh", op=">", threshold=1.7, unit="V")],
                   sim(tpd=100e-12, voh=1.0))
        self.assertEqual(r.overall, OverallStatus.FAILED)

    def test_fail_dominates_unverified(self):
        unverifiable = Criterion(name="looks nice", measurable=False)
        r = verify([crit(threshold=50), unverifiable], sim(tpd=100e-12))
        self.assertEqual(r.overall, OverallStatus.FAILED)

    def test_unverified_blocks_completion(self):
        unverifiable = Criterion(name="low power", measurable=False)
        r = verify([crit(threshold=150), unverifiable], sim(tpd=100e-12))
        self.assertEqual(r.overall, OverallStatus.INCOMPLETE)
        self.assertEqual(r.unresolved, ["low power"])

    def test_no_criteria_is_never_complete(self):
        self.assertEqual(verify([], sim(tpd=1e-10)).overall, OverallStatus.INCOMPLETE)


class SchemaValidation(unittest.TestCase):
    def test_measurable_criterion_needs_threshold(self):
        with self.assertRaises(ValidationError):
            Criterion(name="x", metric="tpd", operator="<", measurable=True)

    def test_measurable_criterion_needs_metric(self):
        with self.assertRaises(ValidationError):
            Criterion(name="x", operator="<", threshold=1.0, measurable=True)

    def test_bad_operator_rejected(self):
        with self.assertRaises(ValidationError):
            Criterion(name="x", metric="tpd", operator="~=", threshold=1.0, measurable=True)

    def test_non_measurable_gets_default_reason(self):
        self.assertTrue(Criterion(name="nice", measurable=False).unverifiable_reason)

    def test_extra_fields_rejected(self):
        with self.assertRaises(ValidationError):
            Criterion(name="x", measurable=False, surprise=1)


if __name__ == "__main__":
    unittest.main()
