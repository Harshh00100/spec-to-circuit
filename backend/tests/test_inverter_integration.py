"""Real ngspice end-to-end test: hardcoded inverter -> simulate -> verify.

Skipped automatically if ngspice is not installed. Thresholds are derived from
the measured value so the test checks verifier behaviour against REAL simulator
output without depending on exact model numbers.
"""
import math
import os
import shutil
import unittest

from app.netlists import hardcoded_inverter_netlist
from app.schemas import Criterion, OverallStatus, Verdict
from app.simulator import run_ngspice
from app.verifier import verify

NGSPICE = os.environ.get("NGSPICE_BIN") or shutil.which("ngspice")


def tpd_criterion(threshold_ps):
    return Criterion(
        name=f"tpd < {threshold_ps:.2f} ps", metric="tpd", operator="<",
        threshold=threshold_ps, unit="ps", measurable=True,
    )


@unittest.skipUnless(NGSPICE, "ngspice not installed -- integration test skipped")
class InverterEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strong = run_ngspice(hardcoded_inverter_netlist(wn_um=2.0, wp_um=4.0, cload_ff=20))
        cls.weak = run_ngspice(hardcoded_inverter_netlist(wn_um=0.2, wp_um=0.4, cload_ff=50))

    def test_simulation_succeeds_with_all_measurements(self):
        self.assertTrue(self.strong.ok, self.strong.error)
        self.assertEqual(self.strong.failed_measures, [])
        for name in ("tphl", "tplh", "tpd", "voh", "vol"):
            self.assertIn(name, self.strong.measurements)

    def test_physical_sanity(self):
        m = self.strong.measurements
        self.assertTrue(1e-12 < m["tpd"] < 2e-9, m["tpd"])
        self.assertGreater(m["voh"], 0.95 * 1.8)
        self.assertLess(abs(m["vol"]), 0.05)
        self.assertAlmostEqual(m["tpd"], (m["tphl"] + m["tplh"]) / 2, delta=m["tpd"] * 0.01)

    def test_weaker_design_is_slower(self):
        self.assertGreater(self.weak.measurements["tpd"], self.strong.measurements["tpd"])

    def test_pass_when_threshold_is_loose(self):
        tpd_ps = self.strong.measurements["tpd"] * 1e12
        report = verify([tpd_criterion(tpd_ps * 3)], self.strong)
        self.assertEqual(report.verdicts[0].verdict, Verdict.PASS)
        self.assertGreater(report.verdicts[0].margin, 0)
        self.assertEqual(report.overall, OverallStatus.COMPLETE)

    def test_fail_when_threshold_is_impossible(self):
        tpd_ps = self.strong.measurements["tpd"] * 1e12
        report = verify([tpd_criterion(tpd_ps * 0.3)], self.strong)
        self.assertEqual(report.verdicts[0].verdict, Verdict.FAIL)
        self.assertLess(report.verdicts[0].margin, 0)
        self.assertEqual(report.overall, OverallStatus.FAILED)

    def test_same_spec_passes_strong_and_fails_weak(self):
        t_strong = self.strong.measurements["tpd"] * 1e12
        t_weak = self.weak.measurements["tpd"] * 1e12
        spec = [tpd_criterion(math.sqrt(t_strong * t_weak))]
        self.assertEqual(verify(spec, self.strong).overall, OverallStatus.COMPLETE)
        self.assertEqual(verify(spec, self.weak).overall, OverallStatus.FAILED)


if __name__ == "__main__":
    unittest.main()
