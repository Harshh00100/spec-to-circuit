"""Runner plumbing tests against a FAKE ngspice (see fake_ngspice.py).

These cover timeout, exit codes, parsing, log-file reading and netlist
rejection. They do not prove the circuit simulates correctly.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from app.netlists import hardcoded_inverter_netlist
from app.schemas import Criterion, Verdict
from app.simulator import declared_measures, parse_measurements, run_ngspice
from app.verifier import verify

FAKE = [sys.executable, str(Path(__file__).parent / "fake_ngspice.py")]
NETLIST = hardcoded_inverter_netlist()


def run(mode, **kw):
    with mock.patch.dict(os.environ, {"FAKE_NGSPICE_MODE": mode}):
        return run_ngspice(NETLIST, ngspice_bin=FAKE, **kw)


class Parsing(unittest.TestCase):
    def test_declared_measures_found(self):
        self.assertEqual(declared_measures(NETLIST), ["tphl", "tplh", "tpd", "voh", "vol"])

    def test_only_declared_names_accepted(self):
        values, failed = parse_measurements("tpd = 1e-10 targ= 1\nfoo = 5\n", ["tpd", "voh"])
        self.assertEqual(values, {"tpd": 1e-10})
        self.assertEqual(failed, ["voh"])

    def test_failed_and_nan_are_failed(self):
        values, failed = parse_measurements("tpd = failed\nvoh = nan\nvol = 0.1\n", ["tpd", "voh", "vol"])
        self.assertEqual(values, {"vol": 0.1})
        self.assertEqual(failed, ["tpd", "voh"])


class Runner(unittest.TestCase):
    def test_ok_run_parses_measurements(self):
        r = run("ok")
        self.assertTrue(r.ok, r.error)
        self.assertAlmostEqual(r.measurements["tpd"], 1.1e-10)
        self.assertEqual(r.failed_measures, [])
        self.assertEqual(set(r.measurements), {"tphl", "tplh", "tpd", "voh", "vol"})

    def test_measurements_read_from_log_file_too(self):
        r = run("log_only")
        self.assertTrue(r.ok, r.error)
        self.assertIn("tpd", r.measurements)

    def test_failed_measure_reported_and_verifier_marks_unverified(self):
        r = run("fail_measure")
        self.assertTrue(r.ok)
        self.assertIn("tphl", r.failed_measures)
        c = Criterion(name="tphl", metric="tphl", operator="<", threshold=1, unit="ns", measurable=True)
        self.assertEqual(verify([c], r).verdicts[0].verdict, Verdict.UNVERIFIED)

    def test_timeout_is_not_ok(self):
        r = run("sleep", timeout_s=1.0)
        self.assertFalse(r.ok)
        self.assertIn("timed out", r.error)
        self.assertLess(r.runtime_s, 15)

    def test_nonzero_exit_is_not_ok(self):
        r = run("crash")
        self.assertFalse(r.ok)
        self.assertEqual(r.returncode, 3)

    def test_no_measure_output_is_not_ok(self):
        r = run("silent")
        self.assertFalse(r.ok)
        self.assertIn("no .measure results", r.error)

    def test_missing_binary_is_not_ok(self):
        r = run_ngspice(NETLIST, ngspice_bin="/nonexistent/ngspice")
        self.assertFalse(r.ok)
        self.assertIn("could not start", r.error)

    def test_netlist_without_measures_rejected(self):
        r = run_ngspice("* nothing\n.end\n", ngspice_bin=FAKE)
        self.assertFalse(r.ok)

    def test_shell_directives_rejected(self):
        bad = NETLIST.replace(".end\n", ".control\nshell echo pwned\n.endc\n.end\n")
        r = run_ngspice(bad, ngspice_bin=FAKE)
        self.assertFalse(r.ok)
        self.assertIn("rejected", r.error)


if __name__ == "__main__":
    unittest.main()
