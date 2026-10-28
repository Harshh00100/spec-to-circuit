"""Trap-mode tests: fake model, fake simulator, fake verifier (the doubles from test_loop).

They prove the RULES of trap mode: the design is built by Python (no model call for it), it is
simulated and verified like any other, the verdict comes only from the verifier, and the page is told
whether the trap was caught, passed or unverified. They do not prove anything about ngspice or a model;
the live check is `POST /api/runs` with {"trap": true} on your PC.
"""
import tempfile
import unittest

from app.llm import FakeLLM
from app.loop import run_spec, trap_result
from app.netlists import L_RANGE_UM, W_RANGE_UM, estimate_delay_s, inverter_netlist
from app.runs import RunManager
from app.schemas import Criterion, OverallStatus, SimulationResult, Verdict
from app.trap import MIN_SLOWDOWN, TrapError, plan_trap
from tests.test_loop import FakeSimulator, fake_verify, intent_reply
from app.units import unit_info

TIMEOUT = 10


def tpd(threshold, unit, op="<"):
    return Criterion(name="tpd", metric="tpd", operator=op, threshold=threshold, unit=unit, measurable=True)


def go_trap(replies=None, sim=None, text="CMOS inverter, delay under 40 ps at 20 fF", **kw):
    llm = FakeLLM(replies if replies is not None else [intent_reply()])
    sim = sim or FakeSimulator()
    events = []
    run = run_spec(text, llm, simulate=sim, verify=fake_verify, trap=True, on_event=events.append, **kw)
    return run, llm, sim, events


class AlwaysFast:
    """A simulator that reports a delay far under any limit (the trap design would 'pass')."""
    def __init__(self):
        self.calls = []

    def __call__(self, netlist):
        self.calls.append(netlist)
        return SimulationResult(ok=True, returncode=0,
                                measurements={"tpd": 1e-12, "tphl": 1e-12, "tplh": 1e-12, "voh": 1.8, "vol": 0.0})


class Planner(unittest.TestCase):
    def test_design_is_inside_the_template_and_clearly_too_slow(self):
        for threshold, unit, cload, vdd in ((40, "ps", 20, 1.8), (100, "ps", 20, 1.8), (0.1, "ns", 50, 1.8),
                                            (200, "ps", 100, 3.3), (300, "ps", 20, 1.2)):
            with self.subTest(threshold=threshold, unit=unit, cload=cload, vdd=vdd):
                plan = plan_trap([tpd(threshold, unit)], cload, vdd)
                for size in plan.design.transistors.values():
                    self.assertGreaterEqual(size.w_um, W_RANGE_UM[0])
                    self.assertLessEqual(size.w_um, W_RANGE_UM[1])
                    self.assertGreaterEqual(size.l_um, L_RANGE_UM[0])
                    self.assertLessEqual(size.l_um, L_RANGE_UM[1])
                limit_s = threshold * unit_info(unit)[1]
                mn, mp = plan.design.transistors["mn"], plan.design.transistors["mp"]
                est = estimate_delay_s(mn.w_um, mn.l_um, mp.w_um, mp.l_um, cload, vdd)
                self.assertGreaterEqual(est, MIN_SLOWDOWN * limit_s)
                inverter_netlist(plan.design, vdd=vdd, cload_ff=cload)    # the real builder accepts it

    def test_the_claim_pretends_to_beat_the_limit(self):
        plan = plan_trap([tpd(40, "ps")], 20, 1.8)
        self.assertEqual((plan.claim_metric, plan.claim_unit, plan.threshold), ("tpd", "ps", 40.0))
        self.assertLess(plan.claim_value, plan.threshold)
        self.assertIn("COMPLETE", plan.claim_text)
        self.assertEqual(plan.design.rationale, plan.claim_text)
        self.assertLessEqual(len(plan.design.rationale), 300)
        self.assertEqual(plan.event_payload()["claimed"], plan.claim_value)

    def test_the_first_usable_delay_limit_is_used(self):
        voh = Criterion(name="voh", metric="voh", operator=">", threshold=1.7, unit="V", measurable=True)
        plan = plan_trap([voh, Criterion(name="low power", measurable=False), tpd(40, "ps")], 20, 1.8)
        self.assertEqual(plan.claim_metric, "tpd")

    def test_no_delay_limit_means_no_trap(self):
        voh = Criterion(name="voh", metric="voh", operator=">", threshold=1.7, unit="V", measurable=True)
        for criteria in ([], [voh], [Criterion(name="low power", measurable=False)], [tpd(40, "ps", op=">")]):
            with self.subTest(criteria=criteria):
                with self.assertRaises(TrapError):
                    plan_trap(criteria, 20, 1.8)

    def test_a_limit_too_relaxed_for_the_allowed_sizes_is_refused_not_faked(self):
        with self.assertRaises(TrapError):
            plan_trap([tpd(2, "ns")], 20, 1.8)

    def test_an_unreachable_limit_is_refused_because_any_design_would_fail_it(self):
        for threshold, unit, cload in ((1, "ps", 20), (5, "ps", 20), (0.001, "ns", 20), (10, "ps", 100)):
            with self.subTest(threshold=threshold, unit=unit, cload=cload):
                with self.assertRaises(TrapError) as ctx:
                    plan_trap([tpd(threshold, unit)], cload, 1.8)
                self.assertIn("too tight", str(ctx.exception))

    def test_a_limit_a_good_design_can_reach_is_still_trapped(self):
        for threshold, unit, cload in ((20, "ps", 20), (40, "ps", 20), (100, "ps", 50)):
            with self.subTest(threshold=threshold, unit=unit, cload=cload):
                plan_trap([tpd(threshold, unit)], cload, 1.8)

    def test_a_voltage_unit_on_a_delay_is_refused(self):
        with self.assertRaises(TrapError):
            plan_trap([tpd(40, "mV")], 20, 1.8)
        with self.assertRaises(TrapError):
            plan_trap([tpd(40, "bogus")], 20, 1.8)


class InTheLoop(unittest.TestCase):
    def test_the_verifier_catches_the_false_claim(self):
        run, llm, sim, events = go_trap()
        self.assertEqual(run.status, OverallStatus.FAILED)
        self.assertEqual(len(run.iterations), 1)
        self.assertEqual(len(sim.calls), 1)
        verdict = run.iterations[0].report.verdicts[0]
        self.assertEqual(verdict.verdict, Verdict.FAIL)
        self.assertGreater(verdict.measured, 40)                      # measured, not claimed
        self.assertIs(run.final_report, run.iterations[0].report)
        self.assertEqual(run.status, run.final_report.overall)

    def test_no_model_call_is_spent_on_the_design(self):
        run, llm, _, _ = go_trap()
        self.assertEqual(len(llm.requests), 1)                       # the intent call only
        self.assertEqual([c.purpose for c in run.llm_calls], ["intent"])

    def test_events_carry_the_claim_and_the_outcome(self):
        _, _, _, events = go_trap()
        self.assertEqual([e["type"] for e in events], ["intent", "conditions", "design", "iteration", "done"])
        design, iteration = events[2], events[3]
        self.assertEqual(design["index"], 1)
        self.assertEqual(design["trap"]["metric"], "tpd")
        self.assertLess(design["trap"]["claimed"], design["trap"]["threshold"])
        self.assertEqual(design["design"]["rationale"], design["trap"]["text"])
        self.assertEqual(iteration["trap_result"], "caught")
        self.assertEqual(iteration["overall"], "FAILED")
        self.assertEqual(events[-1]["status"], "FAILED")

    def test_a_trap_that_happens_to_pass_is_not_called_caught(self):
        run, _, _, events = go_trap(sim=AlwaysFast())
        self.assertEqual(events[3]["trap_result"], "passed")
        self.assertEqual(run.status, OverallStatus.COMPLETE)         # the verifier measured a pass; Python reports it

    def test_a_simulator_crash_leaves_the_trap_unverified(self):
        run, _, _, events = go_trap(sim=FakeSimulator(crash=True))
        self.assertEqual(events[3]["trap_result"], "unverified")
        self.assertEqual(run.status, OverallStatus.INCOMPLETE)

    def test_no_delay_limit_runs_nothing(self):
        reqs = [{"text": "output high above 1.7 V", "metric": "voh", "operator": ">", "threshold": 1.7, "unit": "V"}]
        run, llm, sim, events = go_trap([intent_reply(requirements=reqs)])
        self.assertEqual(run.status, OverallStatus.INCOMPLETE)
        self.assertEqual(sim.calls, [])
        self.assertEqual(run.iterations, [])
        self.assertIsNotNone(run.final_report)
        errors = [e for e in events if e["type"] == "error"]
        self.assertEqual(errors[0]["stage"], "trap")
        self.assertEqual(events[-1]["type"], "done")

    def test_an_impossible_limit_runs_no_simulation(self):
        reqs = [{"text": "propagation delay under 1 ps", "metric": "tpd", "operator": "<", "threshold": 1, "unit": "ps"}]
        run, _, sim, events = go_trap([intent_reply(requirements=reqs)])
        self.assertEqual(run.status, OverallStatus.INCOMPLETE)
        self.assertEqual(sim.calls, [])
        error = [e for e in events if e["type"] == "error"][0]
        self.assertEqual(error["stage"], "trap")
        self.assertIn("too tight", error["message"])

    def test_an_unsupported_circuit_is_never_trapped(self):
        run, _, sim, _ = go_trap([intent_reply(circuit_type="op-amp")], text="Two-stage op-amp, 60 dB gain")
        self.assertEqual(run.status, OverallStatus.INCOMPLETE)
        self.assertEqual(sim.calls, [])

    def test_the_stated_load_and_supply_still_apply(self):
        run, _, sim, _ = go_trap(
            [intent_reply(parameters={"load": "50 fF", "supply": "3.3 V"},
                          requirements=[{"text": "delay under 100 ps", "metric": "tpd",
                                         "operator": "<", "threshold": 100, "unit": "ps"}])],
            text="CMOS inverter, propagation delay under 100 ps at 50 fF load, 3.3 V supply")
        self.assertIn("cl out 0 50f", sim.calls[0])
        self.assertIn("vsupply vdd 0 3.3", sim.calls[0])

    def test_a_normal_run_has_no_trap_fields(self):
        from tests.test_loop import design_reply
        events = []
        run_spec("CMOS inverter, delay under 40 ps at 20 fF", FakeLLM([intent_reply(), design_reply(1.2)]),
                 simulate=FakeSimulator(), verify=fake_verify, on_event=events.append)
        self.assertNotIn("trap", events[2])
        self.assertNotIn("trap_result", events[3])

    def test_the_exact_netlist_is_sent_with_the_design(self):
        _, _, sim, events = go_trap()
        self.assertEqual(events[2]["netlist"], sim.calls[0])         # what ngspice got is what the page shows

    def test_a_normal_run_sends_the_netlist_too(self):
        from tests.test_loop import design_reply
        events = []
        sim = FakeSimulator()
        run_spec("CMOS inverter, delay under 40 ps at 20 fF", FakeLLM([intent_reply(), design_reply(1.2)]),
                 simulate=sim, verify=fake_verify, on_event=events.append)
        self.assertEqual(events[2]["netlist"], sim.calls[0])

    def test_unrelated_conditions_still_block_complete_in_a_trap_run(self):
        run, _, _, _ = go_trap([intent_reply(parameters={"load": "20 fF", "temperature": "85 C"})], sim=AlwaysFast())
        self.assertEqual(run.status, OverallStatus.INCOMPLETE)       # the unmodelled condition is UNVERIFIED


class TrapResultHelper(unittest.TestCase):
    def report(self, sim):
        return fake_verify([tpd(40, "ps")], sim)

    def test_reads_only_the_verifier_report(self):
        slow = SimulationResult(ok=True, measurements={"tpd": 90e-12})
        fast = SimulationResult(ok=True, measurements={"tpd": 10e-12})
        gone = SimulationResult(ok=False, error="boom")
        self.assertEqual(trap_result(self.report(slow), "tpd", 40.0, "ps"), "caught")
        self.assertEqual(trap_result(self.report(fast), "tpd", 40.0, "ps"), "passed")
        self.assertEqual(trap_result(self.report(gone), "tpd", 40.0, "ps"), "unverified")
        self.assertEqual(trap_result(self.report(slow), "tphl", 40.0, "ps"), "unverified")


class WithTheRunManager(unittest.TestCase):
    def manager(self, replies, history_dir=None):
        return RunManager(simulate=FakeSimulator(), verify=fake_verify, llm=FakeLLM(replies), history_dir=history_dir)

    def test_the_flag_reaches_the_loop_and_the_summary(self):
        m = self.manager([intent_reply()])
        h = m.start("CMOS inverter, delay under 40 ps at 20 fF", trap=True)
        self.assertTrue(h.wait(TIMEOUT))
        self.assertTrue(h.trap)
        self.assertTrue(h.summary()["trap"])
        self.assertEqual(h.overall, "FAILED")
        self.assertTrue(any(e.get("trap") for e in h.events if e["type"] == "design"))
        self.assertTrue(m.list()[0]["trap"])

    def test_default_is_not_a_trap(self):
        from tests.test_loop import design_reply
        m = self.manager([intent_reply(), design_reply(1.2)])
        h = m.start("x")
        self.assertTrue(h.wait(TIMEOUT))
        self.assertFalse(h.trap)
        self.assertEqual(h.overall, "COMPLETE")

    def test_the_flag_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = self.manager([intent_reply()], tmp).start("CMOS inverter, delay under 40 ps at 20 fF", trap=True)
            self.assertTrue(h.wait(TIMEOUT))
            fresh = self.manager([], tmp)
            row = [r for r in fresh.list() if r["run_id"] == h.id][0]
            self.assertTrue(row["trap"])
            self.assertTrue(fresh.load_saved(h.id)["trap"])


if __name__ == "__main__":
    unittest.main()
