"""Repair-loop tests with a fake simulator and a fake verifier (no ngspice, no Ollama).

The fakes are test doubles: fake_simulate returns tpd = 33.5 ps*um / Wn (scaled by load), fake_verify
implements the documented PASS/FAIL/UNVERIFIED rules. Wiring to the real simulator.py and verifier.py
is a separate step, so these tests prove the LOOP's rules, not the real modules.
"""
import re
import unittest

from app.loop import Attempt, build_feedback, run_spec
from app.llm import FakeLLM
from app.netlists import inverter_netlist
from app.schemas import (
    Criterion,
    CriterionVerdict,
    OverallStatus,
    SimulationResult,
    Verdict,
    VerificationReport,
)
from app.units import unit_info
from app.schemas import DesignProposal, TransistorSize


# ------------------------------- test doubles -------------------------------
class FakeSimulator:
    def __init__(self, crash=False):
        self.calls = []
        self.crash = crash

    def __call__(self, netlist):
        self.calls.append(netlist)
        if self.crash:
            raise RuntimeError("ngspice exploded")
        wn = float(re.search(r"^mn .* w=(\S+)u l=", netlist, re.M).group(1))
        cload_ff = float(re.search(r"^cl out 0 (\S+)f", netlist, re.M).group(1))
        tpd = 33.5e-12 / wn * (cload_ff / 20.0)
        return SimulationResult(
            ok=True, returncode=0,
            measurements={"tpd": tpd, "tphl": tpd, "tplh": tpd, "voh": 1.8, "vol": 0.0})


def fake_verify(criteria, sim):
    verdicts = []
    for c in criteria:
        if not c.measurable:
            verdicts.append(CriterionVerdict(criterion=c, verdict=Verdict.UNVERIFIED,
                                             reason=c.unverifiable_reason or ""))
            continue
        value = sim.measurements.get(c.metric) if sim.ok else None
        if value is None:
            verdicts.append(CriterionVerdict(criterion=c, verdict=Verdict.UNVERIFIED, reason="no measurement"))
            continue
        _, factor = unit_info(c.unit)
        measured = value / factor
        passed = {"<": measured < c.threshold, "<=": measured <= c.threshold,
                  ">": measured > c.threshold, ">=": measured >= c.threshold}[c.operator]
        margin = (c.threshold - measured) if c.operator in ("<", "<=") else (measured - c.threshold)
        verdicts.append(CriterionVerdict(
            criterion=c, verdict=Verdict.PASS if passed else Verdict.FAIL, measured=measured, margin=margin))
    kinds = [v.verdict for v in verdicts]
    if Verdict.FAIL in kinds:
        overall = OverallStatus.FAILED
    elif not kinds or Verdict.UNVERIFIED in kinds:
        overall = OverallStatus.INCOMPLETE
    else:
        overall = OverallStatus.COMPLETE
    counts = {k.value: kinds.count(k) for k in Verdict}
    return VerificationReport(verdicts=verdicts, overall=overall, counts=counts, evidence_ok=sim.ok)


def intent_reply(parameters=None, circuit_type="inverter", requirements=None):
    return {
        "circuit_type": circuit_type,
        "parameters": {"load": "20 fF"} if parameters is None else parameters,
        "requirements": requirements if requirements is not None else [
            {"text": "propagation delay under 40 ps", "metric": "tpd", "operator": "<",
             "threshold": 40, "unit": "ps"}],
        "questions": [],
    }


def design_reply(wn, wp=None, l=0.18, rationale="balanced"):
    return {"mn": {"w_um": wn, "l_um": l}, "mp": {"w_um": wp or 2 * wn, "l_um": l},
            "rationale": rationale}


def design(wn, wp=None, l=0.18):
    return DesignProposal(
        circuit_type="inverter", stage_count=1,
        transistors={"mn": TransistorSize(w_um=wn, l_um=l), "mp": TransistorSize(w_um=wp or 2 * wn, l_um=l)})


def go(replies, text="CMOS inverter, delay under 40 ps at 20 fF", sim=None, **kw):
    llm = FakeLLM(replies)
    sim = sim or FakeSimulator()
    report = run_spec(text, llm, simulate=sim, verify=fake_verify, **kw)
    return report, llm, sim


# ------------------------------------ tests ----------------------------------
class Outcomes(unittest.TestCase):
    def test_first_design_passes(self):
        run, llm, sim = go([intent_reply(), design_reply(1.2)])
        self.assertEqual(run.status, OverallStatus.COMPLETE)
        self.assertEqual(len(run.iterations), 1)
        self.assertEqual(len(llm.requests), 2)          # intent + one design, nothing more
        self.assertEqual([c.purpose for c in run.llm_calls], ["intent", "design"])
        self.assertEqual(len(sim.calls), 1)
        self.assertIs(run.final_report, run.iterations[-1].report)

    def test_failure_is_repaired_using_measured_feedback(self):
        run, llm, _ = go([intent_reply(), design_reply(0.45), design_reply(1.05)])
        self.assertEqual(run.status, OverallStatus.COMPLETE)
        self.assertEqual([r.report.overall for r in run.iterations],
                         [OverallStatus.FAILED, OverallStatus.COMPLETE])
        second_prompt = llm.requests[2][-1]["content"]
        self.assertIn("Attempt 1", second_prompt)
        self.assertIn("FAIL", second_prompt)
        self.assertIn("74.4 ps", second_prompt)          # measured, from the simulator
        self.assertIn("1.05", second_prompt)             # Python's sizing hint for mn
        self.assertIn("2.09", second_prompt)             # ... and for mp

    def test_never_passes_ends_failed_after_max_iterations(self):
        run, llm, sim = go([intent_reply()] + [design_reply(w) for w in (0.45, 0.46, 0.47, 0.48)])
        self.assertEqual(run.status, OverallStatus.FAILED)
        self.assertEqual(len(run.iterations), 4)
        self.assertEqual(len(sim.calls), 4)
        self.assertEqual(len(llm.requests), 5)

    def test_max_iterations_is_respected(self):
        run, llm, _ = go([intent_reply(), design_reply(0.45), design_reply(0.46)], max_iterations=2)
        self.assertEqual(len(run.iterations), 2)
        self.assertEqual(len(llm.requests), 3)

    def test_model_claiming_success_cannot_override_the_verifier(self):
        claim = "This design definitely meets the 40 ps target. Status: COMPLETE."
        run, llm, _ = go([intent_reply()] + [design_reply(0.45, rationale=claim)] * 4)
        self.assertEqual(run.status, OverallStatus.FAILED)
        for request in llm.requests[2:]:
            self.assertNotIn("definitely", request[-1]["content"])   # model text is never fed back

    def test_status_comes_from_the_final_report_and_report_serialises(self):
        run, _, _ = go([intent_reply(), design_reply(1.2)])
        self.assertEqual(run.status, run.final_report.overall)
        self.assertIn('"COMPLETE"', run.model_dump_json())


class WhenThereIsNothingToVerify(unittest.TestCase):
    def test_no_measurable_criteria(self):
        reqs = [{"text": "low power", "metric": None, "operator": None, "threshold": None, "unit": None}]
        run, llm, sim = go([intent_reply(requirements=reqs)])
        self.assertEqual(run.status, OverallStatus.INCOMPLETE)
        self.assertEqual(run.iterations, [])
        self.assertEqual(sim.calls, [])
        self.assertEqual(len(llm.requests), 1)            # only the intent call
        self.assertIsNotNone(run.final_report)
        self.assertIn("low power", run.final_report.unresolved)

    def test_unsupported_circuit_is_never_designed(self):
        run, llm, sim = go([intent_reply(circuit_type="op-amp")])
        self.assertEqual(run.status, OverallStatus.INCOMPLETE)
        self.assertEqual(sim.calls, [])
        self.assertEqual(len(llm.requests), 1)

    def test_unhonoured_condition_blocks_complete_even_when_the_delay_passes(self):
        run, llm, _ = go([intent_reply(parameters={"load": "20 fF", "temperature": "85 C"}),
                          design_reply(1.2)])
        self.assertEqual(run.iterations[0].report.counts["PASS"], 1)
        self.assertEqual(run.status, OverallStatus.INCOMPLETE)
        self.assertEqual(len(run.iterations), 1)           # a redesign cannot fix it, so no retries
        self.assertEqual(len(llm.requests), 2)
        self.assertTrue(any(name.startswith("condition applied") for name in run.final_report.unresolved))

    def test_stated_load_is_what_gets_simulated(self):
        run, _, sim = go([intent_reply(parameters={"load": "40 fF"}), design_reply(1.2)],
                          text="CMOS inverter, delay under 40 ps at 40 fF")
        self.assertIn("cl out 0 40f", sim.calls[0])
        self.assertAlmostEqual(run.iterations[0].simulation.measurements["tpd"], 33.5e-12 / 1.2 * 2)

    def test_a_repeated_design_is_not_simulated_again(self):
        run, llm, sim = go([intent_reply(), design_reply(0.45), design_reply(0.45), design_reply(0.45)])
        self.assertEqual(len(run.iterations), 1)
        self.assertEqual(len(sim.calls), 1)
        self.assertEqual(run.status, OverallStatus.FAILED)   # still the verifier's last real result


class WhenThingsBreak(unittest.TestCase):
    def test_intent_model_failure(self):
        run, llm, sim = go(["nope", "still nope"])
        self.assertEqual(run.status, OverallStatus.INCOMPLETE)
        self.assertIsNone(run.final_report)
        self.assertFalse(run.llm_calls[0].ok)
        self.assertEqual(run.llm_calls[0].purpose, "intent")
        self.assertEqual(sim.calls, [])

    def test_empty_text(self):
        run = run_spec("   ", FakeLLM([]), simulate=FakeSimulator(), verify=fake_verify)
        self.assertEqual(run.status, OverallStatus.INCOMPLETE)
        self.assertFalse(run.llm_calls[0].ok)

    def test_design_failure_on_a_later_iteration_keeps_the_last_real_result(self):
        run, _, _ = go([intent_reply(), design_reply(0.45), "bad", "bad"])
        self.assertEqual(len(run.iterations), 1)
        self.assertEqual(run.status, OverallStatus.FAILED)
        self.assertIs(run.final_report, run.iterations[0].report)
        self.assertEqual([(c.purpose, c.ok) for c in run.llm_calls][-1], ("design", False))

    def test_design_failure_on_the_first_iteration(self):
        run, _, sim = go([intent_reply(), "bad", "bad"])
        self.assertEqual(run.status, OverallStatus.INCOMPLETE)
        self.assertEqual(sim.calls, [])
        self.assertIsNotNone(run.final_report)

    def test_simulator_crash_is_no_evidence_with_one_retry_only(self):
        run, llm, sim = go([intent_reply(), design_reply(1.2), design_reply(1.3), design_reply(1.4)],
                           sim=FakeSimulator(crash=True))
        self.assertEqual(run.status, OverallStatus.INCOMPLETE)
        self.assertEqual(len(run.iterations), 2)            # tried twice, then stopped
        self.assertEqual(len(llm.requests), 3)
        self.assertFalse(run.iterations[0].simulation.ok)
        self.assertIn("RuntimeError", run.iterations[0].simulation.error)
        self.assertIn("no usable result", llm.requests[2][-1]["content"])


class Events(unittest.TestCase):
    def test_event_order(self):
        events = []
        go([intent_reply(), design_reply(0.45), design_reply(1.05)], on_event=events.append)
        self.assertEqual([e["type"] for e in events],
                         ["intent", "conditions", "design", "iteration", "design", "iteration", "done"])
        self.assertEqual(events[-1]["status"], "COMPLETE")
        self.assertEqual(events[3]["overall"], "FAILED")

    def test_a_broken_callback_cannot_change_the_run(self):
        def boom(event):
            raise ValueError("ui exploded")
        run, _, _ = go([intent_reply(), design_reply(1.2)], on_event=boom)
        self.assertEqual(run.status, OverallStatus.COMPLETE)

    def test_run_ids_are_unique(self):
        a, _, _ = go([intent_reply(), design_reply(1.2)])
        b, _, _ = go([intent_reply(), design_reply(1.2)])
        self.assertNotEqual(a.run_id, b.run_id)


class FeedbackBuilder(unittest.TestCase):
    def attempt(self, d, criteria):
        sim = FakeSimulator()(inverter_netlist(d))
        return Attempt(d, sim, fake_verify(criteria, sim))

    TPD40 = Criterion(name="tpd", metric="tpd", operator="<", threshold=40.0, unit="ps", measurable=True)

    def test_contains_measurements_and_scaled_widths_only(self):
        text = build_feedback([self.attempt(design(0.45), [self.TPD40])])
        self.assertIn("mn w=0.45", text)
        self.assertIn("1.86x", text)
        self.assertIn("1.05", text)
        self.assertLess(len(text), 1400)

    def test_only_the_last_attempts_are_shown_but_numbering_continues(self):
        attempts = [self.attempt(design(0.3 + i / 10), [self.TPD40]) for i in range(5)]
        text = build_feedback(attempts)
        self.assertNotIn("Attempt 1:", text)
        self.assertIn("Attempt 5:", text)

    def test_widths_at_the_maximum_say_the_target_may_be_unreachable(self):
        hard = Criterion(name="tpd", metric="tpd", operator="<", threshold=0.5, unit="ps", measurable=True)
        text = build_feedback([self.attempt(design(50, 50), [hard])])
        self.assertIn("maximum", text)

    def test_level_failures_get_a_supply_hint_not_a_sizing_hint(self):
        voh = Criterion(name="voh", metric="voh", operator=">", threshold=2.0, unit="V", measurable=True)
        text = build_feedback([self.attempt(design(1.0), [voh])])
        self.assertIn("supply", text)
        self.assertNotIn("proportion to width", text)

    def test_unverifiable_criteria_are_not_mentioned_as_results(self):
        low_power = Criterion(name="low power", measurable=False)
        text = build_feedback([self.attempt(design(0.45), [self.TPD40, low_power])])
        self.assertNotIn("low power", text)


if __name__ == "__main__":
    unittest.main()
