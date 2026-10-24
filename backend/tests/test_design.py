"""Design stage tests: conditions parsing, the validated netlist template and the proposer.
They use FakeLLM (no Ollama, no ngspice). They check OUR rules, not any model's quality.
"""
import re
import unittest

from app.design import (
    DesignError,
    conditions_from_intent,
    propose_design,
    unapplied_criteria,
)
from app.llm import FakeLLM, LLMOutputError
from app.netlists import (
    INVERTER_MEASURES,
    hardcoded_inverter_netlist,
    inverter_netlist,
)
from app.schemas import Criterion, DesignProposal, IntentSpec, TransistorSize

TPD = Criterion(name="tpd under 2 ns", metric="tpd", operator="<", threshold=2.0,
                unit="ns", measurable=True)
POWER = Criterion(name="low power", measurable=False)


def spec(parameters=None, criteria=None, circuit_type="inverter"):
    return IntentSpec(circuit_type=circuit_type, parameters=parameters or {},
                      criteria=[TPD] if criteria is None else criteria)


def design(wn=0.5, ln=0.18, wp=1.0, lp=0.18, **kw):
    return DesignProposal(
        circuit_type=kw.get("circuit_type", "inverter"), stage_count=kw.get("stage_count", 1),
        transistors={"mn": TransistorSize(w_um=wn, l_um=ln), "mp": TransistorSize(w_um=wp, l_um=lp)})


def model_reply(**kw):
    base = {"mn": {"w_um": 0.6, "l_um": 0.18}, "mp": {"w_um": 1.2, "l_um": 0.18},
            "rationale": "balanced"}
    base.update(kw)
    return base


class Conditions(unittest.TestCase):
    def test_nothing_stated_means_defaults_are_assumed(self):
        c = conditions_from_intent(spec())
        self.assertEqual((c.cload_ff, c.vdd), (20.0, 1.8))
        self.assertEqual(len(c.assumed), 2)
        self.assertEqual((c.applied, c.unapplied), ([], []))

    def test_stated_load_and_supply_are_applied(self):
        c = conditions_from_intent(spec({"load": "10 pF", "supply": "1.2 V"}))
        self.assertAlmostEqual(c.cload_ff, 10000.0, places=6)
        self.assertAlmostEqual(c.vdd, 1.2)
        self.assertEqual(len(c.applied), 2)
        self.assertEqual((c.assumed, c.unapplied), ([], []))

    def test_key_and_unit_case_variants(self):
        c = conditions_from_intent(spec({"Load_Capacitance": "50 ff", "VDD": "1500 mv"}))
        self.assertAlmostEqual(c.cload_ff, 50.0, places=6)
        self.assertAlmostEqual(c.vdd, 1.5)

    def test_bare_supply_number_is_volts_but_bare_load_is_not_accepted(self):
        c = conditions_from_intent(spec({"vdd": 1.2, "load": 10}))
        self.assertAlmostEqual(c.vdd, 1.2)
        self.assertEqual(c.cload_ff, 20.0)
        self.assertEqual(len(c.unapplied), 1)
        self.assertIn("load", c.unapplied[0])

    def test_out_of_range_supply_is_not_honoured_and_says_so(self):
        c = conditions_from_intent(spec({"supply": "5 V"}))
        self.assertEqual(c.vdd, 1.8)
        self.assertEqual(len(c.unapplied), 1)
        self.assertIn("5 V", c.unapplied[0])

    def test_wrong_dimension_and_huge_load_are_not_honoured(self):
        self.assertEqual(len(conditions_from_intent(spec({"load": "10 ns"})).unapplied), 1)
        self.assertEqual(len(conditions_from_intent(spec({"load": "1 uF"})).unapplied), 1)
        self.assertEqual(len(conditions_from_intent(spec({"load": "0 fF"})).unapplied), 1)

    def test_unknown_condition_is_not_honoured(self):
        c = conditions_from_intent(spec({"temperature": "85 C"}))
        self.assertEqual(len(c.unapplied), 1)
        self.assertIn("not modelled", c.unapplied[0])

    def test_unapplied_conditions_become_unverifiable_criteria(self):
        c = conditions_from_intent(spec({"temperature": "85 C", "supply": "9 V"}))
        crits = unapplied_criteria(c)
        self.assertEqual(len(crits), 2)
        for crit in crits:
            self.assertFalse(crit.measurable)
            self.assertTrue(crit.unverifiable_reason)
        self.assertEqual(unapplied_criteria(conditions_from_intent(spec())), [])


class NetlistTemplate(unittest.TestCase):
    def test_default_design_matches_the_hardcoded_netlist(self):
        built = inverter_netlist(design()).splitlines()[1:]  # skip the differing comment line
        original = hardcoded_inverter_netlist().splitlines()[1:]
        self.assertEqual(built, original)

    def test_values_reach_the_netlist(self):
        text = inverter_netlist(design(wn=2, ln=0.25, wp=4, lp=0.3), vdd=1.2, cload_ff=50)
        self.assertIn("w=4u l=0.3u", text)
        self.assertIn("w=2u l=0.25u", text)
        self.assertIn("vsupply vdd 0 1.2", text)
        self.assertIn("cl out 0 50f", text)

    def test_heavy_load_widens_the_simulation_window(self):
        text = inverter_netlist(design(), cload_ff=10000)  # 10 pF
        step, stop = re.search(r"^\.tran (\S+)p (\S+)n$", text, re.M).groups()
        self.assertGreater(float(step), 5)
        self.assertGreater(float(stop), 10)

    def test_has_all_measures_and_no_dangerous_directives(self):
        text = inverter_netlist(design()).lower()
        for name in INVERTER_MEASURES:
            self.assertIn(f".meas tran {name}", text)
        for bad in (".control", ".include", ".lib", "shell", ".endc"):
            self.assertNotIn(bad, text)

    def test_rejects_anything_off_template(self):
        for bad in (
            lambda: inverter_netlist(design(wn=500)),
            lambda: inverter_netlist(design(ln=0.01)),
            lambda: inverter_netlist(design(), vdd=5),
            lambda: inverter_netlist(design(), vdd=float("nan")),
            lambda: inverter_netlist(design(), cload_ff=0),
            lambda: inverter_netlist(design(), cload_ff=float("inf")),
            lambda: inverter_netlist(design(stage_count=2)),
            lambda: inverter_netlist(design(circuit_type="nand")),
            lambda: inverter_netlist(DesignProposal(
                circuit_type="inverter", stage_count=1,
                transistors={"mn": TransistorSize(w_um=1, l_um=1)})),
        ):
            with self.assertRaises(ValueError):
                bad()


class Proposer(unittest.TestCase):
    def propose(self, reply, s=None, feedback=None, conditions=None):
        s = s or spec()
        fake = FakeLLM([reply])
        cond = conditions or conditions_from_intent(s)
        return propose_design(s, cond, fake, feedback), fake

    def test_happy_path(self):
        result, fake = self.propose(model_reply())
        d = result.design
        self.assertEqual((d.circuit_type, d.stage_count), ("inverter", 1))
        self.assertEqual((d.transistors["mn"].w_um, d.transistors["mp"].w_um), (0.6, 1.2))
        self.assertEqual(d.rationale, "balanced")
        self.assertEqual(result.notes, [])
        self.assertEqual(result.call.purpose, "design")
        self.assertTrue(result.call.ok)
        user = fake.requests[0][-1]["content"]
        self.assertIn("tpd < 2 ns", user)
        self.assertIn("20 fF", user)
        self.assertIn("1.8 V", user)

    def test_out_of_range_sizes_are_clamped_and_reported(self):
        result, _ = self.propose(model_reply(
            mn={"w_um": 500, "l_um": 0.01}, mp={"w_um": 0, "l_um": 9}))
        t = result.design.transistors
        self.assertEqual((t["mn"].w_um, t["mn"].l_um), (50.0, 0.18))
        self.assertEqual((t["mp"].w_um, t["mp"].l_um), (0.18, 2.0))
        self.assertEqual(len(result.notes), 4)
        inverter_netlist(result.design)  # the clamped design is always buildable

    def test_numbers_as_strings_are_accepted(self):
        result, _ = self.propose(model_reply(mn={"w_um": "0.5", "l_um": "0.18"}))
        self.assertEqual(result.design.transistors["mn"].w_um, 0.5)

    def test_model_cannot_change_conditions_or_set_verdicts(self):
        cond = conditions_from_intent(spec({"load": "10 pF"}))
        reply = model_reply(cload_ff=1, vdd=3.3, verdict="PASS", status="COMPLETE")
        result, _ = self.propose(reply, conditions=cond)
        self.assertAlmostEqual(cond.cload_ff, 10000.0, places=6)
        dumped = result.design.model_dump_json()
        self.assertNotIn("PASS", dumped)
        self.assertNotIn("cload", dumped)

    def test_feedback_is_passed_on_and_truncated(self):
        result, fake = self.propose(model_reply(), feedback="attempt 1: tpd 3.1 ns > 2 ns" + "x" * 5000)
        user = fake.requests[0][-1]["content"]
        self.assertIn("attempt 1: tpd 3.1 ns", user)
        self.assertLess(len(user), 2500)

    def test_rationale_is_truncated_and_may_be_null(self):
        result, _ = self.propose(model_reply(rationale="y" * 1000))
        self.assertEqual(len(result.design.rationale), 300)
        result, _ = self.propose(model_reply(rationale=None))
        self.assertEqual(result.design.rationale, "")

    def test_missing_transistor_is_retried_then_raised(self):
        bad = {"mn": {"w_um": 1, "l_um": 0.18}}
        with self.assertRaises(LLMOutputError):
            propose_design(spec(), conditions_from_intent(spec()), FakeLLM([bad, bad]))

    def test_retry_then_success(self):
        fake = FakeLLM(["not json", model_reply()])
        result = propose_design(spec(), conditions_from_intent(spec()), fake)
        self.assertEqual(result.call.retries, 1)

    def test_nothing_to_design_for(self):
        s = spec(criteria=[POWER])
        with self.assertRaises(DesignError):
            propose_design(s, conditions_from_intent(s), FakeLLM([]))

    def test_unsupported_circuit_has_no_template(self):
        s = spec(circuit_type="op-amp")
        fake = FakeLLM([])
        with self.assertRaises(DesignError):
            propose_design(s, conditions_from_intent(s), fake)
        self.assertEqual(fake.requests, [])

    def test_prompt_fits_a_small_context(self):
        _, fake = self.propose(model_reply())
        total = sum(len(m["content"]) for m in fake.requests[0])
        self.assertLess(total, 9000)


if __name__ == "__main__":
    unittest.main()
