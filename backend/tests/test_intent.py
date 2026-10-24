"""Intent compiler tests. They use FakeLLM, so no Ollama and no network are needed.
They verify OUR rules (what becomes measurable, what is demoted to unverifiable),
not the quality of any real model.
"""
import json
import unittest

from app.intent import (
    MAX_CRITERIA,
    MAX_INPUT_CHARS,
    compile_intent,
    is_supported,
)
from app.llm import FakeLLM, LLMOutputError

TEXT = "CMOS inverter, propagation delay under 2 ns"
# Parameters must appear in the requirement (see _ground_parameters), so tests that feed parameters
# through compile_with use a text that states every condition they use.
STATED = (TEXT + ", load 20 fF, load 10 pF, supply 1.2 V, supply voltage 1.8, vcc 1.8, "
          "vdd 1.8 parsecs, temperature 85 C, output above 1.7 V")


def req(text="x", metric="tpd", operator="<", threshold=2, unit="ns"):
    return {"text": text, "metric": metric, "operator": operator,
            "threshold": threshold, "unit": unit}


def reply(requirements, circuit_type="inverter", parameters=None, questions=None):
    return {"circuit_type": circuit_type, "parameters": parameters or {},
            "requirements": requirements, "questions": questions or []}


def compile_with(model_reply, text=STATED):
    fake = FakeLLM([model_reply])
    return compile_intent(text, fake), fake


class HappyPath(unittest.TestCase):
    def test_measurable_and_unverifiable_requirements(self):
        result, _ = compile_with(reply(
            [req("propagation delay under 2 ns"),
             req("output high above 1.7 V", "voh", ">", 1.7, "V"),
             {"text": "low power", "metric": None, "operator": None,
              "threshold": None, "unit": None}],
            parameters={"Load Capacitance": "10 pF", "vdd": 1.8}))
        spec = result.spec
        self.assertEqual(spec.circuit_type, "inverter")
        self.assertEqual(spec.parameters, {"load_capacitance": "10 pF", "vdd": 1.8})
        self.assertEqual(len(spec.criteria), 3)
        first, second, third = spec.criteria
        self.assertTrue(first.measurable)
        self.assertEqual((first.metric, first.operator, first.threshold, first.unit),
                         ("tpd", "<", 2.0, "ns"))
        self.assertEqual((second.metric, second.operator, second.unit), ("voh", ">", "V"))
        self.assertFalse(third.measurable)
        self.assertEqual(third.name, "low power")
        self.assertTrue(third.unverifiable_reason)
        self.assertEqual(spec.questions, [])
        self.assertIn("Will verify by simulation", spec.understood_summary)
        self.assertIn("Cannot verify: low power", spec.understood_summary)
        self.assertTrue(result.call.ok)
        self.assertEqual(result.call.purpose, "intent")
        self.assertTrue(is_supported(spec))

    def test_null_fields_from_a_sloppy_model_are_tolerated(self):
        result, _ = compile_with({"circuit_type": "inverter", "parameters": None,
                                  "requirements": [req()], "questions": None})
        self.assertEqual(len(result.spec.criteria), 1)
        self.assertEqual(result.spec.parameters, {})

    def test_bare_string_requirement_is_unverifiable(self):
        result, _ = compile_with(reply(["be cheap"]))
        crit = result.spec.criteria[0]
        self.assertEqual(crit.name, "be cheap")
        self.assertFalse(crit.measurable)


class Normalisation(unittest.TestCase):
    def test_aliases_operators_and_units_are_normalised(self):
        result, _ = compile_with(reply([
            req("fall", "t_PHL", "\u2264", "500 PS", None),
            req("rise", "propagation delay", "less than", 0.5, "\u00b5s"),
            req("low", "output low", "below", "0.1", "volts"),
            req("low2", "V_OL", "under", "100 mV", ""),
        ]))
        a, b, c, d = result.spec.criteria
        self.assertEqual((a.metric, a.operator, a.threshold, a.unit), ("tphl", "<=", 500.0, "ps"))
        self.assertEqual((b.metric, b.operator, b.threshold, b.unit), ("tpd", "<", 0.5, "us"))
        self.assertEqual((c.metric, c.operator, c.threshold, c.unit), ("vol", "<", 0.1, "V"))
        self.assertEqual((d.metric, d.operator, d.threshold, d.unit), ("vol", "<", 100.0, "mV"))
        self.assertTrue(all(x.measurable for x in (a, b, c, d)))
        self.assertTrue(result.notes)  # changes are reported for the UI diff

    def test_empty_name_gets_a_generated_one(self):
        result, _ = compile_with(reply([req("")]))
        self.assertEqual(result.spec.criteria[0].name, "tpd < 2 ns")

    def test_parameters_are_cleaned(self):
        result, _ = compile_with(reply(
            [req()], parameters={"Supply Voltage": 1.8, "flag": True, "nested": {"a": 1}, "": 3}))
        self.assertEqual(result.spec.parameters, {"supply_voltage": 1.8})
        self.assertTrue(any("dropped" in n for n in result.notes))


class Demotion(unittest.TestCase):
    def assert_unverifiable(self, item, fragment):
        result, _ = compile_with(reply([item]))
        crit = result.spec.criteria[0]
        self.assertFalse(crit.measurable, crit)
        self.assertIn(fragment, crit.unverifiable_reason)
        return result

    def test_unknown_metric_is_never_measurable(self):
        result = self.assert_unverifiable(req("power under 1 mW", "power", "<", 1, "mW"), "power")
        self.assertTrue(result.notes)

    def test_unit_that_does_not_fit_the_metric(self):
        self.assert_unverifiable(req(metric="tpd", unit="V"), "time")
        self.assert_unverifiable(req(metric="voh", operator=">", threshold=1.7, unit="ns"), "voltage")

    def test_missing_unit(self):
        self.assert_unverifiable(req(unit=None), "no unit")

    def test_unrecognised_unit(self):
        self.assert_unverifiable(req(unit="parsecs"), "not recognised")

    def test_conflicting_units(self):
        self.assert_unverifiable(req(threshold="2 ns", unit="ps"), "conflicting")

    def test_bad_thresholds(self):
        for bad in ("fast", "nan", None, True, [1], {"a": 1}):
            self.assert_unverifiable(req(threshold=bad), "threshold")

    def test_bad_operator(self):
        self.assert_unverifiable(req(operator="roughly"), "operator")
        self.assert_unverifiable(req(operator=None), "operator")


class MiskeyedParameters(unittest.TestCase):
    def test_value_with_the_wrong_kind_of_unit_is_dropped(self):
        # seen with a real 3B model: it copied "20 fF" under both load and supply
        result, _ = compile_with(reply([req()], parameters={"load": "20 fF", "supply": "20 fF"}))
        self.assertEqual(result.spec.parameters, {"load": "20 fF"})
        self.assertTrue(any("supply" in n and "dropped" in n for n in result.notes))
        result, _ = compile_with(reply([req()], parameters={"capacitance": "5 V"}))
        self.assertEqual(result.spec.parameters, {})

    def test_matching_unitless_unknown_unit_and_other_keys_are_kept_for_the_design_stage(self):
        params = {"load": "10 pF", "supply": "1.2 V", "vcc": 1.8,
                  "vdd": "1.8 parsecs", "temperature": "85 C"}
        result, _ = compile_with(reply([req()], parameters=params))
        self.assertEqual(result.spec.parameters, {
            "load": "10 pF", "supply": "1.2 V", "vcc": 1.8,
            "vdd": "1.8 parsecs", "temperature": "85 C"})

    def test_prompt_examples_are_clean_and_not_copyable_templates(self):
        from app.intent import SYSTEM_PROMPT, _EXAMPLES, _clean_parameters
        for _, example in _EXAMPLES:
            notes = []
            self.assertEqual(_clean_parameters(example["parameters"], notes), example["parameters"])
            self.assertEqual(notes, [])
        self.assertLessEqual(max(len(e["parameters"]) for _, e in _EXAMPLES), 1)
        self.assertIn("Never add, guess", SYSTEM_PROMPT)


class ConditionsInRequirements(unittest.TestCase):
    """A model sometimes lists 'at 20 fF load' as a requirement of its own."""

    def cond(self, text="at 20 fF load"):
        return {"text": text, "metric": None, "operator": None, "threshold": None, "unit": None}

    def test_restated_load_is_not_a_separate_requirement(self):
        result, _ = compile_with(reply([req(), self.cond()], parameters={"load": "20 fF"}))
        self.assertEqual([c.name for c in result.spec.criteria], ["x"])
        self.assertNotIn("Cannot verify", result.spec.understood_summary)
        self.assertTrue(any("restates" in n for n in result.notes))

    def test_unit_case_and_equivalent_units_still_match(self):
        result, _ = compile_with(reply([req(), self.cond("at 0.02 pf load")],
                                       parameters={"load": "20 fF"}))
        self.assertEqual(len(result.spec.criteria), 1)

    def test_load_missing_from_parameters_is_promoted(self):
        result, _ = compile_with(reply([req(), self.cond()]))
        self.assertEqual(result.spec.parameters, {"load": "20 fF"})
        self.assertEqual(len(result.spec.criteria), 1)
        self.assertTrue(any("applied as load" in n for n in result.notes))

    def test_supply_needs_the_word_supply_and_is_matched_or_promoted(self):
        result, _ = compile_with(reply([req(), self.cond("at 1.2 V supply")]))
        self.assertEqual(result.spec.parameters, {"supply": "1.2 V"})
        self.assertEqual(len(result.spec.criteria), 1)
        result, _ = compile_with(reply([req(), self.cond("1.2 V")]))   # no 'supply': ambiguous
        self.assertEqual(result.spec.parameters, {})
        self.assertEqual(len(result.spec.criteria), 2)

    def test_numeric_supply_parameter_matches_a_restating_requirement(self):
        result, _ = compile_with(reply([req(), self.cond("at 1.8 V supply")],
                                       parameters={"vdd": 1.8}))
        self.assertEqual(len(result.spec.criteria), 1)

    def test_contradicting_or_richer_text_stays_visible_and_unverifiable(self):
        result, _ = compile_with(reply([req(), self.cond("at 5 pF load")],
                                       parameters={"load": "20 fF"}))
        self.assertEqual(len(result.spec.criteria), 2)
        self.assertFalse(result.spec.criteria[1].measurable)
        result, _ = compile_with(reply([req(), self.cond("low power at 20 fF load")]))
        self.assertEqual(len(result.spec.criteria), 2)
        self.assertEqual(result.spec.parameters, {})
        result, _ = compile_with(reply([req(), self.cond("delay under 40 ps")]))
        self.assertEqual(len(result.spec.criteria), 2)

    def test_measurable_requirements_are_never_absorbed(self):
        result, _ = compile_with(reply([req("tpd under 2 ns at 20 fF load")],
                                       parameters={"load": "20 fF"}))
        self.assertTrue(result.spec.criteria[0].measurable)


class RestatedAsMetric(unittest.TestCase):
    """Seen live: the model re-listed the load as 'propagation delay at 20 fF' with no operator."""

    def bad_delay(self, text="propagation delay at 20 fF"):
        return {"text": text, "metric": "tpd", "operator": None, "threshold": 20, "unit": "fF"}

    def good_delay(self):
        return req("propagation delay under 40 ps", threshold=40, unit="ps")

    def test_load_restated_as_a_delay_is_absorbed(self):
        result, _ = compile_with(reply([self.good_delay(), self.bad_delay()],
                                       parameters={"load": "20 fF"}))
        self.assertEqual([c.name for c in result.spec.criteria], ["propagation delay under 40 ps"])
        self.assertNotIn("Cannot verify", result.spec.understood_summary)
        self.assertTrue(any("restates" in n for n in result.notes))

    def test_it_is_promoted_to_the_load_when_no_parameter_exists(self):
        result, _ = compile_with(reply([self.good_delay(), self.bad_delay()]))
        self.assertEqual(result.spec.parameters, {"load": "20 fF"})
        self.assertEqual(len(result.spec.criteria), 1)

    def test_kept_when_the_metric_has_no_measurable_requirement(self):
        result, _ = compile_with(reply([req("out high", "voh", ">", 1.7, "V"), self.bad_delay()],
                                       parameters={"load": "20 fF"}))
        self.assertEqual(len(result.spec.criteria), 2)
        self.assertFalse(result.spec.criteria[1].measurable)

    def test_kept_when_it_contradicts_the_stated_load(self):
        result, _ = compile_with(reply([self.good_delay(), self.bad_delay("propagation delay at 5 pF")],
                                       parameters={"load": "20 fF"}))
        self.assertEqual(len(result.spec.criteria), 2)

    def test_unrelated_words_keep_it_visible(self):
        result, _ = compile_with(reply([self.good_delay(), self.bad_delay("low power delay at 20 fF")],
                                       parameters={"load": "20 fF"}))
        self.assertEqual(len(result.spec.criteria), 2)

    def test_prompt_says_conditions_are_parameters_only(self):
        from app.intent import SYSTEM_PROMPT
        self.assertIn("parameter only", SYSTEM_PROMPT)


class NoiseRequirements(unittest.TestCase):
    """Fragments a small model emits as requirements (seen live: a requirement that was just 'at')."""

    def nometric(self, text):
        return {"text": text, "metric": None, "operator": None, "threshold": None, "unit": None}

    def test_fragments_without_content_are_dropped(self):
        for fragment in ("at", "", "  ", "should be", "the", "and"):
            result, _ = compile_with(reply([req(), self.nometric(fragment)]))
            self.assertEqual([c.name for c in result.spec.criteria], ["x"], fragment)
            self.assertNotIn("Cannot verify", result.spec.understood_summary)
            self.assertTrue(any("no usable content" in n for n in result.notes), fragment)

    def test_bare_string_fragments_are_dropped_too(self):
        result, _ = compile_with(reply([req(), "at", "and"]))
        self.assertEqual(len(result.spec.criteria), 1)

    def test_real_unverifiable_requirements_stay(self):
        result, _ = compile_with(reply([req(), self.nometric("low power"),
                                        self.nometric("cheap"), "be small"]))
        self.assertEqual([c.name for c in result.spec.criteria], ["x", "low power", "cheap", "be small"])

    def test_text_with_numbers_stays_visible(self):
        result, _ = compile_with(reply([req(), self.nometric("at 5 pF load")],
                                       parameters={"load": "20 fF"}))
        self.assertEqual(len(result.spec.criteria), 2)

    def test_broken_requirement_that_names_a_metric_is_never_dropped_as_noise(self):
        result, _ = compile_with(reply([req("", threshold="fast")]))
        self.assertEqual(len(result.spec.criteria), 1)
        self.assertFalse(result.spec.criteria[0].measurable)

    def test_only_noise_leaves_no_criteria_and_asks_what_to_verify(self):
        result, _ = compile_with(reply([self.nometric("at")]))
        self.assertEqual(result.spec.criteria, [])
        self.assertEqual(len(result.spec.questions), 1)


class Questions(unittest.TestCase):
    def test_at_most_two_model_questions_and_duplicates_removed(self):
        result, _ = compile_with(reply(
            [req()], questions=["What load?", "what load?", "What supply?", "Which corner?"]))
        self.assertEqual([q.question for q in result.spec.questions], ["What load?", "What supply?"])

    def test_extra_question_when_nothing_is_measurable(self):
        result, _ = compile_with(reply([req("power", "power", "<", 1, "mW")]))
        self.assertEqual(len(result.spec.questions), 1)
        self.assertIn("measurable targets", result.spec.questions[0].question)

    def test_never_more_than_two_even_when_nothing_is_measurable(self):
        result, _ = compile_with(reply([], questions=["a?", "b?", "c?"]))
        self.assertEqual(len(result.spec.questions), 2)


class CircuitType(unittest.TestCase):
    def test_missing_type_falls_back_to_text(self):
        result, _ = compile_with(reply([req()], circuit_type=""))
        self.assertEqual(result.spec.circuit_type, "inverter")

    def test_unsupported_type_is_kept_and_flagged(self):
        result, _ = compile_with(reply([req()], circuit_type="op-amp"),
                                 text="op-amp with 60 dB gain")
        self.assertEqual(result.spec.circuit_type, "op-amp")
        self.assertFalse(is_supported(result.spec))
        self.assertTrue(any("no template" in n for n in result.notes))


class Robustness(unittest.TestCase):
    def test_invented_fields_such_as_verdict_are_ignored(self):
        model_reply = reply([dict(req(), passed=True)])
        model_reply.update(verdict="PASS", status="COMPLETE")
        result, _ = compile_with(model_reply)
        dumped = result.spec.model_dump_json()
        self.assertNotIn("COMPLETE", dumped)
        self.assertNotIn("verdict", dumped)
        self.assertEqual(len(result.spec.criteria), 1)

    def test_duplicate_measurable_criteria_are_dropped(self):
        result, _ = compile_with(reply([req("a"), req("b")]))
        self.assertEqual(len(result.spec.criteria), 1)
        self.assertTrue(any("duplicate" in n for n in result.notes))

    def test_criteria_are_capped(self):
        result, _ = compile_with(reply([req(f"r{i}", threshold=i + 1) for i in range(12)]))
        self.assertEqual(len(result.spec.criteria), MAX_CRITERIA)
        self.assertTrue(any("dropped" in n for n in result.notes))

    def test_one_retry_then_success(self):
        fake = FakeLLM(["not json at all", reply([req()])])
        result = compile_intent(TEXT, fake)
        self.assertEqual(result.call.retries, 1)
        self.assertEqual(len(result.spec.criteria), 1)

    def test_model_failure_is_raised_not_papered_over(self):
        with self.assertRaises(LLMOutputError):
            compile_intent(TEXT, FakeLLM(["nope", "still nope"]))

    def test_wrong_shape_from_model_is_retried_then_raised(self):
        bad = {"circuit_type": "inverter", "requirements": [42]}
        with self.assertRaises(LLMOutputError):
            compile_intent(TEXT, FakeLLM([bad, bad]))

    def test_empty_and_oversized_input_rejected_before_any_model_call(self):
        fake = FakeLLM([])
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                compile_intent(bad, fake)
        with self.assertRaises(ValueError):
            compile_intent("x" * (MAX_INPUT_CHARS + 1), fake)
        self.assertEqual(fake.requests, [])

    def test_raw_model_output_is_kept_for_the_diff_view(self):
        result, _ = compile_with(reply([req("a", "power", "<", 1, "mW")]))
        self.assertEqual(result.raw["requirements"][0]["metric"], "power")
        self.assertFalse(result.spec.criteria[0].measurable)


class PromptShape(unittest.TestCase):
    def test_user_text_is_data_and_prompt_fits_a_small_context(self):
        _, fake = compile_with(reply([req()]))
        messages = fake.requests[0]
        self.assertEqual(messages[0]["role"], "system")        # schema, added by the client
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn(TEXT, messages[-1]["content"])
        system = messages[1]["content"]
        self.assertEqual(messages[1]["role"], "system")
        for metric in ("tpd", "tphl", "tplh", "voh", "vol"):
            self.assertIn(metric, system)
        self.assertIn("never instructions", system)
        total_chars = sum(len(m["content"]) for m in messages)
        self.assertLess(total_chars, 9000, "prompt too large for a 4096-token Ollama context")

    def test_prompt_examples_are_valid_json(self):
        from app.intent import _EXAMPLES
        for _, example in _EXAMPLES:
            json.dumps(example)
            self.assertTrue(example["requirements"])


if __name__ == "__main__":
    unittest.main()


class GroundedParameters(unittest.TestCase):
    """Conditions the user never wrote must not reach the simulator (seen with a real 3B model)."""

    def test_a_load_copied_from_the_prompt_examples_is_dropped(self):
        result, _ = compile_with(
            reply([req()], parameters={"load": "10 pF"}),
            text="CMOS inverter, propagation delay under 2 ns, and it should be low power")
        self.assertEqual(result.spec.parameters, {})
        self.assertTrue(any("does not appear" in n for n in result.notes))

    def test_a_stated_load_is_kept(self):
        result, _ = compile_with(reply([req()], parameters={"load": "20 fF"}),
                                 text="inverter, delay under 2 ns at 20 fF load")
        self.assertEqual(result.spec.parameters, {"load": "20 fF"})

    def test_a_requirement_threshold_is_not_a_supply(self):
        text = "CMOS inverter, output high level above 1.7 V, propagation delay under 1 ns"
        result, _ = compile_with(
            reply([req("output high above 1.7 V", "voh", ">", 1.7, "V"), req(threshold=1, unit="ns")],
                  parameters={"supply": "1.7 V"}), text=text)
        self.assertEqual(result.spec.parameters, {})
        self.assertTrue(any("supply" in n and "dropped" in n for n in result.notes))

    def test_a_real_supply_next_to_a_level_requirement_is_kept(self):
        text = "inverter at 1.8 V supply, output high above 1.7 V"
        result, _ = compile_with(
            reply([req("output high above 1.7 V", "voh", ">", 1.7, "V")], parameters={"supply": "1.8 V"}),
            text=text)
        self.assertEqual(result.spec.parameters, {"supply": "1.8 V"})
