"""Benchmark scoring tests: fake model, fake simulator, fake verifier (the doubles from test_loop).
They prove the scoring logic, not how good any model is."""
import unittest

from app.llm import FakeLLM
from benchmark.run_benchmark import SPECS, run_all, run_baseline, run_system, summarize, to_markdown
from tests.test_loop import FakeSimulator, design_reply, fake_verify, intent_reply

FAST = SPECS[0]            # 40 ps at 20 fF; FakeSimulator: tpd = 33.5 ps / wn(um)


def baseline_reply(wn, claim, wp=None, l=0.18):
    return {"mn": {"w_um": wn, "l_um": l}, "mp": {"w_um": wp or 2 * wn, "l_um": l},
            "expected_delay_ps": 20, "meets_spec": claim, "rationale": "x"}


class Baseline(unittest.TestCase):
    def test_a_false_claim_is_caught_by_simulation(self):
        row = run_baseline(FAST, FakeLLM([baseline_reply(0.5, True)]), FakeSimulator())   # 67 ps
        self.assertEqual(row["outcome"], "false_claim")
        self.assertFalse(row["actually_met"])
        self.assertAlmostEqual(row["measured_ps"], 67.0, places=1)

    def test_a_correct_claim(self):
        row = run_baseline(FAST, FakeLLM([baseline_reply(1.2, True)]), FakeSimulator())   # 27.9 ps
        self.assertEqual(row["outcome"], "claim_correct")
        self.assertTrue(row["actually_met"])
        row = run_baseline(FAST, FakeLLM([baseline_reply(0.5, False)]), FakeSimulator())
        self.assertEqual(row["outcome"], "claim_correct")

    def test_out_of_range_sizes_are_invalid_not_clamped(self):
        sim = FakeSimulator()
        row = run_baseline(FAST, FakeLLM([baseline_reply(500, True)]), sim)
        self.assertEqual(row["outcome"], "invalid_design")
        self.assertEqual(sim.calls, [])

    def test_a_model_that_cannot_answer(self):
        row = run_baseline(FAST, FakeLLM(["nope", "still nope"]), FakeSimulator())
        self.assertEqual(row["outcome"], "no_answer")


class System(unittest.TestCase):
    def test_complete_is_cross_checked_against_the_limit(self):
        row = run_system(FAST, FakeLLM([intent_reply(), design_reply(0.45), design_reply(1.05)]),
                         FakeSimulator(), fake_verify)
        self.assertEqual(row["status"], "COMPLETE")
        self.assertEqual(row["outcome"], "verified_and_met")
        self.assertEqual(row["iterations"], 2)

    def test_failed_run_is_failed_honestly(self):
        row = run_system(FAST, FakeLLM([intent_reply()] + [design_reply(w) for w in (0.45, 0.46, 0.47, 0.48)]),
                         FakeSimulator(), fake_verify)
        self.assertEqual(row["status"], "FAILED")
        self.assertEqual(row["outcome"], "failed_honestly")

    def test_a_complete_that_misses_the_real_limit_is_flagged(self):
        # the compiled criterion says 100 ps, but the benchmark's own limit is 40 ps
        row = run_system(FAST, FakeLLM([intent_reply(requirements=[{"text": "d", "metric": "tpd", "operator": "<",
                                                                     "threshold": 100, "unit": "ps"}]),
                                        design_reply(0.5)]), FakeSimulator(), fake_verify)
        self.assertEqual(row["status"], "COMPLETE")
        self.assertEqual(row["outcome"], "WRONG_COMPLETE")


class Whole(unittest.TestCase):
    def test_run_all_summary_and_markdown(self):
        replies = [baseline_reply(0.5, True), intent_reply(), design_reply(0.45), design_reply(1.05)]
        results = run_all([FAST], FakeLLM(replies), FakeSimulator(), fake_verify)
        t = summarize(results)
        self.assertEqual((t["baseline_false_claims"], t["system_complete_and_met"], t["system_wrong_complete"]),
                         (1, 1, 0))
        md = to_markdown(results, "fake-model", "2026-10-29")
        self.assertIn("false_claim", md)
        self.assertIn("verified_and_met", md)
        self.assertIn("anecdote", md)


if __name__ == "__main__":
    unittest.main()
