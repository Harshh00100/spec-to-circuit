"""Run-manager tests: fake model, fake simulator, fake verifier (the doubles from test_loop).
No web framework, no ngspice, no Ollama. They prove how runs are started, queued, streamed and
saved, not the quality of any model.
"""
import json
import tempfile
import threading
import unittest
from pathlib import Path

from app.llm import FakeLLM
from app.runs import RunManager
from tests.test_loop import FakeSimulator, design_reply, fake_verify, intent_reply

TIMEOUT = 10


def manager(replies, sim=None, history_dir=None):
    return RunManager(simulate=sim or FakeSimulator(), verify=fake_verify,
                      llm=FakeLLM(replies), history_dir=history_dir)


def parse_sse(chunks):
    events = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


class HappyRun(unittest.TestCase):
    def test_events_status_and_report(self):
        m = manager([intent_reply(), design_reply(0.45), design_reply(1.05)])
        h = m.start("CMOS inverter, delay under 40 ps at 20 fF")
        self.assertTrue(h.wait(TIMEOUT))
        self.assertEqual(h.status, "done")
        self.assertEqual(h.overall, "COMPLETE")
        kinds = [e["type"] for e in h.events]
        self.assertEqual(kinds[0], "started")
        for expected in ("intent", "conditions", "design", "iteration", "done", "report"):
            self.assertIn(expected, kinds)
        self.assertEqual(kinds.count("iteration"), 2)          # FAIL, then PASS
        self.assertEqual(kinds[-1], "report")
        self.assertEqual(h.report["status"], "COMPLETE")
        self.assertEqual(h.report["run_id"], h.id)             # one id everywhere, not two
        done = [e for e in h.events if e["type"] == "done"][0]
        self.assertEqual(done["run_id"], h.id)
        self.assertEqual(len(h.report["llm_calls"]), 3)        # latency/token panel data

    def test_a_failed_run_is_reported_as_failed_not_hidden(self):
        m = manager([intent_reply()] + [design_reply(w) for w in (0.45, 0.46, 0.47, 0.48)])
        h = m.start("x")
        self.assertTrue(h.wait(TIMEOUT))
        self.assertEqual(h.status, "done")                      # the run finished; the verdict is FAILED
        self.assertEqual(h.overall, "FAILED")

    def test_get_and_list(self):
        m = manager([intent_reply(), design_reply(1.2)])
        h = m.start("hello")
        h.wait(TIMEOUT)
        self.assertIs(m.get(h.id), h)
        self.assertIsNone(m.get("deadbeef"))
        rows = m.list()
        self.assertEqual(rows[0]["run_id"], h.id)
        self.assertEqual(rows[0]["overall"], "COMPLETE")


class Streaming(unittest.TestCase):
    def test_stream_replays_everything_and_ends(self):
        m = manager([intent_reply(), design_reply(1.2)])
        h = m.start("x")
        h.wait(TIMEOUT)
        chunks = list(m.stream(h, heartbeat_s=0.05))             # connecting after the run ended
        events = parse_sse(chunks)
        self.assertEqual(events, h.events)
        self.assertTrue(chunks[0].startswith("id: 0\ndata: "))

    def test_stream_follows_a_live_run(self):
        gate = threading.Event()

        class SlowSim(FakeSimulator):
            def __call__(self, netlist):
                gate.wait(TIMEOUT)
                return super().__call__(netlist)

        m = manager([intent_reply(), design_reply(1.2)], sim=SlowSim())
        h = m.start("x")
        collected = []
        reader = threading.Thread(target=lambda: collected.extend(m.stream(h, heartbeat_s=0.05)))
        reader.start()
        # the simulator is blocked, so the reader must have seen the early events and some pings
        for _ in range(200):
            if any("intent" in c for c in collected):
                break
            threading.Event().wait(0.02)
        self.assertTrue(any('"type": "intent"' in c for c in collected))
        self.assertFalse(h.finished)
        threading.Event().wait(0.3)                            # blocked simulator: only heartbeats arrive
        gate.set()
        reader.join(TIMEOUT)
        self.assertFalse(reader.is_alive())
        self.assertEqual(parse_sse(collected)[-1]["type"], "report")
        self.assertTrue(any(c.startswith(": ping") for c in collected))


class OneAtATime(unittest.TestCase):
    def test_second_run_waits_for_the_first(self):
        gate = threading.Event()
        order = []

        class SlowSim(FakeSimulator):
            def __call__(self, netlist):
                order.append("sim")
                gate.wait(TIMEOUT)
                return super().__call__(netlist)

        m = manager([intent_reply(), design_reply(1.2), intent_reply(), design_reply(1.2)], sim=SlowSim())
        first = m.start("one")
        for _ in range(200):
            if first.status == "running" and order:
                break
            threading.Event().wait(0.02)
        second = m.start("two")
        threading.Event().wait(0.2)
        self.assertEqual(first.status, "running")
        self.assertEqual(second.status, "queued")
        self.assertEqual(second.events, [])                    # nothing happens until its turn
        gate.set()
        self.assertTrue(first.wait(TIMEOUT))
        self.assertTrue(second.wait(TIMEOUT))
        self.assertEqual((first.status, second.status), ("done", "done"))


class WhenThingsBreak(unittest.TestCase):
    def test_a_crash_inside_the_worker_ends_the_run_visibly(self):
        m = manager([])
        m._simulate = None
        orig = m._verify

        def boom(criteria, sim):
            raise RuntimeError("verifier exploded")
        m._verify = boom
        m._llm = FakeLLM([intent_reply(), design_reply(1.2)])
        h = m.start("x")
        self.assertTrue(h.wait(TIMEOUT))
        self.assertEqual(h.status, "error")
        self.assertEqual(h.events[-1]["type"], "error")
        self.assertIn("verifier exploded", h.events[-1]["message"])
        self.assertIsNone(h.overall)                           # no verdict is invented
        self.assertEqual(parse_sse(m.stream(h, heartbeat_s=0.05))[-1]["type"], "error")
        m._verify = orig

    def test_model_failure_gives_incomplete_not_a_crash(self):
        m = manager(["nope", "still nope"])
        h = m.start("x")
        self.assertTrue(h.wait(TIMEOUT))
        self.assertEqual(h.overall, "INCOMPLETE")


class History(unittest.TestCase):
    def test_runs_are_saved_and_reloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = manager([intent_reply(), design_reply(1.2)], history_dir=tmp)
            h = m.start("saved run")
            h.wait(TIMEOUT)
            self.assertTrue((Path(tmp) / f"{h.id}.json").is_file())
            fresh = manager([], history_dir=tmp)                 # as after a restart
            self.assertIsNone(fresh.get(h.id))
            saved = fresh.load_saved(h.id)
            self.assertEqual(saved["overall"], "COMPLETE")
            self.assertEqual(saved["report"]["spec_text"], "saved run")
            self.assertEqual(fresh.list()[0]["run_id"], h.id)

    def test_bad_ids_never_touch_the_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp).parent / "secret.json").write_text("{}")
            m = manager([], history_dir=tmp)
            for bad in ("../secret", "..\\secret", "", "zzzzzzzz", "a" * 9, "12345678/../x"):
                self.assertIsNone(m.load_saved(bad), bad)

    def test_no_history_dir_means_nothing_is_written(self):
        m = manager([intent_reply(), design_reply(1.2)])
        h = m.start("x")
        h.wait(TIMEOUT)
        self.assertIsNone(m.load_saved(h.id))


if __name__ == "__main__":
    unittest.main()
