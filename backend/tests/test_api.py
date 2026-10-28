"""HTTP tests: a real uvicorn server on a free local port, driven with urllib (no extra packages).
Skipped automatically when fastapi or uvicorn is not installed. The model, simulator and verifier
are the fakes from test_loop, so no ngspice or Ollama is needed.
"""
import json
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request

try:
    import uvicorn
    from app.main import create_app
    HAVE_WEB = True
except ImportError:  # fastapi / uvicorn not installed
    HAVE_WEB = False

from app.llm import FakeLLM
from app.runs import RunManager
from tests.test_loop import FakeSimulator, design_reply, fake_verify, intent_reply


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(method, url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"null")


@unittest.skipUnless(HAVE_WEB, "fastapi and uvicorn are not installed")
class Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manager = RunManager(simulate=FakeSimulator(), verify=fake_verify,
                                 llm=FakeLLM([intent_reply(), design_reply(0.45), design_reply(1.05)]))
        cls.port = free_port()
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.server = uvicorn.Server(uvicorn.Config(create_app(cls.manager), host="127.0.0.1",
                                                   port=cls.port, log_level="error"))
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        for _ in range(100):
            if cls.server.started:
                break
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(10)

    def test_0_the_page_is_served(self):
        with urllib.request.urlopen(f"{self.base}/", timeout=15) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/html", resp.headers["Content-Type"])
            self.assertIn(b"Spec to Circuit", resp.read())

    def test_1_health(self):
        status, body = call("GET", f"{self.base}/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertIn("ngspice_found", body)

    def test_2_bad_requests(self):
        self.assertEqual(call("POST", f"{self.base}/api/runs", {"text": ""})[0], 422)
        self.assertEqual(call("POST", f"{self.base}/api/runs", {"text": "x" * 2001})[0], 422)
        self.assertEqual(call("POST", f"{self.base}/api/runs", {})[0], 422)
        self.assertEqual(call("GET", f"{self.base}/api/runs/nonexist")[0], 404)
        self.assertEqual(call("GET", f"{self.base}/api/runs/deadbeef")[0], 404)
        self.assertEqual(call("GET", f"{self.base}/api/runs/deadbeef/events")[0], 404)

    def test_3_a_full_run_streams_to_the_end(self):
        status, body = call("POST", f"{self.base}/api/runs", {"text": "CMOS inverter, delay under 40 ps"})
        self.assertEqual(status, 202)
        run_id = body["run_id"]

        events = []
        with urllib.request.urlopen(f"{self.base}/api/runs/{run_id}/events", timeout=30) as resp:
            self.assertIn("text/event-stream", resp.headers["Content-Type"])
            for raw in resp:                               # ends when the server closes the stream
                line = raw.decode().strip()
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds[0], "started")
        self.assertEqual(kinds.count("iteration"), 2)
        self.assertEqual(kinds[-1], "report")
        self.assertEqual(events[-1]["report"]["status"], "COMPLETE")

        status, run = call("GET", f"{self.base}/api/runs/{run_id}")
        self.assertEqual(status, 200)
        self.assertEqual(run["overall"], "COMPLETE")
        self.assertEqual(len(run["events"]), len(events))
        self.assertEqual(run["report"]["status"], "COMPLETE")

        status, rows = call("GET", f"{self.base}/api/runs")
        self.assertEqual(status, 200)
        self.assertIn(run_id, [r["run_id"] for r in rows])


    def test_4_trap_flag(self):
        self.manager._llm = FakeLLM([intent_reply()])      # a trap run needs the intent call only
        self.assertEqual(call("POST", f"{self.base}/api/runs", {"text": "x", "trap": "banana"})[0], 422)
        status, body = call("POST", f"{self.base}/api/runs",
                            {"text": "CMOS inverter, delay under 40 ps at 20 fF", "trap": True})
        self.assertEqual(status, 202)
        run_id = body["run_id"]
        self.assertTrue(self.manager.get(run_id).wait(30))
        status, run = call("GET", f"{self.base}/api/runs/{run_id}")
        self.assertEqual(status, 200)
        self.assertTrue(run["trap"])
        self.assertEqual(run["overall"], "FAILED")
        designs = [e for e in run["events"] if e["type"] == "design"]
        self.assertEqual(len(designs), 1)
        self.assertEqual(designs[0]["trap"]["metric"], "tpd")
        status, rows = call("GET", f"{self.base}/api/runs")
        self.assertTrue([r for r in rows if r["run_id"] == run_id][0]["trap"])


if __name__ == "__main__":
    unittest.main()
