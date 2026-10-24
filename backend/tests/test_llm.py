"""LLM client tests. No network and no API key: a local fake HTTP server stands in
for Token Factory, and FakeLLM covers the offline path. These verify OUR client
(request shape, retry, validation, logging, error handling), not Nebius itself.
"""
import json
import os
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from pydantic import BaseModel

from app.llm import (
    DEFAULT_BASE_URL,
    NEBIUS_BASE_URL,
    FakeLLM,
    LLMError,
    LLMOutputError,
    NebiusClient,
    OpenAICompatClient,
    extract_json,
    list_models,
)

KEY = "test-secret-key-123"


class Answer(BaseModel):
    value: int
    label: str


def chat(content, prompt=11, completion=7, **message_extra):
    return {
        "choices": [{"message": {"role": "assistant", "content": content, **message_extra}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                  "total_tokens": prompt + completion},
    }


GOOD = json.dumps({"value": 3, "label": "ok"})


class _Handler(BaseHTTPRequestHandler):
    script: list = []
    seen: list = []

    def _serve(self, body):
        type(self).seen.append({
            "method": self.command, "path": self.path,
            "auth": self.headers.get("Authorization"), "body": body,
        })
        status, payload, delay = type(self).script.pop(0)
        time.sleep(delay)
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except OSError:
            pass  # client gave up (timeout test)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self._serve(json.loads(self.rfile.read(length) or b"{}"))

    def do_GET(self):
        self._serve(None)

    def log_message(self, *args):
        pass


class HttpClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}/v1/"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _Handler.script = []
        _Handler.seen = []
        self.client = NebiusClient(api_key=KEY, model="nvidia/test-model", base_url=self.base,
                                   timeout_s=5)

    def ask(self, client=None):
        return (client or self.client).complete_json(
            "test", [{"role": "user", "content": "give me an answer"}], Answer)

    def test_success_request_shape_and_logging(self):
        _Handler.script = [(200, chat(GOOD), 0)]
        parsed, record = self.ask()
        self.assertEqual(parsed, Answer(value=3, label="ok"))
        req = _Handler.seen[0]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["auth"], f"Bearer {KEY}")
        self.assertEqual(req["body"]["model"], "nvidia/test-model")
        self.assertEqual(req["body"]["response_format"], {"type": "json_object"})
        self.assertIn("JSON Schema", req["body"]["messages"][0]["content"])
        self.assertEqual(req["body"]["messages"][-1]["content"], "give me an answer")
        self.assertTrue(record.ok)
        self.assertEqual((record.prompt_tokens, record.completion_tokens), (11, 7))
        self.assertEqual(record.retries, 0)
        self.assertGreaterEqual(record.latency_s, 0)
        self.assertEqual(self.client.calls, [record])

    def test_retry_once_on_invalid_json_then_succeed(self):
        _Handler.script = [(200, chat("sorry, here you go: not json"), 0), (200, chat(GOOD), 0)]
        parsed, record = self.ask()
        self.assertEqual(parsed.value, 3)
        self.assertEqual(record.retries, 1)
        self.assertEqual((record.prompt_tokens, record.completion_tokens), (22, 14))  # summed
        self.assertEqual(len(_Handler.seen), 2)
        retry_messages = _Handler.seen[1]["body"]["messages"]
        self.assertIn("invalid", retry_messages[-1]["content"])
        self.assertEqual(retry_messages[-2]["role"], "assistant")

    def test_retry_on_schema_mismatch(self):
        wrong = json.dumps({"value": "not-an-int", "label": "x"})
        _Handler.script = [(200, chat(wrong), 0), (200, chat(GOOD), 0)]
        parsed, record = self.ask()
        self.assertEqual(parsed.label, "ok")
        self.assertEqual(record.retries, 1)
        self.assertIn("value", _Handler.seen[1]["body"]["messages"][-1]["content"])

    def test_fails_after_exactly_one_retry(self):
        _Handler.script = [(200, chat("nope"), 0), (200, chat("still nope"), 0)]
        with self.assertRaises(LLMOutputError):
            self.ask()
        self.assertEqual(len(_Handler.seen), 2)
        record = self.client.calls[-1]
        self.assertFalse(record.ok)
        self.assertTrue(record.error)

    def test_http_error_not_retried_and_key_never_leaks(self):
        _Handler.script = [(401, {"error": f"bad key {KEY}"}, 0)]
        with self.assertRaises(LLMError) as ctx:
            self.ask()
        self.assertIn("401", str(ctx.exception))
        self.assertNotIn(KEY, str(ctx.exception))
        self.assertEqual(len(_Handler.seen), 1)
        self.assertNotIn(KEY, repr(self.client.calls[-1]))
        self.assertFalse(self.client.calls[-1].ok)

    def test_refusal_is_an_error(self):
        _Handler.script = [(200, chat(None, refusal="I can't help with that"), 0)]
        with self.assertRaises(LLMError) as ctx:
            self.ask()
        self.assertIn("refused", str(ctx.exception))

    def test_timeout_is_an_error(self):
        client = NebiusClient(api_key=KEY, model="m", base_url=self.base, timeout_s=0.3)
        _Handler.script = [(200, chat(GOOD), 1.5)]
        with self.assertRaises(LLMError) as ctx:
            self.ask(client)
        self.assertIn("could not reach", str(ctx.exception))

    def test_malformed_response_shape(self):
        _Handler.script = [(200, {"unexpected": True}, 0)]
        with self.assertRaises(LLMError):
            self.ask()

    def test_think_block_and_fences_are_tolerated(self):
        wrapped = "<think>let me reason...</think>\n```json\n" + GOOD + "\n```"
        _Handler.script = [(200, chat(wrapped), 0)]
        parsed, record = self.ask()
        self.assertEqual(parsed.value, 3)
        self.assertEqual(record.retries, 0)

    def test_list_models(self):
        _Handler.script = [(200, {"data": [{"id": "nvidia/nemotron-x"}, {"id": "meta/llama"}]}, 0)]
        ids = list_models(api_key=KEY, base_url=self.base)
        self.assertEqual(ids, ["meta/llama", "nvidia/nemotron-x"])
        self.assertEqual(_Handler.seen[0]["method"], "GET")
        self.assertEqual(_Handler.seen[0]["path"], "/v1/models")

    def test_json_mode_off_omits_response_format(self):
        client = OpenAICompatClient(api_key=KEY, model="m", base_url=self.base, json_mode=False)
        _Handler.script = [(200, chat(GOOD), 0)]
        self.ask(client)
        self.assertNotIn("response_format", _Handler.seen[0]["body"])

    def test_localhost_without_key_sends_no_auth_header(self):
        with mock.patch.dict(os.environ):
            for name in _ENV_NAMES:
                os.environ.pop(name, None)
            client = OpenAICompatClient(model="llama3.1:8b", base_url=self.base)
        _Handler.script = [(200, chat(GOOD), 0)]
        parsed, _ = self.ask(client)
        self.assertEqual(parsed.value, 3)
        self.assertIsNone(_Handler.seen[0]["auth"])

    def test_list_models_without_key_on_localhost(self):
        with mock.patch.dict(os.environ):
            for name in _ENV_NAMES:
                os.environ.pop(name, None)
            _Handler.script = [(200, {"data": [{"id": "llama3.1:8b"}]}, 0)]
            self.assertEqual(list_models(base_url=self.base), ["llama3.1:8b"])
        self.assertIsNone(_Handler.seen[0]["auth"])

    def test_connection_refused_on_localhost_hints_at_ollama(self):
        with socket.socket() as sock:  # grab a port that nothing listens on
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        client = OpenAICompatClient(model="m", base_url=f"http://127.0.0.1:{port}/v1", timeout_s=2)
        with self.assertRaises(LLMError) as ctx:
            self.ask(client)
        self.assertIn("could not reach", str(ctx.exception))
        self.assertIn("ollama", str(ctx.exception).lower())


_ENV_NAMES = (
    "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_TIMEOUT_S", "LLM_JSON_MODE",
    "NEBIUS_BASE_URL", "NEBIUS_API_KEY", "NEBIUS_MODEL",
)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in _ENV_NAMES:
            os.environ.pop(name, None)

    def test_missing_model_is_an_error(self):
        with self.assertRaises(LLMError):
            OpenAICompatClient()
        with self.assertRaises(LLMError):
            OpenAICompatClient(api_key="k")

    def test_default_is_local_ollama_and_needs_no_key(self):
        client = OpenAICompatClient(model="llama3.1:8b")
        self.assertEqual(DEFAULT_BASE_URL, "http://localhost:11434/v1")
        self.assertEqual(client.url, "http://localhost:11434/v1/chat/completions")

    def test_remote_endpoint_requires_a_key(self):
        with self.assertRaises(LLMError):
            OpenAICompatClient(model="m", base_url="https://api.example.com/v1")
        OpenAICompatClient(model="m", api_key="k", base_url="https://api.example.com/v1")

    def test_nebius_url_constant_kept_for_reference(self):
        self.assertEqual(NEBIUS_BASE_URL, "https://api.tokenfactory.nebius.com/v1/")
        client = OpenAICompatClient(api_key="k", model="m", base_url=NEBIUS_BASE_URL)
        self.assertEqual(client.url, "https://api.tokenfactory.nebius.com/v1/chat/completions")

    def test_reads_llm_env_vars(self):
        os.environ.update(LLM_API_KEY="k", LLM_MODEL="qwen2.5:7b", LLM_BASE_URL="http://h/v1")
        client = OpenAICompatClient()
        self.assertEqual(client.model, "qwen2.5:7b")
        self.assertEqual(client.url, "http://h/v1/chat/completions")

    def test_nebius_env_vars_still_work_as_fallback(self):
        os.environ.update(NEBIUS_API_KEY="k", NEBIUS_MODEL="nvidia/x", NEBIUS_BASE_URL="http://h/v1")
        client = OpenAICompatClient()
        self.assertEqual(client.model, "nvidia/x")
        self.assertEqual(client.url, "http://h/v1/chat/completions")

    def test_llm_vars_win_over_nebius_vars(self):
        os.environ.update(LLM_MODEL="new", NEBIUS_MODEL="old")
        self.assertEqual(OpenAICompatClient().model, "new")

    def test_timeout_defaults_and_override(self):
        self.assertEqual(OpenAICompatClient(model="m").timeout_s, 120.0)  # local model may load slowly
        self.assertEqual(
            OpenAICompatClient(model="m", api_key="k", base_url="https://x/v1").timeout_s, 60.0)
        os.environ["LLM_TIMEOUT_S"] = "300"
        self.assertEqual(OpenAICompatClient(model="m").timeout_s, 300.0)
        os.environ["LLM_TIMEOUT_S"] = "soon"
        with self.assertRaises(LLMError):
            OpenAICompatClient(model="m")

    def test_json_mode_env_switch(self):
        self.assertTrue(OpenAICompatClient(model="m").json_mode)
        os.environ["LLM_JSON_MODE"] = "0"
        self.assertFalse(OpenAICompatClient(model="m").json_mode)

    def test_nebius_client_name_is_an_alias(self):
        self.assertIs(NebiusClient, OpenAICompatClient)


class ExtractJsonTests(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('<think>hmm {x}</think>{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('Sure! {"a": 1} Hope that helps'), {"a": 1})

    def test_garbage_raises(self):
        with self.assertRaises(json.JSONDecodeError):
            extract_json("no json here")


class FakeLLMTests(unittest.TestCase):
    def test_scripted_replies_and_logging(self):
        fake = FakeLLM([{"value": 1, "label": "a"}])
        parsed, record = fake.complete_json("t", [{"role": "user", "content": "hi"}], Answer)
        self.assertEqual(parsed.value, 1)
        self.assertTrue(record.ok)
        self.assertEqual(fake.model, "fake-nemotron")
        self.assertEqual(len(fake.requests), 1)

    def test_fake_uses_same_retry_path(self):
        fake = FakeLLM(["garbage", {"value": 2, "label": "b"}])
        parsed, record = fake.complete_json("t", [{"role": "user", "content": "hi"}], Answer)
        self.assertEqual((parsed.value, record.retries), (2, 1))

    def test_fake_fails_after_one_retry(self):
        fake = FakeLLM(["garbage", "garbage again"])
        with self.assertRaises(LLMOutputError):
            fake.complete_json("t", [{"role": "user", "content": "hi"}], Answer)

    def test_exhausted_script_and_scripted_exception(self):
        with self.assertRaises(LLMError):
            FakeLLM([]).complete_json("t", [{"role": "user", "content": "hi"}], Answer)
        with self.assertRaises(LLMError):
            FakeLLM([LLMError("boom")]).complete_json(
                "t", [{"role": "user", "content": "hi"}], Answer)


if __name__ == "__main__":
    unittest.main()
