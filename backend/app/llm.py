"""LLM access: any OpenAI-compatible chat endpoint (Ollama by default) plus an offline fake.

Rules enforced for EVERY call, real or fake, because they share one code path:
  * JSON mode (response_format json_object) and the pydantic schema is spelled
    out in the prompt.
  * The reply is validated against a pydantic model.
  * On a parse or validation failure there is exactly ONE retry, with the
    error fed back to the model. After that, LLMOutputError.
  * Latency and token usage are recorded per call in `client.calls`.

Uses only the standard library (urllib), so no extra dependency is needed.
The LLM never decides pass/fail; it only produces structured data.

Configuration (environment variables; the NEBIUS_* names still work as fallbacks):
    LLM_BASE_URL   default http://localhost:11434/v1  (Ollama)
    LLM_MODEL      required, e.g. the name shown by `ollama list`
    LLM_API_KEY    optional for localhost, required for remote endpoints
    LLM_TIMEOUT_S  default 120 for localhost, 60 otherwise
    LLM_JSON_MODE  set to 0 to omit response_format (some models/providers reject it)

CLI helper to list the model IDs your endpoint offers:
    python -m app.llm models
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .schemas import LLMCallRecord

# Ollama's OpenAI-compatible endpoint (local, free, no key).
DEFAULT_BASE_URL = "http://localhost:11434/v1"
# Target deployment, verified in the docs: https://docs.tokenfactory.nebius.com/quickstart
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

T = TypeVar("T", bound=BaseModel)
Message = dict[str, str]


class LLMError(RuntimeError):
    """Configuration, transport, HTTP or refusal problems (not retried)."""


class LLMOutputError(LLMError):
    """The model's reply could not be parsed/validated, even after one retry."""


class LLMClient(Protocol):
    model: str
    calls: list[LLMCallRecord]

    def complete_json(
        self, purpose: str, messages: list[Message], schema: type[T]
    ) -> tuple[T, LLMCallRecord]: ...


# --------------------------------------------------------------------------
# JSON extraction / validation helpers
# --------------------------------------------------------------------------
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating <think> blocks and code fences."""
    cleaned = _THINK.sub("", text).strip()
    fence = _FENCE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _describe_failure(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"not valid JSON ({exc.msg} at character {exc.pos})"
    if isinstance(exc, ValidationError):
        parts = [
            f"{'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}"
            for err in exc.errors()[:5]
        ]
        return "JSON did not match the schema: " + "; ".join(parts)
    return str(exc)


def _schema_instruction(schema: type[BaseModel]) -> Message:
    return {
        "role": "system",
        "content": (
            "Reply with a single JSON object only: no prose, no markdown fences. "
            "It must validate against this JSON Schema:\n"
            + json.dumps(schema.model_json_schema())
        ),
    }


# --------------------------------------------------------------------------
# Shared client logic (retry, validation, logging)
# --------------------------------------------------------------------------
class _BaseClient:
    model: str

    def __init__(self) -> None:
        self.calls: list[LLMCallRecord] = []

    def _chat(self, messages: list[Message]) -> tuple[str, int | None, int | None]:
        raise NotImplementedError

    def _record(self, **kwargs: Any) -> LLMCallRecord:
        record = LLMCallRecord(model=self.model, **kwargs)
        self.calls.append(record)
        return record

    def complete_json(
        self, purpose: str, messages: list[Message], schema: type[T]
    ) -> tuple[T, LLMCallRecord]:
        started = time.perf_counter()
        convo: list[Message] = [_schema_instruction(schema), *messages]
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        last_error = ""

        def add(current: int | None, new: int | None) -> int | None:
            return current if new is None else (current or 0) + new

        for attempt in range(2):
            try:
                text, p, c = self._chat(convo)
            except LLMError as exc:
                self._record(
                    purpose=purpose, latency_s=time.perf_counter() - started,
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    retries=attempt, ok=False, error=str(exc),
                )
                raise
            prompt_tokens, completion_tokens = add(prompt_tokens, p), add(completion_tokens, c)
            try:
                parsed = schema.model_validate(extract_json(text))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = _describe_failure(exc)
                if attempt == 0:
                    convo = [
                        *convo,
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": (
                                f"Your reply was invalid: {last_error}. Reply again with "
                                "ONLY a JSON object that matches the schema."
                            ),
                        },
                    ]
                continue
            record = self._record(
                purpose=purpose, latency_s=time.perf_counter() - started,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                retries=attempt, ok=True,
            )
            return parsed, record

        self._record(
            purpose=purpose, latency_s=time.perf_counter() - started,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            retries=1, ok=False, error=last_error,
        )
        raise LLMOutputError(f"{purpose}: model output invalid after one retry: {last_error}")


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def _redact(text: str, secret: str | None) -> str:
    return text.replace(secret, "***") if secret else text


def _env(*names: str) -> str:
    """First non-empty environment variable among `names`, else ''."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _is_local(url: str) -> bool:
    return (urllib.parse.urlparse(url).hostname or "") in ("localhost", "127.0.0.1", "::1")


def _http_json(url: str, api_key: str, body: dict | None, timeout_s: float) -> Any:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "spec-to-circuit/0.1",
    }
    if api_key:  # local servers such as Ollama need no key
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        method="GET" if body is None else "POST",
        headers=headers,
    )
    hint = " (is Ollama running? start it with `ollama serve`)" if _is_local(url) else ""
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = _redact(exc.read().decode("utf-8", "replace")[:300], api_key)
        raise LLMError(f"HTTP {exc.code} from LLM endpoint: {detail}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f"could not reach LLM endpoint ({_redact(str(exc), api_key)}){hint}") from None
    try:
        return json.loads(raw)
    except ValueError:
        raise LLMError("LLM endpoint returned a non-JSON response") from None


def _base_url(explicit: str | None) -> str:
    return (explicit or _env("LLM_BASE_URL", "NEBIUS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


class OpenAICompatClient(_BaseClient):
    """Chat completions against any OpenAI-compatible endpoint (Ollama, Nebius, ...)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool | None = None,
    ) -> None:
        super().__init__()
        base = _base_url(base_url)
        local = _is_local(base)
        self._api_key = api_key or _env("LLM_API_KEY", "NEBIUS_API_KEY")
        self.model = model or _env("LLM_MODEL", "NEBIUS_MODEL")
        if not self._api_key and not local:
            raise LLMError("LLM_API_KEY is not set (only localhost endpoints may omit it)")
        if not self.model:
            raise LLMError("LLM_MODEL is not set (run `python -m app.llm models` or `ollama list`)")
        self.url = base + "/chat/completions"
        if timeout_s is None:
            env_timeout = _env("LLM_TIMEOUT_S")
            try:
                timeout_s = float(env_timeout) if env_timeout else (120.0 if local else 60.0)
            except ValueError:
                raise LLMError(f"LLM_TIMEOUT_S is not a number: {env_timeout!r}") from None
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_mode = (_env("LLM_JSON_MODE") != "0") if json_mode is None else json_mode

    def _chat(self, messages: list[Message]) -> tuple[str, int | None, int | None]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        payload = _http_json(self.url, self._api_key, body, self.timeout_s)
        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMError("unexpected response shape from LLM endpoint") from None
        if message.get("refusal"):
            raise LLMError(f"model refused the request: {message['refusal']}")
        content = message.get("content")
        usage = payload.get("usage") or {}
        return (
            content if isinstance(content, str) else "",
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )


def list_models(
    api_key: str | None = None, base_url: str | None = None, timeout_s: float = 30.0
) -> list[str]:
    base = _base_url(base_url)
    key = api_key or _env("LLM_API_KEY", "NEBIUS_API_KEY")
    if not key and not _is_local(base):
        raise LLMError("LLM_API_KEY is not set (only localhost endpoints may omit it)")
    payload = _http_json(base + "/models", key, None, timeout_s)
    try:
        return sorted(str(item["id"]) for item in payload["data"])
    except (KeyError, TypeError):
        raise LLMError("unexpected response shape from the models endpoint") from None


# Backwards-compatible name used by the Step 2 tests.
NebiusClient = OpenAICompatClient


def client_from_env() -> OpenAICompatClient:
    """Real client configured from LLM_BASE_URL / LLM_MODEL / LLM_API_KEY."""
    return OpenAICompatClient()


# --------------------------------------------------------------------------
# Offline fake (same retry/validation path as the real client)
# --------------------------------------------------------------------------
class FakeLLM(_BaseClient):
    """Scripted replies for development and tests. Costs nothing.

    Each scripted item is a str (raw reply), a dict (sent as JSON), or an
    Exception (raised). Token counts are rough estimates (chars // 4).
    """

    def __init__(self, responses: list[str | dict | Exception], model: str = "fake-nemotron") -> None:
        super().__init__()
        self.model = model
        self._queue = list(responses)
        self.requests: list[list[Message]] = []

    def _chat(self, messages: list[Message]) -> tuple[str, int | None, int | None]:
        self.requests.append(messages)
        if not self._queue:
            raise LLMError("FakeLLM has no more scripted responses")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        text = item if isinstance(item, str) else json.dumps(item)
        prompt_estimate = sum(len(m["content"]) for m in messages) // 4
        return text, prompt_estimate, len(text) // 4


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.llm")
    sub = parser.add_subparsers(dest="cmd", required=True)
    models = sub.add_parser("models", help="list model IDs available to your API key")
    models.add_argument("--filter", default="", help="only show IDs containing this text")
    args = parser.parse_args(argv)
    try:
        ids = list_models()
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    needle = args.filter.lower()
    shown = [i for i in ids if needle in i.lower()]
    if not shown:
        print("No matching models found.")
    for model_id in shown:
        print(model_id)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
