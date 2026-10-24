"""Run manager: starts verification runs in background threads and keeps their event history.

No web framework in here, so everything is testable with fakes. main.py is a thin FastAPI wrapper.

  * One run is executed at a time (one CPU, one local model). Others wait as "queued".
  * Every event from loop.run_spec is stored, so a browser that connects late (or reconnects)
    replays the whole history and then follows live. Events are sent as Server-Sent Events.
  * The final RunReport is attached as a last {"type": "report"} event and saved to
    <history_dir>/<run_id>.json so history survives a restart.
  * Run ids are 8 hex characters and are validated before touching the disk (no path tricks).
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .llm import LLMClient
from .loop import Simulate, Verify, run_spec

MAX_TEXT_CHARS = 2000
_ID = re.compile(r"^[0-9a-f]{8}$")


class RunHandle:
    def __init__(self, text: str) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.text = text
        self.status = "queued"            # queued -> running -> done | error
        self.overall: str | None = None   # COMPLETE / FAILED / INCOMPLETE, from the verifier only
        self.created_at = time.time()
        self.events: list[dict[str, Any]] = []
        self.report: dict[str, Any] | None = None
        self.finished = False
        self.cond = threading.Condition()

    def push(self, event: dict[str, Any]) -> None:
        with self.cond:
            self.events.append(event)
            self.cond.notify_all()

    def close(self) -> None:
        with self.cond:
            self.finished = True
            self.cond.notify_all()

    def wait(self, timeout: float | None = None) -> bool:
        with self.cond:
            return self.cond.wait_for(lambda: self.finished, timeout=timeout)

    def summary(self) -> dict[str, Any]:
        return {"run_id": self.id, "text": self.text, "status": self.status,
                "overall": self.overall, "created_at": self.created_at}


def sse(event: dict[str, Any], index: int) -> str:
    return f"id: {index}\ndata: {json.dumps(event, default=str)}\n\n"


class RunManager:
    def __init__(
        self,
        simulate: Simulate,
        verify: Verify,
        llm: LLMClient | None = None,
        history_dir: str | Path | None = None,
        max_in_memory: int = 50,
    ) -> None:
        self._simulate = simulate
        self._verify = verify
        self._llm = llm                   # None: run_spec builds one from the environment
        self._dir = Path(history_dir) if history_dir else None
        self._max = max_in_memory
        self._runs: OrderedDict[str, RunHandle] = OrderedDict()
        self._lock = threading.Lock()     # guards _runs
        self._exec = threading.Lock()     # one run at a time

    # ---- starting and finding runs -------------------------------------
    def start(self, text: str) -> RunHandle:
        handle = RunHandle(text if isinstance(text, str) else "")
        with self._lock:
            self._runs[handle.id] = handle
            self._evict()
        threading.Thread(target=self._work, args=(handle,), daemon=True, name=f"run-{handle.id}").start()
        return handle

    def get(self, run_id: str) -> RunHandle | None:
        with self._lock:
            return self._runs.get(run_id)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            live = [h.summary() for h in reversed(self._runs.values())]
        seen = {row["run_id"] for row in live}
        saved = [row for row in self._saved_summaries() if row["run_id"] not in seen]
        return (live + saved)[:limit]

    def load_saved(self, run_id: str) -> dict[str, Any] | None:
        """The saved final report of an earlier run, or None. Ids that are not 8 hex chars never reach the disk."""
        if self._dir is None or not _ID.match(run_id or ""):
            return None
        path = self._dir / f"{run_id}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    # ---- streaming -----------------------------------------------------
    def stream(self, handle: RunHandle, heartbeat_s: float = 15.0) -> Iterator[str]:
        """Server-Sent Events: replay everything so far, then follow live until the run ends."""
        index = 0
        while True:
            with handle.cond:
                if index >= len(handle.events) and not handle.finished:
                    handle.cond.wait(timeout=heartbeat_s)
                fresh = handle.events[index:]
                finished = handle.finished
            if fresh:
                for event in fresh:
                    yield sse(event, index)
                    index += 1
            elif finished:
                return
            else:
                yield ": ping\n\n"

    # ---- the worker ----------------------------------------------------
    def _work(self, handle: RunHandle) -> None:
        try:
            with self._exec:
                handle.status = "running"
                handle.push({"type": "started", "run_id": handle.id})
                report = run_spec(handle.text, self._llm, simulate=self._simulate,
                                  verify=self._verify, on_event=handle.push, run_id=handle.id)
                data = report.model_dump(mode="json")
                handle.report = data
                handle.overall = report.status.value   # straight from the verifier's report
                handle.push({"type": "report", "report": data})
                handle.status = "done"
                self._save(handle, data)
        except Exception as exc:  # a crashed run must end visibly, never hang a browser
            handle.status = "error"
            handle.push({"type": "error", "stage": "server", "message": f"{type(exc).__name__}: {exc}"[:300]})
        finally:
            handle.close()

    # ---- history -------------------------------------------------------
    def _save(self, handle: RunHandle, data: dict[str, Any]) -> None:
        if self._dir is None:
            return
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            payload = {**handle.summary(), "report": data}
            (self._dir / f"{handle.id}.json").write_text(json.dumps(payload, default=str), encoding="utf-8")
        except OSError:
            handle.push({"type": "warning", "message": "could not save this run to disk"})

    def _saved_summaries(self) -> list[dict[str, Any]]:
        if self._dir is None or not self._dir.is_dir():
            return []
        rows = []
        for path in sorted(self._dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if not _ID.match(path.stem):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rows.append({k: data.get(k) for k in ("run_id", "text", "status", "overall", "created_at")})
            except (OSError, ValueError):
                continue
        return rows

    def _evict(self) -> None:
        while len(self._runs) > self._max:
            oldest = next(iter(self._runs.values()))
            if not oldest.finished:
                break
            self._runs.popitem(last=False)
