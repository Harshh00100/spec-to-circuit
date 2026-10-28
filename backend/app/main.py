"""HTTP API. Start it from the backend folder:

    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

Endpoints
    GET  /                        the web page (frontend/index.html)
    GET  /api/health              is ngspice found, which model is configured
    POST /api/runs                {"text": "..."} -> 202 {"run_id": "..."}   (starts a run in the background)
    GET  /api/runs                recent runs, newest first
    GET  /api/runs/{id}           summary, all events so far, and the final report when finished
    GET  /api/runs/{id}/events    live Server-Sent Events: replay + follow until the run ends

All pass/fail decisions are made in verifier.py; this file only moves events around.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .runs import MAX_TEXT_CHARS, RunManager
from .simulator import _resolve_cmd, run_ngspice
from .verifier import verify

_ID = re.compile(r"^[0-9a-f]{8}$")
FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "index.html"   # <repo>/frontend/index.html
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]   # the Vite dev server


class RunRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


def create_app(manager: RunManager | None = None) -> FastAPI:
    manager = manager or RunManager(
        simulate=run_ngspice,
        verify=verify,
        history_dir=Path(os.environ.get("RUNS_DIR", "runs")),
    )
    app = FastAPI(title="Spec-to-Circuit", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=DEV_ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["*"])

    def find(run_id: str):
        if not _ID.match(run_id):
            raise HTTPException(status_code=404, detail="no such run")
        return manager.get(run_id)

    @app.get("/", include_in_schema=False)
    def page():
        if not FRONTEND.is_file():
            return HTMLResponse("<h1>Spec to Circuit</h1><p>The API is running, but frontend/index.html was not "
                                "found next to the backend folder. See /docs for the API.</p>", status_code=404)
        return FileResponse(FRONTEND, media_type="text/html")

    @app.get("/api/health")
    def health() -> dict:
        try:
            ngspice_found = _resolve_cmd(None) is not None
        except Exception:
            ngspice_found = False
        return {"ok": True, "ngspice_found": ngspice_found,
                "model": os.environ.get("LLM_MODEL") or os.environ.get("NEBIUS_MODEL"),
                "base_url": os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")}

    @app.post("/api/runs", status_code=202)
    def start_run(request: RunRequest) -> dict:
        handle = manager.start(request.text.strip())
        return {"run_id": handle.id, "status": handle.status}

    @app.get("/api/runs")
    def list_runs(limit: int = 50) -> list[dict]:
        return manager.list(max(1, min(limit, 200)))

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        handle = find(run_id)
        if handle is not None:
            return {**handle.summary(), "events": list(handle.events), "report": handle.report}
        saved = manager.load_saved(run_id)
        if saved is None:
            raise HTTPException(status_code=404, detail="no such run")
        return {**{k: saved.get(k) for k in ("run_id", "text", "status", "overall", "created_at")},
                "events": [], "report": saved.get("report")}

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str) -> StreamingResponse:
        handle = find(run_id)
        if handle is None:
            raise HTTPException(status_code=404, detail="no such run (finished runs from before a restart "
                                                        "have a report but no live events)")
        return StreamingResponse(
            manager.stream(handle), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = create_app()
