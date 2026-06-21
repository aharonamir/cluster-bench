"""FastAPI server — POST /api/run, GET /api/runs[/{id}], GET /, WS /ws (T040/T041).

Builds a RunConfig from the request body, launches the orchestrator in a
background task (so /api/run returns immediately), and persists the RunReport
on completion (T042). The WebSocketHub is the orchestrator's emitter, so all
events flow into the bounded replay buffer and out to live subscribers
(FR-21/FR-22).

Re-entrancy: a single in-flight run per server. A second POST /api/run while
one is active returns 409 (FR-15: POST is the only way to start; this guard is
the server-side re-entrancy lock). Runner + source are produced by injected
factories so tests can swap mock_litellm for a fake without going over HTTP.
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from clusterbench.metrics.base import MetricsSource
from clusterbench.metrics.litellm import LiteLLMSource
from clusterbench.miniswerunner import MockRunner, Runner
from clusterbench.models import RunConfig, RunReport, TaskSlice
from clusterbench.orchestrator import Orchestrator
from clusterbench.web.hub import WebSocketHub
from clusterbench.web.persistence import (
    list_reports,
    load_report,
    save_report,
)

# ---------------------------------------------------------------------------
# Request models — pydantic validates the POST body. Field defaults match
# RunConfig's dataclass defaults; the server fills what the client omits.
# ---------------------------------------------------------------------------


class TaskSliceBody(BaseModel):
    subset: str = "verified"
    split: str = "test"
    n: int = 5
    pinned_instance_ids: list[str] = Field(default_factory=list)


class StartRunBody(BaseModel):
    """POST /api/run body. run_id is server-generated when omitted."""

    name: str = ""
    mode: Literal["sweep", "soak"] = "sweep"
    levels: list[int] = Field(default_factory=lambda: [1, 4, 8, 16])
    soak_duration_s: float = 1800.0
    task_slice: TaskSliceBody = Field(default_factory=TaskSliceBody)
    # Guards are intentionally loose in pydantic — RunConfig tolerates all-None.
    guards: dict[str, float | None] = Field(default_factory=dict)
    scrape_interval_s: float = 1.0
    model: str = "gpt-4o-mini"
    streaming: bool = True
    step_limit: int = 0


# ---------------------------------------------------------------------------
# Factory protocols — inject test doubles without HTTP / subprocess plumbing.
# ---------------------------------------------------------------------------


class RunnerFactory(Protocol):
    def __call__(
        self, *, config: RunConfig, pinned: list[str]
    ) -> Runner: ...


class SourceFactory(Protocol):
    def __call__(self, *, config: RunConfig) -> MetricsSource: ...


def default_runner_factory(
    *, base_url: str, runner_root: Path | str
) -> RunnerFactory:
    """Real mock-path runner factory: MockRunner hits mock_litellm at `base_url`."""

    def make(*, config: RunConfig, pinned: list[str]) -> Runner:
        return MockRunner(
            base_url=base_url,
            instance_ids=pinned,
            turns_per_instance=3,
            model=config.miniswe.model,
            streaming=config.miniswe.streaming,
            runner_root=runner_root,
        )

    return make


def default_source_factory() -> SourceFactory:
    """Real source factory: LiteLLMSource against the config's metrics URL."""

    def make(*, config: RunConfig) -> MetricsSource:
        return LiteLLMSource(
            metrics_url=config.litellm_metrics_url,
            scrape_interval_s=config.scrape_interval_s,
        )

    return make


# ---------------------------------------------------------------------------
# Server state — single active run + bounded hub + results dir.
# ---------------------------------------------------------------------------


class _ServerState:
    """Mutable per-app state. Held as a single object so the routes close over
    one well-known thing instead of a forest of module globals."""

    def __init__(
        self,
        *,
        results_dir: Path | str,
        runner_factory: RunnerFactory,
        source_factory: SourceFactory,
        hub: WebSocketHub,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.runner_factory = runner_factory
        self.source_factory = source_factory
        self.hub = hub
        self._lock = asyncio.Lock()
        self._active_run_id: str | None = None
        self._active_task: asyncio.Task[RunReport] | None = None

    @property
    def active_run_id(self) -> str | None:
        return self._active_run_id

    async def try_start(self, config: RunConfig, *, pinned: list[str]) -> str:
        """Try to claim the run slot. Raises HTTPException(409) if busy.

        `pinned` is the already-resolved slice (AC-10). It's set on the config
        (so the orchestrator reuses it) and passed to the runner factory.
        """
        config.task_slice.pinned_instance_ids = list(pinned)
        async with self._lock:
            if self._active_run_id is not None and not self._is_task_done():
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "run_already_active",
                        "active_run_id": self._active_run_id,
                    },
                )
            self._active_run_id = config.run_id
            runner = self.runner_factory(config=config, pinned=list(pinned))
            source = self.source_factory(config=config)
            orchestrator = Orchestrator(
                config,
                source=source,
                runner=runner,
                emitter=self.hub,
            )
            self._active_task = asyncio.create_task(
                self._drive(orchestrator, runner), name=f"run-{config.run_id}"
            )
            return config.run_id

    def _is_task_done(self) -> bool:
        return self._active_task is not None and self._active_task.done()

    async def _drive(self, orchestrator: Orchestrator, runner: Runner) -> RunReport:
        """Run orchestrator.run(), then persist + clear the active slot."""
        try:
            report = await orchestrator.run()
        finally:
            # Release the runner's resources (e.g. close httpx clients).
            aclose = getattr(runner, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass
        try:
            save_report(self.results_dir, report)
        finally:
            async with self._lock:
                if self._active_run_id == orchestrator.config.run_id:
                    self._active_run_id = None
                    self._active_task = None
        return report


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _static_dir() -> Path:
    """Static assets directory, next to this file."""
    return Path(__file__).parent / "static"


def create_app(
    *,
    results_dir: Path | str = "results",
    runner_factory: RunnerFactory | None = None,
    source_factory: SourceFactory | None = None,
    hub: WebSocketHub | None = None,
    index_html: str | None = None,
    serve_static: bool = True,
) -> FastAPI:
    """Build a FastAPI app. Defaults wire the mock path; tests pass their own
    factories + hub to skip HTTP plumbing.

    `index_html` overrides the served `/` payload (tests use this for an
    in-line page). `serve_static=False` skips the /static mount.
    """
    if runner_factory is None:
        runner_factory = default_runner_factory(
            base_url="http://localhost:4000/v1",
            runner_root=Path(results_dir) / "miniswe",
        )
    if source_factory is None:
        source_factory = default_source_factory()
    if hub is None:
        hub = WebSocketHub()

    state = _ServerState(
        results_dir=results_dir,
        runner_factory=runner_factory,
        source_factory=source_factory,
        hub=hub,
    )

    app = FastAPI(title="clusterbench")
    app.state.clusterbench = state

    static_dir = _static_dir()
    if serve_static and static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    _register_routes(
        app,
        state,
        index_html=index_html or _resolve_index_html(static_dir),
    )
    return app


def _resolve_index_html(static_dir: Path) -> str:
    """Read the dashboard HTML from disk if it exists; fall back to the
    placeholder so the server is usable before Phase 5 ships app.js."""
    index_path = static_dir / "index.html"
    if index_path.is_file():
        return index_path.read_text()
    return _DEFAULT_INDEX_HTML


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _register_routes(
    app: FastAPI, state: _ServerState, *, index_html: str
) -> None:
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "active_run_id": state.active_run_id,
            "n_subscribers": state.hub.n_subscribers,
        }

    @app.post("/api/run")
    async def start_run(body: StartRunBody) -> JSONResponse:
        run_id = uuid.uuid4().hex[:12]
        config = _build_run_config(run_id, body)
        pinned = _resolve_pinned(config)
        await state.try_start(config, pinned=pinned)
        return JSONResponse(
            status_code=202,
            content={
                "run_id": run_id,
                "pinned_instance_ids": pinned,
                "config": config.to_dict(),
            },
        )

    @app.get("/api/runs")
    async def list_runs() -> dict[str, Any]:
        reports = list_reports(state.results_dir)
        return {
            "run_ids": [r.run_id for r in reports],
            "runs": [
                {
                    "run_id": r.run_id,
                    "name": r.name,
                    "mode": r.config.mode.value,
                    "levels": list(r.config.levels),
                    "finished_at": r.finished_at,
                    "n_levels": len(r.levels),
                    "wire_metrics_available": r.wire_metrics_available,
                    "ttft_available": r.ttft_available,
                    "knee": r.knee,
                }
                for r in reports
            ],
        }

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        report = load_report(state.results_dir, run_id)
        if report is None:
            raise HTTPException(status_code=404, detail="run_not_found")
        return report.to_dict()

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(index_html)

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await state.hub.subscribe(ws)
        try:
            while True:
                # We don't act on incoming messages; just keep the loop open
                # until the client disconnects. The hub is the only writer.
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await state.hub.unsubscribe(ws)


# ---------------------------------------------------------------------------
# RunConfig assembly from the request body
# ---------------------------------------------------------------------------


def _build_run_config(run_id: str, body: StartRunBody) -> RunConfig:
    from clusterbench.models import (
        DegradationGuard,
        LoadMode,
        MiniSweConfig,
    )

    return RunConfig(
        run_id=run_id,
        name=body.name,
        mode=LoadMode(body.mode),
        levels=list(body.levels),
        soak_duration_s=body.soak_duration_s,
        task_slice=TaskSlice(
            subset=body.task_slice.subset,
            split=body.task_slice.split,
            n=body.task_slice.n,
            pinned_instance_ids=list(body.task_slice.pinned_instance_ids),
        ),
        guards=DegradationGuard(**{k: v for k, v in body.guards.items()}),
        scrape_interval_s=body.scrape_interval_s,
        miniswe=MiniSweConfig(
            model=body.model,
            streaming=body.streaming,
            step_limit=body.step_limit,
        ),
    )


def _resolve_pinned(config: RunConfig) -> list[str]:
    """Pin the slice once. Reused at every level (AC-10)."""
    from clusterbench.miniswerunner import pin_slice

    if config.task_slice.pinned_instance_ids:
        return list(config.task_slice.pinned_instance_ids)
    return pin_slice(
        n=config.task_slice.n,
        subset=config.task_slice.subset,
        split=config.task_slice.split,
        mock=True,
    )


# ---------------------------------------------------------------------------
# Placeholder index — Phase 5 replaces this with the real dashboard.
# ---------------------------------------------------------------------------

_DEFAULT_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ClusterBench</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.4; }
  code { background: #f4f4f4; padding: 0.1rem 0.3rem; border-radius: 3px; }
</style>
</head>
<body>
<h1>ClusterBench</h1>
<p>Dashboard ships in Phase 5. Until then, the JSON API is live:</p>
<ul>
  <li><code>POST /api/run</code> — start a sweep/soak</li>
  <li><code>GET /api/runs</code> — list saved reports</li>
  <li><code>GET /api/runs/{run_id}</code> — fetch one report</li>
  <li><code>WS /ws</code> — live telemetry stream</li>
</ul>
</body>
</html>
"""


__all__ = [
    "create_app",
    "StartRunBody",
    "default_runner_factory",
    "default_source_factory",
]
