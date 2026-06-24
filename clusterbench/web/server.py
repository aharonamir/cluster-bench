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
import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

log = logging.getLogger(__name__)

import httpx

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
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
    """POST /api/run body. run_id is server-generated when omitted.

    `model`, `streaming`, `scrape_interval_s`, and `step_limit` default to None
    here: when omitted, the server fills them from its launch config's
    run-defaults (so the model is set once at startup, not per request). An
    explicit value in the body always wins.
    """

    name: str = ""
    mode: Literal["sweep", "soak"] = "sweep"
    levels: list[int] = Field(default_factory=lambda: [1, 4, 8, 16])
    soak_duration_s: float = 1800.0
    task_slice: TaskSliceBody = Field(default_factory=TaskSliceBody)
    # Guards are intentionally loose in pydantic — RunConfig tolerates all-None.
    guards: dict[str, float | None] = Field(default_factory=dict)
    # None → fall back to the server's run-defaults (see _build_run_config).
    scrape_interval_s: float | None = None
    model: str | None = None
    streaming: bool | None = None
    step_limit: int | None = None


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
        run_defaults: dict[str, Any] | None = None,
        server_info: dict[str, Any] | None = None,
        real: bool = False,
        ssl_verify: bool = True,
        api_key: str = "",
        llm_base_url: str = "",
    ) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.runner_factory = runner_factory
        self.source_factory = source_factory
        self.hub = hub
        self.run_defaults = run_defaults or {}
        # Non-secret server wiring surfaced to the dashboard via /api/config.
        # Never holds the api_key.
        self.server_info = server_info or {}
        # Whether this server was started in real mode (drives pin_slice mock vs real).
        self.real = real
        self.ssl_verify = ssl_verify
        # Used only by /api/runs/{id}/analyze — never surfaced via any route.
        self._api_key = api_key
        self._llm_base_url = llm_base_url
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

    async def cancel_run(self) -> str | None:
        """Cancel the in-flight run. Returns the cancelled run_id, or None if
        nothing was running."""
        async with self._lock:
            if self._active_task is None or self._active_task.done():
                return None
            run_id = self._active_run_id
            self._active_task.cancel()
            return run_id

    def _is_task_done(self) -> bool:
        return self._active_task is not None and self._active_task.done()

    async def _drive(self, orchestrator: Orchestrator, runner: Runner) -> RunReport:
        """Run orchestrator.run(), then persist + clear the active slot."""
        log.info("run starting: run_id=%s", orchestrator.config.run_id)
        try:
            report = await orchestrator.run()
        except Exception:
            log.exception("run failed: run_id=%s", orchestrator.config.run_id)
            raise
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
    run_defaults: dict[str, Any] | None = None,
    server_info: dict[str, Any] | None = None,
    real: bool = False,
    ssl_verify: bool = True,
    api_key: str = "",
    llm_base_url: str = "",
) -> FastAPI:
    """Build a FastAPI app. Defaults wire the mock path; tests pass their own
    factories + hub to skip HTTP plumbing.

    `index_html` overrides the served `/` payload (tests use this for an
    in-line page). `serve_static=False` skips the /static mount.

    `run_defaults` supplies server-level defaults (model, streaming,
    scrape_interval_s, step_limit) used to fill any field a `POST /api/run`
    body leaves unset — so an operator configures the model once in the server
    config instead of repeating it in every run request.

    `server_info` is non-secret server wiring (base_url, metrics_url, path)
    surfaced to the dashboard via GET /api/config. It must never contain the
    api_key.
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
        run_defaults=run_defaults or {},
        server_info=server_info or {},
        real=real,
        ssl_verify=ssl_verify,
        api_key=api_key,
        llm_base_url=llm_base_url,
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

    @app.get("/api/diagnostics")
    async def diagnostics() -> dict[str, Any]:
        """Test metrics URL reachability + mini-extra PATH presence.
        Safe to call at any time — read-only, no side effects."""
        import shutil as _shutil

        metrics_url = state.server_info.get("metrics_url", "")
        metrics_ok = False
        metrics_status = None
        metrics_sample_count = 0
        metrics_error = None
        if metrics_url:
            try:
                async with httpx.AsyncClient(timeout=5.0, verify=state.ssl_verify) as c:
                    r = await c.get(metrics_url)
                    metrics_status = r.status_code
                    metrics_ok = r.status_code == 200
                    if metrics_ok:
                        from clusterbench.metrics.litellm import parse_prometheus_text
                        metrics_sample_count = len(parse_prometheus_text(r.text))
            except Exception as exc:
                metrics_error = str(exc)

        mini_extra_path = _shutil.which("mini-extra")

        return {
            "metrics_url": metrics_url,
            "metrics_reachable": metrics_ok,
            "metrics_http_status": metrics_status,
            "metrics_sample_count": metrics_sample_count,
            "metrics_error": metrics_error,
            "mini_extra_on_path": mini_extra_path is not None,
            "mini_extra_path": mini_extra_path,
            "real_mode": state.real,
            "active_run_id": state.active_run_id,
        }

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        """Non-secret server wiring + run defaults, for the dashboard to
        display and prefill the controls. The api_key is never included."""
        return {
            "server": state.server_info,
            "run_defaults": state.run_defaults,
        }

    @app.post("/api/run")
    async def start_run(body: StartRunBody) -> JSONResponse:
        run_id = uuid.uuid4().hex[:12]
        config = _build_run_config(run_id, body, state.run_defaults)
        pool = _build_pool(config, real=state.real)
        await state.try_start(config, pinned=pool)
        return JSONResponse(
            status_code=202,
            content={
                "run_id": run_id,
                "pinned_instance_ids": pool,
                "config": config.to_dict(),
            },
        )

    @app.delete("/api/run")
    async def cancel_run() -> JSONResponse:
        run_id = await state.cancel_run()
        if run_id is None:
            raise HTTPException(status_code=404, detail={"error": "no_active_run"})
        return JSONResponse(status_code=200, content={"cancelled_run_id": run_id})

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

    @app.get("/api/runs/{run_id}/analyze")
    async def analyze_run(run_id: str) -> StreamingResponse:
        """Stream an LLM analysis of the run as Server-Sent Events.

        Each SSE event is one of:
          data: {"type":"token","text":"..."}   — a chunk of analysis text
          data: {"type":"done","model":"..."}   — stream finished
          data: {"type":"error","detail":"..."}  — failure

        Uses stream:true so even slow models (kimi-k2-r1 etc.) start
        returning tokens immediately without a ReadTimeout.
        """
        report = load_report(state.results_dir, run_id)
        if report is None:
            raise HTTPException(status_code=404, detail="run_not_found")
        if not state._llm_base_url or not state._api_key:
            raise HTTPException(status_code=503, detail="no_llm_configured")

        model = state.run_defaults.get("model", "gpt-4o-mini")
        prompt = _build_analysis_prompt(report)
        log.info(
            "analyze: run_id=%s model=%s base_url=%s prompt_len=%d",
            run_id, model, state._llm_base_url, len(prompt),
        )

        import json as _json

        async def _sse_stream():
            try:
                async with httpx.AsyncClient(
                    base_url=state._llm_base_url,
                    headers={"Authorization": f"Bearer {state._api_key}"},
                    verify=state.ssl_verify,
                    timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=5.0),
                ) as client:
                    async with client.stream(
                        "POST",
                        "/chat/completions",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": True,
                        },
                    ) as resp:
                        log.info("analyze: response status=%d run_id=%s", resp.status_code, run_id)
                        if resp.status_code != 200:
                            body = await resp.aread()
                            log.warning("analyze: llm error status=%d body=%s", resp.status_code, body[:200])
                            yield f"data: {_json.dumps({'type':'error','detail':f'llm_error:{resp.status_code}'})}\n\n"
                            return
                        async for line in resp.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            chunk = line[5:].strip()
                            if chunk == "[DONE]":
                                break
                            try:
                                obj = _json.loads(chunk)
                                text = obj["choices"][0]["delta"].get("content", "")
                                if text:
                                    yield f"data: {_json.dumps({'type':'token','text':text})}\n\n"
                            except Exception:
                                pass
                yield f"data: {_json.dumps({'type':'done','model':model})}\n\n"
            except httpx.RequestError as exc:
                log.warning("analyze: llm unreachable run_id=%s type=%s error=%r",
                            run_id, type(exc).__name__, exc)
                yield f"data: {_json.dumps({'type':'error','detail':f'llm_unreachable:{type(exc).__name__}'})}\n\n"

        return StreamingResponse(_sse_stream(), media_type="text/event-stream")

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


def _build_run_config(
    run_id: str, body: StartRunBody, defaults: dict[str, Any] | None = None
) -> RunConfig:
    from clusterbench.models import (
        DegradationGuard,
        LoadMode,
        MiniSweConfig,
    )

    d = defaults or {}

    def pick(value: Any, key: str, hardcoded: Any) -> Any:
        # body value (if the client set it) > server run-default > hardcoded.
        if value is not None:
            return value
        if key in d and d[key] is not None:
            return d[key]
        return hardcoded

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
        scrape_interval_s=pick(body.scrape_interval_s, "scrape_interval_s", 1.0),
        miniswe=MiniSweConfig(
            model=pick(body.model, "model", "gpt-4o-mini"),
            streaming=pick(body.streaming, "streaming", True),
            step_limit=pick(body.step_limit, "step_limit", 0),
        ),
    )


def _build_pool(config: RunConfig, *, real: bool = False) -> list[str]:
    """Build the instance ID pool that the runner samples from per level.

    Mock mode: n synthetic IDs — MockRunner uses all of them at every level.
    Real mode: all available dataset IDs — MiniSweRunner samples n_per_worker×level
    per call so each level sees a different workload (avoids KV-cache reuse).
    """
    from clusterbench.miniswerunner import pin_slice

    if config.task_slice.pinned_instance_ids:
        return list(config.task_slice.pinned_instance_ids)

    if not real:
        return pin_slice(
            n=config.task_slice.n,
            subset=config.task_slice.subset,
            split=config.task_slice.split,
            mock=True,
        )

    # Real path: load the full dataset so MiniSweRunner can sample per level.
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "real path needs the `datasets` package; install with `uv sync --extra real`"
        ) from exc

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split=config.task_slice.split)
    return [row["instance_id"] for row in ds]


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


def _build_analysis_prompt(report: "RunReport") -> str:
    """Build a structured prompt asking the LLM to analyze a benchmark report."""
    def _f(v: float | None, fmt: str = ".3f") -> str:
        return f"{v:{fmt}}" if v is not None else "n/a"

    lines: list[str] = []
    lines.append(
        "You are an expert in LLM inference infrastructure and cluster performance analysis. "
        "Review the following ClusterBench saturation sweep report and provide a concise "
        "technical analysis covering:\n"
        "1. Throughput scaling and where it plateaus\n"
        "2. Latency behaviour under load (p50/p99 trend)\n"
        "3. TTFT and TPOT trends (if available) — TTFT reveals queueing, TPOT reveals GPU "
        "decode saturation; a rising TPOT means the GPU cannot keep up with decode demand\n"
        "4. KV-cache miss count — high cache misses suggest prompts are not being reused; "
        "recommend prefix caching or request routing if relevant\n"
        "5. The saturation knee — at which concurrency level does the cluster start to degrade "
        "and what is the primary signal (latency, error rate, TTFT spike, TPOT rise)?\n"
        "6. Outcome taxonomy — are timeouts or errors concentrated at specific concurrency levels?\n"
        "7. Process health (if data available) — does RSS grow with concurrency (memory pressure)? "
        "Is the FD ratio approaching the limit (connection pool leak)? "
        "Do gen1/gen2 GC counts spike at high load (Python object churn)?\n"
        "8. Specific recommendations for the inference cluster operator: what to change or test "
        "next (e.g. batch size, number of replicas, model parallelism, KV-cache tuning, "
        "prefix caching, chunked prefill, connection pool sizing)\n\n"
        "Be specific and cite the numbers from the data below.\n\n"
    )

    cfg = report.config
    lines.append(f"## Run: {report.name or report.run_id}")
    lines.append(f"Mode: {cfg.mode.value} | Model: {cfg.miniswe.model} | "
                 f"Streaming: {cfg.miniswe.streaming} | "
                 f"TTFT available: {report.ttft_available} | "
                 f"Wire metrics: {report.wire_metrics_available}")
    if report.knee:
        lines.append(f"Saturation knee detected at level={report.knee['level']}: "
                     f"{report.knee.get('reason', '')}")
    else:
        lines.append("No saturation knee detected within the sweep range.")
    lines.append("")

    lines.append("## Per-level statistics")
    header = (
        f"{'level':>6}  {'n':>5}  {'pass%':>6}  "
        f"{'tok/s':>7}  {'lat_p50':>8}  {'lat_p99':>8}  "
        f"{'ttft_p50':>9}  {'ttft_p95':>9}  "
        f"{'tpot_ms':>8}  {'c_miss':>7}  "
        f"{'inflight':>8}  {'err%':>6}  {'wall_s':>7}  outcomes"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for lv in sorted(report.levels, key=lambda l: l.level):
        d = lv.delta
        outcomes_str = " ".join(f"{k}:{v}" for k, v in sorted((lv.outcome_counts or {}).items()))
        lines.append(
            f"{lv.level:>6}  {lv.n_tasks:>5}  {lv.pass_rate*100:>5.1f}%  "
            f"{_f(d.throughput_tps if d else None, '.1f'):>7}  "
            f"{_f(d.lat_p50 if d else None):>8}  "
            f"{_f(d.lat_p99 if d else None):>8}  "
            f"{_f(d.ttft_p50 if d else None):>9}  "
            f"{_f(d.ttft_p95 if d else None):>9}  "
            f"{_f(d.tpot_ms if d else None, '.1f'):>8}  "
            f"{str(d.cache_misses if d else None) if (d and d.cache_misses is not None) else 'n/a':>7}  "
            f"{_f(d.in_flight_peak if d else None, '.1f'):>8}  "
            f"{((d.error_rate or 0)*100 if d else 0):>5.1f}%  "
            f"{lv.duration_s:>7.1f}  {outcomes_str}"
        )
    lines.append("")

    # Process-health section — only emit if any level has data.
    if any(lv.delta and lv.delta.rss_mb is not None for lv in report.levels):
        lines.append("## Process health (end-of-level snapshot)")
        ph_header = (
            f"{'level':>6}  {'rss_mb':>8}  {'open_fds':>9}  {'max_fds':>8}  "
            f"{'fd%':>5}  {'gc_gen1':>8}  {'gc_gen2':>8}"
        )
        lines.append(ph_header)
        lines.append("-" * len(ph_header))
        for lv in sorted(report.levels, key=lambda l: l.level):
            d = lv.delta
            fd_pct = (
                f"{d.open_fds / d.max_fds * 100:.1f}%"
                if d and d.open_fds and d.max_fds
                else "n/a"
            )
            lines.append(
                f"{lv.level:>6}  "
                f"{_f(d.rss_mb if d else None, '.1f'):>8}  "
                f"{str(d.open_fds) if d and d.open_fds is not None else 'n/a':>9}  "
                f"{str(d.max_fds) if d and d.max_fds is not None else 'n/a':>8}  "
                f"{fd_pct:>5}  "
                f"{str(d.gc_gen1) if d and d.gc_gen1 is not None else 'n/a':>8}  "
                f"{str(d.gc_gen2) if d and d.gc_gen2 is not None else 'n/a':>8}"
            )
        lines.append("")

    lines.append("Provide your analysis:")
    return "\n".join(lines)


__all__ = [
    "create_app",
    "StartRunBody",
    "default_runner_factory",
    "default_source_factory",
]
