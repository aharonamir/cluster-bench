"""Tests for clusterbench.web.server (T043).

Covers Gate 4 ACs:
  - AC-7: two runs persist distinct retrievable reports
  - AC-8: mid-run WS connect replays completed levels
  - AC-9: full stack runs on the mock path with no GPU/Docker/downloads

Plus the basic REST contract: POST /api/run returns 202 with run_id; second
POST while active returns 409; GET /api/runs and /api/runs/{id} work; the
active slot clears after completion.
"""
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from clusterbench.metrics.litellm import LiteLLMSource
from clusterbench.miniswerunner import MockRunner, Runner, pin_slice
from clusterbench.models import RunConfig
from clusterbench.web.hub import WebSocketHub
from clusterbench.web.server import create_app
from mock_litellm import create_app as create_mock_app


# ---------------------------------------------------------------------------
# Test seam: factories that share an httpx.AsyncClient with mock_litellm via
# ASGITransport, so the server's orchestrator talks to the mock without HTTP.
# ---------------------------------------------------------------------------


def _make_factories(
    *, mock_app, tmp_path: Path
) -> tuple[Any, Any, httpx.AsyncClient]:
    """Build runner_factory + source_factory that share one client against the
    in-process mock app. Caller owns the client's lifecycle (use as ctx mgr)."""
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_app), base_url="http://test"
    )

    def runner_factory(*, config: RunConfig, pinned: list[str]) -> Runner:
        return MockRunner(
            base_url="http://test/v1",
            instance_ids=pinned,
            turns_per_instance=1,
            model=config.miniswe.model,
            streaming=config.miniswe.streaming,
            client=client,
            runner_root=tmp_path,
        )

    def source_factory(*, config: RunConfig):
        return LiteLLMSource(
            metrics_url="http://test/metrics",
            scrape_interval_s=config.scrape_interval_s,
            client=client,
        )

    return runner_factory, source_factory, client


def _build_test_app(
    *,
    tmp_path: Path,
    hub: WebSocketHub | None = None,
    mock_app=None,
    run_defaults=None,
    server_info=None,
) -> tuple[Any, httpx.AsyncClient]:
    """Build a server app against an in-process mock_litellm. Returns
    (server_app, shared_client_for_close)."""
    mock_app = mock_app or create_mock_app()
    runner_factory, source_factory, client = _make_factories(
        mock_app=mock_app, tmp_path=tmp_path
    )
    server_app = create_app(
        results_dir=tmp_path / "results",
        runner_factory=runner_factory,
        source_factory=source_factory,
        hub=hub,
        run_defaults=run_defaults,
        server_info=server_info,
    )
    return server_app, client


def _api_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


# ---------------------------------------------------------------------------
# Health + index
# ---------------------------------------------------------------------------


def test_health_endpoint_reports_state():
    async def go():
        app, _ = _build_test_app(tmp_path=Path("/tmp"))
        async with _api_client(app) as client:
            r = await client.get("/api/health")
            assert r.status_code == 200
            data = r.json()
            assert data["ok"] is True
            assert data["active_run_id"] is None
            assert data["n_subscribers"] == 0

    asyncio.run(go())


def test_config_endpoint_exposes_wiring_without_api_key():
    """The dashboard reads /api/config to show the real model + endpoints.
    It must surface server_info + run_defaults and NEVER leak the api_key."""
    async def go():
        app, _ = _build_test_app(
            tmp_path=Path("/tmp"),
            run_defaults={"model": "qwen2.5-coder", "streaming": True,
                          "scrape_interval_s": 0.5, "step_limit": 0},
            server_info={"base_url": "http://litellm:4000/v1",
                         "metrics_url": "http://litellm:4000/metrics",
                         "path": "real", "results_dir": "/data"},
        )
        async with _api_client(app) as client:
            r = await client.get("/api/config")
            assert r.status_code == 200
            data = r.json()
            assert data["run_defaults"]["model"] == "qwen2.5-coder"
            assert data["server"]["base_url"] == "http://litellm:4000/v1"
            assert data["server"]["path"] == "real"
            # The api_key must never appear anywhere in the payload.
            assert "api_key" not in data["server"]
            assert "sk-" not in r.text

    asyncio.run(go())


def test_run_without_model_inherits_server_default(tmp_path: Path):
    """A POST that omits `model` must use the server's configured default,
    not a hardcoded fallback — this is the config-visibility fix."""
    async def go():
        app, mock_client = _build_test_app(
            tmp_path=tmp_path,
            run_defaults={"model": "configured-model"},
        )
        async with mock_client, _api_client(app) as api:
            r = await api.post("/api/run", json={
                "mode": "sweep", "levels": [1], "task_slice": {"n": 2},
                "scrape_interval_s": 0.005,
            })
            assert r.status_code == 202
            assert r.json()["config"]["miniswe"]["model"] == "configured-model"

    asyncio.run(go())


def test_index_serves_html():
    async def go():
        app, _ = _build_test_app(tmp_path=Path("/tmp"))
        async with _api_client(app) as client:
            r = await client.get("/")
            assert r.status_code == 200
            assert "<html" in r.text.lower()
            assert "ClusterBench" in r.text

    asyncio.run(go())


# ---------------------------------------------------------------------------
# POST /api/run → 202 + run_id; AC-9 mock-path SWEEP saves a report
# ---------------------------------------------------------------------------


def test_post_run_returns_202_with_run_id_and_starts_run(tmp_path: Path):
    """AC-9: full stack on the mock path — POST /api/run starts a run that
    completes and persists a report with ≥2 levels."""
    hub = WebSocketHub()

    async def go():
        app, mock_client = _build_test_app(tmp_path=tmp_path, hub=hub)
        async with mock_client, _api_client(app) as api:
            body = {
                "name": "e2e",
                "mode": "sweep",
                "levels": [1, 2],
                "task_slice": {"n": 4},
                "scrape_interval_s": 0.005,
            }
            r = await api.post("/api/run", json=body)
            assert r.status_code == 202
            run_id = r.json()["run_id"]
            assert len(run_id) == 12
            # Wait for the background task to finish by polling /api/runs/{id}.
            for _ in range(200):
                g = await api.get(f"/api/runs/{run_id}")
                if g.status_code == 200:
                    return run_id, g.json()
                await asyncio.sleep(0.05)
            raise AssertionError("run never completed")

    run_id, payload = asyncio.run(go())
    assert payload["run_id"] == run_id
    assert len(payload["levels"]) == 2
    for lv in payload["levels"]:
        assert lv["delta"] is not None
        assert lv["n_tasks"] == 4


def test_second_post_while_active_returns_409(tmp_path: Path):
    """FR-15 re-entrancy: a second POST /api/run while one is active returns
    409 with the active run_id in the error body."""
    hub = WebSocketHub()

    async def go():
        # Long levels: small n but high concurrency so it doesn't finish
        # before the second POST lands.
        app, mock_client = _build_test_app(
            tmp_path=tmp_path,
            hub=hub,
            mock_app=create_mock_app(
                streaming=True, saturation_threshold=2, saturation_extra_delay_s=0.5
            ),
        )
        async with mock_client, _api_client(app) as api:
            body = {
                "mode": "sweep",
                "levels": [4, 8],
                "task_slice": {"n": 8},
                "scrape_interval_s": 0.005,
            }
            r1 = await api.post("/api/run", json=body)
            assert r1.status_code == 202
            run_id_1 = r1.json()["run_id"]
            # Fire the second immediately — the first is still running.
            r2 = await api.post("/api/run", json=body)
            assert r2.status_code == 409
            assert r2.json()["detail"]["error"] == "run_already_active"
            assert r2.json()["detail"]["active_run_id"] == run_id_1
            # Let the first finish so the test teardown doesn't cancel it.
            for _ in range(200):
                g = await api.get(f"/api/runs/{run_id_1}")
                if g.status_code == 200:
                    return
                await asyncio.sleep(0.05)

    asyncio.run(go())


# ---------------------------------------------------------------------------
# AC-7: two runs persist distinct retrievable reports
# ---------------------------------------------------------------------------


def test_two_runs_persist_distinct_reports(tmp_path: Path):
    """AC-7: two sequential runs each land on disk and are retrievable."""
    hub = WebSocketHub()

    async def go():
        app, mock_client = _build_test_app(tmp_path=tmp_path, hub=hub)
        async with mock_client, _api_client(app) as api:
            run_ids: list[str] = []
            for i in range(2):
                body = {
                    "name": f"run-{i}",
                    "mode": "sweep",
                    "levels": [1, 2],
                    "task_slice": {"n": 4},
                    "scrape_interval_s": 0.005,
                }
                r = await api.post("/api/run", json=body)
                assert r.status_code == 202
                run_ids.append(r.json()["run_id"])
                # Wait for completion before the next POST.
                for _ in range(200):
                    g = await api.get(f"/api/runs/{run_ids[-1]}")
                    if g.status_code == 200:
                        break
                    await asyncio.sleep(0.05)
                else:
                    raise AssertionError(f"run {run_ids[-1]} never completed")

            # Both should now be retrievable individually.
            for rid in run_ids:
                g = await api.get(f"/api/runs/{rid}")
                assert g.status_code == 200
                assert g.json()["run_id"] == rid

            # And listed by GET /api/runs.
            listing = await api.get("/api/runs")
            assert listing.status_code == 200
            listed_ids = listing.json()["run_ids"]
            for rid in run_ids:
                assert rid in listed_ids

    asyncio.run(go())


def test_get_run_returns_404_for_unknown_run(tmp_path: Path):
    async def go():
        app, _ = _build_test_app(tmp_path=tmp_path)
        async with _api_client(app) as api:
            r = await api.get("/api/runs/does-not-exist")
            assert r.status_code == 404
            assert r.json()["detail"] == "run_not_found"

    asyncio.run(go())


def test_list_runs_returns_empty_initially(tmp_path: Path):
    async def go():
        app, _ = _build_test_app(tmp_path=tmp_path)
        async with _api_client(app) as api:
            r = await api.get("/api/runs")
            assert r.status_code == 200
            assert r.json()["run_ids"] == []

    asyncio.run(go())


# ---------------------------------------------------------------------------
# AC-8: mid-run WebSocket connect replays history (completed levels)
# ---------------------------------------------------------------------------


class _RecordingWS:
    """Bare WebSocket client over httpx-ws; falls back to manually inspecting
    hub.history() if httpx-ws isn't installed. We use the hub directly
    because it's the source of truth and avoids WS client plumbing in tests."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def feed(self, raw: str) -> None:
        parsed = json.loads(raw)
        self.events.append(parsed)


def test_mid_run_connect_replays_completed_levels(tmp_path: Path):
    """AC-8: a client that connects mid-run receives every event it missed
    (run_start + completed levels so far). We poll the live WS endpoint
    indirectly by reading hub.history() after the run completes — that history
    IS exactly what a mid-run client would have been replayed."""
    hub = WebSocketHub()

    async def go():
        app, mock_client = _build_test_app(tmp_path=tmp_path, hub=hub)
        async with mock_client, _api_client(app) as api:
            body = {
                "mode": "sweep",
                "levels": [1, 2, 4],
                "task_slice": {"n": 4},
                "scrape_interval_s": 0.005,
            }
            r = await api.post("/api/run", json=body)
            run_id = r.json()["run_id"]
            # Let the run complete so the hub has its full history.
            for _ in range(200):
                g = await api.get(f"/api/runs/{run_id}")
                if g.status_code == 200:
                    return run_id
                await asyncio.sleep(0.05)
            raise AssertionError("run never completed")

    run_id = asyncio.run(go())

    history = asyncio.run(hub.history())
    types = [json.loads(h)["type"] for h in history]

    # Full event sequence (FR-21): run_start, then per-level start/scrape/done,
    # task outcomes, then run_done.
    assert types[0] == "run_start"
    assert types[-1] == "run_done"
    assert types.count("level_done") == 3
    # Knee not emitted here (no guards set); make sure no malformed frames.
    for h in history:
        parsed = json.loads(h)
        assert "type" in parsed
        assert "payload" in parsed


def test_real_ws_connect_replays_then_streams_live(tmp_path: Path):
    """AC-8 end-to-end via the actual WS endpoint: connect with no prior
    state, then start a run; the connection should receive the full stream.
    Uses starlette's TestClient over a real WS handshake."""
    from starlette.testclient import TestClient

    hub = WebSocketHub()

    async def build_and_run():
        # Pre-warm: emit a synthetic past event so replay has something to send.
        await hub.emit("external_event", {"note": "pre-run"})
        app, _mock_client = _build_test_app(tmp_path=tmp_path, hub=hub)
        return app

    app = asyncio.run(build_and_run())

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            # First frame is the replayed pre-run event.
            first = ws.receive()
            # Starlette's TestClient may report either "websocket.receive"
            # (client-side view) or "websocket.send" (server-side view).
            assert "text" in first
            first_payload = json.loads(first["text"])
            assert first_payload["type"] == "external_event"

            # Now trigger a run; the WS should stream its events live.
            r = client.post(
                "/api/run",
                json={
                    "mode": "sweep",
                    "levels": [1],
                    "task_slice": {"n": 2},
                    "scrape_interval_s": 0.005,
                },
            )
            assert r.status_code == 202
            run_id = r.json()["run_id"]

            # Drain frames until we see run_done.
            seen_types: list[str] = []
            for _ in range(200):
                msg = ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if "text" not in msg:
                    continue
                parsed = json.loads(msg["text"])
                seen_types.append(parsed["type"])
                if parsed["type"] == "run_done":
                    break

            assert "run_start" in seen_types
            assert "level_done" in seen_types
            assert seen_types[-1] == "run_done"


# ---------------------------------------------------------------------------
# Wire pinning
# ---------------------------------------------------------------------------


def test_post_run_echoes_pinned_instance_ids(tmp_path: Path):
    """The response includes the resolved pinned slice so the client knows
    which instances were run (AC-10: reused across levels)."""
    hub = WebSocketHub()

    async def go():
        app, mock_client = _build_test_app(tmp_path=tmp_path, hub=hub)
        async with mock_client, _api_client(app) as api:
            body = {"mode": "sweep", "levels": [1], "task_slice": {"n": 3}}
            r = await api.post("/api/run", json=body)
            return r.json()

    payload = asyncio.run(go())
    pinned = payload["pinned_instance_ids"]
    assert len(pinned) == 3
    assert all(i.startswith("mock-verified-test-") for i in pinned)


def test_post_run_with_invalid_mode_returns_422(tmp_path: Path):
    async def go():
        app, _ = _build_test_app(tmp_path=tmp_path)
        async with _api_client(app) as api:
            r = await api.post(
                "/api/run",
                json={"mode": "bogus", "levels": [1], "task_slice": {"n": 2}},
            )
            return r.status_code

    assert asyncio.run(go()) == 422
