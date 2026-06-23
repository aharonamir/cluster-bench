"""Headless-browser tests for the ClusterBench dashboard (T055/Gate 5).

Verifies that loading / on a server with seeded saved reports renders the
SVGs with non-empty data, the per-level table with bucket-percentile labels,
and that loading a 2nd saved report overlays both headline series.

Driven by Playwright since the dashboard is vanilla JS over a WS connection
and we need a real DOM + JS runtime to assert on the rendered SVG.
"""
from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path

import httpx
import pytest

try:
    from playwright.async_api import async_playwright

    _HAS_PLAYWRIGHT = True
except ImportError:  # playwright is a dev extra; skip when absent
    _HAS_PLAYWRIGHT = False

from clusterbench.metrics.litellm import LiteLLMSource
from clusterbench.miniswerunner import MockRunner, Runner
from clusterbench.models import RunConfig
from clusterbench.web.hub import WebSocketHub
from clusterbench.web.server import create_app
from mock_litellm import create_app as create_mock_app


pytestmark = pytest.mark.skipif(
    not _HAS_PLAYWRIGHT, reason="playwright not installed"
)


# ---------------------------------------------------------------------------
# Test seam: factories that share one httpx client with the in-process mock.
# ---------------------------------------------------------------------------


def _make_factories(
    *, mock_app, tmp_path: Path
) -> tuple[Any, Any, httpx.AsyncClient]:
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


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _seed_two_runs(tmp_path: Path) -> tuple[Any, httpx.AsyncClient]:
    """Build a server app and drive two SWEEP runs through it, returning the
    app (with its results_dir populated) + the mock client the caller must
    close."""
    mock_app = create_mock_app()
    runner_factory, source_factory, client = _make_factories(
        mock_app=mock_app, tmp_path=tmp_path
    )
    app = create_app(
        results_dir=tmp_path / "results",
        runner_factory=runner_factory,
        source_factory=source_factory,
        run_defaults={"model": "kimi-k2.6", "streaming": True,
                      "scrape_interval_s": 0.5, "step_limit": 0},
        server_info={"base_url": "http://litellm:4000/v1",
                     "metrics_url": "http://litellm:4000/metrics",
                     "path": "real", "results_dir": str(tmp_path / "results")},
    )
    async with client:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as api:
            ids = []
            for spec in (
                {"name": "alpha", "levels": [1, 2, 4]},
                {"name": "beta", "levels": [1, 2, 4, 8]},
            ):
                body = {
                    "name": spec["name"],
                    "mode": "sweep",
                    "levels": spec["levels"],
                    "task_slice": {"n": 4},
                    "scrape_interval_s": 0.005,
                }
                r = await api.post("/api/run", json=body)
                assert r.status_code == 202, r.text
                rid = r.json()["run_id"]
                ids.append(rid)
                # Wait for completion.
                for _ in range(200):
                    g = await api.get(f"/api/runs/{rid}")
                    if g.status_code == 200:
                        break
                    await asyncio.sleep(0.05)
                else:
                    raise AssertionError(f"run {rid} never completed")
    return app, client


async def _serve(app, port: int) -> Any:
    """Serve the app on a free port using uvicorn programmatically."""
    import uvicorn

    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # Give it a moment to bind.
    for _ in range(20):
        if server.started:
            break
        await asyncio.sleep(0.05)
    return server, task


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_saved_report_renders_all_charts(tmp_path: Path):
    """Gate 5 core: load / on a finished run → SVGs contain data points."""
    async def go():
        app, _client = await _seed_two_runs(tmp_path)
        port = _free_port()
        server, task = await _serve(app, port)
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                console_errors: list[str] = []
                page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
                page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))
                await page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")

                # The config readout must show the server's configured model
                # (from /api/config), not a hardcoded default.
                await page.wait_for_function(
                    "() => document.getElementById('rd-model')"
                    "        .textContent === 'kimi-k2.6'"
                )
                rd_path = await page.eval_on_selector("#rd-path", "el => el.textContent")
                assert rd_path == "real"

                # The saved-reports <select> should list both runs.
                sel = page.locator("#saved-select option")
                await page.wait_for_function(
                    "() => document.querySelectorAll('#saved-select option').length >= 2"
                )
                count = await sel.count()
                assert count >= 2

                # Select the first one and load it.
                await sel.nth(0).click()
                await page.click("#load-btn")
                # Wait for the SVGs to contain circles (data points).
                await page.wait_for_function(
                    "() => document.querySelectorAll('#chart-ttft circle').length > 0"
                )
                await page.wait_for_function(
                    "() => document.querySelectorAll('#chart-latency circle').length > 0"
                )
                await page.wait_for_function(
                    "() => document.querySelectorAll('#chart-saturation circle').length > 0"
                )
                await page.wait_for_function(
                    "() => document.querySelectorAll('#chart-litellm rect').length > 0"
                )
                await page.wait_for_function(
                    "() => document.querySelectorAll('#level-table-body tr').length > 0"
                )

                ttft_pts = await page.eval_on_selector_all(
                    "#chart-ttft circle", "els => els.length"
                )
                lat_pts = await page.eval_on_selector_all(
                    "#chart-latency circle", "els => els.length"
                )
                sat_pts = await page.eval_on_selector_all(
                    "#chart-saturation circle", "els => els.length"
                )
                # First option by mtime-desc is whichever run finished last.
                # Either alpha (3 levels) or beta (4 levels). Latency and
                # saturation are single-series (1 circle/level); the TTFT chart
                # draws two series (p50 + p95), so it has twice as many circles.
                assert ttft_pts >= 6, f"ttft points: {ttft_pts}"
                assert lat_pts >= 3, f"latency points: {lat_pts}"
                assert lat_pts == sat_pts, "latency points should match saturation"
                assert ttft_pts == 2 * lat_pts, "ttft has p50+p95 (2x single-series)"

                # Per-level table: one row per level (== single-series points).
                rows = await page.eval_on_selector_all(
                    "#level-table-body tr", "els => els.length"
                )
                assert rows == lat_pts

                # TTFT column headers must be labeled p50/p95 (bucket-edge).
                headers = await page.eval_on_selector_all(
                    "#level-table th", "els => els.map(e => e.textContent.trim())"
                )
                joined = " | ".join(headers)
                assert "p50" in joined
                assert "p95" in joined
                assert "p99" in joined

                # Console must be clean (no JS errors).
                assert console_errors == [], f"console errors: {console_errors}"

                await browser.close()
        finally:
            server.should_exit = True
            await task

    asyncio.run(go())


def test_overlay_two_runs_renders_both_series(tmp_path: Path):
    """Gate 5 / FR-25: loading a 2nd saved report overlays both on the
    headline axes."""
    async def go():
        app, _client = await _seed_two_runs(tmp_path)
        port = _free_port()
        server, task = await _serve(app, port)
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
                await page.wait_for_function(
                    "() => document.querySelectorAll('#saved-select option').length >= 2"
                )
                # Select both options (cmd/ctrl-click metaphor: set selected).
                await page.eval_on_selector_all(
                    "#saved-select option",
                    "els => els.forEach(e => e.selected = true)",
                )
                await page.click("#load-btn")

                # Wait for the overlay to render. Latency is single-series, so
                # 3 + 4 = 7 circles is the stable signal; the TTFT chart draws
                # two series (p50 + p95) and will have twice as many.
                await page.wait_for_function(
                    "() => document.querySelectorAll('#chart-latency circle').length === 7"
                )
                ttft = await page.eval_on_selector_all(
                    "#chart-ttft circle", "els => els.length"
                )
                lat = await page.eval_on_selector_all(
                    "#chart-latency circle", "els => els.length"
                )
                assert lat == 7
                assert ttft == 2 * lat, f"ttft has p50+p95: {ttft} vs 2*lat={2 * lat}"

                # Overlay legend shows both run names.
                legend = await page.eval_on_selector_all(
                    "#overlay-legend .legend-item",
                    "els => els.map(e => e.textContent.trim())",
                )
                assert any("alpha" in s for s in legend)
                assert any("beta" in s for s in legend)

                # Colors used: at least 2 distinct stroke colors on the paths.
                colors = await page.eval_on_selector_all(
                    "#chart-ttft path",
                    "els => els.map(e => e.getAttribute('stroke'))",
                )
                assert len(set(colors)) >= 2

                await browser.close()
        finally:
            server.should_exit = True
            await task

    asyncio.run(go())


def test_streaming_off_renders_ttft_unavailable(tmp_path: Path):
    """AC-11: when a saved report has ttft_available=false, the dashboard
    must NOT draw TTFT points (and ideally show a hint)."""
    async def go():
        mock_app = create_mock_app(streaming=False)
        runner_factory, source_factory, client = _make_factories(
            mock_app=mock_app, tmp_path=tmp_path
        )
        app = create_app(
            results_dir=tmp_path / "results",
            runner_factory=runner_factory,
            source_factory=source_factory,
        )
        async with client:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as api:
                r = await api.post(
                    "/api/run",
                    json={
                        "name": "no-stream",
                        "mode": "sweep",
                        "levels": [1, 2],
                        "task_slice": {"n": 3},
                        "streaming": False,
                        "scrape_interval_s": 0.005,
                    },
                )
                run_id = r.json()["run_id"]
                for _ in range(200):
                    g = await api.get(f"/api/runs/{run_id}")
                    if g.status_code == 200:
                        break
                    await asyncio.sleep(0.05)

        port = _free_port()
        server, task = await _serve(app, port)
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
                await page.wait_for_function(
                    "() => document.querySelectorAll('#saved-select option').length >= 1"
                )
                await page.eval_on_selector_all(
                    "#saved-select option",
                    "els => els.forEach(e => e.selected = true)",
                )
                await page.click("#load-btn")
                # Give it a moment to render.
                await page.wait_for_function(
                    "() => document.querySelectorAll('#level-table-body tr').length >= 1"
                )
                # TTFT chart has 0 circles because ttft_p95 was null at every level.
                ttft_circles = await page.eval_on_selector_all(
                    "#chart-ttft circle", "els => els.length"
                )
                assert ttft_circles == 0
                # The TTFT panel header carries the "unavailable" hint.
                hint = await page.eval_on_selector(
                    "#ttft-hint", "el => el.textContent"
                )
                assert "unavailable" in hint.lower()
                # Latency chart still has data.
                lat_circles = await page.eval_on_selector_all(
                    "#chart-latency circle", "els => els.length"
                )
                assert lat_circles == 2
                await browser.close()
        finally:
            server.should_exit = True
            await task

    asyncio.run(go())
