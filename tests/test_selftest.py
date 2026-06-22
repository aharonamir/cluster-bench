"""Phase 6 self-test (T062): drive a sweep + a soak through /api/run against
the in-process mock path, then assert each landed on disk as a valid
RunReport with non-empty deltas. This is the end-to-end smoke the spec calls
out as Gate 6's "self-test passes sweep+soak on the mock path".
"""
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from clusterbench.metrics.litellm import LiteLLMSource
from clusterbench.miniswerunner import MockRunner, Runner
from clusterbench.models import RunConfig
from clusterbench.web.hub import WebSocketHub
from clusterbench.web.server import create_app
from mock_litellm import create_app as create_mock_app


def _make_factories(
    *, mock_app, tmp_path: Path
) -> tuple[Any, Any, httpx.AsyncClient]:
    """Runner + source factories sharing one in-process httpx client."""
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_app), base_url="http://test"
    )

    def runner_factory(*, config: RunConfig, pinned: list[str]) -> Runner:
        return MockRunner(
            base_url="http://test/v1",
            instance_ids=pinned,
            turns_per_instance=2,
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


def _api_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def _wait_for_run(api: httpx.AsyncClient, run_id: str, timeout_s: float = 30.0):
    """Poll GET /api/runs/{id} until 200. Returns the parsed report."""
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = await api.get(f"/api/runs/{run_id}")
        if r.status_code == 200:
            return r.json()
        await asyncio.sleep(0.05)
    raise AssertionError(f"run {run_id} never completed within {timeout_s}s")


def _assert_valid_report(report: dict, *, expected_levels: int, mode: str) -> None:
    """Common shape checks for both sweep + soak reports."""
    assert report["config"]["mode"] == mode
    assert len(report["levels"]) == expected_levels
    assert report["wire_metrics_available"] is True
    # Pinned slice must be recorded (AC-10) and reused at every level.
    assert len(report["pinned_instance_ids"]) > 0
    # Each level must carry a non-empty delta (the point of scrape-and-delta).
    for lv in report["levels"]:
        d = lv["delta"]
        assert d is not None, "wire metrics unavailable on a level"
        assert d["n_requests"] > 0, "delta saw no traffic"
        assert d["throughput_tps"] > 0, "throughput is zero"
        # in_flight_peak should be at least 1 (every level ran at least one
        # request).
        assert d["in_flight_peak"] >= 1
        # Tasks scored.
        assert lv["n_tasks"] > 0
        assert "outcome_counts" in lv
        # Total outcome counts must equal n_tasks (every attempt got bucketed).
        total = sum(lv["outcome_counts"].values())
        assert total == lv["n_tasks"], (
            f"outcome counts {total} != n_tasks {lv['n_tasks']}"
        )


def test_selftest_sweep_persists_valid_report(tmp_path: Path):
    """End-to-end sweep through /api/run on the mock path."""
    hub = WebSocketHub()

    async def go():
        mock_app = create_mock_app()
        runner_factory, source_factory, client = _make_factories(
            mock_app=mock_app, tmp_path=tmp_path
        )
        app = create_app(
            results_dir=tmp_path / "results",
            runner_factory=runner_factory,
            source_factory=source_factory,
            hub=hub,
        )
        async with client:
            async with _api_client(app) as api:
                # Wire a guard so the knee machinery is exercised too.
                body = {
                    "name": "selftest-sweep",
                    "mode": "sweep",
                    "levels": [1, 2, 4],
                    "task_slice": {"n": 4},
                    "scrape_interval_s": 0.005,
                    "guards": {"max_p99_latency_s": 100.0},
                }
                r = await api.post("/api/run", json=body)
                assert r.status_code == 202, r.text
                run_id = r.json()["run_id"]
                report = await _wait_for_run(api, run_id)
                # The on-disk file should round-trip the same shape.
                on_disk = json.loads(
                    (tmp_path / "results" / f"{run_id}.json").read_text()
                )
                return report, on_disk

    report, on_disk = asyncio.run(go())
    _assert_valid_report(report, expected_levels=3, mode="sweep")
    _assert_valid_report(on_disk, expected_levels=3, mode="sweep")
    # The persisted JSON should match what the API returned (modulo dict order).
    assert report["run_id"] == on_disk["run_id"]
    assert report["levels"] == on_disk["levels"]


def test_selftest_soak_persists_valid_report(tmp_path: Path):
    """End-to-end soak through /api/run on the mock path. Soak holds one
    level and bins it; the report should have ≥1 bin and each bin's delta
    should carry non-zero traffic."""
    hub = WebSocketHub()

    async def go():
        mock_app = create_mock_app()
        runner_factory, source_factory, client = _make_factories(
            mock_app=mock_app, tmp_path=tmp_path
        )
        app = create_app(
            results_dir=tmp_path / "results",
            runner_factory=runner_factory,
            source_factory=source_factory,
            hub=hub,
        )
        async with client:
            async with _api_client(app) as api:
                body = {
                    "name": "selftest-soak",
                    "mode": "soak",
                    "levels": [2],  # single level held for the soak
                    "soak_duration_s": 0.4,
                    "task_slice": {"n": 4},
                    "scrape_interval_s": 0.005,
                }
                r = await api.post("/api/run", json=body)
                assert r.status_code == 202, r.text
                run_id = r.json()["run_id"]
                report = await _wait_for_run(api, run_id, timeout_s=20.0)
                return report

    report = asyncio.run(go())
    assert report["config"]["mode"] == "soak"
    # At least one bin should have run; the mock is fast enough to produce 2+
    # within 0.4s.
    assert len(report["levels"]) >= 1
    for lv in report["levels"]:
        d = lv["delta"]
        assert d is not None
        assert d["n_requests"] > 0
        assert d["throughput_tps"] > 0
        assert lv["n_tasks"] > 0
        total = sum(lv["outcome_counts"].values())
        assert total == lv["n_tasks"]
    assert report["wire_metrics_available"] is True


def test_selftest_two_runs_distinct_and_overlaid(tmp_path: Path):
    """AC-7 final smoke: two sequential runs persist distinct reports, both
    retrievable via the listing endpoint."""
    hub = WebSocketHub()

    async def go():
        mock_app = create_mock_app()
        runner_factory, source_factory, client = _make_factories(
            mock_app=mock_app, tmp_path=tmp_path
        )
        app = create_app(
            results_dir=tmp_path / "results",
            runner_factory=runner_factory,
            source_factory=source_factory,
            hub=hub,
        )
        async with client:
            async with _api_client(app) as api:
                ids = []
                for i in range(2):
                    body = {
                        "name": f"selftest-{i}",
                        "mode": "sweep",
                        "levels": [1, 2],
                        "task_slice": {"n": 3},
                        "scrape_interval_s": 0.005,
                    }
                    r = await api.post("/api/run", json=body)
                    assert r.status_code == 202
                    rid = r.json()["run_id"]
                    await _wait_for_run(api, rid)
                    ids.append(rid)
                listing = await api.get("/api/runs")
                return ids, listing.json()["run_ids"]

    ids, listed = asyncio.run(go())
    assert len(set(ids)) == 2, "two runs must have distinct run_ids"
    for rid in ids:
        assert rid in listed
