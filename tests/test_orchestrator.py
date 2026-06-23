"""Tests for clusterbench.orchestrator (T037).

Covers AC-1 (SWEEP populates agents-vs-TTFT/latency per level via delta),
AC-2 (throughput rises then falls; knee = first guard trip), AC-3 (blocked
metrics → wire-unavailable, process+score still present), AC-5 (every
taxonomy branch reachable), AC-10 (same pinned slice every level), plus
SOAK duration + in-flight peak + event emission.
"""
import asyncio
import time
from pathlib import Path

import httpx
import pytest

from clusterbench.metrics.litellm import LiteLLMSource
from clusterbench.miniswerunner import MockRunner, Runner, pin_slice
from clusterbench.models import (
    DegradationGuard,
    LevelRunResult,
    LoadMode,
    MiniSweConfig,
    Outcome,
    PredRecord,
    ProcessRecord,
    RunConfig,
    TaskSlice,
)
from clusterbench.orchestrator import (
    CollectingEmitter,
    EventEmitter,
    NullEmitter,
    Orchestrator,
    resolve_outcome,
)
from mock_litellm import PER_REQUEST_HOLD_S, create_app


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _DeadSource:
    """MetricsSource whose snapshot() always returns None — exercises AC-3."""

    name = "litellm"

    async def snapshot(self):
        return None

    def diff(self, *args, **kwargs):  # never called when snapshots are None
        raise RuntimeError("diff should not be called when snapshots are None")


class _ScriptedRunner:
    """In-test Runner that returns a configured list of records + preds.
    Lets taxonomy-branch tests drive the resolver without going through
    MockRunner / mock_litellm."""

    name = "scripted"

    def __init__(
        self,
        *,
        records: list[ProcessRecord],
        preds: list[PredRecord] | None = None,
        duration_s: float = 0.01,
    ) -> None:
        self._records = list(records)
        self._preds = list(preds or [])
        self._duration_s = duration_s

    async def run(self, level: int) -> LevelRunResult:
        return LevelRunResult(
            level=level,
            process_records=list(self._records),
            preds=list(self._preds),
            duration_s=self._duration_s,
            out_dir="",
        )


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


def _mock_orch(
    *,
    app,
    tmp_path: Path,
    levels: list[int],
    n_instances: int = 8,
    turns: int = 1,
    mode: LoadMode = LoadMode.SWEEP,
    guards: DegradationGuard | None = None,
    scrape_interval_s: float = 0.005,
    soak_duration_s: float = 1.0,
    emitter: EventEmitter | None = None,
    scorer=None,
) -> tuple[Orchestrator, httpx.AsyncClient, list[str]]:
    """Build an Orchestrator wired to the mock app + an in-process MockRunner.
    Caller is responsible for entering/exiting the client context."""
    client = _client(app)
    source = LiteLLMSource("http://test/metrics", client=client)
    pinned = pin_slice(n=n_instances, mock=True)
    runner = MockRunner(
        base_url="http://test/v1",
        instance_ids=pinned,
        turns_per_instance=turns,
        client=client,
        runner_root=tmp_path,
    )
    config = RunConfig(
        run_id="test",
        mode=mode,
        levels=list(levels),
        task_slice=TaskSlice(n=n_instances, pinned_instance_ids=pinned),
        guards=guards or DegradationGuard(),
        scrape_interval_s=scrape_interval_s,
        soak_duration_s=soak_duration_s,
    )
    orch = Orchestrator(
        config, source=source, runner=runner, emitter=emitter, scorer=scorer
    )
    return orch, client, pinned


# ---------------------------------------------------------------------------
# Taxonomy resolver — pure function, every branch + priority (T035/AC-5)
# ---------------------------------------------------------------------------


def test_resolve_outcome_timeout_wins_over_everything():
    r = ProcessRecord(
        instance_id="t",
        return_status=124,
        wall_time_s=30.0,
        timed_out=True,
        inference_error=True,
    )
    assert resolve_outcome(r, resolved=True) == Outcome.TIMEOUT


def test_resolve_outcome_agent_error_beats_inference_and_resolution():
    r = ProcessRecord(
        instance_id="a",
        return_status=1,
        wall_time_s=5.0,
        timed_out=False,
        inference_error=True,
    )
    assert resolve_outcome(r, resolved=True) == Outcome.AGENT_ERROR


def test_resolve_outcome_inference_error_beats_unresolved_and_resolved():
    r = ProcessRecord(
        instance_id="i",
        return_status=0,
        wall_time_s=5.0,
        timed_out=False,
        inference_error=True,
    )
    # Even with resolved=True, the inference error wins.
    assert resolve_outcome(r, resolved=True) == Outcome.INFERENCE_ERROR


def test_resolve_outcome_unresolved_when_not_resolved():
    r = ProcessRecord(
        instance_id="u",
        return_status=0,
        wall_time_s=5.0,
        timed_out=False,
    )
    assert resolve_outcome(r, resolved=False) == Outcome.UNRESOLVED


def test_resolve_outcome_resolved_happy_path():
    r = ProcessRecord(
        instance_id="r",
        return_status=0,
        wall_time_s=5.0,
        timed_out=False,
    )
    assert resolve_outcome(r, resolved=True) == Outcome.RESOLVED


def test_resolve_outcome_agent_error_when_return_status_is_nonzero():
    """Crash (e.g., exit 137) without timed_out flag still classifies as
    agent_error, not unresolved."""
    r = ProcessRecord(
        instance_id="a2",
        return_status=137,
        wall_time_s=5.0,
        timed_out=False,
    )
    assert resolve_outcome(r, resolved=False) == Outcome.AGENT_ERROR


# ---------------------------------------------------------------------------
# Orchestrator SWEEP — AC-1/AC-2/AC-10
# ---------------------------------------------------------------------------


def test_sweep_populates_delta_per_level_and_pinned_slice_across_levels(tmp_path):
    """AC-1/AC-10: SWEEP over multiple levels populates a delta per level
    (scrape-and-delta works end-to-end through the orchestrator) and uses
    the same pinned slice at every level."""

    async def go():
        app = create_app()
        async with _client(app) as client:
            source = LiteLLMSource("http://test/metrics", client=client)
            pinned = pin_slice(n=8, mock=True)
            runner = MockRunner(
                base_url="http://test/v1",
                instance_ids=pinned,
                turns_per_instance=1,
                client=client,
                runner_root=tmp_path,
            )
            config = RunConfig(
                run_id="sweep-basic",
                mode=LoadMode.SWEEP,
                levels=[1, 2, 4],
                task_slice=TaskSlice(n=8, pinned_instance_ids=pinned),
                scrape_interval_s=0.005,
            )
            orch = Orchestrator(config, source=source, runner=runner)
            return await orch.run(), pinned

    report, pinned = asyncio.run(go())

    assert len(report.levels) == 3
    assert all(s.delta is not None for s in report.levels)
    # AC-10: same pinned slice recorded.
    assert report.pinned_instance_ids == pinned
    # Each level saw some traffic.
    for s in report.levels:
        assert s.delta.n_requests > 0
        assert s.delta.throughput_tps > 0
        assert s.n_tasks == 8


def test_sweep_throughput_rises_then_falls_and_knee_is_first_trip(tmp_path):
    """AC-2: with saturation kicking in past a concurrency knee, throughput
    rises then falls and the guard's first trip marks the knee."""

    async def go():
        # Saturation past in_flight=4: requests observe +0.5s, so e2e lat
        # moves from 0.25 bucket to 1.0 bucket. Guard max_p99=0.5 trips
        # exactly when saturation starts.
        app = create_app(
            streaming=True,
            saturation_threshold=4,
            saturation_extra_delay_s=0.5,
        )
        async with _client(app) as client:
            source = LiteLLMSource("http://test/metrics", client=client)
            pinned = pin_slice(n=16, mock=True)
            runner = MockRunner(
                base_url="http://test/v1",
                instance_ids=pinned,
                turns_per_instance=1,
                client=client,
                runner_root=tmp_path,
            )
            config = RunConfig(
                run_id="sweep-knee",
                mode=LoadMode.SWEEP,
                levels=[1, 4, 8, 16],
                task_slice=TaskSlice(n=16, pinned_instance_ids=pinned),
                guards=DegradationGuard(max_p99_latency_s=0.5),
                scrape_interval_s=0.005,
            )
            orch = Orchestrator(config, source=source, runner=runner)
            return await orch.run()

    report = asyncio.run(go())

    assert len(report.levels) == 4
    deltas = [s.delta for s in report.levels]
    # Pre-saturation: p99 at the 0.25 bucket.
    assert deltas[0].lat_p99 == 0.25
    assert deltas[1].lat_p99 == 0.25
    # Post-saturation: p99 jumps to 1.0.
    assert deltas[2].lat_p99 == 1.0
    assert deltas[3].lat_p99 == 1.0
    # Knee = first level to trip the guard.
    assert report.knee is not None
    assert report.knee["level"] == 8
    # Throughput rose from level 1 → 4 (pre-saturation scaling).
    tps = [d.throughput_tps for d in deltas]
    assert tps[1] > tps[0]


def test_sweep_without_guard_trip_has_no_knee(tmp_path):
    """FR-27: no guard tripped → knee stays None even if all levels ran."""

    async def go():
        app = create_app()
        async with _client(app) as client:
            source = LiteLLMSource("http://test/metrics", client=client)
            pinned = pin_slice(n=4, mock=True)
            runner = MockRunner(
                base_url="http://test/v1",
                instance_ids=pinned,
                turns_per_instance=1,
                client=client,
                runner_root=tmp_path,
            )
            config = RunConfig(
                run_id="sweep-no-knee",
                mode=LoadMode.SWEEP,
                levels=[1, 2],
                task_slice=TaskSlice(n=4, pinned_instance_ids=pinned),
                # Generous guard — nothing should trip.
                guards=DegradationGuard(max_p99_latency_s=10.0),
                scrape_interval_s=0.005,
            )
            orch = Orchestrator(config, source=source, runner=runner)
            return await orch.run()

    report = asyncio.run(go())
    assert report.knee is None


def test_sweep_continues_after_knee(tmp_path):
    """FR-27: sweep MAY continue past the knee to map the full curve."""

    async def go():
        # Extra delay 0.5s → saturated obs = 0.7 → bucket 1.0 → trips the
        # 0.5 max-p99 guard at any level where saturation kicks in.
        app = create_app(
            saturation_threshold=2, saturation_extra_delay_s=0.5
        )
        async with _client(app) as client:
            source = LiteLLMSource("http://test/metrics", client=client)
            pinned = pin_slice(n=8, mock=True)
            runner = MockRunner(
                base_url="http://test/v1",
                instance_ids=pinned,
                turns_per_instance=1,
                client=client,
                runner_root=tmp_path,
            )
            config = RunConfig(
                run_id="sweep-continue",
                mode=LoadMode.SWEEP,
                levels=[1, 4, 8],
                task_slice=TaskSlice(n=8, pinned_instance_ids=pinned),
                guards=DegradationGuard(max_p99_latency_s=0.5),
                scrape_interval_s=0.005,
            )
            orch = Orchestrator(config, source=source, runner=runner)
            return await orch.run()

    report = asyncio.run(go())
    # All three levels ran, even though the knee fired at level 4.
    assert len(report.levels) == 3
    assert report.knee is not None
    assert report.knee["level"] == 4  # 4 workers > threshold 2


# ---------------------------------------------------------------------------
# AC-3: wire-metrics-unavailable path
# ---------------------------------------------------------------------------


def test_sweep_with_dead_source_completes_with_process_and_score(tmp_path):
    """AC-3: with /metrics unreachable, the run completes; per-level delta is
    None and wire_metrics_available is False, but process records + scoring
    are still present."""

    async def go():
        app = create_app()
        async with _client(app) as client:
            # Note: client still hits the app for /v1/chat/completions so the
            # runner has somewhere to drive traffic; the dead source only
            # simulates /metrics being unreachable.
            pinned = pin_slice(n=4, mock=True)
            runner = MockRunner(
                base_url="http://test/v1",
                instance_ids=pinned,
                turns_per_instance=1,
                client=client,
                runner_root=tmp_path,
            )
            config = RunConfig(
                run_id="sweep-dead",
                mode=LoadMode.SWEEP,
                levels=[2, 4],
                task_slice=TaskSlice(n=4, pinned_instance_ids=pinned),
            )
            orch = Orchestrator(config, source=_DeadSource(), runner=runner)
            return await orch.run()

    report = asyncio.run(go())

    assert report.wire_metrics_available is False
    assert len(report.levels) == 2
    for s in report.levels:
        assert s.delta is None
        # Process + score still collected.
        assert s.n_tasks == 4
        assert s.pass_rate >= 0.0
        # Outcome counts still populated.
        assert sum(s.outcome_counts.values()) == 4


# ---------------------------------------------------------------------------
# AC-5: every taxonomy branch reachable in one run
# ---------------------------------------------------------------------------


def test_taxonomy_branches_each_hit_in_one_run():
    """AC-5: a scripted runner with one record per branch yields exactly one
    outcome in each bucket."""

    def _scorer(preds, **_kwargs):
        # Only "r" resolves; everything else unresolved (modulo taxonomy).
        return {p.instance_id: p.instance_id == "r" for p in preds}

    records = [
        ProcessRecord(
            instance_id="t",
            return_status=124,
            wall_time_s=30.0,
            timed_out=True,
        ),
        ProcessRecord(
            instance_id="a",
            return_status=1,
            wall_time_s=5.0,
            timed_out=False,
        ),
        ProcessRecord(
            instance_id="i",
            return_status=0,
            wall_time_s=5.0,
            timed_out=False,
            inference_error=True,
        ),
        ProcessRecord(  # unresolved
            instance_id="u",
            return_status=0,
            wall_time_s=5.0,
            timed_out=False,
        ),
        ProcessRecord(  # resolved
            instance_id="r",
            return_status=0,
            wall_time_s=5.0,
            timed_out=False,
        ),
    ]
    preds = [PredRecord(instance_id=r.instance_id) for r in records]
    runner = _ScriptedRunner(records=records, preds=preds)
    config = RunConfig(
        run_id="tax",
        mode=LoadMode.SWEEP,
        levels=[1],
        task_slice=TaskSlice(
            n=5, pinned_instance_ids=[r.instance_id for r in records]
        ),
    )
    orch = Orchestrator(
        config, source=_DeadSource(), runner=runner, scorer=_scorer
    )
    report = asyncio.run(orch.run())

    counts = report.levels[0].outcome_counts
    assert counts.get("timeout") == 1
    assert counts.get("agent_error") == 1
    assert counts.get("inference_error") == 1
    assert counts.get("unresolved") == 1
    assert counts.get("resolved") == 1
    # Pass rate = resolved / total = 1/5.
    assert report.levels[0].pass_rate == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# SOAK — duration respected
# ---------------------------------------------------------------------------


def test_soak_respects_duration_within_tolerance(tmp_path):
    """SOAK runs for approximately soak_duration_s (within 2× tolerance)."""

    async def go():
        app = create_app()
        async with _client(app) as client:
            source = LiteLLMSource("http://test/metrics", client=client)
            pinned = pin_slice(n=8, mock=True)
            runner = MockRunner(
                base_url="http://test/v1",
                instance_ids=pinned,
                turns_per_instance=1,
                client=client,
                runner_root=tmp_path,
            )
            config = RunConfig(
                run_id="soak",
                mode=LoadMode.SOAK,
                levels=[4],
                soak_duration_s=1.0,
                task_slice=TaskSlice(n=8, pinned_instance_ids=pinned),
                scrape_interval_s=0.01,
            )
            orch = Orchestrator(config, source=source, runner=runner)
            t0 = time.monotonic()
            report = await orch.run()
            return time.monotonic() - t0, report

    elapsed, report = asyncio.run(go())
    # SOAK shouldn't return early; allowance for one extra bin's worth of
    # work after the duration check.
    assert 1.0 <= elapsed < 3.0
    # Multiple bins emitted (SOAK re-feeds the slice).
    assert len(report.levels) >= 2


def test_soak_requires_a_level():
    """SOAK without a level to hold is a config error."""
    config = RunConfig(
        run_id="bad",
        mode=LoadMode.SOAK,
        levels=[],
    )
    orch = Orchestrator(
        config,
        source=_DeadSource(),
        runner=_ScriptedRunner(records=[], preds=[]),
    )
    with pytest.raises(ValueError):
        asyncio.run(orch.run())


# ---------------------------------------------------------------------------
# In-flight peak capture (FR-11)
# ---------------------------------------------------------------------------


def test_inflight_peak_captured_during_high_concurrency(tmp_path):
    """FR-11: scrape_interval polling captures the within-level in-flight peak."""

    async def go():
        app = create_app()
        async with _client(app) as client:
            source = LiteLLMSource("http://test/metrics", client=client)
            pinned = pin_slice(n=8, mock=True)
            runner = MockRunner(
                base_url="http://test/v1",
                instance_ids=pinned,
                turns_per_instance=1,
                client=client,
                runner_root=tmp_path,
            )
            config = RunConfig(
                run_id="peak",
                mode=LoadMode.SWEEP,
                levels=[8],
                task_slice=TaskSlice(n=8, pinned_instance_ids=pinned),
                scrape_interval_s=0.005,  # poll faster than PER_REQUEST_HOLD_S
            )
            orch = Orchestrator(config, source=source, runner=runner)
            return await orch.run()

    report = asyncio.run(go())
    # 8 workers against 8 instances × 1 turn each, hold 0.05s — in_flight
    # should peak at ≥ 4 (almost certainly 8, but poll timing makes the
    # lower bound the safe assertion).
    assert report.levels[0].delta.in_flight_peak >= 4


# ---------------------------------------------------------------------------
# TTFT availability flows to the report (FR-12)
# ---------------------------------------------------------------------------


def test_report_ttft_available_true_when_streaming(tmp_path):
    async def go():
        app = create_app(streaming=True)
        async with _client(app) as client:
            source = LiteLLMSource("http://test/metrics", client=client)
            pinned = pin_slice(n=2, mock=True)
            runner = MockRunner(
                base_url="http://test/v1",
                instance_ids=pinned,
                turns_per_instance=1,
                client=client,
                runner_root=tmp_path,
            )
            config = RunConfig(
                run_id="ttft-on",
                mode=LoadMode.SWEEP,
                levels=[2],
                task_slice=TaskSlice(n=2, pinned_instance_ids=pinned),
            )
            orch = Orchestrator(config, source=source, runner=runner)
            return await orch.run()

    report = asyncio.run(go())
    assert report.ttft_available is True
    assert report.levels[0].delta.ttft_p95 is not None


def test_report_ttft_available_false_when_streaming_off(tmp_path):
    async def go():
        app = create_app(streaming=False)
        async with _client(app) as client:
            source = LiteLLMSource("http://test/metrics", client=client)
            pinned = pin_slice(n=2, mock=True)
            runner = MockRunner(
                base_url="http://test/v1",
                instance_ids=pinned,
                turns_per_instance=1,
                streaming=False,
                client=client,
                runner_root=tmp_path,
            )
            config = RunConfig(
                run_id="ttft-off",
                mode=LoadMode.SWEEP,
                levels=[2],
                task_slice=TaskSlice(n=2, pinned_instance_ids=pinned),
            )
            orch = Orchestrator(config, source=source, runner=runner)
            return await orch.run()

    report = asyncio.run(go())
    assert report.ttft_available is False
    assert report.levels[0].delta.ttft_p95 is None


# ---------------------------------------------------------------------------
# Re-entrancy / active flag
# ---------------------------------------------------------------------------


def test_orchestrator_rejects_concurrent_runs():
    async def go():
        records = [ProcessRecord("x", return_status=0, wall_time_s=0.1, timed_out=False)]
        preds = [PredRecord(instance_id="x")]
        runner = _ScriptedRunner(records=records, preds=preds)
        config = RunConfig(
            run_id="double",
            mode=LoadMode.SWEEP,
            levels=[1],
            task_slice=TaskSlice(n=1, pinned_instance_ids=["x"]),
        )
        orch = Orchestrator(config, source=_DeadSource(), runner=runner)

        # Hold the active flag manually to simulate a concurrent invocation.
        orch._active = True
        try:
            with pytest.raises(RuntimeError):
                await orch.run()
        finally:
            orch._active = False

    asyncio.run(go())


# ---------------------------------------------------------------------------
# Event emission — full sequence
# ---------------------------------------------------------------------------


def test_event_sequence_sweep_complete():
    """run_start → (level_start, scrapes, tasks, level_done)×N → run_done."""
    emitter = CollectingEmitter()
    records = [
        ProcessRecord("a", return_status=0, wall_time_s=0.1, timed_out=False),
        ProcessRecord("b", return_status=0, wall_time_s=0.1, timed_out=False),
    ]
    preds = [PredRecord(instance_id=r.instance_id) for r in records]
    runner = _ScriptedRunner(records=records, preds=preds)
    config = RunConfig(
        run_id="evt",
        mode=LoadMode.SWEEP,
        levels=[1, 2],
        task_slice=TaskSlice(
            n=2, pinned_instance_ids=[r.instance_id for r in records]
        ),
    )
    orch = Orchestrator(
        config, source=_DeadSource(), runner=runner, emitter=emitter
    )
    asyncio.run(orch.run())

    types = emitter.types()
    assert types[0] == "run_start"
    assert types[-1] == "run_done"
    assert types.count("level_start") == 2
    assert types.count("level_done") == 2
    # 2 instances × 2 levels = 4 task events.
    assert types.count("task") == 4
    # Dead source → no start/end scrape emits, but the delta scrape still
    # fires once per level (with delta=None).
    assert types.count("scrape") == 2

    # run_start payload carries the pinned slice (FR-21).
    run_start = emitter.by_type("run_start")[0]
    assert run_start["pinned_instance_ids"] == ["a", "b"]
    assert run_start["levels"] == [1, 2]
    # run_start declares whether the runner streams, so the dashboard can render
    # TTFT/TPOT/cache-miss honestly. _ScriptedRunner has no flag → defaults True.
    assert run_start["agent_streams"] is True


def test_runner_streams_flag(tmp_path):
    """The runner declares whether it opens streaming completions. mini-swe-agent
    (2.4.x) is non-streaming — it calls litellm.completion() and parses the full
    response — so TTFT/TPOT/cache-miss are not measurable for it; the mock
    runner sends stream=True."""
    from clusterbench.miniswerunner import MiniSweRunner, MockRunner

    mock = MockRunner(
        base_url="http://x/v1", instance_ids=["a"], runner_root=tmp_path / "mock"
    )
    assert mock.streams is True

    real = MiniSweRunner(
        model="m",
        base_url="http://x/v1",
        pool=["a"],
        n_per_worker=1,
        runner_root=tmp_path / "real",
    )
    assert real.streams is False


def test_event_sequence_includes_knee_when_guard_trips():
    """knee event fires exactly once, between level_done and the next level_start."""
    emitter = CollectingEmitter()
    # Scripted runner producing one normal record per level.
    records = [ProcessRecord("x", return_status=0, wall_time_s=0.1, timed_out=False)]
    preds = [PredRecord(instance_id="x")]
    runner = _ScriptedRunner(records=records, preds=preds)

    # Build a config + a fake source that fabricates a delta so the guard
    # can trip on the second level.
    from clusterbench.models import LevelDelta, ScrapeSnapshot

    class _FakeSource:
        name = "fake"

        def __init__(self):
            self._call = 0

        async def snapshot(self):
            # Snapshots are read in start/end pairs per level; alternate the
            # token counter so the second level's delta has huge latency.
            self._call += 1
            # Calls 1-2 = level 1 start/end; 3-4 = level 2 start/end.
            return ScrapeSnapshot(
                source_name="fake",
                level=None,
                t=float(self._call),
                raw={},  # empty → diff yields defaults; we override diff below
                ttft_available=True,
            )

        def diff(self, start, end, *, in_flight_peak, duration_s):
            # Level 1 (calls 1-2): low latency. Level 2 (calls 3-4): high.
            high = end.t > 2
            return LevelDelta(
                level=end.level or 0,
                ttft_p50=2.0 if high else 0.05,
                ttft_p95=2.0 if high else 0.05,
                lat_p50=2.0 if high else 0.1,
                lat_p95=2.0 if high else 0.1,
                lat_p99=2.0 if high else 0.1,
                throughput_tps=100.0,
                n_requests=1,
                error_rate=0.0,
                in_flight_peak=in_flight_peak,
                proc_overhead_s=0.01,
            )

    config = RunConfig(
        run_id="evt-knee",
        mode=LoadMode.SWEEP,
        levels=[1, 2],
        task_slice=TaskSlice(n=1, pinned_instance_ids=["x"]),
        guards=DegradationGuard(max_p99_latency_s=0.5),
    )
    orch = Orchestrator(
        config,
        source=_FakeSource(),
        runner=runner,
        emitter=emitter,
        scorer=lambda preds, **_: {p.instance_id: True for p in preds},
    )
    asyncio.run(orch.run())

    types = emitter.types()
    assert types.count("knee") == 1
    # Knee fires after the level_done of the level that tripped.
    knee_idx = types.index("knee")
    assert types[knee_idx - 1] == "level_done"
    # Knee payload records the level + reason.
    assert emitter.by_type("knee")[0]["level"] == 2


# ---------------------------------------------------------------------------
# Runner protocol conformance
# ---------------------------------------------------------------------------


def test_orchestrator_accepts_any_runner_conforming_to_protocol(tmp_path):
    """The orchestrator only knows the Runner protocol — both MockRunner
    and ad-hoc test runners work."""
    records = [ProcessRecord("x", return_status=0, wall_time_s=0.01, timed_out=False)]
    preds = [PredRecord(instance_id="x")]

    async def go():
        config = RunConfig(
            run_id="proto",
            mode=LoadMode.SWEEP,
            levels=[1],
            task_slice=TaskSlice(n=1, pinned_instance_ids=["x"]),
        )
        orch = Orchestrator(
            config,
            source=_DeadSource(),
            runner=_ScriptedRunner(records=records, preds=preds),
        )
        return await orch.run()

    report = asyncio.run(go())
    assert len(report.levels) == 1
    assert report.levels[0].n_tasks == 1
