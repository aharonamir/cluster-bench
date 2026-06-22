"""Orchestrator — SWEEP/SOAK loop with scrape-and-delta per level (Phase 3).

Drives the per-level engine: scrape@start → run Runner(level) while polling
in-flight peak → scrape@end → MetricsSource.diff() → LevelDelta. Combines
with the scorer + taxonomy resolver to build LevelSummary; evaluates guards
to find the knee. Emits events throughout (Phase 4 WebSocket hub is another
EventEmitter implementation).

Cross-cutting guarantees:
  - One level at a time (FR-8).
  - Same pinned slice every level (AC-10).
  - Source unreachable → run completes with delta=None + wire flag (FR-14/AC-3).
  - TTFT availability flows from snapshot to report (FR-12).
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, runtime_checkable

from clusterbench.metrics.base import MetricsSource
from clusterbench.metrics.litellm import SERIES_IN_FLIGHT, compute_live_stats
from clusterbench.miniswerunner import Runner, pin_slice
from clusterbench.models import (
    LevelRunResult,
    LevelSummary,
    LoadMode,
    Outcome,
    ProcessRecord,
    RunConfig,
    RunReport,
    TaskOutcome,
)
from clusterbench.scoring import score_predictions


# ---------------------------------------------------------------------------
# EventEmitter — Phase 4 wires a WebSocket hub as another implementation
# ---------------------------------------------------------------------------


@runtime_checkable
class EventEmitter(Protocol):
    async def emit(self, event_type: str, payload: dict[str, Any]) -> None: ...


class NullEmitter:
    """No-op emitter; default when the caller doesn't supply one."""

    async def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        return None


class CollectingEmitter:
    """Captures all events in order; for tests."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))

    def types(self) -> list[str]:
        return [t for t, _ in self.events]

    def by_type(self, event_type: str) -> list[dict[str, Any]]:
        return [p for t, p in self.events if t == event_type]


# ---------------------------------------------------------------------------
# Outcome taxonomy resolver (T035, FR-19)
# ---------------------------------------------------------------------------


def resolve_outcome(record: ProcessRecord, *, resolved: bool) -> Outcome:
    """Pure function. Priority: timeout → agent_error → inference_error →
    unresolved → resolved. Mutually exclusive (FR-19).

    `resolved` is the scorer's verdict on the instance's patch. Per-instance
    inference_error comes from ProcessRecord.inference_error (log-detected).
    The level-granular failed-request rate from LiteLLM is a separate signal
    on LevelDelta.error_rate — not auto-promoted to per-instance outcome,
    since it can't be attributed reliably to a single instance.
    """
    if record.timed_out:
        return Outcome.TIMEOUT
    if record.return_status not in (None, 0):
        return Outcome.AGENT_ERROR
    if record.inference_error:
        return Outcome.INFERENCE_ERROR
    if not resolved:
        return Outcome.UNRESOLVED
    return Outcome.RESOLVED


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Runs a SWEEP or SOAK against the configured source + runner.

    Construct with all dependencies; `await run()` returns the RunReport and
    emits events along the way. Re-entrancy is rejected (HTTP layer uses this
    to return 409 on a second concurrent /api/run — Phase 4).
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        source: MetricsSource,
        runner: Runner,
        scorer: Callable[[list[Any]], dict[str, bool]] | None = None,
        scorer_pass_rate: float = 0.5,
        emitter: EventEmitter | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.source = source
        self.runner = runner
        if scorer is None:
            # Bind pass_rate so the call site is uniform; custom scorers
            # just take preds and return dict[str, bool].
            pass_rate = scorer_pass_rate
            self._scorer: Callable[[list[Any]], dict[str, bool]] = (
                lambda preds: score_predictions(preds, mock_pass_rate=pass_rate)
            )
        else:
            self._scorer = scorer
        self.emitter = emitter or NullEmitter()
        self._clock = clock
        self._active = False
        # Reset per-run state.
        self._ttft_available_seen = False

    @property
    def active(self) -> bool:
        return self._active

    async def run(self) -> RunReport:
        if self._active:
            raise RuntimeError("orchestrator already running")
        self._active = True
        self._ttft_available_seen = False
        # Accumulate summaries here so a cancellation can save a partial report.
        self._completed_summaries: list[LevelSummary] = []
        self._completed_knee: dict[str, Any] | None = None
        cancelled = False
        pinned: list[str] = list(self.config.task_slice.pinned_instance_ids)
        try:
            pinned = self._resolve_pinned()
            await self.emitter.emit(
                "run_start",
                {
                    "run_id": self.config.run_id,
                    "name": self.config.name,
                    "mode": self.config.mode.value,
                    "levels": list(self.config.levels),
                    "pinned_instance_ids": list(pinned),
                },
            )

            knee: dict[str, Any] | None = None
            if self.config.mode == LoadMode.SWEEP:
                summaries, knee = await self._sweep(pinned)
            else:
                summaries = await self._soak(pinned)

        except asyncio.CancelledError:
            # Build a partial report from whichever levels finished before the
            # cancel arrived. Returning normally (not re-raising) lets _drive
            # reach save_report so the partial data isn't lost.
            summaries = list(self._completed_summaries)
            knee = self._completed_knee
            cancelled = True

        finally:
            self._active = False

        wire_available = any(s.delta is not None for s in summaries)
        report = RunReport(
            run_id=self.config.run_id,
            name=self.config.name,
            config=self.config,
            wire_metrics_available=wire_available,
            ttft_available=self._ttft_available_seen,
            levels=summaries,
            knee=knee,
            pinned_instance_ids=list(pinned),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        await self.emitter.emit(
            "run_done",
            {
                "run_id": self.config.run_id,
                "knee": knee,
                "n_levels": len(summaries),
                "cancelled": cancelled,
            },
        )
        return report

    # ------------------------------------------------------------------
    # Pinned slice
    # ------------------------------------------------------------------

    def _resolve_pinned(self) -> list[str]:
        if self.config.task_slice.pinned_instance_ids:
            return list(self.config.task_slice.pinned_instance_ids)
        return pin_slice(
            n=self.config.task_slice.n,
            subset=self.config.task_slice.subset,
            split=self.config.task_slice.split,
            mock=True,
        )

    # ------------------------------------------------------------------
    # SWEEP (T032, FR-6/FR-8)
    # ------------------------------------------------------------------

    async def _sweep(
        self, pinned: list[str]
    ) -> tuple[list[LevelSummary], dict[str, Any] | None]:
        summaries: list[LevelSummary] = []
        knee: dict[str, Any] | None = None
        for level in self.config.levels:
            summary = await self._run_level(level, pinned)
            summaries.append(summary)
            self._completed_summaries.append(summary)
            await self.emitter.emit("level_done", summary.to_dict())
            if knee is None:
                reason = self._eval_guards(summary)
                if reason is not None:
                    knee = {"level": level, "reason": reason}
                    self._completed_knee = knee
                    await self.emitter.emit("knee", knee)
        return summaries, knee

    # ------------------------------------------------------------------
    # SOAK (T033, FR-6)
    # ------------------------------------------------------------------

    async def _soak(self, pinned: list[str]) -> list[LevelSummary]:
        """Hold one level for soak_duration_s, re-feeding the pinned slice.
        Each iteration of runner.run() produces one bin's LevelSummary; we
        stop once the cumulative wall time crosses the target."""
        if not self.config.levels:
            raise ValueError("SOAK requires at least one level in config.levels")
        level = self.config.levels[0]
        target_s = self.config.soak_duration_s

        summaries: list[LevelSummary] = []
        t_start = self._clock()
        bin_idx = 0
        # Safety bound: 10k bins max — would indicate a runner with sub-millisecond
        # turnaround, which would otherwise spin forever against a tiny duration.
        while bin_idx < 10_000:
            elapsed = self._clock() - t_start
            if elapsed >= target_s:
                break
            summary = await self._run_level(level, pinned, bin_idx=bin_idx)
            summaries.append(summary)
            self._completed_summaries.append(summary)
            await self.emitter.emit("level_done", summary.to_dict())
            bin_idx += 1
        return summaries

    # ------------------------------------------------------------------
    # Per-level engine (T031, FR-10/FR-11)
    # ------------------------------------------------------------------

    async def _run_level(
        self,
        level: int,
        pinned: list[str],
        *,
        bin_idx: int | None = None,
    ) -> LevelSummary:
        level_payload = (
            {"level": level, "bin_idx": bin_idx}
            if bin_idx is not None
            else {"level": level}
        )
        await self.emitter.emit("level_start", level_payload)

        start_snap = await self.source.snapshot()
        if start_snap is not None:
            start_snap.level = level
            if start_snap.ttft_available:
                self._ttft_available_seen = True
            await self.emitter.emit(
                "scrape", {"phase": "start", **level_payload}
            )

        t0 = self._clock()
        result, peak = await self._run_with_inflight_polling(level, start_snap=start_snap)
        duration = self._clock() - t0

        end_snap = await self.source.snapshot()
        if end_snap is not None:
            end_snap.level = level
            if end_snap.ttft_available:
                self._ttft_available_seen = True
            await self.emitter.emit(
                "scrape", {"phase": "end", **level_payload}
            )

        delta = None
        if start_snap is not None and end_snap is not None:
            delta = self.source.diff(
                start_snap,
                end_snap,
                in_flight_peak=peak,
                duration_s=duration,
            )
        await self.emitter.emit(
            "scrape",
            {
                "phase": "delta",
                **level_payload,
                "delta": delta.to_dict() if delta is not None else None,
            },
        )

        resolved_set = self._scorer(result.preds)
        outcomes, outcome_counts = self._build_outcomes(level, result, resolved_set)
        pass_rate = (
            outcome_counts.get(Outcome.RESOLVED.value, 0) / len(outcomes)
            if outcomes
            else 0.0
        )

        for oc in outcomes:
            await self.emitter.emit(
                "task",
                {
                    **level_payload,
                    "instance_id": oc.instance_id,
                    "outcome": oc.outcome.value,
                    "resolved": oc.resolved,
                },
            )

        return LevelSummary(
            level=level,
            delta=delta,
            n_tasks=len(outcomes),
            pass_rate=pass_rate,
            outcome_counts=outcome_counts,
            duration_s=duration,
        )

    async def _run_with_inflight_polling(
        self,
        level: int,
        *,
        start_snap: Any | None = None,
    ) -> tuple[LevelRunResult, float]:
        """Run the batch while polling the source on the scrape interval.

        Tracks in-flight peak (FR-11) and emits `level_live` events so the
        dashboard can show TTFT, latency, tok/s, and TPOT while the level runs.
        """
        peak: float = 0.0
        t_start = self._clock()

        async def poll() -> None:
            nonlocal peak
            while True:
                await asyncio.sleep(self.config.scrape_interval_s)
                snap = await self.source.snapshot()
                if snap is None:
                    continue
                current = snap.raw.get(SERIES_IN_FLIGHT, 0)
                if isinstance(current, (int, float)) and current > peak:
                    peak = current
                if start_snap is not None:
                    elapsed = self._clock() - t_start
                    live = compute_live_stats(start_snap, snap, elapsed, level)
                    await self.emitter.emit("level_live", live)

        poll_task = asyncio.create_task(poll())
        try:
            result = await self.runner.run(level)
        finally:
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass
        return result, peak

    # ------------------------------------------------------------------
    # Combining runner output + scorer → TaskOutcome (T034/T035)
    # ------------------------------------------------------------------

    def _build_outcomes(
        self,
        level: int,
        result: LevelRunResult,
        resolved_set: dict[str, bool],
    ) -> tuple[list[TaskOutcome], dict[str, int]]:
        outcomes: list[TaskOutcome] = []
        counts: dict[str, int] = {}
        for record in result.process_records:
            resolved = bool(resolved_set.get(record.instance_id, False))
            outcome = resolve_outcome(record, resolved=resolved)
            outcomes.append(
                TaskOutcome(
                    run_id=self.config.run_id,
                    level=level,
                    instance_id=record.instance_id,
                    outcome=outcome,
                    resolved=resolved,
                    return_status=record.return_status,
                    wall_time_s=record.wall_time_s,
                    timed_out=record.timed_out,
                    log_tail=record.log_tail,
                )
            )
            counts[outcome.value] = counts.get(outcome.value, 0) + 1
        return outcomes, counts

    # ------------------------------------------------------------------
    # Guards + knee (T036, FR-26/FR-27)
    # ------------------------------------------------------------------

    def _eval_guards(self, summary: LevelSummary) -> str | None:
        if summary.delta is None:
            return None
        return self.config.guards.evaluate(
            p99_latency_s=summary.delta.lat_p99,
            ttft_p95_s=summary.delta.ttft_p95,
            error_rate=summary.delta.error_rate,
            pass_rate=summary.pass_rate,
            in_flight_peak=summary.delta.in_flight_peak,
        )


__all__ = [
    "Orchestrator",
    "EventEmitter",
    "NullEmitter",
    "CollectingEmitter",
    "resolve_outcome",
]
