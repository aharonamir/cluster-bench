"""Dataclasses and enums for ClusterBench.

Shapes are authoritative (see .spec/plan.md "Data shapes"). Phase 0 only
requires importability + RunConfig/RunReport round-trip; later phases fill in
the producers and consumers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class LoadMode(str, Enum):
    SWEEP = "sweep"
    SOAK = "soak"


class Outcome(str, Enum):
    """Per-attempt outcome. Mutually exclusive (FR-19).

    Priority order for resolution: TIMEOUT > AGENT_ERROR > INFERENCE_ERROR >
    UNRESOLVED > RESOLVED. See OUTCOME_PRIORITY.
    """

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    INFERENCE_ERROR = "inference_error"
    AGENT_ERROR = "agent_error"
    TIMEOUT = "timeout"


OUTCOME_PRIORITY: tuple[Outcome, ...] = (
    Outcome.TIMEOUT,
    Outcome.AGENT_ERROR,
    Outcome.INFERENCE_ERROR,
    Outcome.UNRESOLVED,
    Outcome.RESOLVED,
)


@dataclass
class DegradationGuard:
    """Per-level eval guards (FR-26). Any tripping marks the knee."""

    max_p99_latency_s: float | None = None
    max_ttft_p95_s: float | None = None
    max_error_rate: float | None = None
    min_pass_rate: float | None = None
    max_in_flight_peak: float | None = None

    def evaluate(
        self,
        *,
        p99_latency_s: float | None,
        ttft_p95_s: float | None,
        error_rate: float,
        pass_rate: float,
        in_flight_peak: float,
    ) -> str | None:
        """Return the first tripped guard's reason, or None if all pass."""
        if self.max_p99_latency_s is not None and p99_latency_s is not None:
            if p99_latency_s > self.max_p99_latency_s:
                return f"p99_latency {p99_latency_s:.3f}s > {self.max_p99_latency_s:.3f}s"
        if self.max_ttft_p95_s is not None and ttft_p95_s is not None:
            if ttft_p95_s > self.max_ttft_p95_s:
                return f"ttft_p95 {ttft_p95_s:.3f}s > {self.max_ttft_p95_s:.3f}s"
        if self.max_error_rate is not None and error_rate > self.max_error_rate:
            return f"error_rate {error_rate:.3f} > {self.max_error_rate:.3f}"
        if self.min_pass_rate is not None and pass_rate < self.min_pass_rate:
            return f"pass_rate {pass_rate:.3f} < {self.min_pass_rate:.3f}"
        if self.max_in_flight_peak is not None and in_flight_peak > self.max_in_flight_peak:
            return (
                f"in_flight_peak {in_flight_peak:.1f} > {self.max_in_flight_peak:.1f}"
            )
        return None


@dataclass
class TaskSlice:
    """SWE-bench Verified slice (FR-2/FR-7). pinned_instance_ids, when non-empty,
    is reused verbatim at every sweep level (AC-10)."""

    subset: str = "verified"
    split: str = "test"
    n: int = 5
    pinned_instance_ids: list[str] = field(default_factory=list)


@dataclass
class MiniSweConfig:
    """mini-swe-agent invocation knobs (FR-3/FR-12)."""

    model: str = "gpt-4o-mini"
    streaming: bool = True
    step_limit: int = 0  # 0 = unlimited


@dataclass
class RunConfig:
    run_id: str
    name: str = ""
    mode: LoadMode = LoadMode.SWEEP
    # SWEEP: list of worker counts to visit sequentially (FR-8).
    # SOAK: single-element list (the level to hold).
    levels: list[int] = field(default_factory=lambda: [1, 4, 8, 16])
    soak_duration_s: float = 1800.0  # SOAK only
    task_slice: TaskSlice = field(default_factory=TaskSlice)
    guards: DegradationGuard = field(default_factory=DegradationGuard)
    scrape_interval_s: float = 1.0
    miniswe: MiniSweConfig = field(default_factory=MiniSweConfig)
    # LiteLLM is the only reachable layer (rev 3 topology).
    litellm_metrics_url: str = "http://localhost:4000/metrics"
    litellm_base_url: str = "http://localhost:4000/v1"
    # FR-OPT — usually None; vLLM pods unreachable in this topology.
    vllm_metrics_url: str | None = None

    def __post_init__(self) -> None:
        # Accept the enum value (str) on construction for ergonomics.
        if not isinstance(self.mode, LoadMode):
            self.mode = LoadMode(self.mode)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "mode": self.mode.value,
            "levels": list(self.levels),
            "soak_duration_s": self.soak_duration_s,
            "task_slice": asdict(self.task_slice),
            "guards": asdict(self.guards),
            "scrape_interval_s": self.scrape_interval_s,
            "miniswe": asdict(self.miniswe),
            "litellm_metrics_url": self.litellm_metrics_url,
            "litellm_base_url": self.litellm_base_url,
            "vllm_metrics_url": self.vllm_metrics_url,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        return cls(
            run_id=d["run_id"],
            name=d.get("name", ""),
            mode=LoadMode(d.get("mode", LoadMode.SWEEP.value)),
            levels=list(d.get("levels", [1, 4, 8, 16])),
            soak_duration_s=d.get("soak_duration_s", 1800.0),
            task_slice=TaskSlice(**d.get("task_slice", {})),
            guards=DegradationGuard(**d.get("guards", {})),
            scrape_interval_s=d.get("scrape_interval_s", 1.0),
            miniswe=MiniSweConfig(**d.get("miniswe", {})),
            litellm_metrics_url=d.get(
                "litellm_metrics_url", "http://localhost:4000/metrics"
            ),
            litellm_base_url=d.get("litellm_base_url", "http://localhost:4000/v1"),
            vllm_metrics_url=d.get("vllm_metrics_url"),
        )


@dataclass
class ScrapeSnapshot:
    """One timed read of a MetricsSource (FR-9/FR-10). raw is series-name → value
    (counters/gauges) or series-name → {bucket_edge: cumulative_count} for
    histograms. Phase 1 pins the structure."""

    source_name: str
    level: int | None  # None for the baseline scrape
    t: float
    raw: dict[str, Any] = field(default_factory=dict)
    ttft_available: bool = True


@dataclass
class LevelDelta:
    """Differenced stats for one level (FR-10/FR-13). Percentiles are bucket-edge
    values (coarse, per the scrape-and-delta design). ttft_* are None when
    ttft_available is False (FR-12). queue_* are None when the source has no
    queue-time series (e.g. mock); real LiteLLM exposes
    `litellm_request_queue_time_seconds`, which directly measures pre-handler
    queueing (FR-16)."""

    level: int
    ttft_p50: float | None
    ttft_p95: float | None
    lat_p50: float
    lat_p95: float
    lat_p99: float
    throughput_tps: float
    n_requests: int
    error_rate: float
    in_flight_peak: float
    proc_overhead_s: float
    suspect: bool = False  # counter-reset guard (FR-10)
    queue_p50: float | None = None
    queue_p95: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LevelDelta":
        return cls(**d)


@dataclass
class TaskOutcome:
    """Per-attempt record (FR-17/FR-19). One outcome bucket per attempt."""

    run_id: str
    level: int
    instance_id: str
    outcome: Outcome
    resolved: bool
    return_status: int | None
    wall_time_s: float
    timed_out: bool
    log_tail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, Outcome):
            self.outcome = Outcome(self.outcome)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaskOutcome":
        return cls(
            run_id=d["run_id"],
            level=d["level"],
            instance_id=d["instance_id"],
            outcome=Outcome(d["outcome"]),
            resolved=d["resolved"],
            return_status=d["return_status"],
            wall_time_s=d["wall_time_s"],
            timed_out=d["timed_out"],
            log_tail=d.get("log_tail", ""),
        )


@dataclass
class LevelSummary:
    """Aggregated view of one level (FR-20): delta + scorer + outcome taxonomy."""

    level: int
    delta: LevelDelta
    n_tasks: int
    pass_rate: float
    outcome_counts: dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "delta": self.delta.to_dict(),
            "n_tasks": self.n_tasks,
            "pass_rate": self.pass_rate,
            "outcome_counts": dict(self.outcome_counts),
            "duration_s": self.duration_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LevelSummary":
        return cls(
            level=d["level"],
            delta=LevelDelta.from_dict(d["delta"]),
            n_tasks=d["n_tasks"],
            pass_rate=d["pass_rate"],
            outcome_counts=dict(d.get("outcome_counts", {})),
            duration_s=d.get("duration_s", 0.0),
        )


@dataclass
class RunReport:
    run_id: str
    name: str
    config: RunConfig
    wire_metrics_available: bool = True
    ttft_available: bool = True
    levels: list[LevelSummary] = field(default_factory=list)
    knee: dict[str, Any] | None = None  # {"level": int, "reason": str}
    pinned_instance_ids: list[str] = field(default_factory=list)
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "config": self.config.to_dict(),
            "wire_metrics_available": self.wire_metrics_available,
            "ttft_available": self.ttft_available,
            "levels": [lv.to_dict() for lv in self.levels],
            "knee": self.knee,
            "pinned_instance_ids": list(self.pinned_instance_ids),
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunReport":
        return cls(
            run_id=d["run_id"],
            name=d.get("name", ""),
            config=RunConfig.from_dict(d["config"]),
            wire_metrics_available=d.get("wire_metrics_available", True),
            ttft_available=d.get("ttft_available", True),
            levels=[LevelSummary.from_dict(lv) for lv in d.get("levels", [])],
            knee=d.get("knee"),
            pinned_instance_ids=list(d.get("pinned_instance_ids", [])),
            finished_at=d.get("finished_at"),
        )
