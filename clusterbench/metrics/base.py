"""MetricsSource protocol + bucket-edge percentile helpers (FR-9/FR-10/FR-OPT).

The orchestrator depends only on this interface, so swapping LiteLLM for vLLM
(or any future source) needs no orchestrator changes.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from clusterbench.models import LevelDelta, ScrapeSnapshot


@runtime_checkable
class MetricsSource(Protocol):
    """A scrape-and-delta source of wire-level metrics.

    `snapshot()` reads raw series (None if unreachable — FR-14). `diff()` is a
    pure function over two snapshots plus an externally-tracked in-flight peak
    and the level duration; it produces the LevelDelta the orchestrator emits.
    """

    name: str

    async def snapshot(self) -> ScrapeSnapshot | None:
        ...

    def diff(
        self,
        start: ScrapeSnapshot,
        end: ScrapeSnapshot,
        in_flight_peak: float,
        duration_s: float,
    ) -> LevelDelta:
        ...


def percentile_at_bucket_edge(
    buckets: dict[float, int | float], percentile: float
) -> float:
    """Bucket-edge percentile over Prometheus histogram `_bucket` series.

    `buckets` is the cumulative count per edge (Prometheus convention): each
    edge's count is observations with value ≤ edge; the +Inf bucket equals the
    total. The delta (end - start) histogram is what callers usually pass.

    Returns the smallest edge whose cumulative count reaches the percentile
    threshold. No interpolation — the answer is always an actual bucket edge
    (coarse by design; per the locked decision in .spec/README.md). Returns 0.0
    when there are no observations.
    """
    if not buckets:
        return 0.0
    total = buckets.get(float("inf"), 0)
    if total <= 0:
        # Malformed input (no +Inf bucket) — fall back to the max cumulative.
        total = max(buckets.values())
        if total <= 0:
            return 0.0
    target = percentile * total
    finite_edges = sorted(e for e in buckets if e != float("inf"))
    for edge in finite_edges:
        if buckets[edge] >= target:
            return float(edge)
    # Percentile falls above all finite edges — return the last one rather than
    # +Inf so downstream code gets a usable number. (Callers wanting strict
    # +Inf semantics can inspect `buckets` themselves.)
    return float(finite_edges[-1]) if finite_edges else float("inf")


def histogram_mean(buckets: dict[float, int | float], total_sum: float) -> float:
    """Mean of a histogram from its sum and count. Returns 0.0 when empty."""
    total = buckets.get(float("inf"), 0)
    if total <= 0:
        return 0.0
    return float(total_sum) / float(total)


__all__ = [
    "MetricsSource",
    "percentile_at_bucket_edge",
    "histogram_mean",
    "LevelDelta",
    "ScrapeSnapshot",
]
