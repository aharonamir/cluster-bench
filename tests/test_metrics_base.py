"""Tests for clusterbench.metrics.base (Protocol + bucket-edge percentiles)."""
from clusterbench.metrics.base import (
    MetricsSource,
    histogram_mean,
    percentile_at_bucket_edge,
)


def test_percentile_basic():
    # 100 observations spread across edges; cumulative counts below. Fixture
    # is set up so p50/p95/p99 each land on a distinct edge.
    buckets = {
        0.01: 0,
        0.05: 5,
        0.1: 50,   # p50 lands here (count >= 50)
        0.25: 95,  # p95 lands here (count >= 95)
        0.5: 99,   # p99 lands here (count >= 99)
        1.0: 100,
        float("inf"): 100,
    }
    assert percentile_at_bucket_edge(buckets, 0.50) == 0.1
    assert percentile_at_bucket_edge(buckets, 0.95) == 0.25
    assert percentile_at_bucket_edge(buckets, 0.99) == 0.5


def test_percentile_max_returns_last_finite_edge_with_full_count():
    buckets = {
        0.01: 0,
        0.05: 10,
        0.1: 50,
        0.25: 90,
        0.5: 100,
        float("inf"): 100,
    }
    # p100 — smallest edge with count >= 100 is 0.5.
    assert percentile_at_bucket_edge(buckets, 1.0) == 0.5


def test_percentile_all_in_one_bucket():
    # All observations fall in the 0.25 bucket — every percentile returns 0.25.
    buckets = {0.01: 0, 0.05: 0, 0.1: 0, 0.25: 100, 0.5: 100, float("inf"): 100}
    for p in (0.50, 0.95, 0.99):
        assert percentile_at_bucket_edge(buckets, p) == 0.25


def test_percentile_empty_buckets():
    assert percentile_at_bucket_edge({}, 0.5) == 0.0
    assert percentile_at_bucket_edge({0.1: 0, float("inf"): 0}, 0.5) == 0.0


def test_histogram_mean():
    buckets = {0.1: 50, float("inf"): 50}
    assert histogram_mean(buckets, total_sum=10.0) == 0.2
    assert histogram_mean({}, total_sum=0.0) == 0.0


def test_vllm_source_conforms_to_metrics_source_protocol():
    """AC-OPT: VLLMSource registers + conforms to the MetricsSource interface,
    even though it's a stub not selectable in this topology."""
    from clusterbench.metrics.vllm import VLLMSource

    src = VLLMSource("http://vllm:8000/metrics")
    assert isinstance(src, MetricsSource)
    assert src.name == "vllm"
    assert hasattr(src, "snapshot")
    assert hasattr(src, "diff")
