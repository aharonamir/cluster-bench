"""Real-world LiteLLM fixture test (tests/data/lite-llm-metrics).

The fixture is a sanitized capture from a live LiteLLM `/metrics` scrape.
These tests verify the parser handles real multi-label series, that all FR-9
required series are recognized, and that the diff math produces sane values
against a synthetic zero-start snapshot.
"""
from pathlib import Path

from clusterbench.metrics.litellm import (
    SERIES_FAILED_REQUESTS,
    SERIES_IN_FLIGHT,
    SERIES_INPUT_TOKENS,
    SERIES_LATENCY_E2E,
    SERIES_LATENCY_LLM_API,
    SERIES_OUTPUT_TOKENS,
    SERIES_PROC_OVERHEAD,
    SERIES_QUEUE_TIME,
    SERIES_REQUESTS,
    SERIES_TOTAL_TOKENS,
    SERIES_TTFT,
    LiteLLMSource,
    aggregate_histogram,
    parse_prometheus_text,
    sum_by_label,
    sum_unlabeled,
)
from clusterbench.models import ScrapeSnapshot

FIXTURE = Path(__file__).parent / "data" / "lite-llm-metrics"

REQUIRED_SERIES = {
    "input tokens": SERIES_INPUT_TOKENS,
    "output tokens": SERIES_OUTPUT_TOKENS,
    "total tokens": SERIES_TOTAL_TOKENS,
    "requests w/ status_code": SERIES_REQUESTS,
    "failed requests": SERIES_FAILED_REQUESTS,
    "in-flight requests": SERIES_IN_FLIGHT,
    "e2e latency histogram": SERIES_LATENCY_E2E,
    "LLM API latency histogram": SERIES_LATENCY_LLM_API,
    "proc overhead histogram": SERIES_PROC_OVERHEAD,
    "TTFT histogram": SERIES_TTFT,
    "queue time histogram": SERIES_QUEUE_TIME,
}


def _load() -> str:
    return FIXTURE.read_text()


def test_fixture_file_exists_and_was_sanitized():
    text = _load()
    assert "yair" not in text
    assert "KIMI" not in text
    assert "10.244." not in text
    assert "claude-cli" not in text
    assert "e5d81881" not in text
    assert "0b37d306" not in text
    assert len(text) > 10_000  # sanity: fixture still has real bulk


def test_all_required_series_present_in_real_fixture():
    """Every FR-9 series plus queue_time and failed_requests appears in the
    fixture — confirms our names match real LiteLLM output."""
    samples = parse_prometheus_text(_load())
    sample_names = {s.name for s in samples}

    missing: list[str] = []
    for desc, base_name in REQUIRED_SERIES.items():
        # For histograms/counters, samples have suffixes (_bucket/_count/_sum
        # or are unlabeled). Check any sample name starts with the base.
        if not any(name == base_name or name.startswith(f"{base_name}_") for name in sample_names):
            missing.append(f"{desc} ({base_name})")
    assert not missing, f"missing series in real fixture: {missing}"


def test_snapshot_aggregates_real_fixture_to_sane_totals():
    """End-to-end: parse → aggregate → all FR-9 series produce non-trivial
    values (no silent zero / empty bucket dicts)."""
    samples = parse_prometheus_text(_load())
    raw = LiteLLMSource._aggregate(samples)

    # Counters — must be positive (this is a live capture, not the mock).
    assert raw[SERIES_INPUT_TOKENS] > 0
    assert raw[SERIES_OUTPUT_TOKENS] > 0
    assert raw[SERIES_TOTAL_TOKENS] == raw[SERIES_INPUT_TOKENS] + raw[SERIES_OUTPUT_TOKENS]
    assert raw[SERIES_FAILED_REQUESTS] > 0
    assert raw[SERIES_IN_FLIGHT] >= 0

    # Request counter — multi-label, must collapse to status_code-keyed dict.
    requests_by_status = raw[SERIES_REQUESTS]
    assert isinstance(requests_by_status, dict)
    assert "200" in requests_by_status and requests_by_status["200"] > 0
    # The fixture includes a 400 from a ProxyModelNotFoundError request.
    assert "400" in requests_by_status

    # Histograms — must have at least one finite bucket and a +Inf.
    for name in (
        SERIES_LATENCY_E2E,
        SERIES_LATENCY_LLM_API,
        SERIES_PROC_OVERHEAD,
        SERIES_TTFT,
        SERIES_QUEUE_TIME,
    ):
        hist = raw[name]
        assert float("inf") in hist["buckets"], f"{name} missing +Inf bucket"
        assert hist["count"] > 0, f"{name} count should be positive"
        assert hist["sum"] > 0, f"{name} sum should be positive"


def test_snapshot_sets_ttft_available_for_real_data():
    """Real LiteLLM with streaming emits TTFT — snapshot must report
    ttft_available=True (FR-12)."""

    class _FakeResponse:
        text = _load()

        def raise_for_status(self) -> None:
            return None

    class _FakeClient:
        async def get(self, _url):
            return _FakeResponse()

    src = LiteLLMSource("http://test/metrics", client=_FakeClient())  # type: ignore[arg-type]
    import asyncio

    snap = asyncio.run(src.snapshot())
    assert snap is not None
    assert snap.ttft_available is True
    assert SERIES_QUEUE_TIME in snap.raw  # bonus: queue time directly measured


def test_diff_against_real_fixture_with_synthetic_zero_start():
    """diff() with a zero-start snapshot yields deltas equal to the fixture's
    point-in-time values, and bucket-edge percentiles land on real edges."""
    samples = parse_prometheus_text(_load())
    raw = LiteLLMSource._aggregate(samples)

    start = ScrapeSnapshot(
        source_name="litellm",
        level=None,
        t=0.0,
        raw={k: ({} if isinstance(v, dict) else 0.0) for k, v in raw.items()},
        ttft_available=True,
    )
    end = ScrapeSnapshot(
        source_name="litellm",
        level=8,
        t=100.0,
        raw=raw,
        ttft_available=True,
    )

    src = LiteLLMSource("http://test/metrics")
    delta = src.diff(start, end, in_flight_peak=8.0, duration_s=100.0)

    assert delta.level == 8
    assert delta.n_requests > 0
    assert delta.throughput_tps == raw[SERIES_TOTAL_TOKENS] / 100.0
    assert delta.error_rate > 0  # fixture has 400s and failed_requests
    # All bucket-edge percentiles are positive real numbers from the real
    # fixture's bucket schema (no +Inf leakage, no 0).
    assert delta.lat_p50 > 0
    assert delta.lat_p95 >= delta.lat_p50
    assert delta.lat_p99 >= delta.lat_p95
    assert delta.ttft_p50 is not None and delta.ttft_p50 > 0
    assert delta.ttft_p95 is not None
    # Bonus: queue time directly measured (FR-16 upgraded from heuristic).
    assert delta.queue_p50 is not None and delta.queue_p50 >= 0
    assert delta.queue_p95 is not None
    assert delta.suspect is False  # synthetic zero-start never triggers reset
