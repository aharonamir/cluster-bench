"""Tests for clusterbench.metrics.litellm (parser, snapshot, diff).

Covers AC-4 (delta math against deterministic mock), AC-11 (TTFT-absent
detected), and FR-14 (unreachable → None)."""
import asyncio

import httpx

from clusterbench.metrics.litellm import (
    SERIES_FAILED_REQUESTS,
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
from mock_litellm import TOTAL_TOKENS_PER_REQUEST, create_app


def _client(app=None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app or create_app()),
        base_url="http://test",
    )


# ---------------------------------------------------------------------------
# Parser + aggregators
# ---------------------------------------------------------------------------


def test_parser_and_aggregators_handle_multi_label_series():
    """Real LiteLLM emits multi-label counters + multi-label histograms.
    Aggregators must collapse to system-wide totals."""
    text = """# HELP litellm_total_tokens_metric_total Total tokens served
# TYPE litellm_total_tokens_metric_total counter
litellm_total_tokens_metric_total{api_key_alias="a",model="m1"} 100
litellm_total_tokens_metric_total{api_key_alias="b",model="m2"} 50
# HELP litellm_proxy_total_requests_metric_total Client-side requests
# TYPE litellm_proxy_total_requests_metric_total counter
litellm_proxy_total_requests_metric_total{api_key_alias="a",status_code="200",model="m1"} 40
litellm_proxy_total_requests_metric_total{api_key_alias="a",status_code="500",model="m1"} 2
litellm_proxy_total_requests_metric_total{api_key_alias="b",status_code="200",model="m2"} 10
# HELP litellm_in_flight_requests In flight
# TYPE litellm_in_flight_requests gauge
litellm_in_flight_requests 3
# HELP litellm_request_total_latency_metric E2E latency
# TYPE litellm_request_total_latency_metric histogram
litellm_request_total_latency_metric_bucket{api_key_alias="a",le="0.1",model="m1"} 30
litellm_request_total_latency_metric_bucket{api_key_alias="b",le="0.1",model="m2"} 5
litellm_request_total_latency_metric_bucket{api_key_alias="a",le="0.5",model="m1"} 42
litellm_request_total_latency_metric_bucket{api_key_alias="b",le="0.5",model="m2"} 10
litellm_request_total_latency_metric_bucket{api_key_alias="a",le="+Inf",model="m1"} 42
litellm_request_total_latency_metric_bucket{api_key_alias="b",le="+Inf",model="m2"} 10
litellm_request_total_latency_metric_count{api_key_alias="a",model="m1"} 42
litellm_request_total_latency_metric_count{api_key_alias="b",model="m2"} 10
litellm_request_total_latency_metric_sum{api_key_alias="a",model="m1"} 8.4
litellm_request_total_latency_metric_sum{api_key_alias="b",model="m2"} 2.0
"""
    samples = parse_prometheus_text(text)

    # Multi-label counter → summed across all label combos.
    assert sum_unlabeled(samples, "litellm_total_tokens_metric_total") == 150.0
    # Multi-label request counter → grouped by status_code, summed across others.
    assert sum_by_label(samples, "litellm_proxy_total_requests_metric_total", "status_code") == {
        "200": 50.0,
        "500": 2.0,
    }
    # Unlabeled gauge → simple sum.
    assert sum_unlabeled(samples, "litellm_in_flight_requests") == 3.0
    # Multi-label histogram → buckets summed across label combos (except le).
    hist = aggregate_histogram(samples, "litellm_request_total_latency_metric")
    assert hist["buckets"] == {0.1: 35, 0.5: 52, float("inf"): 52}
    assert hist["count"] == 52
    assert hist["sum"] == 10.4


def test_parser_skips_comments_and_garbage():
    text = """# HELP litellm_total_tokens_metric_total Total tokens served
# TYPE litellm_total_tokens_metric_total counter
litellm_total_tokens_metric_total 5
not a metric line
"""
    samples = parse_prometheus_text(text)
    assert sum_unlabeled(samples, "litellm_total_tokens_metric_total") == 5.0
    assert len(samples) == 1  # garbage line skipped


# ---------------------------------------------------------------------------
# snapshot()
# ---------------------------------------------------------------------------


def test_snapshot_returns_parsed_raw_and_detects_ttft_present():
    async def go():
        async with _client(create_app(streaming=True)) as client:
            src = LiteLLMSource("http://test/metrics", client=client)
            # Serve one request so histograms are non-empty.
            await client.post(
                "/v1/chat/completions", json={"model": "m", "messages": []}
            )
            snap = await src.snapshot()
            assert snap is not None
            assert snap.source_name == "litellm"
            assert snap.level is None
            assert snap.ttft_available is True
            assert SERIES_TOTAL_TOKENS in snap.raw
            assert snap.raw[SERIES_TOTAL_TOKENS] == TOTAL_TOKENS_PER_REQUEST

    asyncio.run(go())


def test_snapshot_ttft_absent_when_streaming_off():
    """AC-11 precondition: a non-streaming mock yields ttft_available=False."""

    async def go():
        async with _client(create_app(streaming=False)) as client:
            src = LiteLLMSource("http://test/metrics", client=client)
            snap = await src.snapshot()
            assert snap is not None
            assert snap.ttft_available is False
            # TTFT series should not appear in raw at all.
            assert SERIES_TTFT not in snap.raw

    asyncio.run(go())


def test_snapshot_unreachable_returns_none():
    """FR-14: a dead endpoint yields None so the orchestrator can flag
    wire-metrics-unavailable and continue."""

    def _fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated unreachable")

    async def go():
        client = httpx.AsyncClient(transport=httpx.MockTransport(_fail))
        src = LiteLLMSource("http://test/metrics", client=client)
        snap = await src.snapshot()
        await client.aclose()
        return snap

    assert asyncio.run(go()) is None


# ---------------------------------------------------------------------------
# diff()
# ---------------------------------------------------------------------------


def _delta_after_n_calls(n: int, *, streaming: bool = True, duration_s: float = 1.0):
    """Drive the mock through `n` requests between two scrapes and return the
    LevelDelta + the live end snapshot (for additional asserts)."""

    async def go():
        async with _client(create_app(streaming=streaming)) as client:
            src = LiteLLMSource("http://test/metrics", client=client)
            start = await src.snapshot()
            assert start is not None
            for _ in range(n):
                r = await client.post(
                    "/v1/chat/completions", json={"model": "m", "messages": []}
                )
                assert r.status_code == 200
            end = await src.snapshot()
            assert end is not None
            end.level = 4
            delta = src.diff(start, end, in_flight_peak=4.0, duration_s=duration_s)
            return delta, end

    return asyncio.run(go())


def test_diff_delta_math_exact_against_mock():
    """AC-4: Δtokens/Δrequests/throughput/error-rate match hand-computed values
    against the deterministic mock counters."""
    n = 5
    duration = 2.0
    delta, _end = _delta_after_n_calls(n, duration_s=duration)

    assert delta.level == 4
    assert delta.n_requests == n
    # Mock advances total tokens by TOTAL_TOKENS_PER_REQUEST per call.
    assert delta.throughput_tps == (n * TOTAL_TOKENS_PER_REQUEST) / duration
    assert delta.error_rate == 0.0  # all 200s
    assert delta.in_flight_peak == 4.0
    assert delta.suspect is False


def test_diff_bucket_edge_percentiles_populated():
    """Latency percentiles are bucket-edge values (mock observations fall in
    0.25s for e2e, 0.05s for ttft)."""
    delta, _end = _delta_after_n_calls(5)
    # Mock FAKE_E2E_LATENCY_S = 0.200 → falls in 0.25 bucket.
    assert delta.lat_p50 == 0.25
    assert delta.lat_p95 == 0.25
    assert delta.lat_p99 == 0.25
    # Mock FAKE_TTFT_S = 0.05 → falls in 0.05 bucket.
    assert delta.ttft_p50 == 0.05
    assert delta.ttft_p95 == 0.05


def test_diff_ttft_none_when_streaming_off():
    """AC-11: ttft_p50/p95 are None when the source never had a TTFT series."""
    delta, _end = _delta_after_n_calls(3, streaming=False)
    assert delta.ttft_p50 is None
    assert delta.ttft_p95 is None


def test_diff_error_rate_with_failures():
    """20% failure rate via the direct failed-request counter → error_rate=0.2
    and n_requests counting all status codes."""
    start = ScrapeSnapshot(
        source_name="litellm",
        level=None,
        t=0.0,
        raw={
            SERIES_REQUESTS: {"200": 0.0, "500": 0.0},
            SERIES_FAILED_REQUESTS: 0.0,
        },
        ttft_available=True,
    )
    end = ScrapeSnapshot(
        source_name="litellm",
        level=4,
        t=10.0,
        raw={
            SERIES_REQUESTS: {"200": 80.0, "500": 20.0},
            SERIES_FAILED_REQUESTS: 20.0,
        },
        ttft_available=True,
    )
    src = LiteLLMSource("http://test/metrics")
    delta = src.diff(start, end, in_flight_peak=4.0, duration_s=10.0)
    assert delta.n_requests == 100
    assert delta.error_rate == 0.2
    assert delta.suspect is False


def test_diff_error_rate_falls_back_to_status_code_when_failed_counter_absent():
    """When SERIES_FAILED_REQUESTS is absent (mock without it), diff() falls
    back to counting non-2xx from SERIES_REQUESTS."""
    start = ScrapeSnapshot(
        source_name="litellm",
        level=None,
        t=0.0,
        raw={SERIES_REQUESTS: {"200": 0.0, "500": 0.0}},
        ttft_available=True,
    )
    end = ScrapeSnapshot(
        source_name="litellm",
        level=4,
        t=10.0,
        raw={SERIES_REQUESTS: {"200": 80.0, "500": 20.0}},
        ttft_available=True,
    )
    src = LiteLLMSource("http://test/metrics")
    delta = src.diff(start, end, in_flight_peak=4.0, duration_s=10.0)
    assert delta.n_requests == 100
    assert delta.error_rate == 0.2


def test_diff_counter_reset_marks_suspect_and_clamps_to_zero():
    """FR-10 risk: counter decrease between start/end (LiteLLM restart) must
    flag the delta suspect and never produce negatives."""
    start = ScrapeSnapshot(
        source_name="litellm",
        level=None,
        t=0.0,
        raw={
            SERIES_TOTAL_TOKENS: 1000.0,
            "litellm_input_tokens_metric_total": 200.0,
            "litellm_output_tokens_metric_total": 800.0,
            SERIES_REQUESTS: {"200": 50.0},
        },
        ttft_available=True,
    )
    end = ScrapeSnapshot(
        source_name="litellm",
        level=4,
        t=10.0,
        raw={
            SERIES_TOTAL_TOKENS: 100.0,  # decreased — reset
            "litellm_input_tokens_metric_total": 20.0,
            "litellm_output_tokens_metric_total": 80.0,
            SERIES_REQUESTS: {"200": 5.0},
        },
        ttft_available=True,
    )
    src = LiteLLMSource("http://test/metrics")
    delta = src.diff(start, end, in_flight_peak=1.0, duration_s=10.0)
    assert delta.suspect is True
    # No negatives leak out.
    assert delta.throughput_tps >= 0.0
    assert delta.n_requests >= 0
