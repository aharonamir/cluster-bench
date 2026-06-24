"""LiteLLM MetricsSource — the only active wire source in rev 3 topology.

Scrapes LiteLLM's Prometheus `/metrics` and computes per-level deltas
(scrape-and-delta, FR-9/FR-10). All series parsing lives here per CLAUDE.md so
LiteLLM label/format drift is contained to this file.

Series names pinned from a real LiteLLM `/metrics` capture
(tests/data/lite-llm-metrics). Real LiteLLM emits most series multi-labeled
(api_key_alias, hashed_api_key, model, requested_model, route, status_code,
team, user, user_agent, ...) — we aggregate to system-wide totals per series
at parse time, since ClusterBench's measurement axis is concurrency, not
per-user traffic.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)

from clusterbench.metrics.base import percentile_at_bucket_edge
from clusterbench.models import LevelDelta, ScrapeSnapshot

# Series names pinned from a real LiteLLM `/metrics` capture. Drift here is the
# main integration risk; tests pin to both the sanitized real fixture and the
# mock.
SERIES_INPUT_TOKENS = "litellm_input_tokens_metric_total"
SERIES_OUTPUT_TOKENS = "litellm_output_tokens_metric_total"
SERIES_TOTAL_TOKENS = "litellm_total_tokens_metric_total"
# Client-side request count, labeled with status_code (among others).
SERIES_REQUESTS = "litellm_proxy_total_requests_metric_total"
# Direct failed-request counter from the LLM API side. Better signal than
# inferring failures from non-2xx in SERIES_REQUESTS.
SERIES_FAILED_REQUESTS = "litellm_llm_api_failed_requests_metric_total"
SERIES_IN_FLIGHT = "litellm_in_flight_requests"
SERIES_LATENCY_E2E = "litellm_request_total_latency_metric"
# Health-check-free E2E latency (same Prometheus series, health-check rows excluded
# at parse time). Only streaming / real requests remain, so avg_lat is comparable
# to avg_ttft and TPOT is meaningful. Stored as a separate key in raw.
SERIES_LATENCY_E2E_REAL = SERIES_LATENCY_E2E + ":real"
SERIES_LATENCY_LLM_API = "litellm_llm_api_latency_metric"
SERIES_PROC_OVERHEAD = "litellm_overhead_latency_metric"
SERIES_TTFT = "litellm_llm_api_time_to_first_token_metric"  # streaming-only
# Pre-handler queue time — directly measured (replaces FR-16's heuristic when
# present).
SERIES_QUEUE_TIME = "litellm_request_queue_time_seconds"
# KV-cache misses (counter). In real LiteLLM this tracks the streaming /
# first-token count per model — every uncached prompt that reaches generation.
SERIES_CACHE_MISSES = "litellm_cache_misses_metric_total"
# Health-check-free processing overhead (same Prometheus series, health-check
# rows excluded at parse time). Stored as a separate key so the live gauge and
# the LiteLLM overhead bars reflect bench traffic, not thousands of fast
# health-check probes that would otherwise dominate the histogram.
SERIES_PROC_OVERHEAD_REAL = SERIES_PROC_OVERHEAD + ":real"
# Process-level health gauges (instantaneous; captured at end of each level).
SERIES_PROC_RSS = "process_resident_memory_bytes"
SERIES_PROC_OPEN_FDS = "process_open_fds"
SERIES_PROC_MAX_FDS = "process_max_fds"
# Python GC counter, labeled by generation. gen0 is too noisy; gen1/gen2
# deltas reveal Python object-churn pressure under load.
SERIES_GC_COLLECTIONS = "python_gc_collections_total"

_LINE_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+'
    r'([+-]?[\d.]+(?:[eE][+-]?\d+)?|nan|\+Inf|-Inf)\s*$'
)
_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

# LiteLLM health-check alias — these requests dominate the latency histogram
# (thousands of fast sub-100ms calls) and contaminate avg_lat when mixed with
# real streaming requests, making TPOT appear as 0.
_HEALTH_CHECK_ALIAS = "litellm-internal-health-check"


@dataclass(frozen=True)
class Sample:
    """One parsed Prometheus line."""

    name: str
    labels: tuple[tuple[str, str], ...]  # sorted (key, value) pairs
    value: float


def parse_prometheus_text(text: str) -> list[Sample]:
    """Parse Prometheus text format into a flat list of Samples.

    Aggregation (summing across label combinations, building histograms) is
    done by the helpers below — this function only tokenizes lines.
    """
    samples: list[Sample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        name, label_str, value_str = m.groups()
        labels = tuple(sorted(_LABEL_RE.findall(label_str))) if label_str else ()
        samples.append(Sample(name=name, labels=labels, value=_parse_value(value_str)))
    return samples


def _parse_value(s: str) -> float:
    if s == "+Inf":
        return float("inf")
    if s == "-Inf":
        return float("-inf")
    if s == "nan":
        return float("nan")
    return float(s)


def sum_unlabeled(samples: list[Sample], name: str) -> float:
    """Sum values of all samples with this `name` regardless of labels.

    Used for counters where we want a system-wide total (tokens, failed
    requests, in-flight gauge)."""
    return sum(s.value for s in samples if s.name == name)


def sum_by_label(
    samples: list[Sample],
    name: str,
    label_name: str,
    *,
    exclude_label: str | None = None,
    exclude_value: str | None = None,
) -> dict[str, float]:
    """Sum values of samples with this `name`, grouped by `label_name`'s value.

    Used for the request counter where we want a per-status-code breakdown
    while ignoring the other (api_key_alias, model, route, ...) labels.

    When `exclude_label`/`exclude_value` are given, samples with that label
    pair are skipped (used to drop health-check rows so they don't inflate
    n_requests and deflate error_rate).
    """
    out: dict[str, float] = {}
    for s in samples:
        if s.name != name:
            continue
        if exclude_label is not None and exclude_value is not None:
            if any(k == exclude_label and v == exclude_value for k, v in s.labels):
                continue
        for k, v in s.labels:
            if k == label_name:
                out[v] = out.get(v, 0.0) + s.value
                break
    return out


def aggregate_histogram(
    samples: list[Sample],
    name: str,
    *,
    exclude_label: str | None = None,
    exclude_value: str | None = None,
) -> dict[str, Any]:
    """Build a histogram dict summed across all label combinations except `le`.

    Output shape (matches what `diff()` consumes):
        {"buckets": {edge: count}, "count": int, "sum": float}

    When `exclude_label` and `exclude_value` are given, samples whose labels
    contain that key=value pair are skipped (used to drop health-check rows).
    """
    buckets: dict[float, float] = {}
    count = 0.0
    total_sum = 0.0
    for s in samples:
        if s.name not in (f"{name}_bucket", f"{name}_count", f"{name}_sum"):
            continue
        if exclude_label is not None and exclude_value is not None:
            if any(k == exclude_label and v == exclude_value for k, v in s.labels):
                continue
        if s.name == f"{name}_bucket":
            edge = _le_to_edge(s)
            if edge is not None:
                buckets[edge] = buckets.get(edge, 0.0) + s.value
        elif s.name == f"{name}_count":
            count += s.value
        elif s.name == f"{name}_sum":
            total_sum += s.value
    return {
        "buckets": buckets,
        "count": int(count),
        "sum": float(total_sum),
    }


def _le_to_edge(sample: Sample) -> float | None:
    for k, v in sample.labels:
        if k == "le":
            return float("inf") if v == "+Inf" else float(v)
    return None


class LiteLLMSource:
    """Active MetricsSource for rev 3: scrapes LiteLLM /metrics."""

    name = "litellm"

    def __init__(
        self,
        metrics_url: str,
        scrape_interval_s: float = 1.0,
        client: httpx.AsyncClient | None = None,
        ssl_verify: bool = True,
        scrape_timeout_s: float = 15.0,
    ) -> None:
        self.metrics_url = metrics_url
        self.scrape_interval_s = scrape_interval_s
        # /metrics must stay reachable under load: at high concurrency the
        # LiteLLM proxy is busy serving streams and a 5s scrape can time out,
        # which drops the whole level's wire metrics (delta=None). 15s gives
        # the proxy room to answer even when saturated.
        self._client = client or httpx.AsyncClient(timeout=scrape_timeout_s, verify=ssl_verify)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def snapshot(self) -> ScrapeSnapshot | None:
        """One read of /metrics. Returns None on any network/HTTP error so the
        orchestrator can flag wire-metrics-unavailable and continue (FR-14)."""
        try:
            resp = await self._client.get(self.metrics_url)
            resp.raise_for_status()
        except (httpx.HTTPError, OSError, RuntimeError) as exc:
            # RuntimeError covers httpx's "client has been closed" — can happen
            # when the in-flight poller fires after teardown. Treat like any
            # other scrape failure: None → wire-metrics-unavailable, continue
            # (FR-14).
            log.warning("metrics scrape failed url=%s err=%s", self.metrics_url, exc)
            return None
        samples = parse_prometheus_text(resp.text)
        raw = self._aggregate(samples)
        return ScrapeSnapshot(
            source_name=self.name,
            level=None,
            t=time.time(),
            raw=raw,
            ttft_available=SERIES_TTFT in raw,
        )

    @staticmethod
    def _aggregate(samples: list[Sample]) -> dict[str, Any]:
        """Aggregate parsed samples into the `raw` shape `diff()` consumes."""
        raw: dict[str, Any] = {
            SERIES_INPUT_TOKENS: sum_unlabeled(samples, SERIES_INPUT_TOKENS),
            SERIES_OUTPUT_TOKENS: sum_unlabeled(samples, SERIES_OUTPUT_TOKENS),
            SERIES_TOTAL_TOKENS: sum_unlabeled(samples, SERIES_TOTAL_TOKENS),
            SERIES_REQUESTS: sum_by_label(
                samples, SERIES_REQUESTS, "status_code",
                exclude_label="api_key_alias",
                exclude_value=_HEALTH_CHECK_ALIAS,
            ),
            SERIES_FAILED_REQUESTS: sum_unlabeled(samples, SERIES_FAILED_REQUESTS),
            SERIES_IN_FLIGHT: sum_unlabeled(samples, SERIES_IN_FLIGHT),
            SERIES_CACHE_MISSES: sum_unlabeled(samples, SERIES_CACHE_MISSES),
            SERIES_PROC_RSS: sum_unlabeled(samples, SERIES_PROC_RSS),
            SERIES_PROC_OPEN_FDS: sum_unlabeled(samples, SERIES_PROC_OPEN_FDS),
            SERIES_PROC_MAX_FDS: sum_unlabeled(samples, SERIES_PROC_MAX_FDS),
            SERIES_GC_COLLECTIONS: sum_by_label(samples, SERIES_GC_COLLECTIONS, "generation"),
            SERIES_LATENCY_E2E: aggregate_histogram(samples, SERIES_LATENCY_E2E),
            SERIES_LATENCY_LLM_API: aggregate_histogram(samples, SERIES_LATENCY_LLM_API),
            SERIES_PROC_OVERHEAD: aggregate_histogram(samples, SERIES_PROC_OVERHEAD),
        }
        # Health-check-free E2E latency: exclude rows that are health-check probes.
        # These fast requests (sub-100ms) would otherwise contaminate avg_lat and
        # produce avg_lat < avg_ttft → TPOT = 0.
        lat_real = aggregate_histogram(
            samples, SERIES_LATENCY_E2E,
            exclude_label="api_key_alias", exclude_value=_HEALTH_CHECK_ALIAS,
        )
        if lat_real["count"] or lat_real["buckets"]:
            raw[SERIES_LATENCY_E2E_REAL] = lat_real

        # Same health-check exclusion for processing overhead so the live gauge
        # and the LiteLLM overhead bars reflect bench traffic, not probes.
        overhead_real = aggregate_histogram(
            samples, SERIES_PROC_OVERHEAD,
            exclude_label="api_key_alias", exclude_value=_HEALTH_CHECK_ALIAS,
        )
        if overhead_real["count"] or overhead_real["buckets"]:
            raw[SERIES_PROC_OVERHEAD_REAL] = overhead_real

        # Optional series — include only when present (so ttft_available and
        # queue detection work via `in raw`).
        ttft_hist = aggregate_histogram(samples, SERIES_TTFT)
        if ttft_hist["buckets"] or ttft_hist["count"]:
            raw[SERIES_TTFT] = ttft_hist
        queue_hist = aggregate_histogram(samples, SERIES_QUEUE_TIME)
        if queue_hist["buckets"] or queue_hist["count"]:
            raw[SERIES_QUEUE_TIME] = queue_hist
        return raw

    def diff(
        self,
        start: ScrapeSnapshot,
        end: ScrapeSnapshot,
        in_flight_peak: float,
        duration_s: float,
    ) -> LevelDelta:
        """Compute the per-level LevelDelta from two scrapes (FR-10).

        Counter decreases (LiteLLM restart mid-level) mark the delta suspect
        instead of producing negatives — the orchestrator surfaces this flag.
        """
        level = end.level if end.level is not None else 0

        in_tokens = _counter_delta(start, end, SERIES_INPUT_TOKENS)
        out_tokens = _counter_delta(start, end, SERIES_OUTPUT_TOKENS)
        total_tokens = _counter_delta(start, end, SERIES_TOTAL_TOKENS)

        requests = _labeled_counter_delta(start, end, SERIES_REQUESTS)
        n_requests = max(0, int(round(sum(requests.values()))))
        n_failed = max(0, int(round(_counter_delta(start, end, SERIES_FAILED_REQUESTS))))
        # Backstop: if the failed-counter series is absent (mock without it),
        # fall back to non-2xx in the request counter delta.
        if n_failed == 0:
            n_failed = max(0, int(round(_count_failed_from_status(requests))))

        throughput_tps = (
            max(0.0, total_tokens) / duration_s if duration_s > 0 else 0.0
        )
        error_rate = n_failed / n_requests if n_requests > 0 else 0.0

        lat_buckets = _histogram_delta(start, end, SERIES_LATENCY_E2E)
        lat_p50 = percentile_at_bucket_edge(lat_buckets, 0.50)
        lat_p95 = percentile_at_bucket_edge(lat_buckets, 0.95)
        lat_p99 = percentile_at_bucket_edge(lat_buckets, 0.99)

        # Proc overhead p50, bench-only (health-check probes excluded) when the
        # filtered series is present; falls back to the full series otherwise
        # (e.g. the mock, which emits no health checks). FR-13.
        proc_key = (
            SERIES_PROC_OVERHEAD_REAL
            if SERIES_PROC_OVERHEAD_REAL in end.raw
            else SERIES_PROC_OVERHEAD
        )
        proc_buckets = _histogram_delta(start, end, proc_key)
        proc_overhead_s = percentile_at_bucket_edge(proc_buckets, 0.50)

        ttft_p50: float | None = None
        ttft_p95: float | None = None
        # Require TTFT only in the end snapshot. The start scrape often predates
        # the first streaming request (so SERIES_TTFT is absent there);
        # _histogram_delta treats a missing series as a zero baseline, which is
        # correct since no streaming before the start scrape means nothing to
        # subtract. end.ttft_available=False (streaming off) still → None.
        tpot_ms: float | None = None
        if end.ttft_available:
            ttft_buckets = _histogram_delta(start, end, SERIES_TTFT)
            if ttft_buckets.get(float("inf"), 0) > 0:
                ttft_p50 = percentile_at_bucket_edge(ttft_buckets, 0.50)
                ttft_p95 = percentile_at_bucket_edge(ttft_buckets, 0.95)

                _lat_key = (
                    SERIES_LATENCY_E2E_REAL
                    if SERIES_LATENCY_E2E_REAL in start.raw and SERIES_LATENCY_E2E_REAL in end.raw
                    else SERIES_LATENCY_E2E
                )
                lat_count = (end.raw.get(_lat_key, {}).get("count", 0)
                             - start.raw.get(_lat_key, {}).get("count", 0))
                lat_sum = (end.raw.get(_lat_key, {}).get("sum", 0.0)
                           - start.raw.get(_lat_key, {}).get("sum", 0.0))
                ttft_count = (end.raw.get(SERIES_TTFT, {}).get("count", 0)
                              - start.raw.get(SERIES_TTFT, {}).get("count", 0))
                ttft_sum = (end.raw.get(SERIES_TTFT, {}).get("sum", 0.0)
                            - start.raw.get(SERIES_TTFT, {}).get("sum", 0.0))
                if lat_count > 0 and ttft_count > 0 and out_tokens > 0:
                    avg_lat = lat_sum / lat_count
                    avg_ttft = ttft_sum / ttft_count
                    avg_out_tok = out_tokens / ttft_count
                    gen_s = max(0.0, avg_lat - avg_ttft)
                    if avg_out_tok > 0:
                        tpot_ms = round(gen_s / avg_out_tok * 1000.0, 1)

        cache_misses = int(max(0.0, _counter_delta(start, end, SERIES_CACHE_MISSES)))

        # FR-16: queue time directly measured when the series is present in both
        # scrapes (real LiteLLM); None otherwise (mock).
        queue_p50: float | None = None
        queue_p95: float | None = None
        if SERIES_QUEUE_TIME in start.raw and SERIES_QUEUE_TIME in end.raw:
            queue_buckets = _histogram_delta(start, end, SERIES_QUEUE_TIME)
            if queue_buckets.get(float("inf"), 0) > 0:
                queue_p50 = percentile_at_bucket_edge(queue_buckets, 0.50)
                queue_p95 = percentile_at_bucket_edge(queue_buckets, 0.95)

        # Process-health gauges — take end-of-level snapshot (not delta).
        rss_bytes = end.raw.get(SERIES_PROC_RSS, 0.0)
        rss_mb: float | None = round(rss_bytes / (1024 * 1024), 1) if rss_bytes > 0 else None
        open_fds_v = end.raw.get(SERIES_PROC_OPEN_FDS, 0.0)
        open_fds: int | None = int(open_fds_v) if open_fds_v > 0 else None
        max_fds_v = end.raw.get(SERIES_PROC_MAX_FDS, 0.0)
        max_fds: int | None = int(max_fds_v) if max_fds_v > 0 else None

        # GC gen1/gen2 deltas — skipping gen0 (too frequent to be meaningful).
        gc_gen1: int | None = None
        gc_gen2: int | None = None
        start_gc = start.raw.get(SERIES_GC_COLLECTIONS, {})
        end_gc = end.raw.get(SERIES_GC_COLLECTIONS, {})
        if end_gc:
            _gen1 = int(max(0, end_gc.get("1", 0) - start_gc.get("1", 0)))
            _gen2 = int(max(0, end_gc.get("2", 0) - start_gc.get("2", 0)))
            gc_gen1 = _gen1 if _gen1 > 0 else None
            gc_gen2 = _gen2 if _gen2 > 0 else None

        suspect = any(
            v < 0
            for v in (
                in_tokens,
                out_tokens,
                total_tokens,
                sum(requests.values()),
            )
        )

        return LevelDelta(
            level=level,
            ttft_p50=ttft_p50,
            ttft_p95=ttft_p95,
            lat_p50=lat_p50,
            lat_p95=lat_p95,
            lat_p99=lat_p99,
            throughput_tps=throughput_tps,
            n_requests=n_requests,
            error_rate=error_rate,
            in_flight_peak=in_flight_peak,
            proc_overhead_s=proc_overhead_s,
            suspect=suspect,
            queue_p50=queue_p50,
            queue_p95=queue_p95,
            tpot_ms=tpot_ms,
            cache_misses=cache_misses if cache_misses > 0 else None,
            rss_mb=rss_mb,
            open_fds=open_fds,
            max_fds=max_fds,
            gc_gen1=gc_gen1,
            gc_gen2=gc_gen2,
        )


def _counter_delta(
    start: ScrapeSnapshot, end: ScrapeSnapshot, name: str
) -> float:
    s = start.raw.get(name, 0.0)
    e = end.raw.get(name, 0.0)
    return float(e) - float(s)


def _labeled_counter_delta(
    start: ScrapeSnapshot, end: ScrapeSnapshot, name: str
) -> dict[str, float]:
    s = start.raw.get(name, {})
    e = end.raw.get(name, {})
    if not isinstance(s, dict):
        s = {}
    if not isinstance(e, dict):
        e = {}
    keys = set(s) | set(e)
    return {k: float(e.get(k, 0.0)) - float(s.get(k, 0.0)) for k in keys}


def _histogram_delta(
    start: ScrapeSnapshot, end: ScrapeSnapshot, name: str
) -> dict[float, int]:
    s = (
        start.raw.get(name, {}).get("buckets", {})
        if isinstance(start.raw.get(name), dict)
        else {}
    )
    e = (
        end.raw.get(name, {}).get("buckets", {})
        if isinstance(end.raw.get(name), dict)
        else {}
    )
    edges = set(s) | set(e)
    return {edge: int(e.get(edge, 0)) - int(s.get(edge, 0)) for edge in edges}


def compute_live_stats(
    start: "ScrapeSnapshot",
    current: "ScrapeSnapshot",
    elapsed_s: float,
    level: int,
) -> dict[str, Any]:
    """Lightweight live-stats snapshot for the polling loop.

    Returns a dict suitable for a `level_live` WebSocket event. All fields
    are present; optional ones (ttft_p50, tpot_ms) are None when the TTFT
    series is unavailable (streaming off) or insufficient data.

    TPOT = (avg_latency − avg_TTFT) / avg_output_tokens: rises when the GPU
    is saturated (batch throughput degrades). TTFT rises when queueing before
    generation starts. Together they pinpoint where the saturation is.
    """
    from clusterbench.metrics.base import percentile_at_bucket_edge

    in_flight = float(current.raw.get(SERIES_IN_FLIGHT, 0) or 0)

    total_tok_delta = _counter_delta(start, current, SERIES_TOTAL_TOKENS)
    throughput_tps = max(0.0, total_tok_delta) / elapsed_s if elapsed_s > 0 else 0.0

    lat_buckets = _histogram_delta(start, current, SERIES_LATENCY_E2E)
    lat_p50 = percentile_at_bucket_edge(lat_buckets, 0.50)

    ttft_p50: float | None = None
    tpot_ms: float | None = None

    if SERIES_TTFT in current.raw:
        ttft_buckets = _histogram_delta(start, current, SERIES_TTFT)
        # Only compute TTFT/TPOT when at least one new streaming request completed.
        # +Inf delta = 0 means no new observations; returning 0.0 from the percentile
        # function would display as "0" in the table, which is misleading.
        # Note: we don't require TTFT in start.raw — the level's start scrape
        # often predates the first streaming request, so SERIES_TTFT is absent
        # there. _histogram_delta treats a missing series as an empty (zero)
        # baseline, which is correct: no streaming before the start scrape means
        # nothing to subtract.
        if ttft_buckets.get(float("inf"), 0) > 0:
            ttft_p50 = percentile_at_bucket_edge(ttft_buckets, 0.50)

            # TPOT from histogram sums.
            # Use the health-check-free latency series so avg_lat isn't dragged
            # below avg_ttft by thousands of fast health-check probes.
            # Fall back to the full E2E series only when the filtered series is absent.
            _lat_key = (
                SERIES_LATENCY_E2E_REAL
                if SERIES_LATENCY_E2E_REAL in start.raw and SERIES_LATENCY_E2E_REAL in current.raw
                else SERIES_LATENCY_E2E
            )
            lat_s = start.raw.get(_lat_key, {})
            lat_e = current.raw.get(_lat_key, {})
            ttft_s = start.raw.get(SERIES_TTFT, {})
            ttft_e = current.raw.get(SERIES_TTFT, {})
            lat_count = (lat_e.get("count", 0) - lat_s.get("count", 0))
            lat_sum = (lat_e.get("sum", 0.0) - lat_s.get("sum", 0.0))
            ttft_count = (ttft_e.get("count", 0) - ttft_s.get("count", 0))
            ttft_sum = (ttft_e.get("sum", 0.0) - ttft_s.get("sum", 0.0))
            out_tok = _counter_delta(start, current, SERIES_OUTPUT_TOKENS)

            if lat_count > 0 and ttft_count > 0 and out_tok > 0:
                avg_lat = lat_sum / lat_count
                avg_ttft = ttft_sum / ttft_count
                # Output tokens per streaming request (TTFT count = streaming count).
                avg_out_tok = out_tok / ttft_count
                gen_s = max(0.0, avg_lat - avg_ttft)
                if avg_out_tok > 0:
                    tpot_ms = gen_s / avg_out_tok * 1000.0

    # ---- Proxy-side gauges (independent of the TTFT/TPOT block above) ----
    # Processing overhead, bench-only (health-checks excluded). Falls back to
    # the full series when the filtered one is absent (mock has no probes).
    overhead_p50: float | None = None
    oh_key = (
        SERIES_PROC_OVERHEAD_REAL
        if SERIES_PROC_OVERHEAD_REAL in current.raw
        else SERIES_PROC_OVERHEAD
    )
    oh_buckets = _histogram_delta(start, current, oh_key)
    if oh_buckets.get(float("inf"), 0) > 0:
        overhead_p50 = percentile_at_bucket_edge(oh_buckets, 0.50)

    # Pre-handler queue time — directly measured; rises as concurrency grows.
    # Absent in the mock (→ None).
    queue_p50: float | None = None
    if SERIES_QUEUE_TIME in current.raw:
        q_buckets = _histogram_delta(start, current, SERIES_QUEUE_TIME)
        if q_buckets.get(float("inf"), 0) > 0:
            queue_p50 = percentile_at_bucket_edge(q_buckets, 0.50)

    # KV-cache misses this window (counter delta). For real LiteLLM this tracks
    # the streaming/first-token count.
    cache_misses = int(max(0.0, _counter_delta(start, current, SERIES_CACHE_MISSES)))

    return {
        "level": level,
        "elapsed_s": round(elapsed_s, 1),
        "in_flight": in_flight,
        "throughput_tps": round(throughput_tps, 1),
        "lat_p50": lat_p50,
        "ttft_p50": ttft_p50,
        "tpot_ms": round(tpot_ms, 1) if tpot_ms is not None else None,
        "overhead_p50": overhead_p50,
        "queue_p50": queue_p50,
        "cache_misses": cache_misses,
    }


def _count_failed_from_status(requests_delta: dict[str, float]) -> float:
    """Backstop: count any non-2xx status code as failed. Used when the direct
    failed-request counter is absent (e.g. mock without SERIES_FAILED_REQUESTS)."""
    return sum(
        count
        for status, count in requests_delta.items()
        if not str(status).startswith("2")
    )
