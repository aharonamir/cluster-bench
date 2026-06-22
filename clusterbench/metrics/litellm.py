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
SERIES_LATENCY_LLM_API = "litellm_llm_api_latency_metric"
SERIES_PROC_OVERHEAD = "litellm_overhead_latency_metric"
SERIES_TTFT = "litellm_llm_api_time_to_first_token_metric"  # streaming-only
# Pre-handler queue time — directly measured (replaces FR-16's heuristic when
# present).
SERIES_QUEUE_TIME = "litellm_request_queue_time_seconds"

_LINE_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+'
    r'([+-]?[\d.]+(?:[eE][+-]?\d+)?|nan|\+Inf|-Inf)\s*$'
)
_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


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
    samples: list[Sample], name: str, label_name: str
) -> dict[str, float]:
    """Sum values of samples with this `name`, grouped by `label_name`'s value.

    Used for the request counter where we want a per-status-code breakdown
    while ignoring the other (api_key_alias, model, route, ...) labels.
    """
    out: dict[str, float] = {}
    for s in samples:
        if s.name != name:
            continue
        for k, v in s.labels:
            if k == label_name:
                out[v] = out.get(v, 0.0) + s.value
                break
    return out


def aggregate_histogram(samples: list[Sample], name: str) -> dict[str, Any]:
    """Build a histogram dict summed across all label combinations except `le`.

    Output shape (matches what `diff()` consumes):
        {"buckets": {edge: count}, "count": int, "sum": float}
    """
    buckets: dict[float, float] = {}
    count = 0.0
    total_sum = 0.0
    for s in samples:
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
    ) -> None:
        self.metrics_url = metrics_url
        self.scrape_interval_s = scrape_interval_s
        self._client = client or httpx.AsyncClient(timeout=5.0, verify=ssl_verify)
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
            SERIES_REQUESTS: sum_by_label(samples, SERIES_REQUESTS, "status_code"),
            SERIES_FAILED_REQUESTS: sum_unlabeled(samples, SERIES_FAILED_REQUESTS),
            SERIES_IN_FLIGHT: sum_unlabeled(samples, SERIES_IN_FLIGHT),
            SERIES_LATENCY_E2E: aggregate_histogram(samples, SERIES_LATENCY_E2E),
            SERIES_LATENCY_LLM_API: aggregate_histogram(samples, SERIES_LATENCY_LLM_API),
            SERIES_PROC_OVERHEAD: aggregate_histogram(samples, SERIES_PROC_OVERHEAD),
        }
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

        # Proc overhead is a histogram; report its p50 as the level's
        # representative overhead (LiteLLM reports it directly — FR-13).
        proc_buckets = _histogram_delta(start, end, SERIES_PROC_OVERHEAD)
        proc_overhead_s = percentile_at_bucket_edge(proc_buckets, 0.50)

        ttft_p50: float | None = None
        ttft_p95: float | None = None
        if start.ttft_available and end.ttft_available:
            ttft_buckets = _histogram_delta(start, end, SERIES_TTFT)
            ttft_p50 = percentile_at_bucket_edge(ttft_buckets, 0.50)
            ttft_p95 = percentile_at_bucket_edge(ttft_buckets, 0.95)

        # FR-16: queue time directly measured when the series is present in both
        # scrapes (real LiteLLM); None otherwise (mock).
        queue_p50: float | None = None
        queue_p95: float | None = None
        if SERIES_QUEUE_TIME in start.raw and SERIES_QUEUE_TIME in end.raw:
            queue_buckets = _histogram_delta(start, end, SERIES_QUEUE_TIME)
            queue_p50 = percentile_at_bucket_edge(queue_buckets, 0.50)
            queue_p95 = percentile_at_bucket_edge(queue_buckets, 0.95)

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


def _count_failed_from_status(requests_delta: dict[str, float]) -> float:
    """Backstop: count any non-2xx status code as failed. Used when the direct
    failed-request counter is absent (e.g. mock without SERIES_FAILED_REQUESTS)."""
    return sum(
        count
        for status, count in requests_delta.items()
        if not str(status).startswith("2")
    )
