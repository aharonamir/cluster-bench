"""Fake LiteLLM proxy for the mock path (FR-4/FR-29).

Exposes:
  - POST /v1/chat/completions — fake OpenAI-compatible endpoint. Honors the
    `stream` flag (SSE when true, JSON when false). Streaming is also gated by
    the app-level `streaming` toggle: when `streaming=False`, the TTFT histogram
    is omitted from /metrics, exercising the TTFT-absent path (FR-12/AC-11).
  - GET /metrics — Prometheus text format. Counters advance deterministically
    by a fixed amount per served request, so scrape-and-delta tests are exact
    (AC-4). `litellm_in_flight_requests` rises with concurrency.

Run standalone with:  uvicorn mock_litellm:app --port 4000
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

# Deterministic per-request advance — fixed so delta tests are exact (AC-4).
INPUT_TOKENS_PER_REQUEST = 10
OUTPUT_TOKENS_PER_REQUEST = 40
TOTAL_TOKENS_PER_REQUEST = INPUT_TOKENS_PER_REQUEST + OUTPUT_TOKENS_PER_REQUEST

# Fixed per-request observations (histograms advance deterministically too).
FAKE_TTFT_S = 0.05
FAKE_LLM_API_LATENCY_S = 0.180
FAKE_PROC_OVERHEAD_S = 0.020
FAKE_E2E_LATENCY_S = FAKE_LLM_API_LATENCY_S + FAKE_PROC_OVERHEAD_S

# Bucket schema shared across all latency histograms. Cumulative counts per
# Prometheus convention. Phase 1 will pin exact series names from a fixture.
LATENCY_BUCKETS_S: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
PER_REQUEST_HOLD_S = 0.05  # gives concurrency tests room to observe in_flight > 1


class _State:
    """Mutable counters. Each /v1/chat/completions advances these by a fixed
    amount; each scrape reads them as-is. asyncio.Lock guards multi-field
    updates so concurrent requests can't tear the state mid-write.

    Saturation simulation (used by Orchestrator tests): when in_flight strictly
    exceeds saturation_threshold, each request adds saturation_extra_delay_s
    to its hold + latency observations, so histogram bucket-edge percentiles
    jump to the next bucket. Defaults to None (no saturation) so existing
    tests are unaffected.
    """

    def __init__(
        self,
        *,
        streaming: bool,
        saturation_threshold: int | None = None,
        saturation_extra_delay_s: float = 0.5,
    ) -> None:
        self.streaming = streaming
        self.saturation_threshold = saturation_threshold
        self.saturation_extra_delay_s = saturation_extra_delay_s
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.requests_200 = 0
        self.requests_500 = 0
        self.in_flight = 0
        self.ttft_obs: list[float] = []  # empty when streaming is off
        self.llm_api_lat_obs: list[float] = []
        self.proc_overhead_obs: list[float] = []
        self.e2e_lat_obs: list[float] = []
        self._lock = asyncio.Lock()

    async def begin_request(self) -> None:
        async with self._lock:
            self.in_flight += 1

    async def commit_request(self, *, ok: bool, extra_delay_s: float = 0.0) -> None:
        async with self._lock:
            self.in_flight -= 1
            self.input_tokens += INPUT_TOKENS_PER_REQUEST
            self.output_tokens += OUTPUT_TOKENS_PER_REQUEST
            self.total_tokens += TOTAL_TOKENS_PER_REQUEST
            if ok:
                self.requests_200 += 1
            else:
                self.requests_500 += 1
            # Extra delay inflates e2e + LLM-API latency observations so the
            # histogram bucket-edge percentiles move; overhead is LiteLLM's
            # own processing time and isn't affected by GPU/queueing.
            self.llm_api_lat_obs.append(FAKE_LLM_API_LATENCY_S + extra_delay_s)
            self.proc_overhead_obs.append(FAKE_PROC_OVERHEAD_S)
            self.e2e_lat_obs.append(FAKE_E2E_LATENCY_S + extra_delay_s)
            if self.streaming:
                self.ttft_obs.append(FAKE_TTFT_S)

    async def reset(self) -> None:
        async with self._lock:
            self.input_tokens = 0
            self.output_tokens = 0
            self.total_tokens = 0
            self.requests_200 = 0
            self.requests_500 = 0
            self.in_flight = 0
            self.ttft_obs.clear()
            self.llm_api_lat_obs.clear()
            self.proc_overhead_obs.clear()
            self.e2e_lat_obs.clear()


def create_app(
    *,
    streaming: bool = True,
    saturation_threshold: int | None = None,
    saturation_extra_delay_s: float = 0.5,
) -> FastAPI:
    """Build a fresh mock app with isolated state. Tests should call this so
    they don't share counters; `uvicorn mock_litellm:app` uses the module-level
    `app` below for manual runs.

    `saturation_threshold` (when set) simulates a serving stack that degrades
    past a concurrency knee: requests while in_flight > threshold observe
    extra delay, so latency histograms shift buckets. Used to exercise the
    Orchestrator's knee detection against a non-trivial curve.
    """
    state = _State(
        streaming=streaming,
        saturation_threshold=saturation_threshold,
        saturation_extra_delay_s=saturation_extra_delay_s,
    )
    app = FastAPI(title="mock-litellm", docs_url=None, redoc_url=None)
    app.state.litellm = state

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        body = await request.json()
        want_stream = bool(body.get("stream", False)) and state.streaming
        await state.begin_request()
        # Saturation check after begin so in_flight reflects this request too.
        # Read without lock — int reads are atomic under CPython's GIL and the
        # approximate value is fine for triggering the simulated knee.
        extra = 0.0
        if (
            state.saturation_threshold is not None
            and state.in_flight > state.saturation_threshold
        ):
            extra = state.saturation_extra_delay_s
        try:
            await asyncio.sleep(PER_REQUEST_HOLD_S + extra)
            await state.commit_request(ok=True, extra_delay_s=extra)
        except Exception:
            await state.commit_request(ok=False)
            raise

        if want_stream:
            return StreamingResponse(
                _sse_stream(body), media_type="text/event-stream"
            )
        return JSONResponse(_non_stream_response(body))

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(
            _render_prometheus(state), media_type="text/plain; version=0.0.4"
        )

    @app.get("/__reset")
    async def reset() -> dict[str, bool]:
        await state.reset()
        return {"ok": True}

    @app.get("/__state")
    async def dump_state() -> dict[str, Any]:
        # Test-only: read the live counters without scraping /metrics.
        return {
            "streaming": state.streaming,
            "input_tokens": state.input_tokens,
            "output_tokens": state.output_tokens,
            "total_tokens": state.total_tokens,
            "requests_200": state.requests_200,
            "requests_500": state.requests_500,
            "in_flight": state.in_flight,
        }

    return app


async def _sse_stream(body: dict[str, Any]) -> Any:
    """Fake OpenAI streaming response: a role chunk, a content chunk, then the
    [DONE] sentinel. Just enough shape for callers that expect SSE."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model = body.get("model", "mock-model")
    for chunk in (
        _sse_chunk(completion_id, model, delta={"role": "assistant"}),
        _sse_chunk(completion_id, model, delta={"content": "ok"}),
    ):
        yield f"data: {chunk}\n\n".encode()
        await asyncio.sleep(0)
    yield b"data: [DONE]\n\n"


def _sse_chunk(cid: str, model: str, *, delta: dict[str, str]) -> str:
    import json

    return json.dumps(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
    )


def _non_stream_response(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "mock-model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": INPUT_TOKENS_PER_REQUEST,
            "completion_tokens": OUTPUT_TOKENS_PER_REQUEST,
            "total_tokens": TOTAL_TOKENS_PER_REQUEST,
        },
    }


def _histogram_lines(
    name: str, observations: list[float], help_text: str
) -> list[str]:
    """Emit a Prometheus histogram with cumulative bucket counts (per
    Prometheus convention: `_bucket{le="X"}` is the count of observations
    with value <= X)."""
    per_bucket = [0] * len(LATENCY_BUCKETS_S)
    for obs in observations:
        for i, edge in enumerate(LATENCY_BUCKETS_S):
            if obs <= edge:
                per_bucket[i] += 1
                break
    # Roll up to cumulative; this is what Prometheus clients expect and what
    # the parser's percentile_at_bucket_edge assumes.
    cumulative: list[int] = []
    running = 0
    for c in per_bucket:
        running += c
        cumulative.append(running)
    total = len(observations)
    total_sum = sum(observations)
    lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
    for edge, count in zip(LATENCY_BUCKETS_S, cumulative):
        lines.append(f'{name}_bucket{{le="{edge}"}} {count}')
    lines.append(f'{name}_bucket{{le="+Inf"}} {total}')
    lines.append(f"{name}_count {total}")
    lines.append(f"{name}_sum {total_sum}")
    return lines


def _render_prometheus(state: _State) -> str:
    lines: list[str] = [
        "# HELP litellm_input_tokens_metric_total Input tokens served",
        "# TYPE litellm_input_tokens_metric_total counter",
        f"litellm_input_tokens_metric_total {state.input_tokens}",
        "# HELP litellm_output_tokens_metric_total Output tokens served",
        "# TYPE litellm_output_tokens_metric_total counter",
        f"litellm_output_tokens_metric_total {state.output_tokens}",
        "# HELP litellm_total_tokens_metric_total Total tokens served",
        "# TYPE litellm_total_tokens_metric_total counter",
        f"litellm_total_tokens_metric_total {state.total_tokens}",
        "# HELP litellm_proxy_total_requests_metric_total Client-side requests by status",
        "# TYPE litellm_proxy_total_requests_metric_total counter",
        f'litellm_proxy_total_requests_metric_total{{status_code="200"}} {state.requests_200}',
        f'litellm_proxy_total_requests_metric_total{{status_code="500"}} {state.requests_500}',
        "# HELP litellm_llm_api_failed_requests_metric_total LLM API failed requests",
        "# TYPE litellm_llm_api_failed_requests_metric_total counter",
        f"litellm_llm_api_failed_requests_metric_total {state.requests_500}",
        "# HELP litellm_in_flight_requests Requests currently in flight",
        "# TYPE litellm_in_flight_requests gauge",
        f"litellm_in_flight_requests {state.in_flight}",
    ]
    lines += _histogram_lines(
        "litellm_request_total_latency_metric",
        state.e2e_lat_obs,
        "End-to-end request latency",
    )
    lines += _histogram_lines(
        "litellm_llm_api_latency_metric",
        state.llm_api_lat_obs,
        "LLM API call latency",
    )
    lines += _histogram_lines(
        "litellm_overhead_latency_metric",
        state.proc_overhead_obs,
        "LiteLLM processing overhead",
    )
    if state.streaming:
        lines += _histogram_lines(
            "litellm_llm_api_time_to_first_token_metric",
            state.ttft_obs,
            "Time to first token (streaming only)",
        )
    return "\n".join(lines) + "\n"


# Module-level app for `uvicorn mock_litellm:app` convenience.
app = create_app()
