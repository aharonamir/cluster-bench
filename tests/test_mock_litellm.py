"""Gate 0 (part 2): mock_litellm serves OpenAI /v1/chat/completions and a
parseable Prometheus /metrics whose token/request counters advance across two
scrapes; in_flight rises with concurrency; TTFT series is absent when
streaming is off (AC-11 precondition for Phase 1)."""
import asyncio
import re

import httpx

from mock_litellm import (
    TOTAL_TOKENS_PER_REQUEST,
    create_app,
)


def _scrape_value(text: str, pattern: str) -> float:
    m = re.search(pattern, text, re.MULTILINE)
    assert m, f"pattern not found in /metrics: {pattern!r}"
    return float(m.group(1))


def _client(app=None):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app or create_app()),
        base_url="http://test",
    )


def test_openai_endpoint_non_stream():
    async def go():
        async with _client(create_app()) as client:
            r = await client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["object"] == "chat.completion"
            assert body["usage"]["total_tokens"] == TOTAL_TOKENS_PER_REQUEST

    asyncio.run(go())


def test_openai_endpoint_stream():
    async def go():
        async with _client(create_app(streaming=True)) as client:
            r = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]
            assert "data: " in r.text
            assert "[DONE]" in r.text

    asyncio.run(go())


def test_metrics_parseable_and_counters_increase_across_two_scrapes():
    """The Gate 0 acceptance: counters must advance between two scrapes after
    we serve requests in between. Deterministic — exact deltas."""

    async def go():
        async with _client(create_app()) as client:
            m1 = (await client.get("/metrics")).text
            t1 = _scrape_value(m1, r"^litellm_total_tokens_metric_total (\d+)")
            i1 = _scrape_value(m1, r"^litellm_input_tokens_metric_total (\d+)")
            o1 = _scrape_value(m1, r"^litellm_output_tokens_metric_total (\d+)")
            r1 = _scrape_value(
                m1, r'^litellm_proxy_total_requests_metric_total\{status_code="200"\} (\d+)'
            )
            assert t1 == 0.0
            assert r1 == 0.0

            n_calls = 3
            for _ in range(n_calls):
                resp = await client.post(
                    "/v1/chat/completions", json={"model": "m", "messages": []}
                )
                assert resp.status_code == 200

            m2 = (await client.get("/metrics")).text
            t2 = _scrape_value(m2, r"^litellm_total_tokens_metric_total (\d+)")
            i2 = _scrape_value(m2, r"^litellm_input_tokens_metric_total (\d+)")
            o2 = _scrape_value(m2, r"^litellm_output_tokens_metric_total (\d+)")
            r2 = _scrape_value(
                m2, r'^litellm_proxy_total_requests_metric_total\{status_code="200"\} (\d+)'
            )

            assert t2 - t1 == n_calls * TOTAL_TOKENS_PER_REQUEST
            assert i2 - i1 == n_calls * 10
            assert o2 - o1 == n_calls * 40
            assert r2 - r1 == n_calls

    asyncio.run(go())


def test_in_flight_rises_with_concurrency():
    """litellm_in_flight_requests must rise above 1 when multiple requests run
    concurrently (FR-11 / saturation panel precondition)."""

    async def go():
        async with _client(create_app()) as client:
            tasks = [
                asyncio.create_task(
                    client.post(
                        "/v1/chat/completions", json={"model": "m", "messages": []}
                    )
                )
                for _ in range(4)
            ]
            await asyncio.sleep(0.02)  # let them all enter the handler
            m = (await client.get("/metrics")).text
            in_flight = _scrape_value(m, r"^litellm_in_flight_requests (\d+)")
            await asyncio.gather(*tasks)
            return in_flight

    in_flight = asyncio.run(go())
    assert in_flight >= 2, f"in_flight did not rise with concurrency: {in_flight}"


def test_ttft_absent_when_streaming_off():
    """AC-11 precondition: with streaming=False, the mock omits the TTFT
    histogram series so Phase 1's ttft_available detection can fire."""

    async def go():
        async with _client(create_app(streaming=False)) as client:
            await client.post(
                "/v1/chat/completions", json={"model": "m", "messages": []}
            )
            m = (await client.get("/metrics")).text
            assert "litellm_llm_api_time_to_first_token_metric" not in m

    asyncio.run(go())


def test_ttft_present_when_streaming_on():
    async def go():
        async with _client(create_app(streaming=True)) as client:
            await client.post(
                "/v1/chat/completions", json={"model": "m", "messages": []}
            )
            m = (await client.get("/metrics")).text
            assert "litellm_llm_api_time_to_first_token_metric_bucket" in m

    asyncio.run(go())
