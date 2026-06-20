# Implementation Plan: ClusterBench

**Branch**: `001-clusterbench` | **Spec**: `./spec.md` | rev 3 (LiteLLM /metrics)

## Technical Context

| Aspect | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | matches mini-swe-agent; async orchestration |
| Agent engine | **mini-swe-agent** | owns agent loop + per-task Docker isolation + SWE-bench batch/preds.json |
| Task set + scoring | **SWE-bench Verified** via mini-swe-agent + `swebench` scorer | official per-instance images; preds.json → resolved |
| Concurrency | mini-swe-agent `--workers` per level; ClusterBench sweeps | worker count = concurrent-agent count |
| Wire metrics | **scrape LiteLLM `/metrics`** + scrape-and-delta per level | only reachable layer; no per-request data, no custom proxy |
| Saturation cause | LiteLLM `litellm_in_flight_requests` + processing-overhead | vLLM internals unreachable in this topology |
| Metrics abstraction | **pluggable MetricsSource** (LiteLLM now, vLLM later) | future-proof if vLLM becomes scrapeable (FR-OPT) |
| Server/API | FastAPI + Uvicorn + WebSocket | live telemetry + REST + saved reports |
| Dashboard | static HTML + vanilla JS + SVG | headline graphs + multi-run overlay |
| Persistence | JSON report per run | diffable, supports later/overlay rendering |
| Packaging | Python package + Dockerfile (Docker socket for mini-swe-agent) | single-box |

**NOT built**: agent scaffold, container/workspace manager, SWE-bench verifier,
custom measuring proxy, OTel collector.

## What changed from rev 2
- The **custom measuring proxy is removed.** Wire metrics now come from scraping
  **LiteLLM `/metrics`** and differencing across each level.
- Metrics collection is **scrape-and-delta**, not per-request. Percentiles are
  **bucket-edge** values from LiteLLM histograms.
- The metrics layer is a **pluggable MetricsSource**; vLLM is an optional source
  for later (unreachable now).
- **Proxy-overhead calibration is gone** — LiteLLM reports its processing
  overhead directly.
- New concern: **TTFT is streaming-only** in LiteLLM; availability is detected
  and flagged.

## Architecture

```
                         ┌─────────────────────────────────────┐
   browser ◄──WebSocket──┤            FastAPI server            │
                         │  /api/run /api/runs /api/runs/{id}   │
                         │  /ws  /  static                      │
                         └───────────────┬─────────────────────┘
                                         │ drives
                                         ▼
                          ┌──────────────────────────┐
                          │       Orchestrator        │ SWEEP/SOAK, one level
                          │  for level in levels:     │ at a time
                          │   scrape source @start    │
                          │   run MiniSweRunner(W)     │
                          │   poll in-flight peak      │
                          │   scrape source @end       │
                          │   delta → LevelDelta       │
                          │   score preds.json          │
                          │   aggregate + guard/knee    │
                          └───┬───────────┬────────┬───┘
                  per level    │           │        │
          ┌──────────────────┘           │        └────────────┐
          ▼                              ▼                      ▼
 ┌────────────────────┐       ┌────────────────────┐  ┌──────────────────┐
 │   MiniSweRunner     │       │  SWE-bench scorer  │  │  MetricsSource    │
 │  mini-extra swebench│       │  preds.json →      │  │  (pluggable)      │
 │  --workers W        │       │  resolved set      │  │  ┌──────────────┐ │
 │  --instances slice  │       └────────────────────┘  │  │ LiteLLMSource│ │ active
 │  model→LiteLLM url  │                                │  │ scrape+delta │ │
 └─────────┬───────────┘                                │  └──────────────┘ │
           │ task containers' LLM calls                 │  ┌──────────────┐ │
           ▼                                            │  │ VLLMSource   │ │ FR-OPT
   ┌──────────────────────┐   scrape /metrics ─────────▶│  │ (later/null) │ │ inactive
   │   LiteLLM proxy      │◀──────────────────────────  │  └──────────────┘ │
   │ (OpenAI-compatible)  │                             └──────────────────┘
   └──────────┬───────────┘
              │ routes to (UNREACHABLE to us)
              ▼
       vLLM instances (GPU)   ◀── cannot scrape in this topology
```

## How a level runs (scrape-and-delta)
For each level W (sequential, FR-8):
1. **Scrape@start** — read LiteLLM `/metrics`, store raw counters/histograms as
   the level baseline.
2. **Run batch** — `MiniSweRunner(W)` shells out to
   `mini-extra swebench --subset verified --split test --workers W
    --instances <pinned slice> --model <model> -o <out/level_W>`, model base URL
   → LiteLLM. While it runs, poll `litellm_in_flight_requests` on the scrape
   interval to capture the **in-flight peak** (FR-11).
3. **Scrape@end** — read `/metrics` again.
4. **Delta** — difference end−start: Δtokens, Δrequests, Δfailed; throughput =
   Δtokens / duration; error rate = Δfailed/Δtotal; TTFT/latency percentiles from
   histogram bucket edges over the delta. Build `LevelDelta`.
5. **Score** — parse preds.json + per-instance logs; score via SWE-bench Verified;
   build process records + outcomes.
6. **Aggregate** `LevelSummary`, eval guards, emit `level_done`.

SOAK: hold one W for a duration, re-feeding tasks; periodic scrape-and-delta into
time bins.

## MetricsSource interface (pluggable, FR-OPT)
```
class MetricsSource(Protocol):
    name: str
    async def snapshot(self) -> ScrapeSnapshot | None   # None if unreachable
    def diff(self, start: ScrapeSnapshot, end: ScrapeSnapshot,
             in_flight_peak: float, duration_s: float) -> LevelDelta
```
- `LiteLLMSource` — parses LiteLLM series (FR-9), implements bucket-edge
  percentiles + counter deltas; exposes processing-overhead and in-flight.
- `VLLMSource` — later; same interface; would add KV-cache/scheduler-queue. Ships
  as a stub that conforms to the interface so AC-OPT's conformance test passes,
  but is not selected in this topology.
The orchestrator only knows the interface, so adding vLLM later changes no
orchestrator/dashboard code.

## TTFT availability (FR-12/AC-11)
LiteLLM emits TTFT only for streaming requests. `LiteLLMSource` checks for the
TTFT histogram series; if absent (non-streaming), it sets `ttft_available=False`
and the report/dashboard show "TTFT unavailable (enable streaming)" instead of 0.
mini-swe-agent should be configured to stream.

## Queue time availability (FR-16)
Real LiteLLM emits `litellm_request_queue_time_seconds` — the LiteLLM-internal
queue (time spent after handler entry, before the LLM call). `LiteLLMSource`
surfaces this as `queue_p50`/`queue_p95` on `LevelDelta` when the series is
present in both start and end scrapes; otherwise they are `None` (mock path).
This is distinct from the **pre-ASGI/event-loop** layer LiteLLM cannot see —
the dashboard's heuristic for that layer remains in force regardless of
queue-time availability. The two layers MUST be reported separately.

## Outcome taxonomy (FR-19) — priority
timeout → agent_error → inference_error → unresolved → resolved.
Note: without per-request data, `inference_error` is detected from LiteLLM's
failed-request counter delta within the level (level-granularity), and from
mini-swe-agent's own error surfacing per instance where available. Documented.

## Module layout
```
clusterbench/
  models.py          RunConfig, ScrapeSnapshot, LevelDelta, TaskOutcome,
                     LevelSummary, RunReport; enums LoadMode{sweep,soak},
                     Outcome{...}; DegradationGuard
  metrics/
    base.py          MetricsSource protocol; ScrapeSnapshot; bucket-edge helpers
    litellm.py       LiteLLMSource: scrape + parse + delta + in-flight + overhead
    vllm.py          VLLMSource: interface-conforming stub for later (FR-OPT)
  miniswerunner.py   MiniSweRunner: build+run batch per level; parse out dir,
                     preds.json, per-instance logs; MockRunner for CI
  scoring.py         score_predictions(preds, instances) → resolved set; stub mode
  orchestrator.py    SWEEP/SOAK loop, scrape@start/end, in-flight polling, delta,
                     score, aggregate, guard/knee, emit
  web/
    server.py        FastAPI: /api/run, /api/runs, /api/runs/{id}, /ws, /, static
    index.html       dashboard
    static/app.js    WS client; SVG graphs incl. agents-vs-TTFT/latency + overlay;
                     LiteLLM saturation panel w/ queue-time + pre-handler-queueing
                     annotation
mock_litellm.py      fake OpenAI endpoint + fake LiteLLM /metrics whose counters
                     advance with load and in-flight rises with concurrency
mock_minisweagent.py mock runner: concurrent traffic + fake preds.json
run_server.py        uvicorn entrypoint
Dockerfile           image; Docker socket mount for mini-swe-agent
requirements.txt     fastapi, uvicorn, httpx, aiohttp, mini-swe-agent, swebench
```

## Data shapes (authoritative)
**ScrapeSnapshot**: source_name, level, t, raw{series→value}, ttft_available.

**LevelDelta**: level, ttft_p50/p95 (bucket-edge, or None), lat_p50/p95/p99
(bucket-edge), throughput_tps, n_requests, error_rate, in_flight_peak,
proc_overhead_s, queue_p50/p95 (bucket-edge, or None when the source has no
`litellm_request_queue_time_seconds` — real LiteLLM emits it; the mock does
not).

**TaskOutcome**: run_id, level, instance_id, outcome, resolved, return_status,
wall_time_s, timed_out, log_tail.

**LevelSummary**: LevelDelta + n_tasks + pass_rate + outcome_counts + duration_s.

**RunReport**: run_id, name, config, wire_metrics_available, ttft_available,
levels[LevelSummary], knee{level,reason}|null, pinned_instance_ids, finished_at.

## API surface
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/run` | start (409 if active) |
| GET | `/api/runs` | list saved reports |
| GET | `/api/runs/{id}` | full report (later/overlay graphs) |
| WS | `/ws` | live events + replay |
| GET | `/` | dashboard |

## Event protocol
`run_start` · `level_start` · `scrape` (delta snapshot) · `task` · `level_done` ·
`knee` · `run_done`. Bounded ring buffer (~5000) for replay.

## Risks & mitigations
- **LiteLLM metric/label drift** → isolate parsing in `litellm.py`; pin LiteLLM
  series names from a captured fixture; degrade gracefully if a series is missing.
- **Bucket-edge percentiles are coarse** → accepted; sufficient for knee. Report
  the bucket boundaries so readers know the granularity.
- **Counter resets (LiteLLM restart mid-run)** → detect counter decrease between
  start/end; if so, mark the level's delta suspect rather than emitting negatives.
- **Multi-worker LiteLLM** → `_total` are summed across workers (livesum); doc the
  `PROMETHEUS_MULTIPROC_DIR` requirement so deltas aren't per-worker-partial.
- **TTFT missing** → streaming-only; detect + flag (FR-12).
- **Pre-handler queueing invisible to LiteLLM** → two layers (FR-16):
  `litellm_request_queue_time_seconds` (LiteLLM-internal queue) is measured
  directly when present (real LiteLLM); pre-ASGI/event-loop wait remains the
  documented heuristic. Don't claim GPU-side causation for either layer.
- **mini-swe-agent CLI drift** → isolate in `miniswerunner.py`; pin version.

## Constitution check
- Single-box, one run at a time: satisfied.
- No GPU/Docker/downloads for tests: satisfied (mock LiteLLM + mock runner + stub
  scorer).
- Don't reimplement agent/isolation/verification: satisfied.
- Metrics source pluggable for future vLLM access: satisfied (FR-OPT/AC-OPT).
