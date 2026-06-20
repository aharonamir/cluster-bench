# ClusterBench — Spec Kit artifacts (rev 3: LiteLLM /metrics)

Hand these to Claude Code. Read order: `spec.md` (what) → `plan.md` (how) →
`tasks.md` (phased, gated work).

## Architecture in one line
**ClusterBench = concurrency-sweep orchestrator + LiteLLM /metrics collector +
live dashboard**, wrapping **mini-swe-agent = agent + Docker isolation +
SWE-bench Verified scoring**.

## Topology (what's reachable)
Agents → **LiteLLM proxy** → vLLM (GPU). ClusterBench can reach **only LiteLLM**.
vLLM pods are unreachable even for scraping.

## Locked decisions (do not relitigate)
- **mini-swe-agent is the agent engine** — owns the agent loop, per-task Docker
  isolation, the SWE-bench batch + preds.json. We don't build a spawner,
  workspace isolation, or a verifier.
- **Concurrency axis = mini-swe-agent `--workers`**; ClusterBench sweeps it,
  reusing the **same fixed task slice** at every level.
- **Two modes**: SWEEP (curve) + SOAK (stability).
- **Wire metrics = scrape LiteLLM `/metrics`, scrape-and-delta per level.** No
  custom proxy. No per-request data. Percentiles read at LiteLLM histogram
  **bucket edges** — coarse but fine for locating the knee.
- **Saturation panel is LiteLLM-side**: `litellm_in_flight_requests` +
  processing-overhead. vLLM internals (KV-cache, scheduler queue) are
  **unavailable** in this topology. The metrics source is **pluggable** so a vLLM
  source can be added later if it ever becomes reachable.
- **LiteLLM caveat surfaced**: it measures latency from when its handler starts,
  so pre-ASGI/event-loop queueing is invisible to it. The dashboard shows the
  documented heuristic (low in-flight + high latency ⇒ pre-handler queueing).
  *(rev 3 addendum)* Real LiteLLM additionally emits
  `litellm_request_queue_time_seconds` — a LiteLLM-internal queue layer that IS
  visible and is measured directly (`LevelDelta.queue_p50/p95`). This does not
  replace the pre-ASGI caveat above; the two layers are reported separately.
- **TTFT is streaming-only** in LiteLLM — detected and flagged, never faked as 0.
- **No proxy-overhead calibration** — LiteLLM reports its overhead directly.
- **Headline graphs**: agents-vs-TTFT, agents-vs-latency; live + from saved
  reports; multi-run overlay for config comparison.
- **Mock path** (mock LiteLLM + mock runner + stub scorer) keeps the whole system
  CI-runnable with no GPU, Docker images, or downloads.

## Evolution of the design (so you know why)
- rev 1: our own AgentSpawner + workspace isolation + Docker verification.
- rev 2: replaced all that with **mini-swe-agent** (it does isolation + scoring).
- rev 3 (this): replaced the **custom measuring proxy** with **scraping LiteLLM
  `/metrics`**, because only LiteLLM is reachable. Metrics became scrape-and-delta
  with bucket-edge percentiles; vLLM kept as a pluggable future source.

## Requirements traceability
| Your requirement | Where |
|---|---|
| Spawn isolated agents easily | mini-swe-agent docker env; FR-1; Phase 2 |
| TTFT / latency / tokens | LiteLLM /metrics scrape-and-delta; FR-9/FR-10; Phase 1 (Gate 1) |
| Errors/failures (exit, wall, timeout, stdout/stderr) | mini-swe-agent output; FR-17/FR-19; Phase 2–3 (Gate 3, AC-5) |
| Task score | preds.json → SWE-bench Verified; FR-18; Phase 2 (Gate 2, AC-6) |
| Real-time dashboard | WebSocket + SVG; FR-21/FR-23; Phase 5 |
| Present graphs later (agents-vs-TTFT/latency) | saved reports + overlay; FR-24/FR-25; Phase 5 (Gate 5) |

## Suggested Claude Code flow
Work `tasks.md` top to bottom; run each gate; never advance on a red gate; write
tests with the code; keep the cross-cutting rules green. Isolate all LiteLLM
series parsing in `metrics/litellm.py` (pin series names from a fixture) and all
mini-swe-agent CLI calls in `miniswerunner.py` (pin its version) — these two are
the brittle integration surfaces.
