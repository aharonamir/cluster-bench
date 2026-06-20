# Feature Specification: ClusterBench

**Feature Branch**: `001-clusterbench`
**Status**: Draft (rev 3 — LiteLLM /metrics, no custom proxy)
**Input**: A decision-support benchmark that measures how an inference stack
(model fixed, vLLM behind a LiteLLM proxy) behaves as the number of concurrent
coding agents increases.

---

## 1. Problem & Goal

### Problem
A model is deployed on a GPU cluster served by **vLLM behind a LiteLLM proxy**.
ClusterBench can reach **only LiteLLM** — the vLLM pods are not reachable, even
for scraping. We need to know how the stack behaves as **many coding agents hit
it at once**: how TTFT and latency change with concurrency, where throughput
peaks, where it saturates (the "knee"), and how task success and agent failures
vary with load.

This is **not** a model/agent capability ranking. The model is **fixed**; the
SWE-bench Verified tasks are **realistic load generators**. The unit of interest
is the *stack under concurrent load*.

### Goal
Drive a real coding agent (mini-swe-agent) against a fixed slice of SWE-bench
Verified tasks at increasing concurrency, with all LLM calls flowing through
LiteLLM, and produce a live dashboard + saved report answering:
1. How do **TTFT, latency, tokens/sec** change as concurrent-agent count rises
   (headline graphs: agents-vs-TTFT, agents-vs-latency).
2. Where is the **throughput ceiling** and the **knee**.
3. **Task score** per level from SWE-bench Verified scoring.
4. **Agent reliability** per level: process outcomes (exit, timeout, crash) and
   captured stdout/stderr.

### Architecture in one line
**ClusterBench = concurrency-sweep orchestrator + LiteLLM /metrics collector +
live dashboard**, wrapping **mini-swe-agent = agent + Docker isolation +
SWE-bench Verified scoring**.

### Metric model (load-bearing decision)
Wire-level metrics come from **scraping LiteLLM's Prometheus `/metrics`** on a
timer and **differencing counters/histograms across each level** (scrape-and-
delta). There is **no per-request data** and **no custom proxy**. Percentiles are
read at LiteLLM histogram **bucket edges** — coarse but sufficient to locate a
saturation knee.

### Non-Goals
- Ranking models/agents.
- Building an agent scaffold, container manager, workspace isolation, or
  SWE-bench verification harness — **mini-swe-agent + the swebench scorer own
  these**.
- A custom measuring proxy — **dropped**; LiteLLM `/metrics` is the wire source.
- Per-request telemetry or true (non-bucket) percentiles.
- Reaching vLLM internals (KV-cache, scheduler queue) — **not available** in this
  topology; supported only as an optional future metric source (FR-OPT).
- OTel collectors / agent-internal spans.

---

## 2. Users & Usage

**Primary user**: an ML/infra engineer validating a deployment fronted by LiteLLM.

**Primary flows**:
- *Capacity curves*: "How do TTFT/latency rise from 1→32 concurrent agents?" →
  SWEEP; read agents-vs-TTFT / agents-vs-latency.
- *Ceiling & knee*: "Where does throughput peak / saturate?" → SWEEP; saturation
  curve + knee.
- *Stability*: "Does it hold at 16 for 30 min?" → SOAK.
- *Config comparison*: run twice (different serving/LiteLLM configs), diff/overlay
  the saved reports.

---

## 3. Functional Requirements

### 3.1 Agent execution via mini-swe-agent
- **FR-1**: Use **mini-swe-agent** as the agent engine; each task runs in its
  **docker environment** (isolated container per task). ClusterBench MUST NOT
  implement container/workspace/sandbox logic.
- **FR-2**: Run mini-swe-agent in **batch** over a fixed slice of **SWE-bench
  Verified** (`--subset verified --split test`), with `--workers` = the desired
  concurrent-agent count.
- **FR-3**: Configure mini-swe-agent's model/endpoint so all LLM calls go to the
  **LiteLLM proxy** (OpenAI-compatible base URL / litellm api_base), reachable
  from inside the task containers.
- **FR-4**: A **mock path** MUST exist for CI: a lightweight generator that
  produces concurrent traffic against a **mock LiteLLM** (serving a fake
  `/metrics`) without mini-swe-agent, Docker, GPU, or downloads.

### 3.2 Concurrency control (the measurement axis)
- **FR-5**: Concurrency level = concurrent agents = mini-swe-agent `--workers`.
  ClusterBench owns the sweep; mini-swe-agent has no sweep concept.
- **FR-6**: Two load modes:
  - **SWEEP** — run the **same fixed slice of N tasks** at each worker level
    (e.g. 1,4,8,16,32); concurrency is the only independent variable.
  - **SOAK** — hold one worker level for a duration, continuously feeding tasks.
- **FR-7**: Fixed slice size N is configurable; the **same pinned instance_ids**
  are used at every level of a sweep.
- **FR-8**: Sweep levels run **sequentially** (one fully completes before the
  next) so per-level metric deltas are clean.

### 3.3 Metrics — wire level (LiteLLM /metrics, scrape-and-delta)
- **FR-9**: The system MUST scrape LiteLLM's Prometheus `/metrics` endpoint
  (enabled via LiteLLM's `callbacks: ["prometheus"]`). Required series include:
  TTFT histogram (streaming only), the LLM-API-call latency histogram, the
  end-to-end request latency histogram, the LiteLLM **processing-overhead**
  latency, input/output/total token counters, request/total counters with
  status_code, and `litellm_in_flight_requests`.
- **FR-10**: Per-level stats MUST be computed by **differencing** counters/
  histograms between a clean scrape at level start and level end
  (scrape-and-delta). Throughput = Δtokens / level-duration; error rate =
  Δfailed / Δtotal requests. Percentiles MUST be read from the histogram bucket
  edges over the level's delta.
- **FR-11**: The scrape interval MUST be configurable and frequent enough to
  capture the within-level **peak of `litellm_in_flight_requests`**.
- **FR-12**: TTFT requires streaming; the system MUST configure mini-swe-agent to
  stream where possible, and MUST flag TTFT as unavailable if no streaming TTFT
  series is present (rather than reporting zero).
- **FR-13**: LiteLLM's self-reported **processing overhead** MUST be surfaced
  directly (no calibration step needed — this replaces the old proxy-overhead
  calibration).
- **FR-14**: If LiteLLM `/metrics` is unreachable or the prometheus callback is
  disabled, the run MUST continue, the report MUST flag wire metrics unavailable,
  and process-level + task-score metrics MUST still be collected.

### 3.4 Metrics — saturation cause (LiteLLM-side; vLLM optional later)
- **FR-15**: The saturation-cause signal MUST be **LiteLLM-side**:
  `litellm_in_flight_requests` (proxy queue depth) and the processing-overhead
  latency. The report MUST note this is a proxy-side signal, not GPU-side.
- **FR-16 (caveat, documented, two layers)**: LiteLLM measures latency from when
  its handler starts. Queueing splits into two layers that the report and
  dashboard MUST keep separate:
  - **LiteLLM-internal queue** — `litellm_request_queue_time_seconds` (time in
    LiteLLM's internal queue after handler entry, before the LLM call). When the
    source emits this series, the level's `queue_p50`/`queue_p95` MUST be
    populated from it directly. Real LiteLLM emits it; the mock does not.
  - **Pre-ASGI/event-loop wait** — still invisible to LiteLLM. When `queue_p50`/
    `queue_p95` are unavailable, the dashboard MUST fall back to the documented
    heuristic: low in-flight + high end-to-end latency ⇒ pre-handler queueing.
  The dashboard MUST NOT claim GPU-side causation for either layer.
- **FR-OPT (optional, pluggable)**: The metrics layer MUST be a **pluggable
  source** so a vLLM `/metrics` source (KV-cache, scheduler queue) can be added
  later **if** vLLM ever becomes reachable, without reworking the orchestrator or
  dashboard. Not active in this topology.

### 3.5 Metrics — process level & task score (from mini-swe-agent)
- **FR-17**: Per task attempt, capture from mini-swe-agent output: return status,
  wall-clock, step/time-limit hit (timeout), and bounded stdout/stderr (or
  trajectory log tail).
- **FR-18**: Obtain a **task score** per attempt by feeding mini-swe-agent's
  `preds.json` into **SWE-bench Verified scoring** (resolved/unresolved),
  aggregated to a per-level pass rate.
- **FR-19**: Classify each attempt into exactly one outcome bucket, priority:
  `timeout` → `agent_error` → `inference_error` (a LiteLLM-reported failed
  request in the level) → `unresolved` → `resolved`.
- **FR-20**: Outcome buckets aggregate per level (reliability-vs-load).

### 3.6 Live dashboard & reporting
- **FR-21**: Stream telemetry live over WebSocket: run/level start+done, scrape
  samples (the delta snapshots), per-task outcomes, knee, run done.
- **FR-22**: A mid-run client MUST receive history replay.
- **FR-23**: The dashboard MUST show, live and from saved reports:
  - **agents-vs-TTFT** and **agents-vs-latency** (headline),
  - saturation curve (throughput vs concurrency) with peak + knee,
  - **LiteLLM-side saturation panel**: in-flight peak + processing overhead vs
    concurrency, with **queue-time percentiles when the source emits
    `litellm_request_queue_time_seconds`** plus the pre-handler-queueing
    heuristic annotation for the layer LiteLLM can't see,
  - per-level table (TTFT/latency bucket-percentiles, tokens/sec, pass rate,
    error rate, outcome counts),
  - outcome-taxonomy panel + live event feed.
- **FR-24**: On completion, persist the full **RunReport** JSON keyed by run id,
  retrievable via API.
- **FR-25**: Support rendering graphs from a saved report later, including
  **overlaying multiple runs** on the agents-vs-TTFT/latency axes.

### 3.7 Degradation guards (SWEEP knee)
- **FR-26**: Configurable guards; any tripping marks the knee: max p99 latency
  (bucket-edge), max TTFT p95, max error rate, min pass rate, max in-flight peak.
- **FR-27**: Guards evaluated per completed level; first trip recorded as the knee
  with reason; sweep MAY continue to map the full curve.

### 3.8 Operation
- **FR-28**: Single Docker host. mini-swe-agent's batch spawns per-task
  containers; ClusterBench runs one batch per level.
- **FR-29**: Whole system runnable end-to-end on the mock path (mock agent + mock
  LiteLLM) with no GPU, Docker images, or downloads.

---

## 4. Key Entities

- **RunConfig** — litellm_metrics_url, litellm_base_url (request path), mode,
  worker levels or level+duration, task slice (subset/split/N/pinned ids),
  guards, scrape_interval, mini-swe-agent config (model string, streaming, step
  limit), optional vllm_metrics_url (FR-OPT, usually null).
- **MetricsSource** (pluggable) — `LiteLLMSource` (active) reads + diffs LiteLLM
  /metrics; `VLLMSource` (optional, later) reads vLLM /metrics. Same interface:
  `snapshot()` → raw series; orchestrator diffs across a level.
- **ScrapeSnapshot** — one timed read of a source: raw counter/histogram values +
  timestamp + level.
- **LevelDelta** — differenced stats for a level: ttft/latency bucket-percentiles,
  token throughput, error rate, in-flight peak, processing-overhead.
- **MiniSweRunner** — runs mini-swe-agent batch for one level; collects
  preds.json + per-instance process records.
- **TaskOutcome** — per attempt: instance_id, level, outcome, resolved, return
  status, wall time, timed_out, log tail.
- **LevelSummary** — LevelDelta + pass rate + outcome counts + duration.
- **RunReport** — config + level summaries + knee + wire-metrics-availability +
  ttft-availability + pinned ids + finished_at.

---

## 5. Acceptance Criteria

- **AC-1**: A SWEEP over [1,4,8,16] on the mock path yields agents-vs-TTFT and
  agents-vs-latency populated per level from scrape-and-delta, using the pinned
  slice at every level.
- **AC-2**: Throughput rises then falls; knee = first level a guard trips.
- **AC-3**: With LiteLLM `/metrics` blocked, the run completes with
  wire-metrics-unavailable flagged, and process + task-score metrics still
  present.
- **AC-4**: Per-level token throughput and error rate are computed as **deltas**
  between level-start and level-end scrapes (verified against a mock that
  advances its counters deterministically).
- **AC-5**: Each attempt resolves to exactly one outcome bucket (forced
  timeout/agent_error/inference_error/unresolved/resolved all reachable).
- **AC-6**: preds.json scored via SWE-bench Verified → per-level pass rate (real
  scoring behind a slow marker; mock stubs it).
- **AC-7**: Two runs persist distinct retrievable reports; dashboard loads a saved
  report and overlays two runs on the headline axes.
- **AC-8**: Mid-run WebSocket connect renders completed levels.
- **AC-9**: Full stack runs on the mock path with no GPU/Docker/downloads.
- **AC-10**: Same pinned instance_ids at every sweep level (verified in report).
- **AC-11**: TTFT availability is detected and flagged: with a non-streaming mock
  (no TTFT series) the report marks TTFT unavailable rather than reporting 0.
- **AC-OPT**: The metrics source is pluggable — a `VLLMSource` can be registered
  and selected by config without changing orchestrator/dashboard code (covered by
  an interface-conformance test, even though unused in this topology).

---

## 6. Review Checklist
- [ ] LiteLLM /metrics is the wire source; no custom proxy.
- [ ] Per-level stats = scrape-and-delta; percentiles at bucket edges.
- [ ] Saturation panel is LiteLLM-side (in-flight + overhead + queue-time
      percentiles when emitted) with the pre-handler-queueing caveat surfaced
      for the layer LiteLLM can't see.
- [ ] vLLM source is pluggable-but-inactive (FR-OPT / AC-OPT).
- [ ] mini-swe-agent owns agent + isolation + scoring; same pinned slice each
      level.
- [ ] TTFT-availability handled (streaming-only).
- [ ] Mock path keeps CI hardware-free.
