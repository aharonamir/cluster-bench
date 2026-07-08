# ClusterBench
<img width="1836" height="822" alt="image" src="https://github.com/user-attachments/assets/12bc62bd-e8c7-4429-baf2-b5c2ce02999a" />

**A concurrency-sweep harness for an LLM inference stack.** ClusterBench drives
a real coding agent ([mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent))
against a fixed slice of SWE-bench Verified tasks at *increasing concurrency*,
routes every LLM call through a **LiteLLM proxy**, scrapes LiteLLM's
`/metrics`, and produces a live dashboard + saved report answering:

1. How do **TTFT, latency, tokens/sec** change as the concurrent-agent count
   rises (headline graphs: *agents-vs-TTFT*, *agents-vs-latency*)?
2. Where is the **throughput ceiling** and the **knee** (saturation point)?
3. What's the **task score** per level (SWE-bench Verified resolved/unresolved)?
4. How does **agent reliability** vary with load (timeouts, crashes, inference
   errors)?

This is **not** a model or agent capability ranking. The model is *fixed*; the
SWE-bench tasks are realistic *load generators*. The unit of interest is the
**stack under concurrent load**.

```
                         ┌─────────────────────────────────────┐
   browser ◄──WebSocket──┤            FastAPI server            │
                         │  /api/run  /api/runs  /ws  /         │
                         └───────────────┬─────────────────────┘
                                         │ drives
                                         ▼
                          ┌──────────────────────────┐
                          │       Orchestrator        │  one level at a time
                          │  scrape@start → run(W) →   │  (SWEEP or SOAK)
                          │  poll in-flight peak →     │
                          │  scrape@end → delta →      │
                          │  score → guards → knee     │
                          └───┬───────────┬────────┬───┘
              ┌───────────────┘           │        └───────────────┐
              ▼                           ▼                        ▼
   ┌────────────────────┐     ┌────────────────────┐   ┌──────────────────┐
   │   MiniSweRunner     │     │  SWE-bench scorer  │   │   LiteLLMSource   │
   │ mini-extra swebench │     │ preds.json →       │   │  scrape /metrics  │
   │ --workers W         │     │ resolved set       │   │  + scrape-delta   │
   └─────────┬───────────┘     └────────────────────┘   └────────┬─────────┘
             │ agents' LLM calls                                  │ scrape
             ▼                                                    ▼
   ┌──────────────────────┐   the only reachable wire layer ──► ┌──────────┐
   │   LiteLLM proxy      │◄────────────────────────────────────┤ /metrics │
   └──────────┬───────────┘                                     └──────────┘
              │ routes to (UNREACHABLE to ClusterBench)
              ▼
       vLLM instances (GPU)   ◄── cannot scrape in this topology
```

---

## The metric model (load-bearing decision)

Wire-level metrics come from **scraping LiteLLM's Prometheus `/metrics` on a
timer and differencing counters/histograms across each level** ("scrape-and-
delta"). There is **no per-request data** and **no custom proxy**.

Consequences you must keep in mind when reading the numbers:

- **Percentiles are bucket-edge values.** A reported `p99 = 1.0s` means "the 99th
  percentile falls in the histogram bucket whose upper edge is 1.0s", not a
  precise interpolated value. This is coarse *by design* — it's sufficient to
  locate a saturation knee, which is all we need. Don't read false precision into
  it.
- **Counters are differenced per level.** A LiteLLM restart mid-run decreases a
  counter; ClusterBench flags such a delta `suspect` and clamps to zero rather
  than reporting negative throughput.
- **Throughput = Δtokens / level-duration**, error rate = Δfailed / Δtotal — all
  computed from the start/end scrape pair.

### Saturation cause is LiteLLM-side, never GPU-side

The saturation panel is built from **`litellm_in_flight_requests`** (proxy queue
depth) and LiteLLM's **self-reported processing overhead**. vLLM internals
(KV-cache occupancy, scheduler queue) are **unreachable** in this topology, so
the report **never claims GPU-side causation**.

LiteLLM measures latency *from when its handler starts*, which splits queueing
into two layers the dashboard keeps separate:

- **LiteLLM-internal queue** — `litellm_request_queue_time_seconds`. When the
  source emits it (real LiteLLM does; the mock does not), the level's
  `queue_p50`/`queue_p95` are populated directly.
- **Pre-ASGI / event-loop wait** — *invisible* to LiteLLM. When queue-time is
  unavailable, the dashboard falls back to a documented heuristic: **low
  in-flight + high end-to-end latency ⇒ pre-handler queueing**. This is a
  heuristic annotation, not a measurement.

The metrics layer is a **pluggable `MetricsSource`** so a `VLLMSource` (KV-cache,
scheduler queue) can be added later *if* vLLM ever becomes reachable, without
reworking the orchestrator or dashboard.

### TTFT is streaming-only

Time-to-first-token only exists in LiteLLM's metrics when requests stream.
ClusterBench configures mini-swe-agent to stream, **detects** whether a TTFT
series is present, and **flags TTFT as unavailable** when it isn't — never faking
it as zero. The dashboard renders this as an explicit "TTFT unavailable" state.

---

## Two paths: mock and real

| | **mock path** (default) | **real path** (`--real`) |
|---|---|---|
| Agent | in-process `MockRunner` drives `mock_litellm` | `mini-extra swebench` subprocess |
| LiteLLM | `mock_litellm.py` (fake `/metrics`) | a real LiteLLM proxy |
| Scoring | stub pass-rate | SWE-bench Verified (`swebench`) |
| Needs | nothing — pure Python | Docker daemon + `--extra real` + dataset |
| Use for | CI, the dev loop, demos | actual cluster measurement |

The **whole system is runnable end-to-end on the mock path with no GPU, Docker
images, or downloads** — that's what the test suite and the self-test exercise.

---

## Quickstart (mock path)

This repo is a [uv](https://docs.astral.sh/uv/) project; `pyproject.toml` is the
source of truth and `uv.lock` is committed.

```bash
# 1. Install the mock-path deps.
uv sync

# 2. In one terminal: the fake LiteLLM proxy (serves /v1/chat/completions + /metrics).
uv run uvicorn mock_litellm:app --port 4000

# 3. In another: the ClusterBench server (mock path is the default).
uv run python run_server.py --port 8000

# 4. Open the dashboard.
open http://localhost:8000/
```

Start a sweep from the dashboard's **controls** panel, or via the API:

```bash
curl -X POST http://localhost:8000/api/run \
  -H 'content-type: application/json' \
  -d '{
        "name": "demo",
        "mode": "sweep",
        "levels": [1, 2, 4, 8],
        "task_slice": {"n": 5},
        "scrape_interval_s": 0.5,
        "guards": {"max_p99_latency_s": 2.0}
      }'
```

The response is `202` with a server-generated `run_id`. Watch it stream live on
the dashboard; on completion the full **RunReport** is persisted to
`results/<run_id>.json`.

> **TLS note for this host:** `uv` may not pick up the corporate CA. If `uv sync`
> can't reach PyPI, prefix it:
> `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt uv sync`.

---

## Running the real path

The real path drives mini-swe-agent's batch per level, pointing the agents'
OpenAI client at a **real LiteLLM**. It needs the heavy extra and a Docker
daemon (mini-swe-agent spawns one container per task — FR-28).

```bash
uv sync --extra real

uv run python run_server.py --real \
    --base-url    http://litellm:4000/v1 \
    --metrics-url http://litellm:4000/metrics \
    --api-key     "$LITELLM_KEY"
```

Or, preferably, put it all in a `config.yaml` (`cp config.example.yaml
config.yaml`) and launch with `uv run python run_server.py --config config.yaml`.
Precedence is **defaults < config.yaml < CLI flags**. `base_url`, `api_key`,
`metrics_url`, and `real` are server-level; `model` (and `streaming`,
`scrape_interval_s`, `step_limit`) are defaults a `POST /api/run` body may
override per run. See `config.example.yaml` for every key.

### Configuring LiteLLM so `/metrics` exists

ClusterBench's only wire source is LiteLLM's Prometheus endpoint. Enable it in
your LiteLLM config:

```yaml
litellm_settings:
  callbacks: ["prometheus"]
```

- LiteLLM then serves `/metrics` in Prometheus text format. ClusterBench pins the
  series it reads (tokens, requests-by-status, in-flight, e2e/llm-api/overhead
  latency histograms, streaming TTFT, and `litellm_request_queue_time_seconds`)
  in `clusterbench/metrics/litellm.py` — the single place to touch if LiteLLM's
  series names drift.
- **Multi-worker LiteLLM** (gunicorn/uvicorn with `--workers N`) needs
  `PROMETHEUS_MULTIPROC_DIR` set to a shared writable dir so the per-worker
  counters are aggregated into one `/metrics` response. Without it you'll scrape
  a single worker's partial view and the deltas will undercount.
- **Streaming must be on** for TTFT to appear. If a deployment can't stream,
  ClusterBench detects the missing TTFT series and flags it unavailable rather
  than reporting zero.

If `/metrics` is unreachable or the callback is disabled, **the run still
completes** — the report flags wire metrics unavailable, and process-level +
task-score metrics are still collected (FR-14).

### Sizing & concurrency guidance

- **SWE-bench Verified** task containers are sizeable; budget disk for the images
  and CPU for the per-task builds/tests. mini-swe-agent owns this isolation —
  ClusterBench just sets `--workers`.
- A sweep **level = mini-swe-agent `--workers` = concurrent-agent count**. The
  *same fixed task slice* is reused at every level (so the only variable is
  concurrency). Keep your top level at or below what the host's CPU can actually
  run in parallel — beyond that you're measuring host contention, not the
  inference stack.
- Set `scrape_interval_s` **frequent enough to catch the within-level peak** of
  `litellm_in_flight_requests` (FR-11). Too coarse and you'll miss the peak; the
  default is 1s.

---

## API

| Method & path | Purpose |
|---|---|
| `POST /api/run` | Start a sweep/soak. `202` + `run_id`; **`409` if a run is already active** (one run at a time). |
| `GET /api/runs` | List saved reports (newest first). |
| `GET /api/runs/{run_id}` | Fetch one persisted RunReport. |
| `GET /` | The dashboard. |
| `WS /ws` | Live telemetry stream; a mid-run connect **replays history**. |
| `GET /api/health` | Liveness + active-run id + subscriber count. |

**`POST /api/run` body** (all fields optional except as noted):

| field | default | notes |
|---|---|---|
| `name` | `""` | label shown on the dashboard |
| `mode` | `"sweep"` | `"sweep"` (curve) or `"soak"` (hold one level) |
| `levels` | `[1,4,8,16]` | worker counts to visit; SOAK uses the first |
| `soak_duration_s` | `1800` | SOAK only |
| `task_slice` | `{n: 5}` | `{subset, split, n, pinned_instance_ids}` |
| `guards` | `{}` | any of `max_p99_latency_s`, `max_ttft_p95_s`, `max_error_rate`, `min_pass_rate`, `max_in_flight_peak` |
| `scrape_interval_s` | server default → `1.0` | metrics scrape cadence |
| `model` | server default → `gpt-4o-mini` | model name passed to LiteLLM |
| `streaming` | server default → `true` | needed for TTFT |
| `step_limit` | server default → `0` | per-task agent step cap (0 = unlimited) |

`model`, `streaming`, `scrape_interval_s`, and `step_limit` are **server-level
defaults**: when omitted from the body they're inherited from the server's
`config.yaml` (see [Running the real path](#running-the-real-path)); an explicit
value in the request always wins.

Any tripped **guard** marks the **knee** (first trip wins, with a reason); the
sweep continues so the full curve is still mapped.

---

## The dashboard
<img width="1850" height="890" alt="image" src="https://github.com/user-attachments/assets/12b26156-ea4b-4c7c-9fc7-c7e8e41a8cdd" />

- **Headline graphs** — agents-vs-TTFT and agents-vs-latency, live and from saved
  reports.
- **Saturation curve** — throughput vs concurrency with the peak + knee marked.
- **LiteLLM saturation panel** — in-flight peak + processing overhead vs
  concurrency, with queue-time percentiles when present and the pre-handler
  heuristic annotation when not.
- **Per-level table** — TTFT/latency **bucket-edge** percentiles (labeled as
  such), tokens/sec, pass rate, error rate, outcome counts.
- **Outcome taxonomy** — per attempt, exactly one bucket, priority
  `timeout → agent_error → inference_error → unresolved → resolved`.
- **Live event feed** — run/level start+done, scrape deltas, per-task outcomes,
  knee.

### Overlaying saved runs

Saved reports render later from JSON, and you can **overlay 2+ runs** on the
headline axes to compare configs. In the **saved reports** panel, select two or
more runs (ctrl/cmd-click) and **Load selected**; each run gets its own color and
a legend entry. This is how you compare, e.g., streaming vs non-streaming, or two
different models, on the same agents-vs-latency axis.

---

## Development

```bash
uv run pytest                 # full suite (mock path; no GPU/Docker/downloads)
uv run pytest -m "not slow"   # skip the real-path tests
```

The two brittle integration surfaces are isolated on purpose:

- **All LiteLLM series parsing** lives in `clusterbench/metrics/litellm.py` (pin
  series names from the fixture in `tests/data/`).
- **All mini-swe-agent CLI calls** live in `clusterbench/miniswerunner.py` (pin
  its version in `pyproject.toml`).

The dashboard is vanilla JS + SVG with no build step; its render is verified
headlessly with Playwright in `tests/test_web_dashboard.py`.

---

## Docker

```bash
# Mock path (CI / demo) — no GPU, Docker images, or downloads.
docker build -t clusterbench .
docker run --rm -p 8000:8000 clusterbench

# Real path — needs the host Docker socket (mini-swe-agent spawns task containers).
docker build --build-arg INSTALL_EXTRA=real -t clusterbench:real .
docker run --rm -p 8000:8000 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$PWD/results:/app/results" \
    clusterbench:real --real --base-url http://litellm:4000/v1
```

Mount a volume at `/app/results` to keep persisted reports across container
restarts.

---

## Project layout

```
clusterbench/
  orchestrator.py        SWEEP/SOAK loop, scrape-and-delta, guards, knee
  miniswerunner.py       mini-swe-agent CLI calls + MockRunner (the seam)
  scoring.py             preds.json → resolved set (SWE-bench Verified / stub)
  metrics/
    base.py              MetricsSource protocol + bucket-edge percentile helpers
    litellm.py           LiteLLM /metrics parsing + scrape-and-delta (pinned series)
  web/
    server.py            FastAPI app: /api/run, /api/runs, /ws, /
    hub.py               WebSocket fan-out + bounded history replay
    persistence.py       RunReport JSON save/load/list
    static/              dashboard (index.html, app.js, styles.css)
  config.py              ServerConfig — YAML launch config + CLI-override merge
mock_litellm.py          fake LiteLLM proxy (mock path)
mock_minisweagent.py     fake agent output (mock path)
run_server.py            uvicorn entrypoint (mock by default, --real opt-in)
config.example.yaml      launch-config template (copy to config.yaml)
```

## Further reading

- **[OPERATING.md](OPERATING.md)** — step-by-step operations runbook: full
  mock- and real-path flows, SOAK, the live event vocabulary, outcome taxonomy,
  Docker, and a troubleshooting table.
- **`.spec/`** — the full spec (`README.md` → `spec.md` → `plan.md` →
  `tasks.md`).
