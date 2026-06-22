# Operating ClusterBench

A step-by-step runbook for operating ClusterBench end to end: starting the
services, launching a concurrency sweep (or soak), reading the dashboard, and
retrieving saved reports. Every command here is copy-pasteable.

> **What ClusterBench does, in one line:** drives a real coding agent
> (mini-swe-agent) against a fixed slice of SWE-bench Verified tasks at
> *increasing concurrency*, routes all LLM calls through a LiteLLM proxy,
> scrapes LiteLLM `/metrics` per level, and produces a live dashboard + a saved
> JSON report.

---

## 0. Mental model (read this once)

```
        you ──POST /api/run──►  ClusterBench server  ──drives──►  Orchestrator
                                       │                              │
                                       │ persists                     │ for each level W in levels:
                                       ▼                              │   1. scrape /metrics  (start)
                              results/<run_id>.json                   │   2. run agent batch at W workers
                                       ▲                              │      (poll in-flight peak)
        you ──GET /api/runs/{id}───────┘                              │   3. scrape /metrics  (end)
        you ──WS /ws──► live events ◄────── WebSocketHub ◄────emit────│   4. delta = end − start
                                                                      │   5. score preds → outcomes
                                                                      │   6. eval guards → knee?
                                                                      ▼
                                                            LiteLLM proxy ──► vLLM (GPU, unreachable)
```

There are **two paths**:

| | mock path (default) | real path (`--real`) |
|---|---|---|
| Agent | in-process `MockRunner` | `mini-extra swebench` subprocess |
| LiteLLM | `mock_litellm.py` (fake) | a real LiteLLM proxy |
| Scoring | deterministic stub | SWE-bench Verified |
| Needs | nothing — pure Python | Docker daemon + `--extra real` + dataset |

**Start on the mock path.** It exercises the entire system (orchestrator,
metrics, dashboard, persistence) with no GPU, Docker, or downloads. Switch to the
real path only when you have a real LiteLLM + cluster to measure.

---

## 1. Prerequisites

- [uv](https://docs.astral.sh/uv/) (`uv --version` should print a version)
- Python 3.11+ (uv manages this)
- For the **real path only**: a running Docker daemon and a reachable LiteLLM
  proxy with the Prometheus callback enabled (see §7).

```bash
cd cluster-bench
uv sync                       # mock-path deps
# If uv can't reach PyPI on this host, prefix with the CA bundle:
# SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt uv sync
```

---

## 2. Mock path — full operating flow

### Step 1 — start the fake LiteLLM proxy

It serves `POST /v1/chat/completions` (what the agent calls) and `GET /metrics`
(what ClusterBench scrapes).

```bash
# Terminal A
uv run uvicorn mock_litellm:app --port 4000
```

Sanity check (Terminal C):

```bash
curl -s http://localhost:4000/metrics | head
```

### Step 2 — start the ClusterBench server

```bash
# Terminal B
uv run python run_server.py --port 8000 \
    --base-url    http://localhost:4000/v1 \
    --metrics-url http://localhost:4000/metrics
```

> ⚠️ **`--metrics-url` is authoritative.** The server scrapes exactly this URL.
> If it doesn't point at a live `/metrics`, every level reports
> `wire_metrics_available: false` and empty deltas — the run still completes, but
> with no wire numbers. This is the single most common misconfiguration.

Verify the server is up:

```bash
curl -s http://localhost:8000/api/health
# {"ok":true,"active_run_id":null,"n_subscribers":0}
```

### Step 3 — open the dashboard

```
http://localhost:8000/
```

Leave it open — it connects to `WS /ws` and will render the run live.

### Step 4 — launch a sweep

Either use the dashboard's **controls** panel, or POST directly:

```bash
curl -s -X POST http://localhost:8000/api/run \
  -H 'content-type: application/json' \
  -d '{
        "name": "demo-sweep",
        "mode": "sweep",
        "levels": [1, 2, 4, 8],
        "task_slice": {"n": 5},
        "scrape_interval_s": 0.5,
        "guards": { "max_p99_latency_s": 2.0, "max_error_rate": 0.1 }
      }'
```

Response (`202 Accepted`):

```json
{
  "run_id": "5d7ad68377ab",
  "pinned_instance_ids": ["mock-verified-test-0000", "..."],
  "config": { "...": "the fully-resolved RunConfig" }
}
```

Grab the `run_id` — you'll use it to poll and retrieve.

> **One run at a time.** A second `POST /api/run` while one is active returns
> **`409 Conflict`** with the active run_id in the body. Wait for the first to
> finish (the active slot clears on completion).

### Step 5 — watch it run

On the dashboard you'll see, live:

- **agents-vs-TTFT** and **agents-vs-latency** headline graphs filling in per
  level,
- the **saturation curve** (throughput vs concurrency) with the knee marked if a
  guard trips,
- the **LiteLLM saturation panel** (in-flight peak + processing overhead),
- the **per-level table** and **outcome taxonomy**,
- the **live event feed**.

Or follow from the CLI by polling:

```bash
RID=5d7ad68377ab
until curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/api/runs/$RID | grep -q 200; do
  sleep 0.5
done
echo "run persisted"
```

`GET /api/runs/{id}` returns **404 until the run completes** — completion is what
writes `results/<run_id>.json`.

### Step 6 — read the report

```bash
curl -s http://localhost:8000/api/runs/$RID | python3 -m json.tool | head -40
```

Quick health summary:

```bash
curl -s http://localhost:8000/api/runs/$RID | python3 -c "
import sys, json
r = json.load(sys.stdin)
print('mode :', r['config']['mode'])
print('wire :', r['wire_metrics_available'], '| ttft:', r['ttft_available'])
print('knee :', r['knee'])
for lv in r['levels']:
    d = lv['delta'] or {}
    print(f\"  L{lv['level']:>2}: tps={d.get('throughput_tps',0):8.1f} \"
          f\"p99={d.get('lat_p99')} ttft_p95={d.get('ttft_p95')} \"
          f\"inflight_peak={d.get('in_flight_peak')} \"
          f\"pass={lv['pass_rate']:.2f} outcomes={lv['outcome_counts']}\")
"
```

Expected shape on a healthy mock run: `wire: True`, `ttft: True`, throughput
**rising** with concurrency, every level carrying a non-null `delta`.

### Step 7 — list & compare runs

```bash
curl -s http://localhost:8000/api/runs | python3 -m json.tool
```

To **overlay** two runs on the headline axes: in the dashboard's **saved
reports** panel, ctrl/cmd-click two or more runs and click **Load selected**.
Each run gets its own color + legend entry — this is how you compare configs
(e.g. streaming vs non-streaming, or two models) on the same agents-vs-latency
axis.

### Step 8 — shut down

`Ctrl-C` Terminal B (server), then Terminal A (mock LiteLLM). Saved reports
remain under `results/`.

---

## 3. Running a SOAK (stability at one level)

A sweep maps the curve; a **soak** holds one concurrency level for a duration and
bins it over time — use it to find drift, leaks, or degradation under sustained
load.

```bash
curl -s -X POST http://localhost:8000/api/run \
  -H 'content-type: application/json' \
  -d '{
        "name": "demo-soak",
        "mode": "soak",
        "levels": [8],            # SOAK holds the FIRST level only
        "soak_duration_s": 120,
        "task_slice": {"n": 5},
        "scrape_interval_s": 1.0
      }'
```

Each time bin appears as a `level_done` entry in the report's `levels[]`.

---

## 4. The `POST /api/run` body — every field

| field | default | meaning |
|---|---|---|
| `name` | `""` | label shown on the dashboard / in listings |
| `mode` | `"sweep"` | `"sweep"` (curve) or `"soak"` (hold one level) |
| `levels` | `[1,4,8,16]` | worker counts to visit in order; SOAK uses `levels[0]` |
| `soak_duration_s` | `1800` | SOAK only — how long to hold the level |
| `task_slice.n` | `5` | how many SWE-bench instances to pin |
| `task_slice.subset` | `"verified"` | dataset subset |
| `task_slice.split` | `"test"` | dataset split |
| `task_slice.pinned_instance_ids` | `[]` | pin exact ids (overrides `n`); reused at every level |
| `scrape_interval_s` | `1.0` | metrics scrape cadence — must catch the in-flight peak |
| `model` | `"gpt-4o-mini"` | model name passed to LiteLLM |
| `streaming` | `true` | needed for TTFT to exist |
| `step_limit` | `0` | per-task agent step cap (0 = unlimited) |
| `guards` | `{}` | knee thresholds — see below |

**Guards** (any present key is checked per level; first trip marks the knee, the
sweep continues to map the full curve):

| guard key | trips when |
|---|---|
| `max_p99_latency_s` | level p99 latency exceeds it |
| `max_ttft_p95_s` | level TTFT p95 exceeds it |
| `max_error_rate` | level error rate exceeds it |
| `min_pass_rate` | level pass rate drops below it |
| `max_in_flight_peak` | level in-flight peak exceeds it |

---

## 5. The live event stream (`WS /ws`)

Connect to `ws://<host>/ws`. A mid-run connect **replays history first**, then
streams live. Every frame is JSON:

```json
{ "type": "<event>", "payload": { ... } }
```

Event vocabulary, in order of a typical sweep:

| `type` | when | key payload |
|---|---|---|
| `run_start` | run begins | `run_id`, `mode`, `levels`, `pinned_instance_ids` |
| `level_start` | a level begins | `level` (`bin_idx` for soak) |
| `scrape` | start / end / delta of a level | `phase` ∈ {`start`,`end`,`delta`}, `delta` |
| `task` | one agent attempt resolved | `instance_id`, `outcome`, `resolved` |
| `level_done` | a level's summary is ready | the full `LevelSummary` |
| `knee` | first guard trips | `level`, `reason` |
| `run_done` | run finishes | `run_id`, `knee`, `n_levels` |

Tail it from the CLI with any WS client, e.g.:

```bash
# requires `websocat`; illustrative
websocat ws://localhost:8000/ws
```

---

## 6. Outcome taxonomy (how each attempt is bucketed)

Each task attempt lands in **exactly one** bucket, resolved by priority
(highest first):

1. `timeout` — hit the step/time limit
2. `agent_error` — non-zero exit / crash
3. `inference_error` — a LiteLLM-reported failed request in that level
4. `unresolved` — ran fine, patch didn't resolve the task
5. `resolved` — patch resolved the task

The per-level `outcome_counts` always sums to `n_tasks`.

---

## 7. Real path — operating against a live cluster

### Step 1 — enable LiteLLM's Prometheus endpoint

In your LiteLLM config:

```yaml
litellm_settings:
  callbacks: ["prometheus"]
```

- LiteLLM now serves `/metrics`. ClusterBench pins the series it reads in
  `clusterbench/metrics/litellm.py` (the one file to touch if names drift).
- **Multi-worker LiteLLM** (gunicorn/uvicorn `--workers N`): set
  `PROMETHEUS_MULTIPROC_DIR` to a shared writable dir, or you'll scrape one
  worker's partial counters and the deltas will undercount.
- **Streaming on** ⇒ TTFT series present. Off ⇒ ClusterBench flags TTFT
  unavailable (never fakes 0).

### Step 2 — install the real extra

```bash
uv sync --extra real          # adds mini-swe-agent + swebench
```

### Step 3 — run the server in real mode

```bash
uv run python run_server.py --real \
    --base-url    http://litellm:4000/v1 \
    --metrics-url http://litellm:4000/metrics \
    --api-key     "$LITELLM_KEY" \
    --results-dir ./results
```

mini-swe-agent spawns one Docker container per task, so the **host needs a
running Docker daemon**. In a container, mount the socket (see §8).

### Step 4 — launch & read

Identical to the mock path (§2 Steps 4–7). The difference is only *what's
underneath*: real agents, real LiteLLM metrics, real SWE-bench scoring.

### Sizing guidance

- SWE-bench Verified task images are sizeable — budget disk + CPU.
- A level = `--workers` = concurrent agents. Keep the top level at or below the
  host's real parallelism, or you measure host contention, not the stack.
- Set `scrape_interval_s` frequent enough to catch the within-level in-flight
  peak (the default 1.0s is fine for slow sweeps; tighten for fast ones).

---

## 8. Docker

```bash
# Mock path (CI / demo) — no GPU, images, or downloads.
docker build -t clusterbench .
docker run --rm -p 8000:8000 clusterbench

# Real path — needs the host Docker socket (per-task containers).
docker build --build-arg INSTALL_EXTRA=real -t clusterbench:real .
docker run --rm -p 8000:8000 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$PWD/results:/app/results" \
    clusterbench:real --real --base-url http://litellm:4000/v1
```

Mount a volume at `/app/results` to keep persisted reports across restarts.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `wire_metrics_available: false`, empty deltas | `--metrics-url` not pointing at a live `/metrics` | Point it at the real endpoint; `curl` it to confirm |
| `ttft_available: false` | streaming off (no TTFT series) | Enable streaming on the agent / LiteLLM |
| `POST /api/run` → `409` | a run is already active | Wait for it; one run at a time |
| `GET /api/runs/{id}` → `404` | run hasn't completed yet | Poll until `200` (completion writes the file) |
| `POST /api/run` → `422` | bad body (e.g. `mode` not sweep/soak) | Fix the JSON; `mode` ∈ {`sweep`,`soak`} |
| real path: `mini-extra not found` | `--extra real` not installed | `uv sync --extra real` |
| real path: container errors | Docker daemon down / socket not mounted | Start Docker; mount `/var/run/docker.sock` |
| `uv sync` can't reach PyPI | host CA not picked up | prefix `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt` |
| dashboard blank, "disconnected" | server not reachable on `/ws` | Check the server is up; the client auto-reconnects |

---

## 10. Quick reference

```bash
# Mock path, two terminals:
uv run uvicorn mock_litellm:app --port 4000
uv run python run_server.py --port 8000 \
    --base-url http://localhost:4000/v1 --metrics-url http://localhost:4000/metrics

# Endpoints:
#   GET  /                      dashboard
#   GET  /api/health            liveness + active run + subscribers
#   POST /api/run               start a run  (202 / 409)
#   GET  /api/runs              list saved reports
#   GET  /api/runs/{run_id}     fetch one report (404 until complete)
#   WS   /ws                    live event stream (replays history on connect)

# Artifacts:
#   results/<run_id>.json       one persisted RunReport per run
```
