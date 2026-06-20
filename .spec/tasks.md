# Tasks: ClusterBench

**Branch**: `001-clusterbench` | **Spec**: `./spec.md` | **Plan**: `./plan.md`
rev 3 — LiteLLM /metrics, no custom proxy

Each phase ends with a **verification gate** Claude Code MUST pass before
advancing. `[P]` = parallelizable. Write tests with the code.

---

## Phase 0 — Skeleton & contracts

- **T001** Package layout per plan (`clusterbench/`, `metrics/`, `web/static/`),
  init files, `requirements.txt` (pin `mini-swe-agent`, `swebench`, fastapi,
  uvicorn, httpx, aiohttp), `.gitignore`, empty `results/`.
- **T002** `models.py`: dataclasses + enums (LoadMode{sweep,soak},
  Outcome{resolved,unresolved,inference_error,agent_error,timeout},
  DegradationGuard, RunConfig, ScrapeSnapshot, LevelDelta, TaskOutcome,
  LevelSummary, RunReport) + RunConfig/RunReport `to_dict`/`from_dict`.
- **T003 [P]** `mock_litellm.py`: fake OpenAI `/v1/chat/completions` (streaming
  optional, toggleable to test TTFT-absent) + fake LiteLLM `/metrics` whose
  token/request **counters advance** with served requests and whose
  `litellm_in_flight_requests` rises with concurrency. Deterministic counter
  advance so delta tests are exact.

**Gate 0**: `import clusterbench.models` OK; RunConfig round-trips; mock_litellm
serves OpenAI calls and a parseable `/metrics` whose counters increase across two
scrapes.

---

## Phase 1 — MetricsSource: LiteLLM (scrape-and-delta) ★ metrics backbone

- **T010** `metrics/base.py`: `MetricsSource` protocol (`snapshot()`/`diff()`),
  `ScrapeSnapshot`, and **bucket-edge percentile** helpers over Prometheus
  histogram `_bucket` series.
- **T011** `metrics/litellm.py` `LiteLLMSource.snapshot()`: scrape LiteLLM
  `/metrics`, parse the required series (FR-9): TTFT histogram, LLM-call latency
  histogram, end-to-end latency histogram, processing-overhead, token counters,
  request counters w/ status_code, `litellm_in_flight_requests`. Unreachable →
  None (FR-14). Detect TTFT series presence → `ttft_available` (FR-12).
- **T012** `LiteLLMSource.diff(start,end,in_flight_peak,duration)` → `LevelDelta`
  (FR-10): counter deltas (tokens, requests, failed), throughput, error rate,
  bucket-edge percentiles over the delta, in-flight peak, processing-overhead.
  Guard against counter resets (decrease ⇒ mark suspect, no negatives).
- **T013** `metrics/vllm.py` `VLLMSource`: interface-conforming **stub** for
  later (FR-OPT) — registers + conforms, not selected in this topology.
- **T014** Tests: bucket-edge p50/p95/p99 from a fixture histogram; **delta math**
  exact against the deterministic mock counters (AC-4); unreachable → None +
  flag; TTFT-absent mock → `ttft_available=False` (AC-11); counter-reset →
  suspect not negative; `VLLMSource` passes the same interface-conformance test
  (AC-OPT).

**Gate 1** (AC-4, AC-11, AC-OPT): `pytest tests/test_metrics_litellm.py
tests/test_metrics_base.py` green; scrape the mock twice across known load and
assert Δtokens/Δrequests/throughput/error-rate match hand-computed values;
TTFT-absent path flagged; vLLM stub conforms.

---

## Phase 2 — mini-swe-agent runner + scoring

- **T020** `miniswerunner.py` `build_cmd(level, slice, model, out_dir)`:
  `mini-extra swebench --subset verified --split test --workers <level>
   --instances <pinned slice> --model <model> -o <out_dir>`, model base URL →
  LiteLLM, **streaming enabled** (for TTFT), proxy address reachable from task
  containers (FR-1..FR-3, FR-12).
- **T021** `MiniSweRunner.run(level)`: execute batch (subprocess) under a budget;
  locate + parse out dir: `preds.json` + per-instance trajectory/logs → process
  records (return status, wall time, timeout flag, log tail) (FR-17).
- **T022** `scoring.py` `score_predictions(preds, instances)` → resolved set via
  SWE-bench Verified (FR-18); real scoring behind `slow` marker.
- **T023** `MockRunner` + `mock_minisweagent.py`: emulate a level by driving
  concurrent multi-turn traffic at mock_litellm for N pinned instances (advancing
  its counters) and writing a fake preds.json; `scoring.py` stub marks a
  configurable fraction resolved (CI engine — no Docker/GPU/downloads, FR-4/FR-29).
- **T024** Pinned-slice helper: choose N instance_ids once, reuse every level
  (FR-7/AC-10).
- **T025** Tests: `build_cmd` has correct workers/instances/model/base-url +
  streaming flag; out-dir parser on a fixture dir → records + preds; mock runner
  advances mock counters + writes preds; stub scorer returns configured set;
  pinned slice identical across two levels.

**Gate 2** (AC-6, AC-10): `pytest tests/test_miniswerunner.py tests/test_scoring.py
-m "not slow"` green; mock runner over [2,4] uses identical instance_ids,
advances counters, and yields scored preds.

---

## Phase 3 — Orchestrator (sweep + soak, scrape-and-delta)

- **T030** `Orchestrator` scaffold: holds RunConfig, the selected `MetricsSource`
  (LiteLLMSource), event emitter; resolves pinned slice once.
- **T031** Per-level engine: scrape@start → run `MiniSweRunner(level)` while
  polling in-flight peak on the scrape interval → scrape@end → `source.diff(...)`
  → `LevelDelta` (FR-10/FR-11).
- **T032** SWEEP (FR-6/FR-8): sequential over levels; per level do T031, score
  preds, build `LevelSummary`, eval guards, emit `level_done`.
- **T033** SOAK (FR-6): hold one level for a duration, re-feeding tasks; periodic
  scrape-and-delta into time bins.
- **T034** `LevelSummary` aggregation: LevelDelta + pass rate (scorer) +
  outcome_counts (taxonomy) + duration.
- **T035** Outcome taxonomy resolver (FR-19): pure function, priority
  timeout→agent_error→inference_error→unresolved→resolved; inference_error from
  LiteLLM failed-request delta and/or per-instance error surfacing; mutually
  exclusive.
- **T036** Guards + knee (FR-26/FR-27): per-level eval (bucket-edge percentiles,
  in-flight peak); first trip recorded; sweep may continue.
- **T037** Tests: sweep runs all levels w/ pinned slice; knee = first guard trip;
  soak respects duration ± tol; taxonomy branches each hit; wire-metrics-
  unavailable path → run completes, process+score present (AC-3).

**Gate 3** (AC-1, AC-2, AC-3, AC-5): `pytest tests/test_orchestrator.py` green;
mock SWEEP [1,4,8,16] → agents-vs-TTFT/latency populated per level via delta,
throughput rises then falls, knee = first trip; block `/metrics` →
wire-metrics-unavailable, process+score still present; each taxonomy branch hit.

---

## Phase 4 — Web server, live telemetry, persistence

- **T040** `web/server.py`: `POST /api/run` (409 if active), `GET /api/runs`,
  `GET /api/runs/{id}`, `GET /`, static; build RunConfig from body.
- **T041** WebSocket `/ws` + hub (FR-21/FR-22): fan-out, bounded ring buffer,
  history replay on connect.
- **T042** Persist RunReport JSON to `results/<run_id>.json` (FR-24); retrieve via
  `GET /api/runs/{id}`.
- **T043** Tests (ASGI transport): endpoints serve; start → run id; 2nd run → 409;
  report persists + retrievable; mid-run connect → replay.

**Gate 4** (AC-7, AC-8, AC-9): `pytest tests/test_server.py` green; end-to-end
mock SWEEP via API (no GPU/Docker/downloads) saves a report ≥2 levels; two runs →
two retrievable reports; mid-run connect renders completed levels.

---

## Phase 5 — Dashboard (headline graphs + LiteLLM saturation panel + overlay)

- **T050** `index.html` + `static/app.js`: WS client; SVG: **agents-vs-TTFT**,
  **agents-vs-latency** (headline), saturation curve w/ peak + knee.
- **T051** **LiteLLM saturation panel** (FR-15/FR-16): in-flight peak +
  processing-overhead vs concurrency, with the **pre-handler-queueing annotation**
  (low in-flight + high latency ⇒ delay LiteLLM can't see). TTFT-unavailable
  state rendered explicitly (FR-12).
- **T052** Per-level table (bucket-percentiles labeled as such, tokens/sec, pass
  rate, error rate, outcome counts) + outcome-taxonomy panel + live event feed.
- **T053** Saved-report mode (FR-25): load by id, render all graphs later;
  **overlay 2+ runs** on the headline axes.
- **T054** Controls: mode (sweep/soak), levels or level+duration, slice size,
  LiteLLM base + metrics URLs, model, streaming toggle, guards, scrape interval;
  start button w/ in-progress state.
- **T055** Render verification against a seeded finished run.

**Gate 5**: headless load of `/` on a finished mock run shows non-empty
agents-vs-TTFT, agents-vs-latency, saturation, and the LiteLLM saturation panel
(in-flight + overhead) with the pre-handler annotation; bucket-percentile columns
labeled; loading a 2nd saved report overlays both series.

---

## Phase 6 — Packaging & docs

- **T060** `Dockerfile` + `run_server.py`; document: Docker socket mount (mini-
  swe-agent), enabling LiteLLM `callbacks:["prometheus"]` + (multi-worker)
  `PROMETHEUS_MULTIPROC_DIR`, streaming for TTFT, SWE-bench Verified disk/CPU
  sizing, worker-vs-CPU guidance.
- **T061** `README.md`: wrapper architecture (sweep + LiteLLM-metrics + dashboard
  around mini-swe-agent); the scrape-and-delta model + bucket-edge caveat; the
  LiteLLM-side saturation signal + pre-handler-queueing caveat + that vLLM
  internals are unavailable (pluggable for later); mock path vs real; how to
  overlay saved reports.
- **T062** Self-test: mock_litellm + mock runner + stub scorer → run SWEEP and
  SOAK via API → assert each persists a valid report with headline curves
  populated from deltas.

**Gate 6** (final): `docker build` OK and serves dashboard; self-test passes
sweep+soak on the mock path (no GPU/Docker images/downloads); full `pytest` green;
all earlier gates still pass.

---

## Cross-cutting (verify continuously)
- **LiteLLM /metrics is the only wire source** — scrape-and-delta; no custom
  proxy; no per-request assumptions.
- **vLLM source stays pluggable-but-inactive** — keep the interface clean so it
  can be added if vLLM ever becomes reachable (FR-OPT/AC-OPT).
- **Same pinned slice every sweep level** (AC-10).
- **Don't reimplement** agent loop, container isolation, or SWE-bench
  verification — delegate to mini-swe-agent + swebench scorer.
- **TTFT is streaming-only** — detect + flag, never report 0 as if real.
- **Mock path keeps CI hardware-free** — real batches + real scoring behind a
  `slow` marker.
- **One outcome bucket per attempt.**
