// ClusterBench dashboard — vanilla JS, no framework, no build step.
//
// Layout:
//   state           — single source of truth (active run + loaded runs)
//   transport       — WS for live events; fetch for saved reports + start run
//   renderers       — pure functions of state → SVG/table DOM
//   wire()          — bind controls + boot the WS connection
//
// Percentile values in LevelDelta are bucket-edge (coarse, by design — see
// .spec/README.md). The table labels them as such so users don't read false
// precision into them.

(() => {
"use strict";

// ---------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------

const WILDCARD = "*"; // when no overlay runs are loaded, render the live/active run

// Cyan→magenta heat ramp (low → high concurrency = rising heat). Mirrors the
// --heat-* CSS vars. Index 0 (aqua) is the live/active run.
const SERIES_COLORS = [
  "#38e0c8", "#4cc8e0", "#6aa9ef", "#9b8cf0",
  "#d56fdc", "#ff5da2", "#ff7a6b", "#ffa64d",
];

// Outcome colors mirror the --oc-* CSS vars.
const OUTCOME_COLORS = {
  resolved: "#38e0c8",
  unresolved: "#6aa9ef",
  inference_error: "#ffb454",
  agent_error: "#ff5277",
  timeout: "#d56fdc",
};

// The aqua signal + hot-pink knee, read from CSS so JS and CSS never drift.
const OVERHEAD_COLOR = "#d56fdc"; // the proc-overhead line on the LiteLLM panel
const AXIS_TICK_SIZE = 10;

const OUTCOME_ORDER = ["resolved", "unresolved", "inference_error", "agent_error", "timeout"];

const MAX_EVENTS = 200; // cap event feed to keep DOM small
const RECONNECT_DELAY_MS = 1500;

// ---------------------------------------------------------------------
// State
// ---------------------------------------------------------------------

const state = {
  // Live run being built from WS events. Null when nothing's in flight.
  activeRun: null,
  // Saved reports loaded for overlay. Each has {run_id, name, report}.
  overlayRuns: [],
  // Last N events for the feed.
  events: [],
  // Connection state: "connecting" | "connected" | "disconnected".
  wsState: "disconnected",
  // Server wiring + run defaults from /api/config (model, endpoints, path).
  serverConfig: null,
  // True while a run is active (run_start received, run_done not yet).
  runInProgress: false,
  // ISO timestamp of when the active run started (for elapsed timer).
  runStartedAt: null,
};

let _ws = null;
let _wsReconnectTimer = null;
let _elapsedTimer = null;

// ---------------------------------------------------------------------
// Transport: WebSocket
// ---------------------------------------------------------------------

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}

function connect() {
  setWsState("connecting");
  if (_ws) { try { _ws.close(); } catch (e) { /* ignore */ } }
  const ws = new WebSocket(wsUrl());
  _ws = ws;
  ws.onopen = () => {
    setWsState("connected");
    // Server replays history on connect — no explicit request needed.
  };
  ws.onclose = () => {
    setWsState("disconnected");
    scheduleReconnect();
  };
  ws.onerror = () => {
    // onclose will fire next; just update visual state.
    setWsState("disconnected");
  };
  ws.onmessage = (ev) => {
    let frame;
    try { frame = JSON.parse(ev.data); } catch (e) { return; }
    if (!frame || !frame.type) return;
    onEvent(frame.type, frame.payload || {});
  };
}

function scheduleReconnect() {
  if (_wsReconnectTimer) return;
  _wsReconnectTimer = setTimeout(() => {
    _wsReconnectTimer = null;
    connect();
  }, RECONNECT_DELAY_MS);
}

function setWsState(s) {
  state.wsState = s;
  const el = document.getElementById("connection-indicator");
  if (!el) return;
  el.classList.remove("connected", "disconnected", "connecting");
  el.classList.add(s);
  const labels = { connected: "live", connecting: "linking", disconnected: "offline" };
  el.querySelector(".label").textContent = labels[s] || s;
}

// ---------------------------------------------------------------------
// Event dispatcher — turn WS frames into state mutations + re-render
// ---------------------------------------------------------------------

function onEvent(type, payload) {
  pushEvent(type, payload);

  switch (type) {
    case "run_start":
      state.activeRun = newRunFromStart(payload);
      state.runInProgress = true;
      state.runStartedAt = Date.now();
      showActiveRun();
      setRunInProgress(true);
      startElapsedTimer();
      break;
    case "run_done":
      // Pin the finished-at + knee; the persisted report (fetched later) is
      // the source of truth, but we update what we have so the UI doesn't
      // go blank waiting for /api/runs/{id}.
      if (state.activeRun) {
        state.activeRun.finished_at = new Date().toISOString();
      }
      state.runInProgress = false;
      setRunInProgress(false);
      stopElapsedTimer();
      { const g = document.getElementById("live-gauges"); if (g) g.classList.add("hidden"); }
      // After a run finishes, refresh the saved-reports list so it appears.
      refreshSavedList();
      break;
    case "level_start":
      if (state.activeRun) {
        const lv = ensureLevel(state.activeRun, payload.level);
        if (payload.bin_idx !== undefined) lv.bin_idx = payload.bin_idx;
      }
      break;
    case "scrape":
      if (state.activeRun && payload.delta) {
        const lv = ensureLevel(state.activeRun, payload.level);
        lv.delta = payload.delta;
      }
      break;
    case "task":
      if (state.activeRun) {
        const lv = ensureLevel(state.activeRun, payload.level);
        // The orchestrator emits task events AFTER level_done's scrape, so a
        // level may already exist with a delta.
        lv.tasks = lv.tasks || [];
        lv.tasks.push({
          instance_id: payload.instance_id,
          outcome: payload.outcome,
          resolved: payload.resolved,
        });
        lv.outcome_counts = lv.outcome_counts || {};
        lv.outcome_counts[payload.outcome] = (lv.outcome_counts[payload.outcome] || 0) + 1;
      }
      break;
    case "level_done":
      if (state.activeRun) {
        // level_done payload IS the LevelSummary dict.
        replaceLevel(state.activeRun, payload);
      }
      break;
    case "knee":
      if (state.activeRun) {
        state.activeRun.knee = { level: payload.level, reason: payload.reason };
      }
      break;
    case "level_live":
      if (state.activeRun) {
        state.activeRun.liveStats = payload;
        renderLiveGauges(payload);
      }
      return; // skip full renderAll — gauges update independently
  }
  renderAll();
}

function newRunFromStart(payload) {
  return {
    run_id: payload.run_id,
    name: payload.name || "",
    mode: payload.mode || "sweep",
    levels: payload.levels || [],
    pinned_instance_ids: payload.pinned_instance_ids || [],
    knee: null,
    levels_data: [],
    finished_at: null,
    // Does the runner open streaming completions? When false, TTFT/TPOT/cache
    // are not measurable and the gauges render "n/a" honestly.
    agent_streams: payload.agent_streams !== undefined ? payload.agent_streams : true,
  };
}

function ensureLevel(run, level) {
  if (!run.levels_data) run.levels_data = [];
  let lv = run.levels_data.find((x) => x.level === level);
  if (!lv) {
    lv = { level, delta: null, tasks: [], outcome_counts: {} };
    run.levels_data.push(lv);
  }
  return lv;
}

function replaceLevel(run, summary) {
  if (!run.levels_data) run.levels_data = [];
  const idx = run.levels_data.findIndex((x) => x.level === summary.level);
  // LevelSummary has: level, delta, n_tasks, pass_rate, outcome_counts, duration_s
  const merged = {
    level: summary.level,
    delta: summary.delta,
    tasks: [], // task events came earlier; outcome_counts has the aggregate
    outcome_counts: summary.outcome_counts || {},
    n_tasks: summary.n_tasks,
    pass_rate: summary.pass_rate,
    duration_s: summary.duration_s,
  };
  if (idx >= 0) {
    run.levels_data[idx] = Object.assign(run.levels_data[idx], merged);
  } else {
    run.levels_data.push(merged);
  }
  // Keep levels sorted for stable rendering.
  run.levels_data.sort((a, b) => a.level - b.level);
}

function pushEvent(type, payload) {
  state.events.push({
    t: new Date().toISOString().split("T")[1].replace("Z", ""),
    type,
    payload,
  });
  if (state.events.length > MAX_EVENTS) {
    state.events = state.events.slice(-MAX_EVENTS);
  }
}

// ---------------------------------------------------------------------
// Transport: fetch (config + saved reports + start run)
// ---------------------------------------------------------------------

async function loadConfig() {
  // Pull the server's actual wiring so the readout + controls reflect reality
  // (the model the operator set in config.yaml, not a hardcoded default).
  try {
    const r = await fetch("/api/config");
    if (!r.ok) return;
    state.serverConfig = await r.json();
  } catch (e) {
    /* leave readout as placeholders */
  }
  renderConfigReadout();
  prefillControlsFromConfig();
  syncLiveReadout();
}

function _setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

/** Trim a URL to host[+path tail] so the readout stays compact. */
function _shortUrl(url) {
  if (!url) return "—";
  try {
    const u = new URL(url);
    const tail = u.pathname.replace(/\/$/, "");
    return u.host + (tail && tail !== "" ? tail : "");
  } catch (e) {
    return url;
  }
}

function renderConfigReadout() {
  const cfg = state.serverConfig;
  const server = (cfg && cfg.server) || {};
  const defaults = (cfg && cfg.run_defaults) || {};

  _setText("rd-base", _shortUrl(server.base_url));
  _setText("rd-metrics", _shortUrl(server.metrics_url));
  // model / scrape / streaming are per-run form fields — mirrored live by
  // syncLiveReadout() rather than pinned to the server default here.
  const pathEl = document.getElementById("rd-path");
  if (pathEl) {
    const path = server.path || "—";
    pathEl.textContent = path;
    pathEl.className = "v " + (path === "real" ? "path-real" : "path-mock");
  }
  syncLiveReadout();
}

/** Mirror the per-run form fields (model, scrape, streaming) into the top
 *  readout so it reflects what the current run will actually use, not just the
 *  server default. Called on input and once after the form is prefilled. */
function syncLiveReadout() {
  const form = document.getElementById("start-form");
  if (!form) return;
  const defaults = (state.serverConfig && state.serverConfig.run_defaults) || {};
  const modelVal = (form.model.value || "").trim();
  _setText("rd-model", modelVal || defaults.model || "—");
  const scrape = parseFloat(form.scrape_interval_s.value);
  _setText("rd-scrape", Number.isFinite(scrape) && scrape > 0 ? `${scrape}s` : "—");
  _setText("rd-streaming", form.streaming.checked ? "on" : "off");
}

/** Prefill the controls with the server defaults so a submitted run inherits
 *  the configured model unless the operator deliberately changes it. */
function prefillControlsFromConfig() {
  const defaults = (state.serverConfig && state.serverConfig.run_defaults) || {};
  const form = document.getElementById("start-form");
  if (!form) return;
  if (defaults.model && !form.model.value) form.model.value = defaults.model;
  if (defaults.scrape_interval_s != null) {
    form.scrape_interval_s.value = defaults.scrape_interval_s;
  }
  if (typeof defaults.streaming === "boolean") {
    form.streaming.checked = defaults.streaming;
  }
}

async function refreshSavedList() {
  const sel = document.getElementById("saved-select");
  if (!sel) return;
  try {
    const r = await fetch("/api/runs");
    if (!r.ok) return;
    const data = await r.json();
    // Preserve currently-selected ids across the refresh.
    const selected = new Set(
      Array.from(sel.selectedOptions).map((o) => o.value)
    );
    sel.innerHTML = "";
    for (const run of data.runs || []) {
      const opt = document.createElement("option");
      opt.value = run.run_id;
      const label = run.name
        ? `${run.name} (${run.run_id})`
        : run.run_id;
      opt.textContent = `${label} — ${run.n_levels} levels — ${run.finished_at || ""}`;
      if (selected.has(run.run_id)) opt.selected = true;
      sel.appendChild(opt);
    }
  } catch (e) {
    /* network glitch; another refresh will retry */
  }
}

async function loadSelectedRuns() {
  const sel = document.getElementById("saved-select");
  if (!sel) return;
  const ids = Array.from(sel.selectedOptions).map((o) => o.value);
  if (ids.length === 0) return;
  // Replace overlay set with the current selection.
  state.overlayRuns = [];
  for (const id of ids) {
    try {
      const r = await fetch(`/api/runs/${encodeURIComponent(id)}`);
      if (!r.ok) continue;
      const report = await r.json();
      state.overlayRuns.push({
        run_id: id,
        name: report.name || id,
        report,
      });
    } catch (e) {
      /* skip */
    }
  }
  renderAll();
}

function clearOverlay() {
  state.overlayRuns = [];
  // Deselect everything in the list.
  const sel = document.getElementById("saved-select");
  if (sel) Array.from(sel.options).forEach((o) => (o.selected = false));
  renderAll();
}

// ---------------------------------------------------------------------
// Controls: start a run
// ---------------------------------------------------------------------

function parseLevels(text) {
  // "1, 4, 8, 16" → [1,4,8,16]. Soak mode: take the first only.
  return text
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => parseInt(s, 10))
    .filter((n) => Number.isFinite(n) && n > 0);
}

function buildStartBody() {
  const form = document.getElementById("start-form");
  const fd = new FormData(form);
  const mode = fd.get("mode");
  const levels = parseLevels(fd.get("levels") || "");
  const guards = {};
  const gMaxP99 = parseFloat(fd.get("guard_max_p99"));
  const gMaxTtft = parseFloat(fd.get("guard_max_ttft_p95"));
  const gMaxErr = parseFloat(fd.get("guard_max_error_rate"));
  const gMinPass = parseFloat(fd.get("guard_min_pass_rate"));
  const gMaxInFlight = parseFloat(fd.get("guard_max_in_flight"));
  if (Number.isFinite(gMaxP99)) guards.max_p99_latency_s = gMaxP99;
  if (Number.isFinite(gMaxTtft)) guards.max_ttft_p95_s = gMaxTtft;
  if (Number.isFinite(gMaxErr)) guards.max_error_rate = gMaxErr;
  if (Number.isFinite(gMinPass)) guards.min_pass_rate = gMinPass;
  if (Number.isFinite(gMaxInFlight)) guards.max_in_flight_peak = gMaxInFlight;

  // model is left out of the body when blank, so the server fills it from its
  // configured default (config.yaml) — never hardcode a model here.
  const modelField = (fd.get("model") || "").toString().trim();

  const body = {
    name: (fd.get("name") || "").toString().trim(),
    mode,
    levels: mode === "soak" ? [levels[0] || 1] : levels,
    soak_duration_s: parseFloat(fd.get("soak_duration_s")) || 1800,
    task_slice: { n: parseInt(fd.get("n"), 10) || 5 },
    scrape_interval_s: parseFloat(fd.get("scrape_interval_s")) || 1,
    streaming: fd.get("streaming") === "on",
    guards,
  };
  if (modelField) body.model = modelField;
  return body;
}

// ---------------------------------------------------------------------
// Run in-progress state — button + badge
// ---------------------------------------------------------------------

function setRunInProgress(running) {
  const btn = document.getElementById("start-btn");
  const stopBtn = document.getElementById("stop-btn");
  const badge = document.getElementById("run-status-badge");
  if (btn) {
    btn.disabled = running;
    btn.textContent = running ? "Running…" : "Start run";
  }
  if (stopBtn) {
    stopBtn.classList.toggle("hidden", !running);
  }
  if (badge) {
    if (running) {
      badge.className = "run-badge running";
      const dot = document.createElement("span");
      dot.className = "badge-dot";
      badge.innerHTML = "";
      badge.appendChild(dot);
      badge.appendChild(document.createTextNode("running"));
    } else {
      badge.className = "run-badge hidden";
    }
  }
}

async function stopRun() {
  const stopBtn = document.getElementById("stop-btn");
  if (stopBtn) { stopBtn.disabled = true; stopBtn.textContent = "Stopping…"; }
  try {
    await fetch("/api/run", { method: "DELETE" });
  } catch (_) {}
}

function _fmtElapsed(ms) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}m ${s % 60}s` : `${s}s`;
}

function startElapsedTimer() {
  stopElapsedTimer();
  _elapsedTimer = setInterval(() => {
    if (!state.runStartedAt) return;
    const el = document.getElementById("active-run-elapsed");
    if (el) el.textContent = _fmtElapsed(Date.now() - state.runStartedAt);
  }, 1000);
}

function stopElapsedTimer() {
  if (_elapsedTimer) { clearInterval(_elapsedTimer); _elapsedTimer = null; }
  const el = document.getElementById("active-run-elapsed");
  if (el) el.textContent = "";
}

async function startRun(ev) {
  ev.preventDefault();
  const btn = document.getElementById("start-btn");
  const status = document.getElementById("start-status");
  btn.disabled = true;
  status.className = "";
  status.textContent = "submitting…";
  try {
    const body = buildStartBody();
    const r = await fetch("/api/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (r.status === 202) {
      const data = await r.json();
      status.className = "ok";
      status.textContent = `▸ started ${data.run_id}`;
      // Reset the active run view in case there's stale state from a prior run.
      state.activeRun = null;
      // Collapse the controls form so the charts are visible.
      const details = document.getElementById("controls-details");
      if (details) details.removeAttribute("open");
    } else if (r.status === 409) {
      const data = await r.json();
      status.className = "busy";
      status.textContent = `busy — run ${data.detail.active_run_id} active`;
      btn.disabled = false;
    } else {
      status.className = "err";
      status.textContent = `error: HTTP ${r.status}`;
      btn.disabled = false;
    }
  } catch (e) {
    status.className = "err";
    status.textContent = `error: ${e.message}`;
    btn.disabled = false;
  }
  // Note: button stays disabled while runInProgress=true (WS run_done re-enables it).
}

// ---------------------------------------------------------------------
// Renderers — series extraction
// ---------------------------------------------------------------------

/**
 * Normalize a run (live or saved) into a series-friendly shape.
 * Returns { run_id, name, color, levels: [{level, delta, outcome_counts,
 *           pass_rate, duration_s, n_tasks}], knee, ttft_available }.
 */
function normalizeRun(run, color, isLive) {
  // Saved reports store everything under .report; live runs use the flat shape.
  const report = run.report || run;
  const levelsData = report.levels_data || report.levels || [];
  const levels = levelsData.map((lv) => {
    // Saved LevelSummary: {level, delta, n_tasks, pass_rate, outcome_counts, duration_s}
    // Live: same after replaceLevel(); pre-level_done, it has {level, delta, tasks, outcome_counts}.
    return {
      level: lv.level,
      delta: lv.delta || null,
      outcome_counts: lv.outcome_counts || {},
      pass_rate: lv.pass_rate || 0,
      duration_s: lv.duration_s || 0,
      n_tasks: lv.n_tasks || 0,
    };
  });
  return {
    run_id: run.run_id || report.run_id,
    name: run.name || report.name || run.run_id || report.run_id,
    color,
    isLive,
    levels,
    knee: report.knee || run.knee || null,
    agent_streams: run.agent_streams !== undefined
      ? run.agent_streams
      : (report.agent_streams !== undefined ? report.agent_streams : true),
    ttft_available: report.ttft_available !== undefined
      ? report.ttft_available
      : (run.ttft_available !== undefined ? run.ttft_available : true),
    wire_metrics_available: report.wire_metrics_available !== undefined
      ? report.wire_metrics_available
      : (run.wire_metrics_available !== undefined ? run.wire_metrics_available : true),
  };
}

/** The runs to render: overlay if loaded, else the live active run. */
function runsToRender() {
  if (state.overlayRuns.length > 0) {
    return state.overlayRuns.map((r, i) =>
      normalizeRun(r, SERIES_COLORS[(i + 1) % SERIES_COLORS.length], false)
    );
  }
  if (state.activeRun) {
    return [normalizeRun(state.activeRun, SERIES_COLORS[0], true)];
  }
  return [];
}

// ---------------------------------------------------------------------
// SVG helpers
// ---------------------------------------------------------------------

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(name, attrs = {}) {
  const el = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, v);
  }
  return el;
}

/**
 * Linear scale: maps a value in domain [d0, d1] to range [r0, r1].
 * Returns a function. If d0 === d1, returns the midpoint of the range.
 */
function scaleLinear(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  if (d0 === d1) {
    return (v) => (r0 + r1) / 2;
  }
  const m = (r1 - r0) / (d1 - d0);
  return (v) => r0 + (v - d0) * m;
}

/** Compute domain extent across all runs for a given accessor. */
function extent(runs, accessor) {
  let lo = Infinity, hi = -Infinity;
  for (const run of runs) {
    for (const lv of run.levels) {
      const v = accessor(lv);
      if (v === null || v === undefined || Number.isNaN(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (lo === Infinity) return [0, 1];
  if (lo === hi) return [lo * 0.9, lo * 1.1];
  return [lo, hi];
}

function niceTicks([lo, hi], n = 5) {
  if (lo === hi) return [lo];
  const span = hi - lo;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
  const err = (n * step0) / span;
  let step;
  if (err <= 0.15) step = 10 * step0;
  else if (err <= 0.35) step = 5 * step0;
  else if (err <= 0.75) step = 2 * step0;
  else step = step0;
  const start = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let v = start; v <= hi + step * 0.5; v += step) ticks.push(v);
  return ticks;
}

/** Bar width tuned to the pixel gap between adjacent (sorted) x levels, capped
 *  so a few levels don't produce absurdly wide bars. Bars are then clamped into
 *  the plot area (via clampBarX) so they never cross the axis lines. */
function barWidthForLevels(levels, xScale, plotLeft, plotRight, maxBar = 64) {
  if (levels.length === 0) return 8;
  if (levels.length === 1) return Math.min(maxBar, (plotRight - plotLeft) * 0.22);
  let minGap = Infinity;
  for (let i = 1; i < levels.length; i++) {
    minGap = Math.min(minGap, xScale(levels[i]) - xScale(levels[i - 1]));
  }
  return Math.max(4, Math.min(minGap * 0.6, maxBar));
}

/** Clamp a bar centered at `centerX` so it stays within [plotLeft, plotRight]. */
function clampBarX(centerX, barWidth, plotLeft, plotRight) {
  const x = centerX - barWidth / 2;
  return Math.max(plotLeft, Math.min(x, plotRight - barWidth));
}

function fmt(v, digits = 2) {
  if (v === null || v === undefined) return "—";
  if (Number.isInteger(v)) return v.toString();
  return Number(v).toFixed(digits);
}

/** Clear an SVG element. */
function clearSvg(svg) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
}

// ---------------------------------------------------------------------
// SVG: axes
// ---------------------------------------------------------------------

const PADDING = { top: 12, right: 16, bottom: 28, left: 48 };

function drawAxes(svg, opts) {
  // opts: {xScale, yScale, xTicks, yTicks, xLabel, yLabel, width, height,
  //        xFormat, yFormat}. Colors come from CSS classes (cb-*), so the
  //        chart theme lives in styles.css, not here.
  const w = opts.width, h = opts.height;
  const xFormat = opts.xFormat || ((v) => v);
  const yFormat = opts.yFormat || ((v) => v);

  // Y axis ticks + gridlines.
  for (const tv of opts.yTicks) {
    const y = opts.yScale(tv);
    svg.appendChild(svgEl("line", {
      class: "cb-grid",
      x1: PADDING.left, x2: w - PADDING.right,
      y1: y, y2: y,
    }));
    const lbl = svgEl("text", {
      class: "cb-tick",
      x: PADDING.left - 6, y: y + 3,
      "text-anchor": "end",
    });
    lbl.textContent = yFormat(tv);
    svg.appendChild(lbl);
  }
  // Y axis label.
  if (opts.yLabel) {
    const t = svgEl("text", {
      class: "cb-axis-label",
      x: 12, y: h / 2,
      "text-anchor": "middle", "font-size": 10,
      transform: `rotate(-90 12 ${h / 2})`,
    });
    t.textContent = opts.yLabel;
    svg.appendChild(t);
  }
  // X axis ticks.
  for (const tv of opts.xTicks) {
    const x = opts.xScale(tv);
    svg.appendChild(svgEl("line", {
      class: "cb-axis",
      x1: x, x2: x,
      y1: h - PADDING.bottom, y2: h - PADDING.bottom + 4,
    }));
    const lbl = svgEl("text", {
      class: "cb-tick",
      x: x, y: h - PADDING.bottom + 16,
      "text-anchor": "middle",
    });
    lbl.textContent = xFormat(tv);
    svg.appendChild(lbl);
  }
  if (opts.xLabel) {
    const t = svgEl("text", {
      class: "cb-axis-label",
      x: (w + PADDING.left - PADDING.right) / 2, y: h - 4,
      "text-anchor": "middle", "font-size": 10,
    });
    t.textContent = opts.xLabel;
    svg.appendChild(t);
  }
  // Axis baselines.
  svg.appendChild(svgEl("line", {
    class: "cb-axis",
    x1: PADDING.left, x2: w - PADDING.right,
    y1: h - PADDING.bottom, y2: h - PADDING.bottom,
  }));
  svg.appendChild(svgEl("line", {
    class: "cb-axis",
    x1: PADDING.left, x2: PADDING.left,
    y1: PADDING.top, y2: h - PADDING.bottom,
  }));
}

// ---------------------------------------------------------------------
// SVG: line chart with per-run series
// ---------------------------------------------------------------------

function drawSeriesChart(svg, { width, height }, runs, accessor, opts = {}) {
  clearSvg(svg);
  if (runs.length === 0 || runs.every((r) => r.levels.length === 0)) {
    drawEmptyState(svg, width, height, opts.emptyMessage || "no data yet");
    return;
  }
  // Series: one line per entry in opts.series (e.g. p50 solid + p95 dashed).
  // Falls back to the single `accessor` when no multi-series is given, so the
  // latency/saturation call sites keep working unchanged.
  const seriesList = opts.series && opts.series.length
    ? opts.series
    : [{ accessor, dashed: false, label: null }];
  const primary = seriesList[0].accessor;

  // X domain: union of all levels (integer ticks).
  const allLevels = new Set();
  for (const run of runs) {
    for (const lv of run.levels) allLevels.add(lv.level);
  }
  const xLevels = Array.from(allLevels).sort((a, b) => a - b);
  const xDomain = xLevels.length === 1
    ? [xLevels[0] * 0.5, xLevels[0] * 1.5]
    : [Math.min(...xLevels), Math.max(...xLevels)];
  const xScale = scaleLinear(xDomain, [PADDING.left, width - PADDING.right]);

  // Y domain: union of all series values across all runs, so every line fits.
  let lo = Infinity, hi = -Infinity;
  for (const run of runs) {
    for (const lv of run.levels) {
      for (const s of seriesList) {
        const v = s.accessor(lv);
        if (v === null || v === undefined || Number.isNaN(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
  }
  const hasData = lo !== Infinity;
  if (!hasData) { lo = 0; hi = 1; }
  // Pad y by 5% on top so points don't sit on the top edge.
  const yPad = (hi - lo) * 0.05;
  const yDomain = [Math.min(0, lo), hi + yPad];
  const yScale = scaleLinear(yDomain, [height - PADDING.bottom, PADDING.top]);

  drawAxes(svg, {
    xScale, yScale,
    xTicks: xLevels,
    yTicks: niceTicks(yDomain),
    xLabel: opts.xLabel || "agents (concurrency)",
    yLabel: opts.yLabel || "",
    xFormat: (v) => v,
    yFormat: (v) => fmt(v, opts.yDigits || 2),
    width, height,
  });

  // Legend (only with data and more than one series): explains line STYLE
  // (solid vs dashed), not color, since color tracks the run.
  if (hasData && seriesList.length > 1) {
    let totalW = 0;
    for (const s of seriesList) totalW += 22 + String(s.label).length * 6 + 14;
    let lx = width - PADDING.right - totalW;
    const ly = PADDING.top + 4;
    for (const s of seriesList) {
      svg.appendChild(svgEl("line", {
        x1: lx, x2: lx + 18, y1: ly + 5, y2: ly + 5,
        stroke: "var(--text-dim)", "stroke-width": 2,
        "stroke-dasharray": s.dashed ? "6 3" : "none",
      }));
      const t = svgEl("text", {
        class: "cb-tick", x: lx + 22, y: ly + 9, "font-size": 10,
      });
      t.textContent = s.label;
      svg.appendChild(t);
      lx += 22 + String(s.label).length * 6 + 14;
    }
  }

  // Per-run: one path + dots per series.
  for (const run of runs) {
    for (const s of seriesList) {
      const pts = run.levels
        .filter((lv) => s.accessor(lv) !== null && s.accessor(lv) !== undefined)
        .map((lv) => ({ x: xScale(lv.level), y: yScale(s.accessor(lv)) }));
      if (pts.length === 0) continue;
      if (pts.length > 1) {
        const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
        svg.appendChild(svgEl("path", {
          d, fill: "none", stroke: run.color, "stroke-width": 2,
          "stroke-linejoin": "round", "stroke-linecap": "round",
          "stroke-dasharray": s.dashed ? "6 3" : "none",
        }));
      }
      for (const p of pts) {
        svg.appendChild(svgEl("circle", {
          cx: p.x, cy: p.y, r: 3,
          fill: run.color, stroke: "var(--surface-1)", "stroke-width": 1.5,
        }));
      }
    }
    // Knee marker — drawn on the primary series; always the knee color,
    // regardless of the run's series hue, so "where it bent" reads
    // consistently across overlaid runs.
    if (opts.showKnee && run.knee) {
      const kneeLv = run.levels.find((lv) => lv.level === run.knee.level);
      if (kneeLv && primary(kneeLv) !== null) {
        const kx = xScale(run.knee.level);
        const ky = yScale(primary(kneeLv));
        svg.appendChild(svgEl("line", {
          class: "cb-knee-line",
          x1: kx, x2: kx,
          y1: PADDING.top, y2: height - PADDING.bottom,
        }));
        svg.appendChild(svgEl("circle", {
          class: "cb-knee-ring",
          cx: kx, cy: ky, r: 6,
        }));
      }
    }
  }
}

function drawEmptyState(svg, width, height, msg) {
  const t = svgEl("text", {
    class: "cb-empty",
    x: width / 2, y: height / 2,
    "text-anchor": "middle", "font-size": 12,
  });
  t.textContent = msg;
  svg.appendChild(t);
}

// ---------------------------------------------------------------------
// SVG: dual-axis panel (in-flight peak bars + processing overhead line)
// ---------------------------------------------------------------------

function drawLitellmPanel(svg, { width, height }, runs) {
  clearSvg(svg);
  if (runs.length === 0 || runs.every((r) => r.levels.length === 0)) {
    drawEmptyState(svg, width, height, "no data yet");
    return null;
  }
  const allLevels = new Set();
  for (const run of runs) for (const lv of run.levels) allLevels.add(lv.level);
  const xLevels = Array.from(allLevels).sort((a, b) => a - b);
  const xDomain = xLevels.length === 1
    ? [xLevels[0] * 0.5, xLevels[0] * 1.5]
    : [Math.min(...xLevels), Math.max(...xLevels)];
  const xScale = scaleLinear(xDomain, [PADDING.left + 8, width - PADDING.right - 8]);

  const inFlightDomain = extent(runs, (lv) => lv.delta ? lv.delta.in_flight_peak : null);
  const overheadDomain = extent(runs, (lv) => lv.delta ? lv.delta.proc_overhead_s : null);
  const inFlightScale = scaleLinear(
    [0, inFlightDomain[1] * 1.1 || 1],
    [height - PADDING.bottom, PADDING.top]
  );
  const overheadScale = scaleLinear(
    [0, (overheadDomain[1] || 0.001) * 1.2],
    [height - PADDING.bottom, PADDING.top]
  );

  drawAxes(svg, {
    xScale, yScale: inFlightScale,
    xTicks: xLevels,
    yTicks: niceTicks([0, inFlightDomain[1] * 1.1 || 1]),
    xLabel: "agents (concurrency)",
    yLabel: "in-flight peak",
    yFormat: (v) => fmt(v, 0),
    width, height,
  });
  // Right axis for overhead (colored to match the overhead line).
  const ohTicks = niceTicks([0, (overheadDomain[1] || 0.001) * 1.2]);
  for (const tv of ohTicks) {
    const y = overheadScale(tv);
    if (y < PADDING.top - 2 || y > height - PADDING.bottom + 2) continue;
    const lbl = svgEl("text", {
      x: width - PADDING.right + 6, y: y + 3,
      "text-anchor": "start", "font-size": 10, fill: OVERHEAD_COLOR,
      "font-family": "var(--mono)",
    });
    lbl.textContent = fmt(tv, 3);
    svg.appendChild(lbl);
  }

  const plotLeft = PADDING.left + 8;
  const plotRight = width - PADDING.right - 8;
  const barWidth = barWidthForLevels(xLevels, xScale, plotLeft, plotRight);
  for (const run of runs) {
    // Bars: in-flight peak.
    for (const lv of run.levels) {
      if (!lv.delta) continue;
      const x = clampBarX(xScale(lv.level), barWidth, plotLeft, plotRight);
      const yTop = inFlightScale(lv.delta.in_flight_peak);
      const yBase = inFlightScale(0);
      svg.appendChild(svgEl("rect", {
        x, y: yTop, width: barWidth, height: Math.max(0, yBase - yTop),
        fill: run.color, opacity: 0.35,
      }));
    }
    // Line: proc overhead.
    const pts = run.levels
      .filter((lv) => lv.delta && lv.delta.proc_overhead_s !== null)
      .map((lv) => ({ x: xScale(lv.level), y: overheadScale(lv.delta.proc_overhead_s) }));
    if (pts.length > 1) {
      const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
      svg.appendChild(svgEl("path", {
        d, fill: "none", stroke: OVERHEAD_COLOR, "stroke-width": 2,
        "stroke-linejoin": "round", "stroke-dasharray": "5 3",
      }));
    }
    for (const p of pts) {
      svg.appendChild(svgEl("circle", {
        cx: p.x, cy: p.y, r: 3,
        fill: OVERHEAD_COLOR, stroke: "var(--surface-1)", "stroke-width": 1.5,
      }));
    }
  }

  // Heuristic annotation (FR-16): low in-flight + high latency ⇒ pre-handler
  // queueing LiteLLM can't see. Detect across the run's levels.
  return detectPreHandlerQueueing(runs);
}

function detectPreHandlerQueueing(runs) {
  // Heuristic: if in_flight_peak stays low (below 2× the lowest level) but
  // p99 latency climbs steeply (≥2× between adjacent levels), there's likely
  // pre-handler queueing that LiteLLM's queue_time metric doesn't capture.
  // Returns a string describing the suspect level, or null.
  const flags = [];
  for (const run of runs) {
    if (run.levels.length < 2) continue;
    const inflightValues = run.levels
      .filter((lv) => lv.delta)
      .map((lv) => ({ level: lv.level, inf: lv.delta.in_flight_peak, lat: lv.delta.lat_p99 }))
      .sort((a, b) => a.level - b.level);
    for (let i = 1; i < inflightValues.length; i++) {
      const prev = inflightValues[i - 1];
      const curr = inflightValues[i];
      if (prev.lat > 0 && curr.lat >= prev.lat * 2 && curr.inf < prev.level * 1.5) {
        flags.push({ run_id: run.run_id, level: curr.level, lat: curr.lat, inf: curr.inf });
      }
    }
  }
  return flags.length > 0 ? flags : null;
}

// ---------------------------------------------------------------------
// SVG: outcome taxonomy (stacked-ish bars per level)
// ---------------------------------------------------------------------

function drawTaxonomy(svg, { width, height }, runs) {
  clearSvg(svg);
  if (runs.length === 0 || runs.every((r) => r.levels.length === 0)) {
    drawEmptyState(svg, width, height, "no data yet");
    return;
  }
  // Use the first run (or active) for the taxonomy panel; multiple runs would
  // make this unreadable. Caller renders a legend per run anyway.
  const run = runs[0];
  const levels = run.levels.slice().sort((a, b) => a.level - b.level);
  const allLevels = levels.map((lv) => lv.level);
  const xDomain = allLevels.length === 1 ? [allLevels[0] * 0.5, allLevels[0] * 1.5] : [Math.min(...allLevels), Math.max(...allLevels)];
  const xScale = scaleLinear(xDomain, [PADDING.left + 12, width - PADDING.right - 12]);

  // Max total outcomes determines y domain.
  let maxTotal = 0;
  for (const lv of levels) {
    const total = Object.values(lv.outcome_counts || {}).reduce((a, b) => a + b, 0);
    if (total > maxTotal) maxTotal = total;
  }
  const yDomain = [0, Math.max(1, maxTotal)];
  const yScale = scaleLinear(yDomain, [height - PADDING.bottom, PADDING.top]);

  drawAxes(svg, {
    xScale, yScale,
    xTicks: allLevels,
    yTicks: niceTicks(yDomain, 4),
    xLabel: "agents (concurrency)",
    yLabel: "tasks",
    xFormat: (v) => v,
    yFormat: (v) => fmt(v, 0),
    width, height,
  });

  const plotLeft = PADDING.left + 12;
  const plotRight = width - PADDING.right - 12;
  const barWidth = barWidthForLevels(allLevels, xScale, plotLeft, plotRight);
  for (const lv of levels) {
    let yBase = yScale(0);
    // Stack in OUTCOME_ORDER so resolved is on the bottom (positive outcome
    // on the baseline) and timeout on top.
    for (const oc of OUTCOME_ORDER) {
      const count = (lv.outcome_counts || {})[oc] || 0;
      if (count === 0) continue;
      const yTop = yScale(count);
      svg.appendChild(svgEl("rect", {
        x: clampBarX(xScale(lv.level), barWidth, plotLeft, plotRight),
        y: yTop,
        width: barWidth,
        height: Math.max(0, yBase - yTop),
        fill: OUTCOME_COLORS[oc],
        opacity: 0.85,
      }));
      yBase = yTop;
    }
  }

  // Legend.
  let lx = PADDING.left + 4;
  const ly = PADDING.top + 2;
  for (const oc of OUTCOME_ORDER) {
    svg.appendChild(svgEl("rect", {
      x: lx, y: ly, width: 10, height: 10,
      fill: OUTCOME_COLORS[oc], opacity: 0.85,
    }));
    const t = svgEl("text", {
      class: "cb-tick",
      x: lx + 14, y: ly + 9, "font-size": 10,
    });
    t.textContent = oc;
    svg.appendChild(t);
    lx += 14 + oc.length * 6 + 12;
  }
}

// ---------------------------------------------------------------------
// DOM: per-level table
// ---------------------------------------------------------------------

function renderLevelTable() {
  const tbody = document.getElementById("level-table-body");
  if (!tbody) return;
  tbody.innerHTML = "";
  const runs = runsToRender();
  for (const run of runs) {
    for (const lv of run.levels.slice().sort((a, b) => a.level - b.level)) {
      const tr = document.createElement("tr");
      if (run.knee && run.knee.level === lv.level) tr.classList.add("knee");
      const cells = [
        run.name,
        lv.level,
        ttftCell(lv, "p50"),
        ttftCell(lv, "p95"),
        latencyCell(lv, "p50"),
        latencyCell(lv, "p95"),
        latencyCell(lv, "p99"),
        fmt(lv.delta ? lv.delta.throughput_tps : null, 1),
        fmt((lv.pass_rate || 0) * 100, 1) + "%",
        fmt((lv.delta ? lv.delta.error_rate || 0 : 0) * 100, 1) + "%",
        fmt(lv.delta ? lv.delta.in_flight_peak : null, 0),
        outcomesCell(lv.outcome_counts || {}),
        fmt(lv.duration_s || 0, 2),
      ];
      for (const c of cells) {
        const td = document.createElement("td");
        if (c && typeof c === "object" && c.__html) {
          td.innerHTML = c.__html;
        } else {
          td.textContent = c;
        }
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
  }
}

function ttftCell(lv, key) {
  if (!lv.delta || lv.delta.ttft_p50 === null) {
    return { __html: '<span class="ttft-na">n/a</span>' };
  }
  return key === "p50" ? fmt(lv.delta.ttft_p50) : fmt(lv.delta.ttft_p95);
}

function latencyCell(lv, key) {
  if (!lv.delta) return "—";
  const v = lv.delta[`lat_${key}`];
  return fmt(v);
}

function outcomesCell(counts) {
  const parts = OUTCOME_ORDER
    .filter((oc) => (counts[oc] || 0) > 0)
    .map((oc) => `<span style="color:${OUTCOME_COLORS[oc]}">${counts[oc]} ${oc}</span>`);
  return { __html: parts.join(", ") || "—" };
}

// ---------------------------------------------------------------------
// DOM: event feed + active run + overlay legend
// ---------------------------------------------------------------------

function renderEventFeed() {
  const el = document.getElementById("event-feed");
  if (!el) return;
  el.innerHTML = "";
  const recent = state.events.slice(-MAX_EVENTS).reverse();
  for (const ev of recent) {
    const row = document.createElement("div");
    // ev-<type> drives the per-event-type hue (see styles.css).
    row.className = `event ev-${ev.type}`;
    const t = document.createElement("span");
    t.className = "t"; t.textContent = ev.t;
    const ty = document.createElement("span");
    ty.className = "type"; ty.textContent = ev.type;
    const p = document.createElement("span");
    p.className = "payload";
    p.textContent = JSON.stringify(ev.payload);
    row.appendChild(t); row.appendChild(ty); row.appendChild(p);
    el.appendChild(row);
  }
}

function showActiveRun() {
  const wrap = document.getElementById("active-run");
  if (!wrap) return;
  wrap.classList.remove("hidden");
  renderActiveRunMeta();
}

function renderActiveRunMeta() {
  const el = document.getElementById("active-run-meta");
  if (!el || !state.activeRun) return;
  const r = state.activeRun;
  el.innerHTML = "";
  const fields = [
    ["run", r.run_id, false],
    ["name", r.name || "—", false],
    ["mode", r.mode, false],
    ["levels", (r.levels || []).join(", "), false],
    ["pinned", `${(r.pinned_instance_ids || []).length}`, false],
    ["done", `${(r.levels_data || []).length}/${(r.levels || []).length}`, false],
    ["knee", r.knee ? `L${r.knee.level} · ${r.knee.reason}` : "—", !!r.knee],
  ];
  for (const [k, v, isKnee] of fields) {
    const chip = document.createElement("span");
    chip.className = isKnee ? "meta knee" : "meta";
    const kEl = document.createElement("span");
    kEl.className = "k"; kEl.textContent = k;
    const vEl = document.createElement("span");
    vEl.className = "v"; vEl.textContent = v;
    chip.appendChild(kEl); chip.appendChild(vEl);
    el.appendChild(chip);
  }
}

function renderLiveGauges(stats) {
  const wrap = document.getElementById("live-gauges");
  if (!wrap) return;
  wrap.classList.remove("hidden");

  // Streaming-only metrics (TTFT/TPOT/cache) are not measurable when the agent
  // doesn't stream (real mini-swe-agent). Render "n/a" so a blank gauge reads
  // as "not applicable" rather than "broken / waiting".
  const nonStreaming = !!(state.activeRun && state.activeRun.agent_streams === false);

  function set(id, text) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove("gauge-na");
    el.textContent = text != null ? text : "—";
  }
  function setNa(id) {
    const el = document.getElementById(id);
    if (el) { el.classList.add("gauge-na"); el.textContent = "n/a"; }
  }

  function fmtSec(s) {
    if (s == null) return null;
    return s >= 1 ? s.toFixed(1) + "s" : (s * 1000).toFixed(0) + "ms";
  }

  const inf = stats.in_flight;
  set("g-inflight", inf != null ? inf.toFixed(0) : null);

  const tps = stats.throughput_tps;
  set("g-tps", tps != null && tps > 0 ? tps.toFixed(0) : null);

  // These are meaningful regardless of streaming:
  set("g-lat",  fmtSec(stats.lat_p50));
  set("g-queue", fmtSec(stats.queue_p50));
  set("g-overhead", fmtSec(stats.overhead_p50));

  if (nonStreaming) {
    setNa("g-ttft"); setNa("g-tpot"); setNa("g-cache");
    return;
  }

  set("g-ttft", fmtSec(stats.ttft_p50));

  const tpot = stats.tpot_ms;
  set("g-tpot", tpot != null ? tpot.toFixed(1) + "ms" : null);

  const cache = stats.cache_misses;
  set("g-cache", cache != null && cache > 0 ? String(cache) : null);
}

function renderOverlayLegend() {
  const el = document.getElementById("overlay-legend");
  if (!el) return;
  el.innerHTML = "";
  if (state.overlayRuns.length === 0) {
    const span = document.createElement("span");
    span.className = "muted";
    span.textContent = state.activeRun
      ? "showing the active run"
      : "no runs loaded — start one or load a saved report";
    el.appendChild(span);
    return;
  }
  for (let i = 0; i < state.overlayRuns.length; i++) {
    const run = state.overlayRuns[i];
    const color = SERIES_COLORS[(i + 1) % SERIES_COLORS.length];
    const item = document.createElement("span");
    item.className = "legend-item";
    item.innerHTML = `<span class="legend-swatch" style="background:${color}"></span>${run.name} (${run.run_id})`;
    el.appendChild(item);
  }
}

function renderQueueAnnotation(flags) {
  const el = document.getElementById("queue-annotation");
  if (!el) return;
  if (!flags) {
    el.textContent = "heuristic: no pre-handler bottleneck detected";
    el.className = "annotation";
    return;
  }
  const parts = flags.map(
    (f) => `${f.run_id} L${f.level} (p99=${f.lat.toFixed(2)}s, in-flight=${f.inf.toFixed(1)})`
  );
  el.textContent = `⚠ pre-handler queueing heuristic: ${parts.join("; ")} — LiteLLM can't see pre-ASGI wait; not GPU causation`;
  el.className = "annotation flagged";
}

function renderTtftHint() {
  const el = document.getElementById("ttft-hint");
  if (!el) return;
  const runs = runsToRender();
  if (runs.length === 0) { el.textContent = ""; return; }
  if (runs.some((r) => r.agent_streams === false)) {
    el.textContent = "— agent non-streaming: TTFT/TPOT/cache-miss not measurable";
    return;
  }
  const anyNa = runs.some((r) => r.ttft_available === false);
  el.textContent = anyNa ? "— TTFT unavailable (streaming off)" : "";
}

function renderKneeNote() {
  const el = document.getElementById("knee-note");
  if (!el) return;
  const runs = runsToRender();
  const withKnee = runs.filter((r) => r.knee);
  if (withKnee.length === 0) { el.textContent = ""; return; }
  const parts = withKnee.map(
    (r) => `${r.name}: level ${r.knee.level} (${r.knee.reason})`
  );
  el.textContent = `knee — ${parts.join("; ")}`;
}

// ---------------------------------------------------------------------
// Top-level render
// ---------------------------------------------------------------------

function renderAll() {
  const runs = runsToRender();
  const W = 960, H = 320;
  drawSeriesChart(
    document.getElementById("chart-ttft"),
    { width: W, height: H },
    runs,
    (lv) => lv.delta ? lv.delta.ttft_p95 : null,
    {
      yLabel: "TTFT (s)", yDigits: 3, emptyMessage: "no TTFT data",
      series: [
        { accessor: (lv) => lv.delta ? lv.delta.ttft_p50 : null, label: "p50", dashed: false },
        { accessor: (lv) => lv.delta ? lv.delta.ttft_p95 : null, label: "p95", dashed: true },
      ],
    }
  );
  drawSeriesChart(
    document.getElementById("chart-latency"),
    { width: W, height: H },
    runs,
    (lv) => lv.delta ? lv.delta.lat_p99 : null,
    { yLabel: "latency p99 (s)", yDigits: 2, showKnee: true }
  );
  drawSeriesChart(
    document.getElementById("chart-saturation"),
    { width: W, height: H },
    runs,
    (lv) => lv.delta ? lv.delta.throughput_tps : null,
    { yLabel: "tokens/sec", yDigits: 0, showKnee: true }
  );
  const flags = drawLitellmPanel(
    document.getElementById("chart-litellm"),
    { width: W, height: H },
    runs
  );
  drawTaxonomy(
    document.getElementById("chart-taxonomy"),
    { width: W, height: 280 },
    runs
  );
  renderLevelTable();
  renderEventFeed();
  renderActiveRunMeta();
  renderOverlayLegend();
  renderQueueAnnotation(flags);
  renderTtftHint();
  renderKneeNote();
}

// ---------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------

function initTabs() {
  const nav = document.querySelector(".tab-nav");
  if (!nav) return;
  nav.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    const tab = btn.dataset.tab;
    // Update tab buttons.
    nav.querySelectorAll(".tab").forEach((b) => {
      b.classList.toggle("active", b.dataset.tab === tab);
      b.setAttribute("aria-selected", b.dataset.tab === tab ? "true" : "false");
    });
    // Update panels.
    document.querySelectorAll(".tab-panel").forEach((p) => {
      p.classList.toggle("active", p.dataset.tab === tab);
    });
  });
}

function wire() {
  const form = document.getElementById("start-form");
  if (form) form.addEventListener("submit", startRun);
  // Keep the top readout in sync with the per-run form fields as they're edited.
  if (form) {
    form.scrape_interval_s.addEventListener("input", syncLiveReadout);
    form.streaming.addEventListener("change", syncLiveReadout);
    form.model.addEventListener("input", syncLiveReadout);
  }
  const stopBtn = document.getElementById("stop-btn");
  if (stopBtn) stopBtn.addEventListener("click", stopRun);
  const loadBtn = document.getElementById("load-btn");
  if (loadBtn) loadBtn.addEventListener("click", loadSelectedRuns);
  const clearBtn = document.getElementById("clear-btn");
  if (clearBtn) clearBtn.addEventListener("click", clearOverlay);
  const refreshBtn = document.getElementById("refresh-btn");
  if (refreshBtn) refreshBtn.addEventListener("click", refreshSavedList);

  initTabs();
  loadConfig();
  refreshSavedList();
  connect();
  renderAll();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", wire);
} else {
  wire();
}

})();
