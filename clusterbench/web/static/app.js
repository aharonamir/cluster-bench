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

// Fixed palette for overlay runs. Index 0 is reserved for the active run.
const SERIES_COLORS = [
  "#2563eb", "#dc2626", "#059669", "#7c3aed",
  "#d97706", "#0891b2", "#be185d", "#65a30d",
];

const OUTCOME_COLORS = {
  resolved: "#059669",
  unresolved: "#2563eb",
  inference_error: "#d97706",
  agent_error: "#dc2626",
  timeout: "#7c3aed",
};

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
};

let _ws = null;
let _wsReconnectTimer = null;

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
  el.querySelector(".label").textContent = s;
}

// ---------------------------------------------------------------------
// Event dispatcher — turn WS frames into state mutations + re-render
// ---------------------------------------------------------------------

function onEvent(type, payload) {
  pushEvent(type, payload);

  switch (type) {
    case "run_start":
      state.activeRun = newRunFromStart(payload);
      showActiveRun();
      break;
    case "run_done":
      // Pin the finished-at + knee; the persisted report (fetched later) is
      // the source of truth, but we update what we have so the UI doesn't
      // go blank waiting for /api/runs/{id}.
      if (state.activeRun) {
        state.activeRun.finished_at = new Date().toISOString();
      }
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
// Transport: fetch (saved reports + start run)
// ---------------------------------------------------------------------

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

  return {
    name: (fd.get("name") || "").toString().trim(),
    mode,
    levels: mode === "soak" ? [levels[0] || 1] : levels,
    soak_duration_s: parseFloat(fd.get("soak_duration_s")) || 1800,
    task_slice: { n: parseInt(fd.get("n"), 10) || 5 },
    scrape_interval_s: parseFloat(fd.get("scrape_interval_s")) || 1,
    model: (fd.get("model") || "gpt-4o-mini").toString(),
    streaming: fd.get("streaming") === "on",
    guards,
  };
}

async function startRun(ev) {
  ev.preventDefault();
  const btn = document.getElementById("start-btn");
  const status = document.getElementById("start-status");
  btn.disabled = true;
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
      status.textContent = `started ${data.run_id}`;
      // Reset the active run view in case there's stale state from a prior run.
      state.activeRun = null;
    } else if (r.status === 409) {
      const data = await r.json();
      status.textContent = `busy — run ${data.detail.active_run_id} is active`;
    } else {
      status.textContent = `error: HTTP ${r.status}`;
    }
  } catch (e) {
    status.textContent = `error: ${e.message}`;
  } finally {
    btn.disabled = false;
  }
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
  //        xFormat, yFormat}
  const w = opts.width, h = opts.height;
  const xFormat = opts.xFormat || ((v) => v);
  const yFormat = opts.yFormat || ((v) => v);
  const axisColor = "#999";

  // Y axis ticks + gridlines.
  for (const tv of opts.yTicks) {
    const y = opts.yScale(tv);
    svg.appendChild(svgEl("line", {
      x1: PADDING.left, x2: w - PADDING.right,
      y1: y, y2: y,
      stroke: "#eee", "stroke-width": 1,
    }));
    const lbl = svgEl("text", {
      x: PADDING.left - 6, y: y + 3,
      "text-anchor": "end", "font-size": 10, fill: axisColor,
    });
    lbl.textContent = yFormat(tv);
    svg.appendChild(lbl);
  }
  // Y axis label.
  if (opts.yLabel) {
    const t = svgEl("text", {
      x: 12, y: h / 2,
      "text-anchor": "middle", "font-size": 10, fill: axisColor,
      transform: `rotate(-90 12 ${h / 2})`,
    });
    t.textContent = opts.yLabel;
    svg.appendChild(t);
  }
  // X axis ticks.
  for (const tv of opts.xTicks) {
    const x = opts.xScale(tv);
    svg.appendChild(svgEl("line", {
      x1: x, x2: x,
      y1: h - PADDING.bottom, y2: h - PADDING.bottom + 4,
      stroke: axisColor, "stroke-width": 1,
    }));
    const lbl = svgEl("text", {
      x: x, y: h - PADDING.bottom + 16,
      "text-anchor": "middle", "font-size": 10, fill: axisColor,
    });
    lbl.textContent = xFormat(tv);
    svg.appendChild(lbl);
  }
  if (opts.xLabel) {
    const t = svgEl("text", {
      x: (w + PADDING.left - PADDING.right) / 2, y: h - 4,
      "text-anchor": "middle", "font-size": 10, fill: axisColor,
    });
    t.textContent = opts.xLabel;
    svg.appendChild(t);
  }
  // Axis baselines.
  svg.appendChild(svgEl("line", {
    x1: PADDING.left, x2: w - PADDING.right,
    y1: h - PADDING.bottom, y2: h - PADDING.bottom,
    stroke: axisColor, "stroke-width": 1,
  }));
  svg.appendChild(svgEl("line", {
    x1: PADDING.left, x2: PADDING.left,
    y1: PADDING.top, y2: h - PADDING.bottom,
    stroke: axisColor, "stroke-width": 1,
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
  // Y domain: union of all accessor values.
  const yDomainRaw = extent(runs, accessor);
  // Pad y by 5% on top so points don't sit on the top edge.
  const yPad = (yDomainRaw[1] - yDomainRaw[0]) * 0.05;
  const yDomain = [Math.min(0, yDomainRaw[0]), yDomainRaw[1] + yPad];
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

  // Per-run: one path + per-point dots.
  for (const run of runs) {
    const pts = run.levels
      .filter((lv) => accessor(lv) !== null && accessor(lv) !== undefined)
      .map((lv) => ({ x: xScale(lv.level), y: yScale(accessor(lv)) }));
    if (pts.length === 0) continue;
    if (pts.length > 1) {
      const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
      svg.appendChild(svgEl("path", {
        d, fill: "none", stroke: run.color, "stroke-width": 1.75,
        "stroke-linejoin": "round", "stroke-linecap": "round",
      }));
    }
    for (const p of pts) {
      svg.appendChild(svgEl("circle", {
        cx: p.x, cy: p.y, r: 3.5,
        fill: run.color, stroke: "#fff", "stroke-width": 1,
      }));
    }
    // Knee marker.
    if (opts.showKnee && run.knee) {
      const kneeLv = run.levels.find((lv) => lv.level === run.knee.level);
      if (kneeLv && accessor(kneeLv) !== null) {
        const kx = xScale(run.knee.level);
        const ky = yScale(accessor(kneeLv));
        svg.appendChild(svgEl("line", {
          x1: kx, x2: kx,
          y1: PADDING.top, y2: height - PADDING.bottom,
          stroke: run.color, "stroke-width": 1, "stroke-dasharray": "4 3", opacity: 0.5,
        }));
        svg.appendChild(svgEl("circle", {
          cx: kx, cy: ky, r: 6,
          fill: "none", stroke: run.color, "stroke-width": 2,
        }));
      }
    }
  }
}

function drawEmptyState(svg, width, height, msg) {
  const t = svgEl("text", {
    x: width / 2, y: height / 2,
    "text-anchor": "middle", "font-size": 12, fill: "#999",
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
  // Right axis for overhead.
  const ohTicks = niceTicks([0, (overheadDomain[1] || 0.001) * 1.2]);
  for (const tv of ohTicks) {
    const y = overheadScale(tv);
    if (y < PADDING.top - 2 || y > height - PADDING.bottom + 2) continue;
    const lbl = svgEl("text", {
      x: width - PADDING.right + 6, y: y + 3,
      "text-anchor": "start", "font-size": 10, fill: "#7c3aed",
    });
    lbl.textContent = fmt(tv, 3);
    svg.appendChild(lbl);
  }

  const barWidth = Math.max(4, (width - PADDING.left - PADDING.right) / (xLevels.length * 4));
  for (const run of runs) {
    // Bars: in-flight peak.
    for (const lv of run.levels) {
      if (!lv.delta) continue;
      const x = xScale(lv.level) - barWidth / 2;
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
        d, fill: "none", stroke: "#7c3aed", "stroke-width": 1.75,
        "stroke-linejoin": "round",
      }));
    }
    for (const p of pts) {
      svg.appendChild(svgEl("circle", {
        cx: p.x, cy: p.y, r: 3,
        fill: "#7c3aed", stroke: "#fff", "stroke-width": 1,
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

  const barWidth = Math.max(8, (width - PADDING.left - PADDING.right) / (allLevels.length * 2.2));
  for (const lv of levels) {
    let yBase = yScale(0);
    // Stack in OUTCOME_ORDER so resolved is on the bottom (positive outcome
    // on the baseline) and timeout on top.
    for (const oc of OUTCOME_ORDER) {
      const count = (lv.outcome_counts || {})[oc] || 0;
      if (count === 0) continue;
      const yTop = yScale(count);
      svg.appendChild(svgEl("rect", {
        x: xScale(lv.level) - barWidth / 2,
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
      x: lx + 14, y: ly + 9, "font-size": 10, fill: "#333",
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
        fmt(lv.delta ? lv.delta.error_rate : null, 3),
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
    row.className = "event";
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
    ["run_id", r.run_id],
    ["name", r.name || "—"],
    ["mode", r.mode],
    ["levels", (r.levels || []).join(", ")],
    ["pinned", `${(r.pinned_instance_ids || []).length} instances`],
    ["levels done", `${(r.levels_data || []).length} / ${(r.levels || []).length}`],
    ["knee", r.knee ? `level ${r.knee.level} (${r.knee.reason})` : "—"],
  ];
  for (const [k, v] of fields) {
    const div = document.createElement("div");
    div.innerHTML = `<strong>${k}:</strong> ${v}`;
    el.appendChild(div);
  }
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
    el.textContent = "queue-time metric: heuristic shows no pre-handler bottleneck";
    el.className = "annotation muted";
    return;
  }
  const parts = flags.map(
    (f) => `${f.run_id} level ${f.level} (p99=${f.lat.toFixed(2)}s, in-flight=${f.inf.toFixed(1)})`
  );
  el.textContent = `⚠ pre-handler queueing heuristic: ${parts.join("; ")} (LiteLLM can't see pre-ASGI wait; not GPU causation)`;
  el.className = "annotation";
  el.style.color = "var(--warning)";
}

function renderTtftHint() {
  const el = document.getElementById("ttft-hint");
  if (!el) return;
  const runs = runsToRender();
  if (runs.length === 0) { el.textContent = ""; return; }
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
  // Headline charts.
  drawSeriesChart(
    document.getElementById("chart-ttft"),
    { width: 480, height: 280 },
    runs,
    (lv) => lv.delta ? lv.delta.ttft_p95 : null,
    { yLabel: "TTFT p95 (s)", yDigits: 3, emptyMessage: "no TTFT data" }
  );
  drawSeriesChart(
    document.getElementById("chart-latency"),
    { width: 480, height: 280 },
    runs,
    (lv) => lv.delta ? lv.delta.lat_p99 : null,
    { yLabel: "latency p99 (s)", yDigits: 2, showKnee: true }
  );
  drawSeriesChart(
    document.getElementById("chart-saturation"),
    { width: 960, height: 280 },
    runs,
    (lv) => lv.delta ? lv.delta.throughput_tps : null,
    { yLabel: "tokens/sec", yDigits: 0, showKnee: true }
  );
  const flags = drawLitellmPanel(
    document.getElementById("chart-litellm"),
    { width: 960, height: 280 },
    runs
  );
  drawTaxonomy(
    document.getElementById("chart-taxonomy"),
    { width: 640, height: 200 },
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

function wire() {
  const form = document.getElementById("start-form");
  if (form) form.addEventListener("submit", startRun);
  const loadBtn = document.getElementById("load-btn");
  if (loadBtn) loadBtn.addEventListener("click", loadSelectedRuns);
  const clearBtn = document.getElementById("clear-btn");
  if (clearBtn) clearBtn.addEventListener("click", clearOverlay);
  const refreshBtn = document.getElementById("refresh-btn");
  if (refreshBtn) refreshBtn.addEventListener("click", refreshSavedList);

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
