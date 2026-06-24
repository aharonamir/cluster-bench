"""MiniSweRunner + MockRunner + build_cmd + parse_out_dir + pin_slice.

Covers FR-1..FR-4 (mini-swe-agent batch + mock), FR-7/AC-10 (pinned slice),
FR-12 (streaming for TTFT), FR-17/FR-18 (process records + preds).

The Runner protocol is the seam: the orchestrator only knows the interface,
so swapping real mini-swe-agent for the in-process mock (CI path) is a config
choice.

All mini-swe-agent CLI assumptions live in `build_cmd`; all out_dir layout
assumptions live in `parse_out_dir`. Version drift touches only these.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

import httpx

log = logging.getLogger(__name__)

from clusterbench.models import LevelRunResult, PredRecord, ProcessRecord


# ---------------------------------------------------------------------------
# Pinned slice (T024, FR-7/AC-10)
# ---------------------------------------------------------------------------

def pin_slice(
    *,
    n: int,
    subset: str = "verified",
    split: str = "test",
    mock: bool = True,
    dataset_loader: Callable[..., list[str]] | None = None,
) -> list[str]:
    """Choose N instance_ids once, reused at every sweep level (FR-7/AC-10).

    Mock mode (default): deterministic synthetic IDs so tests are reproducible
    and don't need the SWE-bench dataset on disk. The IDs depend only on
    (n, subset, split) — every level of a sweep gets the same list.

    Real mode (mock=False): invoke `dataset_loader(subset=, split=)` to pull
    the first N IDs from the SWE-bench Verified dataset. The loader is
    injected so this module doesn't depend on swebench at import time; the
    orchestrator wires a real loader behind the `slow` marker.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if mock:
        return [f"mock-{subset}-{split}-{i:04d}" for i in range(n)]
    if dataset_loader is None:
        raise NotImplementedError(
            "real pin_slice needs a dataset_loader; pass one or use mock=True"
        )
    ids = list(dataset_loader(subset=subset, split=split))
    if len(ids) < n:
        raise ValueError(f"requested {n} ids but dataset has {len(ids)}")
    return ids[:n]


# ---------------------------------------------------------------------------
# build_cmd (T020)
# ---------------------------------------------------------------------------

ENV_OPENAI_BASE = "OPENAI_API_BASE"
ENV_OPENAI_KEY = "OPENAI_API_KEY"


_STREAMING_MODEL_CLASS = "clusterbench.streaming_model.StreamingLitellmModel"


def build_cmd(
    *,
    level: int,
    instance_ids: list[str],
    model: str,
    out_dir: Path,
    subset: str = "verified",
    split: str = "test",
    step_limit: int = 0,
    streaming: bool = False,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Construct the `mini-extra swebench` invocation for one level (FR-1..FR-3).

    Workers = concurrent-agent count (FR-5). The model/endpoint is set via
    environment on the subprocess (see `env_for_subprocess`); the cmd itself
    just names the model so mini-swe-agent knows which LiteLLM route to hit.
    Step limit 0 = unlimited.

    When streaming=True, passes --model-class pointing at StreamingLitellmModel
    so every LLM call uses stream=True — making TTFT / cache-miss visible on
    the proxy, matching real coding-agent behaviour (opencode, Claude Code).

    CLI flags verified against minisweagent 2.4.1 swebench.py:
      --filter  : regex matched against instance_id (re.match, anchored at start)
      --workers : concurrency
      --model   : model_name; prefixed with "openai/" so litellm routes through
                  OPENAI_API_BASE (the LiteLLM proxy set in env_for_subprocess)
      -c        : config spec override; used for agent.step_limit
      --redo-existing : always re-run since we clear out_dir before the call
    """
    if level < 1:
        raise ValueError(f"level must be >= 1, got {level}")
    if not instance_ids:
        raise ValueError("instance_ids must not be empty")

    # Build an exact-match regex for our pinned instance IDs.
    # re.match anchors at the start; "$" ensures we don't match a prefix.
    filter_regex = "^(" + "|".join(re.escape(iid) for iid in instance_ids) + ")$"

    # Prefix model with "openai/" so litellm routes through OPENAI_API_BASE
    # (our LiteLLM proxy). If the caller already includes a provider prefix
    # (e.g. "anthropic/..."), leave it as-is.
    model_arg = model if "/" in model else f"openai/{model}"

    cmd: list[str] = [
        "mini-extra", "swebench",
        "--subset", subset,
        "--split", split,
        "--workers", str(level),
        "--filter", filter_regex,
        "--model", model_arg,
        "-o", str(out_dir),
        "--redo-existing",   # we already cleared out_dir; skip stale-check
    ]
    if streaming:
        cmd += ["--model-class", _STREAMING_MODEL_CLASS]
    if step_limit > 0:
        # Passing any -c replaces typer's default [swebench.yaml] list, so we
        # must re-include the base config before our override.
        cmd += ["-c", "swebench.yaml", "-c", f"agent.step_limit={step_limit}"]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def env_for_subprocess(
    *,
    base_url: str,
    api_key: str = "sk-mock",
    ssl_verify: bool = True,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the env for the mini-swe-agent subprocess so its OpenAI client
    hits LiteLLM (FR-3). Streaming is mini-swe-agent's default; we don't
    disable it (FR-12 — TTFT needs streaming).

    ssl_verify=False sets several env vars that disable TLS verification in
    Python HTTP stacks (urllib, requests, httpx via HTTPX_SSL_VERIFY, openai
    via OPENAI_VERIFY_SSL). Needed when LiteLLM uses a corporate CA not
    present inside the container.
    """
    env = dict(os.environ if base is None else base)
    env[ENV_OPENAI_BASE] = base_url
    env[ENV_OPENAI_KEY] = api_key
    # Suppress cost-tracking errors for models not in litellm's price table
    # (e.g. custom LiteLLM routes). A missing price is never a reason to abort.
    env["MSWEA_COST_TRACKING"] = "ignore_errors"
    if not ssl_verify:
        # Cover the main Python HTTP stacks. httpx (used by openai >= 1.x)
        # reads HTTPX_SSL_VERIFY; requests reads REQUESTS_CA_BUNDLE (empty =
        # system default, so set CURL_CA_BUNDLE too for curl-based paths).
        env["HTTPX_SSL_VERIFY"] = "0"
        env["OPENAI_VERIFY_SSL"] = "false"
        env["REQUESTS_CA_BUNDLE"] = ""
        env["CURL_CA_BUNDLE"] = ""
        env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"
    return env


# ---------------------------------------------------------------------------
# Runner protocol — the seam (orchestrator only knows this)
# ---------------------------------------------------------------------------

@runtime_checkable
class Runner(Protocol):
    """Drives one level's batch. Real MiniSweRunner shells out to mini-extra;
    MockRunner drives mock_litellm in-process for CI (FR-4/FR-29)."""

    name: str
    # Whether this runner opens streaming completions, i.e. whether TTFT / TPOT
    # / cache-miss are measurable for its traffic. mini-swe-agent 2.4.x calls
    # litellm.completion() without stream=True and parses the full response, so
    # it is architecturally non-streaming; MockRunner sends stream=True.
    streams: bool

    async def run(self, level: int) -> LevelRunResult:
        ...


# ---------------------------------------------------------------------------
# out_dir parsing (T021) — shared by MiniSweRunner + fixture tests
# ---------------------------------------------------------------------------

def _parse_exit_statuses_yaml(out_dir: Path) -> dict[str, str]:
    """Parse exit_statuses_*.yaml written by mini-swe-agent ≥2.4.

    Returns {instance_id: exit_status_string} (e.g. "TimeoutExpired",
    "Submitted", "LimitsExceeded", "InternalServerError"). Empty dict if no
    file found or yaml unavailable — callers fall back to log scanning only.
    """
    candidates = list(out_dir.glob("exit_statuses_*.yaml"))
    if not candidates or not _YAML_AVAILABLE:
        return {}
    result: dict[str, str] = {}
    for path in candidates:
        try:
            data = _yaml.safe_load(path.read_text()) or {}
        except Exception:
            continue
        by_status = data.get("instances_by_exit_status") or {}
        for status, ids in by_status.items():
            if isinstance(ids, list):
                for iid in ids:
                    result[str(iid)] = str(status)
    return result


def parse_out_dir(out_dir: Path) -> tuple[list[ProcessRecord], list[PredRecord]]:
    """Parse a mini-swe-agent out_dir (FR-17/FR-18).

    Reads:
      - exit_statuses_*.yaml (mini-swe-agent ≥2.4): canonical per-instance
        exit status (TimeoutExpired, Submitted, LimitsExceeded, …). Instances
        here may have no per-instance log directory.
      - preds.json: list of {instance_id, model_patch?, ...}
      - per-instance log/trajectory files for process info (return status,
        wall time, timeout flag, log tail).

    Returns (process_records, preds). Tolerant of missing pieces: if
    preds.json is absent, returns empty preds; instances appearing only in
    logs still get a ProcessRecord.

    The exact out_dir layout depends on the mini-swe-agent version; this
    function is the single place to update if it drifts.
    """
    out_dir = Path(out_dir)

    exit_status_map = _parse_exit_statuses_yaml(out_dir)

    preds: list[PredRecord] = []
    preds_file = out_dir / "preds.json"
    if preds_file.is_file():
        try:
            raw_preds = json.loads(preds_file.read_text())
        except json.JSONDecodeError:
            raw_preds = []
        if isinstance(raw_preds, dict):
            # Some versions key preds by instance_id.
            raw_preds = [
                {"instance_id": k, **(v if isinstance(v, dict) else {})}
                for k, v in raw_preds.items()
            ]
        for entry in raw_preds:
            if not isinstance(entry, dict):
                continue
            instance_id = entry.get("instance_id", "")
            if not instance_id:
                continue
            preds.append(
                PredRecord(
                    instance_id=str(instance_id),
                    model_patch=str(entry.get("model_patch", "") or ""),
                )
            )

    instance_ids: set[str] = {p.instance_id for p in preds}
    # Also include any IDs from the YAML that didn't make it to preds.json.
    instance_ids.update(exit_status_map.keys())
    if not instance_ids:
        for pattern in ("**/*.traj", "**/*.log"):
            for f in out_dir.glob(pattern):
                iid = _instance_id_from_path(f)
                if iid:
                    instance_ids.add(iid)

    process_records: list[ProcessRecord] = []
    for iid in sorted(instance_ids):
        _log_path, tail, timed_out, return_status, wall, inference_error = _scan_instance_logs(
            out_dir, iid
        )
        # exit_statuses_*.yaml is authoritative when present; override log heuristics.
        yaml_status = exit_status_map.get(iid)
        if yaml_status == "TimeoutExpired":
            timed_out = True
        elif yaml_status == "InternalServerError":
            if return_status is None:
                return_status = 1
        process_records.append(
            ProcessRecord(
                instance_id=iid,
                return_status=return_status,
                wall_time_s=wall,
                timed_out=timed_out,
                log_tail=tail,
                inference_error=inference_error,
            )
        )

    return process_records, preds


def _instance_id_from_path(p: Path) -> str | None:
    """Best-effort: extract instance_id from a log/traj filename."""
    stem = p.stem
    return stem.split(".")[0] or None


def _scan_instance_logs(
    out_dir: Path, instance_id: str
) -> tuple[Path | None, str, bool, int | None, float, bool]:
    """Find logs for `instance_id`; surface (log_path, tail, timed_out,
    return_status, wall_time_s, inference_error) best-effort.

    inference_error is per-instance (FR-19): True when the log indicates the
    agent's LLM call failed at the inference layer (LiteLLM/OpenAI/connection
    keywords). The level-granular failed-request rate from LiteLLM stays on
    LevelDelta.error_rate — documented but not auto-promoted here, since
    level-granularity can't be attributed to a specific instance reliably.
    """
    candidates: list[Path] = []
    for pattern in (
        f"**/{instance_id}*.traj",
        f"**/{instance_id}.log",
        f"**/{instance_id}*.log",
        f"**/{instance_id}*",
    ):
        candidates.extend(out_dir.glob(pattern))
    candidates = sorted(
        set(candidates), key=lambda p: (p.suffix != ".traj", str(p))
    )
    if not candidates:
        return None, "", False, None, 0.0, False

    log_path = candidates[0]
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        text = ""
    tail = "\n".join(text.splitlines()[-20:])

    low = text.lower()
    timed_out = any(
        kw in low
        for kw in ("step limit reached", "timed out", "timeout", "hit step limit")
    )

    # Per-instance inference_error (FR-19). Match common LiteLLM/OpenAI
    # transport-error patterns; the level-granular LiteLLM counter is a
    # separate signal on LevelDelta.error_rate.
    inference_error = any(
        kw in low
        for kw in (
            "inference_error",
            "litellm.error",
            "litellm_timeout",
            "openai.error",
            "apiconnectionerror",
            "ratelimiterror",
            "service_unavailable",
            "bad gateway",
            "connection reset",
            "connection refused",
            "connection aborted",
            "connection error",
            "read timeout",
            "remote disconnected",
        )
    )

    return_status: int | None = None
    for line in text.splitlines():
        ll = line.lower()
        if "return_status" in ll or "exit code" in ll or "return code" in ll:
            for tok in reversed(line.split()):
                cleaned = tok.rstrip(",.;:")
                if cleaned.lstrip("-").isdigit():
                    return_status = int(cleaned)
                    break
            if return_status is not None:
                break

    wall_time_s = 0.0
    for line in text.splitlines():
        ll = line.lower()
        if ("wall" in ll and "time" in ll) or "elapsed" in ll:
            for tok in reversed(line.split()):
                try:
                    wall_time_s = float(tok.rstrip("s,.;:"))
                    break
                except ValueError:
                    continue
            if wall_time_s > 0:
                break

    return log_path, tail, timed_out, return_status, wall_time_s, inference_error


# ---------------------------------------------------------------------------
# MiniSweRunner (T021) — real subprocess path
# ---------------------------------------------------------------------------

async def _stop_minisweagent_containers() -> None:
    """Kill any running minisweagent-* Docker containers left by a cancelled run."""
    try:
        list_proc = await asyncio.create_subprocess_exec(
            "docker", "ps", "-q", "--filter", "name=minisweagent-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(list_proc.communicate(), timeout=5)
        container_ids = stdout.decode().split()
        if not container_ids:
            return
        log.info("stopping %d orphaned minisweagent container(s)", len(container_ids))
        stop_proc = await asyncio.create_subprocess_exec(
            "docker", "stop", *container_ids,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(stop_proc.wait(), timeout=30)
    except Exception:
        log.warning("failed to stop minisweagent containers", exc_info=True)


class MiniSweRunner:
    """Real runner: shells out to mini-extra per level (FR-1..FR-3).

    Not exercised by the default test suite — needs mini-swe-agent installed
    + Docker + the SWE-bench dataset (the `real` extra). CI uses MockRunner.
    `parse_out_dir` is unit-tested independently against a fixture.

    `pool` is the full set of available instance IDs. Each call to `run(level)`
    samples `n_per_worker * level` IDs randomly from the pool so each level
    sees a different workload — preventing KV-cache reuse from flattering
    latency at higher concurrency.
    """

    name = "miniswe"
    # streams is set to match the streaming config flag in __init__ so the
    # dashboard can show TTFT/cache-miss when StreamingLitellmModel is active.
    streams: bool = False

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        pool: list[str],
        n_per_worker: int,
        subset: str = "verified",
        split: str = "test",
        step_limit: int = 0,
        streaming: bool = True,
        api_key: str = "sk-mock",
        ssl_verify: bool = True,
        timeout_s: float | None = None,
        runner_root: Path | str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.pool = list(pool)
        self.n_per_worker = n_per_worker
        self.subset = subset
        self.split = split
        self.step_limit = step_limit
        self.streaming = streaming
        self.streams = streaming  # Runner protocol flag — controls dashboard gauges
        self.api_key = api_key
        self.ssl_verify = ssl_verify
        self.timeout_s = timeout_s
        self.runner_root = Path(runner_root) if runner_root else Path("results/miniswe")
        self.runner_root.mkdir(parents=True, exist_ok=True)

    def _sample_instance_ids(self, level: int) -> list[str]:
        total = min(self.n_per_worker * level, len(self.pool))
        return random.sample(self.pool, total)

    def _cmd(self, level: int, out_dir: Path, instance_ids: list[str]) -> list[str]:
        return build_cmd(
            level=level,
            instance_ids=instance_ids,
            model=self.model,
            out_dir=out_dir,
            subset=self.subset,
            split=self.split,
            step_limit=self.step_limit,
            streaming=self.streaming,
        )

    async def run(self, level: int) -> LevelRunResult:
        instance_ids = self._sample_instance_ids(level)
        out_dir = self.runner_root / f"level_{level:04d}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        cmd = self._cmd(level, out_dir, instance_ids)
        env = env_for_subprocess(
            base_url=self.base_url,
            api_key=self.api_key,
            ssl_verify=self.ssl_verify,
        )
        t0 = time.monotonic()
        log.info("mini-swe-agent starting: level=%d cmd=%s", level, " ".join(cmd))
        try:
            # Inherit parent stdout/stderr so mini-swe-agent output appears in
            # `docker logs` in real-time. PIPE would silently swallow everything.
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=None,
                stderr=None,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "mini-extra not found on PATH; install with `uv sync --extra real`"
            ) from exc
        try:
            await asyncio.wait_for(proc.wait(), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            log.warning("mini-swe-agent timed out at level %d, killing", level)
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
        except asyncio.CancelledError:
            log.info("run cancelled: stopping mini-swe-agent and containers (level %d)", level)
            proc.terminate()
            try:
                await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=10)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                proc.kill()
            await _stop_minisweagent_containers()
            raise
        duration = time.monotonic() - t0
        log.info("mini-swe-agent done: level=%d rc=%s duration=%.1fs",
                 level, proc.returncode, duration)

        process_records, preds = parse_out_dir(out_dir)
        return LevelRunResult(
            level=level,
            process_records=process_records,
            preds=preds,
            duration_s=duration,
            out_dir=str(out_dir),
        )


# ---------------------------------------------------------------------------
# MockRunner (T023) — CI engine, drives mock_litellm in-process
# ---------------------------------------------------------------------------

class MockRunner:
    """Emulates a level by driving concurrent multi-turn traffic at
    mock_litellm for the N pinned instances (advancing its counters +
    exercising in_flight), then writes a fake preds.json + per-instance logs
    to out_dir so `parse_out_dir` produces the same shape as the real runner.

    CI engine — no Docker/GPU/downloads (FR-4/FR-29).
    """

    name = "mock"
    # MockRunner sends stream=True, so TTFT/TPOT/cache-miss are measurable.
    streams = True

    def __init__(
        self,
        *,
        base_url: str,
        instance_ids: list[str],
        turns_per_instance: int = 3,
        model: str = "mock-model",
        streaming: bool = True,
        runner_root: Path | str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        # base_url is the OpenAI-style base (ends in /v1).
        self.base_url = base_url
        self.server_url = base_url.rstrip("/").removesuffix("/v1")
        self.instance_ids = list(instance_ids)
        self.turns_per_instance = turns_per_instance
        self.model = model
        self.streaming = streaming
        self.runner_root = Path(runner_root) if runner_root else Path("results/mock")
        self.runner_root.mkdir(parents=True, exist_ok=True)
        self._client = client

    async def run(self, level: int) -> LevelRunResult:
        out_dir = self.runner_root / f"level_{level:04d}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        t0 = time.monotonic()
        await self._drive_level(level, out_dir)
        duration = time.monotonic() - t0

        process_records, preds = parse_out_dir(out_dir)
        return LevelRunResult(
            level=level,
            process_records=process_records,
            preds=preds,
            duration_s=duration,
            out_dir=str(out_dir),
        )

    async def _drive_level(self, level: int, out_dir: Path) -> None:
        sem = asyncio.Semaphore(level)

        async def run_one(client: httpx.AsyncClient, iid: str) -> None:
            async with sem:
                t0 = time.monotonic()
                status_ok = True
                err_msg = ""
                for turn in range(self.turns_per_instance):
                    payload: dict[str, Any] = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": f"mock agent for {iid}"},
                            {"role": "user", "content": f"turn {turn}"},
                        ],
                    }
                    if self.streaming:
                        payload["stream"] = True
                    try:
                        r = await client.post("/v1/chat/completions", json=payload)
                        if r.status_code >= 400:
                            status_ok = False
                            err_msg = f"HTTP {r.status_code}"
                    except httpx.HTTPError as exc:
                        status_ok = False
                        err_msg = str(exc)
                wall = time.monotonic() - t0
                self._write_instance_log(out_dir, iid, wall, status_ok, err_msg)

        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(base_url=self.server_url)
        try:
            await asyncio.gather(
                *(run_one(client, iid) for iid in self.instance_ids)
            )
        finally:
            if owns_client:
                await client.aclose()

        self._write_preds(out_dir)

    def _write_instance_log(
        self,
        out_dir: Path,
        instance_id: str,
        wall_time_s: float,
        status_ok: bool,
        err_msg: str,
    ) -> None:
        return_status = 0 if status_ok else 1
        lines = [
            f"instance_id: {instance_id}",
            f"turns: {self.turns_per_instance}",
            f"wall_time_s: {wall_time_s:.6f}",
            f"return_status: {return_status}",
        ]
        if not status_ok:
            lines.append(f"error: {err_msg}")
        (out_dir / f"{instance_id}.traj").write_text("\n".join(lines) + "\n")

    def _write_preds(self, out_dir: Path) -> None:
        preds = [
            {"instance_id": iid, "model_patch": "", "model_name_or_path": self.model}
            for iid in self.instance_ids
        ]
        (out_dir / "preds.json").write_text(json.dumps(preds, indent=2))


__all__ = [
    "Runner",
    "MiniSweRunner",
    "MockRunner",
    "build_cmd",
    "env_for_subprocess",
    "parse_out_dir",
    "pin_slice",
    "ENV_OPENAI_BASE",
    "ENV_OPENAI_KEY",
]
