"""uvicorn entrypoint for the ClusterBench server.

Configuration comes from a YAML file (`--config config.yaml`), with individual
CLI flags overriding file values for ad-hoc runs. Precedence:

    dataclass defaults  <  YAML file  <  CLI flags

Two paths:

  mock path (default)
      Drives an in-process mock_litellm via MockRunner. No GPU, Docker, or
      downloads — what CI and the dev loop use.

      uv run python run_server.py
      uv run python run_server.py --config config.yaml

  real path (real: true, or --real)
      Drives mini-swe-agent's `mini-extra swebench` subprocess per level,
      pointing the agents' OpenAI client at a real LiteLLM at base_url.
      Requires the `real` extra (`uv sync --extra real`) and Docker (the
      per-task containers mini-swe-agent spawns need the daemon).

      uv run python run_server.py --config config.yaml

Both paths scrape the same /metrics endpoint (metrics_url) for wire-level
deltas; only the runner differs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from clusterbench.config import ServerConfig
from clusterbench.web.server import create_app, default_runner_factory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the ClusterBench server (mock path by default)."
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file (see config.example.yaml).",
    )
    # All flags default to None so we can tell 'unset' from an explicit value
    # and only override the file when the operator actually passed one.
    p.add_argument("--host", default=None, help="Bind host (default 127.0.0.1).")
    p.add_argument("--port", type=int, default=None, help="Bind port (default 8000).")
    p.add_argument(
        "--results-dir",
        default=None,
        help="Where to persist RunReport JSON files (default ./results).",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-style base URL (mock_litellm by default; LiteLLM on real path).",
    )
    p.add_argument(
        "--metrics-url",
        default=None,
        help="LiteLLM /metrics endpoint to scrape.",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API key mini-swe-agent passes to LiteLLM (default sk-mock).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Default model name (a POST /api/run body may override per run).",
    )
    p.add_argument(
        "--real",
        action="store_true",
        default=None,
        help="Wire the real MiniSweRunner (requires --extra real + Docker).",
    )
    p.add_argument(
        "--step-limit",
        type=int,
        default=None,
        help="Per-task step limit for mini-swe-agent (0 = unlimited).",
    )
    p.add_argument(
        "--runner-root",
        default=None,
        help="Root directory for per-level out_dir (default <results>/miniswe).",
    )
    p.add_argument(
        "--log-level",
        default=None,
        choices=["critical", "error", "warning", "info", "debug"],
    )
    return p.parse_args()


def resolve_config(args: argparse.Namespace) -> ServerConfig:
    """Build the effective ServerConfig: file (if any) overlaid with CLI flags."""
    cfg = ServerConfig.load(args.config) if args.config else ServerConfig()
    # argparse uses dashes; ServerConfig uses underscores.
    overrides = {
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "results_dir": args.results_dir,
        "runner_root": args.runner_root,
        "base_url": args.base_url,
        "metrics_url": args.metrics_url,
        "api_key": args.api_key,
        "model": args.model,
        "real": args.real,  # None unless --real was passed (store_true default None)
        "step_limit": args.step_limit,
    }
    return cfg.merge_overrides(overrides)


def _real_runner_factory(
    *, base_url: str, api_key: str, step_limit: int, runner_root: Path, ssl_verify: bool = True
):
    """Build a RunnerFactory that drives mini-swe-agent per level."""
    try:
        from clusterbench.miniswerunner import MiniSweRunner
    except ImportError as exc:
        raise SystemExit(
            "real path needs mini-swe-agent; install with `uv sync --extra real`"
        ) from exc

    def make(*, config, pinned):
        return MiniSweRunner(
            model=config.miniswe.model,
            base_url=base_url,
            pool=pinned,
            n_per_worker=config.task_slice.n,
            subset=config.task_slice.subset,
            split=config.task_slice.split,
            step_limit=step_limit or config.miniswe.step_limit,
            streaming=config.miniswe.streaming,
            api_key=api_key,
            ssl_verify=ssl_verify,
            runner_root=runner_root,
        )

    return make


def build_app(cfg: ServerConfig):
    """Build the FastAPI app from a resolved ServerConfig. Hook for tests."""
    results_dir = Path(cfg.results_dir)
    runner_root = Path(cfg.runner_root) if cfg.runner_root else results_dir / "miniswe"

    if cfg.real:
        runner_factory = _real_runner_factory(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            step_limit=cfg.step_limit,
            runner_root=runner_root,
            ssl_verify=cfg.ssl_verify,
        )
    else:
        runner_factory = default_runner_factory(
            base_url=cfg.base_url,
            runner_root=runner_root,
        )

    # Source factory always honors cfg.metrics_url: it's the server-level wire
    # endpoint (the only reachable layer). The API body doesn't expose a
    # per-run metrics URL.
    def source_factory(*, config):
        from clusterbench.metrics.litellm import LiteLLMSource

        return LiteLLMSource(
            metrics_url=cfg.metrics_url,
            scrape_interval_s=config.scrape_interval_s,
            ssl_verify=cfg.ssl_verify,
        )

    return create_app(
        results_dir=results_dir,
        runner_factory=runner_factory,
        source_factory=source_factory,
        real=cfg.real,
        ssl_verify=cfg.ssl_verify,
        run_defaults={
            "model": cfg.model,
            "streaming": cfg.streaming,
            "scrape_interval_s": cfg.scrape_interval_s,
            "step_limit": cfg.step_limit,
        },
        # Non-secret wiring for the dashboard's config readout. NOT the api_key.
        server_info={
            "base_url": cfg.base_url,
            "metrics_url": cfg.metrics_url,
            "path": "real" if cfg.real else "mock",
            "results_dir": str(results_dir),
        },
    )


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    app = build_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=cfg.log_level)


if __name__ == "__main__":
    main()
