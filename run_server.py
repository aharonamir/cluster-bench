"""uvicorn entrypoint for the ClusterBench server.

Two paths:

  mock path (default)
      Drives an in-process mock_litellm via MockRunner. No GPU, Docker, or
      downloads — what CI and the dev loop use.

      uv run python run_server.py

  real path (--real)
      Drives mini-swe-agent's `mini-extra swebench` subprocess per level,
      pointing the agents' OpenAI client at a real LiteLLM at --base-url.
      Requires the `real` extra (`uv sync --extra real`) and Docker (the
      per-task containers mini-swe-agent spawns need the daemon).

      uv run python run_server.py --real --base-url http://litellm:4000/v1

Both paths scrape the same /metrics endpoint (--metrics-url) for wire-level
deltas; only the runner differs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from clusterbench.web.server import (
    create_app,
    default_runner_factory,
    default_source_factory,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the ClusterBench server (mock path by default)."
    )
    p.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)."
    )
    p.add_argument(
        "--port", type=int, default=8000, help="Bind port (default 8000)."
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Where to persist RunReport JSON files (default ./results).",
    )
    p.add_argument(
        "--base-url",
        default="http://localhost:4000/v1",
        help="OpenAI-style base URL (mock_litellm by default; LiteLLM on real path).",
    )
    p.add_argument(
        "--metrics-url",
        default="http://localhost:4000/metrics",
        help="LiteLLM /metrics endpoint to scrape.",
    )
    p.add_argument(
        "--real",
        action="store_true",
        help="Wire the real MiniSweRunner (requires --extra real + Docker).",
    )
    p.add_argument(
        "--api-key",
        default="sk-mock",
        help="API key mini-swe-agent passes to LiteLLM (default sk-mock).",
    )
    p.add_argument(
        "--step-limit",
        type=int,
        default=0,
        help="Per-task step limit for mini-swe-agent (0 = unlimited).",
    )
    p.add_argument(
        "--runner-root",
        default=None,
        help="Root directory for per-level out_dir (default <results>/miniswe).",
    )
    p.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )
    return p.parse_args()


def _real_runner_factory(
    *, base_url: str, api_key: str, step_limit: int, runner_root: Path
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
            instance_ids=pinned,
            subset=config.task_slice.subset,
            split=config.task_slice.split,
            step_limit=step_limit or config.miniswe.step_limit,
            streaming=config.miniswe.streaming,
            api_key=api_key,
            runner_root=runner_root,
        )

    return make


def build_app(args: argparse.Namespace):
    """Build the FastAPI app from CLI args. Hook for run_server.py + tests."""
    results_dir = Path(args.results_dir)
    runner_root = Path(args.runner_root) if args.runner_root else results_dir / "miniswe"

    if args.real:
        runner_factory = _real_runner_factory(
            base_url=args.base_url,
            api_key=args.api_key,
            step_limit=args.step_limit,
            runner_root=runner_root,
        )
    else:
        runner_factory = default_runner_factory(
            base_url=args.base_url,
            runner_root=runner_root,
        )

    # Source factory always honors --metrics-url: it's the server-level wire
    # endpoint (the only reachable layer). The API body doesn't expose a
    # per-run metrics URL, so config.litellm_metrics_url is always its dataclass
    # default — the CLI flag is what the operator actually pointed at.
    def source_factory(*, config):
        from clusterbench.metrics.litellm import LiteLLMSource

        return LiteLLMSource(
            metrics_url=args.metrics_url,
            scrape_interval_s=config.scrape_interval_s,
        )

    return create_app(
        results_dir=results_dir,
        runner_factory=runner_factory,
        source_factory=source_factory,
    )


def main() -> None:
    args = parse_args()
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
