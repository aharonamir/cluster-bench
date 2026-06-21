"""uvicorn entrypoint for the ClusterBench server.

Defaults to the mock path: expects mock_litellm at --base-url (default
http://localhost:4000/v1). For a real cluster, swap --mock for --real once the
real runner is wired (Phase 6 wires MiniSweRunner behind a flag).

Run with:
  uv run python run_server.py
  uv run python run_server.py --port 8000 --base-url http://localhost:4000/v1
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
        help="OpenAI-style base URL mini-swe-agent's tasks hit (mock_litellm).",
    )
    p.add_argument(
        "--metrics-url",
        default="http://localhost:4000/metrics",
        help="LiteLLM /metrics endpoint to scrape.",
    )
    p.add_argument(
        "--real",
        action="store_true",
        help="Wire the real MiniSweRunner (default: mock path).",
    )
    p.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )
    return p.parse_args()


def build_app(args: argparse.Namespace):
    """Build the FastAPI app from CLI args. Hook for run_server.py + tests."""
    if args.real:
        # Phase 6 will wire the real runner factory behind this flag.
        raise NotImplementedError(
            "real path not wired yet; use the default mock path"
        )
    results_dir = Path(args.results_dir)
    runner_factory = default_runner_factory(
        base_url=args.base_url,
        runner_root=results_dir / "miniswe",
    )

    # Override the default source factory to honor --metrics-url.
    def source_factory(*, config):
        from clusterbench.metrics.litellm import LiteLLMSource

        # Prefer the per-run config URL, fall back to the CLI flag.
        metrics_url = config.litellm_metrics_url or args.metrics_url
        return LiteLLMSource(
            metrics_url=metrics_url,
            scrape_interval_s=config.scrape_interval_s,
        )

    # Pre-fill the default metrics URL by mutating the runner config when the
    # client omits it — but since the client can override per run, we do this
    # in the source factory above.
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
