"""CLI wrapper around MockRunner for manual / debug invocation (FR-4).

Run with:

    uv run python mock_minisweagent.py \\
        --base-url http://localhost:4000/v1 \\
        --level 4 --n 8 --out-dir results/mock-manual

Drives `level` concurrent workers against mock_litellm for `n` pinned
instances (each doing `--turns` sequential requests), writes preds.json +
per-instance logs to out_dir. Pair with `uv run uvicorn mock_litellm:app
--port 4000` in another terminal.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from clusterbench.miniswerunner import MockRunner, pin_slice


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drive mock_litellm as if a mini-swe-agent batch were running."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:4000/v1",
        help="mock_litellm OpenAI-style base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--level", type=int, default=4, help="concurrency / workers (default: %(default)s)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="number of pinned instances to drive (default: %(default)s)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=3,
        help="sequential requests per instance (default: %(default)s)",
    )
    parser.add_argument(
        "--model", default="mock-model", help="model name to send (default: %(default)s)"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="disable streaming (exercises the TTFT-absent path)",
    )
    parser.add_argument(
        "--out-dir",
        default="results/mock-manual",
        help="where to write preds.json + per-instance logs (default: %(default)s)",
    )
    parser.add_argument(
        "--subset", default="verified", help="subset label for pinned IDs (default: %(default)s)"
    )
    parser.add_argument(
        "--split", default="test", help="split label for pinned IDs (default: %(default)s)"
    )
    args = parser.parse_args()

    instance_ids = pin_slice(
        n=args.n, subset=args.subset, split=args.split, mock=True
    )
    runner = MockRunner(
        base_url=args.base_url,
        instance_ids=instance_ids,
        turns_per_instance=args.turns,
        model=args.model,
        streaming=not args.no_stream,
        runner_root=Path(args.out_dir).parent,
    )

    result = asyncio.run(runner.run(args.level))
    print(
        json.dumps(
            {
                "level": result.level,
                "n_instances": len(instance_ids),
                "n_process_records": len(result.process_records),
                "n_preds": len(result.preds),
                "duration_s": result.duration_s,
                "out_dir": result.out_dir,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
