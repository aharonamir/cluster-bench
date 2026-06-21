"""SWE-bench Verified scoring (FR-18).

Stub mode (default, CI): deterministic hash → resolved/not-resolved so re-runs
are reproducible without Docker/GPU/downloads (FR-29).

Real mode (slow): invokes the swebench Verified harness; the runner is
injected so this module doesn't import swebench at module load. Tests use the
stub; the orchestrator wires a real runner behind the `slow` marker.
"""
from __future__ import annotations

import hashlib
from typing import Callable

from clusterbench.models import PredRecord


def _hash_fraction(s: str) -> float:
    """Stable [0, 1) from a string. Makes stub scoring reproducible."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / (1 << 64)


def score_predictions(
    preds: list[PredRecord],
    *,
    mock_pass_rate: float = 0.5,
) -> dict[str, bool]:
    """Score predictions against SWE-bench Verified (FR-18).

    Stub mode (default): for each pred, hash the instance_id to a stable
    value in [0, 1) and resolve if that value is below `mock_pass_rate`.
    Deterministic, no external deps. With pass_rate=0.0 nothing resolves;
    1.0 everything; 0.5 roughly half (depending on instance_id hashes).

    Real mode lives in `score_predictions_real` (slow).
    """
    if not 0.0 <= mock_pass_rate <= 1.0:
        raise ValueError(
            f"mock_pass_rate must be in [0, 1], got {mock_pass_rate}"
        )
    return {
        p.instance_id: _hash_fraction(p.instance_id) < mock_pass_rate
        for p in preds
    }


def score_predictions_real(
    preds: list[PredRecord],
    *,
    subset: str = "verified",
    split: str = "test",
    runner: Callable[..., dict[str, bool]] | None = None,
) -> dict[str, bool]:
    """Real SWE-bench Verified scoring. Slow path — invoke via tests marked
    `slow`; the default suite uses `score_predictions`.

    The runner is injected so this module doesn't import swebench at module
    load. The orchestrator wires a runner that calls into the swebench
    harness (writes preds to a temp file, runs per-instance Docker images,
    parses resolved set).
    """
    if runner is None:
        raise NotImplementedError(
            "real scoring needs a runner (swebench harness); pass one or use "
            "the stub via score_predictions()"
        )
    return runner(preds=preds, subset=subset, split=split)


__all__ = ["score_predictions", "score_predictions_real"]
