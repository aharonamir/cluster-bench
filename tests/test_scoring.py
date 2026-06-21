"""Tests for clusterbench.scoring (T025).

Covers:
  - Stub scoring is deterministic (AC-6 stub).
  - Pass-rate extremes: 0.0 → nothing resolved; 1.0 → all resolved.
  - Distribution at 0.5 is plausible (roughly half).
  - Real path raises without a runner (slow marker on a wired test).
"""
import pytest

from clusterbench.models import PredRecord
from clusterbench.scoring import score_predictions, score_predictions_real


def _preds(n: int = 20) -> list[PredRecord]:
    return [PredRecord(instance_id=f"mock-verified-test-{i:04d}") for i in range(n)]


# ---------------------------------------------------------------------------
# Stub scoring (default path)
# ---------------------------------------------------------------------------


def test_stub_scoring_is_deterministic():
    preds = _preds(10)
    a = score_predictions(preds, mock_pass_rate=0.5)
    b = score_predictions(preds, mock_pass_rate=0.5)
    assert a == b


def test_stub_scoring_zero_pass_rate_resolves_none():
    preds = _preds(10)
    resolved = score_predictions(preds, mock_pass_rate=0.0)
    assert not any(resolved.values())


def test_stub_scoring_full_pass_rate_resolves_all():
    preds = _preds(10)
    resolved = score_predictions(preds, mock_pass_rate=1.0)
    assert all(resolved.values())


def test_stub_scoring_returns_one_entry_per_pred():
    preds = _preds(7)
    resolved = score_predictions(preds, mock_pass_rate=0.5)
    assert set(resolved.keys()) == {p.instance_id for p in preds}


def test_stub_scoring_mid_rate_is_plausible():
    """At pass_rate=0.5, with 100 distinct instance_ids, the resolved fraction
    should be in a plausible range — not all, not none. (Deterministic, so
    the exact fraction is fixed; just guarding against pathological hash.)"""
    preds = _preds(100)
    resolved = score_predictions(preds, mock_pass_rate=0.5)
    fraction = sum(resolved.values()) / len(preds)
    # Generous bounds — the hash is uniform, so this should be very close
    # to 0.5, but we don't want a brittle test.
    assert 0.3 < fraction < 0.7


def test_stub_scoring_rejects_invalid_pass_rate():
    with pytest.raises(ValueError):
        score_predictions(_preds(3), mock_pass_rate=-0.1)
    with pytest.raises(ValueError):
        score_predictions(_preds(3), mock_pass_rate=1.5)


def test_stub_scoring_independent_of_pred_order():
    preds = _preds(10)
    a = score_predictions(preds, mock_pass_rate=0.5)
    b = score_predictions(list(reversed(preds)), mock_pass_rate=0.5)
    assert a == b


# ---------------------------------------------------------------------------
# Real path (slow) — covered by conformance test, not actually invoked
# ---------------------------------------------------------------------------


def test_real_scoring_without_runner_raises():
    """The slow path is gated — without an injected runner it must refuse
    rather than silently no-op."""
    with pytest.raises(NotImplementedError):
        score_predictions_real(_preds(3))


def test_real_scoring_with_runner_delegates():
    """A wired runner is invoked as expected; the orchestrator injects one
    that hits the swebench harness behind the slow marker."""

    captured: dict[str, object] = {}

    def fake_runner(*, preds, subset, split):
        captured["preds"] = list(preds)
        captured["subset"] = subset
        captured["split"] = split
        return {p.instance_id: True for p in preds}

    preds = _preds(3)
    resolved = score_predictions_real(preds, runner=fake_runner)
    assert all(resolved.values())
    assert captured["subset"] == "verified"
    assert captured["split"] == "test"
    assert len(captured["preds"]) == 3
