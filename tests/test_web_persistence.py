"""Tests for clusterbench.web.persistence — save/load/list RunReports."""
from pathlib import Path

from clusterbench.models import LevelSummary, RunConfig, RunReport
from clusterbench.web.persistence import (
    list_reports,
    list_run_ids,
    load_report,
    report_path,
    save_report,
)


def _report(run_id: str = "abc123") -> RunReport:
    return RunReport(
        run_id=run_id,
        name=f"name-{run_id}",
        config=RunConfig(run_id=run_id, levels=[1, 4]),
        wire_metrics_available=True,
        ttft_available=True,
        levels=[LevelSummary(level=1, delta=None, n_tasks=2, pass_rate=0.5)],
        knee={"level": 4, "reason": "p99"},
        pinned_instance_ids=["mock-verified-test-0000"],
        finished_at="2026-06-21T00:00:00Z",
    )


def test_save_then_load_round_trips(tmp_path: Path):
    r = _report()
    path = save_report(tmp_path, r)
    assert path == report_path(tmp_path, r.run_id)
    assert path.is_file()
    loaded = load_report(tmp_path, r.run_id)
    assert loaded is not None
    assert loaded.run_id == r.run_id
    assert loaded.name == r.name
    assert loaded.config.levels == [1, 4]
    assert loaded.knee == {"level": 4, "reason": "p99"}
    assert loaded.pinned_instance_ids == ["mock-verified-test-0000"]
    assert loaded.levels[0].level == 1
    assert loaded.levels[0].delta is None
    assert loaded.levels[0].pass_rate == 0.5


def test_load_returns_none_for_missing_run(tmp_path: Path):
    assert load_report(tmp_path, "nope") is None


def test_load_returns_none_for_corrupt_file(tmp_path: Path):
    """A co-located non-JSON file shouldn't crash load — surface None."""
    path = report_path(tmp_path, "broken")
    path.write_text("{not valid json")
    assert load_report(tmp_path, "broken") is None


def test_save_creates_results_dir_if_missing(tmp_path: Path):
    target = tmp_path / "deeper" / "results"
    assert not target.exists()
    save_report(target, _report())
    assert target.is_dir()


def test_list_reports_returns_newest_first(tmp_path: Path):
    """Multiple saves should list newest-first by mtime."""
    import time

    r1 = _report("run1")
    save_report(tmp_path, r1)
    time.sleep(0.05)  # ensure distinct mtimes
    r2 = _report("run2")
    save_report(tmp_path, r2)

    ids = list_run_ids(tmp_path)
    assert ids == ["run2", "run1"]


def test_list_reports_skips_non_run_files(tmp_path: Path):
    """READMEs, hidden temp files, and other files don't pollute the listing."""
    save_report(tmp_path, _report())
    (tmp_path / "README.md").write_text("not a run")
    (tmp_path / ".partial.tmp").write_text("ignored")
    (tmp_path / "no-extension").write_text("ignored")

    reports = list_reports(tmp_path)
    assert len(reports) == 1
    assert reports[0].run_id == "abc123"


def test_list_reports_empty_when_dir_missing(tmp_path: Path):
    assert list_reports(tmp_path / "nope") == []
    assert list_run_ids(tmp_path / "nope") == []


def test_save_is_atomic_on_write_failure(tmp_path: Path):
    """Even when the write fails mid-flight, no leftover temp file is left
    behind and the canonical path is not created. Verifies the tempfile+rename
    pattern in save_report actually rolls back on error."""
    from unittest.mock import patch

    r = _report()
    real_fdopen = open  # save_import order doesn't matter; we patch by attribute

    # Patch os.fdopen so the wrapped file's .write raises.
    import os

    class _BrokenFile:
        def write(self, _s):
            raise OSError("disk full")

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    with patch("os.fdopen", return_value=_BrokenFile()):
        try:
            save_report(tmp_path, r)
        except OSError:
            pass

    # No leftover .tmp file or canonical .json from the failed save.
    leftovers = list(tmp_path.glob("*.tmp")) + list(
        tmp_path.glob(f"*{r.run_id}*")
    )
    assert leftovers == []
    assert not report_path(tmp_path, r.run_id).exists()
