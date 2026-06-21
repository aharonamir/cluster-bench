"""Persistence for RunReports (FR-24/FR-25).

One JSON file per run, keyed by run_id, under `results/`. Stable on-disk shape
because `RunReport.to_dict()` round-trips through `RunReport.from_dict()`.

Read paths are tolerant of partial writes — if a JSON parse fails or the file
is missing, the caller gets None and can decide whether to surface it.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from clusterbench.models import RunReport

# Only files matching this pattern are surfaced by list_reports, so co-located
# helpers (e.g. .tmp files, READMEs) don't appear as runs.
RUN_FILE_RE = re.compile(r"^[0-9a-zA-Z_-]+\.json$")


def report_path(results_dir: Path | str, run_id: str) -> Path:
    """Canonical path for one run's report."""
    return Path(results_dir) / f"{run_id}.json"


def save_report(results_dir: Path | str, report: RunReport) -> Path:
    """Persist a RunReport to `<results_dir>/<run_id>.json` (FR-24).

    Atomic write: stage in a sibling tempfile then rename, so a crash mid-write
    never leaves a half-finished file behind for list/retrieve to pick up.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = report_path(results_dir, report.run_id)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=f".{report.run_id}.", suffix=".tmp", dir=str(results_dir)
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(payload)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return path


def load_report(results_dir: Path | str, run_id: str) -> RunReport | None:
    """Read back a report by run_id. Returns None if the file is missing or
    unparseable so the server can 404 rather than 500."""
    path = report_path(results_dir, run_id)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return RunReport.from_dict(raw)
    except (KeyError, TypeError, ValueError):
        return None


def list_reports(results_dir: Path | str) -> list[RunReport]:
    """Return all reports in `results_dir`, newest-first by mtime. Files that
    don't parse are skipped rather than failing the listing."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    candidates = [
        p
        for p in results_dir.iterdir()
        if p.is_file() and RUN_FILE_RE.match(p.name)
    ]
    reports: list[tuple[float, RunReport]] = []
    for p in candidates:
        try:
            raw = json.loads(p.read_text())
            report = RunReport.from_dict(raw)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        reports.append((mtime, report))
    reports.sort(key=lambda pair: pair[0], reverse=True)
    return [r for _, r in reports]


def list_run_ids(results_dir: Path | str) -> list[str]:
    """Convenience: just the run_ids, newest-first."""
    return [r.run_id for r in list_reports(results_dir)]


__all__ = [
    "report_path",
    "save_report",
    "load_report",
    "list_reports",
    "list_run_ids",
]
