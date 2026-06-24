"""Reprocess outcome_counts in saved report JSONs using exit_statuses_*.yaml.

mini-swe-agent ≥2.4 writes exit_statuses_*.yaml per level, but the original
parse_out_dir ignored it — so TimeoutExpired/InternalServerError instances were
misclassified as 'unresolved'. This script re-derives the correct outcome_counts
for each level and rewrites the report JSON in-place (backing up the original).

Usage:
    uv run python scripts/reprocess_outcomes.py [results_dir [miniswe_dir]]

Defaults:
    results_dir = results/
    miniswe_dir = results/miniswe/   (or auto-detected alongside results_dir)

Examples:
    uv run python scripts/reprocess_outcomes.py
    uv run python scripts/reprocess_outcomes.py results.org results.org/miniswe
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# Make sure clusterbench is importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from clusterbench.miniswerunner import parse_out_dir
from clusterbench.orchestrator import resolve_outcome


def _outcome_counts_from_out_dir(
    out_dir: Path, pass_rate: float, n_tasks: int
) -> dict[str, int] | None:
    """Re-derive outcome_counts for one level.

    We don't re-run the swebench scorer here. Instead we estimate the resolved
    set from pass_rate * n_tasks (rounded) — the same count that was produced
    originally. The key improvement is correct timed_out / return_status flags
    from the YAML, which re-classifies timeouts/agent_errors that were
    previously lumped into 'unresolved'.

    Returns None if the out_dir has no YAML (nothing to fix for this level).
    """
    yaml_candidates = list(out_dir.glob("exit_statuses_*.yaml"))
    if not yaml_candidates:
        return None

    process_records, preds = parse_out_dir(out_dir)
    if not process_records:
        return None

    # Recover which instances were resolved from the stored pass_rate.
    # Sort by instance_id (deterministic) and mark the first `resolved_n` as
    # resolved — this matches what the scorer would have produced for a fixed
    # pool (same instances, same order).  It isn't perfect, but it keeps the
    # overall pass_rate intact while correctly splitting non-resolved into
    # timeout / agent_error / unresolved.
    resolved_n = round(pass_rate * n_tasks)
    submitted_ids = {
        r.instance_id
        for r in process_records
        if not r.timed_out and r.return_status in (None, 0)
    }
    # Sort submitted instances to get a stable resolved set.
    resolved_ids = set(sorted(submitted_ids)[:resolved_n])

    counts: dict[str, int] = {}
    for record in process_records:
        resolved = record.instance_id in resolved_ids
        outcome = resolve_outcome(record, resolved=resolved)
        counts[outcome.value] = counts.get(outcome.value, 0) + 1
    return counts


def _find_out_dir(
    miniswe_dir: Path, pinned_ids: list[str], level: int
) -> Path | None:
    """Find the level_XXXX out_dir whose preds match the report's pinned pool."""
    pinned_set = set(pinned_ids)
    candidate = miniswe_dir / f"level_{level:04d}"
    if not candidate.is_dir():
        return None
    preds_file = candidate / "preds.json"
    if not preds_file.is_file():
        return None
    try:
        raw = json.loads(preds_file.read_text())
    except Exception:
        return None
    if isinstance(raw, dict):
        ids = set(raw.keys())
    else:
        ids = {e["instance_id"] for e in raw if isinstance(e, dict) and "instance_id" in e}
    # All preds must be from this report's pinned pool.
    if not ids or not ids.issubset(pinned_set):
        return None
    return candidate


def reprocess(results_dir: Path, miniswe_dir: Path, *, dry_run: bool = False) -> None:
    report_files = sorted(results_dir.glob("*.json"))
    if not report_files:
        print(f"No report JSON files found in {results_dir}")
        return

    for report_path in report_files:
        try:
            report = json.loads(report_path.read_text())
        except Exception as exc:
            print(f"  SKIP {report_path.name}: unreadable ({exc})")
            continue

        levels = report.get("levels", [])
        if not levels:
            continue

        pinned_ids = report.get("pinned_instance_ids", [])
        if not pinned_ids:
            print(f"  SKIP {report_path.name}: no pinned_instance_ids")
            continue

        changed = False
        for lvl in levels:
            level = lvl.get("level")
            pass_rate = lvl.get("pass_rate", 0.0)
            n_tasks = lvl.get("n_tasks", 0)
            if level is None or not n_tasks:
                continue

            out_dir = _find_out_dir(miniswe_dir, pinned_ids, level)
            if out_dir is None:
                continue

            new_counts = _outcome_counts_from_out_dir(out_dir, pass_rate, n_tasks)
            if new_counts is None:
                continue  # no YAML → nothing to fix

            old_counts = lvl.get("outcome_counts", {})
            if old_counts == new_counts:
                continue

            print(
                f"  {report_path.name} level={level}: "
                f"{old_counts} → {new_counts}"
            )
            lvl["outcome_counts"] = new_counts
            changed = True

        if changed and not dry_run:
            backup = report_path.with_suffix(".json.bak")
            shutil.copy2(report_path, backup)
            report_path.write_text(json.dumps(report, indent=2))
            print(f"  → wrote {report_path.name} (backup: {backup.name})")

    if dry_run:
        print("\n[dry-run] no files written")


def main() -> None:
    args = sys.argv[1:]
    results_dir = Path(args[0]) if args else Path("results")
    if len(args) >= 2:
        miniswe_dir = Path(args[1])
    else:
        miniswe_dir = results_dir / "miniswe"

    if not results_dir.is_dir():
        print(f"results_dir not found: {results_dir}")
        sys.exit(1)
    if not miniswe_dir.is_dir():
        print(f"miniswe_dir not found: {miniswe_dir}")
        sys.exit(1)

    print(f"results_dir : {results_dir}")
    print(f"miniswe_dir : {miniswe_dir}")
    print()
    reprocess(results_dir, miniswe_dir)


if __name__ == "__main__":
    main()
