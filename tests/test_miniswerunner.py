"""Tests for clusterbench.miniswerunner (T025).

Covers:
  - build_cmd shape (workers, instances, model, out_dir) — FR-1..FR-3.
  - pin_slice determinism + same slice every level — FR-7/AC-10.
  - parse_out_dir on a synthetic fixture — FR-17/FR-18.
  - MockRunner advances mock counters + writes preds (AC-6/AC-10).
  - MockRunner over [2,4] uses identical instance_ids — AC-10.
  - Runner protocol conformance for both runners.
"""
import asyncio
import json
from pathlib import Path

import httpx
import pytest

from clusterbench.miniswerunner import (
    MockRunner,
    Runner,
    build_cmd,
    env_for_subprocess,
    parse_out_dir,
    pin_slice,
)
from clusterbench.models import LevelRunResult
from mock_litellm import (
    INPUT_TOKENS_PER_REQUEST,
    TOTAL_TOKENS_PER_REQUEST,
    create_app,
)


# ---------------------------------------------------------------------------
# build_cmd (T020)
# ---------------------------------------------------------------------------


def test_build_cmd_has_required_flags():
    cmd = build_cmd(
        level=8,
        instance_ids=["a", "b", "c"],
        model="gpt-4o-mini",
        out_dir=Path("/tmp/out"),
    )
    # Program + subcommand first.
    assert cmd[:2] == ["mini-extra", "swebench"]
    # Workers reflect the level (FR-5).
    assert "--workers" in cmd and cmd[cmd.index("--workers") + 1] == "8"
    # Instances passed as an exact-match regex via --filter (not --instances).
    assert "--filter" in cmd
    filter_val = cmd[cmd.index("--filter") + 1]
    import re
    for iid in ["a", "b", "c"]:
        assert re.match(filter_val, iid), f"filter should match {iid!r}"
    assert not re.match(filter_val, "a_extra"), "filter should not match prefix"
    # Model prefixed with openai/ so litellm routes through OPENAI_API_BASE.
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "openai/gpt-4o-mini"
    # out_dir present.
    assert "-o" in cmd and cmd[cmd.index("-o") + 1] == "/tmp/out"
    # Subset/split default to verified/test (FR-2).
    assert "--subset" in cmd and cmd[cmd.index("--subset") + 1] == "verified"
    assert "--split" in cmd and cmd[cmd.index("--split") + 1] == "test"
    # --redo-existing always present.
    assert "--redo-existing" in cmd


def test_build_cmd_model_with_provider_prefix_not_doubled():
    cmd = build_cmd(
        level=1, instance_ids=["x"], model="anthropic/claude-3-5-haiku", out_dir=Path("/o")
    )
    assert cmd[cmd.index("--model") + 1] == "anthropic/claude-3-5-haiku"


def test_build_cmd_step_limit_omitted_when_zero():
    cmd = build_cmd(
        level=1,
        instance_ids=["x"],
        model="m",
        out_dir=Path("/tmp/o"),
        step_limit=0,
    )
    assert "-c" not in cmd or "agent.step_limit" not in " ".join(cmd)


def test_build_cmd_step_limit_emitted_when_positive():
    cmd = build_cmd(
        level=1,
        instance_ids=["x"],
        model="m",
        out_dir=Path("/tmp/o"),
        step_limit=20,
    )
    assert "-c" in cmd
    c_indices = [i for i, tok in enumerate(cmd) if tok == "-c"]
    config_values = [cmd[i + 1] for i in c_indices]
    # swebench.yaml must be included so typer doesn't drop the default config.
    assert any("swebench" in v for v in config_values)
    assert any("agent.step_limit=20" in v for v in config_values)


def test_build_cmd_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        build_cmd(level=0, instance_ids=["x"], model="m", out_dir=Path("/o"))
    with pytest.raises(ValueError):
        build_cmd(level=1, instance_ids=[], model="m", out_dir=Path("/o"))


def test_env_for_subprocess_routes_to_litellm():
    env = env_for_subprocess(base_url="http://litellm:4000/v1", api_key="k")
    assert env["OPENAI_API_BASE"] == "http://litellm:4000/v1"
    assert env["OPENAI_API_KEY"] == "k"


# ---------------------------------------------------------------------------
# pin_slice (T024, AC-10)
# ---------------------------------------------------------------------------


def test_pin_slice_mock_is_deterministic():
    a = pin_slice(n=5, mock=True)
    b = pin_slice(n=5, mock=True)
    assert a == b
    assert len(a) == 5


def test_pin_slice_same_ids_across_levels():
    """AC-10: every sweep level uses the same pinned slice."""
    slice_for_run = pin_slice(n=8, mock=True)
    # The orchestrator would call this once and reuse for every level.
    for level in (1, 4, 8, 16):
        # Same slice used; nothing re-derived per level.
        assert pin_slice(n=8, mock=True) == slice_for_run


def test_pin_slice_mock_ids_are_distinct():
    ids = pin_slice(n=10, mock=True)
    assert len(set(ids)) == 10


def test_pin_slice_subset_split_appear_in_ids():
    ids = pin_slice(n=2, subset="verified", split="test", mock=True)
    assert all("verified" in i and "test" in i for i in ids)


def test_pin_slice_real_without_loader_raises():
    with pytest.raises(NotImplementedError):
        pin_slice(n=5, mock=False)


def test_pin_slice_real_with_loader_uses_first_n():
    seen: dict[str, list[str]] = {}

    def loader(*, subset, split):
        seen["args"] = [subset, split]
        return [f"real-{i}" for i in range(20)]

    ids = pin_slice(n=3, mock=False, dataset_loader=loader)
    assert ids == ["real-0", "real-1", "real-2"]
    assert seen["args"] == ["verified", "test"]


# ---------------------------------------------------------------------------
# parse_out_dir (T021) — fixture-driven
# ---------------------------------------------------------------------------


def _write_fixture(out_dir: Path) -> None:
    """Synthesize a mini-swe-agent-style out_dir: preds.json + per-instance
    trajectory logs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = [
        {"instance_id": "django__django-1234", "model_patch": "diff --git a/..."},
        {"instance_id": "flask__flask-5678", "model_patch": ""},
    ]
    (out_dir / "preds.json").write_text(json.dumps(preds))
    (out_dir / "django__django-1234.traj").write_text(
        "instance_id: django__django-1234\n"
        "turns: 5\n"
        "wall_time_s: 12.5\n"
        "return_status: 0\n"
        "agent: done\n"
    )
    (out_dir / "flask__flask-5678.traj").write_text(
        "instance_id: flask__flask-5678\n"
        "turns: 5\n"
        "step limit reached\n"
        "wall_time_s: 30.0\n"
        "return_status: 124\n"
    )


def test_parse_out_dir_reads_preds_and_process_records(tmp_path):
    _write_fixture(tmp_path)
    process_records, preds = parse_out_dir(tmp_path)

    # preds.json → PredRecord list.
    assert len(preds) == 2
    by_iid = {p.instance_id: p for p in preds}
    assert "django__django-1234" in by_iid
    assert by_iid["django__django-1234"].model_patch.startswith("diff --git")
    assert by_iid["flask__flask-5678"].model_patch == ""

    # Per-instance logs → ProcessRecord list.
    assert len(process_records) == 2
    by_iid_pr = {r.instance_id: r for r in process_records}
    django = by_iid_pr["django__django-1234"]
    assert django.return_status == 0
    assert django.wall_time_s == 12.5
    assert django.timed_out is False
    assert "agent: done" in django.log_tail

    flask = by_iid_pr["flask__flask-5678"]
    assert flask.return_status == 124
    assert flask.wall_time_s == 30.0
    assert flask.timed_out is True  # "step limit reached" detected


def test_parse_out_dir_handles_missing_preds(tmp_path):
    """If preds.json is missing but per-instance logs exist, we still get
    ProcessRecords (instance_ids derived from filenames)."""
    (tmp_path / "solo__case-9.traj").write_text(
        "wall_time_s: 5.0\nreturn_status: 0\n"
    )
    process_records, preds = parse_out_dir(tmp_path)
    assert preds == []
    assert len(process_records) == 1
    assert process_records[0].instance_id == "solo__case-9"
    assert process_records[0].wall_time_s == 5.0


def test_parse_out_dir_handles_empty_dir(tmp_path):
    process_records, preds = parse_out_dir(tmp_path)
    assert process_records == []
    assert preds == []


def test_parse_out_dir_handles_dict_keyed_preds(tmp_path):
    """Some swebench versions key preds.json by instance_id."""
    (tmp_path / "preds.json").write_text(
        json.dumps(
            {
                "django__django-1": {"model_patch": "p"},
                "django__django-2": {"model_patch": ""},
            }
        )
    )
    _process, preds = parse_out_dir(tmp_path)
    assert {p.instance_id for p in preds} == {"django__django-1", "django__django-2"}


def test_parse_out_dir_reads_exit_statuses_yaml(tmp_path):
    """exit_statuses_*.yaml is the authoritative source for timeout/error
    classification; instances listed only in the YAML (no per-instance log)
    still produce ProcessRecords with the correct timed_out/return_status."""
    (tmp_path / "preds.json").write_text(
        json.dumps(
            {
                "pkg__pkg-timeout": {"model_patch": ""},
                "pkg__pkg-error": {"model_patch": ""},
                "pkg__pkg-submitted": {"model_patch": "p"},
            }
        )
    )
    (tmp_path / "exit_statuses_123.yaml").write_text(
        "instances_by_exit_status:\n"
        "    TimeoutExpired:\n"
        "    - pkg__pkg-timeout\n"
        "    InternalServerError:\n"
        "    - pkg__pkg-error\n"
        "    Submitted:\n"
        "    - pkg__pkg-submitted\n"
    )
    records, preds = parse_out_dir(tmp_path)
    by_id = {r.instance_id: r for r in records}
    assert by_id["pkg__pkg-timeout"].timed_out is True
    assert by_id["pkg__pkg-error"].return_status == 1
    assert not by_id["pkg__pkg-submitted"].timed_out
    assert by_id["pkg__pkg-submitted"].return_status in (None, 0)
    assert len(preds) == 3


# ---------------------------------------------------------------------------
# MockRunner (T023) — integration against mock_litellm
# ---------------------------------------------------------------------------


def _client(app=None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app or create_app()),
        base_url="http://test",
    )


def test_mock_runner_writes_preds_and_process_records(tmp_path):
    """MockRunner writes a parseable out_dir; parse_out_dir yields the same
    shape as for the real runner."""

    async def go():
        async with _client() as client:
            runner = MockRunner(
                base_url="http://test/v1",
                instance_ids=["mock-verified-test-0000", "mock-verified-test-0001"],
                turns_per_instance=2,
                client=client,
                runner_root=tmp_path,
            )
            result = await runner.run(level=2)
        return result

    result = asyncio.run(go())
    assert isinstance(result, LevelRunResult)
    assert result.level == 2
    assert len(result.preds) == 2
    assert len(result.process_records) == 2
    assert {r.return_status for r in result.process_records} == {0}
    assert all(r.wall_time_s >= 0 for r in result.process_records)


def test_mock_runner_advances_mock_counters(tmp_path):
    """MockRunner's traffic advances mock_litellm counters; the delta matches
    the expected per-request amounts (AC-4-style exact check at the runner
    boundary)."""

    async def go():
        async with _client() as client:
            # Baseline state.
            before = (await client.get("/__state")).json()
            runner = MockRunner(
                base_url="http://test/v1",
                instance_ids=[
                    "mock-verified-test-0000",
                    "mock-verified-test-0001",
                    "mock-verified-test-0002",
                    "mock-verified-test-0003",
                ],
                turns_per_instance=2,  # 4 instances × 2 turns = 8 requests
                client=client,
                runner_root=tmp_path,
            )
            await runner.run(level=2)
            after = (await client.get("/__state")).json()
        return before, after

    before, after = asyncio.run(go())
    n_requests = 4 * 2
    assert after["requests_200"] - before["requests_200"] == n_requests
    assert after["total_tokens"] - before["total_tokens"] == (
        n_requests * TOTAL_TOKENS_PER_REQUEST
    )
    assert after["input_tokens"] - before["input_tokens"] == (
        n_requests * INPUT_TOKENS_PER_REQUEST
    )


def test_mock_runner_uses_identical_instance_ids_across_levels(tmp_path):
    """AC-10: every level of a sweep uses the same pinned instance_ids."""

    async def go():
        async with _client() as client:
            pinned = pin_slice(n=4, mock=True)
            runner = MockRunner(
                base_url="http://test/v1",
                instance_ids=pinned,
                turns_per_instance=1,
                client=client,
                runner_root=tmp_path,
            )
            r2 = await runner.run(level=2)
            r4 = await runner.run(level=4)
        return pinned, r2, r4

    pinned, r2, r4 = asyncio.run(go())
    ids_2 = {r.instance_id for r in r2.process_records}
    ids_4 = {r.instance_id for r in r4.process_records}
    assert ids_2 == set(pinned)
    assert ids_4 == set(pinned)
    # Preds also pinned.
    assert {p.instance_id for p in r2.preds} == set(pinned)
    assert {p.instance_id for p in r4.preds} == set(pinned)


def test_mock_runner_level_bounds_concurrency(tmp_path):
    """At level=1, the in-flight peak observed on the mock should be ≤ 1; at
    level=4 with 4 instances running one turn each, the peak should reach 4
    (mock holds each request PER_REQUEST_HOLD_S)."""
    from mock_litellm import PER_REQUEST_HOLD_S

    async def measure_peak(level: int, n_instances: int) -> int:
        """Run MockRunner while polling the mock's in_flight; return max seen."""
        peaks: list[int] = []

        async def poll(client: httpx.AsyncClient) -> None:
            while True:
                state = (await client.get("/__state")).json()
                peaks.append(state["in_flight"])
                await asyncio.sleep(PER_REQUEST_HOLD_S / 8)

        async with _client() as client:
            runner = MockRunner(
                base_url="http://test/v1",
                instance_ids=pin_slice(n=n_instances, mock=True),
                turns_per_instance=1,
                client=client,
                runner_root=tmp_path,
            )
            poll_task = asyncio.create_task(poll(client))
            try:
                await asyncio.wait_for(runner.run(level=level), timeout=5.0)
            finally:
                poll_task.cancel()
                try:
                    await poll_task
                except asyncio.CancelledError:
                    pass
        return max(peaks) if peaks else 0

    # Force sequential execution — tmp_path is shared and MockRunner wipes
    # level_*/ on each call, but the mock app is per-call here so counters
    # don't bleed.
    peak_low = asyncio.run(measure_peak(1, 4))
    peak_high = asyncio.run(measure_peak(4, 4))
    # Sanity: higher concurrency observes more in-flight.
    assert peak_high >= peak_low
    assert peak_low <= 1
    assert peak_high >= 2  # at least some overlap with 4 workers


# ---------------------------------------------------------------------------
# Runner protocol conformance
# ---------------------------------------------------------------------------


def test_miniswe_runner_conforms_to_protocol():
    runner = MockRunner(
        base_url="http://test/v1",
        instance_ids=["x"],
    )
    assert isinstance(runner, Runner)


def test_mock_runner_conforms_to_protocol():
    from clusterbench.miniswerunner import MiniSweRunner

    runner = MiniSweRunner(
        model="m",
        base_url="http://test/v1",
        pool=["x", "y", "z"],
        n_per_worker=1,
    )
    assert isinstance(runner, Runner)


# ---------------------------------------------------------------------------
# Gate 2 integration: runner over [2,4] → scored preds (AC-6/AC-10)
# ---------------------------------------------------------------------------


def test_gate2_integration_runner_yields_scored_preds_across_levels(tmp_path):
    """Gate 2 explicit check: mock runner over [2,4] uses identical
    instance_ids, advances counters, and yields preds that the stub scorer
    turns into a resolved set (AC-6/AC-10)."""
    from clusterbench.scoring import score_predictions

    async def go():
        async with _client() as client:
            before = (await client.get("/__state")).json()
            pinned = pin_slice(n=4, mock=True)
            runner = MockRunner(
                base_url="http://test/v1",
                instance_ids=pinned,
                turns_per_instance=2,
                client=client,
                runner_root=tmp_path,
            )
            results: dict[int, LevelRunResult] = {}
            for level in (2, 4):
                results[level] = await runner.run(level)
            after = (await client.get("/__state")).json()
        return pinned, before, after, results

    pinned, before, after, results = asyncio.run(go())

    # AC-10: same ids at both levels.
    for level in (2, 4):
        ids = {r.instance_id for r in results[level].process_records}
        assert ids == set(pinned)
        assert {p.instance_id for p in results[level].preds} == set(pinned)

    # Counters advanced across both levels (8 instances-levels × 2 turns = 16 reqs).
    delta_reqs = after["requests_200"] - before["requests_200"]
    assert delta_reqs == 4 * 2 * 2  # 4 instances × 2 turns × 2 levels

    # AC-6: preds are scoreable; resolved set is per-level, deterministic.
    for level in (2, 4):
        resolved = score_predictions(results[level].preds, mock_pass_rate=0.5)
        assert set(resolved.keys()) == set(pinned)
        # Same pinned slice ⇒ same resolved decisions at both levels.
    assert score_predictions(results[2].preds) == score_predictions(results[4].preds)
