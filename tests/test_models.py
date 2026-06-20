"""Gate 0 (part 1): `import clusterbench.models` works; RunConfig + RunReport
round-trip through to_dict/from_dict; enums and DegradationGuard behave."""
from clusterbench.models import (
    OUTCOME_PRIORITY,
    DegradationGuard,
    LevelDelta,
    LevelSummary,
    LoadMode,
    MiniSweConfig,
    Outcome,
    RunConfig,
    RunReport,
    TaskOutcome,
    TaskSlice,
)


def test_import_ok():
    import clusterbench.models as m

    assert m.__name__ == "clusterbench.models"


def test_loadmode_values():
    assert LoadMode.SWEEP.value == "sweep"
    assert LoadMode.SOAK.value == "soak"


def test_outcome_priority_order():
    # FR-19: timeout > agent_error > inference_error > unresolved > resolved.
    assert OUTCOME_PRIORITY == (
        Outcome.TIMEOUT,
        Outcome.AGENT_ERROR,
        Outcome.INFERENCE_ERROR,
        Outcome.UNRESOLVED,
        Outcome.RESOLVED,
    )


def test_runconfig_roundtrip_default():
    cfg = RunConfig(run_id="r1")
    assert RunConfig.from_dict(cfg.to_dict()) == cfg


def test_runconfig_roundtrip_populated():
    cfg = RunConfig(
        run_id="r2",
        name="probe",
        mode=LoadMode.SOAK,
        levels=[16],
        soak_duration_s=600.0,
        task_slice=TaskSlice(n=8, pinned_instance_ids=["a", "b"]),
        guards=DegradationGuard(max_p99_latency_s=2.0, min_pass_rate=0.5),
        scrape_interval_s=0.5,
        miniswe=MiniSweConfig(model="custom", streaming=False, step_limit=10),
        litellm_metrics_url="http://litellm:4000/metrics",
        litellm_base_url="http://litellm:4000/v1",
        vllm_metrics_url=None,
    )
    assert RunConfig.from_dict(cfg.to_dict()) == cfg


def test_runconfig_vllm_optional_is_none_by_default():
    # FR-OPT — vllm_metrics_url is None unless explicitly set.
    assert RunConfig(run_id="r4").vllm_metrics_url is None


def test_runreport_roundtrip_with_levels():
    cfg = RunConfig(run_id="r3")
    delta = LevelDelta(
        level=4,
        ttft_p50=0.05,
        ttft_p95=0.08,
        lat_p50=0.20,
        lat_p95=0.40,
        lat_p99=0.80,
        throughput_tps=125.0,
        n_requests=10,
        error_rate=0.0,
        in_flight_peak=4.0,
        proc_overhead_s=0.02,
    )
    summary = LevelSummary(
        level=4,
        delta=delta,
        n_tasks=5,
        pass_rate=0.6,
        outcome_counts={"resolved": 3, "unresolved": 2},
        duration_s=12.5,
    )
    report = RunReport(
        run_id="r3",
        name="probe",
        config=cfg,
        wire_metrics_available=True,
        ttft_available=True,
        levels=[summary],
        knee={"level": 8, "reason": "p99_latency 2.1s > 2.0s"},
        pinned_instance_ids=["a", "b", "c", "d", "e"],
        finished_at="2026-06-18T12:00:00Z",
    )
    assert RunReport.from_dict(report.to_dict()) == report


def test_taskoutcome_roundtrip():
    o = TaskOutcome(
        run_id="r",
        level=4,
        instance_id="i-1",
        outcome=Outcome.AGENT_ERROR,
        resolved=False,
        return_status=137,
        wall_time_s=42.0,
        timed_out=False,
        log_tail="boom",
    )
    d = o.to_dict()
    assert d["outcome"] == "agent_error"
    assert TaskOutcome.from_dict(d) == o


def test_degradation_guard_no_trip():
    g = DegradationGuard(max_p99_latency_s=1.0, min_pass_rate=0.5)
    assert (
        g.evaluate(
            p99_latency_s=0.5,
            ttft_p95_s=0.1,
            error_rate=0.0,
            pass_rate=0.8,
            in_flight_peak=2.0,
        )
        is None
    )


def test_degradation_guard_trips_p99():
    g = DegradationGuard(max_p99_latency_s=1.0)
    reason = g.evaluate(
        p99_latency_s=1.5,
        ttft_p95_s=None,
        error_rate=0.0,
        pass_rate=1.0,
        in_flight_peak=0.0,
    )
    assert reason is not None and "p99_latency" in reason
