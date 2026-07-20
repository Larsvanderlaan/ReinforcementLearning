from __future__ import annotations

import sys

from occupancy_ratio_benchmark.calibration_supervisor import (
    classify_returncode,
    run_supervised_units,
    worker_environment,
)


def _manifest():
    return {"run_id": "occ-cal-supervisor", "provenance": {"fingerprint": "a" * 64}}


def _unit(name):
    return {"unit_id": name}


def test_supervisor_runs_two_atomic_children_and_forces_thread_limits(tmp_path) -> None:
    completed = set()

    def command(unit, attempt):
        completed.add(unit["unit_id"])
        return [sys.executable, "-c", "import sys; sys.exit(0)"]

    outcomes = run_supervised_units(
        units=[_unit("a"), _unit("b")],
        manifest=_manifest(),
        run_root=tmp_path,
        command_builder=command,
        completion_check=lambda unit: unit["unit_id"] in completed,
        maximum_concurrency=2,
        timeout_sec=2.0,
        heartbeat_interval_sec=0.1,
        stale_after_sec=0.3,
        maximum_transient_retries=0,
        poll_interval_sec=0.01,
    )
    assert [row.status for row in outcomes] == ["ok", "ok"]
    environment = worker_environment()
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"


def test_signal_style_exit_is_worker_lost_and_timeout_is_exact_unit_scope(tmp_path) -> None:
    lost = run_supervised_units(
        units=[_unit("lost")],
        manifest=_manifest(),
        run_root=tmp_path / "lost",
        command_builder=lambda unit, attempt: [sys.executable, "-c", "import sys; sys.exit(139)"],
        completion_check=lambda unit: False,
        maximum_concurrency=1,
        timeout_sec=2.0,
        heartbeat_interval_sec=0.1,
        stale_after_sec=0.3,
        maximum_transient_retries=0,
        poll_interval_sec=0.01,
    )
    assert lost[0].failure_type == "worker_lost"
    assert classify_returncode(139) == ("worker_lost", True)

    timed = run_supervised_units(
        units=[_unit("slow")],
        manifest=_manifest(),
        run_root=tmp_path / "slow",
        command_builder=lambda unit, attempt: [sys.executable, "-c", "import time; time.sleep(2)"],
        completion_check=lambda unit: False,
        maximum_concurrency=1,
        timeout_sec=0.1,
        heartbeat_interval_sec=0.03,
        stale_after_sec=0.09,
        maximum_transient_retries=0,
        poll_interval_sec=0.01,
    )
    assert timed[0].failure_type == "timeout"
