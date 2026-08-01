from __future__ import annotations

import pytest

from occupancy_ratio_benchmark.calibration_eta import (
    build_eta_gate,
    eta_execution_limits,
)


def _manifest(count: int, *, family: str, estimator: str, sample_size: int):
    return {
        "atomic_fold_units": [
            {
                "cell": {"benchmark_family": family},
                "identity": {
                    "axis_values": {"sample_size": sample_size},
                    "estimator_id": estimator,
                },
            }
            for _ in range(count)
        ]
    }


def _aggregation_unit(*, family: str, estimator: str, sample_size: int):
    return {
        "cell": {"benchmark_family": family},
        "identity": {
            "axis_values": {"sample_size": sample_size},
            "estimator_id": estimator,
        },
        "operation": {"base_fit_required": True},
    }


def _execution_manifest(
    concurrency: int,
    *,
    aggregation_concurrency: int = 5,
    ceiling: float = 84.0,
):
    return {
        "resolved_config": {
            "execution": {
                "maximum_concurrent_estimator_processes": concurrency,
                "maximum_concurrent_aggregation_processes": aggregation_concurrency,
                "pilot_eta_gate": {
                    "maximum_concurrency": concurrency,
                    "launch_only_if_projected_hours_at_most": ceiling,
                },
            }
        }
    }


def test_eta_execution_limits_are_read_from_confirmatory_manifests() -> None:
    manifests = [_execution_manifest(5), _execution_manifest(5)]
    assert eta_execution_limits(manifests) == (5, 5, 84.0)


def test_eta_execution_limits_reject_worker_model_mismatch() -> None:
    manifest = _execution_manifest(5)
    manifest["resolved_config"]["execution"]["pilot_eta_gate"][
        "maximum_concurrency"
    ] = 4
    with pytest.raises(ValueError, match="does not match"):
        eta_execution_limits([manifest])


def test_eta_gate_models_fold_and_aggregation_concurrency_separately() -> None:
    full = _manifest(
        10,
        family="d4rl_matched",
        estimator="bestdice",
        sample_size=50_000,
    )
    full["aggregation_units"] = [
        _aggregation_unit(
            family="d4rl_matched",
            estimator="bestdice",
            sample_size=50_000,
        )
    ]
    gate = build_eta_gate(
        pilot_fold_rows=[
            {
                "status": "ok",
                "benchmark_family": "d4rl_matched",
                "estimator_id": "bestdice",
                "sample_size": 50_000,
                "fit_runtime_sec": 2.0,
            }
        ],
        confirmatory_manifests=[full],
        maximum_concurrency=5,
        maximum_aggregation_concurrency=2,
        expected_full_fold_units=10,
        expected_full_data_fit_units=1,
        ceiling_hours=1.0,
    )
    assert gate["projected_p95_fold_walltime_hours"] == 4.0 / 3600.0
    assert gate["projected_p95_full_data_fit_walltime_hours"] == 1.25 / 3600.0


def test_eta_gate_uses_pilot_runtime_only_and_passes_under_ceiling() -> None:
    full = [
        _manifest(5, family="random_tabular", estimator="neural_fori", sample_size=5_000),
        _manifest(5, family="linear_gaussian", estimator="neural_fori", sample_size=50_000),
    ]
    pilot = [
        {
            "status": "ok",
            "benchmark_family": "random_tabular",
            "estimator_id": "neural_fori",
            "sample_size": 5_000,
            "fit_runtime_sec": 2.0,
        },
        {
            "status": "ok",
            "benchmark_family": "linear_gaussian",
            "estimator_id": "neural_fori",
            "sample_size": 50_000,
            "fit_runtime_sec": 4.0,
        },
    ]
    gate = build_eta_gate(
        pilot_fold_rows=pilot,
        confirmatory_manifests=full,
        expected_full_fold_units=10,
        ceiling_hours=1.0,
    )
    assert gate["status"] == "pass"
    assert gate["scientific_selection_performed"] is False
    assert len(gate["strata"]) == 2


def test_eta_gate_closes_on_missing_or_failed_strata() -> None:
    full = [_manifest(10, family="d4rl_matched", estimator="bestdice", sample_size=50_000)]
    gate = build_eta_gate(
        pilot_fold_rows=[{"status": "failed"}],
        confirmatory_manifests=full,
        expected_full_fold_units=10,
    )
    assert gate["status"] == "blocked"
    assert gate["missing_strata"]
    assert gate["pilot_failed_rows"] == 1


def test_eta_gate_includes_sequential_dataset_build_cost() -> None:
    full = [
        _manifest(
            10,
            family="dice_rl_cartpole",
            estimator="bestdice",
            sample_size=50_000,
        )
    ]
    pilot_fold = [
        {
            "status": "ok",
            "benchmark_family": "dice_rl_cartpole",
            "estimator_id": "bestdice",
            "sample_size": 50_000,
            "fit_runtime_sec": 1.0,
        }
    ]
    pilot_dataset = [
        {
            "status": "created",
            "benchmark_family": "dice_rl_cartpole",
            "sample_size": 50_000,
            "dataset_runtime_sec": 12.0,
        }
    ]
    gate = build_eta_gate(
        pilot_fold_rows=pilot_fold,
        pilot_dataset_rows=pilot_dataset,
        confirmatory_manifests=full,
        expected_full_fold_units=10,
        ceiling_hours=1.0,
    )
    assert gate["status"] == "pass"
    assert gate["projected_p95_dataset_walltime_hours"] == 12.0 / 3600.0
    assert gate["dataset_strata"][0]["full_dataset_units"] == 1


def test_eta_gate_charges_full_data_raw_fit_cost() -> None:
    full = _manifest(
        10,
        family="d4rl_matched",
        estimator="bestdice",
        sample_size=50_000,
    )
    full["aggregation_units"] = [
        _aggregation_unit(
            family="d4rl_matched",
            estimator="bestdice",
            sample_size=50_000,
        )
    ]
    gate = build_eta_gate(
        pilot_fold_rows=[
            {
                "status": "ok",
                "benchmark_family": "d4rl_matched",
                "estimator_id": "bestdice",
                "sample_size": 50_000,
                "fit_runtime_sec": 2.0,
            }
        ],
        confirmatory_manifests=[full],
        expected_full_fold_units=10,
        expected_full_data_fit_units=1,
        ceiling_hours=1.0,
    )
    assert gate["status"] == "pass"
    assert gate["full_data_fit_units"] == 1
    assert gate["strata"][0]["projected_fold_serial_hours"] == 20.0 / 3600.0
    assert (
        gate["strata"][0]["projected_full_data_fit_serial_hours"]
        == 2.5 / 3600.0
    )
