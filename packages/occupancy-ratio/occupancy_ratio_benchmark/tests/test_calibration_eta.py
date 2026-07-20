from __future__ import annotations

from occupancy_ratio_benchmark.calibration_eta import build_eta_gate


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
