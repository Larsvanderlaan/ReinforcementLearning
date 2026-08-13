from __future__ import annotations

import numpy as np
import pytest

from occupancy_ratio_benchmark.calibration_crossfit import (
    CrossCalibrationConfig,
    fit_cross_calibrated_matrices,
    make_grouped_fold_assignment,
)
from occupancy_ratio_benchmark.calibration_evaluation import (
    evaluate_cross_calibrated_result,
    weight_diagnostics,
)
from occupancy_ratio_benchmark.discrete import make_discrete_dataset


def _fixture():
    dataset = make_discrete_dataset(
        setting="random_tabular_mdp",
        gamma=0.8,
        sample_size=256,
        seed=14,
        policy_shift=0.35,
        n_states=8,
        n_actions=2,
    )
    source_groups = np.arange(dataset.n, dtype=np.int64)
    initial_groups = np.arange(dataset.initial_states.shape[0], dtype=np.int64) + 10_000
    assignment = make_grouped_fold_assignment(
        source_groups,
        initial_groups,
        num_folds=2,
        seed=7,
    )
    source = np.stack([dataset.true_ratio * 0.5, dataset.true_ratio * 0.55])
    next_truth = np.ones(dataset.n, dtype=np.float64)
    initial_truth = np.ones(dataset.initial_states.shape[0], dtype=np.float64)
    result = fit_cross_calibrated_matrices(
        source_q_by_fold=source,
        next_q_by_fold=np.stack([next_truth, 1.1 * next_truth]),
        initial_q_by_fold=np.stack([initial_truth, 1.1 * initial_truth]),
        assignment=assignment,
        gamma=dataset.gamma,
        config=CrossCalibrationConfig(num_folds=2, seed=7, pava_num_iterations=20),
    )
    return dataset, source_groups, initial_groups, result


def test_evaluation_emits_three_candidates_and_prespecified_endpoints() -> None:
    dataset, source_groups, initial_groups, result = _fixture()
    rows, arrays = evaluate_cross_calibrated_result(
        dataset=dataset,
        result=result,
        source_groups=source_groups,
        initial_groups=initial_groups,
        identity={
            "study_id": "controlled",
            "cell_id": "tabular",
            "benchmark_family": "random_tabular",
            "seed": 14,
        },
        estimator_id="fixture",
        fold_runtime_sec=(0.1, 0.2),
    )

    assert len(rows) == 3
    assert set(arrays) == {
        "native_pointwise_median",
        "scalar_normalized_pointwise_median",
        "pava_pointwise_median",
    }
    for row in rows:
        assert row["status"] == "ok"
        assert row["cross_moment_halves_used_for_fit"] is False
        assert row["cross_moment_signed_squared_error"] is not None
        assert row["ratio_mse_untruncated"] >= 0.0
        assert row["ratio_generalized_kl_extended"] >= 0.0
        assert row["oracle_floor_smallest_positive"] > 0.0
        assert row["policy_value_absolute_error"] >= 0.0
        assert row["fold_runtime_total_sec"] == pytest.approx(0.3)


def test_evaluation_rejects_coverage_stopping() -> None:
    dataset, source_groups, initial_groups, result = _fixture()
    dataset.next_retention = np.zeros(dataset.n, dtype=np.float64)
    with pytest.raises(ValueError, match="coverage-stopped"):
        evaluate_cross_calibrated_result(
            dataset=dataset,
            result=result,
            source_groups=source_groups,
            initial_groups=initial_groups,
            identity={"cell_id": "bad"},
            estimator_id="fixture",
        )


def test_weight_diagnostics_have_exact_mass_and_ess() -> None:
    diagnostics = weight_diagnostics(np.ones(100, dtype=np.float64))
    assert diagnostics["empirical_mass"] == pytest.approx(1.0)
    assert diagnostics["effective_sample_size"] == pytest.approx(100.0)
    assert diagnostics["effective_sample_size_fraction"] == pytest.approx(1.0)
    assert diagnostics["top_one_percent_weight_mass"] == pytest.approx(0.01)


def test_evaluation_adds_full_data_raw_and_multi_reward_endpoints() -> None:
    dataset, source_groups, initial_groups, result = _fixture()
    take = np.arange(80)
    dataset.target_occupancy_states = np.concatenate(
        [dataset.states[take], dataset.states[take]],
        axis=0,
    )
    dataset.target_occupancy_actions = np.concatenate(
        [dataset.actions[take], dataset.actions[take]],
        axis=0,
    )
    dataset.target_occupancy_episode_ids = np.arange(160)
    dataset.target_occupancy_pool_ids = np.repeat([0, 1], 80)
    rows, arrays = evaluate_cross_calibrated_result(
        dataset=dataset,
        result=result,
        source_groups=source_groups,
        initial_groups=initial_groups,
        identity={
            "study_id": "controlled",
            "cell_id": "tabular",
            "benchmark_family": "random_tabular",
            "gamma": dataset.gamma,
            "seed": dataset.seed,
        },
        estimator_id="fixture",
        full_data_predictions={
            "current": np.ones(dataset.n),
            "next": np.ones(dataset.n),
            "initial": np.ones(dataset.initial_states.shape[0]),
        },
        full_data_fit_runtime_sec=0.4,
    )

    assert len(rows) == 4
    assert set(arrays) == {
        "native_pointwise_median",
        "scalar_normalized_pointwise_median",
        "pava_pointwise_median",
        "full_data_raw",
    }
    full = next(row for row in rows if row["candidate_id"] == "full_data_raw")
    assert full["fit_scope"] == "full_data"
    assert full["num_base_fits"] == 1
    assert full["calibration_fit"] == "none"
    assert full["full_data_fit_runtime_sec"] == pytest.approx(0.4)
    for row in rows:
        assert row["occupancy_functional_reward_count"] == 256
        assert np.isfinite(row["occupancy_functional_cross_pool_signed_mse"])
        assert len(arrays[row["candidate_id"]]["functional_pooled_error"]) == 256


def test_external_audit_drives_cross_moment_without_changing_ope_sample() -> None:
    dataset, source_groups, initial_groups, result = _fixture()
    audit = make_discrete_dataset(
        setting="random_tabular_mdp",
        gamma=dataset.gamma,
        sample_size=300,
        seed=dataset.seed,
        sample_seed=91,
        policy_shift=0.35,
        n_states=8,
        n_actions=2,
    )
    audit_source_groups = np.arange(audit.n, dtype=np.int64) + 50_000
    audit_initial_groups = (
        np.arange(audit.initial_states.shape[0], dtype=np.int64) + 60_000
    )

    def audit_predictions(value: float):
        roles = {
            "current": np.full(audit.n, value),
            "next": np.full(audit.n, value),
            "initial": np.full(audit.initial_states.shape[0], value),
        }
        return {
            candidate_id: {key: value.copy() for key, value in roles.items()}
            for candidate_id in (
                "native_pointwise_median",
                "scalar_normalized_pointwise_median",
                "pava_pointwise_median",
            )
        }

    common = {
        "dataset": dataset,
        "result": result,
        "source_groups": source_groups,
        "initial_groups": initial_groups,
        "identity": {
            "study_id": "controlled",
            "cell_id": "tabular",
            "benchmark_family": "random_tabular",
            "seed": 14,
        },
        "estimator_id": "fixture",
        "audit_dataset": audit,
        "audit_source_groups": audit_source_groups,
        "audit_initial_groups": audit_initial_groups,
    }
    calibrated_rows, _ = evaluate_cross_calibrated_result(
        **common,
        audit_candidate_predictions=audit_predictions(1.0),
    )
    misspecified_rows, _ = evaluate_cross_calibrated_result(
        **common,
        audit_candidate_predictions=audit_predictions(2.0),
    )

    for calibrated, misspecified in zip(calibrated_rows, misspecified_rows):
        assert calibrated["cross_moment_external_behavior_audit"] is True
        assert calibrated["cross_moment_basis_audit_groups_disjoint"] is True
        assert calibrated["cross_moment_target"] == (
            "finite_bin_projected_bellman_calibration_error"
        )
        assert calibrated["cross_moment_signed_squared_error"] == pytest.approx(
            0.0, abs=1e-28
        )
        assert misspecified["cross_moment_signed_squared_error"] > 0.0
        assert calibrated["policy_value_estimate"] == misspecified[
            "policy_value_estimate"
        ]
        assert calibrated["cross_moment_n_basis_groups"] > 0
        assert calibrated["cross_moment_n_transition_rows_a"] > 0
        assert calibrated["cross_moment_n_transition_rows_b"] > 0
