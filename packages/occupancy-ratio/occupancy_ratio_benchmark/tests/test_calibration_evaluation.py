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
