from __future__ import annotations

import numpy as np

from occupancy_ratio_benchmark.calibration_pipeline import (
    execute_aggregation,
    execute_deterministic_score,
)
from occupancy_ratio_benchmark.discrete import make_discrete_dataset


def _manifest():
    return {
        "run_id": "occ-cal-fixture",
        "resolved_config": {
            "cross_calibration": {"folds": 2},
            "pava": {
                "maximum_iterations": 300,
                "relative_tolerance": 1e-8,
                "direction": "increasing",
                "damping": 1.0,
                "boundary_rule": "constant_endpoint_extrapolation",
            },
            "estimator_registry": {
                "oracle_transform_score": {"crossfit_base_fit_required": False}
            },
        },
    }


def _unit():
    return {
        "unit_id": "aggregate-fixture",
        "cell": {
            "cell_id": "tabular",
            "benchmark_family": "random_tabular",
            "states": 8,
            "actions": 2,
            "policy_shift": 0.35,
        },
        "conceptual_fold_count": 2,
        "depends_on": ["score-fixture"],
        "identity": {
            "study_id": "mechanism",
            "cell_id": "tabular",
            "estimator_id": "oracle_transform_score",
            "config_sha256": "0" * 64,
            "axis_values": {
                "sample_size": 512,
                "gamma": 0.8,
                "seed": 3,
                "score_distortion": "half_oracle",
            },
        },
    }


def test_deterministic_scores_flow_through_cross_calibrated_aggregation() -> None:
    dataset = make_discrete_dataset(
        setting="random_tabular_mdp",
        gamma=0.8,
        sample_size=512,
        seed=3,
        policy_shift=0.35,
        n_states=8,
        n_actions=2,
    )
    source_groups = np.arange(dataset.n, dtype=np.int64)
    initial_groups = np.arange(dataset.initial_states.shape[0], dtype=np.int64) + 20_000
    score = execute_deterministic_score(
        manifest=_manifest(),
        unit=_unit(),
        dataset=dataset,
    )
    output = execute_aggregation(
        manifest=_manifest(),
        unit=_unit(),
        dataset=dataset,
        source_groups=source_groups,
        initial_groups=initial_groups,
        source_q_by_fold=score.source_q_by_fold,
        next_q_by_fold=score.next_q_by_fold,
        initial_q_by_fold=score.initial_q_by_fold,
    )
    assert len(output.rows) == 3
    assert output.diagnostics["pooled_calibrator_count"] == 1
    assert output.diagnostics["aggregation"] == "pointwise_median"
    assert {row["candidate_id"] for row in output.rows} == set(output.candidate_arrays)
