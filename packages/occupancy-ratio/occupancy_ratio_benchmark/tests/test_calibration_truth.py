from __future__ import annotations

import numpy as np

from occupancy_ratio_benchmark.calibration_truth import (
    SCORE_DISTORTIONS,
    exact_ratio_at,
    has_exact_finite_support,
    oracle_score_matrices,
)
from occupancy_ratio_benchmark.discrete import make_discrete_dataset
from occupancy_ratio_benchmark.gaussian import make_linear_gaussian_dataset


def test_exact_ratio_evaluator_reproduces_generator_truth() -> None:
    for dataset in (
        make_discrete_dataset(
            setting="random_tabular_mdp",
            gamma=0.99,
            sample_size=400,
            seed=7,
            policy_shift=1.0,
            n_states=12,
            n_actions=4,
        ),
        make_linear_gaussian_dataset(
            gamma=0.9,
            sample_size=400,
            seed=8,
            policy_shift=2.0,
        ),
    ):
        evaluated = exact_ratio_at(dataset, dataset.states, dataset.actions)
        assert np.allclose(evaluated, dataset.true_ratio, rtol=1e-10, atol=1e-12)


def test_oracle_mechanisms_emit_identical_conceptual_folds() -> None:
    dataset = make_discrete_dataset(
        setting="random_tabular_mdp",
        gamma=0.9,
        sample_size=300,
        seed=3,
        policy_shift=0.35,
        n_states=10,
        n_actions=3,
    )
    for distortion in SCORE_DISTORTIONS:
        result = oracle_score_matrices(dataset, distortion=distortion, num_folds=4)
        assert result.source_q_by_fold.shape == (4, dataset.n)
        assert np.allclose(result.source_q_by_fold[0], result.source_q_by_fold[-1])
        assert np.all(result.source_q_by_fold >= 0.0)
    half = oracle_score_matrices(dataset, distortion="half_oracle", num_folds=2)
    assert np.allclose(half.source_q_by_fold[0], 0.5 * dataset.true_ratio)
    negative = oracle_score_matrices(
        dataset, distortion="reciprocal_normalized_oracle", num_folds=2
    )
    order = np.argsort(negative.source_oracle)
    assert np.all(np.diff(negative.source_q_by_fold[0, order]) <= 1e-12)
    assert np.unique(negative.source_q_by_fold[0]).size > 1
    assert np.isclose(np.mean(negative.source_q_by_fold[0]), 1.0)


def test_floor_sensitivity_is_finite_support_only() -> None:
    tabular = make_discrete_dataset(
        setting="random_tabular_mdp", gamma=0.9, sample_size=20, seed=1
    )
    gaussian = make_linear_gaussian_dataset(
        gamma=0.9, sample_size=20, seed=1
    )
    assert has_exact_finite_support(tabular)
    assert not has_exact_finite_support(gaussian)
