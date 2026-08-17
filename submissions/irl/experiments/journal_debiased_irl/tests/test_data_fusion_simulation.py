from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import data_fusion_simulation as data_fusion_module

from data_fusion_simulation import (
    FrozenOutcomeRegression,
    build_transition_factors_from_sieve,
    build_data_fusion_truth,
    data_fusion_if_contribution,
    data_fusion_normalization_policy,
    data_fusion_readiness,
    evaluate_policy_with_transition_factors,
    fit_gaussian_transition_sieve,
    fit_sieve_logit_behavior_policy,
    generate_outcome_source,
    run_data_fusion_replication,
)
from jrssb_simulation import JRSSBConfig, JRSSBOracle


class _ConstantGrid:
    def sample_states(self, weights, rng, n_samples, jitter):
        del weights, rng, jitter
        return np.zeros((n_samples, 2), dtype=float)


class _FakeOracle:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            state_jitter=True,
            state_low=-2.5,
            state_high=2.5,
            n_actions=2,
        )
        self.main_grid = _ConstantGrid()
        self.stationary_behavior = np.array([1.0])
        self.pi0 = np.array([[0.25, 0.75]])
        self.reward_dagger = np.array([[0.0, 2.0]])

    def policy_probs(self, states, policy_grid):
        del policy_grid
        return np.repeat(self.pi0, np.asarray(states).shape[0], axis=0)

    def action_values(self, states, value_grid):
        return np.repeat(np.asarray(value_grid), np.asarray(states).shape[0], axis=0)


def test_outcome_source_omits_actions_and_uses_known_randomization_mean():
    source = generate_outcome_source(
        _FakeOracle(), sample_size=20_000, seed=31, outcome_noise_sd=0.0
    )
    assert not hasattr(source, "actions")
    assert source.states.shape == (20_000, 2)
    assert source.outcomes.shape == (20_000,)
    assert np.mean(source.outcomes) == pytest.approx(1.0, abs=0.03)


def test_data_fusion_normalization_policy_is_known_uniform():
    probabilities = data_fusion_normalization_policy(
        _FakeOracle(), np.zeros((7, 2), dtype=float)
    )
    np.testing.assert_allclose(probabilities, 0.5)
    np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0)


def test_outcome_source_is_deterministic_by_seed_and_independent_across_seeds():
    oracle = _FakeOracle()
    first = generate_outcome_source(oracle, sample_size=100, seed=7)
    repeat = generate_outcome_source(oracle, sample_size=100, seed=7)
    other = generate_outcome_source(oracle, sample_size=100, seed=8)
    np.testing.assert_array_equal(first.outcomes, repeat.outcomes)
    assert not np.array_equal(first.outcomes, other.outcomes)


def test_readiness_gate_uses_ten_percent_of_pilot_standard_error():
    frozen = FrozenOutcomeRegression(
        model=object(),
        sample_size=1_000_000,
        seed=1,
        outcome_noise_sd=0.5,
        g_rmse=0.01,
        estimand_shift=0.004,
    )
    passed = data_fusion_readiness(frozen, pilot_median_se=0.05)
    assert passed["allowed_absolute_shift"] == pytest.approx(0.005)
    assert passed["passed"] is True
    failed = data_fusion_readiness(
        FrozenOutcomeRegression(
            model=object(),
            sample_size=1_000_000,
            seed=1,
            outcome_noise_sd=0.5,
            g_rmse=0.01,
            estimand_shift=0.006,
        ),
        pilot_median_se=0.05,
    )
    assert failed["passed"] is False


def test_known_softmax_scale_recovers_the_structural_reward_on_the_oracle_grid():
    oracle = JRSSBOracle(
        JRSSBConfig(
            main_grid_points=7,
            coarse_grid_points=5,
            max_iterations=500,
            use_oracle_cache=False,
        )
    )
    truth = build_data_fusion_truth(oracle)
    np.testing.assert_allclose(
        truth.reward_grid,
        oracle.reward_dagger,
        atol=1e-10,
        rtol=1e-10,
    )


def test_data_fusion_eif_scales_only_the_direct_log_policy_contrast():
    value = np.array([1.0, 2.0])
    d = np.array([3.0, 4.0])
    d_normalization = np.array([2.5, 4.5])
    target_residual = np.array([0.2, -0.1])
    normalization_residual = np.array([-0.3, 0.4])
    scale = 0.8
    expected = (
        value
        + d * target_residual
        + (d - d_normalization) * normalization_residual
        + scale * (d - d_normalization)
    )
    np.testing.assert_allclose(
        data_fusion_if_contribution(
            value=value,
            state_action_ratio=d,
            normalization_state_action_ratio=d_normalization,
            target_bellman_residual=target_residual,
            normalization_bellman_residual=normalization_residual,
            behavior_log_scale=scale,
        ),
        expected,
    )


def test_data_fusion_honors_configurable_crossfit_fold_count(monkeypatch):
    n = 25

    class FakeOracle:
        config = SimpleNamespace(
            crossfit_folds=5,
            crossfit_se_method="iid",
            crossfit_ci_method="normal",
            data_fusion_policy_mode="sieve-logit",
            data_fusion_transition_mode="sieve",
            data_fusion_g_mode="frozen",
            data_fusion_ratio_mode="neural-fore",
            data_fusion_target_gamma=0.80,
            data_fusion_repeated_splits=1,
            data_fusion_probability_floor=0.02,
        )

        @staticmethod
        def sample_stationary_transitions(n, seed):
            del seed
            return {
                "states": np.zeros((n, 2)),
                "actions": np.zeros(n, dtype=int),
                "next_states": np.zeros((n, 2)),
            }

    diagnostic = {
        "selected_iterations": 10.0,
        "apbv_score": 0.1,
        "fit_seconds": 0.0,
        "normalized_mass": 1.0,
        "logit_cap_fraction": 0.0,
        "behavior_policy_selected_c": 1.0,
        "behavior_policy_selected_degree": 2.0,
        "behavior_policy_validation_nll": 1.0,
        "behavior_probability_clipping_fraction": 0.0,
        "transition_selected_alpha": 1.0,
        "transition_validation_mse": 0.1,
        "transition_residual_sd_x": 0.2,
        "transition_residual_sd_z": 0.2,
        "nonfinite": 0.0,
    }

    def fake_fold(**kwargs):
        size = len(kwargs["eval_idx"])
        return {
            "plugin": np.zeros(size),
            "if": np.arange(size, dtype=float),
            "reward_squared_error": np.zeros(size),
            "ratio": np.ones(size),
            "ratio_diagnostics": diagnostic,
        }

    monkeypatch.setattr(
        data_fusion_module,
        "build_data_fusion_truth",
        lambda oracle: SimpleNamespace(psi=0.0),
    )
    monkeypatch.setattr(data_fusion_module, "_evaluate_data_fusion_fold", fake_fold)
    frozen = SimpleNamespace(
        sample_size=1_000_000,
        g_rmse=0.0,
        estimand_shift=0.0,
        normalization_policy_mode="known-uniform",
    )
    result = run_data_fusion_replication(
        oracle=FakeOracle(),
        outcome_regression=frozen,
        n=n,
        seed=19,
    )
    assert result.crossfit_fold_count == 5.0
    assert len(json.loads(result.fore_selected_iterations_by_fold)) == 5


def test_sieve_logit_selection_is_deterministic_and_probability_valid():
    rng = np.random.default_rng(91)
    states = rng.normal(size=(800, 2))
    logits = np.column_stack(
        [
            0.2 + 0.3 * states[:, 0],
            -0.1 + 0.2 * states[:, 1],
            -0.3 * states[:, 0] + 0.1 * states[:, 1],
            np.zeros(states.shape[0]),
        ]
    )
    probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    uniforms = rng.random(states.shape[0])
    actions = np.sum(uniforms[:, None] > np.cumsum(probabilities, axis=1), axis=1)
    first = fit_sieve_logit_behavior_policy(
        states=states, actions=actions, n_actions=4, seed=12
    )
    repeat = fit_sieve_logit_behavior_policy(
        states=states, actions=actions, n_actions=4, seed=12
    )
    assert first.selected_c == repeat.selected_c
    assert first.selected_degree == repeat.selected_degree
    assert first.selected_degree in {2, 3, 4}
    assert first.validation_nll == pytest.approx(repeat.validation_nll)
    assert all("paired_excess_nll" in row for row in first.candidate_scores)
    assert all("paired_excess_se" in row for row in first.candidate_scores)
    predicted = first.policy.predict_proba(states[:25])
    assert np.all(predicted > 0.0)
    np.testing.assert_allclose(np.sum(predicted, axis=1), 1.0)


def test_transition_sieve_is_source_only_and_discretizes_to_valid_factors():
    oracle = JRSSBOracle(
        JRSSBConfig(
            main_grid_points=7,
            coarse_grid_points=5,
            max_iterations=500,
            use_oracle_cache=False,
        )
    )
    source = oracle.sample_stationary_transitions(n=1_000, seed=72)
    selection = fit_gaussian_transition_sieve(
        states=source["states"],
        actions=source["actions"],
        next_states=source["next_states"],
        n_actions=oracle.config.n_actions,
        seed=81,
    )
    assert selection.selected_alpha in {0.01, 0.1, 1.0, 10.0}
    assert np.isfinite(selection.validation_mse)
    assert np.all(selection.model.residual_sds >= 0.05)
    factors = build_transition_factors_from_sieve(
        oracle=oracle, transition=selection.model
    )
    np.testing.assert_allclose(np.sum(factors["px"], axis=2), 1.0)
    np.testing.assert_allclose(np.sum(factors["pz"], axis=2), 1.0)
    zero_reward = np.zeros((oracle.main_grid.n_states, oracle.config.n_actions))
    q_grid, value_grid = evaluate_policy_with_transition_factors(
        oracle=oracle,
        reward_grid=zero_reward,
        policy_grid=oracle.pi0,
        gamma=oracle.config.gamma_behavior,
        transition_factors=factors,
    )
    np.testing.assert_allclose(q_grid, 0.0)
    np.testing.assert_allclose(value_grid, 0.0)
