from __future__ import annotations

import numpy as np

from jrssb_simulation import (
    JRSSBConfig,
    JRSSBOracle,
    crossfit_critical_value,
    crossfit_se_diagnostics,
    deterministic_fore_training_indices,
    exact_adaptive_ratio_bundle,
    fit_behavior_policy_example1a,
    fold_splits,
    ordinary_ratio_if_contribution,
    reconstruct_total_signed_ratios,
    signed_ratio_if_contribution,
)


def test_fore_training_cap_is_deterministic_truth_blind_and_optional():
    uncapped = deterministic_fore_training_indices(23, max_rows=0, seed=11)
    np.testing.assert_array_equal(uncapped, np.arange(23))
    first = deterministic_fore_training_indices(100, max_rows=20, seed=19)
    second = deterministic_fore_training_indices(100, max_rows=20, seed=19)
    third = deterministic_fore_training_indices(100, max_rows=20, seed=20)
    assert first.shape == (20,)
    assert np.all(np.diff(first) > 0)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_outer_crossfit_folds_are_deterministic_disjoint_and_complete():
    first = fold_splits(101, seed=409, n_folds=2)
    second = fold_splits(101, seed=409, n_folds=2)
    assert len(first) == 2
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert set(first[0]).isdisjoint(set(first[1]))
    assert sorted(np.concatenate(first).tolist()) == list(range(101))


def test_five_and_ten_fold_partitions_are_complete_and_balanced():
    for fold_count in (5, 10):
        folds = fold_splits(103, seed=811, n_folds=fold_count)
        assert len(folds) == fold_count
        assert max(map(len, folds)) - min(map(len, folds)) <= 1
        assert sorted(np.concatenate(folds).tolist()) == list(range(103))


def test_crossfit_cluster_se_matches_hand_calculation_and_fold_t():
    contributions = np.array([1.0, 2.0, 3.0, 4.0])
    diagnostics = crossfit_se_diagnostics(
        contributions,
        [np.array([0, 1]), np.array([2, 3])],
    )
    assert diagnostics["iid_se"] == np.std(contributions, ddof=1) / 2.0
    assert diagnostics["fold_cluster_se"] == 1.0
    assert crossfit_critical_value(1.96, 5, "fold-t") > 2.77


def test_oracle_policy_and_adaptive_ratio_diagnostics_match_grid_truth():
    oracle = JRSSBOracle(
        JRSSBConfig(
            main_grid_points=7,
            coarse_grid_points=5,
            max_iterations=500,
            example1a_policy_estimator="oracle",
            use_oracle_cache=False,
        )
    )
    policy = fit_behavior_policy_example1a(
        oracle,
        oracle.main_grid.states[:10],
        np.zeros(10, dtype=int),
        seed=7,
    )
    np.testing.assert_allclose(
        policy.predict_proba(oracle.main_grid.states),
        oracle.pi0,
    )
    fixed = exact_adaptive_ratio_bundle(
        oracle,
        oracle.pi_fix,
        oracle.config.gamma_behavior,
    )
    np.testing.assert_allclose(fixed["eta"], oracle.eta_fix)
    np.testing.assert_allclose(fixed["rho"], oracle.rho_fix)
    signed = exact_adaptive_ratio_bundle(
        oracle,
        oracle.pi_star,
        oracle.config.gamma_behavior,
        q_grid=oracle.q_1b,
        v_grid=oracle.v_1b,
    )
    np.testing.assert_allclose(signed["eta"], oracle.eta_star)
    np.testing.assert_allclose(signed["tilde_eta"], oracle.tilde_eta_star)
    np.testing.assert_allclose(signed["tilde_d"], oracle.tilde_d_star)
    np.testing.assert_allclose(signed["tilde_rho"], oracle.tilde_rho_star)


def test_ordinary_state_action_form_equals_propensity_ratio_form():
    value = np.array([0.2, -0.4, 1.1])
    rho = np.array([2.0, 1.5, 3.0])
    policy_ratio = np.array([0.5, 1.2, 0.8])
    bellman = np.array([0.3, -0.2, 0.1])
    state_action = rho * policy_ratio
    old = value + rho * policy_ratio * bellman + rho * (policy_ratio - 1.0)
    new = ordinary_ratio_if_contribution(value, state_action, rho, bellman)
    np.testing.assert_allclose(new, old)


def test_signed_state_action_form_equals_factorized_form_when_current_term_zero():
    value = np.array([0.2, -0.4, 1.1])
    rho = np.array([2.0, 1.5, 3.0])
    tilde_rho = np.array([0.4, -0.3, 0.1])
    policy_ratio = np.array([0.5, 1.2, 0.8])
    bellman = np.array([0.3, -0.2, 0.1])
    policy_residual = np.array([0.1, -0.4, 0.2])
    gamma = 0.9
    temperature = 0.6
    state_action = rho * policy_ratio
    signed_state_action = tilde_rho * policy_ratio
    old = (
        value
        + rho * policy_ratio * bellman
        + (gamma / temperature) * tilde_rho * policy_ratio * policy_residual
        + ((tilde_rho / temperature) + rho) * (policy_ratio - 1.0)
    )
    new = signed_ratio_if_contribution(
        value,
        state_action,
        rho,
        signed_state_action,
        tilde_rho,
        bellman,
        policy_residual,
        gamma=gamma,
        temperature=temperature,
    )
    np.testing.assert_allclose(new, old)


def test_total_signed_reconstruction_adds_current_term_and_state_average():
    ordinary = np.array([[2.0, 1.0], [0.5, 3.0]])
    future = np.array([[0.2, -0.1], [0.4, 0.7]])
    advantage = np.array([[0.3, -0.6], [-0.2, 0.5]])
    behavior = np.array([[0.75, 0.25], [0.4, 0.6]])
    actions = np.array([1, 0])
    expected_all = ordinary * advantage + future
    signed_d, signed_rho = reconstruct_total_signed_ratios(
        ordinary, future, advantage, behavior, actions
    )
    np.testing.assert_allclose(signed_d, [expected_all[0, 1], expected_all[1, 0]])
    np.testing.assert_allclose(signed_rho, np.sum(behavior * expected_all, axis=1))
    # Estimated ordinary ratios need not factor through the target policy, so
    # the immediate state average is generally nonzero and must be retained.
    assert not np.allclose(np.sum(behavior * ordinary * advantage, axis=1), 0.0)


def test_gamma_zero_signed_resolvent_is_current_source_not_zero():
    ordinary = np.array([[1.4, 0.6]])
    advantage = np.array([[0.25, -0.75]])
    behavior = np.array([[0.5, 0.5]])
    signed_d, signed_rho = reconstruct_total_signed_ratios(
        ordinary,
        np.zeros_like(ordinary),
        advantage,
        behavior,
        np.array([0]),
    )
    np.testing.assert_allclose(signed_d, [1.4 * 0.25])
    np.testing.assert_allclose(
        signed_rho, [0.5 * 1.4 * 0.25 + 0.5 * 0.6 * -0.75]
    )


def test_total_signed_grid_ratio_solves_adjoint_resolvent():
    rng = np.random.default_rng(271)
    n_states, n_actions = 3, 2
    gamma = 0.73
    behavior_state = rng.dirichlet(np.ones(n_states))
    behavior_policy = rng.dirichlet(np.ones(n_actions), size=n_states)
    target_policy = rng.dirichlet(np.ones(n_actions), size=n_states)
    transition = rng.dirichlet(np.ones(n_states), size=(n_actions, n_states))
    eta = rng.uniform(0.4, 1.2, size=n_states)
    q = rng.normal(size=(n_states, n_actions))
    value = np.sum(target_policy * q, axis=1)
    behavior_joint = behavior_state[:, None] * behavior_policy
    ordinary = eta[:, None] * target_policy / behavior_joint
    source_measure = eta[:, None] * target_policy * (q - value[:, None])

    # State-action target transition, with rows indexed by current (s,a).
    p_target = np.zeros((n_states * n_actions, n_states * n_actions))
    for state in range(n_states):
        for action in range(n_actions):
            row = state * n_actions + action
            for next_state in range(n_states):
                for next_action in range(n_actions):
                    column = next_state * n_actions + next_action
                    p_target[row, column] = (
                        transition[action, state, next_state]
                        * target_policy[next_state, next_action]
                    )
    total_measure = np.linalg.solve(
        np.eye(n_states * n_actions) - gamma * p_target.T,
        source_measure.reshape(-1),
    ).reshape(n_states, n_actions)
    future_ratio = (total_measure - source_measure) / behavior_joint
    all_actions = np.tile(np.arange(n_actions), n_states)
    signed_d, signed_rho = reconstruct_total_signed_ratios(
        ordinary,
        future_ratio,
        q - value[:, None],
        behavior_policy,
        all_actions[:n_states],
    )
    np.testing.assert_allclose(
        ordinary * (q - value[:, None]) + future_ratio,
        total_measure / behavior_joint,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        signed_rho,
        np.sum(total_measure, axis=1) / behavior_state,
        atol=1e-12,
    )
    del signed_d


def test_reward_finite_difference_requires_current_signed_term():
    rng = np.random.default_rng(902)
    n_states, n_actions = 3, 2
    gamma, temperature = 0.71, 0.8
    initial = rng.dirichlet(np.ones(n_states))
    behavior = rng.dirichlet(np.ones(n_actions), size=n_states)
    transition = rng.dirichlet(np.ones(n_states), size=(n_actions, n_states))
    reward = rng.normal(size=(n_states, n_actions))
    direction = rng.normal(size=(n_states, n_actions))

    def soft_policy(reward_grid):
        value = np.zeros(n_states)
        for _ in range(20_000):
            q_soft = reward_grid + gamma * np.einsum(
                "asn,n->sa", transition, value
            )
            maximum = np.max(q_soft / temperature, axis=1, keepdims=True)
            updated = temperature * (
                maximum[:, 0]
                + np.log(np.sum(np.exp(q_soft / temperature - maximum), axis=1))
            )
            if np.max(np.abs(updated - value)) < 1e-13:
                value = updated
                break
            value = updated
        logits = q_soft / temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        policy = np.exp(logits)
        return policy / np.sum(policy, axis=1, keepdims=True)

    def standard_value(reward_grid, policy):
        kernel = np.einsum("sa,asn->sn", policy, transition)
        state_reward = np.sum(policy * reward_grid, axis=1)
        value = np.linalg.solve(np.eye(n_states) - gamma * kernel, state_reward)
        q = reward_grid + gamma * np.einsum("asn,n->sa", transition, value)
        return value, q, kernel

    policy = soft_policy(reward)
    value, q, kernel = standard_value(reward, policy)
    eta = np.linalg.solve(np.eye(n_states) - gamma * kernel.T, initial)
    behavior_joint = initial[:, None] * behavior
    ordinary_measure = eta[:, None] * policy
    ordinary_ratio = ordinary_measure / behavior_joint
    source_measure = ordinary_measure * (q - value[:, None])
    p_target = np.zeros((n_states * n_actions, n_states * n_actions))
    for state in range(n_states):
        for action in range(n_actions):
            row = state * n_actions + action
            for next_state in range(n_states):
                for next_action in range(n_actions):
                    column = next_state * n_actions + next_action
                    p_target[row, column] = (
                        transition[action, state, next_state]
                        * policy[next_state, next_action]
                    )
    total_signed_measure = np.linalg.solve(
        np.eye(n_states * n_actions) - gamma * p_target.T,
        source_measure.reshape(-1),
    ).reshape(n_states, n_actions)
    future_signed_measure = total_signed_measure - source_measure
    predicted = np.sum(
        behavior_joint
        * (ordinary_ratio + total_signed_measure / behavior_joint / temperature)
        * direction
    )
    predicted_future_only = np.sum(
        behavior_joint
        * (ordinary_ratio + future_signed_measure / behavior_joint / temperature)
        * direction
    )
    epsilon = 1e-5
    policy_plus = soft_policy(reward + epsilon * direction)
    policy_minus = soft_policy(reward - epsilon * direction)
    value_plus = standard_value(reward + epsilon * direction, policy_plus)[0]
    value_minus = standard_value(reward - epsilon * direction, policy_minus)[0]
    finite_difference = (
        np.dot(initial, value_plus) - np.dot(initial, value_minus)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        predicted, finite_difference, rtol=2e-6, atol=2e-7
    )
    assert abs(predicted_future_only - finite_difference) > 1e-3


def test_signed_fore_jordan_convention_matches_oracle_grid_ratio():
    oracle = JRSSBOracle(
        JRSSBConfig(
            main_grid_points=7,
            coarse_grid_points=5,
            max_iterations=500,
            use_oracle_cache=False,
        )
    )
    diagnostic = oracle.signed_jordan_oracle_diagnostic()
    assert diagnostic["signed_jordan_positive_mass"] > 0.0
    assert diagnostic["signed_jordan_negative_mass"] > 0.0
    assert diagnostic["signed_jordan_ratio_rmse"] < 1e-10
    assert diagnostic["signed_jordan_ratio_sup_error"] < 1e-9
    assert diagnostic["signed_jordan_future_ratio_sup_error"] < 1e-9
    assert diagnostic["signed_jordan_state_marginal_sup_error"] < 1e-9
