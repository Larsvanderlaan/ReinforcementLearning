from __future__ import annotations

import numpy as np
import pytest

from occupancy_ratio_benchmark.calibration_metrics import (
    controlled_ratio_errors,
    estimate_bellman_cross_moment_error,
    estimate_multi_reward_occupancy_functional_error,
    generalized_kl_divergence,
    independent_audit_basis_masks,
    oracle_floor_kl_sensitivity,
)


def _constant_candidate(value: float, *, basis_group: object = "basis"):
    return estimate_bellman_cross_moment_error(
        basis_candidate_weights=np.full(8, value),
        audit_current_weights=np.full(8, value),
        audit_next_weights=np.full(8, value),
        audit_initial_weights=np.full(4, value),
        audit_transition_group_ids=np.repeat(np.arange(4), 2),
        audit_initial_group_ids=np.arange(4),
        basis_group_ids=np.full(8, basis_group),
        gamma=0.9,
    )


def test_normalized_constant_ratio_has_zero_cross_moment_error() -> None:
    result = _constant_candidate(1.0)

    assert result.basis_near_constant
    assert result.base.effective_bins == 1
    assert result.half_bins.effective_bins == 1
    assert result.double_bins.effective_bins == 1
    assert result.mass.effective_bins == 1
    assert result.base.moment_a == pytest.approx((0.0,), abs=1e-15)
    assert result.base.moment_b == pytest.approx((0.0,), abs=1e-15)
    assert result.base.signed_squared_error == pytest.approx(0.0, abs=1e-30)
    assert result.base.positive_part_root_error == pytest.approx(0.0)
    assert result.basis_audit_groups_disjoint is True


def test_mass_miscalibration_matches_closed_form() -> None:
    result = _constant_candidate(2.0)

    expected_moment = (1.0 - 0.9) * (1.0 - 2.0)
    expected_signed = expected_moment**2 / (1.0 + 1e-8)
    assert result.mass.ridge == pytest.approx(1e-8)
    assert result.mass.moment_a == pytest.approx((expected_moment,))
    assert result.mass.moment_b == pytest.approx((expected_moment,))
    assert result.mass.signed_squared_error == pytest.approx(expected_signed)
    assert result.mass.positive_part_root_error == pytest.approx(np.sqrt(expected_signed))


def test_signed_cross_moment_is_not_clipped_when_halves_disagree() -> None:
    result = estimate_bellman_cross_moment_error(
        basis_candidate_weights=np.ones(4),
        audit_current_weights=np.array([0.5, 0.5, 1.5, 1.5]),
        audit_next_weights=np.ones(4),
        audit_initial_weights=np.ones(2),
        audit_transition_group_ids=np.array([10, 10, 20, 20]),
        audit_initial_group_ids=np.array([10, 20]),
        gamma=0.0,
        bin_count=1,
    )

    assert sorted((result.mass.moment_a[0], result.mass.moment_b[0])) == pytest.approx([-0.5, 0.5])
    assert result.mass.signed_squared_error == pytest.approx(-0.25 / (1.0 + 1e-8))
    assert result.mass.positive_part_root_error == 0.0
    assert result.basis_audit_groups_disjoint is None


def test_quantile_resolutions_and_group_overlap_are_auditable() -> None:
    basis = np.arange(1.0, 101.0)
    current = np.linspace(0.25, 3.0, 40)
    group_ids = np.repeat(np.arange(10), 4)
    result = estimate_bellman_cross_moment_error(
        basis_candidate_weights=basis,
        audit_current_weights=current,
        audit_next_weights=current[::-1],
        audit_initial_weights=np.linspace(0.5, 1.5, 10),
        audit_transition_group_ids=group_ids,
        audit_initial_group_ids=np.arange(10),
        basis_group_ids=np.arange(1_000, 1_100),
        gamma=0.95,
        bin_count=6,
        split_seed=17,
    )

    assert result.base.requested_bins == 6
    assert result.base.effective_bins == 6
    assert result.half_bins.requested_bins == 3
    assert result.half_bins.effective_bins == 3
    assert result.double_bins.requested_bins == 12
    assert result.double_bins.effective_bins == 12
    assert sum(result.base.gram_diagonal) == pytest.approx(1.0)
    assert all(value > 0.0 for value in result.base.gram_diagonal)
    assert sum(result.base.moment_a) == pytest.approx(result.mass.moment_a[0])
    assert sum(result.base.moment_b) == pytest.approx(result.mass.moment_b[0])
    assert result.n_groups_a == 5
    assert result.n_groups_b == 5
    assert result.n_transition_groups == 10
    assert result.n_initial_groups == 10
    assert result.n_union_groups == 10
    assert result.n_audit_groups == result.n_union_groups
    assert result.n_transition_groups_a == 5
    assert result.n_transition_groups_b == 5
    assert result.n_initial_groups_a == 5
    assert result.n_initial_groups_b == 5
    assert result.basis_audit_groups_disjoint is True

    overlap = estimate_bellman_cross_moment_error(
        basis_candidate_weights=np.linspace(0.5, 1.5, 10),
        audit_current_weights=current,
        audit_next_weights=current[::-1],
        audit_initial_weights=np.linspace(0.5, 1.5, 10),
        audit_transition_group_ids=group_ids,
        audit_initial_group_ids=np.arange(10),
        basis_group_ids=np.arange(10),
        gamma=0.95,
        bin_count=3,
    )
    assert overlap.basis_audit_groups_disjoint is False


def test_group_split_is_deterministic_under_initial_row_reordering() -> None:
    kwargs = {
        "basis_candidate_weights": np.linspace(0.1, 2.0, 16),
        "audit_current_weights": np.linspace(0.2, 1.8, 16),
        "audit_next_weights": np.linspace(1.8, 0.2, 16),
        "audit_transition_group_ids": np.repeat(np.array(["a", "b", "c", "d"]), 4),
        "gamma": 0.8,
        "bin_count": 4,
        "split_seed": 29,
    }
    initial_weight = np.array([0.3, 0.7, 1.1, 1.5])
    initial_group = np.array(["a", "b", "c", "d"])
    first = estimate_bellman_cross_moment_error(
        **kwargs,
        audit_initial_weights=initial_weight,
        audit_initial_group_ids=initial_group,
    )
    order = np.array([2, 0, 3, 1])
    second = estimate_bellman_cross_moment_error(
        **kwargs,
        audit_initial_weights=initial_weight[order],
        audit_initial_group_ids=initial_group[order],
    )

    assert second.base == first.base
    assert second.mass == first.mass


def test_disjoint_transition_and_initial_group_sets_split_both_samples() -> None:
    result = estimate_bellman_cross_moment_error(
        basis_candidate_weights=np.linspace(0.5, 1.5, 8),
        audit_current_weights=np.linspace(0.4, 1.6, 8),
        audit_next_weights=np.linspace(1.6, 0.4, 8),
        audit_initial_weights=np.linspace(0.7, 1.3, 4),
        audit_transition_group_ids=np.repeat(
            np.array(["t0", "t1", "t2", "t3"]),
            2,
        ),
        audit_initial_group_ids=np.array(["i0", "i1", "i2", "i3"]),
        basis_group_ids=np.repeat(np.array(["i0", "i1", "i2", "i3"]), 2),
        gamma=0.9,
        bin_count=3,
        split_seed=5,
    )

    assert result.n_transition_groups == 4
    assert result.n_initial_groups == 4
    assert result.n_union_groups == 8
    assert result.n_audit_groups == 8
    assert result.n_groups_a + result.n_groups_b == 8
    assert result.n_transition_groups_a > 0
    assert result.n_transition_groups_b > 0
    assert result.n_initial_groups_a > 0
    assert result.n_initial_groups_b > 0
    assert result.basis_audit_groups_disjoint is False
    assert np.isfinite(result.base.signed_squared_error)


def test_controlled_ratio_errors_are_untruncated_and_kl_is_exact() -> None:
    estimated = np.array([0.0, 1.0, 2.0])
    oracle = np.array([1.0, 1.0, 2.0])
    result = controlled_ratio_errors(estimated, oracle)

    assert result.mse == pytest.approx(1.0 / 3.0)
    assert result.relative_mse == pytest.approx(1.0 / 6.0)
    assert result.rmse == pytest.approx(np.sqrt(1.0 / 3.0))
    assert result.l1 == pytest.approx(1.0 / 3.0)
    assert result.generalized_kl == pytest.approx(1.0 / 3.0)
    assert generalized_kl_divergence(np.array([0.0]), np.array([0.0])) == 0.0
    assert np.isinf(generalized_kl_divergence(np.array([0.5]), np.array([0.0])))


def test_zero_reference_weight_drops_an_infinite_kl_support_point() -> None:
    value = generalized_kl_divergence(
        np.array([1.0, 1.0]),
        np.array([0.0, 1.0]),
        sample_weight=np.array([0.0, 2.0]),
    )
    assert value == pytest.approx(0.0)


def test_ratio_metrics_avoid_avoidable_intermediate_overflow() -> None:
    result = controlled_ratio_errors(np.array([1e200]), np.array([2e200]))
    assert np.isinf(result.mse)
    assert result.relative_mse == pytest.approx(0.25)
    assert np.isfinite(generalized_kl_divergence(np.array([1e100]), np.array([1e-300])))


def test_oracle_floor_sensitivity_changes_only_the_oracle_kl() -> None:
    estimated = np.array([0.5, 1.0, 2.0])
    oracle = np.array([0.0, 1.0, 2.0])
    result = oracle_floor_kl_sensitivity(
        estimated,
        oracle,
        oracle_is_exact_finite_support=True,
    )

    assert result.smallest_positive_oracle == 1.0
    assert np.isinf(result.exact_generalized_kl)
    assert [point.multiplier for point in result.points] == [0.1, 1.0, 10.0]
    assert [point.floor for point in result.points] == pytest.approx([0.1, 1.0, 10.0])
    for point in result.points:
        expected = generalized_kl_divergence(estimated, np.maximum(oracle, point.floor))
        assert point.generalized_kl == pytest.approx(expected)
    assert controlled_ratio_errors(estimated, oracle).mse == pytest.approx(1.0 / 12.0)


@pytest.mark.parametrize(
    ("function", "message"),
    [
        (
            lambda: generalized_kl_divergence(np.array([-0.1, 1.0]), np.ones(2)),
            "nonnegative",
        ),
        (
            lambda: controlled_ratio_errors(np.ones((2, 1)), np.ones(2)),
            "one-dimensional",
        ),
        (
            lambda: oracle_floor_kl_sensitivity(
                np.ones(2),
                np.ones(2),
                oracle_is_exact_finite_support=False,
            ),
            "exact finite-support",
        ),
        (
            lambda: oracle_floor_kl_sensitivity(
                np.zeros(2),
                np.zeros(2),
                oracle_is_exact_finite_support=True,
            ),
            "positive",
        ),
    ],
)
def test_ratio_metric_validation(function, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        function()


def test_multi_reward_panel_is_zero_for_identical_occupancies() -> None:
    rng = np.random.default_rng(91)
    states = rng.normal(size=(40, 3))
    actions = rng.normal(size=(40, 2))
    result = estimate_multi_reward_occupancy_functional_error(
        source_states=states,
        source_actions=actions,
        candidate_weights=np.ones(40),
        target_states_a=states,
        target_actions_a=actions,
        target_states_b=states,
        target_actions_b=actions,
        reward_count=17,
        seed=8,
        evaluation_batch_size=11,
    )

    assert result.signed_cross_pool_mse == pytest.approx(0.0, abs=1e-30)
    assert result.positive_part_root_mse == pytest.approx(0.0, abs=1e-15)
    assert result.pooled_rmse == pytest.approx(0.0, abs=1e-15)
    assert result.panel.reward_count == 17
    assert sum(result.panel.rewards_per_bandwidth) == 17
    assert len(result.panel.panel_sha256) == 64


def test_multi_reward_panel_is_deterministic_and_detects_error() -> None:
    rng = np.random.default_rng(92)
    source_states = rng.normal(size=(50, 2))
    source_actions = rng.normal(size=(50, 1))
    target_states = rng.normal(loc=0.8, size=(60, 2))
    target_actions = rng.normal(loc=-0.5, size=(60, 1))
    kwargs = {
        "source_states": source_states,
        "source_actions": source_actions,
        "candidate_weights": np.ones(50),
        "target_states_a": target_states[:30],
        "target_actions_a": target_actions[:30],
        "target_states_b": target_states[30:],
        "target_actions_b": target_actions[30:],
        "reward_count": 19,
        "seed": 9,
    }
    first = estimate_multi_reward_occupancy_functional_error(**kwargs)
    second = estimate_multi_reward_occupancy_functional_error(**kwargs)

    assert second == first
    assert first.pooled_rmse > 0.0
    assert first.n_target_a == first.n_target_b == 30


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"gamma": 1.0}, "gamma"),
        ({"bin_count": 0}, "bin_count"),
        ({"ridge_scale": -1.0}, "ridge_scale"),
        (
            {"audit_transition_group_ids": np.array([0, 0, 0, 0])},
            "two audit transition groups",
        ),
        (
            {"audit_initial_group_ids": np.array([0, 0])},
            "two audit initial groups",
        ),
    ],
)
def test_cross_moment_validation(replacement: dict[str, object], message: str) -> None:
    kwargs: dict[str, object] = {
        "basis_candidate_weights": np.ones(4),
        "audit_current_weights": np.ones(4),
        "audit_next_weights": np.ones(4),
        "audit_initial_weights": np.ones(2),
        "audit_transition_group_ids": np.array([0, 0, 1, 1]),
        "audit_initial_group_ids": np.array([0, 1]),
        "gamma": 0.9,
    }
    kwargs.update(replacement)
    with pytest.raises(ValueError, match=message):
        estimate_bellman_cross_moment_error(**kwargs)


def test_independent_audit_basis_masks_preserve_group_disjointness() -> None:
    transition_groups = np.repeat(np.arange(12), 3)
    initial_groups = np.arange(12)

    basis, transition_audit, initial_audit = independent_audit_basis_masks(
        transition_group_ids=transition_groups,
        initial_group_ids=initial_groups,
        split_seed=71,
    )

    basis_groups = set(transition_groups[basis].tolist())
    moment_groups = set(transition_groups[transition_audit].tolist()) | set(
        initial_groups[initial_audit].tolist()
    )
    assert basis_groups
    assert moment_groups
    assert basis_groups.isdisjoint(moment_groups)
    np.testing.assert_array_equal(transition_audit, ~basis)
    assert np.all(np.isin(initial_groups[~initial_audit], list(basis_groups)))


def test_independent_audit_basis_masks_are_seed_deterministic() -> None:
    kwargs = {
        "transition_group_ids": np.repeat(np.arange(15), 2),
        "initial_group_ids": np.arange(15),
        "split_seed": 17,
    }
    first = independent_audit_basis_masks(**kwargs)
    second = independent_audit_basis_masks(**kwargs)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
