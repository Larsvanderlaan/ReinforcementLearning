from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import fore_ratio as fore_ratio_module
from fore_ratio import (
    FORECandidateConfig,
    FittedFORERatio,
    SignedFORERatio,
    APBVSelectionResult,
    FOREFitOptions,
    _fit_candidate_library,
    _fit_signed_component,
    aggregate_pilot_selections,
    adversarial_pairwise_bellman_validation,
    deterministic_three_way_split,
    fit_selected_fore_ratio,
    fit_selected_signed_fore_ratio,
    one_hot_actions,
)


class _TableRatioModel:
    def __init__(self, state_slope: float, action_slope: float, *, nonfinite: bool = False) -> None:
        self.state_slope = state_slope
        self.action_slope = action_slope
        self.nonfinite = nonfinite
        self.logit_clip = 10.0

    def predict_state_action_log_ratio(self, states, actions, clip=True):
        del clip
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=float)
        action_ids = np.argmax(actions, axis=1)
        values = self.state_slope * states[:, 0] + self.action_slope * action_ids
        if self.nonfinite:
            values[0] = np.nan
        return values

    def predict_state_action_ratio(self, states, actions, clip=True):
        return np.exp(self.predict_state_action_log_ratio(states, actions, clip=clip))


class _Policy:
    def predict_proba(self, states):
        states = np.asarray(states, dtype=float)
        p1 = np.clip(0.35 + 0.1 * states[:, 0], 0.05, 0.95)
        return np.column_stack([1.0 - p1, p1])


def _ratio(state_slope: float, action_slope: float, *, nonfinite: bool = False):
    return FittedFORERatio(
        model=_TableRatioModel(state_slope, action_slope, nonfinite=nonfinite),
        gamma=0.8,
        n_actions=2,
        candidate=FORECandidateConfig(
            hidden_dims=(8,),
            learning_rate=1e-3,
            weight_decay=1e-3,
            num_iterations=int(10 + 100 * abs(state_slope)),
        ),
        fit_seconds=0.0,
    )


def test_one_hot_actions_does_not_encode_actions_as_ordinal_scalars():
    encoded = one_hot_actions(np.array([0, 1, 0]), n_actions=2)
    np.testing.assert_array_equal(encoded, np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))


def test_apbv_matches_direct_pairwise_score_with_exact_action_average():
    candidates = [_ratio(0.10, 0.20), _ratio(-0.05, 0.35)]
    states = np.array([[-0.5], [0.25], [0.9]])
    actions = np.array([0, 1, 0])
    next_states = np.array([[0.1], [-0.2], [0.4]])
    initial_states = np.array([[-0.25], [0.75]])
    gamma = 0.8
    policy = _Policy()

    result = adversarial_pairwise_bellman_validation(
        candidates,
        states=states,
        actions=actions,
        next_states=next_states,
        initial_states=initial_states,
        target_policy=policy,
        gamma=gamma,
    )

    wi = candidates[0].predict_normalized(states, actions)
    wj = candidates[1].predict_normalized(states, actions)
    ell_current = np.log(wi / wj)
    all_actions = np.arange(2)
    ell_next = []
    for state, probs in zip(next_states, policy.predict_proba(next_states)):
        block = np.repeat(state[None, :], 2, axis=0)
        ell = np.log(
            candidates[0].predict_normalized(block, all_actions)
            / candidates[1].predict_normalized(block, all_actions)
        )
        ell_next.append(np.sum(probs * ell))
    ell_initial = []
    for state, probs in zip(initial_states, policy.predict_proba(initial_states)):
        block = np.repeat(state[None, :], 2, axis=0)
        ell = np.log(
            candidates[0].predict_normalized(block, all_actions)
            / candidates[1].predict_normalized(block, all_actions)
        )
        ell_initial.append(np.sum(probs * ell))
    expected = (
        np.mean(wi * (ell_current - gamma * np.asarray(ell_next)))
        - (1.0 - gamma) * np.mean(ell_initial)
        - 0.25 * (1.0 - gamma) * np.mean((np.sqrt(wi) - np.sqrt(wj)) ** 2)
    )
    assert result.score_matrix[0, 1] == pytest.approx(expected)
    np.testing.assert_allclose(np.diag(result.score_matrix), 0.0, atol=1e-12)


def test_apbv_ties_select_first_predeclared_candidate():
    candidates = [_ratio(0.1, 0.2), _ratio(0.1, 0.2)]
    states = np.array([[0.0], [1.0]])
    result = adversarial_pairwise_bellman_validation(
        candidates,
        states=states,
        actions=np.array([0, 1]),
        next_states=states[::-1],
        initial_states=states,
        target_policy=_Policy(),
        gamma=0.8,
    )
    assert result.selected_index == 0


def test_apbv_rejects_nonfinite_candidate_predictions():
    states = np.array([[0.0], [1.0]])
    with pytest.raises(FloatingPointError, match="finite"):
        adversarial_pairwise_bellman_validation(
            [_ratio(0.1, 0.2), _ratio(0.1, 0.2, nonfinite=True)],
            states=states,
            actions=np.array([0, 1]),
            next_states=states[::-1],
            initial_states=states,
            target_policy=_Policy(),
            gamma=0.8,
        )


def test_fore_scaling_and_behavior_averaging_match_jasa_convention():
    ratio = _ratio(0.0, np.log(2.0))
    states = np.zeros((2, 1))
    behavior = np.array([[0.75, 0.25], [0.20, 0.80]])
    expected_normalized = np.array([0.75 * 1.0 + 0.25 * 2.0, 0.20 * 1.0 + 0.80 * 2.0])
    np.testing.assert_allclose(
        ratio.predict_state_ratio(states, behavior),
        expected_normalized / (1.0 - ratio.gamma),
    )


def test_signed_jordan_reconstruction_uses_component_masses_and_discount_scale():
    positive = _ratio(0.0, 0.0)
    negative = _ratio(0.0, np.log(2.0))
    signed = SignedFORERatio(
        gamma=0.8,
        n_actions=2,
        positive=positive,
        negative=negative,
        positive_mass=0.5,
        negative_mass=0.25,
        positive_selection=None,
        negative_selection=None,
    )
    states = np.zeros((2, 1))
    actions = np.array([0, 1])
    expected = (0.8 / 0.2) * np.array([0.5 - 0.25, 0.5 - 0.50])
    np.testing.assert_allclose(
        signed.predict_future_unnormalized(states, actions), expected
    )


def test_signed_jordan_reconstruction_covers_one_sided_and_negligible_sources():
    positive = _ratio(0.0, 0.0)
    states = np.zeros((2, 1))
    actions = np.array([0, 1])
    positive_only = SignedFORERatio(
        gamma=0.8,
        n_actions=2,
        positive=positive,
        negative=None,
        positive_mass=0.5,
        negative_mass=0.0,
        positive_selection=None,
        negative_selection=None,
    )
    negative_only = SignedFORERatio(
        gamma=0.8,
        n_actions=2,
        positive=None,
        negative=positive,
        positive_mass=0.0,
        negative_mass=0.25,
        positive_selection=None,
        negative_selection=None,
    )
    np.testing.assert_allclose(
        positive_only.predict_future_unnormalized(states, actions), 2.0
    )
    np.testing.assert_allclose(
        negative_only.predict_future_unnormalized(states, actions), -1.0
    )

    component = _fit_signed_component(
        component_weights=np.full(5, 1e-12),
        final_component_weights=np.full(5, 1e-12),
        component_name="positive",
        states=np.zeros((5, 1)),
        actions=np.array([0, 1, 0, 1, 0]),
        next_states=np.ones((5, 1)),
        fit_idx=np.array([0, 1, 2]),
        validation_idx=np.array([3, 4]),
        target_policy=_Policy(),
        gamma=0.8,
        n_actions=2,
        candidates=(_ratio(0.0, 0.0).candidate,),
        seed=11,
        options=FOREFitOptions(),
        mass_tolerance=1e-10,
    )
    assert component[0] is None
    assert component[1] is None


def test_three_way_split_is_deterministic_disjoint_and_complete():
    first = deterministic_three_way_split(101, seed=17)
    second = deterministic_three_way_split(101, seed=17)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert tuple(len(part) for part in first) == (60, 20, 21)
    combined = np.concatenate(first)
    assert np.unique(combined).size == 101
    assert set(combined) == set(range(101))


def test_candidate_library_uses_common_target_action_draws(monkeypatch):
    calls = []

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_fit(**kwargs):
        calls.append(kwargs)
        return _TableRatioModel(0.0, 0.0)

    monkeypatch.setitem(
        sys.modules,
        "occupancy_ratio",
        SimpleNamespace(KLFORIConfig=FakeConfig, fit_kl_fori_neural=fake_fit),
    )
    candidates = (
        FORECandidateConfig((8,), 1e-3, 1e-3, 10),
        FORECandidateConfig((16,), 3e-4, 1e-2, 30),
    )
    states = np.linspace(-1.0, 1.0, 10)[:, None]
    _fit_candidate_library(
        states=states,
        actions=np.arange(10) % 2,
        next_states=states[::-1],
        initial_states=states,
        initial_weights=None,
        target_policy=_Policy(),
        gamma=0.8,
        n_actions=2,
        candidates=candidates,
        seed=91,
        options=FOREFitOptions(target_action_draws=4),
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(
        calls[0]["target_next_actions"], calls[1]["target_next_actions"]
    )
    np.testing.assert_array_equal(
        calls[0]["initial_actions"], calls[1]["initial_actions"]
    )


def test_selection_uses_fit_holdout_then_refits_complete_outer_training_fold(
    monkeypatch,
):
    fitted_state_rows = []
    validation_rows = {}

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_fit(**kwargs):
        fitted_state_rows.append(np.asarray(kwargs["states"]).copy())
        return _TableRatioModel(0.01, 0.02)

    monkeypatch.setitem(
        sys.modules,
        "occupancy_ratio",
        SimpleNamespace(KLFORIConfig=FakeConfig, fit_kl_fori_neural=fake_fit),
    )
    real_apbv = fore_ratio_module.adversarial_pairwise_bellman_validation

    def capture_apbv(*args, **kwargs):
        validation_rows["transition"] = np.asarray(kwargs["states"]).copy()
        validation_rows["initial"] = np.asarray(kwargs["initial_states"]).copy()
        return real_apbv(*args, **kwargs)

    monkeypatch.setattr(
        fore_ratio_module,
        "adversarial_pairwise_bellman_validation",
        capture_apbv,
    )
    states = np.arange(101, dtype=float)[:, None]
    split = deterministic_three_way_split(101, seed=17)
    selected = fit_selected_fore_ratio(
        states=states,
        actions=np.arange(101) % 2,
        next_states=states + 0.5,
        target_policy=_Policy(),
        gamma=0.8,
        n_actions=2,
        candidates=(
            FORECandidateConfig((8,), 1e-3, 1e-3, 10),
            FORECandidateConfig((16,), 1e-3, 1e-3, 10),
        ),
        seed=31,
        split=split,
    )
    assert [rows.shape[0] for rows in fitted_state_rows] == [60, 60, 101]
    assert set(fitted_state_rows[0][:, 0]) == set(states[split[0], 0])
    assert set(fitted_state_rows[-1][:, 0]) == set(states[:, 0])
    # Adaptive signed-source construction uses this selected fit-only model,
    # not the all-row refit, on its independent final-20% validation block.
    assert selected.selection_ratio.model is not selected.ratio.model
    # The same held-out block supplies both empirical terms. This preserves
    # the A-PBV objective while making efficient use of validation data.
    np.testing.assert_array_equal(
        validation_rows["transition"], states[split[1]]
    )
    np.testing.assert_array_equal(validation_rows["initial"], states[split[1]])


def test_selector_interfaces_cannot_receive_oracle_ratios_or_estimand_truth():
    forbidden = {"oracle", "oracle_ratio", "truth", "estimand_truth"}
    for function in (
        adversarial_pairwise_bellman_validation,
        fit_selected_fore_ratio,
        fit_selected_signed_fore_ratio,
    ):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)


def test_pilot_aggregation_uses_frequency_then_predeclared_complexity_tie_break():
    configs = (
        FORECandidateConfig((64, 64), 1e-3, 1e-3, 10),
        FORECandidateConfig((64, 64), 1e-3, 1e-3, 30),
        FORECandidateConfig((128, 128), 1e-3, 1e-2, 10),
        FORECandidateConfig((128, 128), 1e-3, 1e-2, 30),
    )

    def selection(selected_index, scores):
        return APBVSelectionResult(
            candidate_ids=tuple(config.candidate_id for config in configs),
            score_matrix=np.zeros((4, 4)),
            worst_case_scores=np.asarray(scores, dtype=float),
            selected_index=selected_index,
        )

    decision = aggregate_pilot_selections(
        [
            (configs, selection(0, [0.1, 0.2, 0.3, 0.4])),
            (configs, selection(2, [0.4, 0.5, 0.1, 0.2])),
        ]
    )
    # Each base wins once and has the same mean rank, so the smaller network wins.
    assert decision["selected"]["hidden_dims"] == (64, 64)
    assert decision["uses_oracle_truth"] is False
