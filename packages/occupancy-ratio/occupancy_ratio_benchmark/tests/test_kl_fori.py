from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from occupancy_ratio_benchmark import estimators as estimators_module
    from occupancy_ratio_benchmark.config import (
        DIRECT_ESTIMATORS,
        OccupancyRatioBenchmarkConfig,
    )
    from occupancy_ratio_benchmark.discrete import make_discrete_dataset
    from occupancy_ratio_benchmark.estimators import (
        EstimatorResult,
        estimate_kl_fori,
        run_estimator,
    )
    from occupancy_ratio_benchmark.external_baselines import (
        GoogleDICERLPreflight,
        GoogleDualDICEPreflight,
    )
    from occupancy_ratio_benchmark.run import load_config_file

from occupancy_ratio import (
    KLFORIConfig,
    KLFORIModel,
    fit_discounted_occupancy_ratio,
    fit_kl_fori,
    fit_kl_fori_boosting,
    fit_kl_fori_neural,
    fit_regression_fori_lgbm,
)
from occupancy_ratio import _kl_fori_impl as kl_impl
from occupancy_ratio._kl_fori_impl import _projection_loss_from_scores, _projection_objective_and_grad
from occupancy_ratio.neural import fit_regression_fori_neural


_UNRELATED_BENCHMARK_REFACTOR = pytest.mark.skip(
    reason="requires the separate dirty-tree general benchmark refactor"
)


def _toy_arrays(n: int = 80):
    state_id = np.arange(n) % 2
    states = np.column_stack([state_id == 0, state_id == 1]).astype(np.float64)
    actions = np.zeros((n, 1), dtype=np.float64)
    next_states = states.copy()
    target_next_actions = actions.copy()
    initial_id = np.r_[np.zeros(3 * n // 4, dtype=int), np.ones(n // 4, dtype=int)]
    initial_states = np.column_stack([initial_id == 0, initial_id == 1]).astype(np.float64)
    initial_actions = np.zeros((initial_states.shape[0], 1), dtype=np.float64)
    return states, actions, next_states, target_next_actions, initial_states, initial_actions


def test_kl_projection_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(3)
    phi = rng.normal(size=(9, 4))
    phi_init = rng.normal(size=(5, 4))
    phi_plus = rng.normal(size=(7, 4))
    init_probs = np.linspace(1.0, 2.0, phi_init.shape[0])
    init_probs = init_probs / np.sum(init_probs)
    target_phi = rng.normal(size=4) * 0.2
    theta = rng.normal(size=4) * 0.1
    blocks = ((phi, None), (phi_init, init_probs), (phi_plus, None))
    value, grad = _projection_objective_and_grad(
        theta,
        Phi_ref=phi,
        target_phi=target_phi,
        l2_penalty=0.03,
        score_tikhonov_penalty=0.07,
        tikhonov_blocks=blocks,
    )
    assert np.isfinite(value)
    eps = 1e-6
    numeric = np.empty_like(theta)
    for j in range(theta.shape[0]):
        step = np.zeros_like(theta)
        step[j] = eps
        plus, _ = _projection_objective_and_grad(
            theta + step,
            Phi_ref=phi,
            target_phi=target_phi,
            l2_penalty=0.03,
            score_tikhonov_penalty=0.07,
            tikhonov_blocks=blocks,
        )
        minus, _ = _projection_objective_and_grad(
            theta - step,
            Phi_ref=phi,
            target_phi=target_phi,
            l2_penalty=0.03,
            score_tikhonov_penalty=0.07,
            tikhonov_blocks=blocks,
        )
        numeric[j] = (plus - minus) / (2.0 * eps)
    np.testing.assert_allclose(grad, numeric, atol=1e-5, rtol=1e-5)


def test_kl_projection_loss_self_normalizes_successor_term() -> None:
    scores_ref = np.array([0.0, np.log(2.0), -0.2])
    scores_init = np.array([1.5])
    scores_plus = np.array([1.0, 3.0])
    init_probs = np.array([1.0])
    weights_ref = np.array([2.0, 4.0])
    successor_row_index = np.array([0, 1])
    continuation_plus = np.array([1.0, 0.5])
    gamma = 0.7

    loss = _projection_loss_from_scores(
        scores_ref=scores_ref,
        scores_init=scores_init,
        scores_plus=scores_plus,
        init_probs=init_probs,
        weights_ref=weights_ref,
        successor_row_index=successor_row_index,
        continuation_plus=continuation_plus,
        gamma=gamma,
    )
    successor_term = np.mean(weights_ref * continuation_plus * scores_plus) / np.mean(weights_ref)
    expected = kl_impl._logmeanexp(scores_ref) - (1.0 - gamma) * scores_init[0] - gamma * successor_term
    assert loss == pytest.approx(expected)


def test_variational_log_partition_identity_at_optimal_intercept() -> None:
    scores = np.array([-3.0, -0.4, 0.2, 1.7, 4.0])
    a_star = kl_impl._logmeanexp(scores)
    variational_value = a_star - 1.0 + float(np.mean(np.exp(scores - a_star)))
    variational_score_grad = np.exp(scores - a_star) / scores.shape[0]

    assert variational_value == pytest.approx(a_star)
    np.testing.assert_allclose(variational_score_grad, kl_impl._softmax_mean(scores))
    assert float(np.sum(variational_score_grad)) == pytest.approx(1.0)


def test_variational_gauge_recenter_preserves_exp_h_minus_a() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(19)
    model = kl_impl._build_torch_mlp(torch, input_dim=3, hidden_dims=(5,))
    final_layer = kl_impl._torch_final_linear_layer(torch, model)
    reference = torch.randn(8, 3)
    with torch.no_grad():
        final_layer.weight.normal_(mean=0.0, std=0.2)
        final_layer.bias.fill_(2.5)
    a = torch.nn.Parameter(torch.tensor(1.25))

    before_scores = model(reference).reshape(-1).detach().clone()
    before_ratio_kernel = torch.exp(before_scores - a.detach()).numpy()
    shift = kl_impl._recenter_torch_score_gauge(
        torch=torch,
        model=model,
        final_linear_layer=final_layer,
        variational_a=a,
        reference_batch=reference,
    )
    after_scores = model(reference).reshape(-1).detach()
    after_ratio_kernel = torch.exp(after_scores - a.detach()).numpy()

    assert shift == pytest.approx(float(before_scores.mean().item()))
    assert float(after_scores.mean().item()) == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(after_ratio_kernel, before_ratio_kernel, rtol=1e-6, atol=1e-7)


def test_kl_fori_tikhonov_history_separates_paper_and_regularized_objectives() -> None:
    states, actions, next_states, target_next_actions, initial_states, initial_actions = _toy_arrays(20)
    penalty = 0.2
    model = fit_kl_fori(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.3,
        config=KLFORIConfig(
            backend="linear",
            num_iterations=1,
            optimizer_steps=3,
            learning_rate=0.01,
            l2_penalty=0.0,
            score_tikhonov_penalty=penalty,
            validation_fraction=0.0,
            logit_clip=None,
        ),
    )
    row = model.history[-1]
    assert row["score_tikhonov_value"] >= 0.0
    assert row["score_regularized_objective"] == pytest.approx(
        row["objective"] + penalty * row["score_tikhonov_value"]
    )
    assert row["training_objective"] == pytest.approx(row["score_regularized_objective"])


def test_kl_fori_history_records_external_selection_objective() -> None:
    states, actions, next_states, target_next_actions, initial_states, initial_actions = _toy_arrays(24)
    reward_like = np.linspace(0.0, 1.0, states.shape[0])
    seen: list[float] = []

    def selection_objective(weights: np.ndarray) -> float:
        value = float(np.mean(np.asarray(weights, dtype=np.float64).reshape(-1) * reward_like))
        seen.append(value)
        return value

    model = fit_kl_fori(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.5,
        config=KLFORIConfig(
            backend="linear",
            num_iterations=3,
            optimizer_steps=4,
            learning_rate=0.02,
            l2_penalty=0.0,
            validation_fraction=0.0,
            logit_clip=None,
            selection_objective=selection_objective,
            selection_objective_name="toy_external",
        ),
    )

    assert len(seen) == 3
    assert [row["selection_objective_name"] for row in model.history] == ["toy_external"] * 3
    np.testing.assert_allclose([row["selection_objective_value"] for row in model.history], seen)
    assert model.diagnostics["selection_objective_name"] == "toy_external"
    assert np.isfinite(float(model.diagnostics["selection_objective_best"]))


def test_kl_fori_history_reports_weight_path_divergence() -> None:
    states, actions, next_states, target_next_actions, initial_states, initial_actions = _toy_arrays(24)
    model = fit_kl_fori(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.6,
        config=KLFORIConfig(
            backend="linear",
            num_iterations=2,
            optimizer_steps=4,
            learning_rate=0.02,
            l2_penalty=0.0,
            validation_fraction=0.25,
            logit_clip=None,
            seed=4,
        ),
    )

    first, second = model.history
    for row in (first, second):
        assert row["weight_kl_to_uniform"] >= 0.0
        assert row["weight_reverse_kl_to_uniform"] >= 0.0
        assert row["weight_step_kl"] >= 0.0
        assert row["weight_step_reverse_kl"] >= 0.0
        assert row["weight_step_l1"] >= 0.0
        assert row["valid_weight_kl_to_uniform"] >= 0.0
        assert row["valid_weight_step_kl"] >= 0.0
        assert row["valid_weight_step_reverse_kl"] >= 0.0
        assert row["valid_weight_step_l1"] >= 0.0
    assert first["weight_step_kl"] == pytest.approx(first["weight_kl_to_uniform"])
    assert first["valid_weight_step_kl"] == pytest.approx(first["valid_weight_kl_to_uniform"])
    assert model.diagnostics["weight_kl_to_uniform"] == pytest.approx(second["weight_kl_to_uniform"])
    assert model.diagnostics["weight_step_kl_final"] == pytest.approx(second["weight_step_kl"])
    assert model.diagnostics["valid_weight_step_kl_final"] == pytest.approx(second["valid_weight_step_kl"])


@pytest.mark.parametrize(
    ("backend", "fit_fn", "module_name"),
    [
        ("boosting", fit_kl_fori_boosting, "lightgbm"),
        ("neural", fit_kl_fori_neural, "torch"),
    ],
)
def test_kl_fori_tikhonov_history_optional_backends(
    backend: str,
    fit_fn,
    module_name: str,
) -> None:
    pytest.importorskip(module_name)
    states, actions, next_states, target_next_actions, initial_states, initial_actions = _toy_arrays(18)
    penalty = 0.15
    model = fit_fn(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.3,
        config=KLFORIConfig(
            backend=backend,
            num_iterations=1,
            optimizer_steps=1,
            learning_rate=0.02,
            l2_penalty=0.0,
            score_tikhonov_penalty=penalty,
            validation_fraction=0.0,
            early_stopping=False,
            neural_hidden_dims=(4,),
            neural_weight_decay=0.0,
            logit_clip=None,
        ),
    )
    row = model.history[-1]
    assert row["score_tikhonov_value"] >= 0.0
    assert row["score_regularized_objective"] == pytest.approx(
        row["objective"] + penalty * row["score_tikhonov_value"]
    )


def test_kl_projection_loss_has_paper_signs() -> None:
    scores_ref = np.array([-0.2, 0.4, 0.1])
    scores_init = np.array([0.7, -0.3])
    scores_plus = np.array([0.2, -0.5, 0.8])
    init_probs = np.array([0.25, 0.75])
    weights_ref = np.array([0.5, 1.0, 2.5])
    successor_idx = np.array([0, 1, 2])
    continuation = np.array([1.0, 0.0, 1.0])
    gamma = 0.4
    expected = (
        np.log(np.mean(np.exp(scores_ref)))
        - (1.0 - gamma) * np.dot(init_probs, scores_init)
        - gamma
        * np.mean(weights_ref[successor_idx] * continuation * scores_plus)
        / np.mean(weights_ref[successor_idx])
    )
    observed = _projection_loss_from_scores(
        scores_ref=scores_ref,
        scores_init=scores_init,
        scores_plus=scores_plus,
        init_probs=init_probs,
        weights_ref=weights_ref,
        successor_row_index=successor_idx,
        continuation_plus=continuation,
        gamma=gamma,
    )
    assert observed == pytest.approx(expected)


def test_kl_fori_positive_empirically_normalized_gamma_zero_projection() -> None:
    states, actions, next_states, target_next_actions, initial_states, initial_actions = _toy_arrays()
    model = fit_kl_fori(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.0,
        config=KLFORIConfig(num_iterations=1, optimizer_steps=1200, learning_rate=0.03, l2_penalty=1e-8),
    )
    weights = model.predict_state_action_ratio(states, actions)
    assert np.all(weights > 0.0)
    assert np.mean(weights) == pytest.approx(1.0, abs=1e-6)
    expected = np.where(states[:, 0] > 0.5, 1.5, 0.5)
    assert np.mean(np.abs(weights - expected)) < 0.08
    assert model.diagnostics["algorithm"] == "kl_fori"
    assert model.diagnostics["backend"] == "linear"


def test_kl_fori_positive_gamma_identity_dynamics_recovers_initial_projection() -> None:
    states, actions, next_states, target_next_actions, initial_states, initial_actions = _toy_arrays(90)
    model = fit_kl_fori(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.5,
        config=KLFORIConfig(num_iterations=7, optimizer_steps=500, learning_rate=0.03, l2_penalty=1e-8),
    )
    weights = model.predict_state_action_ratio(states, actions)
    expected = np.where(states[:, 0] > 0.5, 1.5, 0.5)
    assert np.mean(weights) == pytest.approx(1.0, abs=1e-6)
    assert np.mean(np.abs(weights - expected)) < 0.12


def test_kl_fori_boosting_backend_is_positive_normalized_and_not_regression_fori() -> None:
    states, actions, next_states, target_next_actions, initial_states, initial_actions = _toy_arrays(60)
    model = fit_kl_fori_boosting(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.0,
        config=KLFORIConfig(
            backend="boosting",
            num_iterations=1,
            optimizer_steps=12,
            boosting_learning_rate=0.25,
            boosting_num_leaves=3,
            boosting_min_data_in_leaf=1,
            logit_clip=None,
        ),
    )
    weights = model.predict_state_action_ratio(states, actions)
    assert model.diagnostics["algorithm"] == "kl_fori"
    assert model.diagnostics["backend"] == "boosting"
    assert "k_fit" not in model.fit_payload
    assert np.all(weights > 0.0)
    assert np.mean(weights) == pytest.approx(1.0, abs=1e-6)


def test_kl_fori_model_save_load_preserves_linear_and_boosting_predictions(tmp_path) -> None:
    states, actions, next_states, target_next_actions, initial_states, initial_actions = _toy_arrays(30)
    linear = fit_kl_fori(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.0,
        config=KLFORIConfig(num_iterations=1, optimizer_steps=20),
    )
    linear_path = tmp_path / "linear.pkl"
    linear.save(linear_path)
    linear_loaded = KLFORIModel.load(linear_path)
    np.testing.assert_allclose(
        linear_loaded.predict_state_action_ratio(states, actions),
        linear.predict_state_action_ratio(states, actions),
    )

    boosting = fit_kl_fori_boosting(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.0,
        config=KLFORIConfig(
            num_iterations=1,
            optimizer_steps=2,
            boosting_learning_rate=0.2,
            boosting_num_leaves=3,
            boosting_min_data_in_leaf=1,
        ),
    )
    boosting_path = tmp_path / "boosting.pkl"
    boosting.save(boosting_path)
    boosting_loaded = KLFORIModel.load(boosting_path)
    np.testing.assert_allclose(
        boosting_loaded.predict_state_action_ratio(states, actions),
        boosting.predict_state_action_ratio(states, actions),
    )

    torch = pytest.importorskip("torch")
    del torch
    neural = fit_kl_fori_neural(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.0,
        config=KLFORIConfig(
            num_iterations=1,
            optimizer_steps=20,
            learning_rate=0.03,
            neural_hidden_dims=(4,),
            neural_weight_decay=0.0,
        ),
    )
    neural_path = tmp_path / "neural.pkl"
    neural.save(neural_path)
    neural_loaded = KLFORIModel.load(neural_path)
    np.testing.assert_allclose(
        neural_loaded.predict_state_action_ratio(states, actions),
        neural.predict_state_action_ratio(states, actions),
    )


def test_kl_fori_neural_backend_smoke() -> None:
    pytest.importorskip("torch")
    states, actions, next_states, target_next_actions, initial_states, initial_actions = _toy_arrays(40)
    model = fit_kl_fori_neural(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.0,
        config=KLFORIConfig(
            backend="neural",
            num_iterations=1,
            optimizer_steps=80,
            learning_rate=0.03,
            neural_hidden_dims=(8,),
            neural_weight_decay=0.0,
            logit_clip=None,
        ),
    )
    weights = model.predict_state_action_ratio(states, actions)
    assert model.diagnostics["backend"] == "neural"
    assert np.all(weights > 0.0)
    assert np.mean(weights) == pytest.approx(1.0, abs=1e-6)


def test_kl_fori_neural_variational_backend_smoke() -> None:
    pytest.importorskip("torch")
    states, actions, next_states, target_next_actions, initial_states, initial_actions = _toy_arrays(48)
    model = fit_kl_fori_neural(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.2,
        config=KLFORIConfig(
            backend="neural",
            neural_log_partition_mode="variational",
            neural_batch_size=16,
            neural_variational_gauge_fix="batch",
            num_iterations=2,
            optimizer_steps=5,
            learning_rate=0.01,
            validation_fraction=0.0,
            early_stopping=False,
            neural_hidden_dims=(8,),
            neural_weight_decay=0.0,
            logit_clip=None,
        ),
    )
    weights = model.predict_state_action_ratio(states, actions)
    assert model.diagnostics["backend"] == "neural"
    assert model.diagnostics["neural_log_partition_mode"] == "variational"
    assert model.diagnostics["neural_batch_size"] == 16
    assert model.diagnostics["neural_variational_gauge_fix"] == "batch"
    assert model.diagnostics["neural_variational_gauge_active"] is True
    assert model.diagnostics["neural_variational_gauge_mode"] == "batch"
    assert np.isfinite(model.diagnostics["neural_variational_a"])
    assert np.all(weights > 0.0)
    assert np.mean(weights) == pytest.approx(1.0, abs=1e-6)


def test_kl_fori_neural_variational_full_batch_backend_smoke() -> None:
    pytest.importorskip("torch")
    states, actions, next_states, target_next_actions, initial_states, initial_actions = _toy_arrays(36)
    model = fit_kl_fori_neural(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.2,
        config=KLFORIConfig(
            backend="neural",
            neural_log_partition_mode="variational",
            neural_batch_size=None,
            neural_variational_gauge_fix="full",
            num_iterations=2,
            optimizer_steps=5,
            learning_rate=0.01,
            validation_fraction=0.0,
            early_stopping=False,
            neural_hidden_dims=(8,),
            neural_weight_decay=0.0,
            logit_clip=None,
        ),
    )
    weights = model.predict_state_action_ratio(states, actions)
    assert model.diagnostics["backend"] == "neural"
    assert model.diagnostics["neural_log_partition_mode"] == "variational"
    assert model.diagnostics["neural_batch_size"] == ""
    assert model.diagnostics["neural_variational_gauge_fix"] == "full"
    assert model.diagnostics["neural_variational_gauge_active"] is True
    assert model.diagnostics["neural_variational_gauge_mode"] == "full"
    assert np.isfinite(model.diagnostics["neural_variational_a"])
    assert np.all(weights > 0.0)
    assert np.mean(weights) == pytest.approx(1.0, abs=1e-6)


def test_kl_fori_neural_early_stopping_uses_validation_projection_loss() -> None:
    pytest.importorskip("torch")
    states, actions, next_states, target_next_actions, initial_states, initial_actions = _toy_arrays(40)
    model = fit_kl_fori_neural(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.2,
        config=KLFORIConfig(
            backend="neural",
            num_iterations=5,
            optimizer_steps=1,
            learning_rate=0.03,
            validation_fraction=0.5,
            early_stopping=True,
            patience=1,
            min_improvement=1e9,
            validation_warmup_iterations=0,
            neural_hidden_dims=(4,),
            neural_weight_decay=0.0,
            logit_clip=None,
        ),
    )
    assert model.diagnostics["backend"] == "neural"
    assert model.diagnostics["early_stopping"] is True
    assert model.diagnostics["stopped_early"] is True
    assert model.diagnostics["stop_reason"] == "validation_projection_loss"
    assert model.diagnostics["iterations_completed"] < 5
    assert model.diagnostics["accepted_count"] == 0
    assert model.history[-1]["accepted"] is False
    assert model.history[-1]["validation_loss"] == model.history[-1]["selected_validation_loss"]
    assert model.history[-1]["candidate_validation_loss"] != model.history[-1]["selected_validation_loss"]


def test_kl_fori_neural_validation_loss_uses_all_3d_successor_samples(monkeypatch) -> None:
    pytest.importorskip("torch")
    states, actions, next_states, _target_next_actions, initial_states, initial_actions = _toy_arrays(30)
    target_next_actions = np.zeros((states.shape[0], 3, actions.shape[1]), dtype=np.float64)
    original_loss = kl_impl._projection_loss_from_scores
    observed_validation_lengths: list[tuple[int, int]] = []

    def wrapped_projection_loss(**kwargs):
        scores_ref = np.asarray(kwargs["scores_ref"], dtype=np.float64).reshape(-1)
        scores_plus = np.asarray(kwargs["scores_plus"], dtype=np.float64).reshape(-1)
        if 0 < scores_ref.shape[0] < states.shape[0]:
            observed_validation_lengths.append((int(scores_ref.shape[0]), int(scores_plus.shape[0])))
        return original_loss(**kwargs)

    monkeypatch.setattr(kl_impl, "_projection_loss_from_scores", wrapped_projection_loss)
    model = fit_kl_fori_neural(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.2,
        config=KLFORIConfig(
            backend="neural",
            num_iterations=1,
            optimizer_steps=1,
            learning_rate=0.02,
            validation_fraction=0.4,
            early_stopping=True,
            validation_warmup_iterations=1,
            neural_hidden_dims=(4,),
            neural_weight_decay=0.0,
            logit_clip=None,
        ),
    )
    assert model.diagnostics["successor_rows"] == 3 * states.shape[0]
    assert model.diagnostics["early_stopping"] is True
    assert observed_validation_lengths
    assert all(plus_rows == 3 * ref_rows for ref_rows, plus_rows in observed_validation_lengths)


def test_kl_fori_neural_early_stopping_uses_actual_train_validation_split() -> None:
    pytest.importorskip("torch")
    states, actions, next_states, _target_next_actions, initial_states, initial_actions = _toy_arrays(30)
    target_next_actions = np.zeros((states.shape[0], 3, actions.shape[1]), dtype=np.float64)
    model = fit_kl_fori_neural(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.2,
        config=KLFORIConfig(
            backend="neural",
            num_iterations=1,
            optimizer_steps=1,
            learning_rate=0.02,
            validation_fraction=0.4,
            early_stopping=True,
            validation_warmup_iterations=1,
            neural_hidden_dims=(4,),
            neural_weight_decay=0.0,
            logit_clip=None,
        ),
    )
    row = next(row for row in model.history if row.get("stage") == "selection")
    final = model.history[-1]
    ref_valid = max(1, int(round(0.4 * states.shape[0])))
    init_valid = max(1, int(round(0.4 * initial_states.shape[0])))
    ref_train = states.shape[0] - ref_valid
    init_train = initial_states.shape[0] - init_valid
    assert row["split_training"] is True
    assert row["objective_train_rows"] == ref_train
    assert row["objective_init_rows"] == init_train
    assert row["objective_successor_rows"] == 3 * ref_train
    assert row["validation_init_rows"] == init_valid
    assert row["objective_train_rows"] < states.shape[0]
    assert row["objective_init_rows"] < initial_states.shape[0]
    assert final["stage"] == "refit"
    assert final["split_training"] is False
    assert final["objective_train_rows"] == states.shape[0]
    assert final["objective_init_rows"] == initial_states.shape[0]
    assert model.diagnostics["refit_after_validation"] is True
    assert model.diagnostics["refit_num_iterations"] == model.diagnostics["selection_accepted_count"]


def test_kl_fori_grouped_validation_keeps_source_groups_intact() -> None:
    pytest.importorskip("torch")
    states, actions, next_states, _target_next_actions, initial_states, initial_actions = _toy_arrays(30)
    target_next_actions = np.zeros((states.shape[0], 2, actions.shape[1]), dtype=np.float64)
    groups = np.repeat(np.arange(8), [1, 2, 3, 4, 5, 6, 4, 5])
    initial_groups = np.repeat(np.arange(6), [5, 5, 5, 5, 5, 4])
    seed = 19
    train_idx, valid_idx = kl_impl._train_valid_indices_from_sources(groups, 0.25, seed)
    init_train_idx, init_valid_idx = kl_impl._train_valid_indices_from_sources(initial_groups, 0.25, seed + 71)
    assert set(groups[train_idx]).isdisjoint(set(groups[valid_idx]))
    assert set(initial_groups[init_train_idx]).isdisjoint(set(initial_groups[init_valid_idx]))
    model = fit_kl_fori_neural(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        groups=groups,
        initial_groups=initial_groups,
        gamma=0.2,
        config=KLFORIConfig(
            backend="neural",
            num_iterations=1,
            optimizer_steps=1,
            learning_rate=0.02,
            validation_fraction=0.25,
            early_stopping=True,
            validation_warmup_iterations=1,
            neural_hidden_dims=(4,),
            neural_weight_decay=0.0,
            seed=seed,
            logit_clip=None,
        ),
    )
    row = next(row for row in model.history if row.get("stage") == "selection")
    assert row["objective_train_rows"] == train_idx.size
    assert row["objective_successor_rows"] == 2 * train_idx.size
    assert row["objective_init_rows"] == init_train_idx.size
    assert row["validation_init_rows"] == init_valid_idx.size
    assert model.diagnostics["reference_split_grouped"] is True
    assert model.diagnostics["initial_split_grouped"] is True


def test_kl_fori_rejects_nonfinite_arrays_and_wrong_config_type() -> None:
    states, actions, next_states, target_next_actions, initial_states, initial_actions = _toy_arrays(12)
    bad_states = states.copy()
    bad_states[0, 0] = np.nan
    with pytest.raises(ValueError, match="states"):
        fit_kl_fori(
            states=bad_states,
            actions=actions,
            next_states=next_states,
            target_next_actions=target_next_actions,
            initial_states=initial_states,
            initial_actions=initial_actions,
            gamma=0.5,
            config=KLFORIConfig(num_iterations=1, optimizer_steps=2),
        )
    with pytest.raises(TypeError, match="KLFORIConfig"):
        fit_kl_fori(
            states=states,
            actions=actions,
            next_states=next_states,
            target_next_actions=target_next_actions,
            initial_states=initial_states,
            initial_actions=initial_actions,
            gamma=0.5,
            config={"backend": "linear"},
        )


def test_kl_fori_requires_successor_target_actions_or_sampler() -> None:
    states, actions, next_states, _, initial_states, initial_actions = _toy_arrays(12)
    with pytest.raises(ValueError, match="target_next_actions"):
        fit_kl_fori(
            states=states,
            actions=actions,
            next_states=next_states,
            initial_states=initial_states,
            initial_actions=initial_actions,
            gamma=0.5,
            config=KLFORIConfig(num_iterations=1, optimizer_steps=2),
        )


def test_kl_fori_successor_action_sampler_uses_next_states() -> None:
    states, actions, next_states, _, initial_states, initial_actions = _toy_arrays(12)
    seen: dict[str, np.ndarray] = {}

    def sampler(query_states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        del rng
        seen["states"] = np.asarray(query_states)
        return np.zeros((query_states.shape[0], 1), dtype=np.float64)

    model = fit_kl_fori(
        states=states,
        actions=actions,
        next_states=next_states,
        target_action_sampler=sampler,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.2,
        config=KLFORIConfig(num_iterations=1, optimizer_steps=2),
    )
    np.testing.assert_allclose(seen["states"], next_states)
    assert model.diagnostics["successor_action_source"] == "target_action_sampler"


def test_kl_fori_accepts_3d_successor_action_samples() -> None:
    states, actions, next_states, _target_next_actions, initial_states, initial_actions = _toy_arrays(10)
    target_next_actions = np.zeros((states.shape[0], 3, actions.shape[1]), dtype=np.float64)
    model = fit_kl_fori(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.3,
        config=KLFORIConfig(num_iterations=1, optimizer_steps=5),
    )
    assert model.diagnostics["successor_action_source"] == "target_next_actions"
    assert model.diagnostics["successor_rows"] == 3 * states.shape[0]


def test_kl_fori_accepts_3d_initial_action_samples() -> None:
    states, actions, next_states, target_next_actions, initial_states, initial_actions = _toy_arrays(10)
    initial_action_samples = np.repeat(initial_actions[:, None, :], 4, axis=1)
    model = fit_kl_fori(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_action_samples,
        gamma=0.0,
        config=KLFORIConfig(num_iterations=1, optimizer_steps=5),
    )
    assert model.diagnostics["initial_action_source"] == "initial_actions"
    assert model.diagnostics["initial_rows"] == 4 * initial_states.shape[0]
    weights = model.predict_state_action_ratio(states, actions)
    assert np.mean(weights) == pytest.approx(1.0, abs=1e-6)


def test_kl_fori_samples_initial_actions_from_target_sampler_when_missing() -> None:
    states, actions, next_states, _, initial_states, _initial_actions = _toy_arrays(12)
    seen: list[np.ndarray] = []

    def sampler(query_states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        del rng
        seen.append(np.asarray(query_states))
        return np.zeros((query_states.shape[0], 1), dtype=np.float64)

    model = fit_kl_fori(
        states=states,
        actions=actions,
        next_states=next_states,
        target_action_sampler=sampler,
        initial_states=initial_states,
        initial_actions=None,
        gamma=0.2,
        config=KLFORIConfig(num_iterations=1, optimizer_steps=2),
    )
    assert len(seen) == 2
    np.testing.assert_allclose(seen[0], initial_states)
    np.testing.assert_allclose(seen[1], next_states)
    assert model.diagnostics["initial_action_source"] == "target_action_sampler"
    assert model.diagnostics["successor_action_source"] == "target_action_sampler"


def test_kl_fori_clipped_predictions_remain_empirically_normalized() -> None:
    states, actions, next_states, target_next_actions, initial_states, initial_actions = _toy_arrays(80)
    model = fit_kl_fori(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.0,
        config=KLFORIConfig(
            num_iterations=1,
            optimizer_steps=800,
            learning_rate=0.05,
            l2_penalty=1e-8,
            logit_clip=0.05,
        ),
    )
    weights = model.predict_state_action_ratio(states, actions, clip=True)
    assert np.mean(weights) == pytest.approx(1.0, abs=1e-6)
    assert model.diagnostics["logit_clip_applied_to_iteration"] is True
    raw = np.asarray(model.fit_payload["pred_state_action_ratio_beh_raw"], dtype=np.float64)
    clipped = np.asarray(model.fit_payload["pred_state_action_ratio_beh"], dtype=np.float64)
    assert np.mean(raw) == pytest.approx(1.0, abs=1e-6)
    assert not np.allclose(raw, clipped)


@_UNRELATED_BENCHMARK_REFACTOR
def test_default_fit_discounted_occupancy_ratio_returns_kl_fori_model() -> None:
    states, actions, next_states, target_next_actions, initial_states, initial_actions = _toy_arrays(20)
    model = fit_discounted_occupancy_ratio(
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=actions,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        gamma=0.1,
        config=KLFORIConfig(num_iterations=1, optimizer_steps=10),
    )
    assert isinstance(model, KLFORIModel)
    assert model.diagnostics["algorithm"] == "kl_fori"


def test_regression_fori_aliases_remain_importable() -> None:
    assert callable(fit_kl_fori_boosting)
    assert callable(fit_kl_fori_neural)
    assert callable(fit_regression_fori_lgbm)
    assert callable(fit_regression_fori_neural)


@_UNRELATED_BENCHMARK_REFACTOR
def test_benchmark_dispatch_distinguishes_kl_and_regression_labels() -> None:
    assert {
        "kl_fori",
        "kl_fori_boosting",
        "kl_fori_neural",
        "regression_fori_lgbm",
        "regression_fori_neural",
    }.issubset(DIRECT_ESTIMATORS)
    assert {
        "kl_fori",
        "kl_fori_boosting",
        "kl_fori_neural",
    }.issubset(OccupancyRatioBenchmarkConfig(stage="smoke").estimators)
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=40, seed=5)
    config = OccupancyRatioBenchmarkConfig(
        stage="smoke",
        kl_num_iterations=2,
        boosted_num_iterations=99,
        kl_learning_rate=0.02,
        kl_neural_learning_rate=0.003,
        kl_score_tikhonov_penalty=0.17,
        kl_logit_clip=7.0,
        kl_neural_weight_decay=1e-3,
        kl_seed_offset=11,
    )
    result = run_estimator("kl_fori", dataset, config, google_preflight=None)
    assert result.estimator == "kl_fori"
    assert result.status == "ok"
    assert result.diagnostics["algorithm"] == "kl_fori"
    assert result.diagnostics["backend"] == "linear"
    assert result.diagnostics["num_iterations"] == 2
    assert result.diagnostics["kl_learning_rate"] == pytest.approx(0.02)
    assert result.diagnostics["kl_score_tikhonov_penalty"] == pytest.approx(0.17)
    assert result.diagnostics["kl_logit_clip"] == pytest.approx(7.0)
    assert result.diagnostics["kl_seed_offset"] == pytest.approx(11.0)
    assert np.all(result.weights > 0.0)
    boosted = run_estimator("kl_fori_boosting", dataset, config, google_preflight=None)
    assert boosted.status == "ok"
    assert boosted.diagnostics["algorithm"] == "kl_fori"
    assert boosted.diagnostics["backend"] == "boosting"


@_UNRELATED_BENCHMARK_REFACTOR
def test_capacity_benchmark_labels_dispatch_fixed_neural_widths(monkeypatch) -> None:
    labels = {
        "kl_fori_neural_64x64",
        "kl_fori_neural_128x128",
        "kl_fori_neural_256x256",
        "regression_fori_neural_64x64",
        "regression_fori_neural_128x128",
        "regression_fori_neural_256x256",
        "google_dualdice_64x64",
        "google_dualdice_128x128",
        "google_dualdice_256x256",
        "dice_rl_dualdice_recovered_64x64",
        "dice_rl_dualdice_recovered_128x128",
        "dice_rl_dualdice_recovered_256x256",
        "dice_rl_best_regularized_64x64",
        "dice_rl_best_regularized_128x128",
        "dice_rl_best_regularized_256x256",
    }
    assert labels.issubset(DIRECT_ESTIMATORS)
    filtered = OccupancyRatioBenchmarkConfig(
        stage="smoke",
        estimators=("kl_fori_neural_64x64", "google_dualdice_64x64", "dice_rl_dualdice_recovered_64x64"),
        include_google_dual_dice=False,
        include_dice_rl=False,
    ).resolved_estimators()
    assert filtered == ("kl_fori_neural_64x64",)

    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=30, seed=11)
    config = OccupancyRatioBenchmarkConfig(
        stage="smoke",
        estimators=tuple(sorted(labels)),
        google_num_updates=17,
        google_batch_size=19,
        dice_rl_num_steps=23,
        dice_rl_batch_size=29,
    )
    google_preflight = GoogleDualDICEPreflight(False, "skip", Path("/tmp/google-research"))
    dice_preflight = GoogleDICERLPreflight(False, "skip", Path("/tmp/dice_rl"))
    captured: dict[str, object] = {}

    def fake_result(estimator: str, diagnostics: dict[str, object] | None = None) -> EstimatorResult:
        weights = np.ones(dataset.n, dtype=np.float64)
        return EstimatorResult(
            estimator=estimator,
            status="ok",
            weights=weights,
            raw_weights=weights,
            runtime_sec=0.0,
            diagnostics={} if diagnostics is None else dict(diagnostics),
        )

    def fake_kl(_dataset, fit_config, *, backend):
        captured["kl_dims"] = tuple(fit_config.kl_neural_hidden_dims)
        captured["kl_backend"] = backend
        return fake_result("kl_fori_neural", {"algorithm": "kl_fori", "backend": backend})

    def fake_regression(_dataset, fit_config, **_kwargs):
        captured["regression_dims"] = tuple(fit_config.neural_hidden_dims)
        captured["regression_action_dims"] = tuple(fit_config.neural_action_hidden_dims)
        captured["regression_transition_dims"] = tuple(fit_config.neural_transition_hidden_dims)
        captured["regression_activation"] = fit_config.neural_activation
        return fake_result("neural_network_stable", {"algorithm": "regression_fori_neural"})

    def fake_dualdice_candidate(_dataset, _config, _preflight, candidate):
        captured["google_candidate"] = dict(candidate)
        return fake_result("google_dualdice_candidate", {"algorithm": "google_dualdice"})

    def fake_dice_candidate(_dataset, _config, _preflight, candidate):
        captured["dice_candidate"] = dict(candidate)
        return fake_result("dice_rl_candidate", {"algorithm": "dice_rl_dualdice"})

    monkeypatch.setattr(estimators_module, "estimate_kl_fori", fake_kl)
    monkeypatch.setattr(estimators_module, "estimate_neural_network", fake_regression)
    monkeypatch.setattr(estimators_module, "_estimate_dualdice_candidate", fake_dualdice_candidate)
    monkeypatch.setattr(estimators_module, "_estimate_dice_rl_candidate", fake_dice_candidate)

    kl = estimators_module.run_estimator("kl_fori_neural_128x128", dataset, config, google_preflight, dice_preflight)
    assert captured["kl_backend"] == "neural"
    assert captured["kl_dims"] == (128, 128)
    assert kl.estimator == "kl_fori_neural_128x128"
    assert kl.diagnostics["capacity_hidden_dims"] == "128x128"
    assert kl.diagnostics["capacity_match_family"] == "kl_fori"

    regression = estimators_module.run_estimator(
        "regression_fori_neural_256x256",
        dataset,
        config,
        google_preflight,
        dice_preflight,
    )
    assert captured["regression_dims"] == (256, 256)
    assert captured["regression_action_dims"] == (256, 256)
    assert captured["regression_transition_dims"] == (256, 256)
    assert captured["regression_activation"] == "relu"
    assert regression.diagnostics["capacity_match_family"] == "regression_fori"

    google = estimators_module.run_estimator("google_dualdice_64x64", dataset, config, google_preflight, dice_preflight)
    assert captured["google_candidate"]["hidden_dims"] == (64, 64)
    assert captured["google_candidate"]["num_updates"] == 17
    assert captured["google_candidate"]["batch_size"] == 19
    assert google.diagnostics["capacity_match_family"] == "google_dualdice"

    dice = estimators_module.run_estimator(
        "dice_rl_dualdice_recovered_128x128",
        dataset,
        config,
        google_preflight,
        dice_preflight,
    )
    assert captured["dice_candidate"]["hidden_dims"] == (128, 128)
    assert captured["dice_candidate"]["num_steps"] == 23
    assert captured["dice_candidate"]["batch_size"] == 29
    assert dice.diagnostics["capacity_match_family"] == "dice_rl_dualdice_recovered"

    best = estimators_module.run_estimator(
        "dice_rl_best_regularized_256x256",
        dataset,
        config,
        google_preflight,
        dice_preflight,
    )
    assert captured["dice_candidate"]["hidden_dims"] == (256, 256)
    assert best.diagnostics["capacity_match_family"] == "dice_rl_best_regularized"


@_UNRELATED_BENCHMARK_REFACTOR
def test_fori_dualdice_parity_configs_load_and_use_fixed_capacity_labels() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    loaded = {}
    for name in (
        "fori_dualdice_parity_smoke.json",
        "fori_dualdice_parity_synthetic.json",
        "fori_dualdice_parity_gym.json",
        "fori_dualdice_paper_overnight_synthetic.json",
        "fori_dualdice_paper_overnight_gym.json",
    ):
        config = load_config_file(config_dir / name)
        loaded[name] = config
        assert config.config_path is not None
        assert config.config_sha256
        assert "dualdice_gmm_tuned" not in config.estimators
        if "paper_overnight" in name:
            assert config.require_external_baselines is True
        for width in (64, 128, 256):
            assert f"kl_fori_neural_{width}x{width}" in config.estimators
            assert f"regression_fori_neural_{width}x{width}" in config.estimators
            assert f"google_dualdice_{width}x{width}" in config.estimators
            assert f"dice_rl_dualdice_recovered_{width}x{width}" in config.estimators
            assert f"dice_rl_best_regularized_{width}x{width}" in config.estimators
    assert (
        loaded["fori_dualdice_paper_overnight_synthetic.json"].output_root
        != loaded["fori_dualdice_paper_overnight_gym.json"].output_root
    )


@_UNRELATED_BENCHMARK_REFACTOR
def test_benchmark_kl_fori_uses_masks_as_terminal_continuation() -> None:
    dataset = make_discrete_dataset(setting="discrete_chain", gamma=0.5, sample_size=40, seed=7)
    masks = np.ones_like(dataset.masks, dtype=np.float64)
    masks[:10] = 0.0
    masked = replace(dataset, masks=masks)
    config = OccupancyRatioBenchmarkConfig(stage="smoke", boosted_num_iterations=1)
    result = estimate_kl_fori(masked, config)
    assert result.status == "ok"
    assert result.diagnostics["continuation_mean"] == pytest.approx(float(np.mean(masks)))
