"""Linear backend for recursively clipped KL-FORI.

The module owns the linear feature map and both inner optimizers.  The public
facade injects the objective callables into the optimizers so that numerical
specification tests remain independent of the backend implementation.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from occupancy_ratio import _clipped_kl_fori_backend_common as common
from occupancy_ratio import _clipped_kl_fori_diagnostics as diagnostics
from occupancy_ratio import _clipped_kl_fori_objectives as objectives
from occupancy_ratio._clipped_kl_fori_types import (
    FitResult,
    InnerOptimizerResult,
    IterationRecord,
)
from occupancy_ratio._fori_data import as_2d


Array = np.ndarray
ObjectiveAndGradient = Callable[..., tuple[float, Array]]
InnerOptimizer = Callable[..., InnerOptimizerResult]


def linear_features(
    x: Array, mean: Array, scale: Array, include_quadratic: bool
) -> Array:
    """Create the intercept, standardized linear, and optional square terms."""
    z = (np.asarray(x, dtype=np.float64) - mean.reshape(1, -1)) / scale.reshape(1, -1)
    parts = [np.ones((z.shape[0], 1), dtype=np.float64), z]
    if include_quadratic:
        parts.append(z * z)
    return np.concatenate(parts, axis=1)


def linear_features_from_state_action(
    states: Array,
    actions: Array,
    *,
    state_dim: int,
    action_dim: int,
    mean: Array,
    scale: Array,
    include_quadratic: bool,
) -> Array:
    """Validate state-action rows and create the fitted linear feature map."""
    state = as_2d(states, "states")
    action = as_2d(actions, "actions")
    if state.shape[0] != action.shape[0]:
        raise ValueError("states and actions must have the same number of rows.")
    if state.shape[1] != int(state_dim) or action.shape[1] != int(action_dim):
        raise ValueError("states or actions have the wrong feature dimension.")
    return linear_features(
        np.concatenate([state, action], axis=1), mean, scale, include_quadratic
    )


def fit_linear_gate_adam(
    *,
    coef: Array,
    learning_rate: float,
    steps: int,
    relative_tolerance: float,
    gradient_tolerance: float,
    patience: int,
    objective_and_grad: ObjectiveAndGradient = objectives.linear_gate_objective_and_grad,
    **objective_kwargs: Any,
) -> InnerOptimizerResult:
    """Minimize the weighted logistic gate objective with deterministic Adam."""
    theta = np.asarray(coef, dtype=np.float64).copy()
    first_moment = np.zeros_like(theta)
    second_moment = np.zeros_like(theta)
    previous_objective: Optional[float] = None
    stable_count = 0
    reason = "max_steps"
    objective = float("nan")
    gradient_norm = float("nan")
    for step in range(1, int(steps) + 1):
        objective, gradient = objective_and_grad(theta, **objective_kwargs)
        gradient_norm = float(np.linalg.norm(gradient))
        common.require_finite_optimizer_state(objective, gradient, theta, "linear gate")
        if gradient_norm <= float(gradient_tolerance):
            reason = "gradient_tolerance"
            break
        stable_count = common.updated_inner_stability_count(
            objective,
            previous_objective,
            relative_tolerance=relative_tolerance,
            stable_count=stable_count,
        )
        if stable_count >= int(patience):
            reason = "objective_stability"
            break
        theta, first_moment, second_moment = common.adam_step(
            theta,
            gradient,
            first_moment,
            second_moment,
            step,
            learning_rate,
        )
        common.require_finite(theta, "linear gate parameters")
        previous_objective = objective
    return InnerOptimizerResult(theta, objective, gradient_norm, step, reason)


def fit_linear_ratio_adam(
    *,
    coef: Array,
    learning_rate: float,
    steps: int,
    relative_tolerance: float,
    gradient_tolerance: float,
    patience: int,
    objective_and_grad: ObjectiveAndGradient = objectives.linear_ratio_objective_and_grad,
    **objective_kwargs: Any,
) -> InnerOptimizerResult:
    """Minimize the unnormalized generalized-KL projection with Adam."""
    theta = np.asarray(coef, dtype=np.float64).copy()
    first_moment = np.zeros_like(theta)
    second_moment = np.zeros_like(theta)
    previous_objective: Optional[float] = None
    stable_count = 0
    reason = "max_steps"
    objective = float("nan")
    gradient_norm = float("nan")
    for step in range(1, int(steps) + 1):
        objective, gradient = objective_and_grad(theta, **objective_kwargs)
        gradient_norm = float(np.linalg.norm(gradient))
        common.require_finite_optimizer_state(
            objective, gradient, theta, "linear ratio"
        )
        if gradient_norm <= float(gradient_tolerance):
            reason = "gradient_tolerance"
            break
        stable_count = common.updated_inner_stability_count(
            objective,
            previous_objective,
            relative_tolerance=relative_tolerance,
            stable_count=stable_count,
        )
        if stable_count >= int(patience):
            reason = "objective_stability"
            break
        theta, first_moment, second_moment = common.adam_step(
            theta,
            gradient,
            first_moment,
            second_moment,
            step,
            learning_rate,
        )
        common.require_finite(theta, "linear ratio parameters")
        previous_objective = objective
    return InnerOptimizerResult(theta, objective, gradient_norm, step, reason)


def fit_linear_backend(
    *,
    cfg: Any,
    gamma: float,
    X_ref: Array,
    X_init: Array,
    X_plus: Array,
    mean: Array,
    scale: Array,
    init_probs: Array,
    successor_row_index: Array,
    continuation_plus: Array,
    objective_ref_idx: Array,
    objective_init_idx: Array,
    valid_ref_idx: Array,
    valid_init_idx: Array,
    gate_optimizer: InnerOptimizer = fit_linear_gate_adam,
    ratio_optimizer: InnerOptimizer = fit_linear_ratio_adam,
) -> FitResult:
    """Run deployable outer iteration for the linear backend."""
    phi_ref = linear_features(X_ref, mean, scale, cfg.feature_include_quadratic)
    phi_init = linear_features(X_init, mean, scale, cfg.feature_include_quadratic)
    phi_plus = linear_features(X_plus, mean, scale, cfg.feature_include_quadratic)
    gate_coef = np.zeros(phi_ref.shape[1], dtype=np.float64)
    ratio_coef = np.zeros(phi_ref.shape[1], dtype=np.float64)
    ratio_coef[0] = objectives.uniform_raw_score(cfg.tau_lower, cfg.tau_upper)
    if cfg.initialization_perturbation_scale > 0.0:
        rng = np.random.default_rng(int(cfg.seed) + 91_177)
        gate_coef += rng.normal(
            scale=cfg.initialization_perturbation_scale, size=gate_coef.shape
        )
        ratio_coef += rng.normal(
            scale=cfg.initialization_perturbation_scale, size=ratio_coef.shape
        )
    history: list[dict[str, Any]] = []
    convergence_count = 0
    weights_ref = np.ones(X_ref.shape[0], dtype=np.float64)
    gate_ref = np.ones(X_ref.shape[0], dtype=np.float64)

    train_plus_idx = common.plus_indices_for_sources(
        successor_row_index, objective_ref_idx
    )
    valid_plus_idx = common.plus_indices_for_sources(successor_row_index, valid_ref_idx)
    train_init_probs = common.probability_subset(init_probs, objective_init_idx)
    valid_init_probs = (
        common.probability_subset(init_probs, valid_init_idx)
        if valid_init_idx.size
        else None
    )

    for iteration in range(int(cfg.num_iterations)):
        previous_weights = weights_ref.copy()
        previous_gate = gate_ref.copy()
        gate_result = gate_optimizer(
            coef=gate_coef,
            Phi_ref=phi_ref[objective_ref_idx],
            Phi_init=phi_init[objective_init_idx],
            Phi_plus=phi_plus[train_plus_idx],
            init_probs=train_init_probs,
            source_weights=previous_weights[successor_row_index[train_plus_idx]],
            continuation=continuation_plus[train_plus_idx],
            gamma=gamma,
            tau_upper=cfg.tau_upper,
            learning_rate=cfg.resolved_gate_learning_rate,
            steps=cfg.gate_optimizer_steps,
            l2_penalty=cfg.gate_l2_penalty,
            relative_tolerance=cfg.inner_relative_tolerance,
            gradient_tolerance=cfg.inner_gradient_tolerance,
            patience=cfg.inner_patience,
        )
        gate_coef = np.asarray(gate_result.parameters, dtype=np.float64)
        gate_score_ref = phi_ref @ gate_coef
        gate_score_init = phi_init @ gate_coef
        gate_score_plus = phi_plus @ gate_coef
        gate_ref = (gate_score_ref >= 0.0).astype(np.float64)
        gate_init = (gate_score_init >= 0.0).astype(np.float64)
        gate_plus = (gate_score_plus >= 0.0).astype(np.float64)

        # The arrays below are fixed before entering the ratio optimizer.
        ratio_result = ratio_optimizer(
            coef=ratio_coef,
            Phi_ref=phi_ref[objective_ref_idx],
            Phi_init=phi_init[objective_init_idx],
            Phi_plus=phi_plus[train_plus_idx],
            init_probs=train_init_probs,
            source_weights=previous_weights[successor_row_index[train_plus_idx]],
            continuation=continuation_plus[train_plus_idx],
            gate_ref=gate_ref[objective_ref_idx],
            gate_init=gate_init[objective_init_idx],
            gate_plus=gate_plus[train_plus_idx],
            gamma=gamma,
            tau_lower=cfg.tau_lower,
            tau_upper=cfg.tau_upper,
            learning_rate=cfg.resolved_ratio_learning_rate,
            steps=cfg.ratio_optimizer_steps,
            l2_penalty=cfg.ratio_l2_penalty,
            relative_tolerance=cfg.inner_relative_tolerance,
            gradient_tolerance=cfg.inner_gradient_tolerance,
            patience=cfg.inner_patience,
        )
        ratio_coef = np.asarray(ratio_result.parameters, dtype=np.float64)
        h_ref = objectives.bounded_log_ratio(
            phi_ref @ ratio_coef,
            tau_lower=cfg.tau_lower,
            tau_upper=cfg.tau_upper,
        )[0]
        h_init = objectives.bounded_log_ratio(
            phi_init @ ratio_coef,
            tau_lower=cfg.tau_lower,
            tau_upper=cfg.tau_upper,
        )[0]
        h_plus = objectives.bounded_log_ratio(
            phi_plus @ ratio_coef,
            tau_lower=cfg.tau_lower,
            tau_upper=cfg.tau_upper,
        )[0]
        weights_ref = np.exp(h_ref)
        common.require_finite(weights_ref, "clipped ratio iterate")
        ratio_residual = common.relative_weight_change(
            weights_ref, previous_weights, cfg.normalize_eps
        )
        gate_change = float(np.mean(gate_ref != previous_gate))
        residual = max(ratio_residual, gate_change)
        convergence_count = common.updated_convergence_count(
            cfg, iteration, residual, convergence_count
        )

        train_gate_loss = objectives.gate_loss_from_scores(
            scores_ref=gate_score_ref[objective_ref_idx],
            scores_init=gate_score_init[objective_init_idx],
            scores_plus=gate_score_plus[train_plus_idx],
            init_probs=train_init_probs,
            source_weights=previous_weights[successor_row_index[train_plus_idx]],
            continuation=continuation_plus[train_plus_idx],
            gamma=gamma,
            tau_upper=cfg.tau_upper,
        )
        train_projection_loss = objectives.projection_loss_from_log_ratios(
            log_ratio_ref=h_ref[objective_ref_idx],
            log_ratio_init=h_init[objective_init_idx],
            log_ratio_plus=h_plus[train_plus_idx],
            init_probs=train_init_probs,
            source_weights=previous_weights[successor_row_index[train_plus_idx]],
            continuation=continuation_plus[train_plus_idx],
            gate_ref=gate_ref[objective_ref_idx],
            gate_init=gate_init[objective_init_idx],
            gate_plus=gate_plus[train_plus_idx],
            gamma=gamma,
            tau_upper=cfg.tau_upper,
        )
        valid = common.validation_losses(
            valid_ref_idx=valid_ref_idx,
            valid_init_idx=valid_init_idx,
            valid_plus_idx=valid_plus_idx,
            init_probs=valid_init_probs,
            successor_row_index=successor_row_index,
            continuation_plus=continuation_plus,
            previous_weights=previous_weights,
            gate_score_ref=gate_score_ref,
            gate_score_init=gate_score_init,
            gate_score_plus=gate_score_plus,
            gate_ref=gate_ref,
            gate_init=gate_init,
            gate_plus=gate_plus,
            h_ref=h_ref,
            h_init=h_init,
            h_plus=h_plus,
            gamma=gamma,
            tau_upper=cfg.tau_upper,
        )
        history.append(
            IterationRecord(
                diagnostics.history_row(
                    iteration=iteration,
                    weights_ref=weights_ref,
                    previous_weights=previous_weights,
                    gate_ref=gate_ref,
                    gate_init=gate_init,
                    gate_plus=gate_plus,
                    init_probs=init_probs,
                    successor_row_index=successor_row_index,
                    continuation_plus=continuation_plus,
                    gamma=gamma,
                    train_gate_loss=train_gate_loss,
                    train_projection_loss=train_projection_loss,
                    valid=valid,
                    gate_result=gate_result,
                    ratio_result=ratio_result,
                    ratio_residual=ratio_residual,
                    gate_change=gate_change,
                    residual=residual,
                    cfg=cfg,
                )
            ).to_dict()
        )
        if convergence_count >= int(cfg.outer_patience):
            break

    return FitResult(
        ratio_coef=ratio_coef,
        gate_coef=gate_coef,
        ratio_neural_state_dict={},
        gate_neural_state_dict={},
        history=history,
        weights_ref=weights_ref,
        gate_ref=gate_ref,
        iterations_completed=len(history),
        neural_deterministic_enabled=False,
        neural_deterministic_error="",
    )


__all__ = [
    "fit_linear_backend",
    "fit_linear_gate_adam",
    "fit_linear_ratio_adam",
    "linear_features",
    "linear_features_from_state_action",
]
