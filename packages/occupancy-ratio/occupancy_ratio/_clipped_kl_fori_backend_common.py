"""Shared numerical helpers for clipped KL-FORI backends."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from occupancy_ratio import _clipped_kl_fori_objectives as objectives


Array = np.ndarray


def validation_losses(**kwargs: Any) -> dict[str, float]:
    """Evaluate held-out objectives without selecting on them."""
    valid_ref_idx = kwargs["valid_ref_idx"]
    valid_init_idx = kwargs["valid_init_idx"]
    valid_plus_idx = kwargs["valid_plus_idx"]
    init_probs = kwargs["init_probs"]
    if not (
        valid_ref_idx.size
        and valid_init_idx.size
        and valid_plus_idx.size
        and init_probs is not None
    ):
        return {"valid_gate_loss": float("nan"), "valid_projection_loss": float("nan")}
    source = kwargs["successor_row_index"][valid_plus_idx]
    return {
        "valid_gate_loss": objectives.gate_loss_from_scores(
            scores_ref=kwargs["gate_score_ref"][valid_ref_idx],
            scores_init=kwargs["gate_score_init"][valid_init_idx],
            scores_plus=kwargs["gate_score_plus"][valid_plus_idx],
            init_probs=init_probs,
            source_weights=kwargs["previous_weights"][source],
            continuation=kwargs["continuation_plus"][valid_plus_idx],
            gamma=kwargs["gamma"],
            tau_upper=kwargs["tau_upper"],
        ),
        "valid_projection_loss": objectives.projection_loss_from_log_ratios(
            log_ratio_ref=kwargs["h_ref"][valid_ref_idx],
            log_ratio_init=kwargs["h_init"][valid_init_idx],
            log_ratio_plus=kwargs["h_plus"][valid_plus_idx],
            init_probs=init_probs,
            source_weights=kwargs["previous_weights"][source],
            continuation=kwargs["continuation_plus"][valid_plus_idx],
            gate_ref=kwargs["gate_ref"][valid_ref_idx],
            gate_init=kwargs["gate_init"][valid_init_idx],
            gate_plus=kwargs["gate_plus"][valid_plus_idx],
            gamma=kwargs["gamma"],
            tau_upper=kwargs["tau_upper"],
        ),
    }


def normalize_probability(weight: Array, name: str) -> Array:
    value = np.asarray(weight, dtype=np.float64).reshape(-1)
    total = float(np.sum(value))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have positive finite mass.")
    return value / total


def probability_subset(probability: Array, index: Array) -> Array:
    index = np.asarray(index, dtype=np.int64)
    if index.size == 0:
        return np.array([], dtype=np.float64)
    return normalize_probability(np.asarray(probability)[index], "probability subset")


def plus_indices_for_sources(successor_row_index: Array, source_index: Array) -> Array:
    if np.asarray(source_index).size == 0:
        return np.array([], dtype=np.int64)
    return np.flatnonzero(
        np.isin(successor_row_index, np.asarray(source_index))
    ).astype(np.int64)


def relative_weight_change(current: Array, previous: Array, eps: float) -> float:
    return float(
        np.mean(np.abs(np.asarray(current) - np.asarray(previous)))
        / (np.mean(np.abs(previous)) + eps)
    )


def updated_convergence_count(
    cfg: Any, iteration: int, residual: float, previous_count: int
) -> int:
    if cfg.outer_tolerance is None or iteration + 1 < int(cfg.min_iterations):
        return 0
    return previous_count + 1 if residual <= float(cfg.outer_tolerance) else 0


def require_finite(value: Array, name: str) -> None:
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(f"{name} contains nonfinite values.")


def require_finite_optimizer_state(
    objective: float, gradient: Array, parameters: Array, name: str
) -> None:
    if not np.isfinite(objective):
        raise FloatingPointError(f"{name} objective is nonfinite.")
    require_finite(np.asarray(gradient), f"{name} gradient")
    require_finite(np.asarray(parameters), f"{name} parameters")


def updated_inner_stability_count(
    objective: float,
    previous_objective: Optional[float],
    *,
    relative_tolerance: float,
    stable_count: int,
) -> int:
    if previous_objective is None:
        return 0
    relative_change = abs(float(objective) - float(previous_objective)) / max(
        1.0, abs(float(previous_objective))
    )
    return stable_count + 1 if relative_change <= float(relative_tolerance) else 0


def adam_step(
    theta: Array,
    gradient: Array,
    first_moment: Array,
    second_moment: Array,
    step: int,
    learning_rate: float,
) -> tuple[Array, Array, Array]:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
    second_moment = beta2 * second_moment + (1.0 - beta2) * (gradient * gradient)
    first_hat = first_moment / (1.0 - beta1**step)
    second_hat = second_moment / (1.0 - beta2**step)
    theta = theta - float(learning_rate) * first_hat / (np.sqrt(second_hat) + eps)
    return theta, first_moment, second_moment


__all__ = [
    "adam_step",
    "normalize_probability",
    "plus_indices_for_sources",
    "probability_subset",
    "relative_weight_change",
    "require_finite",
    "require_finite_optimizer_state",
    "updated_convergence_count",
    "updated_inner_stability_count",
    "validation_losses",
]
