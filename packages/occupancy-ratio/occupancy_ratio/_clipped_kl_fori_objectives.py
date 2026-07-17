"""Independent NumPy specification of clipped KL-FORI empirical objectives.

This module intentionally contains no fitting loop, model state, benchmark
truth, or normalization helper.  It is both the executable objective
specification and the source used by the fitted linear backend.
"""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def sigmoid(value: Array) -> Array:
    """Evaluate the logistic function without overflow."""
    x = np.asarray(value, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def bounded_log_ratio(
    raw: Array, *, tau_lower: float, tau_upper: float
) -> tuple[Array, Array]:
    """Map raw scores smoothly into the log-envelope and return derivatives.

    The mapping is an approximate bounded parameterization: finite raw scores
    cannot attain the envelope endpoints exactly.
    """
    probability = sigmoid(raw)
    lower = float(np.log(tau_lower))
    width = float(np.log(tau_upper) - lower)
    log_ratio = lower + width * probability
    derivative = width * probability * (1.0 - probability)
    return log_ratio, derivative


def uniform_raw_score(tau_lower: float, tau_upper: float) -> float:
    """Return the raw intercept yielding an initial ratio exactly equal to one."""
    lower = float(np.log(tau_lower))
    width = float(np.log(tau_upper) - lower)
    probability = -lower / width
    return float(np.log(probability) - np.log1p(-probability))


def gate_loss_from_scores(
    *,
    scores_ref: Array,
    scores_init: Array,
    scores_plus: Array,
    init_probs: Array,
    source_weights: Array,
    continuation: Array,
    gamma: float,
    tau_upper: float,
) -> float:
    """Evaluate the unbalanced weighted-logistic gate objective."""
    return float(
        float(tau_upper) * np.mean(np.logaddexp(0.0, -np.asarray(scores_ref)))
        + (1.0 - gamma)
        * np.sum(np.asarray(init_probs) * np.logaddexp(0.0, np.asarray(scores_init)))
        + gamma
        * np.mean(
            np.asarray(source_weights)
            * np.asarray(continuation)
            * np.logaddexp(0.0, np.asarray(scores_plus))
        )
    )


def projection_loss_from_log_ratios(
    *,
    log_ratio_ref: Array,
    log_ratio_init: Array,
    log_ratio_plus: Array,
    init_probs: Array,
    source_weights: Array,
    continuation: Array,
    gate_ref: Array,
    gate_init: Array,
    gate_plus: Array,
    gamma: float,
    tau_upper: float,
) -> float:
    """Evaluate the unnormalized generalized-KL projection objective."""
    return float(
        np.mean(np.exp(np.asarray(log_ratio_ref)))
        - (1.0 - gamma)
        * np.sum(
            np.asarray(init_probs) * np.asarray(gate_init) * np.asarray(log_ratio_init)
        )
        - gamma
        * np.mean(
            np.asarray(source_weights)
            * np.asarray(continuation)
            * np.asarray(gate_plus)
            * np.asarray(log_ratio_plus)
        )
        - float(tau_upper)
        * np.mean((1.0 - np.asarray(gate_ref)) * np.asarray(log_ratio_ref))
    )


def linear_gate_objective_and_grad(
    theta: Array,
    *,
    Phi_ref: Array,
    Phi_init: Array,
    Phi_plus: Array,
    init_probs: Array,
    source_weights: Array,
    continuation: Array,
    gamma: float,
    tau_upper: float,
    l2_penalty: float,
) -> tuple[float, Array]:
    """Evaluate the linear gate objective and analytic gradient."""
    score_ref = Phi_ref @ theta
    score_init = Phi_init @ theta
    score_plus = Phi_plus @ theta
    successor_weight = np.asarray(source_weights) * np.asarray(continuation)
    value = gate_loss_from_scores(
        scores_ref=score_ref,
        scores_init=score_init,
        scores_plus=score_plus,
        init_probs=init_probs,
        source_weights=source_weights,
        continuation=continuation,
        gamma=gamma,
        tau_upper=tau_upper,
    )
    penalized = np.asarray(theta, dtype=np.float64).copy()
    penalized[0] = 0.0
    value += 0.5 * float(l2_penalty) * float(penalized @ penalized)
    gradient = (
        float(tau_upper)
        * ((sigmoid(score_ref) - 1.0) @ Phi_ref)
        / Phi_ref.shape[0]
        + (1.0 - gamma)
        * ((init_probs * sigmoid(score_init)) @ Phi_init)
        + gamma
        * ((successor_weight * sigmoid(score_plus)) @ Phi_plus)
        / Phi_plus.shape[0]
        + float(l2_penalty) * penalized
    )
    return float(value), gradient


def linear_ratio_objective_and_grad(
    theta: Array,
    *,
    Phi_ref: Array,
    Phi_init: Array,
    Phi_plus: Array,
    init_probs: Array,
    source_weights: Array,
    continuation: Array,
    gate_ref: Array,
    gate_init: Array,
    gate_plus: Array,
    gamma: float,
    tau_lower: float,
    tau_upper: float,
    l2_penalty: float,
) -> tuple[float, Array]:
    """Evaluate the bounded linear projection objective and gradient."""
    h_ref, dh_ref = bounded_log_ratio(
        Phi_ref @ theta, tau_lower=tau_lower, tau_upper=tau_upper
    )
    h_init, dh_init = bounded_log_ratio(
        Phi_init @ theta, tau_lower=tau_lower, tau_upper=tau_upper
    )
    h_plus, dh_plus = bounded_log_ratio(
        Phi_plus @ theta, tau_lower=tau_lower, tau_upper=tau_upper
    )
    value = projection_loss_from_log_ratios(
        log_ratio_ref=h_ref,
        log_ratio_init=h_init,
        log_ratio_plus=h_plus,
        init_probs=init_probs,
        source_weights=source_weights,
        continuation=continuation,
        gate_ref=gate_ref,
        gate_init=gate_init,
        gate_plus=gate_plus,
        gamma=gamma,
        tau_upper=tau_upper,
    )
    successor_weight = (
        np.asarray(source_weights) * np.asarray(continuation) * np.asarray(gate_plus)
    )
    penalized = np.asarray(theta, dtype=np.float64).copy()
    penalized[0] = 0.0
    value += 0.5 * float(l2_penalty) * float(penalized @ penalized)
    gradient = (
        ((np.exp(h_ref) * dh_ref) @ Phi_ref) / Phi_ref.shape[0]
        - (1.0 - gamma)
        * ((init_probs * gate_init * dh_init) @ Phi_init)
        - gamma * ((successor_weight * dh_plus) @ Phi_plus) / Phi_plus.shape[0]
        - float(tau_upper)
        * (((1.0 - gate_ref) * dh_ref) @ Phi_ref)
        / Phi_ref.shape[0]
        + float(l2_penalty) * penalized
    )
    return float(value), gradient


__all__ = [
    "bounded_log_ratio",
    "gate_loss_from_scores",
    "linear_gate_objective_and_grad",
    "linear_ratio_objective_and_grad",
    "projection_loss_from_log_ratios",
    "sigmoid",
    "uniform_raw_score",
]
