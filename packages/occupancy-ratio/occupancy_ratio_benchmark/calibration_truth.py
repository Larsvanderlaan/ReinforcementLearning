"""Controlled-truth helpers for occupancy calibration experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from occupancy_ratio_benchmark.data import BenchmarkDataset, state_action_indices
from occupancy_ratio_benchmark.discrete import (
    exact_ratio_table,
    make_chain_mdp,
    make_grid_mdp,
    make_random_tabular_mdp,
)
from occupancy_ratio_benchmark.gaussian import (
    build_reference_joint,
    build_target_occupancy_mixture,
    make_linear_gaussian_system,
)


Array = np.ndarray
SCORE_DISTORTIONS = (
    "half_oracle",
    "double_oracle",
    "sqrt_normalized_oracle",
    "q90_clipped_oracle",
    "rank_permuted_oracle",
)


@dataclass(frozen=True)
class OracleScoreMatrices:
    """Identical conceptual-fold scores for one deterministic mechanism."""

    source_q_by_fold: Array
    next_q_by_fold: Array
    initial_q_by_fold: Array
    source_oracle: Array
    next_oracle: Array
    initial_oracle: Array
    distortion: str


def exact_ratio_at(dataset: BenchmarkDataset, states: Array, actions: Array) -> Array:
    """Evaluate the exact normalized ratio for a controlled generator."""

    if dataset.setting in {"random_tabular_mdp", "discrete_chain", "discrete_grid"}:
        n_states = int(dataset.metadata["n_states"])
        n_actions = int(dataset.metadata["n_actions"])
        shift = float(dataset.metadata.get("policy_shift", 1.0))
        if dataset.setting == "random_tabular_mdp":
            mdp = make_random_tabular_mdp(
                n_states=n_states,
                n_actions=n_actions,
                policy_shift=shift,
                seed=int(dataset.seed) + 31_337,
            )
        elif dataset.setting == "discrete_grid":
            mdp = make_grid_mdp(policy_shift=shift)
        else:
            mdp = make_chain_mdp(n_states=n_states, policy_shift=shift)
        state_index = _one_hot_index(states, n_states, "states")
        action_index = _one_hot_index(actions, n_actions, "actions")
        table = exact_ratio_table(mdp, float(dataset.gamma))
        return table.reshape(-1)[state_action_indices(state_index, action_index, n_actions)]
    if dataset.setting == "linear_gaussian":
        system = make_linear_gaussian_system(policy_shift=float(dataset.metadata["policy_shift"]))
        target = build_target_occupancy_mixture(system, float(dataset.gamma))
        reference = build_reference_joint(system)
        points = np.concatenate(
            [np.asarray(states, dtype=np.float64), np.asarray(actions, dtype=np.float64)],
            axis=1,
        )
        log_ratio = np.clip(target.logpdf(points) - reference.logpdf(points), -60.0, 60.0)
        return np.exp(log_ratio)
    raise ValueError(f"exact ratio is unavailable for setting {dataset.setting!r}")


def oracle_score_matrices(
    dataset: BenchmarkDataset,
    *,
    distortion: str,
    num_folds: int,
) -> OracleScoreMatrices:
    """Construct a predeclared exact-ratio mechanism without fitting a model."""

    name = str(distortion)
    if name not in SCORE_DISTORTIONS:
        raise ValueError(f"unknown oracle score distortion {name!r}")
    if int(num_folds) < 2:
        raise ValueError("num_folds must be at least two")
    source = exact_ratio_at(dataset, dataset.states, dataset.actions)
    next_ratio = exact_ratio_at(dataset, dataset.next_states, dataset.next_target_actions)
    initial = exact_ratio_at(dataset, dataset.initial_states, dataset.initial_actions)
    transform = _distortion_map(name, source)
    source_q = transform(source)
    next_q = transform(next_ratio)
    initial_q = transform(initial)
    return OracleScoreMatrices(
        source_q_by_fold=np.repeat(source_q[None, :], int(num_folds), axis=0),
        next_q_by_fold=np.repeat(next_q[None, :], int(num_folds), axis=0),
        initial_q_by_fold=np.repeat(initial_q[None, :], int(num_folds), axis=0),
        source_oracle=source,
        next_oracle=next_ratio,
        initial_oracle=initial,
        distortion=name,
    )


def has_exact_finite_support(dataset: BenchmarkDataset) -> bool:
    """Whether oracle-floor sensitivity is a valid finite-support analysis."""

    return dataset.setting in {"random_tabular_mdp", "discrete_chain", "discrete_grid"}


def _distortion_map(name: str, source_oracle: Array):
    source = np.asarray(source_oracle, dtype=np.float64).reshape(-1)
    if name == "half_oracle":
        return lambda value: 0.5 * np.asarray(value, dtype=np.float64)
    if name == "double_oracle":
        return lambda value: 2.0 * np.asarray(value, dtype=np.float64)
    if name == "sqrt_normalized_oracle":
        scale = float(np.mean(np.sqrt(source)))
        if scale <= 0.0:
            raise ValueError("cannot normalize a zero square-root oracle score")
        return lambda value: np.sqrt(np.asarray(value, dtype=np.float64)) / scale
    if name == "q90_clipped_oracle":
        threshold = float(np.quantile(source, 0.90))
        return lambda value: np.minimum(np.asarray(value, dtype=np.float64), threshold)
    ordered = np.sort(source)
    if ordered.size == 0:
        raise ValueError("source oracle must be nonempty")

    def reverse_rank(value: Array) -> Array:
        query = np.asarray(value, dtype=np.float64)
        rank = np.searchsorted(ordered, query, side="right") - 1
        reverse = ordered.size - 1 - np.clip(rank, 0, ordered.size - 1)
        return ordered[reverse]

    return reverse_rank


def _one_hot_index(value: Array, width: int, name: str) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != int(width):
        raise ValueError(f"{name} must be a two-dimensional one-hot matrix with width {width}")
    index = np.argmax(array, axis=1)
    expected = np.zeros_like(array)
    expected[np.arange(array.shape[0]), index] = 1.0
    if not np.allclose(array, expected, atol=1e-10, rtol=0.0):
        raise ValueError(f"{name} must contain exact one-hot rows")
    return index.astype(np.int64)


__all__ = [
    "OracleScoreMatrices",
    "SCORE_DISTORTIONS",
    "exact_ratio_at",
    "has_exact_finite_support",
    "oracle_score_matrices",
]
