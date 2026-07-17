"""Shared input preparation for fitted occupancy-ratio estimators.

The helpers in this module contain no estimator objective or normalization
logic.  Standard and recursively clipped KL-FORI share them so their public
state-action input contracts cannot drift independently.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np


Array = np.ndarray
TargetActionSampler = Callable[[Array, np.random.Generator], Array]


def as_2d(value: Array, name: str) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        out = arr.reshape(-1, 1)
    elif arr.ndim == 2:
        out = arr
    else:
        raise ValueError(f"{name} must be a 1D or 2D array.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values.")
    return out


def prepare_initial_rows(
    *,
    initial_states: Optional[Array],
    initial_actions: Optional[Array],
    initial_weights: Optional[Array],
    action_dim: int,
    target_policy: Any,
    target_action_sampler: Optional[TargetActionSampler],
    seed: int,
) -> tuple[Array, Array, Array, Array, str]:
    if initial_states is None:
        raise ValueError("KL-FORI requires initial_states to estimate P0 h.")
    states = as_2d(initial_states, "initial_states")
    source_row_index = np.arange(states.shape[0], dtype=np.int64)
    source = "initial_actions"
    if initial_actions is None:
        rng = np.random.default_rng(seed)
        if target_action_sampler is not None:
            actions = as_2d(
                target_action_sampler(states, rng),
                "target_action_sampler initial output",
            )
            source = "target_action_sampler"
        elif target_policy is not None:
            actions = as_2d(
                sample_policy_actions(target_policy, states, rng),
                "target_policy initial actions",
            )
            source = "target_policy"
        else:
            raise ValueError(
                "KL-FORI requires initial_actions, or a target_policy/target_action_sampler "
                "to sample initial target actions."
            )
    else:
        raw_actions = np.asarray(initial_actions, dtype=np.float64)
        if not np.all(np.isfinite(raw_actions)):
            raise ValueError("initial_actions must contain only finite values.")
        if raw_actions.ndim == 3:
            if raw_actions.shape[0] != states.shape[0]:
                raise ValueError(
                    "initial_actions must have one block per initial state."
                )
            if raw_actions.shape[2] != int(action_dim):
                raise ValueError(
                    "initial_actions must have the same feature dimension as actions."
                )
            n_samples = int(raw_actions.shape[1])
            if n_samples <= 0:
                raise ValueError(
                    "initial_actions must include at least one sample per initial state."
                )
            actions = raw_actions.reshape(states.shape[0] * n_samples, int(action_dim))
            source_row_index = np.repeat(source_row_index, n_samples)
            states = np.repeat(states, n_samples, axis=0)
        else:
            actions = as_2d(raw_actions, "initial_actions")
    if states.shape[0] != actions.shape[0]:
        raise ValueError(
            "initial_states and initial_actions must have the same number of rows."
        )
    if actions.shape[1] != int(action_dim):
        raise ValueError(
            "initial_actions must have the same feature dimension as actions."
        )
    if states.shape[0] == 0:
        raise ValueError("initial_states must be nonempty.")
    weights = (
        np.ones(states.shape[0], dtype=np.float64)
        if initial_weights is None
        else np.asarray(initial_weights, dtype=np.float64).reshape(-1)
    )
    if (
        initial_weights is not None
        and weights.shape[0] != states.shape[0]
        and weights.shape[0] > 0
        and states.shape[0] % weights.shape[0] == 0
    ):
        repeats = states.shape[0] // weights.shape[0]
        weights = np.repeat(weights / float(repeats), repeats)
    if weights.shape[0] != states.shape[0]:
        raise ValueError("initial_weights must match initial_states rows.")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("initial_weights must be finite and nonnegative.")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("initial_weights must have positive mass.")
    return states, actions, weights, source_row_index, source


def prepare_successor_actions(
    *,
    next_states: Array,
    actions: Array,
    target_next_actions: Optional[Array],
    target_policy: Any,
    target_action_sampler: Optional[TargetActionSampler],
    seed: int,
) -> tuple[Array, Array, str]:
    n_rows = next_states.shape[0]
    if target_next_actions is not None:
        value = np.asarray(target_next_actions, dtype=np.float64)
        if not np.all(np.isfinite(value)):
            raise ValueError("target_next_actions must contain only finite values.")
        if value.ndim == 1:
            value = value.reshape(-1, 1)
        if value.ndim == 2:
            if value.shape[0] != n_rows:
                raise ValueError(
                    "target_next_actions must have one row per transition."
                )
            if value.shape[1] != actions.shape[1]:
                raise ValueError(
                    "target_next_actions must match the action feature dimension."
                )
            return value, np.arange(n_rows, dtype=np.int64), "target_next_actions"
        if value.ndim == 3:
            if value.shape[0] != n_rows:
                raise ValueError(
                    "target_next_actions must have one block per transition."
                )
            if value.shape[2] != actions.shape[1]:
                raise ValueError(
                    "target_next_actions must match the action feature dimension."
                )
            n_samples = int(value.shape[1])
            if n_samples <= 0:
                raise ValueError(
                    "target_next_actions must include at least one sample per transition."
                )
            return (
                value.reshape(n_rows * n_samples, actions.shape[1]),
                np.repeat(np.arange(n_rows), n_samples),
                "target_next_actions",
            )
        raise ValueError("target_next_actions must be a 1D, 2D, or 3D array.")
    rng = np.random.default_rng(seed)
    if target_action_sampler is not None:
        value = as_2d(
            target_action_sampler(next_states, rng), "target_action_sampler output"
        )
        source = "target_action_sampler"
    elif target_policy is not None:
        value = as_2d(
            sample_policy_actions(target_policy, next_states, rng),
            "target_policy actions",
        )
        source = "target_policy"
    else:
        raise ValueError(
            "KL-FORI requires target_next_actions or a target_policy/target_action_sampler for X_i^+."
        )
    if value.shape[0] != n_rows or value.shape[1] != actions.shape[1]:
        raise ValueError(
            f"{source} output must match next_states rows and action dimension."
        )
    return value, np.arange(n_rows, dtype=np.int64), source


def sample_policy_actions(
    target_policy: Any, states: Array, rng: np.random.Generator
) -> Array:
    if hasattr(target_policy, "sample_action"):
        try:
            return np.asarray(
                target_policy.sample_action(states, rng), dtype=np.float64
            )
        except TypeError:
            return np.asarray(target_policy.sample_action(states), dtype=np.float64)
    if hasattr(target_policy, "sample"):
        try:
            return np.asarray(target_policy.sample(states, rng), dtype=np.float64)
        except TypeError:
            return np.asarray(target_policy.sample(states), dtype=np.float64)
    if hasattr(target_policy, "predict"):
        return np.asarray(target_policy.predict(states), dtype=np.float64)
    if callable(target_policy):
        try:
            return np.asarray(target_policy(states, rng), dtype=np.float64)
        except TypeError:
            return np.asarray(target_policy(states), dtype=np.float64)
    raise ValueError(
        "target_policy must be callable or expose sample_action/sample/predict."
    )


def resolve_continuation(
    *,
    n_rows: int,
    terminals: Optional[Array],
    timeouts: Optional[Array],
    handle_timeouts: str,
    absorbing_state: bool,
) -> Array:
    continuation = np.ones(int(n_rows), dtype=np.float64)
    if terminals is not None:
        terminal = np.asarray(terminals, dtype=bool).reshape(-1)
        if terminal.shape[0] != int(n_rows):
            raise ValueError("terminals must match states rows.")
        continuation[terminal] = 0.0
    if timeouts is not None and str(handle_timeouts) != "nonterminal":
        timeout = np.asarray(timeouts, dtype=bool).reshape(-1)
        if timeout.shape[0] != int(n_rows):
            raise ValueError("timeouts must match states rows.")
        continuation[timeout] = 0.0
    if absorbing_state:
        return np.ones(int(n_rows), dtype=np.float64)
    return continuation


def fit_standardizer(x: Array) -> tuple[Array, Array]:
    value = np.asarray(x, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] == 0 or not np.all(np.isfinite(value)):
        raise ValueError("standardizer input must be a nonempty finite 2D array.")
    mean = np.mean(value, axis=0)
    scale = np.std(value, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return mean.astype(np.float64), scale.astype(np.float64)


def standardize(x: Array, mean: Array, scale: Array) -> Array:
    return (
        (np.asarray(x, dtype=np.float64) - mean.reshape(1, -1)) / scale.reshape(1, -1)
    ).astype(np.float64, copy=False)


def make_features(
    x: Array, mean: Array, scale: Array, *, include_quadratic: bool
) -> Array:
    standardized = standardize(x, mean, scale)
    parts = [standardized]
    if include_quadratic:
        parts.append(standardized * standardized)
    return np.concatenate(parts, axis=1).astype(np.float64, copy=False)


def features_from_state_action(
    states: Array,
    actions: Array,
    *,
    state_dim: int,
    action_dim: int,
    mean: Array,
    scale: Array,
    include_quadratic: bool,
) -> Array:
    state_rows = as_2d(states, "states")
    action_rows = as_2d(actions, "actions")
    _validate_state_action_dimensions(state_rows, action_rows, state_dim, action_dim)
    return make_features(
        np.concatenate([state_rows, action_rows], axis=1),
        mean,
        scale,
        include_quadratic=include_quadratic,
    )


def standardized_from_state_action(
    states: Array,
    actions: Array,
    *,
    state_dim: int,
    action_dim: int,
    mean: Array,
    scale: Array,
) -> Array:
    state_rows = as_2d(states, "states")
    action_rows = as_2d(actions, "actions")
    _validate_state_action_dimensions(state_rows, action_rows, state_dim, action_dim)
    return standardize(np.concatenate([state_rows, action_rows], axis=1), mean, scale)


def normalize_weights(weights: Array, *, n_rows: int, name: str) -> Array:
    value = np.asarray(weights, dtype=np.float64).reshape(-1)
    if value.shape[0] != int(n_rows):
        raise ValueError(f"{name} has wrong length.")
    mass = float(np.sum(value))
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"{name} must have positive finite mass.")
    return value / mass


def optional_split_groups(
    groups: Optional[Array], *, n_rows: int, name: str
) -> Optional[Array]:
    if groups is None:
        return None
    value = np.asarray(groups).reshape(-1)
    if value.shape[0] != int(n_rows):
        raise ValueError(f"{name} must have {n_rows} rows.")
    return value


def train_valid_indices(
    n_rows: int, validation_fraction: float, seed: int
) -> tuple[Array, Array]:
    n_rows = int(n_rows)
    if n_rows <= 1 or validation_fraction <= 0.0:
        return np.arange(n_rows, dtype=np.int64), np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed + 53_017)
    order = rng.permutation(n_rows)
    n_valid = max(1, int(round(float(validation_fraction) * n_rows)))
    valid = np.sort(order[:n_valid]).astype(np.int64, copy=False)
    train = np.sort(order[n_valid:]).astype(np.int64, copy=False)
    if train.size == 0:
        return np.arange(n_rows, dtype=np.int64), np.array([], dtype=np.int64)
    return train, valid


def train_valid_indices_from_sources(
    source_row_index: Array, validation_fraction: float, seed: int
) -> tuple[Array, Array]:
    source = np.asarray(source_row_index).reshape(-1)
    if source.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    unique_sources = np.unique(source)
    train_sources, valid_sources = train_valid_indices(
        unique_sources.shape[0], validation_fraction, seed
    )
    if valid_sources.size == 0:
        return np.arange(source.shape[0], dtype=np.int64), np.array([], dtype=np.int64)
    train = np.flatnonzero(np.isin(source, unique_sources[train_sources])).astype(
        np.int64, copy=False
    )
    valid = np.flatnonzero(np.isin(source, unique_sources[valid_sources])).astype(
        np.int64, copy=False
    )
    if train.size == 0:
        return np.arange(source.shape[0], dtype=np.int64), np.array([], dtype=np.int64)
    return train, valid


def _validate_state_action_dimensions(
    states: Array, actions: Array, state_dim: int, action_dim: int
) -> None:
    if states.shape[0] != actions.shape[0]:
        raise ValueError("states and actions must have the same number of rows.")
    if states.shape[1] != int(state_dim):
        raise ValueError(f"states must have {state_dim} columns.")
    if actions.shape[1] != int(action_dim):
        raise ValueError(f"actions must have {action_dim} columns.")


# Private aliases keep the standard implementation diff small while making the
# shared module's public surface descriptive and testable.
_as_2d = as_2d
_prepare_initial_rows = prepare_initial_rows
_prepare_successor_actions = prepare_successor_actions
_sample_policy_actions = sample_policy_actions
_resolve_continuation = resolve_continuation
_fit_standardizer = fit_standardizer
_standardize = standardize
_make_features = make_features
_features_from_state_action = features_from_state_action
_standardized_from_state_action = standardized_from_state_action
_normalize_weights = normalize_weights
_optional_split_groups = optional_split_groups
_train_valid_indices = train_valid_indices
_train_valid_indices_from_sources = train_valid_indices_from_sources


__all__ = [
    "TargetActionSampler",
    "as_2d",
    "features_from_state_action",
    "fit_standardizer",
    "make_features",
    "normalize_weights",
    "optional_split_groups",
    "prepare_initial_rows",
    "prepare_successor_actions",
    "resolve_continuation",
    "standardize",
    "standardized_from_state_action",
    "train_valid_indices",
    "train_valid_indices_from_sources",
]
