from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass
class BenchmarkDataset:
    """One benchmark dataset and all truth needed for diagnostics."""

    setting: str
    states: Array
    actions: Array
    next_states: Array
    target_actions: Array
    next_target_actions: Array
    rewards: Array
    true_ratio: Array | None
    initial_states: Array
    initial_actions: Array
    initial_weights: Array
    masks: Array
    gamma: float
    seed: int
    sample_size: int
    true_action_ratio: Array | None = None
    true_transition_ratio: Array | None = None
    reference_weights: Array | None = None
    episode_ids: Array | None = None
    timesteps: Array | None = None
    initial_episode_ids: Array | None = None
    is_absorbing: Array | None = None
    next_retention: Array | None = None
    initial_retention: Array | None = None
    target_occupancy_states: Array | None = None
    target_occupancy_actions: Array | None = None
    target_occupancy_episode_ids: Array | None = None
    target_occupancy_pool_ids: Array | None = None
    target_occupancy_mass: float | None = None
    target_policy_value: float | None = None
    target_policy_value_se: float | None = None
    target_policy_value_kind: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = int(np.asarray(self.states).shape[0])
        for name in (
            "actions",
            "next_states",
            "target_actions",
            "next_target_actions",
            "rewards",
            "masks",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape[0] != n:
                raise ValueError(f"{name} must have {n} rows.")
        if self.true_ratio is not None:
            value = np.asarray(self.true_ratio)
            if value.shape[0] != n:
                raise ValueError(f"true_ratio must have {n} rows.")
        initial_n = int(np.asarray(self.initial_states).shape[0])
        if np.asarray(self.initial_actions).shape[0] != initial_n:
            raise ValueError(f"initial_actions must have {initial_n} rows.")
        if np.asarray(self.initial_weights).shape[0] != initial_n:
            raise ValueError(f"initial_weights must have {initial_n} rows.")
        for name in ("episode_ids", "timesteps", "is_absorbing"):
            value = getattr(self, name)
            if value is not None and np.asarray(value).shape[0] != n:
                raise ValueError(f"{name} must have {n} rows.")
        if self.initial_episode_ids is not None and np.asarray(self.initial_episode_ids).shape[0] != initial_n:
            raise ValueError(f"initial_episode_ids must have {initial_n} rows.")
        for name, expected in (("next_retention", n), ("initial_retention", initial_n)):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float64).reshape(-1)
            if array.shape[0] != expected:
                raise ValueError(f"{name} must have {expected} rows.")
            if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
                raise ValueError(f"{name} must be finite and lie in [0, 1].")
        target_arrays = (
            self.target_occupancy_states,
            self.target_occupancy_actions,
            self.target_occupancy_episode_ids,
            self.target_occupancy_pool_ids,
        )
        supplied = [value is not None for value in target_arrays]
        if any(supplied) and not all(supplied):
            raise ValueError("target occupancy states, actions, episode ids, and pool ids must be supplied together.")
        if all(supplied):
            target_n = int(np.asarray(self.target_occupancy_states).shape[0])
            for name, value in (
                ("target_occupancy_actions", self.target_occupancy_actions),
                ("target_occupancy_episode_ids", self.target_occupancy_episode_ids),
                ("target_occupancy_pool_ids", self.target_occupancy_pool_ids),
            ):
                if np.asarray(value).shape[0] != target_n:
                    raise ValueError(f"{name} must have {target_n} rows.")
        for name in ("target_policy_value", "target_policy_value_se"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when supplied.")
        if self.target_occupancy_mass is not None:
            mass = float(self.target_occupancy_mass)
            if not np.isfinite(mass) or mass < 0.0:
                raise ValueError("target_occupancy_mass must be finite and nonnegative.")

    @property
    def n(self) -> int:
        return int(np.asarray(self.states).shape[0])

    @property
    def state_dim(self) -> int:
        return int(np.asarray(self.states).reshape(self.n, -1).shape[1])

    @property
    def action_dim(self) -> int:
        return int(np.asarray(self.actions).reshape(self.n, -1).shape[1])

    @property
    def coverage_next_retention(self) -> Array:
        if self.next_retention is None:
            return np.ones(self.n, dtype=np.float64)
        return np.asarray(self.next_retention, dtype=np.float64).reshape(-1)

    @property
    def coverage_initial_retention(self) -> Array:
        initial_n = int(np.asarray(self.initial_states).shape[0])
        if self.initial_retention is None:
            return np.ones(initial_n, dtype=np.float64)
        return np.asarray(self.initial_retention, dtype=np.float64).reshape(-1)


def as_2d(x: Array) -> Array:
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError("Expected a 1D or 2D array.")


def one_hot(indices: Array, size: int) -> Array:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    out = np.zeros((idx.shape[0], int(size)), dtype=np.float64)
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


def state_action_indices(states: Array, actions: Array, n_actions: int) -> Array:
    return np.asarray(states, dtype=np.int64).reshape(-1) * int(n_actions) + np.asarray(
        actions,
        dtype=np.int64,
    ).reshape(-1)
