from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class AbsorbingOccupancySample:
    """I.i.d. rows from a normalized discounted episodic occupancy."""

    states: Array
    actions: Array
    next_states: Array
    rewards: Array
    masks: Array
    episode_ids: Array
    timesteps: Array
    source_indices: Array
    is_absorbing: Array
    next_is_absorbing: Array


def sample_absorbing_discounted_occupancy(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    rewards: Array,
    episode_ids: Array,
    timesteps: Array,
    gamma: float,
    sample_size: int,
    rng: np.random.Generator,
    replace_episodes: bool = True,
) -> AbsorbingOccupancySample:
    """Sample normalized discounted occupancy rows with an absorbing tail.

    An episode is sampled uniformly and then ``T ~ Geometric(1-gamma)-1``.
    If ``T`` lies beyond the observed episode, the returned row is the
    absorbing zero-reward self-loop. The final observed row of every episode
    transitions to absorption. A final binary feature identifies absorption.
    """

    gamma_f = float(gamma)
    if not 0.0 <= gamma_f < 1.0:
        raise ValueError("gamma must be in [0, 1).")
    if int(sample_size) <= 0:
        raise ValueError("sample_size must be positive.")

    state_arr = _as_2d(states, "states")
    action_arr = _as_2d(actions, "actions")
    next_state_arr = _as_2d(next_states, "next_states")
    reward_arr = np.asarray(rewards, dtype=np.float64).reshape(-1)
    episode_arr = np.asarray(episode_ids).reshape(-1)
    time_arr = np.asarray(timesteps, dtype=np.int64).reshape(-1)
    n = state_arr.shape[0]
    for name, arr in (
        ("actions", action_arr),
        ("next_states", next_state_arr),
        ("rewards", reward_arr),
        ("episode_ids", episode_arr),
        ("timesteps", time_arr),
    ):
        if arr.shape[0] != n:
            raise ValueError(f"{name} must have {n} rows.")
    if next_state_arr.shape[1] != state_arr.shape[1]:
        raise ValueError("states and next_states must have the same feature dimension.")
    if n == 0:
        raise ValueError("at least one transition is required.")
    if not np.all(np.isfinite(state_arr)) or not np.all(np.isfinite(next_state_arr)):
        raise ValueError("states and next_states must be finite.")
    if not np.all(np.isfinite(action_arr)) or not np.all(np.isfinite(reward_arr)):
        raise ValueError("actions and rewards must be finite.")
    if np.any(time_arr < 0):
        raise ValueError("timesteps must be nonnegative.")

    episode_rows = _episode_row_lookup(episode_arr, time_arr)
    episode_keys = np.asarray(list(episode_rows))
    if not replace_episodes and int(sample_size) > episode_keys.shape[0]:
        raise ValueError("sample_size cannot exceed the episode count without replacement.")
    chosen_episode_pos = rng.choice(
        episode_keys.shape[0],
        size=int(sample_size),
        replace=bool(replace_episodes),
    )
    chosen_episodes = episode_keys[chosen_episode_pos]
    if gamma_f == 0.0:
        sampled_times = np.zeros(int(sample_size), dtype=np.int64)
    else:
        sampled_times = rng.geometric(1.0 - gamma_f, size=int(sample_size)).astype(np.int64) - 1

    state_dim = state_arr.shape[1]
    action_dim = action_arr.shape[1]
    absorbing_state = np.zeros(state_dim + 1, dtype=np.float64)
    absorbing_state[-1] = 1.0
    sampled_states = np.repeat(absorbing_state[None, :], int(sample_size), axis=0)
    sampled_next_states = sampled_states.copy()
    sampled_actions = np.zeros((int(sample_size), action_dim), dtype=np.float64)
    sampled_rewards = np.zeros(int(sample_size), dtype=np.float64)
    source_indices = np.full(int(sample_size), -1, dtype=np.int64)
    is_absorbing = np.ones(int(sample_size), dtype=bool)
    next_is_absorbing = np.ones(int(sample_size), dtype=bool)

    for out_idx, (episode, timestep) in enumerate(zip(chosen_episodes, sampled_times)):
        rows = episode_rows[episode]
        row_idx = rows.get(int(timestep))
        if row_idx is None:
            continue
        source_indices[out_idx] = int(row_idx)
        is_absorbing[out_idx] = False
        sampled_states[out_idx, :-1] = state_arr[row_idx]
        sampled_states[out_idx, -1] = 0.0
        sampled_actions[out_idx] = action_arr[row_idx]
        sampled_rewards[out_idx] = reward_arr[row_idx]
        if int(timestep) < max(rows):
            sampled_next_states[out_idx, :-1] = next_state_arr[row_idx]
            sampled_next_states[out_idx, -1] = 0.0
            next_is_absorbing[out_idx] = False

    return AbsorbingOccupancySample(
        states=sampled_states,
        actions=sampled_actions,
        next_states=sampled_next_states,
        rewards=sampled_rewards,
        masks=np.ones(int(sample_size), dtype=np.float64),
        episode_ids=np.asarray(chosen_episodes),
        timesteps=sampled_times,
        source_indices=source_indices,
        is_absorbing=is_absorbing,
        next_is_absorbing=next_is_absorbing,
    )


def append_nonabsorbing_indicator(states: Array) -> Array:
    """Append a zero absorbing-state indicator to ordinary states."""

    arr = _as_2d(states, "states")
    return np.column_stack([arr, np.zeros(arr.shape[0], dtype=np.float64)])


def _episode_row_lookup(episode_ids: Array, timesteps: Array) -> dict[object, dict[int, int]]:
    lookup: dict[object, dict[int, int]] = {}
    for row_idx, (episode, timestep) in enumerate(zip(episode_ids, timesteps)):
        key = episode.item() if isinstance(episode, np.generic) else episode
        rows = lookup.setdefault(key, {})
        time = int(timestep)
        if time in rows:
            raise ValueError(f"episode {key!r} has duplicate timestep {time}.")
        rows[time] = int(row_idx)
    for episode, rows in lookup.items():
        expected = set(range(max(rows) + 1))
        if set(rows) != expected:
            raise ValueError(f"episode {episode!r} must contain contiguous timesteps starting at zero.")
    return lookup


def _as_2d(values: Array, name: str) -> Array:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")
    return arr
