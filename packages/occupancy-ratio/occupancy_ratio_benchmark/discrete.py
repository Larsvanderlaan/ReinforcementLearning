from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from occupancy_ratio_benchmark.data import BenchmarkDataset, one_hot, state_action_indices


Array = np.ndarray


@dataclass(frozen=True)
class DiscreteMDP:
    transition: Array
    rewards: Array
    behavior_policy: Array
    target_policy: Array
    reference_state_dist: Array

    @property
    def n_states(self) -> int:
        return int(self.transition.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.transition.shape[1])


def make_chain_mdp(n_states: int = 5, policy_shift: float = 0.65) -> DiscreteMDP:
    """Small chain where target policy moves right more often than behavior."""
    n_actions = 2
    transition = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    for state in range(n_states):
        left = max(0, state - 1)
        right = min(n_states - 1, state + 1)
        transition[state, 0, left] += 0.85
        transition[state, 0, state] += 0.15
        transition[state, 1, right] += 0.85
        transition[state, 1, state] += 0.15

    positions = np.linspace(-1.0, 1.0, n_states)
    behavior_right = 0.45 + 0.10 * positions
    target_right = behavior_right + float(policy_shift) * (0.25 + 0.10 * positions)
    behavior_right = np.clip(behavior_right, 0.05, 0.95)
    target_right = np.clip(target_right, 0.05, 0.95)
    behavior_policy = np.column_stack([1.0 - behavior_right, behavior_right])
    target_policy = np.column_stack([1.0 - target_right, target_right])
    rewards = np.column_stack([positions - 0.05, positions + 0.05])
    reference_state_dist = np.ones(n_states, dtype=np.float64) / n_states
    return DiscreteMDP(
        transition=transition,
        rewards=rewards,
        behavior_policy=behavior_policy,
        target_policy=target_policy,
        reference_state_dist=reference_state_dist,
    )


def make_grid_mdp(width: int = 3, height: int = 3, policy_shift: float = 0.45) -> DiscreteMDP:
    """Tiny gridworld with four actions and a target biased toward the goal."""
    n_states = width * height
    n_actions = 4
    transition = np.zeros((n_states, n_actions, n_states), dtype=np.float64)

    def idx(x: int, y: int) -> int:
        return y * width + x

    moves = ((0, -1), (0, 1), (-1, 0), (1, 0))
    for y in range(height):
        for x in range(width):
            state = idx(x, y)
            for action, (dx, dy) in enumerate(moves):
                nx = min(width - 1, max(0, x + dx))
                ny = min(height - 1, max(0, y + dy))
                transition[state, action, idx(nx, ny)] += 0.80
                transition[state, action, state] += 0.20

    behavior_policy = np.ones((n_states, n_actions), dtype=np.float64) / n_actions
    target_policy = behavior_policy.copy()
    rewards = np.zeros((n_states, n_actions), dtype=np.float64)
    goal = np.array([width - 1, height - 1], dtype=np.float64)
    for y in range(height):
        for x in range(width):
            state = idx(x, y)
            to_goal = goal - np.array([x, y], dtype=np.float64)
            action_scores = np.array([-to_goal[1], to_goal[1], -to_goal[0], to_goal[0]], dtype=np.float64)
            logits = float(policy_shift) * action_scores
            logits -= np.max(logits)
            target_policy[state] = np.exp(logits) / np.sum(np.exp(logits))
            rewards[state, :] = -np.linalg.norm(to_goal, ord=1)
    reference_state_dist = np.ones(n_states, dtype=np.float64) / n_states
    return DiscreteMDP(
        transition=transition,
        rewards=rewards,
        behavior_policy=behavior_policy,
        target_policy=target_policy,
        reference_state_dist=reference_state_dist,
    )


def make_random_tabular_mdp(
    *,
    n_states: int = 20,
    n_actions: int = 2,
    policy_shift: float = 1.0,
    transition_concentration: float = 1.0,
    overlap_floor: float = 0.02,
    seed: int = 0,
) -> DiscreteMDP:
    """Full-support random finite MDP with controlled target-behavior shift."""
    rng = np.random.default_rng(int(seed))
    transition = rng.dirichlet(
        np.full(int(n_states), float(transition_concentration), dtype=np.float64),
        size=(int(n_states), int(n_actions)),
    ).reshape(int(n_states), int(n_actions), int(n_states))
    behavior_logits = rng.normal(scale=0.7, size=(int(n_states), int(n_actions)))
    shift_logits = rng.normal(scale=1.0, size=(int(n_states), int(n_actions)))
    behavior_policy = _floor_probabilities(_softmax(behavior_logits, axis=1), overlap_floor)
    target_policy = _floor_probabilities(
        _softmax(behavior_logits + float(policy_shift) * shift_logits, axis=1),
        overlap_floor,
    )
    reference_state_dist = _floor_probabilities(
        rng.dirichlet(np.ones(int(n_states), dtype=np.float64)),
        overlap_floor / max(int(n_states), 1),
    )
    rewards = rng.uniform(-1.0, 1.0, size=(int(n_states), int(n_actions)))
    return DiscreteMDP(
        transition=transition,
        rewards=rewards,
        behavior_policy=behavior_policy,
        target_policy=target_policy,
        reference_state_dist=reference_state_dist,
    )


def induced_state_transition(mdp: DiscreteMDP, policy: Array) -> Array:
    return np.einsum("sa,san->sn", policy, mdp.transition)


def exact_discounted_state_occupancy(mdp: DiscreteMDP, gamma: float) -> Array:
    p_pi = induced_state_transition(mdp, mdp.target_policy)
    system = np.eye(mdp.n_states, dtype=np.float64) - float(gamma) * p_pi.T
    rhs = (1.0 - float(gamma)) * mdp.reference_state_dist
    occupancy = np.linalg.solve(system, rhs)
    occupancy = np.maximum(occupancy, 0.0)
    return occupancy / np.sum(occupancy)


def exact_ratio_table(mdp: DiscreteMDP, gamma: float) -> Array:
    target_state = exact_discounted_state_occupancy(mdp, gamma)
    target_joint = target_state[:, None] * mdp.target_policy
    reference_joint = mdp.reference_state_dist[:, None] * mdp.behavior_policy
    return target_joint / np.maximum(reference_joint, 1e-12)


def sample_policy_actions(policy: Array, states: Array, rng: np.random.Generator) -> Array:
    probs = policy[np.asarray(states, dtype=np.int64).reshape(-1)]
    cdf = np.cumsum(probs, axis=1)
    draws = rng.random(size=probs.shape[0])[:, None]
    return (draws > cdf[:, :-1]).sum(axis=1).astype(np.int64)


def sample_next_states(mdp: DiscreteMDP, states: Array, actions: Array, rng: np.random.Generator) -> Array:
    probs = mdp.transition[np.asarray(states, dtype=np.int64), np.asarray(actions, dtype=np.int64)]
    cdf = np.cumsum(probs, axis=1)
    draws = rng.random(size=probs.shape[0])[:, None]
    return (draws > cdf[:, :-1]).sum(axis=1).astype(np.int64)


def make_discrete_dataset(
    *,
    setting: str,
    gamma: float,
    sample_size: int,
    seed: int,
    policy_shift: float | None = None,
    n_states: int | None = None,
    n_actions: int | None = None,
    sample_seed: int | None = None,
) -> BenchmarkDataset:
    if setting == "random_tabular_mdp":
        shift = 1.0 if policy_shift is None else float(policy_shift)
        mdp = make_random_tabular_mdp(
            n_states=20 if n_states is None else int(n_states),
            n_actions=2 if n_actions is None else int(n_actions),
            policy_shift=shift,
            seed=int(seed + 31_337),
        )
    elif policy_shift is None:
        mdp = make_grid_mdp() if setting == "discrete_grid" else make_chain_mdp()
    else:
        shift = float(policy_shift)
        mdp = make_grid_mdp(policy_shift=shift) if setting == "discrete_grid" else make_chain_mdp(policy_shift=shift)
    rng = np.random.default_rng(seed if sample_seed is None else int(sample_seed))
    states_i = rng.choice(mdp.n_states, size=int(sample_size), p=mdp.reference_state_dist)
    actions_i = sample_policy_actions(mdp.behavior_policy, states_i, rng)
    next_states_i = sample_next_states(mdp, states_i, actions_i, rng)
    target_actions_i = sample_policy_actions(mdp.target_policy, states_i, rng)
    next_target_actions_i = sample_policy_actions(mdp.target_policy, next_states_i, rng)
    initial_states_i = rng.choice(mdp.n_states, size=max(256, min(2_000, sample_size)), p=mdp.reference_state_dist)
    initial_actions_i = sample_policy_actions(mdp.target_policy, initial_states_i, rng)

    ratio_table = exact_ratio_table(mdp, gamma)
    target_state_occupancy = exact_discounted_state_occupancy(mdp, gamma)
    target_policy_value = float(
        np.sum(target_state_occupancy[:, None] * mdp.target_policy * mdp.rewards)
    )
    row_idx = state_action_indices(states_i, actions_i, mdp.n_actions)
    true_ratio = ratio_table.reshape(-1)[row_idx]
    true_action_ratio = (
        mdp.target_policy[states_i, actions_i] / np.maximum(mdp.behavior_policy[states_i, actions_i], 1e-12)
    )
    true_transition_ratio = (
        mdp.transition[states_i, actions_i, next_states_i]
        / np.maximum(mdp.reference_state_dist[next_states_i], 1e-12)
    )

    rewards = mdp.rewards[states_i, actions_i]
    metadata = {
        "truth_source": "tabular_linear_solve",
        "reference_distribution": "fixed_reference_state_distribution",
        "n_states": mdp.n_states,
        "n_actions": mdp.n_actions,
        "sample_seed": int(seed if sample_seed is None else sample_seed),
    }
    if policy_shift is not None:
        metadata["policy_shift"] = float(policy_shift)
    if setting == "random_tabular_mdp":
        metadata["transition_source"] = "random_dirichlet"

    return BenchmarkDataset(
        setting=setting,
        states=one_hot(states_i, mdp.n_states),
        actions=one_hot(actions_i, mdp.n_actions),
        next_states=one_hot(next_states_i, mdp.n_states),
        target_actions=one_hot(target_actions_i, mdp.n_actions),
        next_target_actions=one_hot(next_target_actions_i, mdp.n_actions),
        rewards=rewards,
        true_ratio=true_ratio,
        true_action_ratio=true_action_ratio,
        true_transition_ratio=true_transition_ratio,
        initial_states=one_hot(initial_states_i, mdp.n_states),
        initial_actions=one_hot(initial_actions_i, mdp.n_actions),
        initial_weights=np.ones(initial_states_i.shape[0], dtype=np.float64),
        masks=np.ones(int(sample_size), dtype=np.float64),
        gamma=float(gamma),
        seed=int(seed),
        sample_size=int(sample_size),
        target_policy_value=target_policy_value,
        target_policy_value_se=0.0,
        target_policy_value_kind="exact_normalized_discounted_step_reward",
        metadata=metadata,
    )


def _softmax(values: Array, axis: int) -> Array:
    shifted = np.asarray(values, dtype=np.float64) - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _floor_probabilities(probabilities: Array, floor: float) -> Array:
    probs = np.maximum(np.asarray(probabilities, dtype=np.float64), float(floor))
    return probs / np.sum(probs, axis=-1, keepdims=True)
