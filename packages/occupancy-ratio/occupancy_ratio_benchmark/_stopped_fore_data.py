"""Structural-support data for the external-test stopped-FORE benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


Array = np.ndarray
DEFAULT_SUPPORT_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_FAILURE_MODES = ("initial", "successor")


@dataclass(frozen=True)
class StructuralStoppedTruth:
    """Exact finite-measure stopped occupancy for one shared-hub process."""

    gamma: float
    context_probability: Array
    initial_behavior_probability: Array
    hub_behavior_probability: Array
    context_reward: Array
    failure_mode: str
    support_fraction: float

    @property
    def initial_supported(self) -> Array:
        return self.initial_behavior_probability > 0.0

    @property
    def hub_supported(self) -> Array:
        return self.hub_behavior_probability > 0.0

    @property
    def fully_supported(self) -> Array:
        return self.initial_supported & self.hub_supported

    @property
    def retained_mass(self) -> float:
        initial_mass = (1.0 - self.gamma) * float(
            self.context_probability @ self.initial_supported.astype(float)
        )
        hub_mass = self.gamma * float(
            self.context_probability @ self.fully_supported.astype(float)
        )
        return initial_mass + hub_mass

    @property
    def max_positive_ratio(self) -> float:
        probabilities = np.concatenate(
            [self.initial_behavior_probability, self.hub_behavior_probability]
        )
        positive = probabilities[probabilities > 0.0]
        return 0.0 if positive.size == 0 else float(np.max(1.0 / positive))

    def ratio(self, context_id: Array, stage: Array, action: Array) -> Array:
        """Evaluate the exact stopped ratio on reference-distribution rows."""
        context = np.asarray(context_id, dtype=np.int64).reshape(-1)
        stage_array = np.asarray(stage, dtype=np.int64).reshape(-1)
        action_array = np.asarray(action, dtype=np.int64).reshape(-1)
        if not (context.shape == stage_array.shape == action_array.shape):
            raise ValueError("context_id, stage, and action must have equal shapes.")
        if np.any((stage_array < 0) | (stage_array > 1)):
            raise ValueError("stage must be 0 (initial) or 1 (hub).")
        if np.any((action_array < 0) | (action_array > 1)):
            raise ValueError("action must be 0 (target) or 1 (alternative).")

        ratio = np.zeros(context.shape[0], dtype=np.float64)
        initial = (stage_array == 0) & (action_array == 0)
        initial_probability = self.initial_behavior_probability[context]
        initial_positive = initial & (initial_probability > 0.0)
        ratio[initial_positive] = 1.0 / initial_probability[initial_positive]

        hub = (stage_array == 1) & (action_array == 0)
        hub_probability = self.hub_behavior_probability[context]
        reaches_hub = self.fully_supported[context]
        hub_positive = hub & reaches_hub & (hub_probability > 0.0)
        ratio[hub_positive] = 1.0 / hub_probability[hub_positive]
        return ratio

    def values(self) -> dict[str, float]:
        """Return exact stopped values for the benchmark reward functions."""
        initial_weight = self.context_probability * self.initial_supported
        hub_weight = self.context_probability * self.fully_supported
        return {
            "constant": self.retained_mass,
            "initial": float((1.0 - self.gamma) * np.sum(initial_weight)),
            "hub": float(self.gamma * np.sum(hub_weight)),
            "context": float(
                (1.0 - self.gamma) * (initial_weight @ self.context_reward)
                + self.gamma * (hub_weight @ self.context_reward)
            ),
        }


@dataclass(frozen=True)
class StructuralStoppedDataset:
    """Independent source and target-moment rows with exact stopped truth."""

    states: Array
    actions: Array
    next_states: Array
    target_next_actions: Array
    initial_states: Array
    initial_actions: Array
    context_id: Array
    stage: Array
    action_id: Array
    rewards: dict[str, Array]
    stopped_ratio: Array
    truth: StructuralStoppedTruth

    @property
    def n(self) -> int:
        return int(self.states.shape[0])


@dataclass(frozen=True)
class StoppedFOREExternalConfig:
    """Configuration for a train-once, external-test stopped-FORE study."""

    n_train: int = 10_000
    n_test: int = 50_000
    repetitions: int = 5
    gamma: float = 0.95
    contexts: int = 8
    behavior_probability: float = 0.25
    support_fractions: Sequence[float] = DEFAULT_SUPPORT_FRACTIONS
    failure_modes: Sequence[str] = DEFAULT_FAILURE_MODES
    backends: Sequence[str] = ("neural",)
    methods: Sequence[str] = (
        "stopped_fori_learned_gate",
        "standard_fori",
        "posthoc_winsorized",
    )
    tau_lower: float = 1e-6
    tau_upper: float = 20.0
    seed: int = 83_000
    irrelevant_features: int = 0
    stopped_num_iterations: int = 60
    stopped_gate_steps: int = 100
    stopped_ratio_steps: int = 100
    stopped_gate_learning_rate: float | None = None
    stopped_ratio_learning_rate: float | None = None
    stopped_hidden_dims: Sequence[int] = (64, 64)
    standard_num_iterations: int = 30
    standard_optimizer_steps: int = 100
    standard_hidden_dims: Sequence[int] = (64, 64)
    validation_fraction: float = 0.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.n_train <= 0 or self.n_test <= 0 or self.repetitions <= 0:
            raise ValueError("n_train, n_test, and repetitions must be positive.")
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError("gamma must be in [0, 1).")
        if self.contexts <= 0:
            raise ValueError("contexts must be positive.")
        if not (0.0 < self.behavior_probability <= 1.0):
            raise ValueError("behavior_probability must lie in (0, 1].")
        if not (0.0 < self.tau_lower <= 1.0 <= self.tau_upper):
            raise ValueError(
                "the numerical envelope must satisfy 0 < tau_lower <= 1 <= tau_upper."
            )
        if self.tau_upper <= 1.0 / self.behavior_probability:
            raise ValueError(
                "tau_upper must exceed every positive true ratio so upper clipping "
                "is inactive in the main stopped-ratio experiment."
            )
        fractions = tuple(float(value) for value in self.support_fractions)
        if not fractions or len(set(fractions)) != len(fractions):
            raise ValueError("support_fractions must be nonempty and unique.")
        if any(not (0.0 <= value <= 1.0) for value in fractions):
            raise ValueError("support_fractions must lie in [0, 1].")
        if any(
            not np.isclose(value * self.contexts, round(value * self.contexts))
            for value in fractions
        ):
            raise ValueError(
                "each support fraction must select an integer number of contexts."
            )
        if any(mode not in DEFAULT_FAILURE_MODES for mode in self.failure_modes):
            raise ValueError("failure_modes contains an unsupported mechanism.")
        if any(backend not in {"linear", "neural"} for backend in self.backends):
            raise ValueError("backends must contain only 'linear' and 'neural'.")
        valid_methods = {
            "stopped_fori_learned_gate",
            "standard_fori",
            "posthoc_winsorized",
        }
        methods = tuple(str(method) for method in self.methods)
        if not methods or len(set(methods)) != len(methods):
            raise ValueError("methods must be nonempty and unique.")
        if any(method not in valid_methods for method in methods):
            raise ValueError("methods contains an unsupported estimator.")
        if "posthoc_winsorized" in methods and "standard_fori" not in methods:
            raise ValueError("posthoc_winsorized requires standard_fori.")
        if self.irrelevant_features < 0:
            raise ValueError("irrelevant_features must be nonnegative.")
        if self.stopped_num_iterations <= 0:
            raise ValueError("stopped_num_iterations must be positive.")
        if self.stopped_gate_steps <= 0 or self.stopped_ratio_steps <= 0:
            raise ValueError("stopped inner optimizer budgets must be positive.")
        if self.standard_num_iterations <= 0 or self.standard_optimizer_steps <= 0:
            raise ValueError("standard optimizer budgets must be positive.")
        if not (0.0 <= self.validation_fraction < 1.0):
            raise ValueError("validation_fraction must lie in [0, 1).")
        if any(int(width) <= 0 for width in self.stopped_hidden_dims):
            raise ValueError("stopped_hidden_dims must contain positive widths.")
        if any(int(width) <= 0 for width in self.standard_hidden_dims):
            raise ValueError("standard_hidden_dims must contain positive widths.")


def make_structural_stopped_dataset(
    *,
    n: int,
    gamma: float,
    contexts: int,
    behavior_probability: float,
    support_fraction: float,
    failure_mode: str,
    seed: int,
    initial_rows: int | None = None,
    irrelevant_features: int = 0,
) -> StructuralStoppedDataset:
    """Sample directly from a behavior occupancy with structural support gaps."""
    if n <= 0 or contexts <= 0:
        raise ValueError("n and contexts must be positive.")
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    if not (0.0 < behavior_probability <= 1.0):
        raise ValueError("behavior_probability must lie in (0, 1].")
    if failure_mode not in DEFAULT_FAILURE_MODES:
        raise ValueError("failure_mode must be 'initial' or 'successor'.")
    selected = support_fraction * contexts
    if not (0.0 <= support_fraction <= 1.0) or not np.isclose(
        selected, round(selected)
    ):
        raise ValueError(
            "support_fraction must select an integer number of contexts."
        )
    if irrelevant_features < 0:
        raise ValueError("irrelevant_features must be nonnegative.")

    support = np.zeros(contexts, dtype=bool)
    support[_support_order(contexts)[: int(round(selected))]] = True
    safe = float(behavior_probability)
    if failure_mode == "initial":
        initial_probability = np.where(support, safe, 0.0)
        hub_probability = np.full(contexts, safe, dtype=np.float64)
    else:
        initial_probability = np.full(contexts, safe, dtype=np.float64)
        hub_probability = np.where(support, safe, 0.0)
    context_probability = np.full(contexts, 1.0 / contexts, dtype=np.float64)
    context_reward = np.linspace(0.1, 1.0, contexts, dtype=np.float64)
    truth = StructuralStoppedTruth(
        gamma=float(gamma),
        context_probability=context_probability,
        initial_behavior_probability=initial_probability.astype(np.float64),
        hub_behavior_probability=hub_probability.astype(np.float64),
        context_reward=context_reward,
        failure_mode=str(failure_mode),
        support_fraction=float(support_fraction),
    )

    rng = np.random.default_rng(seed)
    context_id = rng.choice(contexts, size=int(n), p=context_probability)
    stage = (rng.random(int(n)) < gamma).astype(np.int64)
    target_probability = np.where(
        stage == 0,
        initial_probability[context_id],
        hub_probability[context_id],
    )
    action_id = (rng.random(int(n)) >= target_probability).astype(np.int64)
    states = _state_features(context_id, stage, contexts)
    actions = _action_features(action_id)
    next_context = context_id.copy()
    next_stage = np.ones(int(n), dtype=np.int64)
    next_states = _state_features(next_context, next_stage, contexts)
    target_next_actions = _action_features(np.zeros(int(n), dtype=np.int64))

    n_initial = int(n if initial_rows is None else initial_rows)
    initial_context = rng.choice(contexts, size=n_initial, p=context_probability)
    initial_states = _state_features(
        initial_context, np.zeros(n_initial, dtype=np.int64), contexts
    )
    initial_actions = _action_features(np.zeros(n_initial, dtype=np.int64))
    if irrelevant_features:
        states = np.column_stack(
            [states, rng.normal(size=(int(n), int(irrelevant_features)))]
        )
        next_states = np.column_stack(
            [next_states, rng.normal(size=(int(n), int(irrelevant_features)))]
        )
        initial_states = np.column_stack(
            [
                initial_states,
                rng.normal(size=(n_initial, int(irrelevant_features))),
            ]
        )

    target_action = action_id == 0
    stopped_ratio = truth.ratio(context_id, stage, action_id)
    rewards = {
        "constant": np.ones(int(n), dtype=np.float64),
        "initial": ((stage == 0) & target_action).astype(np.float64),
        "hub": ((stage == 1) & target_action).astype(np.float64),
        "context": context_reward[context_id] * target_action.astype(np.float64),
    }
    return StructuralStoppedDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        context_id=context_id,
        stage=stage,
        action_id=action_id,
        rewards=rewards,
        stopped_ratio=stopped_ratio,
        truth=truth,
    )


def context_target_rows(
    truth: StructuralStoppedTruth,
    *,
    irrelevant_features: int = 0,
) -> dict[str, Array]:
    """Return deterministic context rows for support-classifier diagnostics."""
    contexts = truth.context_probability.shape[0]
    context_id = np.arange(contexts, dtype=np.int64)
    initial_states = _state_features(
        context_id, np.zeros(contexts, dtype=np.int64), contexts
    )
    successor_states = _state_features(
        context_id, np.ones(contexts, dtype=np.int64), contexts
    )
    if irrelevant_features:
        zeros = np.zeros((contexts, int(irrelevant_features)), dtype=np.float64)
        initial_states = np.column_stack([initial_states, zeros])
        successor_states = np.column_stack([successor_states, zeros])
    target_actions = _action_features(np.zeros(contexts, dtype=np.int64))
    successor_relevant = truth.initial_supported.copy()
    return {
        "initial_states": initial_states,
        "successor_states": successor_states,
        "target_actions": target_actions,
        "initial_retained": truth.initial_supported.astype(np.float64),
        "successor_retained": truth.hub_supported.astype(np.float64),
        "successor_relevant": successor_relevant.astype(np.float64),
    }


def _support_order(contexts: int) -> Array:
    """Return a fixed, spread-out nested order without using outcome data."""
    if contexts <= 0:
        raise ValueError("contexts must be positive.")
    values = np.arange(contexts, dtype=np.int64)
    bits = max(1, int(np.ceil(np.log2(contexts))))

    def reversed_bits(value: int) -> int:
        return int(f"{value:0{bits}b}"[::-1], 2)

    return np.asarray(sorted(values.tolist(), key=reversed_bits), dtype=np.int64)


def _state_features(context_id: Array, stage: Array, contexts: int) -> Array:
    context = np.asarray(context_id, dtype=np.int64).reshape(-1)
    stage_array = np.asarray(stage, dtype=np.int64).reshape(-1)
    if context.shape != stage_array.shape:
        raise ValueError("context_id and stage must have equal shapes.")
    if np.any((context < 0) | (context >= contexts)):
        raise ValueError("context_id is outside the configured context range.")
    if np.any((stage_array < 0) | (stage_array > 1)):
        raise ValueError("stage must be 0 or 1.")
    state_id = stage_array * int(contexts) + context
    return np.eye(2 * int(contexts), dtype=np.float64)[state_id]


def _action_features(action_id: Array) -> Array:
    action = np.asarray(action_id, dtype=np.int64).reshape(-1)
    if np.any((action < 0) | (action > 1)):
        raise ValueError("action_id must be 0 or 1.")
    return np.eye(2, dtype=np.float64)[action]


__all__ = [
    "DEFAULT_FAILURE_MODES",
    "DEFAULT_SUPPORT_FRACTIONS",
    "StoppedFOREExternalConfig",
    "StructuralStoppedDataset",
    "StructuralStoppedTruth",
    "context_target_rows",
    "make_structural_stopped_dataset",
]
