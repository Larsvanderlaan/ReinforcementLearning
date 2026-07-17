"""Truth definitions and data generation for the clipped coverage benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from occupancy_ratio_benchmark._clipped_coverage_oracle import shared_hub_box_oracle


Array = np.ndarray
DEFAULT_MASS_GRID = (0.0, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98, 1.0)


@dataclass(frozen=True)
class SharedHubTruth:
    """Analytic truth for a shared-hub coverage setting."""

    gamma: float
    tau_upper: float
    q_by_context: Array
    alpha_by_context: Array
    context_probability: Array

    @property
    def retained_mass(self) -> float:
        return float(self.context_probability @ self.alpha_by_context)

    @property
    def posthoc_mass(self) -> float:
        return float(self.gamma + (1.0 - self.gamma) * self.retained_mass)

    def projected_mass(self, tau_lower: float) -> float:
        """Return the exact tabular lower-envelope projected mass."""
        oracle = shared_hub_box_oracle(
            q=self.q_by_context,
            gamma=self.gamma,
            tau_lower=tau_lower,
            tau_upper=self.tau_upper,
            context_probability=self.context_probability,
        )
        return oracle.projected_mass

    def values(self, reward: str) -> dict[str, float]:
        """Return full, recursively clipped, and post-hoc values."""
        if reward == "constant":
            return {
                "full": 1.0,
                "clipped": self.retained_mass,
                "posthoc": self.posthoc_mass,
            }
        if reward == "hub":
            return {
                "full": float(self.gamma),
                "clipped": float(self.gamma * self.retained_mass),
                "posthoc": float(self.gamma),
            }
        if reward == "gate":
            clipped = float((1.0 - self.gamma) * self.retained_mass)
            return {
                "full": float(1.0 - self.gamma),
                "clipped": clipped,
                "posthoc": clipped,
            }
        raise ValueError("reward must be 'constant', 'hub', or 'gate'.")


@dataclass(frozen=True)
class SharedHubDataset:
    """Sampled reference rows and analytic truth for the coverage benchmark."""

    states: Array
    actions: Array
    next_states: Array
    target_next_actions: Array
    initial_states: Array
    initial_actions: Array
    context_id: Array
    category: Array
    rewards: dict[str, Array]
    recursive_ratio: Array
    posthoc_ratio: Array
    full_ratio: Array
    gate_indicator: Array
    terminals: Array | None
    timeouts: Array | None
    handle_timeouts: str
    absorbing_state: bool
    truth: SharedHubTruth

    @property
    def n(self) -> int:
        return int(self.states.shape[0])


@dataclass(frozen=True)
class CoverageRunConfig:
    """Configuration for one reproducible coverage-benchmark screen."""

    n: int = 2_000
    repetitions: int = 1
    gamma: float = 0.95
    tau_lower: float = 1e-4
    tau_upper: float = 10.0
    mass_grid: Sequence[float] = DEFAULT_MASS_GRID
    contexts: int = 1
    backends: Sequence[str] = ("linear",)
    methods: Sequence[str] = (
        "clipped_fori",
        "standard_fori",
        "posthoc_winsorized",
    )
    seed: int = 12_000
    crossfit_folds: int = 2
    clipped_num_iterations: int = 300
    clipped_gate_steps: int = 200
    clipped_ratio_steps: int = 300
    clipped_gate_learning_rate: float | None = None
    clipped_ratio_learning_rate: float | None = None
    clipped_inner_relative_tolerance: float = 1e-8
    clipped_inner_gradient_tolerance: float = 1e-6
    clipped_inner_patience: int = 5
    standard_num_iterations: int = 30
    standard_optimizer_steps: int = 300
    standard_outer_tolerance: float = 1e-4
    standard_objective_tolerance: float = 1e-5
    standard_mass_tolerance: float = 1e-10
    standard_require_convergence: bool = False
    validation_fraction: float = 0.0
    optimizer_stability_restarts: int = 1
    optimizer_perturbation_scale: float = 1e-3
    sampling_mode: str = "occupancy"
    irrelevant_features: int = 0
    termination_encoding: str = "none"

    def __post_init__(self) -> None:
        if self.n <= 0 or self.repetitions <= 0:
            raise ValueError("n and repetitions must be positive.")
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError("gamma must be in [0, 1).")
        if not (0.0 < self.tau_lower <= 1.0 <= self.tau_upper):
            raise ValueError(
                "clipping levels must satisfy 0 < tau_lower <= 1 <= tau_upper."
            )
        if self.contexts <= 0:
            raise ValueError("contexts must be positive.")
        if self.crossfit_folds != 2:
            raise ValueError(
                "this benchmark currently implements exactly two cross-fitting folds."
            )
        if any(not (0.0 <= float(mass) <= 1.0) for mass in self.mass_grid):
            raise ValueError("mass_grid values must lie in [0, 1].")
        if any(str(backend) not in {"linear", "neural"} for backend in self.backends):
            raise ValueError("backends must contain only 'linear' and 'neural'.")
        valid_methods = {
            "clipped_fori",
            "standard_fori",
            "posthoc_winsorized",
        }
        methods = tuple(str(method) for method in self.methods)
        if not methods or len(set(methods)) != len(methods):
            raise ValueError("methods must be a nonempty sequence without duplicates.")
        if any(method not in valid_methods for method in methods):
            raise ValueError("methods contains an unsupported estimator.")
        if "posthoc_winsorized" in methods and "standard_fori" not in methods:
            raise ValueError("posthoc_winsorized requires standard_fori.")
        if self.optimizer_stability_restarts not in {1, 3}:
            raise ValueError("optimizer_stability_restarts must be 1 or 3.")
        if self.optimizer_perturbation_scale <= 0.0:
            raise ValueError("optimizer_perturbation_scale must be positive.")
        if self.clipped_inner_relative_tolerance <= 0.0:
            raise ValueError("clipped_inner_relative_tolerance must be positive.")
        if self.clipped_inner_gradient_tolerance <= 0.0:
            raise ValueError("clipped_inner_gradient_tolerance must be positive.")
        if self.clipped_inner_patience <= 0:
            raise ValueError("clipped_inner_patience must be positive.")
        if self.standard_num_iterations < 0 or self.standard_optimizer_steps <= 0:
            raise ValueError("standard optimizer budgets are invalid.")
        if (
            self.standard_outer_tolerance <= 0.0
            or self.standard_objective_tolerance <= 0.0
            or self.standard_mass_tolerance <= 0.0
        ):
            raise ValueError("standard convergence tolerances must be positive.")
        if self.sampling_mode not in {"occupancy", "trajectory"}:
            raise ValueError("sampling_mode must be 'occupancy' or 'trajectory'.")
        if self.irrelevant_features < 0:
            raise ValueError("irrelevant_features must be nonnegative.")
        if self.termination_encoding not in {
            "none",
            "terminal_absorbing",
            "timeout_absorbing",
        }:
            raise ValueError("termination_encoding is invalid.")


def make_shared_hub_dataset(
    *,
    n: int,
    gamma: float,
    tau_upper: float,
    oracle_mass: float,
    contexts: int = 1,
    seed: int = 0,
    initial_rows: int | None = None,
    sampling_mode: str = "occupancy",
    irrelevant_features: int = 0,
    termination_encoding: str = "none",
) -> SharedHubDataset:
    """Sample the branch-merge benchmark directly from behavior occupancy."""
    if n <= 0 or contexts <= 0:
        raise ValueError("n and contexts must be positive.")
    if sampling_mode not in {"occupancy", "trajectory"}:
        raise ValueError("sampling_mode must be 'occupancy' or 'trajectory'.")
    if irrelevant_features < 0:
        raise ValueError("irrelevant_features must be nonnegative.")
    if termination_encoding not in {
        "none",
        "terminal_absorbing",
        "timeout_absorbing",
    }:
        raise ValueError("termination_encoding is invalid.")
    if not (0.0 <= oracle_mass <= 1.0):
        raise ValueError("oracle_mass must be in [0, 1].")
    if not (0.0 <= gamma < 1.0) or tau_upper < 1.0:
        raise ValueError("gamma or tau_upper is invalid.")

    theta = (
        2.0 * np.pi * (np.arange(contexts, dtype=np.float64) + 0.5) / float(contexts)
    )
    context_probability = np.full(contexts, 1.0 / float(contexts), dtype=np.float64)
    if contexts == 1:
        q = np.array(
            [oracle_mass / tau_upper if oracle_mass < 1.0 else min(0.2, 1.0)],
            dtype=np.float64,
        )
    else:
        base = 0.1 + 0.4 * _sigmoid(3.0 * np.sin(3.0 * theta))
        rho = solve_context_coverage_scale(
            oracle_mass=oracle_mass,
            base_overlap=base,
            tau_upper=tau_upper,
            context_probability=context_probability,
        )
        q = rho * base
    alpha = np.minimum(1.0, tau_upper * q)
    truth = SharedHubTruth(
        gamma=float(gamma),
        tau_upper=float(tau_upper),
        q_by_context=q,
        alpha_by_context=alpha,
        context_probability=context_probability,
    )

    probabilities = np.column_stack(
        [
            (1.0 - gamma) * context_probability * q,
            (1.0 - gamma) * context_probability * (1.0 - q),
            gamma * context_probability,
        ]
    ).reshape(-1)
    probabilities /= np.sum(probabilities)
    rng = np.random.default_rng(seed)
    if sampling_mode == "occupancy":
        flat_category = rng.choice(probabilities.shape[0], size=int(n), p=probabilities)
        context_id = flat_category // 3
        category = flat_category % 3
    else:
        context_id = rng.choice(contexts, size=int(n), p=context_probability)
        at_initial_time = rng.random(int(n)) < (1.0 - gamma)
        takes_target = rng.random(int(n)) < q[context_id]
        category = np.where(at_initial_time, np.where(takes_target, 0, 1), 2)
    states = _state_features(theta[context_id], category == 2)
    actions = _action_features(category)
    next_states = _state_features(theta[context_id], np.ones(n, dtype=bool))
    target_next_actions = _action_features(np.full(n, 2, dtype=np.int64))

    n_initial = int(n if initial_rows is None else initial_rows)
    initial_context = rng.choice(contexts, size=n_initial, p=context_probability)
    initial_states = _state_features(
        theta[initial_context], np.zeros(n_initial, dtype=bool)
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
            [initial_states, rng.normal(size=(n_initial, int(irrelevant_features)))]
        )

    q_row = q[context_id]
    alpha_row = alpha[context_id]
    recursive_ratio = np.where(
        category == 0,
        np.where(
            q_row > 0.0,
            np.minimum(1.0 / np.maximum(q_row, 1e-300), tau_upper),
            tau_upper,
        ),
        np.where(category == 1, 0.0, alpha_row),
    )
    posthoc_ratio = np.where(
        category == 0,
        np.where(
            q_row > 0.0,
            np.minimum(1.0 / np.maximum(q_row, 1e-300), tau_upper),
            tau_upper,
        ),
        np.where(category == 1, 0.0, 1.0),
    )
    full_ratio = np.where(
        category == 0,
        np.where(q_row > 0.0, 1.0 / np.maximum(q_row, 1e-300), np.nan),
        np.where(category == 1, 0.0, 1.0),
    )
    gate_indicator = np.where(
        category == 0, (q_row * tau_upper >= 1.0).astype(float), 1.0
    )
    rewards = {
        "constant": np.ones(n, dtype=np.float64),
        "hub": (category == 2).astype(np.float64),
        "gate": (category == 0).astype(np.float64),
    }
    terminals = (
        (category == 2) if termination_encoding == "terminal_absorbing" else None
    )
    timeouts = (category == 2) if termination_encoding == "timeout_absorbing" else None
    return SharedHubDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        target_next_actions=target_next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        context_id=context_id,
        category=category,
        rewards=rewards,
        recursive_ratio=recursive_ratio.astype(np.float64),
        posthoc_ratio=posthoc_ratio.astype(np.float64),
        full_ratio=full_ratio.astype(np.float64),
        gate_indicator=gate_indicator.astype(np.float64),
        terminals=terminals,
        timeouts=timeouts,
        handle_timeouts=(
            "terminal" if termination_encoding == "timeout_absorbing" else "nonterminal"
        ),
        absorbing_state=termination_encoding != "none",
        truth=truth,
    )


def solve_context_coverage_scale(
    *,
    oracle_mass: float,
    base_overlap: Array,
    tau_upper: float,
    context_probability: Array,
) -> float:
    """Invert the monotone contextual retained-mass curve by bisection."""
    target = float(oracle_mass)
    if target <= 0.0:
        return 0.0
    base = np.asarray(base_overlap, dtype=np.float64)
    probability = np.asarray(context_probability, dtype=np.float64)
    at_one = float(probability @ np.minimum(1.0, tau_upper * base))
    if target > at_one + 1e-12:
        raise ValueError("requested oracle_mass is not attainable on rho in [0, 1].")
    if target >= at_one - 1e-12:
        return 1.0
    lower, upper = 0.0, 1.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        mass = float(probability @ np.minimum(1.0, tau_upper * midpoint * base))
        if mass < target:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def analytic_shared_hub_rows(
    *,
    gamma: float = 0.95,
    tau_upper: float = 10.0,
    mass_grid: Sequence[float] = DEFAULT_MASS_GRID,
) -> list[dict[str, float | str]]:
    """Return exact mechanism rows without fitting an estimator."""
    rows: list[dict[str, float | str]] = []
    for requested_mass in mass_grid:
        dataset = make_shared_hub_dataset(
            n=1,
            gamma=gamma,
            tau_upper=tau_upper,
            oracle_mass=float(requested_mass),
            contexts=1,
            seed=0,
        )
        truth = dataset.truth
        q = float(truth.q_by_context[0])
        for reward in ("constant", "hub", "gate"):
            values = truth.values(reward)
            rows.append(
                {
                    "requested_mass": float(requested_mass),
                    "oracle_mass": truth.retained_mass,
                    "q": q,
                    "reward": reward,
                    "full_value": values["full"],
                    "clipped_value": values["clipped"],
                    "posthoc_value": values["posthoc"],
                    "posthoc_mass": truth.posthoc_mass,
                }
            )
    return rows


def _state_features(theta: Array, hub: Array) -> Array:
    angle = np.asarray(theta, dtype=np.float64)
    return np.column_stack(
        [np.asarray(hub, dtype=np.float64), np.cos(angle), np.sin(angle)]
    )


def _action_features(category: Array) -> Array:
    value = np.asarray(category, dtype=np.int64).reshape(-1)
    if np.any((value < 0) | (value > 2)):
        raise ValueError("action category must be 0, 1, or 2.")
    return np.eye(3, dtype=np.float64)[value]


def _sigmoid(value: Array) -> Array:
    value = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-value))


__all__ = [
    "CoverageRunConfig",
    "DEFAULT_MASS_GRID",
    "SharedHubDataset",
    "SharedHubTruth",
    "analytic_shared_hub_rows",
    "make_shared_hub_dataset",
    "solve_context_coverage_scale",
]
