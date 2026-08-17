"""Complementary-source data construction for the GenPQR experiment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

try:
    from .data_generation import (
        ExactSoftQPlanner,
        LinearGaussianDynamics,
        SimulationConfig,
        exact_anchor_pqr_reward_matrix,
        generate_deeppqr_style_data,
        make_anchor_mu,
        make_zero_g,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from data_generation import (
        ExactSoftQPlanner,
        LinearGaussianDynamics,
        SimulationConfig,
        exact_anchor_pqr_reward_matrix,
        generate_deeppqr_style_data,
        make_anchor_mu,
        make_zero_g,
    )


@dataclass(frozen=True)
class OutcomeSource:
    """State-outcome observations with latent actions."""

    states: np.ndarray
    outcomes: np.ndarray
    trajectory_ids: np.ndarray
    n_trajectories: int

    def subset_trajectories(self, n_trajectories: int) -> "OutcomeSource":
        """Return a nested prefix containing complete trajectories."""
        if not 1 <= n_trajectories <= self.n_trajectories:
            raise ValueError(
                f"n_trajectories must be in [1, {self.n_trajectories}], got {n_trajectories}."
            )
        keep = self.trajectory_ids < n_trajectories
        return OutcomeSource(
            states=self.states[keep],
            outcomes=self.outcomes[keep],
            trajectory_ids=self.trajectory_ids[keep],
            n_trajectories=n_trajectories,
        )


@dataclass(frozen=True)
class BehaviorSource:
    """State-action transition observations with rewards withheld."""

    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    trajectory_ids: np.ndarray
    n_trajectories: int


@dataclass(frozen=True)
class OracleTestSource:
    """Independent test transitions with truth exposed only for reporting."""

    states: np.ndarray
    actions: np.ndarray
    true_reward_matrix: np.ndarray
    true_policy_probs: np.ndarray
    true_g: np.ndarray
    trajectory_ids: np.ndarray
    n_trajectories: int


@dataclass(frozen=True)
class OracleWorld:
    """Simulator truth used by data generation and labeled oracle diagnostics."""

    config: SimulationConfig
    parameters: LinearGaussianDynamics
    planner: ExactSoftQPlanner

    def reward_matrix(self, states: np.ndarray) -> np.ndarray:
        """Return the noiseless zero-anchor reward matrix."""
        return exact_anchor_pqr_reward_matrix(
            states=np.asarray(states, dtype=float),
            planner=self.planner,
            params=self.parameters,
            config=self.config,
        )

    def policy_probs(self, states: np.ndarray) -> np.ndarray:
        """Return the true reference/behavior policy."""
        return self.planner.predict_proba(np.asarray(states, dtype=float))

    def g(self, states: np.ndarray) -> np.ndarray:
        """Return ``E[r(S,A) | S]`` for ``A`` drawn from the true policy."""
        reward_matrix = self.reward_matrix(states)
        return np.sum(self.policy_probs(states) * reward_matrix, axis=1)


@dataclass(frozen=True)
class DataFusionWorld:
    """The two observed sources and an independent oracle test source."""

    outcome: OutcomeSource
    behavior: BehaviorSource
    test: OracleTestSource
    oracle: OracleWorld


@dataclass(frozen=True)
class StateRewardRegressor:
    """Prediction interface for the auxiliary ``g(s)`` regression."""

    model: object

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Predict the conditional mean reward."""
        states = np.asarray(states, dtype=float)
        if hasattr(self.model, "booster_"):
            prediction = np.asarray(self.model.booster_.predict(states), dtype=float)
        else:
            prediction = np.asarray(self.model.predict(states), dtype=float)
        return prediction.reshape(-1)


def _validate_generated_world(world: DataFusionWorld, anchor_tolerance: float = 1e-8) -> None:
    arrays = [
        world.outcome.states,
        world.outcome.outcomes,
        world.behavior.states,
        world.behavior.actions,
        world.behavior.next_states,
        world.test.states,
        world.test.true_reward_matrix,
        world.test.true_policy_probs,
        world.test.true_g,
    ]
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Generated data contain nonfinite values.")
    if world.behavior.states.shape != world.behavior.next_states.shape:
        raise ValueError("Behavior states and next_states must have identical shapes.")
    if world.behavior.states.shape[0] != world.behavior.actions.shape[0]:
        raise ValueError("Behavior states and actions have inconsistent lengths.")
    if world.test.true_reward_matrix.shape != world.test.true_policy_probs.shape:
        raise ValueError("Test reward and policy matrices must have identical shapes.")
    anchor_errors = [
        np.max(np.abs(world.oracle.reward_matrix(world.outcome.states)[:, 0])),
        np.max(np.abs(world.oracle.reward_matrix(world.behavior.states)[:, 0])),
        np.max(np.abs(world.test.true_reward_matrix[:, 0])),
    ]
    max_anchor_error = float(np.max(anchor_errors))
    if max_anchor_error > anchor_tolerance:
        raise ValueError(
            f"The unchanged DGP violated r(s, 0)=0: max absolute error {max_anchor_error:.3e}."
        )
    policy_row_error = float(np.max(np.abs(np.sum(world.test.true_policy_probs, axis=1) - 1.0)))
    if policy_row_error > 1e-10:
        raise ValueError(f"True policy rows do not sum to one: max error {policy_row_error:.3e}.")


def generate_data_fusion_world(
    *,
    seed: int,
    config: SimulationConfig,
    behavior_trajectories: int,
    outcome_trajectories: int,
    test_trajectories: int,
) -> DataFusionWorld:
    """Generate independent complementary sources from one shared MDP.

    The unchanged simulator constructs a reward satisfying ``r(s, 0)=0``.
    The outcome source retains only ``(S, Y)``; the behavior source retains
    only ``(S, A, S')``. Both sources use the same soft behavior/reference
    policy but independent trajectories and random-number streams.
    """
    for name, value in [
        ("behavior_trajectories", behavior_trajectories),
        ("outcome_trajectories", outcome_trajectories),
        ("test_trajectories", test_trajectories),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if config.n_actions < 2:
        raise ValueError("The data-fusion experiment requires at least two actions.")

    anchor_mu = make_anchor_mu(0, config.n_actions)
    zero_g = make_zero_g()
    behavior_config = replace(config, seed=seed)
    behavior_raw = generate_deeppqr_style_data(
        n_trajectories=behavior_trajectories,
        config=behavior_config,
        mu=anchor_mu,
        g=zero_g,
    )
    parameters = behavior_raw["simulation_parameters"]

    outcome_config = replace(config, seed=seed + 1_000_003)
    outcome_raw = generate_deeppqr_style_data(
        n_trajectories=outcome_trajectories,
        config=outcome_config,
        mu=anchor_mu,
        g=zero_g,
        simulation_parameters=parameters,
    )
    test_config = replace(config, seed=seed + 2_000_003)
    test_raw = generate_deeppqr_style_data(
        n_trajectories=test_trajectories,
        config=test_config,
        mu=anchor_mu,
        g=zero_g,
        simulation_parameters=parameters,
    )

    oracle = OracleWorld(
        config=behavior_config,
        parameters=parameters,
        planner=behavior_raw["planner"],
    )
    test_states = np.asarray(test_raw["states"], dtype=float)
    test_reward_matrix = oracle.reward_matrix(test_states)
    test_policy_probs = oracle.policy_probs(test_states)
    world = DataFusionWorld(
        outcome=OutcomeSource(
            states=np.asarray(outcome_raw["states"], dtype=float),
            outcomes=np.asarray(outcome_raw["rewards"], dtype=float),
            trajectory_ids=np.asarray(outcome_raw["trajectory_id"], dtype=int),
            n_trajectories=outcome_trajectories,
        ),
        behavior=BehaviorSource(
            states=np.asarray(behavior_raw["states"], dtype=float),
            actions=np.asarray(behavior_raw["actions"], dtype=int),
            next_states=np.asarray(behavior_raw["next_states"], dtype=float),
            dones=np.asarray(behavior_raw["dones"], dtype=float),
            trajectory_ids=np.asarray(behavior_raw["trajectory_id"], dtype=int),
            n_trajectories=behavior_trajectories,
        ),
        test=OracleTestSource(
            states=test_states,
            actions=np.asarray(test_raw["actions"], dtype=int),
            true_reward_matrix=test_reward_matrix,
            true_policy_probs=test_policy_probs,
            true_g=np.sum(test_policy_probs * test_reward_matrix, axis=1),
            trajectory_ids=np.asarray(test_raw["trajectory_id"], dtype=int),
            n_trajectories=test_trajectories,
        ),
        oracle=oracle,
    )
    _validate_generated_world(world)
    return world


def fit_state_reward_regressor(
    source: OutcomeSource,
    *,
    seed: int,
    n_estimators: int = 150,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 20,
) -> StateRewardRegressor:
    """Fit ``g(s)=E[Y|S=s]`` using only the state-outcome source.

    The hyperparameters are fixed before the confirmatory run; no simulator
    truth is used for fitting or model selection.
    """
    states = np.asarray(source.states, dtype=float)
    outcomes = np.asarray(source.outcomes, dtype=float).reshape(-1)
    if states.ndim != 2:
        raise ValueError(f"states must be two-dimensional, got shape {states.shape}.")
    if states.shape[0] != outcomes.shape[0]:
        raise ValueError("states and outcomes have inconsistent lengths.")
    if states.shape[0] < 2:
        raise ValueError("At least two state-outcome observations are required.")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(outcomes)):
        raise ValueError("State-outcome data must be finite.")

    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - dependency is in the repro requirements
        raise RuntimeError(
            "The data-fusion experiment requires lightgbm for the g(s) regression."
        ) from exc

    model = LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1e-3,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(states, outcomes)
    return StateRewardRegressor(model=model)


def recover_reward_matrix(
    *,
    q_estimate,
    normalization_policy: Callable[[np.ndarray], np.ndarray],
    normalization_function: Callable[[np.ndarray], np.ndarray],
    states: np.ndarray,
) -> np.ndarray:
    """Apply the GenPQR plug-in reward formula for every action."""
    states = np.asarray(states, dtype=float)
    q_matrix = np.asarray(q_estimate.predict_all_actions(states), dtype=float)
    policy_probs = np.asarray(normalization_policy(states), dtype=float)
    g_values = np.asarray(normalization_function(states), dtype=float).reshape(-1)
    if q_matrix.shape != policy_probs.shape:
        raise ValueError(
            f"Q and normalization-policy shapes differ: {q_matrix.shape} versus {policy_probs.shape}."
        )
    if q_matrix.shape[0] != g_values.shape[0]:
        raise ValueError("g(s) predictions have the wrong number of rows.")
    if np.any(policy_probs < 0.0):
        raise ValueError("Normalization-policy probabilities must be nonnegative.")
    row_sums = np.sum(policy_probs, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Normalization-policy rows must sum to one.")
    mu_q = np.sum(policy_probs * q_matrix, axis=1)
    reward_matrix = q_matrix - mu_q[:, None] + g_values[:, None]
    if not np.all(np.isfinite(reward_matrix)):
        raise ValueError("Recovered reward matrix contains nonfinite values.")
    return reward_matrix
