from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from occupancy_ratio_benchmark.data import BenchmarkDataset


Array = np.ndarray


def _logsumexp(values: Array, axis: int = 0) -> Array:
    max_val = np.max(values, axis=axis, keepdims=True)
    out = max_val + np.log(np.sum(np.exp(values - max_val), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


@dataclass(frozen=True)
class GaussianPolicy:
    gain: Array
    action_sd: float

    def mean_action(self, states: Array) -> Array:
        states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
        return states_arr @ np.asarray(self.gain, dtype=np.float64).reshape(1, 2).T

    def sample(self, states: Array, rng: np.random.Generator) -> Array:
        return self.mean_action(states) + float(self.action_sd) * rng.normal(size=(states.shape[0], 1))

    def logpdf(self, states: Array, actions: Array) -> Array:
        mean = self.mean_action(states).reshape(-1)
        actions_arr = np.asarray(actions, dtype=np.float64).reshape(-1)
        var = float(self.action_sd) ** 2
        return -0.5 * (np.log(2.0 * np.pi * var) + ((actions_arr - mean) ** 2) / var)

    def joint_moments(self, state_mean: Array, state_cov: Array) -> tuple[Array, Array]:
        mean_s = np.asarray(state_mean, dtype=np.float64).reshape(2)
        cov_s = np.asarray(state_cov, dtype=np.float64).reshape(2, 2)
        gain = np.asarray(self.gain, dtype=np.float64).reshape(1, 2)
        mean_a = (gain @ mean_s).reshape(1)
        cross = cov_s @ gain.T
        var_a = float((gain @ cov_s @ gain.T).item() + float(self.action_sd) ** 2)
        return (
            np.concatenate([mean_s, mean_a], axis=0),
            np.block([[cov_s, cross], [cross.T, np.array([[var_a]], dtype=np.float64)]]),
        )


@dataclass(frozen=True)
class GaussianMixture:
    weights: Array
    means: Array
    covariances: Array

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        weights = weights / np.sum(weights)
        means = np.asarray(self.means, dtype=np.float64)
        covariances = np.asarray(self.covariances, dtype=np.float64)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "covariances", covariances)
        cholesky = []
        log_norms = []
        dim = int(means.shape[1])
        for cov in covariances:
            chol = np.linalg.cholesky(cov + 1e-10 * np.eye(dim, dtype=np.float64))
            cholesky.append(chol)
            log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
            log_norms.append(-0.5 * (dim * np.log(2.0 * np.pi) + log_det))
        object.__setattr__(self, "_cholesky", cholesky)
        object.__setattr__(self, "_log_norms", np.asarray(log_norms, dtype=np.float64))

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])

    def logpdf(self, points: Array) -> Array:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, self.dim)
        logs = []
        for idx, weight in enumerate(self.weights):
            diff = pts - self.means[idx]
            solved = np.linalg.solve(self._cholesky[idx], diff.T)
            maha = np.sum(solved * solved, axis=0)
            logs.append(np.log(max(float(weight), 1e-300)) + self._log_norms[idx] - 0.5 * maha)
        return _logsumexp(np.vstack(logs), axis=0)


@dataclass(frozen=True)
class LinearGaussianSystem:
    b_matrix: Array
    c_matrix: Array
    process_noise_sd: float
    initial_mean: Array
    initial_cov: Array
    behavior_policy: GaussianPolicy
    target_policy: GaussianPolicy

    def step_mean(self, states: Array, actions: Array) -> Array:
        s = np.asarray(states, dtype=np.float64).reshape(-1, 2)
        a = np.asarray(actions, dtype=np.float64).reshape(-1, 1)
        return s @ self.b_matrix.T + a @ self.c_matrix.T

    def step(self, states: Array, actions: Array, rng: np.random.Generator) -> Array:
        return self.step_mean(states, actions) + float(self.process_noise_sd) * rng.normal(size=(states.shape[0], 2))

    def transition_logpdf(self, states: Array, actions: Array, next_states: Array) -> Array:
        mean = self.step_mean(states, actions)
        diff = np.asarray(next_states, dtype=np.float64).reshape(-1, 2) - mean
        var = float(self.process_noise_sd) ** 2
        return -0.5 * (2 * np.log(2.0 * np.pi * var) + np.sum(diff**2, axis=1) / var)


def make_linear_gaussian_system(policy_shift: float = 1.0) -> LinearGaussianSystem:
    b_matrix = np.array([[0.86, 0.08], [0.03, 0.78]], dtype=np.float64)
    c_matrix = np.array([[0.55], [0.25]], dtype=np.float64)
    target_gain = np.array([[-0.72, -0.20]], dtype=np.float64)
    behavior_gain = target_gain + float(policy_shift) * np.array([[0.55, -0.35]], dtype=np.float64)
    return LinearGaussianSystem(
        b_matrix=b_matrix,
        c_matrix=c_matrix,
        process_noise_sd=0.08,
        initial_mean=np.array([1.2, -0.75], dtype=np.float64),
        initial_cov=np.diag([0.55, 0.35]).astype(np.float64),
        behavior_policy=GaussianPolicy(gain=behavior_gain, action_sd=0.20),
        target_policy=GaussianPolicy(gain=target_gain, action_sd=0.16),
    )


def _stationary_cov(system: LinearGaussianSystem, policy: GaussianPolicy) -> Array:
    closed_loop = system.b_matrix + system.c_matrix @ np.asarray(policy.gain, dtype=np.float64).reshape(1, 2)
    noise = (system.process_noise_sd**2) * np.eye(2) + (policy.action_sd**2) * (system.c_matrix @ system.c_matrix.T)
    cov = np.zeros((2, 2), dtype=np.float64)
    for _ in range(20_000):
        new_cov = closed_loop @ cov @ closed_loop.T + noise
        if np.max(np.abs(new_cov - cov)) < 1e-10:
            return new_cov
        cov = new_cov
    return cov


def build_target_occupancy_mixture(system: LinearGaussianSystem, gamma: float, max_steps: int = 140) -> GaussianMixture:
    policy = system.target_policy
    closed_loop = system.b_matrix + system.c_matrix @ np.asarray(policy.gain, dtype=np.float64).reshape(1, 2)
    noise = (system.process_noise_sd**2) * np.eye(2) + (policy.action_sd**2) * (system.c_matrix @ system.c_matrix.T)
    means = []
    covs = []
    weights = []
    mean = system.initial_mean.copy()
    cov = system.initial_cov.copy()
    for step in range(max_steps):
        joint_mean, joint_cov = policy.joint_moments(mean, cov)
        weight = (1.0 - float(gamma)) * (float(gamma) ** step)
        means.append(joint_mean)
        covs.append(joint_cov)
        weights.append(weight)
        mean = closed_loop @ mean
        cov = closed_loop @ cov @ closed_loop.T + noise
        if step > 40 and float(gamma) ** (step + 1) < 1e-8:
            break
    tail_weight = float(gamma) ** len(weights)
    if tail_weight > 1e-10:
        joint_mean, joint_cov = policy.joint_moments(np.zeros(2, dtype=np.float64), _stationary_cov(system, policy))
        means.append(joint_mean)
        covs.append(joint_cov)
        weights.append(tail_weight)
    return GaussianMixture(
        weights=np.asarray(weights, dtype=np.float64),
        means=np.asarray(means, dtype=np.float64),
        covariances=np.asarray(covs, dtype=np.float64),
    )


def build_reference_joint(system: LinearGaussianSystem) -> GaussianMixture:
    mean, cov = system.behavior_policy.joint_moments(system.initial_mean, system.initial_cov)
    return GaussianMixture(
        weights=np.ones(1, dtype=np.float64),
        means=mean.reshape(1, -1),
        covariances=cov.reshape(1, 3, 3),
    )


def _state_logpdf(system: LinearGaussianSystem, states: Array) -> Array:
    state_mix = GaussianMixture(
        weights=np.ones(1, dtype=np.float64),
        means=system.initial_mean.reshape(1, 2),
        covariances=system.initial_cov.reshape(1, 2, 2),
    )
    return state_mix.logpdf(states)


def make_linear_gaussian_dataset(
    *,
    gamma: float,
    sample_size: int,
    seed: int,
    policy_shift: float = 1.0,
    sample_seed: int | None = None,
) -> BenchmarkDataset:
    system = make_linear_gaussian_system(policy_shift=float(policy_shift))
    rng = np.random.default_rng(seed if sample_seed is None else int(sample_seed))
    states = rng.multivariate_normal(system.initial_mean, system.initial_cov, size=int(sample_size))
    actions = system.behavior_policy.sample(states, rng)
    next_states = system.step(states, actions, rng)
    target_actions = system.target_policy.sample(states, rng)
    next_target_actions = system.target_policy.sample(next_states, rng)
    initial_states = rng.multivariate_normal(system.initial_mean, system.initial_cov, size=max(256, min(2_000, sample_size)))
    initial_actions = system.target_policy.sample(initial_states, rng)

    target_mix = build_target_occupancy_mixture(system, gamma)
    reference_joint = build_reference_joint(system)
    points = np.concatenate([states, actions], axis=1)
    log_ratio = np.clip(target_mix.logpdf(points) - reference_joint.logpdf(points), -60.0, 60.0)
    true_ratio = np.exp(log_ratio)
    true_action_ratio = np.exp(system.target_policy.logpdf(states, actions) - system.behavior_policy.logpdf(states, actions))
    true_transition_ratio = np.exp(system.transition_logpdf(states, actions, next_states) - _state_logpdf(system, next_states))
    rewards = -np.sum(states**2, axis=1) - 0.2 * np.sum(actions**2, axis=1)
    target_policy_value = 0.0
    for mixture_weight, mean, covariance in zip(
        target_mix.weights,
        target_mix.means,
        target_mix.covariances,
        strict=True,
    ):
        state_second_moment = float(np.sum(mean[:2] ** 2) + np.trace(covariance[:2, :2]))
        action_second_moment = float(mean[2] ** 2 + covariance[2, 2])
        target_policy_value -= float(mixture_weight) * (
            state_second_moment + 0.2 * action_second_moment
        )

    return BenchmarkDataset(
        setting="linear_gaussian",
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=target_actions,
        next_target_actions=next_target_actions,
        rewards=rewards,
        true_ratio=true_ratio,
        true_action_ratio=true_action_ratio,
        true_transition_ratio=true_transition_ratio,
        initial_states=initial_states,
        initial_actions=initial_actions,
        initial_weights=np.ones(initial_states.shape[0], dtype=np.float64),
        masks=np.ones(int(sample_size), dtype=np.float64),
        gamma=float(gamma),
        seed=int(seed),
        sample_size=int(sample_size),
        target_policy_value=float(target_policy_value),
        target_policy_value_se=0.0,
        target_policy_value_kind="analytic_normalized_discounted_step_reward",
        metadata={
            "truth_source": "analytic_linear_gaussian_mixture",
            "reference_distribution": "initial_state_gaussian",
            "policy_shift": float(policy_shift),
            "state_dim": 2,
            "action_dim": 1,
            "sample_seed": int(seed if sample_seed is None else sample_seed),
        },
    )
