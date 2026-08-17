"""Policy-estimation routines for the IRL experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import numpy as np

try:
    from .utils import MLP, EPS, set_random_seed, softmax, standardize_fit, state_action_features
except ImportError:  # pragma: no cover - supports direct script execution
    from utils import MLP, EPS, set_random_seed, softmax, standardize_fit, state_action_features

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


@dataclass
class EstimatedPolicy:
    """Unified interface for policy estimators."""

    n_actions: int
    kind: str
    parameters: Dict[str, object]

    def predict_logits(self, states: np.ndarray) -> np.ndarray:
        """Return policy logits for each action."""
        if self.kind == "linear-softmax":
            features = self.parameters["feature_map"](states, self.n_actions)
            return features @ self.parameters["theta"]
        if self.kind == "mlp":
            return self.parameters["network"].predict(states)
        if self.kind == "sklearn":
            x = self.parameters["transform"](states)
            probs = np.clip(self.parameters["model"].predict_proba(x), EPS, 1.0)
            return np.log(probs)
        if self.kind == "torch":
            model = self.parameters["model"]
            model.eval()
            x = self.parameters["transform"](states)
            with torch.no_grad():
                logits = model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
            return logits
        raise ValueError(f"Unsupported policy kind: {self.kind}")

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        """Return action probabilities."""
        logits = self.predict_logits(states)
        temperature = float(self.parameters.get("temperature", 1.0))
        logits = logits / max(temperature, EPS)
        if logits.ndim == 1:
            logits = logits[:, None]
        probs = softmax(logits, axis=1)
        clip_min = self.parameters.get("prob_clip_min")
        clip_max = self.parameters.get("prob_clip_max")
        if clip_min is not None or clip_max is not None:
            lo = EPS if clip_min is None else float(clip_min)
            hi = 1.0 if clip_max is None else float(clip_max)
            probs = np.clip(probs, lo, hi)
            probs = probs / np.clip(np.sum(probs, axis=1, keepdims=True), EPS, None)
        return probs

    def sample_actions(self, states: np.ndarray, seed: int | None = None) -> np.ndarray:
        """Sample actions from the estimated policy."""
        rng = set_random_seed(0 if seed is None else seed)
        probs = self.predict_proba(states)
        return np.array([rng.choice(self.n_actions, p=row) for row in probs], dtype=int)

    def predict_q(self, states: np.ndarray) -> np.ndarray:
        """Return the internal Q/logit surrogate when available."""
        if self.kind == "linear-softmax":
            return self.predict_logits(states)
        if self.kind == "torch":
            model = self.parameters["model"]
            transform = self.parameters["transform"]
            model.eval()
            x = transform(states)
            with torch.no_grad():
                return model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
        if self.kind == "mlp":
            return self.parameters["network"].predict(states)
        raise ValueError("This policy object does not expose a Q surrogate.")

    def predict_state_reward(self, states: np.ndarray) -> np.ndarray:
        """Return an AIRL-style state-only reward estimate when available."""
        if "airl_reward_model" not in self.parameters:
            raise ValueError("This policy object does not expose an AIRL state reward.")
        model = self.parameters["airl_reward_model"]
        transform = self.parameters["transform"]
        x = transform(states)
        if self.kind == "torch":
            model.eval()
            with torch.no_grad():
                return model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy().reshape(-1)
        raise ValueError("AIRL state reward is only exposed for torch policies.")


@dataclass
class LinearRewardFunction:
    """Linear reward parameterization for SPL-GD/DDC."""

    n_actions: int
    theta: np.ndarray

    def features(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=int).reshape(-1)
        base = default_state_feature_map(states, self.n_actions)
        out = np.zeros((states.shape[0], base.shape[1] * self.n_actions), dtype=float)
        d = base.shape[1]
        for a in range(self.n_actions):
            mask = actions == a
            if np.any(mask):
                out[mask, a * d : (a + 1) * d] = base[mask]
        return out

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self.features(states, actions) @ self.theta


def default_state_feature_map(states: np.ndarray, n_actions: int) -> np.ndarray:
    """Feature map for linear MaxEnt IRL."""
    states = np.asarray(states, dtype=float)
    intercept = np.ones((states.shape[0], 1), dtype=float)
    quadratic = states**2
    shared = np.concatenate([intercept, states, quadratic], axis=1)
    return shared


def _expand_action_features(states: np.ndarray, n_actions: int) -> np.ndarray:
    """Build action-specific feature tensors of shape (n, k, a)."""
    shared = default_state_feature_map(states, n_actions)
    n, d = shared.shape
    tensor = np.zeros((n, n_actions, d * n_actions), dtype=float)
    for action in range(n_actions):
        tensor[:, action, action * d : (action + 1) * d] = shared
    return tensor


def _torch_batch_indices(n_samples: int, batch_size: int):
    """Yield shuffled minibatch indices without DataLoader overhead."""
    if torch is None:
        raise RuntimeError("Torch batching requested without torch installed.")
    order = torch.randperm(n_samples)
    for start in range(0, n_samples, batch_size):
        yield order[start : start + batch_size]


def fit_maxent_irl_policy(
    states: np.ndarray,
    actions: np.ndarray,
    n_actions: int,
    hidden_sizes: Sequence[int] = (128, 128),
    learning_rate: float = 5e-3,
    n_iters: int = 200,
    l2: float = 1e-4,
    temperature: float = 1.0,
    prob_clip_min: float = 0.01,
    prob_clip_max: float = 0.99,
    seed: int = 0,
    verbose: bool = False,
) -> EstimatedPolicy:
    """Fit a neural maximum-entropy IRL policy/Q surrogate.

    Following the DeepPQR comparison description, this estimator learns a
    neural state-action score interpreted as the MaxEnt IRL Q-function
    surrogate, then induces the policy by softmax over that Q estimate.
    """
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    if torch is None:
        rng = set_random_seed(seed)
        feature_tensor = _expand_action_features(states, n_actions)
        theta = rng.normal(scale=0.1, size=feature_tensor.shape[-1])
        for iteration in range(n_iters):
            logits = feature_tensor @ theta
            probs = softmax(logits / max(temperature, EPS), axis=1)
            empirical = feature_tensor[np.arange(states.shape[0]), actions]
            expected = np.sum(feature_tensor * probs[:, :, None], axis=1)
            grad = np.mean(empirical - expected, axis=0) - l2 * theta
            theta += learning_rate * grad
        class LinearWrapper:
            def __call__(self, x: np.ndarray, action_count: int) -> np.ndarray:
                return _expand_action_features(np.asarray(x, dtype=float), action_count)
        return EstimatedPolicy(
            n_actions=n_actions,
            kind="linear-softmax",
            parameters={
                "theta": theta / max(temperature, EPS),
                "feature_map": LinearWrapper(),
                "prob_clip_min": prob_clip_min,
                "prob_clip_max": prob_clip_max,
            },
        )

    mean, std = standardize_fit(states)

    def transform(x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - mean) / std

    class QNet(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden: Sequence[int]) -> None:
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                prev = h
            layers.append(nn.Linear(prev, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    x = torch.as_tensor(transform(states), dtype=torch.float32)
    y = torch.as_tensor(actions, dtype=torch.long)
    batch_size = min(256, states.shape[0])
    model = QNet(states.shape[1], n_actions, hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)
    for iteration in range(n_iters):
        model.train()
        for batch_idx in _torch_batch_indices(states.shape[0], batch_size):
            xb = x[batch_idx]
            yb = y[batch_idx]
            q_logits = model(xb) / max(temperature, EPS)
            log_probs = torch.log_softmax(q_logits, dim=1)
            loss = torch.nn.functional.nll_loss(log_probs, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if verbose and (iteration + 1) % 50 == 0:
            with torch.no_grad():
                nll = torch.nn.functional.nll_loss(torch.log_softmax(model(x) / max(temperature, EPS), dim=1), y).item()
            print(f"[MaxEntIRL-NN] iter={iteration + 1} nll={nll:.6f}")
    return EstimatedPolicy(
        n_actions=n_actions,
        kind="torch",
        parameters={
            "model": model,
            "transform": transform,
            "temperature": temperature,
            "surrogate": "maxent_q",
            "prob_clip_min": prob_clip_min,
            "prob_clip_max": prob_clip_max,
        },
    )


def fit_behavior_cloning_policy(
    states: np.ndarray,
    actions: np.ndarray,
    n_actions: int,
    hidden_sizes: Sequence[int] = (64, 64),
    learning_rate: float = 5e-3,
    n_epochs: int = 400,
    prob_clip_min: float = 0.01,
    prob_clip_max: float = 0.99,
    seed: int = 0,
    verbose: bool = False,
) -> EstimatedPolicy:
    """Fit a multiclass behavior-cloning policy using a small MLP."""
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    if torch is not None:
        torch.manual_seed(seed)
        mean, std = standardize_fit(states)

        def transform(x: np.ndarray) -> np.ndarray:
            return (np.asarray(x, dtype=float) - mean) / std

        class PolicyNet(nn.Module):
            def __init__(self, input_dim: int, output_dim: int, hidden: Sequence[int]) -> None:
                super().__init__()
                layers = []
                prev = input_dim
                for h in hidden:
                    layers.append(nn.Linear(prev, h))
                    layers.append(nn.ReLU())
                    prev = h
                layers.append(nn.Linear(prev, output_dim))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        x = torch.as_tensor(transform(states), dtype=torch.float32)
        y = torch.as_tensor(actions, dtype=torch.long)
        batch_size = min(256, states.shape[0])
        model = PolicyNet(states.shape[1], n_actions, hidden_sizes)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for _ in range(n_epochs):
            model.train()
            for batch_idx in _torch_batch_indices(states.shape[0], batch_size):
                xb = x[batch_idx]
                yb = y[batch_idx]
                logits = model(xb)
                loss = torch.nn.functional.cross_entropy(logits, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        return EstimatedPolicy(
            n_actions=n_actions,
            kind="torch",
            parameters={
                "model": model,
                "transform": transform,
                "prob_clip_min": prob_clip_min,
                "prob_clip_max": prob_clip_max,
            },
        )
    mean, std = standardize_fit(states)
    network = MLP.initialize(
        input_dim=states.shape[1],
        output_dim=n_actions,
        hidden_sizes=hidden_sizes,
        rng=set_random_seed(seed),
        task="multiclass",
        x_mean=mean,
        x_std=std,
    )
    network.fit_multiclass(
        x=states,
        y=actions,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        verbose=verbose,
        rng=set_random_seed(seed + 1),
    )
    return EstimatedPolicy(
        n_actions=n_actions,
        kind="mlp",
        parameters={"network": network, "prob_clip_min": prob_clip_min, "prob_clip_max": prob_clip_max},
    )


def fit_airl_policy(
    states: np.ndarray,
    actions: np.ndarray,
    n_actions: int,
    next_states: np.ndarray | None = None,
    dones: np.ndarray | None = None,
    gamma: float = 0.95,
    transition_model: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    hidden_sizes: Sequence[int] = (64, 64),
    learning_rate: float = 1e-3,
    n_iters: int = 200,
    pretrain_epochs: int = 60,
    bc_weight: float = 0.5,
    entropy_weight: float = 1e-2,
    prob_clip_min: float = 0.01,
    prob_clip_max: float = 0.99,
    seed: int = 0,
    verbose: bool = False,
) -> EstimatedPolicy:
    """Fit a simplified AIRL-style adversarial policy estimator.

    When `torch` is available, this uses a standard AIRL-style decomposition:
    a state-only reward network `g(s)` and a potential/value network `h(s)`,
    with discriminator logit

        f(s, a, s') - log pi(a | s)
        where f(s, a, s') = g(s) + gamma * h(s') - h(s).

    This aligns with the common action-independent, deterministic-transition
    AIRL setup described in the DeepPQR comparison section. A deterministic
    `transition_model` can be supplied so sampled policy actions induce their
    own next states during adversarial training.
    """
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    if torch is not None:
        next_states = states if next_states is None else np.asarray(next_states, dtype=float)
        dones = np.zeros(states.shape[0], dtype=float) if dones is None else np.asarray(dones, dtype=float).reshape(-1)
        mean, std = standardize_fit(states)

        def transform(x: np.ndarray) -> np.ndarray:
            return (np.asarray(x, dtype=float) - mean) / std

        class PolicyNet(nn.Module):
            def __init__(self, input_dim: int, output_dim: int, hidden: Sequence[int]) -> None:
                super().__init__()
                layers = []
                prev = input_dim
                for h in hidden:
                    layers.append(nn.Linear(prev, h))
                    layers.append(nn.ReLU())
                    prev = h
                layers.append(nn.Linear(prev, output_dim))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        class RewardNet(nn.Module):
            def __init__(self, input_dim: int, hidden: Sequence[int]) -> None:
                super().__init__()
                layers = []
                prev = input_dim
                for h in hidden:
                    layers.append(nn.Linear(prev, h))
                    layers.append(nn.ReLU())
                    prev = h
                layers.append(nn.Linear(prev, 1))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x).squeeze(-1)

        rng = set_random_seed(seed)
        x_states = transform(states)
        x_next_states = transform(next_states)
        x_tensor = torch.as_tensor(x_states, dtype=torch.float32)
        x_next_tensor = torch.as_tensor(x_next_states, dtype=torch.float32)
        y_tensor = torch.as_tensor(actions, dtype=torch.long)
        d_tensor = torch.as_tensor(dones, dtype=torch.float32)
        index_tensor = torch.arange(states.shape[0], dtype=torch.long)
        dataset = TensorDataset(x_tensor, x_next_tensor, y_tensor, d_tensor, index_tensor)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        policy_net = PolicyNet(states.shape[1], n_actions, hidden_sizes)
        reward_net = RewardNet(states.shape[1], hidden_sizes)
        value_net = RewardNet(states.shape[1], hidden_sizes)
        opt_policy = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        opt_reward = torch.optim.Adam(list(reward_net.parameters()) + list(value_net.parameters()), lr=learning_rate)

        if pretrain_epochs > 0:
            for _ in range(pretrain_epochs):
                for xb, _, ab, _, _ in loader:
                    logits = policy_net(xb)
                    bc_loss = torch.nn.functional.cross_entropy(logits, ab)
                    opt_policy.zero_grad()
                    bc_loss.backward()
                    opt_policy.step()

        for iteration in range(n_iters):
            for xb, xnb, ab, db, ib in loader:
                with torch.no_grad():
                    probs = torch.softmax(policy_net(xb), dim=1)
                    sampled_actions = torch.multinomial(probs, num_samples=1).squeeze(1)
                    if transition_model is not None:
                        sampled_next_np = transition_model(
                            states=states[ib.cpu().numpy()],
                            actions=sampled_actions.cpu().numpy(),
                        )
                        sampled_next = torch.as_tensor(transform(sampled_next_np), dtype=torch.float32)
                    else:
                        sampled_next = xnb
                reward_expert = reward_net(xb)
                reward_sampled = reward_net(xb)
                value_current = value_net(xb)
                value_next_expert = value_net(xnb)
                value_next_sampled = value_net(sampled_next)
                f_expert = reward_expert + gamma * (1.0 - db) * value_next_expert - value_current
                f_sampled = reward_sampled + gamma * (1.0 - db) * value_next_sampled - value_current
                log_pi_expert = torch.log_softmax(policy_net(xb), dim=1).gather(1, ab[:, None]).squeeze(1)
                log_pi_sampled = torch.log_softmax(policy_net(xb), dim=1).gather(1, sampled_actions[:, None]).squeeze(1)
                disc_expert = f_expert - log_pi_expert
                disc_sampled = f_sampled - log_pi_sampled
                reward_loss = (
                    torch.nn.functional.binary_cross_entropy_with_logits(disc_expert, torch.ones_like(disc_expert))
                    + torch.nn.functional.binary_cross_entropy_with_logits(disc_sampled, torch.zeros_like(disc_sampled))
                )
                opt_reward.zero_grad()
                reward_loss.backward()
                opt_reward.step()

                sampled_actions = torch.multinomial(torch.softmax(policy_net(xb), dim=1), num_samples=1).squeeze(1)
                log_probs = torch.log_softmax(policy_net(xb), dim=1)
                selected_log_prob = log_probs.gather(1, sampled_actions[:, None]).squeeze(1)
                with torch.no_grad():
                    if transition_model is not None:
                        sampled_next_np = transition_model(
                            states=states[ib.cpu().numpy()],
                            actions=sampled_actions.cpu().numpy(),
                        )
                        sampled_next = torch.as_tensor(transform(sampled_next_np), dtype=torch.float32)
                    else:
                        sampled_next = xnb
                    advantages = reward_net(xb) + gamma * (1.0 - db) * value_net(sampled_next) - value_net(xb)
                    advantages = advantages - advantages.mean()
                entropy = -(torch.softmax(policy_net(xb), dim=1) * log_probs).sum(dim=1).mean()
                bc_loss = torch.nn.functional.cross_entropy(policy_net(xb), ab)
                policy_loss = -(advantages * selected_log_prob).mean() + bc_weight * bc_loss - entropy_weight * entropy
                opt_policy.zero_grad()
                policy_loss.backward()
                opt_policy.step()

            if verbose and (iteration + 1) % 50 == 0:
                with torch.no_grad():
                    probs = torch.softmax(policy_net(x_tensor), dim=1).cpu().numpy()
                nll = -np.mean(np.log(np.clip(probs[np.arange(states.shape[0]), actions], EPS, None)))
                print(f"[AIRL] iter={iteration + 1} imitation_nll={nll:.6f}")

        return EstimatedPolicy(
            n_actions=n_actions,
            kind="torch",
            parameters={
                "model": policy_net,
                "transform": transform,
                "airl_reward_model": reward_net,
                "airl_value_model": value_net,
                "prob_clip_min": prob_clip_min,
                "prob_clip_max": prob_clip_max,
            },
        )

    rng = set_random_seed(seed)
    mean, std = standardize_fit(states)

    policy_net = MLP.initialize(
        input_dim=states.shape[1],
        output_dim=n_actions,
        hidden_sizes=hidden_sizes,
        rng=rng,
        task="multiclass",
        x_mean=mean,
        x_std=std,
    )
    reward_features = state_action_features(states, actions, n_actions)
    reward_mean, reward_std = standardize_fit(reward_features)
    reward_net = MLP.initialize(
        input_dim=reward_features.shape[1],
        output_dim=1,
        hidden_sizes=hidden_sizes,
        rng=set_random_seed(seed + 7),
        task="regression",
        x_mean=reward_mean,
        x_std=reward_std,
    )

    for iteration in range(n_iters):
        neg_actions = np.array(
            [rng.choice(n_actions, p=row) for row in policy_net.predict_proba(states)],
            dtype=int,
        )
        expert_x = state_action_features(states, actions, n_actions)
        negative_x = state_action_features(states, neg_actions, n_actions)
        x_disc = np.concatenate([expert_x, negative_x], axis=0)
        log_pi_expert = np.log(np.clip(policy_net.predict_proba(states)[np.arange(states.shape[0]), actions], EPS, None))
        log_pi_negative = np.log(np.clip(policy_net.predict_proba(states)[np.arange(states.shape[0]), neg_actions], EPS, None))
        y_disc = np.concatenate([np.ones(states.shape[0]), np.zeros(states.shape[0])], axis=0)
        logits_offset = np.concatenate([-log_pi_expert, -log_pi_negative], axis=0)[:, None]
        target_logits = np.log(np.clip(y_disc[:, None] + 0.05, EPS, None)) - np.log(np.clip(1.05 - y_disc[:, None], EPS, None))
        reward_targets = target_logits + logits_offset
        reward_net.fit_regression(
            x=x_disc,
            y=reward_targets,
            learning_rate=learning_rate,
            n_epochs=8,
            batch_size=256,
            rng=set_random_seed(seed + 100 + iteration),
        )

        all_action_scores = []
        for action in range(n_actions):
            x_action = state_action_features(states, np.full(states.shape[0], action, dtype=int), n_actions)
            all_action_scores.append(reward_net.predict(x_action).reshape(-1))
        reward_logits = np.stack(all_action_scores, axis=1)
        target_actions = np.argmax(reward_logits, axis=1)
        policy_net.fit_multiclass(
            x=states,
            y=target_actions,
            learning_rate=learning_rate,
            n_epochs=4,
            batch_size=256,
            rng=set_random_seed(seed + 1000 + iteration),
        )
        if verbose and (iteration + 1) % 50 == 0:
            probs = policy_net.predict_proba(states)
            nll = -np.mean(np.log(np.clip(probs[np.arange(states.shape[0]), actions], EPS, None)))
            print(f"[AIRL] iter={iteration + 1} imitation_nll={nll:.6f}")

    return EstimatedPolicy(
        n_actions=n_actions,
        kind="mlp",
        parameters={"network": policy_net, "prob_clip_min": prob_clip_min, "prob_clip_max": prob_clip_max},
    )


def fit_spl_gd_policy(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    rewards: np.ndarray,
    n_actions: int,
    gamma: float = 0.95,
    regression_targets: np.ndarray | None = None,
    learning_rate: float = 0.05,
    n_iters: int = 200,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[EstimatedPolicy, LinearRewardFunction]:
    """Linear SPL-GD/DDC baseline.

    The reward is parameterized linearly in state-action features, and a soft
    value iteration style target is optimized by gradient descent.
    """
    rng = set_random_seed(seed)
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    rewards = np.asarray(rewards, dtype=float).reshape(-1)
    targets = rewards if regression_targets is None else np.asarray(regression_targets, dtype=float).reshape(-1)
    theta = rng.normal(scale=0.05, size=_expand_action_features(states[:1], n_actions).shape[-1])
    reward_fn = LinearRewardFunction(n_actions=n_actions, theta=theta)
    feats = reward_fn.features(states, actions)
    for iteration in range(n_iters):
        pred = feats @ reward_fn.theta
        grad = feats.T @ (pred - targets) / max(states.shape[0], 1)
        reward_fn.theta -= learning_rate * grad
        if verbose and (iteration + 1) % 50 == 0:
            mse = np.mean((pred - targets) ** 2)
            print(f"[SPL-GD] iter={iteration + 1} mse={mse:.6f}")
    class LinearWrapper:
        def __call__(self, x: np.ndarray, action_count: int) -> np.ndarray:
            feat = _expand_action_features(np.asarray(x, dtype=float), action_count)
            return feat
    policy = EstimatedPolicy(
        n_actions=n_actions,
        kind="linear-softmax",
        parameters={"theta": reward_fn.theta, "feature_map": LinearWrapper()},
    )
    return policy, reward_fn
