"""Self-contained fitted Q evaluation implementations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

try:
    from .utils import MLP, fit_gradient_boosted_regressor, one_hot, standardize_fit, state_action_features
except ImportError:  # pragma: no cover - supports direct script execution
    from utils import MLP, fit_gradient_boosted_regressor, one_hot, standardize_fit, state_action_features

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None

try:
    try:
        from .lightgbm_fqe import fit_fqe_boosted as fit_fqe_boosted_lgbm
    except ImportError:  # pragma: no cover - supports direct script execution
        from lightgbm_fqe import fit_fqe_boosted as fit_fqe_boosted_lgbm
except Exception:  # pragma: no cover - optional dependency
    fit_fqe_boosted_lgbm = None


@dataclass
class QFunctionEstimate:
    """Unified prediction interface for estimated Q-functions."""

    n_actions: int
    kind: str
    parameters: Dict[str, object]

    def predict_q(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Predict Q(s, a) for a batch of state-action pairs."""
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=int).reshape(-1)
        if self.kind == "neural":
            pred = self.parameters["network"].predict(states)
            return pred[np.arange(states.shape[0]), actions]
        if self.kind == "torch":
            x = self.parameters["transform"](states)
            model = self.parameters["model"]
            model.eval()
            with torch.no_grad():
                pred = model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
            return pred[np.arange(states.shape[0]), actions]
        if self.kind == "boosted":
            x = state_action_features(states, actions, self.n_actions)
            return self.parameters["model"].predict(x)
        raise ValueError(f"Unsupported Q estimator: {self.kind}")

    def predict_all_actions(self, states: np.ndarray) -> np.ndarray:
        """Predict Q(s, a) for every action."""
        states = np.asarray(states, dtype=float)
        if self.kind == "neural":
            return self.parameters["network"].predict(states)
        if self.kind == "torch":
            x = self.parameters["transform"](states)
            model = self.parameters["model"]
            model.eval()
            with torch.no_grad():
                return model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
        if self.kind == "boosted":
            predictions = []
            for action in range(self.n_actions):
                action_vec = np.full(states.shape[0], action, dtype=int)
                x = state_action_features(states, action_vec, self.n_actions)
                predictions.append(self.parameters["model"].predict(x))
            return np.stack(predictions, axis=1)
        raise ValueError(f"Unsupported Q estimator: {self.kind}")


def _expected_next_value(policy, q_estimate: QFunctionEstimate | None, next_states: np.ndarray, n_actions: int) -> np.ndarray:
    """Compute E_pi[Q(next_state, A)] under the current Q estimate."""
    if q_estimate is None:
        return np.zeros(next_states.shape[0], dtype=float)
    q_next = q_estimate.predict_all_actions(next_states)
    pi_next = policy.predict_proba(next_states)
    return np.sum(pi_next * q_next, axis=1)


def _torch_batch_indices(n_samples: int, batch_size: int):
    """Yield shuffled minibatch indices without DataLoader overhead."""
    if torch is None:
        raise RuntimeError("Torch batching requested without torch installed.")
    order = torch.randperm(n_samples)
    for start in range(0, n_samples, batch_size):
        yield order[start : start + batch_size]


def fit_fqe_neural(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    policy,
    n_actions: int,
    gamma: float = 0.95,
    hidden_sizes: Sequence[int] = (128, 128),
    learning_rate: float = 5e-3,
    n_fqe_iters: int = 60,
    epochs_per_iter: int = 30,
    seed: int = 0,
    verbose: bool = False,
) -> QFunctionEstimate:
    """Fitted Q evaluation with a pooled-but-action-flexible neural Q model.

    The torch implementation uses a shared state trunk together with a dueling
    decomposition ``Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)``. This keeps
    all actions equally expressive while still pooling information through the
    common trunk and value baseline.
    """
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    rewards = np.asarray(rewards, dtype=float).reshape(-1)
    next_states = np.asarray(next_states, dtype=float)
    dones = np.asarray(dones, dtype=float).reshape(-1)

    if torch is not None:
        torch.manual_seed(seed)
        mean, std = standardize_fit(states)

        def transform(x: np.ndarray) -> np.ndarray:
            return (np.asarray(x, dtype=float) - mean) / std

        class DuelingQNet(nn.Module):
            def __init__(self, input_dim: int, output_dim: int, hidden: Sequence[int]) -> None:
                super().__init__()
                layers = []
                prev = input_dim
                for h in hidden:
                    layers.append(nn.Linear(prev, h))
                    layers.append(nn.ReLU())
                    prev = h
                self.trunk = nn.Sequential(*layers)
                self.value_head = nn.Linear(prev, 1)
                self.advantage_head = nn.Linear(prev, output_dim)

            def forward(self, x):
                trunk = self.trunk(x)
                value = self.value_head(trunk)
                advantage = self.advantage_head(trunk)
                centered_advantage = advantage - advantage.mean(dim=1, keepdim=True)
                return value + centered_advantage

        x = transform(states)
        x_t = torch.as_tensor(x, dtype=torch.float32)
        a_t = torch.as_tensor(actions, dtype=torch.long)
        batch_size = min(256, states.shape[0])
        q_net = DuelingQNet(states.shape[1], n_actions, hidden_sizes)
        optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
        q_estimate = None
        for iteration in range(n_fqe_iters):
            continuation = _expected_next_value(policy, q_estimate, next_states, n_actions)
            targets = rewards + gamma * (1.0 - dones) * continuation
            y_t = torch.as_tensor(targets, dtype=torch.float32)
            for _ in range(epochs_per_iter):
                q_net.train()
                for batch_idx in _torch_batch_indices(states.shape[0], batch_size):
                    xb = x_t[batch_idx]
                    ab = a_t[batch_idx]
                    yb = y_t[batch_idx]
                    pred = q_net(xb).gather(1, ab[:, None]).squeeze(1)
                    loss = ((pred - yb) ** 2).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            q_estimate = QFunctionEstimate(n_actions=n_actions, kind="torch", parameters={"model": q_net, "transform": transform})
            if verbose and (iteration + 1) % 10 == 0:
                mse = np.mean((q_estimate.predict_q(states, actions) - targets) ** 2)
                print(f"[Neural FQE] iter={iteration + 1} bellman_mse={mse:.6f}")
        return q_estimate

    mean, std = standardize_fit(states)
    network = MLP.initialize(
        input_dim=states.shape[1],
        output_dim=n_actions,
        hidden_sizes=hidden_sizes,
        rng=np.random.default_rng(seed),
        task="regression",
        x_mean=mean,
        x_std=std,
    )
    q_estimate = None
    for iteration in range(n_fqe_iters):
        continuation = _expected_next_value(policy, q_estimate, next_states, n_actions)
        targets = rewards + gamma * (1.0 - dones) * continuation
        target_matrix = network.predict(states)
        target_matrix[np.arange(states.shape[0]), actions] = targets
        network.fit_regression(
            x=states,
            y=target_matrix,
            learning_rate=learning_rate,
            n_epochs=epochs_per_iter,
            batch_size=256,
            rng=np.random.default_rng(seed + iteration + 1),
        )
        q_estimate = QFunctionEstimate(n_actions=n_actions, kind="neural", parameters={"network": network})
        if verbose and (iteration + 1) % 10 == 0:
            mse = np.mean((q_estimate.predict_q(states, actions) - targets) ** 2)
            print(f"[Neural FQE] iter={iteration + 1} bellman_mse={mse:.6f}")
    return q_estimate


def fit_fqe_boosted(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    policy,
    n_actions: int,
    gamma: float = 0.95,
    n_fqe_iters: int = 40,
    n_estimators: int = 80,
    learning_rate: float = 0.05,
    max_depth: int = 2,
    min_leaf: int = 20,
    verbose: bool = False,
) -> QFunctionEstimate:
    """Fitted Q evaluation with lightweight gradient-boosted trees."""
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    rewards = np.asarray(rewards, dtype=float).reshape(-1)
    next_states = np.asarray(next_states, dtype=float)
    dones = np.asarray(dones, dtype=float).reshape(-1)

    if fit_fqe_boosted_lgbm is not None:
        next_actions = policy.sample_actions(next_states, seed=0)
        result = fit_fqe_boosted_lgbm(
            S=states,
            A=one_hot(actions, n_actions),
            S_next=next_states,
            A_next=one_hot(next_actions, n_actions),
            rewards=rewards,
            discount_factor=gamma,
            is_terminal_outcome=dones,
            lgb_params={
                "learning_rate": learning_rate,
                "num_leaves": 32,
                "min_data_in_leaf": min_leaf,
            },
            fit_control={
                "num_boost_rounds": n_estimators,
                "early_stopping_rounds": max(10, n_fqe_iters),
            },
            refit_on_all_data=True,
            verbose=verbose,
        )
        return QFunctionEstimate(n_actions=n_actions, kind="boosted", parameters={"model": result["model"]})

    q_estimate = None
    model = None
    x_train = state_action_features(states, actions, n_actions)
    for iteration in range(n_fqe_iters):
        if q_estimate is None:
            continuation = np.zeros(states.shape[0], dtype=float)
        else:
            continuation = _expected_next_value(policy, q_estimate, next_states, n_actions)
        targets = rewards + gamma * (1.0 - dones) * continuation
        model = fit_gradient_boosted_regressor(
            x=x_train,
            y=targets,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_leaf=min_leaf,
            verbose=False,
        )
        q_estimate = QFunctionEstimate(n_actions=n_actions, kind="boosted", parameters={"model": model})
        if verbose and (iteration + 1) % 10 == 0:
            mse = np.mean((q_estimate.predict_q(states, actions) - targets) ** 2)
            print(f"[Boosted FQE] iter={iteration + 1} bellman_mse={mse:.6f}")
    return q_estimate
