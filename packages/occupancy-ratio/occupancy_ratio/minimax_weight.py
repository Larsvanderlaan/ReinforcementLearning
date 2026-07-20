from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Literal, Optional

import numpy as np

from occupancy_ratio.diagnostics import postprocess_weights, weight_summary
from occupancy_ratio.fit_occupancy_ratio import (
    Array,
    _as_2d,
    _safe_divide,
    _validate_aligned_inputs,
    _validate_initial_action_inputs,
    _validate_initial_state_inputs,
    _validate_next_target_actions,
)
from occupancy_ratio.google_dualdice import (
    GoogleDualDICEConfig,
    fit_google_dualdice_occupancy_ratio,
    preflight_google_dualdice,
)


MinimaxWeightMethod = Literal[
    "auto",
    "google_policy_eval_dualdice",
    "google_dice_rl_dualdice_exact",
    "google_dice_rl_recommended",
    "scope_rl_minimax_state_action",
    "scope_rl_minimax_state",
]

DEFAULT_MINIMAX_WEIGHT_METHOD: Literal["google_policy_eval_dualdice"] = "google_policy_eval_dualdice"

GOOGLE_DICE_RL_DUALDICE_EXACT_FLAGS: dict[str, float | bool] = {
    "primal_regularizer": 0.0,
    "dual_regularizer": 1.0,
    "zero_reward": True,
    "norm_regularizer": 0.0,
    "zeta_pos": False,
}

GOOGLE_DICE_RL_RECOMMENDED_FLAGS: dict[str, float | bool] = {
    "primal_regularizer": 0.0,
    "dual_regularizer": 1.0,
    "zero_reward": False,
    "norm_regularizer": 1.0,
    "zeta_pos": True,
}


@dataclass(frozen=True)
class GoogleDICERLConfig:
    """Configuration for the optional official Google DICE-RL NeuralDice backend.

    Parameters
    ----------
    dice_rl_repo_path:
        Path to a clone or importable checkout of
        ``https://github.com/google-research/dice_rl``.
    num_steps:
        Number of NeuralDice training steps.
    batch_size:
        Transition minibatch size.
    learning_rate:
        Learning rate for the Nu, Zeta, and Lagrange optimizers.
    hidden_dims:
        Fully connected hidden dimensions for the Nu and Zeta networks.
    seed:
        Random seed used for NumPy and TensorFlow.
    prediction_max:
        Optional upper cap applied to predicted Zeta weights.
    normalize_predictions:
        Whether to normalize predicted weights to mean one on each query block.
    limit_tf_threads:
        Whether to request single-threaded TensorFlow execution when possible.
    """

    dice_rl_repo_path: str | Path = Path("/tmp/dice_rl")
    num_steps: int = 1000
    batch_size: int = 128
    learning_rate: float = 1e-4
    hidden_dims: tuple[int, ...] = (64, 64)
    seed: int = 123
    prediction_max: Optional[float] = None
    normalize_predictions: bool = False
    limit_tf_threads: bool = True

    def __post_init__(self) -> None:
        if int(self.num_steps) <= 0:
            raise ValueError("num_steps must be positive.")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if not self.hidden_dims or any(int(width) <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths.")
        if self.prediction_max is not None and float(self.prediction_max) <= 0.0:
            raise ValueError("prediction_max must be positive when supplied.")


@dataclass(frozen=True)
class ScopeRLMinimaxWeightConfig:
    """Configuration for optional SCOPE-RL minimax weight learners.

    Parameters
    ----------
    scope_rl_repo_path:
        Optional path to a local SCOPE-RL checkout. When omitted, imports use
        the active Python environment.
    n_steps:
        Number of SCOPE-RL minimax optimization steps.
    n_steps_per_epoch:
        Number of optimization steps reported as one SCOPE-RL epoch.
    batch_size:
        SCOPE-RL learner minibatch size.
    learning_rate:
        Optimizer learning rate.
    hidden_dim:
        Hidden width for the SCOPE-RL weight function.
    bandwidth:
        Gaussian-kernel bandwidth used by SCOPE-RL.
    regularization_weight:
        Weight for the mean-one regularization term.
    seed:
        Random seed passed to SCOPE-RL.
    device:
        Torch device. Defaults to CPU to keep package defaults portable.
    prediction_max:
        Optional upper cap applied to predicted weights.
    normalize_predictions:
        Whether to normalize predicted weights to mean one on each query block.
    limit_torch_threads:
        Whether to pin Torch/BLAS thread counts before importing SCOPE-RL. This
        avoids native crashes observed in SCOPE-RL's Gaussian-kernel objective
        on some macOS/PyTorch stacks.
    """

    scope_rl_repo_path: str | Path | None = None
    n_steps: int = 10000
    n_steps_per_epoch: int = 10000
    batch_size: int = 128
    learning_rate: float = 1e-4
    hidden_dim: int = 100
    bandwidth: float = 1.0
    bandwidth_selection: Literal["fixed", "median"] = "fixed"
    bandwidth_num_pairs: int = 4096
    standardize_inputs: bool = False
    regularization_weight: float = 1.0
    seed: int = 123
    device: str = "cpu"
    prediction_max: Optional[float] = None
    normalize_predictions: bool = False
    limit_torch_threads: bool = True

    def __post_init__(self) -> None:
        if int(self.n_steps) <= 0:
            raise ValueError("n_steps must be positive.")
        if int(self.n_steps_per_epoch) <= 0:
            raise ValueError("n_steps_per_epoch must be positive.")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive.")
        if float(self.bandwidth) <= 0.0:
            raise ValueError("bandwidth must be positive.")
        if self.bandwidth_selection not in {"fixed", "median"}:
            raise ValueError("bandwidth_selection must be 'fixed' or 'median'.")
        if int(self.bandwidth_num_pairs) <= 0:
            raise ValueError("bandwidth_num_pairs must be positive.")
        if float(self.regularization_weight) < 0.0:
            raise ValueError("regularization_weight must be nonnegative.")
        if self.prediction_max is not None and float(self.prediction_max) <= 0.0:
            raise ValueError("prediction_max must be positive when supplied.")


@dataclass(frozen=True)
class MinimaxWeightConfig:
    """Shared configuration for :func:`fit_minimax_weight`.

    Parameters
    ----------
    method:
        Minimax-weight backend. ``"auto"`` resolves through
        :data:`DEFAULT_MINIMAX_WEIGHT_METHOD`.
    google_policy_eval:
        Configuration for Google Research ``policy_eval.dual_dice``.
    google_dice_rl:
        Configuration for Google Research DICE-RL NeuralDice.
    scope_rl:
        Configuration for SCOPE-RL minimax weight learners.
    """

    method: MinimaxWeightMethod = "auto"
    google_policy_eval: GoogleDualDICEConfig = field(default_factory=GoogleDualDICEConfig)
    google_dice_rl: GoogleDICERLConfig = field(default_factory=GoogleDICERLConfig)
    scope_rl: ScopeRLMinimaxWeightConfig = field(default_factory=ScopeRLMinimaxWeightConfig)

    def __post_init__(self) -> None:
        _resolve_minimax_method(self.method)


@dataclass(frozen=True)
class MinimaxWeightPreflight:
    """Availability check for one minimax-weight backend."""

    available: bool
    reason: str
    method: str
    repo_path: Path | None = None


@dataclass
class MinimaxWeightModel:
    """Common occupancy-ratio model wrapper for minimax-weight backends."""

    backend_model: Any
    method: str
    gamma: float
    state_dim: int
    action_dim: int
    diagnostics: dict[str, Any]
    config: MinimaxWeightConfig
    _state_action_predictor: Callable[[Array, Array], Array] | None = None
    _state_predictor: Callable[[Array], Array] | None = None
    prediction_max: Optional[float] = None
    normalize_predictions: bool = False

    def predict_state_action_ratio(self, states: Array, actions: Array, *, clip: bool = True) -> Array:
        """Predict state-action occupancy weights on query rows."""
        S, A = self._validate_query(states, actions)
        if self._state_action_predictor is not None:
            raw = np.asarray(self._state_action_predictor(S, A), dtype=np.float64).reshape(-1)
        elif hasattr(self.backend_model, "predict_state_action_ratio"):
            return np.asarray(self.backend_model.predict_state_action_ratio(S, A, clip=clip), dtype=np.float64).reshape(-1)
        elif self._state_predictor is not None:
            raw = np.asarray(self._state_predictor(S), dtype=np.float64).reshape(-1)
        else:
            raise TypeError(f"{self.method} model cannot predict state-action ratios.")
        if raw.shape[0] != S.shape[0]:
            raise ValueError("state-action ratio predictor returned the wrong number of rows.")
        if not clip:
            return raw
        return _postprocess_minimax_weights(
            raw,
            prediction_max=self.prediction_max,
            normalize=self.normalize_predictions,
        )

    def predict_action_ratio(self, states: Array, actions: Array, *, clip: bool = True) -> Array:
        """Return ones because these minimax backends estimate marginal weights directly."""
        S, _ = self._validate_query(states, actions)
        return np.ones(S.shape[0], dtype=np.float64)

    def predict_state_ratio(self, states: Array, actions: Array, *, clip: bool = True) -> Array:
        """Predict the implied state ratio."""
        if self._state_predictor is not None:
            S = _as_2d(states, "states").astype(np.float32, copy=False)
            if S.shape[1] != self.state_dim:
                raise ValueError(f"states must have {self.state_dim} columns.")
            raw = np.asarray(self._state_predictor(S), dtype=np.float64).reshape(-1)
            if raw.shape[0] != S.shape[0]:
                raise ValueError("state ratio predictor returned the wrong number of rows.")
            if not clip:
                return raw
            return _postprocess_minimax_weights(
                raw,
                prediction_max=self.prediction_max,
                normalize=self.normalize_predictions,
            )
        state_action = self.predict_state_action_ratio(states, actions, clip=clip)
        action = self.predict_action_ratio(states, actions, clip=clip)
        return _safe_divide(state_action, action)

    def predict_for_target_actions(
        self,
        states: Array,
        target_actions: Array,
        *,
        observed_actions: Optional[Array] = None,
        clip: bool = True,
    ) -> dict[str, Array]:
        """Predict ratios for target and, optionally, observed action rows."""
        out = dict(
            target_state_action_ratio=self.predict_state_action_ratio(states, target_actions, clip=clip),
            target_action_ratio=self.predict_action_ratio(states, target_actions, clip=clip),
        )
        out["target_state_ratio"] = _safe_divide(
            out["target_state_action_ratio"],
            out["target_action_ratio"],
        )
        if observed_actions is not None:
            out["observed_state_action_ratio"] = self.predict_state_action_ratio(states, observed_actions, clip=clip)
            out["observed_action_ratio"] = self.predict_action_ratio(states, observed_actions, clip=clip)
            out["observed_state_ratio"] = _safe_divide(
                out["observed_state_action_ratio"],
                out["observed_action_ratio"],
            )
        return out

    def _validate_query(self, states: Array, actions: Array) -> tuple[Array, Array]:
        S = _as_2d(states, "states").astype(np.float32, copy=False)
        A = _as_2d(actions, "actions").astype(np.float32, copy=False)
        if S.shape[0] != A.shape[0]:
            raise ValueError("states and actions must have the same number of rows.")
        if S.shape[1] != self.state_dim:
            raise ValueError(f"states must have {self.state_dim} columns.")
        if A.shape[1] != self.action_dim:
            raise ValueError(f"actions must have {self.action_dim} columns.")
        return S, A


def preflight_minimax_weight(
    method: MinimaxWeightMethod = "auto",
    config: MinimaxWeightConfig | None = None,
) -> MinimaxWeightPreflight:
    """Check whether a minimax-weight backend can run.

    Parameters
    ----------
    method:
        Backend to check. ``"auto"`` resolves to the package default.
    config:
        Optional shared config containing backend paths.

    Returns
    -------
    MinimaxWeightPreflight
        Availability status and a user-facing reason when unavailable.
    """
    cfg = MinimaxWeightConfig(method=method) if config is None else config
    resolved = _resolve_minimax_method(method if method != "auto" else cfg.method)
    if resolved == "google_policy_eval_dualdice":
        preflight = preflight_google_dualdice(cfg.google_policy_eval.google_research_path)
        return MinimaxWeightPreflight(
            available=bool(preflight.available),
            reason=str(preflight.reason),
            method=resolved,
            repo_path=preflight.repo_path,
        )
    if resolved in {"google_dice_rl_dualdice_exact", "google_dice_rl_recommended"}:
        return preflight_google_dice_rl(cfg.google_dice_rl.dice_rl_repo_path, method=resolved)
    if resolved in {"scope_rl_minimax_state_action", "scope_rl_minimax_state"}:
        return preflight_scope_rl(cfg.scope_rl.scope_rl_repo_path, method=resolved)
    raise AssertionError(f"Unhandled minimax-weight method {resolved!r}.")


def fit_minimax_weight(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    target_actions: Array,
    target_next_actions: Array | None,
    gamma: float,
    initial_states: Array | None = None,
    initial_actions: Array | None = None,
    terminals: Optional[Array] = None,
    rewards: Optional[Array] = None,
    sample_weight: Optional[Array] = None,
    initial_weights: Optional[Array] = None,
    method: MinimaxWeightMethod | None = None,
    config: MinimaxWeightConfig | None = None,
    episode_ids: Optional[Array] = None,
    timesteps: Optional[Array] = None,
    step_per_trajectory: int | None = None,
    behavior_action_pscore: Optional[Array] = None,
) -> MinimaxWeightModel:
    """Fit a supported minimax or DICE-style occupancy-weight estimator.

    Parameters
    ----------
    states, actions, next_states, target_actions, target_next_actions:
        Logged transition rows and evaluation-policy actions. These match the
        core :func:`fit_discounted_occupancy_ratio` array contract.
    gamma:
        Discount factor for the occupancy weights. Use a large value below one
        for near-stationary weights.
    initial_states, initial_actions:
        Target-policy initial state-action rows. Required by Google DICE
        methods and used when available by trajectory learners.
    terminals:
        Optional terminal indicators for Google DICE methods.
    rewards:
        Optional reward rows for Google DICE-RL's regularized NeuralDice
        objective. Exact DualDICE-recovery mode ignores rewards.
    sample_weight:
        Optional nonnegative transition-row weights.
    initial_weights:
        Optional nonnegative weights for initial rows. Google DICE-RL uses
        these as sampling probabilities for initial minibatches.
    method:
        Backend to fit. Overrides ``config.method`` when supplied.
    config:
        Shared backend configuration.
    episode_ids, timesteps, step_per_trajectory:
        Optional trajectory metadata for SCOPE-RL. If omitted, SCOPE-RL treats
        the input rows as one flattened trajectory.
    behavior_action_pscore:
        Behavior action propensities for ``"scope_rl_minimax_state"``.

    Returns
    -------
    MinimaxWeightModel
        Common wrapper exposing occupancy-ratio prediction helpers.
    """
    cfg = MinimaxWeightConfig() if config is None else config
    resolved = _resolve_minimax_method(method if method is not None else cfg.method)
    common = _prepare_common_inputs(
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=target_actions,
        target_next_actions=target_next_actions,
        gamma=gamma,
        initial_states=initial_states,
        initial_actions=initial_actions,
        terminals=terminals,
        rewards=rewards,
        sample_weight=sample_weight,
        initial_weights=initial_weights,
        require_target_next_actions=resolved.startswith("google_"),
        require_initial_actions=resolved.startswith("google_"),
    )
    if resolved == "google_policy_eval_dualdice":
        return _fit_google_policy_eval_dualdice(common, cfg)
    if resolved in {"google_dice_rl_dualdice_exact", "google_dice_rl_recommended"}:
        flags = (
            GOOGLE_DICE_RL_DUALDICE_EXACT_FLAGS
            if resolved == "google_dice_rl_dualdice_exact"
            else GOOGLE_DICE_RL_RECOMMENDED_FLAGS
        )
        return _fit_google_dice_rl(common, cfg, method=resolved, flags=flags)
    if resolved in {"scope_rl_minimax_state_action", "scope_rl_minimax_state"}:
        return _fit_scope_rl_minimax_weight(
            common,
            cfg,
            method=resolved,
            episode_ids=episode_ids,
            timesteps=timesteps,
            step_per_trajectory=step_per_trajectory,
            behavior_action_pscore=behavior_action_pscore,
        )
    raise AssertionError(f"Unhandled minimax-weight method {resolved!r}.")


def preflight_google_dice_rl(
    repo_path: str | Path = Path("/tmp/dice_rl"),
    *,
    method: str = "google_dice_rl_recommended",
) -> MinimaxWeightPreflight:
    """Check whether the optional Google Research DICE-RL backend can run."""
    path = Path(repo_path)
    package_dir = _dice_rl_package_dir(path)
    if package_dir is None:
        return MinimaxWeightPreflight(
            available=False,
            reason=(
                f"Missing DICE-RL source under {path}. Clone "
                "https://github.com/google-research/dice_rl to an importable "
                "directory such as /tmp/dice_rl and pass dice_rl_repo_path."
            ),
            method=method,
            repo_path=path,
        )
    if package_dir.name != "dice_rl":
        return MinimaxWeightPreflight(
            available=False,
            reason=f"DICE-RL package directory must be named 'dice_rl' for absolute imports; got {package_dir}.",
            method=method,
            repo_path=path,
        )
    try:
        _ensure_dice_rl_importable(path)
        import tensorflow  # noqa: F401
        import tf_agents  # noqa: F401
        from dice_rl.data import dataset as dice_dataset  # noqa: F401
        from dice_rl.estimators.neural_dice import NeuralDice  # noqa: F401
        from dice_rl.networks.value_network import ValueNetwork  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return MinimaxWeightPreflight(
            available=False,
            reason=f"Google DICE-RL import failed: {type(exc).__name__}: {exc}",
            method=method,
            repo_path=path,
        )
    return MinimaxWeightPreflight(available=True, reason="", method=method, repo_path=path)


def preflight_scope_rl(
    repo_path: str | Path | None = None,
    *,
    method: str = "scope_rl_minimax_state_action",
) -> MinimaxWeightPreflight:
    """Check whether the optional SCOPE-RL minimax backend can run."""
    path = None if repo_path is None else Path(repo_path)
    try:
        _ensure_scope_rl_importable(path)
        from scope_rl.ope.weight_value_learning import (  # noqa: F401
            ContinuousMinimaxStateActionWeightLearning,
            ContinuousMinimaxStateWeightLearning,
        )
        from scope_rl.ope.weight_value_learning.function import (  # noqa: F401
            ContinuousStateActionWeightFunction,
            StateWeightFunction,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return MinimaxWeightPreflight(
            available=False,
            reason=f"SCOPE-RL import failed: {type(exc).__name__}: {exc}",
            method=method,
            repo_path=path,
        )
    return MinimaxWeightPreflight(available=True, reason="", method=method, repo_path=path)


def _fit_google_policy_eval_dualdice(common: dict[str, Any], cfg: MinimaxWeightConfig) -> MinimaxWeightModel:
    google_model = fit_google_dualdice_occupancy_ratio(
        states=common["S"],
        actions=common["A"],
        next_states=common["S_next"],
        target_actions=common["A_pi"],
        gamma=common["gamma"],
        initial_states=common["S_initial"],
        initial_actions=common["A_initial"],
        initial_weights=common["initial_weight"],
        target_next_actions=common["A_pi_next"],
        terminals=common["terminals"],
        sample_weight=common["sample_weight"],
        config=cfg.google_policy_eval,
    )
    diagnostics = _prefixed_backend_diagnostics(
        method="google_policy_eval_dualdice",
        backend="google_policy_eval_dualdice",
        gamma=common["gamma"],
        raw=google_model.predict_state_action_ratio(common["S"], common["A"], clip=False),
        weights=google_model.predict_state_action_ratio(common["S"], common["A"], clip=True),
        extra=google_model.diagnostics,
        prediction_max=cfg.google_policy_eval.prediction_max,
    )
    return MinimaxWeightModel(
        backend_model=google_model,
        method="google_policy_eval_dualdice",
        gamma=common["gamma"],
        state_dim=common["S"].shape[1],
        action_dim=common["A"].shape[1],
        diagnostics=diagnostics,
        config=cfg,
        prediction_max=cfg.google_policy_eval.prediction_max,
        normalize_predictions=bool(cfg.google_policy_eval.normalize_predictions),
    )


def _fit_google_dice_rl(
    common: dict[str, Any],
    cfg: MinimaxWeightConfig,
    *,
    method: str,
    flags: dict[str, float | bool],
) -> MinimaxWeightModel:
    dice_cfg = cfg.google_dice_rl
    preflight = preflight_google_dice_rl(dice_cfg.dice_rl_repo_path, method=method)
    if not preflight.available:
        raise ModuleNotFoundError(preflight.reason)
    _ensure_dice_rl_importable(preflight.repo_path or dice_cfg.dice_rl_repo_path)
    import tensorflow as tf  # noqa: PLC0415
    from tf_agents.specs import tensor_spec  # noqa: PLC0415
    from dice_rl.data import dataset as dice_dataset  # noqa: PLC0415
    from dice_rl.estimators.neural_dice import NeuralDice  # noqa: PLC0415
    from dice_rl.networks.value_network import ValueNetwork  # noqa: PLC0415

    if dice_cfg.limit_tf_threads:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass
    np.random.seed(int(dice_cfg.seed))
    tf.random.set_seed(int(dice_cfg.seed))

    S = common["S"]
    A = common["A"]
    S_next = common["S_next"]
    A_pi_next = common["A_pi_next"]
    S_initial = common["S_initial"]
    A_initial = common["A_initial"]
    rewards = common["rewards"].astype(np.float32, copy=False)
    masks = (1.0 - common["terminals"]).astype(np.float32, copy=False)

    step_spec = dice_dataset.EnvStep(
        step_type=tensor_spec.TensorSpec((), tf.int32),
        step_num=tensor_spec.TensorSpec((), tf.int32),
        observation=tensor_spec.TensorSpec((S.shape[1],), tf.float32),
        action=tensor_spec.TensorSpec((A.shape[1],), tf.float32),
        reward=tensor_spec.TensorSpec((), tf.float32),
        discount=tensor_spec.TensorSpec((), tf.float32),
        policy_info={},
        env_info={},
        other_info={},
    )
    input_spec = (step_spec.observation, step_spec.action)
    hidden_dims = tuple(int(width) for width in dice_cfg.hidden_dims)
    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=int(dice_cfg.seed))
    last_kernel_initializer = tf.keras.initializers.GlorotUniform(seed=int(dice_cfg.seed + 1_003))
    zeta_activation = tf.math.square if bool(flags["zeta_pos"]) else tf.identity
    nu_network = ValueNetwork(
        input_spec,
        fc_layer_params=hidden_dims,
        activation_fn=tf.nn.relu,
        kernel_initializer=kernel_initializer,
        last_kernel_initializer=last_kernel_initializer,
    )
    zeta_network = ValueNetwork(
        input_spec,
        fc_layer_params=hidden_dims,
        activation_fn=tf.nn.relu,
        output_activation_fn=zeta_activation,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(dice_cfg.seed + 2_003)),
        last_kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(dice_cfg.seed + 3_003)),
    )
    optimizer_kwargs = {"learning_rate": float(dice_cfg.learning_rate), "clipvalue": 1.0}
    DirectActionNeuralDice = _direct_action_neural_dice_class(NeuralDice)
    estimator = DirectActionNeuralDice(
        step_spec,
        nu_network,
        zeta_network,
        _dice_rl_adam_optimizer(tf, **optimizer_kwargs),
        _dice_rl_adam_optimizer(tf, **optimizer_kwargs),
        _dice_rl_adam_optimizer(tf, **optimizer_kwargs),
        float(common["gamma"]),
        zero_reward=bool(flags["zero_reward"]),
        f_exponent=2.0,
        primal_regularizer=float(flags["primal_regularizer"]),
        dual_regularizer=float(flags["dual_regularizer"]),
        norm_regularizer=float(flags["norm_regularizer"]),
    )
    states_tf = tf.convert_to_tensor(S, dtype=tf.float32)
    actions_tf = tf.convert_to_tensor(A, dtype=tf.float32)
    next_states_tf = tf.convert_to_tensor(S_next, dtype=tf.float32)
    next_actions_tf = tf.convert_to_tensor(A_pi_next, dtype=tf.float32)
    rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
    masks_tf = tf.convert_to_tensor(masks, dtype=tf.float32)
    initial_states_tf = tf.convert_to_tensor(S_initial, dtype=tf.float32)
    initial_actions_tf = tf.convert_to_tensor(A_initial, dtype=tf.float32)

    rng = np.random.default_rng(int(dice_cfg.seed) + 77_003)
    row_probs = _probabilities(common["sample_weight"])
    init_probs = _probabilities(common["initial_weight"])
    actual_batch_size = min(int(dice_cfg.batch_size), S.shape[0])
    losses: list[tuple[float, float, float]] = []
    start = time.perf_counter()
    for step in range(int(dice_cfg.num_steps)):
        idx = rng.choice(S.shape[0], size=actual_batch_size, replace=True, p=row_probs)
        init_idx = rng.choice(S_initial.shape[0], size=actual_batch_size, replace=True, p=init_probs)
        experience = _dice_rl_experience_batch(
            dice_dataset,
            tf,
            states=tf.gather(states_tf, idx),
            actions=tf.gather(actions_tf, idx),
            next_states=tf.gather(next_states_tf, idx),
            next_actions=tf.gather(next_actions_tf, idx),
            rewards=tf.gather(rewards_tf, idx),
            masks=tf.gather(masks_tf, idx),
        )
        initial_step = _dice_rl_initial_batch(
            dice_dataset,
            tf,
            states=tf.gather(initial_states_tf, init_idx),
            actions=tf.gather(initial_actions_tf, init_idx),
        )
        loss_tuple = estimator.train_step(initial_step, experience, None)
        if step == 0 or step == int(dice_cfg.num_steps) - 1 or (step + 1) % 250 == 0:
            losses.append(tuple(float(value.numpy()) for value in loss_tuple))

    def predict_zeta(states_query: Array, actions_query: Array) -> Array:
        states_tf_query = tf.convert_to_tensor(states_query.astype(np.float32, copy=False), dtype=tf.float32)
        actions_tf_query = tf.convert_to_tensor(actions_query.astype(np.float32, copy=False), dtype=tf.float32)
        return zeta_network((states_tf_query, actions_tf_query))[0].numpy().astype(np.float64).reshape(-1)

    raw = predict_zeta(S, A)
    weights = _postprocess_minimax_weights(
        raw,
        prediction_max=dice_cfg.prediction_max,
        normalize=bool(dice_cfg.normalize_predictions),
    )
    extra: dict[str, Any] = {
        "num_steps": float(dice_cfg.num_steps),
        "batch_size": float(actual_batch_size),
        "initial_batch_size": float(actual_batch_size),
        "learning_rate": float(dice_cfg.learning_rate),
        "hidden_dims": "x".join(str(width) for width in hidden_dims),
        "runtime_sec": float(time.perf_counter() - start),
        "exact_dualdice_flags": float(_flags_match(flags, GOOGLE_DICE_RL_DUALDICE_EXACT_FLAGS)),
        "recommended_flags": float(_flags_match(flags, GOOGLE_DICE_RL_RECOMMENDED_FLAGS)),
        "direct_action_target_policy": 1.0,
        "target_policy_lookup_reference_rows": 0.0,
    }
    for key, value in flags.items():
        extra[key] = float(value)
    if losses:
        extra["final_nu_loss"] = float(losses[-1][0])
        extra["final_zeta_loss"] = float(losses[-1][1])
        extra["final_lam_loss"] = float(losses[-1][2])
    diagnostics = _prefixed_backend_diagnostics(
        method=method,
        backend="google_dice_rl",
        gamma=common["gamma"],
        raw=raw,
        weights=weights,
        extra=extra,
        prediction_max=dice_cfg.prediction_max,
    )
    return MinimaxWeightModel(
        backend_model={"estimator": estimator, "nu_network": nu_network, "zeta_network": zeta_network},
        method=method,
        gamma=common["gamma"],
        state_dim=S.shape[1],
        action_dim=A.shape[1],
        diagnostics=diagnostics,
        config=cfg,
        _state_action_predictor=predict_zeta,
        prediction_max=dice_cfg.prediction_max,
        normalize_predictions=bool(dice_cfg.normalize_predictions),
    )


def _maybe_limit_scope_rl_torch_threads(enabled: bool) -> bool:
    if not enabled:
        return False
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return False
    try:
        torch.set_num_threads(1)
    except Exception:
        return False
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    except Exception:
        pass
    return True


def _fit_scope_rl_minimax_weight(
    common: dict[str, Any],
    cfg: MinimaxWeightConfig,
    *,
    method: str,
    episode_ids: Optional[Array],
    timesteps: Optional[Array],
    step_per_trajectory: int | None,
    behavior_action_pscore: Optional[Array],
) -> MinimaxWeightModel:
    scope_cfg = cfg.scope_rl
    preflight = preflight_scope_rl(scope_cfg.scope_rl_repo_path, method=method)
    if not preflight.available:
        raise ModuleNotFoundError(preflight.reason)
    _ensure_scope_rl_importable(None if scope_cfg.scope_rl_repo_path is None else Path(scope_cfg.scope_rl_repo_path))
    torch_threads_limited = _maybe_limit_scope_rl_torch_threads(bool(scope_cfg.limit_torch_threads))
    from scope_rl.ope.weight_value_learning import (  # noqa: PLC0415
        ContinuousMinimaxStateActionWeightLearning,
        ContinuousMinimaxStateWeightLearning,
    )
    from scope_rl.ope.weight_value_learning.function import (  # noqa: PLC0415
        ContinuousStateActionWeightFunction,
        StateWeightFunction,
    )

    S = common["S"]
    A = common["A"]
    A_pi = common["A_pi"]
    scope_common, state_center, state_scale, action_center, action_scale = _scope_rl_preprocess_inputs(
        common,
        standardize=bool(scope_cfg.standardize_inputs),
    )
    scope_bandwidth = _scope_rl_bandwidth(
        scope_common["S"],
        scope_common["A"],
        configured=float(scope_cfg.bandwidth),
        selection=str(scope_cfg.bandwidth_selection),
        num_pairs=int(scope_cfg.bandwidth_num_pairs),
        seed=int(scope_cfg.seed),
    )
    S_fit = scope_common["S"]
    A_fit = scope_common["A"]
    A_pi_fit = scope_common["A_pi"]
    iid_transition_fit = method == "scope_rl_minimax_state_action" and step_per_trajectory == 1
    fit_rows = (
        {
            "states": S_fit,
            "actions": A_fit,
            "target_actions": A_pi_fit,
            "row_index": np.arange(S_fit.shape[0], dtype=np.int64),
            "step_per_trajectory": 1,
            "inferred_single_trajectory": False,
        }
        if iid_transition_fit
        else _prepare_scope_rl_trajectory_rows(
            states=S_fit,
            actions=A_fit,
            target_actions=A_pi_fit,
            episode_ids=episode_ids,
            timesteps=timesteps,
            step_per_trajectory=step_per_trajectory,
        )
    )
    start = time.perf_counter()
    if method == "scope_rl_minimax_state_action":
        weight_function = ContinuousStateActionWeightFunction(
            action_dim=A.shape[1],
            state_dim=S.shape[1],
            hidden_dim=int(scope_cfg.hidden_dim),
        )
        learner = ContinuousMinimaxStateActionWeightLearning(
            w_function=weight_function,
            gamma=float(common["gamma"]),
            bandwidth=float(scope_bandwidth),
            batch_size=int(scope_cfg.batch_size),
            lr=float(scope_cfg.learning_rate),
            device=str(scope_cfg.device),
        )
        if iid_transition_fit:
            objective_final, normalization_final = _fit_scope_rl_iid_state_action(
                learner,
                common=scope_common,
                n_steps=int(scope_cfg.n_steps),
                batch_size=int(scope_cfg.batch_size),
                learning_rate=float(scope_cfg.learning_rate),
                regularization_weight=float(scope_cfg.regularization_weight),
                seed=int(scope_cfg.seed),
                device=str(scope_cfg.device),
            )
        else:
            learner.fit(
                step_per_trajectory=int(fit_rows["step_per_trajectory"]),
                state=fit_rows["states"],
                action=fit_rows["actions"],
                evaluation_policy_action=fit_rows["target_actions"],
                n_steps=int(scope_cfg.n_steps),
                n_steps_per_epoch=int(scope_cfg.n_steps_per_epoch),
                regularization_weight=float(scope_cfg.regularization_weight),
                random_state=int(scope_cfg.seed),
            )
            objective_final = normalization_final = float("nan")

        def predict_scope_state_action(states_query: Array, actions_query: Array) -> Array:
            states_query = _apply_scope_standardization(states_query, state_center, state_scale)
            actions_query = _apply_scope_standardization(actions_query, action_center, action_scale)
            if hasattr(learner, "predict_weight"):
                return learner.predict_weight(states_query, actions_query)
            return learner.predict(states_query, actions_query)

        raw = np.asarray(predict_scope_state_action(S, A), dtype=np.float64).reshape(-1)
        state_predictor = None
        state_action_predictor = predict_scope_state_action
    else:
        if behavior_action_pscore is None:
            raise ValueError("behavior_action_pscore is required for method='scope_rl_minimax_state'.")
        pscore = _as_pscore_2d(behavior_action_pscore, n_rows=S.shape[0], action_dim=A.shape[1])
        fit_pscore = pscore[fit_rows["row_index"]]
        weight_function = StateWeightFunction(
            state_dim=S.shape[1],
            hidden_dim=int(scope_cfg.hidden_dim),
        )
        learner = ContinuousMinimaxStateWeightLearning(
            w_function=weight_function,
            gamma=float(common["gamma"]),
            bandwidth=float(scope_bandwidth),
            batch_size=int(scope_cfg.batch_size),
            lr=float(scope_cfg.learning_rate),
            device=str(scope_cfg.device),
        )
        learner.fit(
            step_per_trajectory=int(fit_rows["step_per_trajectory"]),
            state=fit_rows["states"],
            action=fit_rows["actions"],
            pscore=fit_pscore,
            evaluation_policy_action=fit_rows["target_actions"],
            n_steps=int(scope_cfg.n_steps),
            n_steps_per_epoch=int(scope_cfg.n_steps_per_epoch),
            regularization_weight=float(scope_cfg.regularization_weight),
            random_state=int(scope_cfg.seed),
        )

        def predict_scope_state(states_query: Array) -> Array:
            states_query = _apply_scope_standardization(states_query, state_center, state_scale)
            if hasattr(learner, "predict_state_marginal_importance_weight"):
                return learner.predict_state_marginal_importance_weight(states_query)
            if hasattr(learner, "predict_weight"):
                return learner.predict_weight(states_query)
            return learner.predict(states_query)

        def predict_scope_state_action(states_query: Array, actions_query: Array) -> Array:
            if (
                np.asarray(states_query).shape == S.shape
                and np.asarray(actions_query).shape == A.shape
                and np.allclose(states_query, S)
                and np.allclose(actions_query, A)
                and hasattr(learner, "predict_state_action_marginal_importance_weight")
            ):
                return learner.predict_state_action_marginal_importance_weight(
                    state=states_query,
                    action=actions_query,
                    pscore=pscore,
                    evaluation_policy_action=A_pi,
                )
            return predict_scope_state(states_query)

        raw = np.asarray(predict_scope_state_action(S, A), dtype=np.float64).reshape(-1)
        state_predictor = predict_scope_state
        state_action_predictor = predict_scope_state_action

    weights = _postprocess_minimax_weights(
        raw,
        prediction_max=scope_cfg.prediction_max,
        normalize=bool(scope_cfg.normalize_predictions),
    )
    extra = {
        "n_steps": float(scope_cfg.n_steps),
        "n_steps_per_epoch": float(scope_cfg.n_steps_per_epoch),
        "batch_size": float(scope_cfg.batch_size),
        "learning_rate": float(scope_cfg.learning_rate),
        "hidden_dim": float(scope_cfg.hidden_dim),
        "bandwidth": float(scope_bandwidth),
        "bandwidth_configured": float(scope_cfg.bandwidth),
        "bandwidth_selection": str(scope_cfg.bandwidth_selection),
        "bandwidth_num_pairs": float(scope_cfg.bandwidth_num_pairs),
        "standardize_inputs": bool(scope_cfg.standardize_inputs),
        "regularization_weight": float(scope_cfg.regularization_weight),
        "device": str(scope_cfg.device),
        "limit_torch_threads": bool(scope_cfg.limit_torch_threads),
        "torch_threads_limited": bool(torch_threads_limited),
        "runtime_sec": float(time.perf_counter() - start),
        "step_per_trajectory": float(fit_rows["step_per_trajectory"]),
        "trajectory_rows_used": float(fit_rows["states"].shape[0]),
        "trajectory_rows_dropped": float(S.shape[0] - fit_rows["states"].shape[0]),
        "trajectory_inferred_single": bool(fit_rows["inferred_single_trajectory"]),
        "iid_transition_fit": bool(iid_transition_fit),
        "evaluation_policy_actions_used": bool(iid_transition_fit),
        "normalization_penalty_sign_corrected": bool(iid_transition_fit),
        "objective_final": float(objective_final),
        "normalization_loss_final": float(normalization_final),
    }
    diagnostics = _prefixed_backend_diagnostics(
        method=method,
        backend="scope_rl",
        gamma=common["gamma"],
        raw=raw,
        weights=weights,
        extra=extra,
        prediction_max=scope_cfg.prediction_max,
    )
    return MinimaxWeightModel(
        backend_model=learner,
        method=method,
        gamma=common["gamma"],
        state_dim=S.shape[1],
        action_dim=A.shape[1],
        diagnostics=diagnostics,
        config=cfg,
        _state_action_predictor=state_action_predictor,
        _state_predictor=state_predictor,
        prediction_max=scope_cfg.prediction_max,
        normalize_predictions=bool(scope_cfg.normalize_predictions),
    )


def _scope_rl_preprocess_inputs(
    common: dict[str, Any],
    *,
    standardize: bool,
) -> tuple[dict[str, Any], Array, Array, Array, Array]:
    state_center = np.zeros(common["S"].shape[1], dtype=np.float32)
    state_scale = np.ones(common["S"].shape[1], dtype=np.float32)
    action_center = np.zeros(common["A"].shape[1], dtype=np.float32)
    action_scale = np.ones(common["A"].shape[1], dtype=np.float32)
    if standardize:
        state_center = np.mean(common["S"], axis=0, dtype=np.float64).astype(np.float32)
        state_scale = np.std(common["S"], axis=0, dtype=np.float64).astype(np.float32)
        action_center = np.mean(common["A"], axis=0, dtype=np.float64).astype(np.float32)
        action_scale = np.std(common["A"], axis=0, dtype=np.float64).astype(np.float32)
        state_scale = np.where(state_scale > 1e-6, state_scale, 1.0).astype(np.float32)
        action_scale = np.where(action_scale > 1e-6, action_scale, 1.0).astype(np.float32)

    transformed = dict(common)
    for key in ("S", "S_next", "S_initial"):
        transformed[key] = _apply_scope_standardization(common[key], state_center, state_scale)
    for key in ("A", "A_pi", "A_pi_next", "A_initial"):
        transformed[key] = _apply_scope_standardization(common[key], action_center, action_scale)
    return transformed, state_center, state_scale, action_center, action_scale


def _apply_scope_standardization(values: Array, center: Array, scale: Array) -> Array:
    out = (np.asarray(values, dtype=np.float32) - center) / scale
    return np.asarray(out, dtype=np.float32)


def _scope_rl_bandwidth(
    states: Array,
    actions: Array,
    *,
    configured: float,
    selection: str,
    num_pairs: int,
    seed: int,
) -> float:
    if selection == "fixed":
        return float(configured)
    features = np.concatenate(
        [np.asarray(states, dtype=np.float64), np.asarray(actions, dtype=np.float64)],
        axis=1,
    )
    rng = np.random.default_rng(int(seed) + 91_337)
    left = rng.integers(0, features.shape[0], size=int(num_pairs))
    right = rng.integers(0, features.shape[0], size=int(num_pairs))
    distance = np.linalg.norm(features[left] - features[right], axis=1)
    positive = distance[np.isfinite(distance) & (distance > 1e-8)]
    if positive.size == 0:
        return float(configured)
    return max(float(np.median(positive)), 1e-6)


def _fit_scope_rl_iid_state_action(
    learner: Any,
    *,
    common: dict[str, Any],
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    regularization_weight: float,
    seed: int,
    device: str,
) -> tuple[float, float]:
    """Fit SCOPE-RL's MWL objective on independent transition rows."""

    import torch  # noqa: PLC0415
    from torch.nn.utils import clip_grad_norm_  # noqa: PLC0415

    torch.manual_seed(int(seed))
    state = torch.as_tensor(common["S"], dtype=torch.float32, device=device)
    action = torch.as_tensor(common["A"], dtype=torch.float32, device=device)
    next_state = torch.as_tensor(common["S_next"], dtype=torch.float32, device=device)
    next_action = torch.as_tensor(common["A_pi_next"], dtype=torch.float32, device=device)
    initial_state = torch.as_tensor(common["S_initial"], dtype=torch.float32, device=device)
    initial_action = torch.as_tensor(common["A_initial"], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(learner.w_function.parameters(), lr=float(learning_rate))
    actual_batch = min(int(batch_size), state.shape[0])
    initial_batch = min(int(batch_size), initial_state.shape[0])
    objective_loss = normalization_loss = torch.tensor(float("nan"), device=device)
    for _ in range(int(n_steps)):
        row_idx = torch.randint(state.shape[0], size=(actual_batch,), device=device)
        initial_idx = torch.randint(initial_state.shape[0], size=(initial_batch,), device=device)
        pair_batch = min(actual_batch, initial_batch)
        row_idx = row_idx[:pair_batch]
        initial_idx = initial_idx[:pair_batch]
        objective_loss, normalization_loss = learner._objective_function(
            initial_state=initial_state[initial_idx],
            initial_action=initial_action[initial_idx],
            state=state[row_idx],
            action=action[row_idx],
            next_state=next_state[row_idx],
            next_action=next_action[row_idx],
        )
        loss = objective_loss + float(regularization_weight) * normalization_loss
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(learner.w_function.parameters(), max_norm=0.01)
        optimizer.step()
    return float(objective_loss.detach().cpu()), float(normalization_loss.detach().cpu())


def _prepare_common_inputs(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    target_actions: Array,
    target_next_actions: Array | None,
    gamma: float,
    initial_states: Array | None,
    initial_actions: Array | None,
    terminals: Optional[Array],
    rewards: Optional[Array],
    sample_weight: Optional[Array],
    initial_weights: Optional[Array],
    require_target_next_actions: bool,
    require_initial_actions: bool,
) -> dict[str, Any]:
    gamma_value = float(gamma)
    if not (0.0 <= gamma_value < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    S = _as_2d(states, "states").astype(np.float32, copy=False)
    A = _as_2d(actions, "actions").astype(np.float32, copy=False)
    S_next = _as_2d(next_states, "next_states").astype(np.float32, copy=False)
    A_pi = _as_2d(target_actions, "target_actions").astype(np.float32, copy=False)
    A_pi_next = None if target_next_actions is None else _as_2d(target_next_actions, "target_next_actions").astype(np.float32, copy=False)
    S_initial = None if initial_states is None else _as_2d(initial_states, "initial_states").astype(np.float32, copy=False)
    A_initial = None if initial_actions is None else _as_2d(initial_actions, "initial_actions").astype(np.float32, copy=False)
    _validate_aligned_inputs(S=S, A=A, S_next=S_next, A_pi=A_pi)
    _validate_next_target_actions(A=A, S=S, A_pi_next=A_pi_next)
    _validate_initial_state_inputs(S=S, S_initial=S_initial, initial_weights=initial_weights)
    _validate_initial_action_inputs(A=A, S_initial=S_initial, A_initial=A_initial)
    rewards_1d = _optional_rewards(rewards, S.shape[0])
    if require_target_next_actions and A_pi_next is None:
        raise ValueError("target_next_actions is required for Google minimax-weight methods.")
    if require_initial_actions and (S_initial is None or A_initial is None):
        raise ValueError("initial_states and initial_actions are required for Google minimax-weight methods.")
    return {
        "S": S,
        "A": A,
        "S_next": S_next,
        "A_pi": A_pi,
        "A_pi_next": A_pi_next if A_pi_next is not None else A_pi,
        "S_initial": S_initial if S_initial is not None else S[:1],
        "A_initial": A_initial if A_initial is not None else A_pi[:1],
        "gamma": gamma_value,
        "terminals": _optional_terminals(terminals, S.shape[0]),
        "rewards": rewards_1d,
        "sample_weight": _optional_weights(sample_weight, S.shape[0], "sample_weight"),
        "initial_weight": _optional_weights(initial_weights, S_initial.shape[0] if S_initial is not None else 1, "initial_weights"),
    }


def _resolve_minimax_method(method: str) -> str:
    if method == "auto":
        return DEFAULT_MINIMAX_WEIGHT_METHOD
    allowed = {
        "google_policy_eval_dualdice",
        "google_dice_rl_dualdice_exact",
        "google_dice_rl_recommended",
        "scope_rl_minimax_state_action",
        "scope_rl_minimax_state",
    }
    if method not in allowed:
        raise ValueError(
            "method must be one of 'auto', 'google_policy_eval_dualdice', "
            "'google_dice_rl_dualdice_exact', 'google_dice_rl_recommended', "
            "'scope_rl_minimax_state_action', or 'scope_rl_minimax_state'."
        )
    return method


def _postprocess_minimax_weights(
    raw: Array,
    *,
    prediction_max: Optional[float],
    normalize: bool,
) -> Array:
    return postprocess_weights(raw, cap=prediction_max, normalize=normalize)


def _prefixed_backend_diagnostics(
    *,
    method: str,
    backend: str,
    gamma: float,
    raw: Array,
    weights: Array,
    extra: dict[str, Any],
    prediction_max: Optional[float],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "backend": "minimax_weight",
        "minimax_method": method,
        "minimax_backend": backend,
        "gamma": float(gamma),
    }
    raw_summary = weight_summary(raw)
    weight_diag = weight_summary(weights, cap=prediction_max)
    diagnostics.update({f"raw_weight_{key}": value for key, value in raw_summary.items()})
    diagnostics.update({f"weight_{key}": value for key, value in weight_diag.items()})
    diagnostics.update({f"{backend}_{key}": value for key, value in extra.items()})
    return diagnostics


def _optional_weights(weights: Optional[Array], n_rows: int, name: str) -> Array:
    if weights is None:
        return np.ones(int(n_rows), dtype=np.float64)
    arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if arr.shape[0] != int(n_rows):
        raise ValueError(f"{name} must have {n_rows} rows.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    if float(np.sum(arr)) <= 0.0:
        raise ValueError(f"{name} must contain positive total weight.")
    return arr


def _optional_terminals(terminals: Optional[Array], n_rows: int) -> Array:
    if terminals is None:
        return np.zeros(int(n_rows), dtype=np.float64)
    arr = np.asarray(terminals, dtype=np.float64).reshape(-1)
    if arr.shape[0] != int(n_rows):
        raise ValueError(f"terminals must have {n_rows} rows.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("terminals must contain only finite values.")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("terminals must be in [0, 1].")
    return arr


def _optional_rewards(rewards: Optional[Array], n_rows: int) -> Array:
    if rewards is None:
        return np.zeros(int(n_rows), dtype=np.float64)
    arr = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if arr.shape[0] != int(n_rows):
        raise ValueError(f"rewards must have {n_rows} rows.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("rewards must contain only finite values.")
    return arr


def _probabilities(weights: Array) -> Array | None:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(np.sum(w))
    if total <= 0.0 or not np.isfinite(total):
        return None
    return w / total


def _flags_match(flags: dict[str, float | bool], target: dict[str, float | bool]) -> bool:
    return all(flags.get(key) == value for key, value in target.items())


def _dice_rl_package_dir(repo_path: Path) -> Path | None:
    if (repo_path / "__init__.py").exists() and (repo_path / "estimators" / "neural_dice.py").exists():
        return repo_path
    nested = repo_path / "dice_rl"
    if (nested / "__init__.py").exists() and (nested / "estimators" / "neural_dice.py").exists():
        return nested
    return None


def _ensure_dice_rl_importable(repo_path: str | Path) -> None:
    package_dir = _dice_rl_package_dir(Path(repo_path))
    if package_dir is None:
        raise FileNotFoundError(f"Missing DICE-RL package under {repo_path}.")
    if package_dir.name != "dice_rl":
        raise ImportError(f"DICE-RL package directory must be named 'dice_rl', got {package_dir}.")
    import_root = str(package_dir.parent)
    if import_root not in sys.path:
        sys.path.insert(0, import_root)
    existing = sys.modules.get("dice_rl")
    existing_file = Path(str(getattr(existing, "__file__", ""))).resolve() if existing is not None else None
    if existing_file is not None and existing_file != (package_dir / "__init__.py").resolve():
        for name in tuple(sys.modules):
            if name == "dice_rl" or name.startswith("dice_rl."):
                del sys.modules[name]
    importlib.import_module("dice_rl")


def _ensure_scope_rl_importable(repo_path: Path | None) -> None:
    if repo_path is not None:
        if (repo_path / "__init__.py").exists() and repo_path.name == "scope_rl":
            import_root = repo_path.parent
        elif (repo_path / "scope_rl" / "__init__.py").exists():
            import_root = repo_path
        else:
            raise FileNotFoundError(f"Missing SCOPE-RL package under {repo_path}.")
        root = str(import_root)
        if root not in sys.path:
            sys.path.insert(0, root)
    importlib.import_module("scope_rl")


def _direct_action_neural_dice_class(neural_dice_cls):  # noqa: ANN001
    """Build NeuralDice that consumes target actions stored on each row.

    NeuralDice normally calls a policy for initial and successor observations.
    The benchmark has already sampled those actions and stores them in the
    corresponding ``EnvStep``. Reusing them is the same one-action Monte Carlo
    objective without a nearest-neighbor policy lookup.
    """

    class DirectActionNeuralDice(neural_dice_cls):
        def _get_average_value(self, network, env_step, policy):  # noqa: ANN001
            del policy
            return self._get_value(network, env_step)

    DirectActionNeuralDice.__name__ = "DirectActionNeuralDice"
    return DirectActionNeuralDice


def _dice_rl_adam_optimizer(tf, **kwargs):  # noqa: ANN001
    legacy = getattr(getattr(tf.keras.optimizers, "legacy", None), "Adam", None)
    return legacy(**kwargs) if legacy is not None else tf.keras.optimizers.Adam(**kwargs)


def _as_float32_2d(values: Array) -> Array:
    arr = np.asarray(values, dtype=np.float32)
    return arr.reshape(arr.shape[0], -1)


def _dice_rl_experience_batch(
    dice_dataset,
    tf,
    *,
    states,
    actions,
    next_states,
    next_actions,
    rewards,
    masks,
):
    batch_size = tf.shape(states)[0]
    step_type = tf.zeros((batch_size, 2), dtype=tf.int32)
    step_num = tf.zeros((batch_size, 2), dtype=tf.int32)
    return dice_dataset.EnvStep(
        step_type=step_type,
        step_num=step_num,
        observation=tf.stack([states, next_states], axis=1),
        action=tf.stack([actions, next_actions], axis=1),
        reward=tf.stack([rewards, tf.zeros_like(rewards)], axis=1),
        discount=tf.stack([tf.ones_like(masks), masks], axis=1),
        policy_info={},
        env_info={},
        other_info={},
    )


def _dice_rl_initial_batch(dice_dataset, tf, *, states, actions):
    batch_size = tf.shape(states)[0]
    return dice_dataset.EnvStep(
        step_type=tf.zeros((batch_size,), dtype=tf.int32),
        step_num=tf.zeros((batch_size,), dtype=tf.int32),
        observation=states,
        action=actions,
        reward=tf.zeros((batch_size,), dtype=tf.float32),
        discount=tf.ones((batch_size,), dtype=tf.float32),
        policy_info={},
        env_info={},
        other_info={},
    )


def _prepare_scope_rl_trajectory_rows(
    *,
    states: Array,
    actions: Array,
    target_actions: Array,
    episode_ids: Optional[Array],
    timesteps: Optional[Array],
    step_per_trajectory: int | None,
) -> dict[str, Any]:
    n = states.shape[0]
    if step_per_trajectory is not None:
        step = int(step_per_trajectory)
        if step < 3:
            raise ValueError("step_per_trajectory must be at least 3 for SCOPE-RL minimax weights.")
        usable = (n // step) * step
        if usable < step:
            raise ValueError("Not enough rows for one complete SCOPE-RL trajectory.")
        row_index = np.arange(usable, dtype=np.int64)
        return {
            "states": states[row_index],
            "actions": actions[row_index],
            "target_actions": target_actions[row_index],
            "row_index": row_index,
            "step_per_trajectory": step,
            "inferred_single_trajectory": False,
        }
    if episode_ids is not None and timesteps is not None:
        row_index, step = _complete_trajectory_indices_from_ids(episode_ids, timesteps)
        if row_index.size == 0:
            raise ValueError("No complete SCOPE-RL trajectory blocks were found in episode_ids/timesteps.")
        return {
            "states": states[row_index],
            "actions": actions[row_index],
            "target_actions": target_actions[row_index],
            "row_index": row_index,
            "step_per_trajectory": step,
            "inferred_single_trajectory": False,
        }
    if n < 3:
        raise ValueError("SCOPE-RL minimax weights need at least three ordered rows.")
    row_index = np.arange(n, dtype=np.int64)
    return {
        "states": states,
        "actions": actions,
        "target_actions": target_actions,
        "row_index": row_index,
        "step_per_trajectory": n,
        "inferred_single_trajectory": True,
    }


def _complete_trajectory_indices_from_ids(episode_ids: Array, timesteps: Array) -> tuple[Array, int]:
    episodes = np.asarray(episode_ids).reshape(-1)
    times = np.asarray(timesteps).reshape(-1)
    if episodes.shape[0] != times.shape[0]:
        raise ValueError("episode_ids and timesteps must have the same number of rows.")
    groups: list[np.ndarray] = []
    for episode in np.unique(episodes):
        idx = np.flatnonzero(episodes == episode)
        order = idx[np.argsort(times[idx], kind="stable")]
        ordered_times = times[order]
        if order.size < 3:
            continue
        start = 0
        while start < order.size:
            end = start + 1
            while end < order.size and ordered_times[end] == ordered_times[end - 1] + 1:
                end += 1
            if end - start >= 3:
                groups.append(order[start:end])
            start = end
    if not groups:
        return np.array([], dtype=np.int64), 0
    step = int(min(group.size for group in groups))
    step = max(3, step)
    chunks = [group[:step] for group in groups if group.size >= step]
    if not chunks:
        return np.array([], dtype=np.int64), 0
    return np.concatenate(chunks).astype(np.int64, copy=False), step


def _as_pscore_2d(values: Array, *, n_rows: int, action_dim: int) -> Array:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("behavior_action_pscore must be 1D or 2D.")
    if arr.shape[0] != int(n_rows):
        raise ValueError(f"behavior_action_pscore must have {n_rows} rows.")
    if arr.shape[1] != int(action_dim):
        raise ValueError("behavior_action_pscore must have the same feature dimension as actions.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("behavior_action_pscore must contain only finite values.")
    if np.any(arr <= 0.0):
        raise ValueError("behavior_action_pscore must be positive.")
    return arr


__all__ = [
    "DEFAULT_MINIMAX_WEIGHT_METHOD",
    "GOOGLE_DICE_RL_DUALDICE_EXACT_FLAGS",
    "GOOGLE_DICE_RL_RECOMMENDED_FLAGS",
    "GoogleDICERLConfig",
    "MinimaxWeightConfig",
    "MinimaxWeightMethod",
    "MinimaxWeightModel",
    "MinimaxWeightPreflight",
    "ScopeRLMinimaxWeightConfig",
    "fit_minimax_weight",
    "preflight_google_dice_rl",
    "preflight_minimax_weight",
    "preflight_scope_rl",
]
