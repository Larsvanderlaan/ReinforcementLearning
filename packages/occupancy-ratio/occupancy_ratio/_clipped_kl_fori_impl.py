"""Recursively clipped KL-projected fitted occupancy-ratio iteration.

This module implements the two empirical ERM steps in Algorithm 2 of the
coverage-clipped FORE manuscript: a weighted retention gate followed by an
unnormalized generalized-KL projection.  Unlike standard KL-FORI, neither the
successor weights nor the fitted ratio are normalized.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from occupancy_ratio import _clipped_kl_fori_objectives as _objective_spec
from occupancy_ratio import _clipped_kl_fori_diagnostics as _diagnostic_spec
from occupancy_ratio import _clipped_kl_fori_backend_common as _backend_common
from occupancy_ratio import _clipped_kl_fori_linear as _linear_backend
from occupancy_ratio import _clipped_kl_fori_neural as _neural_backend
from occupancy_ratio._clipped_kl_fori_types import (
    FitDiagnostics,
    FitResult,
    InnerOptimizerResult,
)
from occupancy_ratio._fori_data import (
    as_2d as _as_2d,
    fit_standardizer as _fit_standardizer,
    optional_split_groups as _optional_split_groups,
    prepare_initial_rows as _prepare_initial_rows,
    prepare_successor_actions as _prepare_successor_actions,
    resolve_continuation as _resolve_continuation,
    standardized_from_state_action as _standardized_from_state_action,
    train_valid_indices as _train_valid_indices,
    train_valid_indices_from_sources as _train_valid_indices_from_sources,
)


Array = np.ndarray
TargetActionSampler = Callable[[Array, np.random.Generator], Array]
MODEL_SCHEMA = "clipped-kl-fori-model-v1"


class ClippedKLFORIConvergenceError(RuntimeError):
    """Raised when a caller requires convergence but the final refit misses it."""

    def __init__(self, model: "ClippedKLFORIModel") -> None:
        super().__init__(
            "clipped KL-FORI did not meet the deployable outer convergence criterion"
        )
        self.model = model
        self.diagnostics = model.diagnostics


@dataclass
class ClippedKLFORIConfig:
    """Configuration for recursively clipped KL-FORI.

    The lower envelope is deliberately independent of ``tau_upper``.  A
    symmetric lower envelope such as ``1 / tau_upper`` can dominate the
    retained-mass diagnostic when the clipped target has little mass.
    """

    tau_lower: float = 1e-4
    tau_upper: float = 10.0
    backend: str = "linear"
    num_iterations: int = 300
    min_iterations: int = 5
    outer_tolerance: Optional[float] = 1e-5
    outer_patience: int = 3
    gate_optimizer_steps: int = 200
    ratio_optimizer_steps: int = 300
    gate_learning_rate: Optional[float] = None
    ratio_learning_rate: Optional[float] = None
    gate_l2_penalty: float = 1e-4
    ratio_l2_penalty: float = 1e-4
    validation_fraction: float = 0.2
    seed: int = 123
    feature_include_quadratic: bool = False
    neural_hidden_dims: Sequence[int] = (64, 64)
    neural_weight_decay: float = 1e-4
    neural_grad_clip_norm: Optional[float] = 10.0
    normalize_eps: float = 1e-12
    device: str = "cpu"
    show_progress: bool = False
    retain_fit_payload: bool = False
    require_convergence: bool = False
    inner_relative_tolerance: float = 1e-8
    inner_gradient_tolerance: float = 1e-6
    inner_patience: int = 5
    neural_deterministic: bool = True
    initialization_perturbation_scale: float = 0.0

    def __post_init__(self) -> None:
        self.backend = _normalize_clipped_backend(self.backend)
        self.neural_hidden_dims = tuple(int(width) for width in self.neural_hidden_dims)
        if not np.isfinite(self.tau_lower) or not np.isfinite(self.tau_upper):
            raise ValueError("tau_lower and tau_upper must be finite.")
        if not (0.0 < self.tau_lower <= 1.0 <= self.tau_upper):
            raise ValueError(
                "clipping levels must satisfy 0 < tau_lower <= 1 <= tau_upper."
            )
        if self.num_iterations < 0:
            raise ValueError("num_iterations must be nonnegative.")
        if self.min_iterations < 0:
            raise ValueError("min_iterations must be nonnegative.")
        if self.min_iterations > self.num_iterations:
            raise ValueError("min_iterations cannot exceed num_iterations.")
        if self.outer_tolerance is not None and self.outer_tolerance <= 0.0:
            raise ValueError("outer_tolerance must be positive when supplied.")
        if self.outer_patience <= 0:
            raise ValueError("outer_patience must be positive.")
        if self.gate_optimizer_steps <= 0 or self.ratio_optimizer_steps <= 0:
            raise ValueError(
                "gate_optimizer_steps and ratio_optimizer_steps must be positive."
            )
        for name, value in (
            ("gate_learning_rate", self.gate_learning_rate),
            ("ratio_learning_rate", self.ratio_learning_rate),
        ):
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive when supplied.")
        if self.gate_l2_penalty < 0.0 or self.ratio_l2_penalty < 0.0:
            raise ValueError("gate and ratio L2 penalties must be nonnegative.")
        if not (0.0 <= self.validation_fraction < 1.0):
            raise ValueError("validation_fraction must be in [0, 1).")
        if any(width <= 0 for width in self.neural_hidden_dims):
            raise ValueError("neural_hidden_dims must contain positive widths.")
        if self.neural_weight_decay < 0.0:
            raise ValueError("neural_weight_decay must be nonnegative.")
        if self.neural_grad_clip_norm is not None and self.neural_grad_clip_norm <= 0.0:
            raise ValueError("neural_grad_clip_norm must be positive when supplied.")
        if self.normalize_eps <= 0.0:
            raise ValueError("normalize_eps must be positive.")
        if self.inner_relative_tolerance <= 0.0:
            raise ValueError("inner_relative_tolerance must be positive.")
        if self.inner_gradient_tolerance <= 0.0:
            raise ValueError("inner_gradient_tolerance must be positive.")
        if self.inner_patience <= 0:
            raise ValueError("inner_patience must be positive.")
        if self.initialization_perturbation_scale < 0.0:
            raise ValueError("initialization_perturbation_scale must be nonnegative.")

    @property
    def resolved_gate_learning_rate(self) -> float:
        """Return the backend-specific gate learning rate."""
        if self.gate_learning_rate is not None:
            return float(self.gate_learning_rate)
        return 5e-2 if self.backend == "linear" else 1e-3

    @property
    def resolved_ratio_learning_rate(self) -> float:
        """Return the backend-specific ratio learning rate."""
        if self.ratio_learning_rate is not None:
            return float(self.ratio_learning_rate)
        return 5e-2 if self.backend == "linear" else 1e-3


@dataclass
class ClippedKLFORIModel:
    """Fitted recursively clipped occupancy-ratio model."""

    ratio_coef: Array
    gate_coef: Array
    mean: Array
    scale: Array
    gamma: float
    state_dim: int
    action_dim: int
    tau_lower: float
    tau_upper: float
    feature_include_quadratic: bool
    backend: str
    neural_hidden_dims: tuple[int, ...]
    ratio_neural_state_dict: dict[str, Array]
    gate_neural_state_dict: dict[str, Array]
    history: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    fit_payload: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Serialize the fitted model as a versioned, pickle-free NPZ file."""
        arrays: dict[str, Array] = {
            "ratio_coef": np.asarray(self.ratio_coef),
            "gate_coef": np.asarray(self.gate_coef),
            "mean": np.asarray(self.mean),
            "scale": np.asarray(self.scale),
        }
        ratio_keys: dict[str, str] = {}
        gate_keys: dict[str, str] = {}
        for index, (name, value) in enumerate(
            sorted(self.ratio_neural_state_dict.items())
        ):
            key = f"ratio_state_{index}"
            ratio_keys[name] = key
            arrays[key] = np.asarray(value)
        for index, (name, value) in enumerate(
            sorted(self.gate_neural_state_dict.items())
        ):
            key = f"gate_state_{index}"
            gate_keys[name] = key
            arrays[key] = np.asarray(value)
        metadata = {
            "schema": MODEL_SCHEMA,
            "gamma": self.gamma,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "tau_lower": self.tau_lower,
            "tau_upper": self.tau_upper,
            "feature_include_quadratic": self.feature_include_quadratic,
            "backend": self.backend,
            "neural_hidden_dims": list(self.neural_hidden_dims),
            "ratio_state_keys": ratio_keys,
            "gate_state_keys": gate_keys,
            "history": _json_safe(self.history),
            "diagnostics": _json_safe(self.diagnostics),
            # Training arrays are deliberately never persisted in model files.
            "fit_payload": _compact_fit_payload(self.fit_payload),
        }
        arrays["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, allow_nan=False)
        )
        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "ClippedKLFORIModel":
        """Load and validate a versioned clipped KL-FORI NPZ model."""
        try:
            archive = np.load(Path(path), allow_pickle=False)
        except Exception as exc:
            raise ValueError("invalid clipped KL-FORI model artifact") from exc
        with archive:
            if "metadata_json" not in archive.files:
                raise ValueError("model artifact is missing metadata_json")
            try:
                metadata = json.loads(str(archive["metadata_json"].item()))
            except Exception as exc:
                raise ValueError("model artifact metadata is not valid JSON") from exc
            if metadata.get("schema") != MODEL_SCHEMA:
                raise ValueError(
                    f"unsupported clipped KL-FORI model schema {metadata.get('schema')!r}"
                )
            metadata_fields = {
                "gamma",
                "state_dim",
                "action_dim",
                "tau_lower",
                "tau_upper",
                "feature_include_quadratic",
                "backend",
                "neural_hidden_dims",
            }
            missing_metadata = sorted(metadata_fields - metadata.keys())
            if missing_metadata:
                raise ValueError(
                    "model artifact is missing metadata fields "
                    + ", ".join(missing_metadata)
                )
            if metadata["backend"] not in {"linear", "neural"}:
                raise ValueError("model artifact has incompatible backend metadata")
            if not isinstance(metadata["neural_hidden_dims"], list):
                raise ValueError("model artifact neural_hidden_dims must be a list")
            if not isinstance(metadata.get("history", []), list):
                raise ValueError("model artifact history must be a list")
            if not isinstance(metadata.get("diagnostics", {}), dict):
                raise ValueError("model artifact diagnostics must be an object")
            if not isinstance(metadata.get("fit_payload", {}), dict):
                raise ValueError("model artifact fit_payload must be an object")
            required = ("ratio_coef", "gate_coef", "mean", "scale")
            if any(name not in archive.files for name in required):
                raise ValueError("model artifact is missing required parameter arrays")
            ratio_state = _load_state_dict(
                archive, metadata.get("ratio_state_keys", {})
            )
            gate_state = _load_state_dict(archive, metadata.get("gate_state_keys", {}))
            model = cls(
                ratio_coef=np.asarray(archive["ratio_coef"], dtype=np.float64),
                gate_coef=np.asarray(archive["gate_coef"], dtype=np.float64),
                mean=np.asarray(archive["mean"], dtype=np.float64),
                scale=np.asarray(archive["scale"], dtype=np.float64),
                gamma=float(metadata["gamma"]),
                state_dim=int(metadata["state_dim"]),
                action_dim=int(metadata["action_dim"]),
                tau_lower=float(metadata["tau_lower"]),
                tau_upper=float(metadata["tau_upper"]),
                feature_include_quadratic=bool(metadata["feature_include_quadratic"]),
                backend=str(metadata["backend"]),
                neural_hidden_dims=tuple(
                    int(x) for x in metadata["neural_hidden_dims"]
                ),
                ratio_neural_state_dict=ratio_state,
                gate_neural_state_dict=gate_state,
                history=list(metadata.get("history", [])),
                diagnostics=dict(metadata.get("diagnostics", {})),
                fit_payload=dict(metadata.get("fit_payload", {})),
            )
        try:
            _validate_loaded_model(model)
        except FloatingPointError as exc:
            raise ValueError(f"invalid clipped KL-FORI model artifact: {exc}") from exc
        return model

    def _raw_ratio_score(self, states: Array, actions: Array) -> Array:
        if self.backend == "linear":
            features = _linear_features_from_state_action(
                states,
                actions,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                mean=self.mean,
                scale=self.scale,
                include_quadratic=self.feature_include_quadratic,
            )
            return features @ np.asarray(self.ratio_coef, dtype=np.float64)
        z = _standardized_from_state_action(
            states,
            actions,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            mean=self.mean,
            scale=self.scale,
        )
        return _predict_neural_scores(
            z,
            state_dict=self.ratio_neural_state_dict,
            hidden_dims=self.neural_hidden_dims,
            device="cpu",
        )

    def _raw_gate_score(self, states: Array, actions: Array) -> Array:
        if self.backend == "linear":
            features = _linear_features_from_state_action(
                states,
                actions,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                mean=self.mean,
                scale=self.scale,
                include_quadratic=self.feature_include_quadratic,
            )
            return features @ np.asarray(self.gate_coef, dtype=np.float64)
        z = _standardized_from_state_action(
            states,
            actions,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            mean=self.mean,
            scale=self.scale,
        )
        return _predict_neural_scores(
            z,
            state_dict=self.gate_neural_state_dict,
            hidden_dims=self.neural_hidden_dims,
            device="cpu",
        )

    def predict_state_action_log_ratio(self, states: Array, actions: Array) -> Array:
        """Predict an unnormalized log ratio inside the configured envelope."""
        return _bounded_log_ratio(
            self._raw_ratio_score(states, actions),
            tau_lower=self.tau_lower,
            tau_upper=self.tau_upper,
        )[0]

    def predict_state_action_ratio(self, states: Array, actions: Array) -> Array:
        """Predict an unnormalized recursively clipped occupancy ratio."""
        return np.exp(self.predict_state_action_log_ratio(states, actions))

    def predict_gate_score(self, states: Array, actions: Array) -> Array:
        """Predict the weighted-classification gate score from the final update."""
        return self._raw_gate_score(states, actions).astype(np.float64, copy=False)

    def predict_gate_probability(self, states: Array, actions: Array) -> Array:
        """Predict the logistic surrogate class probability.

        This is not the trajectory continuation probability in the stopped-flow
        representation.
        """
        return _sigmoid(self.predict_gate_score(states, actions))

    def predict_gate_indicator(self, states: Array, actions: Array) -> Array:
        """Predict the hard retention indicator used by the final update."""
        return (self.predict_gate_score(states, actions) >= 0.0).astype(np.float64)

    def estimate_retained_mass(
        self,
        states: Array,
        actions: Array,
        *,
        sample_weight: Optional[Array] = None,
    ) -> float:
        """Estimate retained occupancy mass on reference-distribution rows."""
        ratio = self.predict_state_action_ratio(states, actions)
        weight = _evaluation_weights(sample_weight, ratio.shape[0])
        return float(weight @ ratio)

    def estimate_stopped_value(
        self,
        states: Array,
        actions: Array,
        rewards: Array,
        *,
        sample_weight: Optional[Array] = None,
    ) -> float:
        """Estimate the stopped discounted value on reference rows."""
        ratio = self.predict_state_action_ratio(states, actions)
        reward = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if reward.shape[0] != ratio.shape[0]:
            raise ValueError("rewards must match states rows.")
        if not np.all(np.isfinite(reward)):
            raise ValueError("rewards must contain only finite values.")
        weight = _evaluation_weights(sample_weight, ratio.shape[0])
        return float(weight @ (ratio * reward))


def fit_clipped_kl_fori(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    target_actions: Array | None = None,
    gamma: float,
    initial_states: Optional[Array] = None,
    initial_actions: Optional[Array] = None,
    initial_weights: Optional[Array] = None,
    target_next_actions: Optional[Array] = None,
    target_policy: Any = None,
    target_action_sampler: Optional[TargetActionSampler] = None,
    terminals: Optional[Array] = None,
    timeouts: Optional[Array] = None,
    groups: Optional[Array] = None,
    initial_groups: Optional[Array] = None,
    handle_timeouts: str = "nonterminal",
    absorbing_state: bool = False,
    config: Optional[ClippedKLFORIConfig] = None,
    **unused: Any,
) -> ClippedKLFORIModel:
    """Fit recursively clipped KL-FORI from offline transition moments.

    The estimator follows the same array contract as :func:`fit_kl_fori`, but
    its empirical generalized-KL projection is intentionally unnormalized.
    """
    del target_actions
    if unused:
        ignored = ", ".join(sorted(unused))
        raise TypeError(f"Unsupported clipped KL-FORI arguments: {ignored}")
    if config is not None and not isinstance(config, ClippedKLFORIConfig):
        raise TypeError("config must be a ClippedKLFORIConfig when supplied.")
    cfg = ClippedKLFORIConfig() if config is None else config
    fit_start = time.perf_counter()
    if not (0.0 <= float(gamma) < 1.0):
        raise ValueError("gamma must be in [0, 1).")

    S = _as_2d(states, "states")
    A = _as_2d(actions, "actions")
    S_next = _as_2d(next_states, "next_states")
    if S.shape[0] == 0:
        raise ValueError("states must be nonempty.")
    if S.shape[0] != A.shape[0] or S.shape[0] != S_next.shape[0]:
        raise ValueError(
            "states, actions, and next_states must have the same number of rows."
        )
    if S.shape[1] != S_next.shape[1]:
        raise ValueError("states and next_states must have the same feature dimension.")

    split_groups = _optional_split_groups(groups, n_rows=S.shape[0], name="groups")
    S0, A0, init_weight, initial_row_index, initial_action_source = (
        _prepare_initial_rows(
            initial_states=initial_states,
            initial_actions=initial_actions,
            initial_weights=initial_weights,
            action_dim=A.shape[1],
            target_policy=target_policy,
            target_action_sampler=target_action_sampler,
            seed=int(cfg.seed) + 17,
        )
    )
    initial_split_groups = _optional_split_groups(
        initial_groups,
        n_rows=int(np.max(initial_row_index, initial=-1)) + 1,
        name="initial_groups",
    )
    if initial_split_groups is not None:
        initial_split_groups = initial_split_groups[
            np.asarray(initial_row_index, dtype=np.int64)
        ]
    A_next, successor_row_index, successor_action_source = _prepare_successor_actions(
        next_states=S_next,
        actions=A,
        target_next_actions=target_next_actions,
        target_policy=target_policy,
        target_action_sampler=target_action_sampler,
        seed=int(cfg.seed),
    )
    continuation = _resolve_continuation(
        n_rows=S.shape[0],
        terminals=terminals,
        timeouts=timeouts,
        handle_timeouts=handle_timeouts,
        absorbing_state=absorbing_state,
    )
    continuation_plus = continuation[np.asarray(successor_row_index, dtype=np.int64)]

    X_ref = np.concatenate([S, A], axis=1)
    X_init = np.concatenate([S0, A0], axis=1)
    X_plus = np.concatenate([S_next[successor_row_index], A_next], axis=1)
    init_probs = _normalize_probability(init_weight, "initial_weights")

    if split_groups is None:
        train_idx, valid_idx = _train_valid_indices(
            S.shape[0], cfg.validation_fraction, cfg.seed
        )
    else:
        train_idx, valid_idx = _train_valid_indices_from_sources(
            split_groups, cfg.validation_fraction, cfg.seed
        )
    initial_source = (
        initial_row_index if initial_split_groups is None else initial_split_groups
    )
    init_train_idx, init_valid_idx = _train_valid_indices_from_sources(
        initial_source, cfg.validation_fraction, cfg.seed + 71
    )

    has_validation = bool(
        valid_idx.size and init_valid_idx.size and cfg.validation_fraction > 0.0
    )
    selection: Optional[FitResult] = None
    if has_validation:
        selection_mean, selection_scale = _fit_standardizer(X_ref[train_idx])
        selection = _fit_backend(
            cfg=cfg,
            gamma=float(gamma),
            X_ref=X_ref,
            X_init=X_init,
            X_plus=X_plus,
            mean=selection_mean,
            scale=selection_scale,
            init_probs=init_probs,
            successor_row_index=np.asarray(successor_row_index, dtype=np.int64),
            continuation_plus=np.asarray(continuation_plus, dtype=np.float64),
            objective_ref_idx=train_idx,
            objective_init_idx=init_train_idx,
            valid_ref_idx=valid_idx,
            valid_init_idx=init_valid_idx,
        )
        selected_iterations = int(selection["iterations_completed"])
        refit_cfg = replace(
            cfg,
            num_iterations=selected_iterations,
            min_iterations=min(int(cfg.min_iterations), selected_iterations),
            outer_tolerance=None,
            validation_fraction=0.0,
        )
        full_idx = np.arange(X_ref.shape[0], dtype=np.int64)
        full_init_idx = np.arange(X_init.shape[0], dtype=np.int64)
        mean, scale = _fit_standardizer(X_ref)
        fit = _fit_backend(
            cfg=refit_cfg,
            gamma=float(gamma),
            X_ref=X_ref,
            X_init=X_init,
            X_plus=X_plus,
            mean=mean,
            scale=scale,
            init_probs=init_probs,
            successor_row_index=np.asarray(successor_row_index, dtype=np.int64),
            continuation_plus=np.asarray(continuation_plus, dtype=np.float64),
            objective_ref_idx=full_idx,
            objective_init_idx=full_init_idx,
            valid_ref_idx=np.array([], dtype=np.int64),
            valid_init_idx=np.array([], dtype=np.int64),
        )
        selection_history = [
            {**row, "stage": "selection"} for row in selection["history"]
        ]
        refit_history = [{**row, "stage": "refit"} for row in fit["history"]]
        fit.history = selection_history + refit_history
    else:
        mean, scale = _fit_standardizer(X_ref)
        full_idx = np.arange(X_ref.shape[0], dtype=np.int64)
        full_init_idx = np.arange(X_init.shape[0], dtype=np.int64)
        fit = _fit_backend(
            cfg=cfg,
            gamma=float(gamma),
            X_ref=X_ref,
            X_init=X_init,
            X_plus=X_plus,
            mean=mean,
            scale=scale,
            init_probs=init_probs,
            successor_row_index=np.asarray(successor_row_index, dtype=np.int64),
            continuation_plus=np.asarray(continuation_plus, dtype=np.float64),
            objective_ref_idx=full_idx,
            objective_init_idx=full_init_idx,
            valid_ref_idx=np.array([], dtype=np.int64),
            valid_init_idx=np.array([], dtype=np.int64),
        )

    diagnostics = FitDiagnostics(
        _final_diagnostics(
            cfg=cfg,
            gamma=float(gamma),
            fit=fit,
            continuation_plus=continuation_plus,
            selection=selection,
        )
    ).to_dict()
    diagnostics.update(
        {
            "runtime_sec": float(time.perf_counter() - fit_start),
            "initial_action_source": initial_action_source,
            "successor_action_source": successor_action_source,
            "reference_rows": int(X_ref.shape[0]),
            "initial_rows": int(X_init.shape[0]),
            "successor_rows": int(X_plus.shape[0]),
            "reference_split_grouped": bool(split_groups is not None),
            "initial_split_grouped": bool(initial_split_groups is not None),
            "refit_after_validation": bool(has_validation),
        }
    )
    compact_payload = {
        "algorithm": "clipped_kl_fori",
        "backend": cfg.backend,
        "reference_rows": int(X_ref.shape[0]),
        "initial_rows": int(X_init.shape[0]),
        "successor_rows": int(X_plus.shape[0]),
    }
    if selection is not None:
        compact_payload["selection_iterations_completed"] = int(
            selection["iterations_completed"]
        )
    fit_payload = compact_payload
    if cfg.retain_fit_payload:
        fit_payload = {
            **compact_payload,
            "pred_state_action_ratio_beh": fit["weights_ref"],
            "pred_gate_indicator_beh": fit["gate_ref"],
            "X_ref": X_ref,
            "X_init": X_init,
            "X_plus": X_plus,
            "successor_row_index": np.asarray(successor_row_index, dtype=np.int64),
            "continuation_plus": np.asarray(continuation_plus, dtype=np.float64),
        }
        if selection is not None:
            fit_payload["selection_history"] = selection["history"]

    model = ClippedKLFORIModel(
        ratio_coef=fit["ratio_coef"],
        gate_coef=fit["gate_coef"],
        mean=mean,
        scale=scale,
        gamma=float(gamma),
        state_dim=S.shape[1],
        action_dim=A.shape[1],
        tau_lower=float(cfg.tau_lower),
        tau_upper=float(cfg.tau_upper),
        feature_include_quadratic=bool(cfg.feature_include_quadratic),
        backend=cfg.backend,
        neural_hidden_dims=tuple(int(width) for width in cfg.neural_hidden_dims),
        ratio_neural_state_dict=fit["ratio_neural_state_dict"],
        gate_neural_state_dict=fit["gate_neural_state_dict"],
        history=fit["history"],
        diagnostics=diagnostics,
        fit_payload=fit_payload,
    )
    if cfg.require_convergence and not bool(model.diagnostics.get("converged", False)):
        raise ClippedKLFORIConvergenceError(model)
    return model


def fit_clipped_kl_fori_neural(**kwargs: Any) -> ClippedKLFORIModel:
    """Fit recursively clipped KL-FORI with the lazy Torch backend."""
    config = kwargs.get("config")
    if config is None:
        kwargs["config"] = ClippedKLFORIConfig(backend="neural")
    elif not isinstance(config, ClippedKLFORIConfig):
        raise TypeError("config must be a ClippedKLFORIConfig when supplied.")
    else:
        kwargs["config"] = replace(config, backend="neural")
    return fit_clipped_kl_fori(**kwargs)


def _fit_backend(**kwargs: Any) -> FitResult:
    cfg = kwargs["cfg"]
    if cfg.backend == "linear":
        return _linear_backend.fit_linear_backend(
            **kwargs,
            gate_optimizer=_fit_linear_gate_adam,
            ratio_optimizer=_fit_linear_ratio_adam,
        )
    if cfg.backend == "neural":
        return _neural_backend.fit_neural_backend(**kwargs)
    raise ValueError(f"Unknown clipped KL-FORI backend {cfg.backend!r}.")


def _fit_linear_gate_adam(**kwargs: Any) -> InnerOptimizerResult:
    """Private test hook around the isolated linear gate optimizer."""
    return _linear_backend.fit_linear_gate_adam(
        **kwargs, objective_and_grad=_linear_gate_objective_and_grad
    )


def _fit_linear_ratio_adam(**kwargs: Any) -> InnerOptimizerResult:
    """Private test hook around the isolated linear ratio optimizer."""
    return _linear_backend.fit_linear_ratio_adam(
        **kwargs, objective_and_grad=_linear_ratio_objective_and_grad
    )


def _evaluation_weights(sample_weight: Optional[Array], n_rows: int) -> Array:
    if sample_weight is None:
        return np.full(int(n_rows), 1.0 / float(n_rows), dtype=np.float64)
    weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if weight.shape[0] != int(n_rows):
        raise ValueError("sample_weight must match states rows.")
    if np.any(weight < 0.0) or not np.all(np.isfinite(weight)):
        raise ValueError("sample_weight must be finite and nonnegative.")
    return _normalize_probability(weight, "sample_weight")


def _normalize_clipped_backend(backend: str) -> str:
    value = str(backend).strip().lower()
    aliases = {
        "linear": "linear",
        "lin": "linear",
        "neural": "neural",
        "torch": "neural",
        "mlp": "neural",
    }
    if value not in aliases:
        raise ValueError("backend must be 'linear' or 'neural'.")
    return aliases[value]


def _load_state_dict(archive: Any, mapping: Any) -> dict[str, Array]:
    if not isinstance(mapping, dict):
        raise ValueError("model state_dict mapping must be an object")
    state: dict[str, Array] = {}
    for name, key in mapping.items():
        if (
            not isinstance(name, str)
            or not isinstance(key, str)
            or key not in archive.files
        ):
            raise ValueError("model artifact has an invalid state_dict mapping")
        value = np.asarray(archive[key])
        _require_finite(value, f"state_dict parameter {name}")
        state[name] = value.astype(np.float32, copy=False)
    return state


def _validate_loaded_model(model: ClippedKLFORIModel) -> None:
    if not (0.0 <= model.gamma < 1.0):
        raise ValueError("model gamma must lie in [0, 1)")
    if not (0.0 < model.tau_lower <= 1.0 <= model.tau_upper):
        raise ValueError("model clipping envelope is invalid")
    if model.state_dim <= 0 or model.action_dim <= 0:
        raise ValueError("model state/action dimensions must be positive")
    expected_input = model.state_dim + model.action_dim
    if model.mean.shape != (expected_input,) or model.scale.shape != (expected_input,):
        raise ValueError("model standardizer arrays have incompatible shapes")
    _require_finite(model.mean, "model mean")
    _require_finite(model.scale, "model scale")
    if np.any(model.scale <= 0.0):
        raise ValueError("model scale must be positive")
    backend = _normalize_clipped_backend(model.backend)
    if backend == "linear":
        expected_coef = 1 + expected_input * (
            2 if model.feature_include_quadratic else 1
        )
        if model.ratio_coef.shape != (expected_coef,) or model.gate_coef.shape != (
            expected_coef,
        ):
            raise ValueError("linear model coefficient arrays have incompatible shapes")
        _require_finite(model.ratio_coef, "model ratio coefficients")
        _require_finite(model.gate_coef, "model gate coefficients")
        if model.ratio_neural_state_dict or model.gate_neural_state_dict:
            raise ValueError(
                "linear model artifact unexpectedly contains neural parameters"
            )
    else:
        if model.ratio_coef.size or model.gate_coef.size:
            raise ValueError(
                "neural model artifact unexpectedly contains linear coefficients"
            )
        if not model.ratio_neural_state_dict or not model.gate_neural_state_dict:
            raise ValueError("neural model artifact is missing state_dict parameters")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _compact_fit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _json_safe(value)
        for key, value in payload.items()
        if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_))
    }


# The fitted backends use the independent executable objective specification.
# Keeping these private aliases preserves the established test/debug surface.
_bounded_log_ratio = _objective_spec.bounded_log_ratio
_gate_loss_from_scores = _objective_spec.gate_loss_from_scores
_linear_gate_objective_and_grad = _objective_spec.linear_gate_objective_and_grad
_linear_ratio_objective_and_grad = _objective_spec.linear_ratio_objective_and_grad
_projection_loss_from_log_ratios = _objective_spec.projection_loss_from_log_ratios
_sigmoid = _objective_spec.sigmoid
_uniform_raw_score = _objective_spec.uniform_raw_score
_normalize_probability = _backend_common.normalize_probability
_require_finite = _backend_common.require_finite
_linear_features_from_state_action = _linear_backend.linear_features_from_state_action
_predict_neural_scores = _neural_backend.predict_neural_scores
_ess_fraction = _diagnostic_spec.ess_fraction
_final_diagnostics = _diagnostic_spec.final_diagnostics
_history_row = _diagnostic_spec.history_row


__all__ = [
    "ClippedKLFORIConvergenceError",
    "ClippedKLFORIConfig",
    "ClippedKLFORIModel",
    "fit_clipped_kl_fori",
    "fit_clipped_kl_fori_neural",
]
