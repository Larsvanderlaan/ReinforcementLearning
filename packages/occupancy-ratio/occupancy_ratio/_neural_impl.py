from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
try:  # pragma: no cover - exercised when PyTorch is installed.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - environment dependent.
    torch = None
    nn = None

from occupancy_ratio.fit_importance_and_transition_ratios import _postprocess_ratio_predictions
from occupancy_ratio.fit_occupancy_ratio import (
    _as_2d,
    _damped_update,
    _ess,
    _fold_initial_states,
    _fold_initial_weights,
    _fit_prediction_max,
    _make_occupancy_sample_weights,
    _make_stabilized_fixed_point_target,
    _occupancy_loss_value,
    _lookup_known_training_predictions,
    _project_nonnegative_normalized,
    _prepare_target_action_samples,
    _replace_known_training_predictions,
    _resolve_continuation,
    _resolve_huber_delta,
    _resolve_known_action_ratio_inputs,
    _ratio_query_cap_fraction,
    _safe_divide,
    _source_state_ratio_summary,
    _validate_base_transition_inputs,
    _validate_initial_action_inputs,
    _validate_initial_state_inputs,
    _validate_occupancy_stabilization_config,
    _validate_ratio_prediction_config,
    _validate_target_action_rows,
    _resolve_initial_ratio_mode,
    _resolve_one_step_ratio_mode,
)


Array = np.ndarray

__all__ = [
    "NeuralActionRatioConfig",
    "NeuralSourceStateRatioConfig",
    "NeuralTransitionRatioConfig",
    "NeuralOccupancyRegressionConfig",
    "NeuralDiscountedOccupancyRatioModel",
    "fit_action_ratio_neural",
    "fit_source_state_ratio_neural",
    "fit_transition_ratio_neural",
    "fit_discounted_occupancy_ratio_neural",
    "tune_discounted_occupancy_ratio_neural_cv",
]


@dataclass
class NeuralActionRatioConfig:
    """Tuning for neural action importance-ratio nuisance fits."""

    hidden_dims: Sequence[int] = (64, 64)
    activation: str = "silu"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_steps: int = 400
    validation_fraction: float = 0.2
    patience: int = 20
    min_improvement: float = 1e-5
    grad_clip_norm: Optional[float] = 5.0
    normalization_penalty: float = 10.0
    initial_ratio: float = 1.0
    prediction_max: Optional[float] = 50.0
    prediction_power: float = 1.0
    normalize_predictions: bool = False
    moment_calibration: str = "none"
    crossfit_folds: int = 1
    crossfit_seed: Optional[int] = None
    density_ratio_loss: str = "lsif"
    logistic_logit_clip: Optional[float] = 20.0
    device: str = "cpu"
    seed: int = 123

    def __post_init__(self) -> None:
        _validate_neural_common(
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_steps=self.max_steps,
            validation_fraction=self.validation_fraction,
            patience=self.patience,
            min_improvement=self.min_improvement,
            grad_clip_norm=self.grad_clip_norm,
        )
        if self.normalization_penalty < 0.0:
            raise ValueError("normalization_penalty must be nonnegative.")
        if self.initial_ratio <= 0.0:
            raise ValueError("initial_ratio must be positive.")
        _validate_ratio_prediction_config(
            prediction_max=self.prediction_max,
            prediction_power=self.prediction_power,
            moment_calibration=self.moment_calibration,
            crossfit_folds=self.crossfit_folds,
        )
        _validate_density_ratio_loss(self.density_ratio_loss, self.logistic_logit_clip)

    @classmethod
    def stable_defaults(cls, **overrides: Any) -> "NeuralActionRatioConfig":
        params = dict(
            normalization_penalty=10.0,
            prediction_max=50.0,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=400,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def balanced_defaults(cls, **overrides: Any) -> "NeuralActionRatioConfig":
        params = dict(
            normalization_penalty=1.0,
            prediction_max=200.0,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=800,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def dualdice_comparable_defaults(cls, **overrides: Any) -> "NeuralActionRatioConfig":
        params = dict(
            normalization_penalty=0.0,
            prediction_max=None,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=1200,
        )
        params.update(overrides)
        return cls(**params)


@dataclass
class NeuralSourceStateRatioConfig:
    """Tuning for neural initial/source density-ratio nuisance fits."""

    hidden_dims: Sequence[int] = (64, 64)
    activation: str = "silu"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_steps: int = 400
    validation_fraction: float = 0.2
    patience: int = 20
    min_improvement: float = 1e-5
    grad_clip_norm: Optional[float] = 5.0
    normalization_penalty: float = 10.0
    initial_ratio: float = 1.0
    prediction_max: Optional[float] = 50.0
    prediction_power: float = 1.0
    normalize_predictions: bool = False
    moment_calibration: str = "none"
    crossfit_folds: int = 1
    crossfit_seed: Optional[int] = None
    density_ratio_loss: str = "lsif"
    logistic_logit_clip: Optional[float] = 20.0
    device: str = "cpu"
    seed: int = 123

    def __post_init__(self) -> None:
        _validate_neural_common(
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_steps=self.max_steps,
            validation_fraction=self.validation_fraction,
            patience=self.patience,
            min_improvement=self.min_improvement,
            grad_clip_norm=self.grad_clip_norm,
        )
        if self.normalization_penalty < 0.0:
            raise ValueError("normalization_penalty must be nonnegative.")
        if self.initial_ratio <= 0.0:
            raise ValueError("initial_ratio must be positive.")
        _validate_ratio_prediction_config(
            prediction_max=self.prediction_max,
            prediction_power=self.prediction_power,
            moment_calibration=self.moment_calibration,
            crossfit_folds=self.crossfit_folds,
        )
        _validate_density_ratio_loss(self.density_ratio_loss, self.logistic_logit_clip)

    @classmethod
    def stable_defaults(cls, **overrides: Any) -> "NeuralSourceStateRatioConfig":
        params = dict(
            normalization_penalty=10.0,
            prediction_max=50.0,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=400,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def balanced_defaults(cls, **overrides: Any) -> "NeuralSourceStateRatioConfig":
        params = dict(
            normalization_penalty=1.0,
            prediction_max=200.0,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=800,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def dualdice_comparable_defaults(cls, **overrides: Any) -> "NeuralSourceStateRatioConfig":
        params = dict(
            normalization_penalty=0.0,
            prediction_max=None,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=1200,
        )
        params.update(overrides)
        return cls(**params)


@dataclass
class NeuralTransitionRatioConfig:
    """Tuning for neural transition density-ratio nuisance fits."""

    hidden_dims: Sequence[int] = (64, 64)
    activation: str = "silu"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_steps: int = 600
    permutation_samples: int = 4
    validation_fraction: float = 0.2
    patience: int = 20
    min_improvement: float = 1e-5
    grad_clip_norm: Optional[float] = 5.0
    normalization_penalty: float = 10.0
    initial_ratio: float = 1.0
    prediction_max: Optional[float] = 50.0
    prediction_power: float = 1.0
    normalize_predictions: bool = False
    moment_calibration: str = "none"
    crossfit_folds: int = 1
    crossfit_seed: Optional[int] = None
    density_ratio_loss: str = "lsif"
    logistic_logit_clip: Optional[float] = 20.0
    device: str = "cpu"
    seed: int = 123

    def __post_init__(self) -> None:
        _validate_neural_common(
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_steps=self.max_steps,
            validation_fraction=self.validation_fraction,
            patience=self.patience,
            min_improvement=self.min_improvement,
            grad_clip_norm=self.grad_clip_norm,
        )
        if self.permutation_samples <= 0:
            raise ValueError("permutation_samples must be positive.")
        if self.normalization_penalty < 0.0:
            raise ValueError("normalization_penalty must be nonnegative.")
        if self.initial_ratio <= 0.0:
            raise ValueError("initial_ratio must be positive.")
        _validate_ratio_prediction_config(
            prediction_max=self.prediction_max,
            prediction_power=self.prediction_power,
            moment_calibration=self.moment_calibration,
            crossfit_folds=self.crossfit_folds,
        )
        _validate_density_ratio_loss(self.density_ratio_loss, self.logistic_logit_clip)

    @classmethod
    def stable_defaults(cls, **overrides: Any) -> "NeuralTransitionRatioConfig":
        params = dict(
            normalization_penalty=10.0,
            prediction_max=50.0,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=600,
            permutation_samples=4,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def balanced_defaults(cls, **overrides: Any) -> "NeuralTransitionRatioConfig":
        params = dict(
            normalization_penalty=1.0,
            prediction_max=200.0,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=800,
            permutation_samples=16,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def dualdice_comparable_defaults(cls, **overrides: Any) -> "NeuralTransitionRatioConfig":
        params = dict(
            normalization_penalty=0.0,
            prediction_max=None,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=1200,
            permutation_samples=32,
        )
        params.update(overrides)
        return cls(**params)


@dataclass
class NeuralOccupancyRegressionConfig:
    """Tuning for neural fixed-point occupancy regression."""

    hidden_dims: Sequence[int] = (64, 64)
    activation: str = "silu"
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    batch_size: int = 512
    num_iterations: int = 30
    gradient_steps_per_iteration: int = 4
    mcmc_samples: int = 24
    initial_ratio: float = 1.0
    fixed_point_initialization: str = "one_step"
    loss: str = "huber"
    huber_delta: Optional[float] = None
    huber_delta_scale: float = 1.345
    huber_delta_quantile_power: Optional[float] = 0.25
    huber_delta_min_quantile: float = 0.80
    fixed_point_damping: float = 0.5
    normalize_occupancy: bool = True
    occupancy_ratio_max: Optional[float] = 50.0
    occupancy_projection_eps: float = 1e-12
    clip_pseudo_outcomes: bool = True
    pseudo_outcome_max: Optional[float] = None
    pseudo_outcome_upper_quantile: float = 0.995
    pseudo_outcome_min: float = 0.0
    normalize_transition_cache: bool = False
    transition_cache_norm_eps: float = 1e-12
    occupancy_sample_weight_mode: str = "uniform"
    occupancy_sample_weight_max: Optional[float] = 20.0
    occupancy_ess_penalty: float = 0.0
    occupancy_ess_mean_penalty: float = 10.0
    fixed_point_tol: Optional[float] = None
    fixed_point_patience: int = 3
    min_outer_iterations: int = 3
    target_min: Optional[float] = 0.0
    target_max: Optional[float] = None
    early_stopping: bool = True
    validation_fraction: float = 0.2
    min_improvement: float = 1e-6
    patience: int = 10
    validation_warmup_iterations: int = 1
    refresh_on_plateau: bool = True
    refresh_after_plateaus: int = 1
    eval_mcmc_multiplier: int = 5
    eval_seed_offset: int = 777_777
    direct_adjoint_steps: Optional[int] = 128
    direct_adjoint_learning_rate: Optional[float] = 1e-3
    direct_adjoint_weight_decay: Optional[float] = 0.0
    direct_one_step_density_ratio_loss: str = "lsif"
    direct_one_step_prediction_max: Optional[float] = 10.0
    direct_one_step_logistic_logit_clip: Optional[float] = 10.0
    direct_one_step_moment_calibration: str = "scalar"
    direct_one_step_max_steps: Optional[int] = None
    direct_one_step_hidden_dims: Optional[Sequence[int]] = None
    grad_clip_norm: Optional[float] = 5.0
    device: str = "cpu"
    seed: int = 123
    show_progress: bool = False

    def __post_init__(self) -> None:
        _validate_neural_common(
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_steps=self.num_iterations,
            validation_fraction=self.validation_fraction,
            patience=self.patience,
            min_improvement=self.min_improvement,
            grad_clip_norm=self.grad_clip_norm,
        )
        if self.gradient_steps_per_iteration <= 0:
            raise ValueError("gradient_steps_per_iteration must be positive.")
        if self.mcmc_samples <= 0:
            raise ValueError("mcmc_samples must be positive.")
        if self.initial_ratio <= 0.0:
            raise ValueError("initial_ratio must be positive.")
        _normalize_fixed_point_initialization(self.fixed_point_initialization)
        self._normalized_loss()
        if self.huber_delta is not None and self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive when supplied.")
        if self.huber_delta_scale <= 0.0:
            raise ValueError("huber_delta_scale must be positive.")
        if self.huber_delta_quantile_power is not None and self.huber_delta_quantile_power <= 0.0:
            raise ValueError("huber_delta_quantile_power must be positive when supplied.")
        if not (0.0 < self.huber_delta_min_quantile < 1.0):
            raise ValueError("huber_delta_min_quantile must be in (0, 1).")
        if self.validation_warmup_iterations < 0:
            raise ValueError("validation_warmup_iterations must be nonnegative.")
        if self.occupancy_ess_penalty < 0.0:
            raise ValueError("occupancy_ess_penalty must be nonnegative.")
        if self.occupancy_ess_mean_penalty < 0.0:
            raise ValueError("occupancy_ess_mean_penalty must be nonnegative.")
        if self.direct_adjoint_steps is not None and self.direct_adjoint_steps <= 0:
            raise ValueError("direct_adjoint_steps must be positive when supplied.")
        if self.direct_adjoint_learning_rate is not None and self.direct_adjoint_learning_rate <= 0.0:
            raise ValueError("direct_adjoint_learning_rate must be positive when supplied.")
        if self.direct_adjoint_weight_decay is not None and self.direct_adjoint_weight_decay < 0.0:
            raise ValueError("direct_adjoint_weight_decay must be nonnegative when supplied.")
        _validate_density_ratio_loss(self.direct_one_step_density_ratio_loss, self.direct_one_step_logistic_logit_clip)
        if self.direct_one_step_prediction_max is not None and self.direct_one_step_prediction_max <= 0.0:
            raise ValueError("direct_one_step_prediction_max must be positive when supplied.")
        if str(self.direct_one_step_moment_calibration) not in {"none", "scalar"}:
            raise ValueError("direct_one_step_moment_calibration must be 'none' or 'scalar'.")
        if self.direct_one_step_max_steps is not None and self.direct_one_step_max_steps <= 0:
            raise ValueError("direct_one_step_max_steps must be positive when supplied.")
        if self.direct_one_step_hidden_dims is not None and (
            not tuple(self.direct_one_step_hidden_dims)
            or any(int(width) <= 0 for width in self.direct_one_step_hidden_dims)
        ):
            raise ValueError("direct_one_step_hidden_dims must contain positive widths when supplied.")
        _validate_occupancy_stabilization_config(
            fixed_point_damping=self.fixed_point_damping,
            occupancy_ratio_max=self.occupancy_ratio_max,
            occupancy_projection_eps=self.occupancy_projection_eps,
            pseudo_outcome_max=self.pseudo_outcome_max,
            pseudo_outcome_upper_quantile=self.pseudo_outcome_upper_quantile,
            pseudo_outcome_min=self.pseudo_outcome_min,
            transition_cache_norm_eps=self.transition_cache_norm_eps,
            occupancy_sample_weight_mode=self.occupancy_sample_weight_mode,
            occupancy_sample_weight_max=self.occupancy_sample_weight_max,
            fixed_point_tol=self.fixed_point_tol,
            fixed_point_patience=self.fixed_point_patience,
            min_outer_iterations=self.min_outer_iterations,
        )

    def _normalized_loss(self) -> str:
        aliases = {
            "l2": "squared",
            "mse": "squared",
            "squared_error": "squared",
            "squared": "squared",
            "huber": "huber",
            "robust": "huber",
        }
        normalized = str(self.loss).strip().lower()
        if normalized not in aliases:
            raise ValueError("loss must be 'squared' or 'huber'.")
        return aliases[normalized]

    @classmethod
    def stable_defaults(cls, **overrides: Any) -> "NeuralOccupancyRegressionConfig":
        params = dict(
            loss="huber",
            fixed_point_damping=0.5,
            normalize_occupancy=True,
            occupancy_ratio_max=50.0,
            clip_pseudo_outcomes=True,
            pseudo_outcome_upper_quantile=0.995,
            occupancy_sample_weight_mode="uniform",
            direct_adjoint_steps=128,
            direct_one_step_prediction_max=10.0,
            direct_one_step_moment_calibration="scalar",
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def balanced_defaults(cls, **overrides: Any) -> "NeuralOccupancyRegressionConfig":
        params = dict(
            num_iterations=80,
            gradient_steps_per_iteration=8,
            mcmc_samples=64,
            loss="squared",
            fixed_point_damping=1.0,
            normalize_occupancy=True,
            occupancy_ratio_max=200.0,
            clip_pseudo_outcomes=False,
            occupancy_sample_weight_mode="sqrt_target",
            direct_adjoint_steps=128,
            direct_one_step_prediction_max=200.0,
            direct_one_step_moment_calibration="none",
            learning_rate=5e-4,
            weight_decay=1e-5,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def dualdice_comparable_defaults(cls, **overrides: Any) -> "NeuralOccupancyRegressionConfig":
        params = dict(
            num_iterations=120,
            gradient_steps_per_iteration=12,
            mcmc_samples=96,
            loss="squared",
            fixed_point_damping=1.0,
            normalize_occupancy=False,
            occupancy_ratio_max=None,
            clip_pseudo_outcomes=False,
            occupancy_sample_weight_mode="sqrt_target",
            occupancy_sample_weight_max=None,
            direct_adjoint_steps=256,
            direct_one_step_prediction_max=None,
            direct_one_step_moment_calibration="none",
            learning_rate=5e-4,
            weight_decay=0.0,
        )
        params.update(overrides)
        return cls(**params)


@dataclass
class NeuralDiscountedOccupancyRatioModel:
    """Fitted neural discounted occupancy ratio with LightGBM-compatible helpers."""

    occupancy_predictor: "_RatioPredictor"
    action_ratio_predictor: Optional["_RatioPredictor"]
    transition_ratio_predictor: "_RatioPredictor"
    gamma: float
    state_dim: int
    action_dim: int
    history: List[Dict[str, Any]]
    diagnostics: Dict[str, Any]
    fit_payload: Dict[str, Any]
    occupancy_normalize: bool = False
    occupancy_ratio_max: Optional[float] = None
    occupancy_projection_eps: float = 1e-12
    occupancy_prediction_scale: Optional[float] = None
    occupancy_training_features: Optional[Array] = None
    occupancy_training_predictions: Optional[Array] = None
    action_ratio_training_features: Optional[Array] = None
    action_ratio_training_predictions: Optional[Array] = None

    def save(self, path: str | Path) -> None:
        """Serialize the fitted neural model with pickle."""
        with Path(path).open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralDiscountedOccupancyRatioModel":
        with Path(path).open("rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, cls):
            raise TypeError(f"Serialized object is {type(model).__name__}, not {cls.__name__}.")
        return model

    def predict_state_action_ratio(self, states: Array, actions: Array, *, clip: bool = True) -> Array:
        features = self._state_action_features(states, actions)
        raw = self.occupancy_predictor.predict(features, postprocess=False)
        if not clip:
            return raw
        projected = _project_nonnegative_normalized(
            raw,
            max_value=self.occupancy_ratio_max,
            normalize=self.occupancy_normalize,
            eps=self.occupancy_projection_eps,
            normalization_scale=self.occupancy_prediction_scale,
        )
        return _replace_known_training_predictions(
            features,
            projected,
            known_features=self.occupancy_training_features,
            known_predictions=self.occupancy_training_predictions,
        )

    def predict_action_ratio(self, states: Array, actions: Array, *, clip: bool = True) -> Array:
        features = self._state_action_features(states, actions)
        if self.action_ratio_predictor is None:
            if self.action_ratio_training_features is None or self.action_ratio_training_predictions is None:
                raise ValueError("This model was fit with known action ratios and has no action-ratio predictor.")
            out = _lookup_known_training_predictions(
                features,
                known_features=self.action_ratio_training_features,
                known_predictions=self.action_ratio_training_predictions,
            )
            if out is None:
                raise ValueError("Known action-ratio model can only predict exact fitted state-action rows.")
            return out
        return self.action_ratio_predictor.predict(features, postprocess=clip)

    def predict_state_ratio(self, states: Array, actions: Array, *, clip: bool = True) -> Array:
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
    ) -> Dict[str, Array]:
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

    def _state_action_features(self, states: Array, actions: Array) -> Array:
        states = _as_2d(states, "states")
        actions = _as_2d(actions, "actions")
        if states.shape[0] != actions.shape[0]:
            raise ValueError("states and actions must have the same number of rows.")
        if states.shape[1] != self.state_dim:
            raise ValueError(f"states must have {self.state_dim} columns.")
        if actions.shape[1] != self.action_dim:
            raise ValueError(f"actions must have {self.action_dim} columns.")
        return np.concatenate([states, actions], axis=1)


def fit_discounted_occupancy_ratio_neural(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    target_actions: Array,
    gamma: float,
    initial_states: Optional[Array] = None,
    initial_actions: Optional[Array] = None,
    initial_weights: Optional[Array] = None,
    target_next_actions: Optional[Array] = None,
    terminals: Optional[Array] = None,
    timeouts: Optional[Array] = None,
    handle_timeouts: str = "nonterminal",
    absorbing_state: bool = False,
    action_ratio_values: Optional[Array] = None,
    behavior_log_prob: Optional[Array] = None,
    target_log_prob: Optional[Array] = None,
    known_action_ratio_clip_max: Optional[float] = None,
    known_action_ratio_normalize: bool = False,
    initial_ratio_mode: str = "auto",
    one_step_ratio_mode: str = "auto",
    occupancy: Optional[NeuralOccupancyRegressionConfig] = None,
    action_ratio: Optional[NeuralActionRatioConfig] = None,
    source_state_ratio: Optional[NeuralSourceStateRatioConfig] = None,
    transition_ratio: Optional[NeuralTransitionRatioConfig] = None,
    action_ratio_predictor: Optional["_RatioPredictor"] = None,
    transition_ratio_predictor: Optional["_RatioPredictor"] = None,
    _prefit_nuisance: Optional[Dict[str, Any]] = None,
) -> NeuralDiscountedOccupancyRatioModel:
    """Fit a discounted occupancy density-ratio model with neural gradient updates."""
    _require_torch()

    occupancy = NeuralOccupancyRegressionConfig() if occupancy is None else occupancy
    action_ratio = NeuralActionRatioConfig(seed=occupancy.seed) if action_ratio is None else action_ratio
    source_state_ratio = (
        NeuralSourceStateRatioConfig(seed=occupancy.seed)
        if source_state_ratio is None
        else source_state_ratio
    )
    transition_ratio = NeuralTransitionRatioConfig(seed=occupancy.seed) if transition_ratio is None else transition_ratio

    if not (0.0 <= float(gamma) < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    S = _as_2d(states, "states").astype(np.float32, copy=False)
    A = _as_2d(actions, "actions").astype(np.float32, copy=False)
    S_next = _as_2d(next_states, "next_states").astype(np.float32, copy=False)
    A_pi_info = _prepare_target_action_samples(S, A, target_actions, name="target_actions")
    A_pi = np.asarray(A_pi_info["actions"], dtype=np.float32)
    S_pi = np.asarray(A_pi_info["states"], dtype=np.float32)
    target_row_index = np.asarray(A_pi_info["row_index"], dtype=np.int64)
    num_target_action_samples = int(A_pi_info["num_samples"])
    S_initial = None if initial_states is None else _as_2d(initial_states, "initial_states").astype(np.float32, copy=False)
    A_initial = None if initial_actions is None else _as_2d(initial_actions, "initial_actions").astype(np.float32, copy=False)
    A_pi_next_info = (
        None
        if target_next_actions is None
        else _prepare_target_action_samples(S_next, A, target_next_actions, name="target_next_actions")
    )
    A_pi_next = None if A_pi_next_info is None else np.asarray(A_pi_next_info["actions"], dtype=np.float32)
    S_next_pi = None if A_pi_next_info is None else np.asarray(A_pi_next_info["states"], dtype=np.float32)
    next_target_row_index = None if A_pi_next_info is None else np.asarray(A_pi_next_info["row_index"], dtype=np.int64)
    _validate_base_transition_inputs(S=S, A=A, S_next=S_next)
    _validate_target_action_rows(A=A, A_pi=A_pi, name="target_actions")
    if A_pi_next is not None:
        _validate_target_action_rows(A=A, A_pi=A_pi_next, name="target_next_actions")
    _validate_initial_state_inputs(S=S, S_initial=S_initial, initial_weights=initial_weights)
    _validate_initial_action_inputs(A=A, S_initial=S_initial, A_initial=A_initial)
    if A_pi_next_info is not None and int(A_pi_next_info["num_samples"]) != num_target_action_samples:
        raise ValueError("target_next_actions must have the same number of target-action samples as target_actions.")
    resolved_initial_mode = _resolve_initial_ratio_mode(
        initial_ratio_mode,
        S_initial=S_initial,
        A_initial=A_initial,
    )
    resolved_one_step_mode = _resolve_one_step_ratio_mode(one_step_ratio_mode, A_pi_next=A_pi_next)

    X_sa_beh = np.concatenate([S, A], axis=1).astype(np.float32, copy=False)
    X_sa_pi = np.concatenate([S_pi, A_pi], axis=1).astype(np.float32, copy=False)
    X_sa_query = np.vstack([X_sa_pi, X_sa_beh]).astype(np.float32, copy=False)
    X_s_query = np.vstack([S_pi, S]).astype(np.float32, copy=False)
    X_sa_initial = None if S_initial is None or A_initial is None else np.concatenate([S_initial, A_initial], axis=1)
    X_sa_next_pi = None if A_pi_next is None or S_next_pi is None else np.concatenate([S_next_pi, A_pi_next], axis=1)
    n = S.shape[0]
    q = X_sa_query.shape[0]
    n_target = X_sa_pi.shape[0]
    continuation = _resolve_continuation(
        terminals=terminals,
        timeouts=timeouts,
        handle_timeouts=handle_timeouts,
        absorbing_state=absorbing_state,
        n_rows=n,
    )
    continuation_query = np.concatenate([continuation[target_row_index], continuation]).astype(np.float64, copy=False)
    known_iw_beh, known_iw_query = _resolve_known_action_ratio_inputs(
        action_ratio_values=action_ratio_values,
        behavior_log_prob=behavior_log_prob,
        target_log_prob=target_log_prob,
        n_rows=n,
        q_rows=q,
        n_target_rows=n_target,
        prediction_max=known_action_ratio_clip_max,
        normalize=known_action_ratio_normalize,
    )

    prefit_nuisance = {} if _prefit_nuisance is None else dict(_prefit_nuisance)
    use_crossfit_context = (
        not prefit_nuisance
        and action_ratio_predictor is None
        and transition_ratio_predictor is None
        and known_iw_query is None
        and num_target_action_samples == 1
        and max(int(action_ratio.crossfit_folds), int(transition_ratio.crossfit_folds)) > 1
    )
    action_fit_config = replace(action_ratio, crossfit_folds=1) if use_crossfit_context else action_ratio
    transition_fit_config = replace(transition_ratio, crossfit_folds=1) if use_crossfit_context else transition_ratio

    if prefit_nuisance:
        action_fit = dict(prefit_nuisance["action_fit"])
        transition_fit = dict(prefit_nuisance["transition_fit"])
    elif known_iw_beh is not None:
        action_fit = dict(
            predictor=None,
            w_hat=np.asarray(known_iw_beh, dtype=np.float64),
            w_hat_raw=np.asarray(known_iw_beh, dtype=np.float64),
            w_hat_summary=_summarize_array(np.asarray(known_iw_beh, dtype=np.float64)),
            Xw_beh=X_sa_beh,
            X_sa=X_sa_beh,
            X_sa_pi=X_sa_pi,
            history=[],
            updates=0,
            known_action_ratio=True,
            prediction_offset=0.0,
            prediction_max=known_action_ratio_clip_max,
            prediction_power=1.0,
            normalize_predictions=bool(known_action_ratio_normalize),
            prediction_scale=1.0,
            known_action_ratio_features=X_sa_query,
            known_action_ratio_predictions=np.asarray(known_iw_query, dtype=np.float64),
        )
    elif action_ratio_predictor is None:
        X_sa_beh_for_action = X_sa_beh if X_sa_pi.shape[0] == X_sa_beh.shape[0] else np.concatenate(
            [S_pi, A[target_row_index]],
            axis=1,
        ).astype(np.float32, copy=False)
        action_fit = fit_action_ratio_neural(X_sa_beh_for_action, X_sa_pi, action_fit_config)
    else:
        w_raw = action_ratio_predictor.predict(X_sa_beh, postprocess=False)
        w_hat = action_ratio_predictor.predict(X_sa_beh, postprocess=True)
        action_fit = dict(
            predictor=action_ratio_predictor,
            w_hat=w_hat,
            w_hat_raw=w_raw,
            w_hat_summary=_postprocess_summary(action_ratio_predictor, w_raw),
            Xw_beh=X_sa_beh,
            X_sa=X_sa_beh,
            X_sa_pi=X_sa_pi,
            history=[],
            updates=0,
            prefit=True,
            prediction_offset=0.0,
            prediction_max=action_ratio_predictor.prediction_max,
            prediction_power=float(action_ratio_predictor.prediction_power),
            normalize_predictions=bool(action_ratio_predictor.normalize_predictions),
            prediction_scale=float(action_ratio_predictor.prediction_scale),
        )
    if not prefit_nuisance:
        if transition_ratio_predictor is None:
            transition_fit = fit_transition_ratio_neural(X_sa_beh, S_next, S, transition_fit_config)
        else:
            Xk_beh = np.concatenate([X_sa_beh, S_next], axis=1)
            k_raw = transition_ratio_predictor.predict(Xk_beh, postprocess=False)
            k_hat = transition_ratio_predictor.predict(Xk_beh, postprocess=True)
            transition_fit = dict(
                predictor=transition_ratio_predictor,
                k_hat=k_hat,
                k_hat_raw=k_raw,
                k_hat_summary=_postprocess_summary(transition_ratio_predictor, k_raw),
                Xk_beh=Xk_beh,
                X_sa=X_sa_beh,
                S_feat=S,
                S_next_feat=S_next,
                history=[],
                updates=0,
                prefit=True,
                prediction_offset=0.0,
                prediction_max=transition_ratio_predictor.prediction_max,
                prediction_power=float(transition_ratio_predictor.prediction_power),
                normalize_predictions=bool(transition_ratio_predictor.normalize_predictions),
                prediction_scale=float(transition_ratio_predictor.prediction_scale),
            )
    if prefit_nuisance:
        source_fit = prefit_nuisance.get("source_fit")
        source_weight_query = _checked_optional_neural_query_vector(
            prefit_nuisance.get("source_weight_query"),
            q,
            "source_weight_query",
        )
        source_state_query = _checked_optional_neural_query_vector(
            prefit_nuisance.get("source_state_query"),
            q,
            "source_state_query",
        )
        source_diagnostics = dict(prefit_nuisance.get("source_diagnostics", _source_state_ratio_diagnostics(None, None)))
    else:
        source_fit, source_weight_query, source_state_query, source_diagnostics = _fit_initial_ratio_neural_for_occupancy(
            S=S,
            X_sa_beh=X_sa_beh,
            S_query=X_s_query,
            X_sa_query=X_sa_query,
            S_initial=S_initial,
            X_sa_initial=X_sa_initial,
            initial_weights=initial_weights,
            config=source_state_ratio,
            initial_ratio_mode=resolved_initial_mode,
        )
    if source_weight_query is None:
        source_weight_query = _make_neural_factored_initial_source_weights(
            action_predictor=action_fit["predictor"],
            X_sa_query=X_sa_query,
            source_state_query=source_state_query,
            known_iw_query=known_iw_query,
        )
    if prefit_nuisance:
        c_fit = prefit_nuisance.get("c_fit")
        c_predictor = None if c_fit is None else c_fit.get("predictor")
        c_ratio_query = _checked_optional_neural_query_vector(prefit_nuisance.get("c_ratio_query"), q, "c_ratio_query")
        c_diagnostics = dict(prefit_nuisance.get("c_diagnostics", _one_step_direct_ratio_diagnostics(None, None)))
    else:
        c_fit, c_predictor, c_ratio_query, c_diagnostics = _fit_direct_one_step_ratio_neural(
            X_ref=X_sa_beh,
            X_next_pi=X_sa_next_pi,
            X_query=X_sa_query,
            source_config=source_state_ratio,
            occupancy_config=occupancy,
            one_step_ratio_mode=resolved_one_step_mode,
        )
    crossfit_context = None
    if use_crossfit_context:
        crossfit_context = _fit_neural_crossfit_nuisance_context(
            X_sa_beh=X_sa_beh,
            X_sa_pi=X_sa_pi,
            S_next=S_next,
            S_ref=S,
            action_config=action_ratio,
            transition_config=transition_ratio,
            action_predictor_final=action_fit["predictor"],
            transition_predictor_final=transition_fit["predictor"],
            seed=int(occupancy.seed),
        )
        diagnostics = crossfit_context["diagnostics"]
        action_fit["crossfit_folds"] = int(diagnostics["action_crossfit_folds"])
        transition_fit["crossfit_folds"] = int(diagnostics["transition_crossfit_folds"])
        action_fit["crossfit"] = diagnostics
        transition_fit["crossfit"] = diagnostics

    occ_predictor = _fit_occupancy_neural(
        X_sa_beh=X_sa_beh,
        X_sa_query=X_sa_query,
        X_s_query=X_s_query,
        gamma=float(gamma),
        action_predictor=action_fit["predictor"],
        transition_predictor=transition_fit["predictor"],
        config=occupancy,
        crossfit_context=crossfit_context,
        source_state_ratio_query=source_state_query if resolved_initial_mode == "factored" else None,
        source_weight_query=source_weight_query,
        continuation_query=continuation_query,
        w_query_override=known_iw_query,
        one_step_ratio_mode=resolved_one_step_mode,
        X_sa_next_pi=X_sa_next_pi,
        successor_row_index=next_target_row_index,
        c_ratio_query=c_ratio_query,
    )
    history = occ_predictor["history"]
    pred_query_raw = occ_predictor["pred_query_raw"]
    pred_beh_raw = occ_predictor["pred_beh_raw"]
    pred_query_state = occ_predictor["pred_query_state"]
    pred_beh_state = occ_predictor["pred_beh_state"]

    if known_iw_query is not None:
        iw_query_hat = np.asarray(known_iw_query, dtype=np.float64).reshape(-1)
    else:
        iw_query_hat = action_fit["predictor"].predict(X_sa_query, postprocess=True)
    iw_hat_beh = action_fit["w_hat"]
    iw_pi_hat = iw_query_hat[:n_target]
    state_ratio_beh = _safe_divide(pred_beh_state, iw_hat_beh)
    state_ratio_pi = _safe_divide(pred_query_state[:n_target], iw_pi_hat)

    prediction_scale = None
    if occupancy.normalize_occupancy:
        projected_beh = _project_nonnegative_normalized(
            pred_beh_raw,
            max_value=occupancy.occupancy_ratio_max,
            normalize=False,
            eps=occupancy.occupancy_projection_eps,
        )
        mean_projected = float(np.mean(projected_beh)) if projected_beh.size else 1.0
        prediction_scale = (
            mean_projected
            if np.isfinite(mean_projected) and mean_projected > occupancy.occupancy_projection_eps
            else 1.0
        )

    fit_payload = dict(
        bst_w=None,
        bst_iw=None,
        bst_k=None,
        neural_w=occ_predictor["predictor"],
        neural_iw=action_fit["predictor"],
        neural_k=transition_fit["predictor"],
        k_fit=transition_fit,
        iw_fit=action_fit,
        source_fit=source_fit,
        c_fit=c_fit,
        loss=occupancy._normalized_loss(),
        pred_query=pred_query_raw,
        pred_query_raw=pred_query_raw,
        pred_query_clipped=pred_query_state,
        pred_query_stabilized=pred_query_state,
        pred_beh=pred_beh_raw,
        pred_beh_raw=pred_beh_raw,
        pred_beh_stabilized=pred_beh_state,
        X_sa_query=X_sa_query,
        X_s_query=X_s_query,
        pred_pi=pred_query_raw[:n_target],
        pred_pi_raw=pred_query_raw[:n_target],
        pred_pi_clipped=pred_query_state[:n_target],
        pred_iw=iw_hat_beh,
        pred_iw_query=iw_query_hat,
        pred_iw_pi=iw_pi_hat,
        pred_iw_beh_in_query=iw_query_hat[n_target:],
        pred_state_action_ratio_beh=pred_beh_state,
        pred_state_action_ratio_beh_raw=pred_beh_raw,
        pred_state_action_ratio_pi=pred_query_state[:n_target],
        pred_state_action_ratio_pi_raw=pred_query_raw[:n_target],
        pred_state_ratio_beh=state_ratio_beh,
        pred_state_ratio_pi=state_ratio_pi,
        pred_sa_iw_in_query=pred_query_raw[n_target:],
        pred_sa_iw_in_query_raw=pred_query_raw[n_target:],
        pred_sa_iw_in_query_clipped=pred_query_state[n_target:],
        history=history,
        stopped_early=occ_predictor["stopped_early"],
        stop_iter=occ_predictor["stop_iter"],
        stop_reason=occ_predictor["stop_reason"],
        refresh_count=occ_predictor["refresh_count"],
        gradient_steps_used=occ_predictor["gradient_steps_used"],
        accepted_count=occ_predictor["accepted_count"],
        validation_warmup_accepts=occ_predictor["validation_warmup_accepts"],
        validation_warmup_iterations=int(occupancy.validation_warmup_iterations),
        action_prediction_scale=float(action_fit.get("prediction_scale", 1.0)),
        transition_prediction_scale=float(transition_fit.get("prediction_scale", 1.0)),
        action_moment_calibration=str(action_fit.get("moment_calibration", "none")),
        transition_moment_calibration=str(transition_fit.get("moment_calibration", "none")),
        action_density_ratio_loss=str(action_fit.get("density_ratio_loss", "lsif")),
        transition_density_ratio_loss=str(transition_fit.get("density_ratio_loss", "lsif")),
        action_logistic_logit_clip=action_fit.get("logistic_logit_clip"),
        transition_logistic_logit_clip=transition_fit.get("logistic_logit_clip"),
        action_prior_correction=float(action_fit.get("prior_correction", 1.0)),
        transition_prior_correction=float(transition_fit.get("prior_correction", 1.0)),
        **source_diagnostics,
        **c_diagnostics,
        **occ_predictor["fixed_point_initialization_diag"],
        initial_ratio_mode=resolved_initial_mode,
        one_step_ratio_mode=resolved_one_step_mode,
        action_crossfit=action_fit.get("crossfit"),
        transition_crossfit=transition_fit.get("crossfit"),
        nuisance_crossfit=None if crossfit_context is None else crossfit_context.get("diagnostics", {}),
        fixed_point_damping=float(occupancy.fixed_point_damping),
        occupancy_ess_penalty=float(occupancy.occupancy_ess_penalty),
        occupancy_ess_mean_penalty=float(occupancy.occupancy_ess_mean_penalty),
        normalize_occupancy=bool(occupancy.normalize_occupancy),
        occupancy_ratio_max=occupancy.occupancy_ratio_max,
        occupancy_projection_eps=float(occupancy.occupancy_projection_eps),
        occupancy_prediction_scale=prediction_scale,
        clip_pseudo_outcomes=bool(occupancy.clip_pseudo_outcomes),
        pseudo_outcome_upper_quantile=float(occupancy.pseudo_outcome_upper_quantile),
        occupancy_sample_weight_mode=str(occupancy.occupancy_sample_weight_mode),
        num_target_action_samples=int(num_target_action_samples),
        continuation_mean=float(np.mean(continuation_query)),
        continuation_min=float(np.min(continuation_query)),
        known_action_ratio=bool(known_iw_query is not None),
        known_action_ratio_features=X_sa_query if known_iw_query is not None else None,
        known_action_ratio_predictions=known_iw_query,
    )
    validation_risks = [
        float(row["risk_new"])
        for row in history
        if "risk_new" in row and np.isfinite(float(row["risk_new"]))
    ]
    diagnostics = dict(
        stopped_early=occ_predictor["stopped_early"],
        stop_iter=occ_predictor["stop_iter"],
        stop_reason=occ_predictor["stop_reason"],
        refresh_count=occ_predictor["refresh_count"],
        gradient_steps_used=occ_predictor["gradient_steps_used"],
        accepted_count=occ_predictor["accepted_count"],
        validation_warmup_accepts=occ_predictor["validation_warmup_accepts"],
        validation_warmup_iterations=int(occupancy.validation_warmup_iterations),
        action_updates=action_fit["updates"],
        transition_updates=transition_fit["updates"],
        occupancy_updates=occ_predictor["gradient_steps_used"],
        action_best_valid_loss=action_fit.get("best_valid_loss"),
        transition_best_valid_loss=transition_fit.get("best_valid_loss"),
        occupancy_best_valid_loss=float(np.min(validation_risks)) if validation_risks else None,
        occupancy_final_valid_loss=float(validation_risks[-1]) if validation_risks else None,
        action_prediction_scale=float(action_fit.get("prediction_scale", 1.0)),
        transition_prediction_scale=float(transition_fit.get("prediction_scale", 1.0)),
        action_moment_calibration=str(action_fit.get("moment_calibration", "none")),
        transition_moment_calibration=str(transition_fit.get("moment_calibration", "none")),
        action_density_ratio_loss=str(action_fit.get("density_ratio_loss", "lsif")),
        transition_density_ratio_loss=str(transition_fit.get("density_ratio_loss", "lsif")),
        action_logistic_logit_clip=action_fit.get("logistic_logit_clip"),
        transition_logistic_logit_clip=transition_fit.get("logistic_logit_clip"),
        action_prior_correction=float(action_fit.get("prior_correction", 1.0)),
        transition_prior_correction=float(transition_fit.get("prior_correction", 1.0)),
        **source_diagnostics,
        **c_diagnostics,
        **occ_predictor["fixed_point_initialization_diag"],
        initial_ratio_mode=resolved_initial_mode,
        one_step_ratio_mode=resolved_one_step_mode,
        action_crossfit_folds=float(action_fit.get("crossfit_folds", 1)),
        transition_crossfit_folds=float(transition_fit.get("crossfit_folds", 1)),
        nuisance_crossfit_enabled=bool(crossfit_context is not None),
        loss=occupancy._normalized_loss(),
        fixed_point_damping=float(occupancy.fixed_point_damping),
        occupancy_ess_penalty=float(occupancy.occupancy_ess_penalty),
        occupancy_ess_mean_penalty=float(occupancy.occupancy_ess_mean_penalty),
        normalize_occupancy=bool(occupancy.normalize_occupancy),
        normalize_transition_cache=bool(occupancy.normalize_transition_cache),
        occupancy_ratio_max=occupancy.occupancy_ratio_max,
        occupancy_prediction_scale=prediction_scale,
        num_target_action_samples=int(num_target_action_samples),
        continuation_mean=float(np.mean(continuation_query)),
        continuation_min=float(np.min(continuation_query)),
        known_action_ratio=bool(known_iw_query is not None),
    )

    return NeuralDiscountedOccupancyRatioModel(
        occupancy_predictor=occ_predictor["predictor"],
        action_ratio_predictor=action_fit["predictor"],
        transition_ratio_predictor=transition_fit["predictor"],
        gamma=float(gamma),
        state_dim=S.shape[1],
        action_dim=A.shape[1],
        history=history,
        diagnostics=diagnostics,
        fit_payload=fit_payload,
        occupancy_normalize=bool(occupancy.normalize_occupancy),
        occupancy_ratio_max=occupancy.occupancy_ratio_max,
        occupancy_projection_eps=float(occupancy.occupancy_projection_eps),
        occupancy_prediction_scale=prediction_scale,
        occupancy_training_features=np.vstack([X_sa_query.astype(np.float64), X_sa_beh.astype(np.float64)]),
        occupancy_training_predictions=np.concatenate(
            [
                np.asarray(pred_query_state, dtype=np.float64).reshape(-1),
                np.asarray(pred_beh_state, dtype=np.float64).reshape(-1),
            ]
        ),
        action_ratio_training_features=X_sa_query if known_iw_query is not None else None,
        action_ratio_training_predictions=known_iw_query,
    )


def tune_discounted_occupancy_ratio_neural_cv(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    target_actions: Array,
    gamma: float,
    initial_states: Optional[Array] = None,
    initial_actions: Optional[Array] = None,
    initial_weights: Optional[Array] = None,
    target_next_actions: Optional[Array] = None,
    initial_ratio_mode: str = "auto",
    one_step_ratio_mode: str = "auto",
    occupancy: Optional[NeuralOccupancyRegressionConfig] = None,
    action_ratio: Optional[NeuralActionRatioConfig] = None,
    source_state_ratio: Optional[NeuralSourceStateRatioConfig] = None,
    transition_ratio: Optional[NeuralTransitionRatioConfig] = None,
    occupancy_grid: Sequence[Dict[str, Any]] = ({},),
    action_grid: Sequence[Dict[str, Any]] = ({},),
    transition_grid: Sequence[Dict[str, Any]] = ({},),
    cv_folds: int = 3,
    seed: int = 123,
    fit_final: bool = True,
) -> Dict[str, Any]:
    """Lightweight config-grid tuning for neural occupancy ratios.

    The score uses each fold fit's internal validation Bellman risk. This keeps
    tuning scalable for neural fixed-point regression while still selecting
    among stabilization and optimization knobs with the same fitted API.
    """
    _require_torch()
    S = _as_2d(states, "states")
    A = _as_2d(actions, "actions")
    S_next = _as_2d(next_states, "next_states")
    A_pi_input = np.asarray(target_actions)
    A_pi_info = _prepare_target_action_samples(S, A, A_pi_input, name="target_actions")
    A_pi = np.asarray(A_pi_info["actions"], dtype=np.float64)
    if int(A_pi_info["num_samples"]) != 1:
        raise ValueError("neural CV tuning currently expects one target action sample per row.")
    S_initial = None if initial_states is None else _as_2d(initial_states, "initial_states")
    A_initial = None if initial_actions is None else _as_2d(initial_actions, "initial_actions")
    A_pi_next = None if target_next_actions is None else _as_2d(target_next_actions, "target_next_actions")
    _validate_base_transition_inputs(S=S, A=A, S_next=S_next)
    _validate_target_action_rows(A=A, A_pi=A_pi, name="target_actions")
    _validate_initial_state_inputs(S=S, S_initial=S_initial, initial_weights=initial_weights)
    _validate_initial_action_inputs(A=A, S_initial=S_initial, A_initial=A_initial)
    if A_pi_next is not None:
        _validate_target_action_rows(A=A, A_pi=A_pi_next, name="target_next_actions")
    resolved_initial_mode = _resolve_initial_ratio_mode(initial_ratio_mode, S_initial=S_initial, A_initial=A_initial)
    resolved_one_step_mode = _resolve_one_step_ratio_mode(one_step_ratio_mode, A_pi_next=A_pi_next)
    if int(cv_folds) < 1:
        raise ValueError("cv_folds must be >= 1.")

    occupancy = NeuralOccupancyRegressionConfig() if occupancy is None else occupancy
    action_ratio = NeuralActionRatioConfig(seed=occupancy.seed) if action_ratio is None else action_ratio
    source_state_ratio = (
        NeuralSourceStateRatioConfig(seed=occupancy.seed)
        if source_state_ratio is None
        else source_state_ratio
    )
    transition_ratio = NeuralTransitionRatioConfig(seed=occupancy.seed) if transition_ratio is None else transition_ratio
    folds = _make_fold_indices(S.shape[0], int(cv_folds), int(seed))
    rows: list[dict[str, Any]] = []
    best_score = float("inf")
    best_configs: tuple[
        NeuralOccupancyRegressionConfig,
        NeuralActionRatioConfig,
        NeuralTransitionRatioConfig,
    ] | None = None

    for occ_over, act_over, trans_over in product(occupancy_grid, action_grid, transition_grid):
        occ_cfg = replace(occupancy, **dict(occ_over))
        act_cfg = replace(action_ratio, **dict(act_over))
        trans_cfg = replace(transition_ratio, **dict(trans_over))
        fold_scores = []
        for fold_id, valid_idx in enumerate(folds):
            if valid_idx.size == 0:
                continue
            train_mask = np.ones(S.shape[0], dtype=bool)
            train_mask[valid_idx] = False
            train_idx = np.flatnonzero(train_mask)
            fit = fit_discounted_occupancy_ratio_neural(
                states=S[train_idx],
                actions=A[train_idx],
                next_states=S_next[train_idx],
                target_actions=A_pi[train_idx],
                gamma=gamma,
                initial_states=_fold_initial_states(S_initial, train_idx, S.shape[0]),
                initial_actions=_fold_initial_states(A_initial, train_idx, S.shape[0]),
                initial_weights=_fold_initial_weights(initial_weights, S_initial, train_idx, S.shape[0]),
                target_next_actions=None if A_pi_next is None else A_pi_next[train_idx],
                initial_ratio_mode=resolved_initial_mode,
                one_step_ratio_mode=resolved_one_step_mode,
                occupancy=replace(occ_cfg, seed=int(seed) + 10_000 * (fold_id + 1)),
                action_ratio=replace(act_cfg, seed=int(seed) + 11_000 * (fold_id + 1)),
                source_state_ratio=replace(source_state_ratio, seed=int(seed) + 13_000 * (fold_id + 1)),
                transition_ratio=replace(trans_cfg, seed=int(seed) + 12_000 * (fold_id + 1)),
            )
            risks = [
                float(row["risk_new"])
                for row in fit.history
                if "risk_new" in row and np.isfinite(float(row["risk_new"]))
            ]
            score = float(np.min(risks)) if risks else float("inf")
            fold_scores.append(score)
            rows.append(
                dict(
                    fold=int(fold_id),
                    score=score,
                    occupancy_overrides=dict(occ_over),
                    action_overrides=dict(act_over),
                    transition_overrides=dict(trans_over),
                )
            )
        mean_score = float(np.mean(fold_scores)) if fold_scores else float("inf")
        if mean_score < best_score:
            best_score = mean_score
            best_configs = (occ_cfg, act_cfg, trans_cfg)

    if best_configs is None:
        raise RuntimeError("No valid neural CV folds were evaluated.")
    final_model = None
    if fit_final:
        final_model = fit_discounted_occupancy_ratio_neural(
            states=S,
            actions=A,
            next_states=S_next,
            target_actions=A_pi,
            gamma=gamma,
            initial_states=S_initial,
            initial_actions=A_initial,
            initial_weights=initial_weights,
            target_next_actions=A_pi_next,
            initial_ratio_mode=resolved_initial_mode,
            one_step_ratio_mode=resolved_one_step_mode,
            occupancy=best_configs[0],
            action_ratio=best_configs[1],
            source_state_ratio=source_state_ratio,
            transition_ratio=best_configs[2],
        )
    return dict(
        best_score=float(best_score),
        best_occupancy=best_configs[0],
        best_action_ratio=best_configs[1],
        best_transition_ratio=best_configs[2],
        cv_rows=rows,
        model=final_model,
    )


def fit_action_ratio_neural(X_sa_beh: Array, X_sa_pi: Array, config: NeuralActionRatioConfig) -> Dict[str, Any]:
    _require_torch()
    X_sa_beh = np.asarray(X_sa_beh, dtype=np.float32)
    X_sa_pi = np.asarray(X_sa_pi, dtype=np.float32)
    if X_sa_beh.shape != X_sa_pi.shape:
        raise ValueError("X_sa_beh and X_sa_pi must have the same shape.")
    density_ratio_loss = _normalized_density_ratio_loss(config.density_ratio_loss)
    if np.allclose(X_sa_beh, X_sa_pi, atol=1e-7, rtol=1e-7):
        predictor = _RatioPredictor.constant(
            config.initial_ratio,
            prediction_max=config.prediction_max,
            prediction_power=config.prediction_power,
            normalize_predictions=config.normalize_predictions,
        )
        w_raw = predictor.predict(X_sa_beh, postprocess=False)
        _apply_scalar_calibration(
            predictor,
            w_raw,
            moment_calibration=config.moment_calibration,
            target_mean=1.0,
        )
        w_hat, w_summary = _postprocess_with_summary(predictor, w_raw)
        return dict(
            predictor=predictor,
            w_hat=w_hat,
            w_hat_raw=w_raw,
            w_hat_summary=w_summary,
            Xw_beh=X_sa_beh,
            X_sa=X_sa_beh,
            X_sa_pi=X_sa_pi,
            history=[],
            updates=0,
            prediction_scale=float(predictor.prediction_scale),
            moment_calibration=str(config.moment_calibration),
            calibration=_calibration_dict(config.moment_calibration, predictor.prediction_scale),
            crossfit_folds=int(config.crossfit_folds),
            crossfit_seed=None if config.crossfit_seed is None else int(config.crossfit_seed),
            prediction_offset=0.0,
            prediction_max=config.prediction_max,
            prediction_power=float(config.prediction_power),
            normalize_predictions=bool(config.normalize_predictions),
            density_ratio_loss=density_ratio_loss,
            logistic_logit_clip=config.logistic_logit_clip,
            prior_correction=1.0,
            logit_summary=_ratio_summary(np.zeros(X_sa_beh.shape[0], dtype=np.float64))
            if density_ratio_loss == "logistic"
            else None,
        )

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    train_idx, valid_idx = _split_indices(X_sa_beh.shape[0], config.validation_fraction, rng)
    mean, scale = _fit_standardizer(np.vstack([X_sa_beh[train_idx], X_sa_pi[train_idx]]))
    if density_ratio_loss == "logistic":
        model = _LogitMLP(
            input_dim=X_sa_beh.shape[1],
            hidden_dims=config.hidden_dims,
            activation=config.activation,
        ).to(device)
    else:
        model = _PositiveMLP(
            input_dim=X_sa_beh.shape[1],
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            initial_output=config.initial_ratio,
        ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    xb = torch.as_tensor(_standardize(X_sa_beh, mean, scale), dtype=torch.float32, device=device)
    xp = torch.as_tensor(_standardize(X_sa_pi, mean, scale), dtype=torch.float32, device=device)
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    valid_idx_t = torch.as_tensor(valid_idx, dtype=torch.long, device=device)

    history: list[dict[str, float]] = []
    best_state = deepcopy(model.state_dict())
    best_valid = float("inf")
    patience = 0
    updates = 0
    batch = min(int(config.batch_size), max(int(train_idx.size), 1))
    for step in range(int(config.max_steps)):
        idx = train_idx_t[torch.randint(train_idx_t.numel(), (batch,), device=device)]
        if density_ratio_loss == "logistic":
            idx_pi = train_idx_t[torch.randint(train_idx_t.numel(), (batch,), device=device)]
            loss = _binary_ratio_loss(model(xb[idx]), model(xp[idx_pi]))
        else:
            loss = _action_lsif_loss(model(xb[idx]), model(xp[idx]), config.normalization_penalty)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
        opt.step()
        updates += 1

        if step == 0 or step == int(config.max_steps) - 1 or (step + 1) % 10 == 0:
            with torch.no_grad():
                idx_eval = valid_idx_t if valid_idx_t.numel() else train_idx_t
                if density_ratio_loss == "logistic":
                    valid = float(_binary_ratio_loss(model(xb[idx_eval]), model(xp[idx_eval])).item())
                else:
                    valid = float(
                        _action_lsif_loss(
                            model(xb[idx_eval]),
                            model(xp[idx_eval]),
                            config.normalization_penalty,
                        ).item()
                    )
            history.append({"step": float(step + 1), "valid_loss": valid})
            if valid + float(config.min_improvement) < best_valid:
                best_valid = valid
                best_state = deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= int(config.patience):
                    break
    model.load_state_dict(best_state)
    predictor = _RatioPredictor(
        model=_cpu_inference_model(model),
        mean=mean,
        scale=scale,
        prediction_max=config.prediction_max,
        prediction_power=float(config.prediction_power),
        normalize_predictions=bool(config.normalize_predictions),
        output_transform="logistic_ratio" if density_ratio_loss == "logistic" else "identity",
        logistic_logit_clip=config.logistic_logit_clip,
        prior_correction=1.0,
        device="cpu",
    )
    logits_beh = (
        predictor.predict_model_output(X_sa_beh)
        if density_ratio_loss == "logistic"
        else None
    )
    w_raw = predictor.predict(X_sa_beh, postprocess=False)
    _apply_scalar_calibration(
        predictor,
        w_raw,
        moment_calibration=config.moment_calibration,
        target_mean=1.0,
    )
    w_hat, w_summary = _postprocess_with_summary(predictor, w_raw)
    crossfit = _action_crossfit_diagnostics(X_sa_beh, X_sa_pi, config) if int(config.crossfit_folds) > 1 else None
    return dict(
        predictor=predictor,
        w_hat=w_hat,
        w_hat_raw=w_raw,
        w_hat_summary=w_summary,
        Xw_beh=X_sa_beh,
        X_sa=X_sa_beh,
        X_sa_pi=X_sa_pi,
        history=history,
        updates=updates,
        best_valid_loss=best_valid,
        prediction_scale=float(predictor.prediction_scale),
        moment_calibration=str(config.moment_calibration),
        calibration=_calibration_dict(config.moment_calibration, predictor.prediction_scale),
        crossfit_folds=int(config.crossfit_folds),
        crossfit_seed=None if config.crossfit_seed is None else int(config.crossfit_seed),
        crossfit=crossfit,
        prediction_offset=0.0,
        prediction_max=config.prediction_max,
        prediction_power=float(config.prediction_power),
        normalize_predictions=bool(config.normalize_predictions),
        density_ratio_loss=density_ratio_loss,
        logistic_logit_clip=config.logistic_logit_clip,
        prior_correction=float(predictor.prior_correction),
        logit_summary=_ratio_summary(logits_beh) if logits_beh is not None else None,
    )


def fit_source_state_ratio_neural(
    X_ref: Array,
    X_num: Array,
    config: NeuralSourceStateRatioConfig,
    *,
    numerator_weights: Optional[Array] = None,
) -> Dict[str, Any]:
    _require_torch()
    X_ref = np.asarray(X_ref, dtype=np.float32)
    X_num = np.asarray(X_num, dtype=np.float32)
    if X_ref.ndim != 2 or X_num.ndim != 2:
        raise ValueError("X_ref and X_num must be 2D arrays.")
    if X_ref.shape[1] != X_num.shape[1]:
        raise ValueError("X_ref and X_num must have the same number of columns.")
    if X_ref.shape[0] == 0 or X_num.shape[0] == 0:
        raise ValueError("X_ref and X_num must both contain at least one row.")
    density_ratio_loss = _normalized_density_ratio_loss(config.density_ratio_loss)
    weights_num = _normalize_source_weights(numerator_weights, X_num.shape[0], target_sum=float(X_ref.shape[0]))
    if X_ref.shape == X_num.shape and np.allclose(X_ref, X_num, atol=1e-7, rtol=1e-7) and np.allclose(
        weights_num,
        1.0,
        atol=1e-7,
        rtol=1e-7,
    ):
        predictor = _RatioPredictor.constant(
            config.initial_ratio,
            prediction_max=config.prediction_max,
            prediction_power=config.prediction_power,
            normalize_predictions=config.normalize_predictions,
        )
        raw = predictor.predict(X_ref, postprocess=False)
        _apply_scalar_calibration(
            predictor,
            raw,
            moment_calibration=config.moment_calibration,
            target_mean=1.0,
        )
        source_hat, source_summary = _postprocess_with_summary(predictor, raw)
        return dict(
            predictor=predictor,
            source_hat=source_hat,
            source_hat_raw=raw,
            source_hat_summary=source_summary,
            X_source_ref=X_ref,
            X_source_num=X_num,
            history=[],
            updates=0,
            prediction_scale=float(predictor.prediction_scale),
            moment_calibration=str(config.moment_calibration),
            calibration=_calibration_dict(config.moment_calibration, predictor.prediction_scale),
            crossfit_folds=int(config.crossfit_folds),
            crossfit_seed=None if config.crossfit_seed is None else int(config.crossfit_seed),
            prediction_offset=0.0,
            prediction_max=config.prediction_max,
            prediction_power=float(config.prediction_power),
            normalize_predictions=bool(config.normalize_predictions),
            density_ratio_loss=density_ratio_loss,
            logistic_logit_clip=config.logistic_logit_clip,
            prior_correction=1.0,
            best_valid_loss=0.0,
            logit_summary=_ratio_summary(np.zeros(X_ref.shape[0], dtype=np.float64))
            if density_ratio_loss == "logistic"
            else None,
        )

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    ref_train, ref_valid = _split_indices(X_ref.shape[0], config.validation_fraction, rng)
    num_train, num_valid = _split_indices(X_num.shape[0], config.validation_fraction, np.random.default_rng(config.seed + 71_777))
    mean, scale = _fit_standardizer(np.vstack([X_ref[ref_train], X_num[num_train]]))
    if density_ratio_loss == "logistic":
        model = _LogitMLP(
            input_dim=X_ref.shape[1],
            hidden_dims=config.hidden_dims,
            activation=config.activation,
        ).to(device)
    else:
        model = _PositiveMLP(
            input_dim=X_ref.shape[1],
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            initial_output=config.initial_ratio,
        ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    x_ref = torch.as_tensor(_standardize(X_ref, mean, scale), dtype=torch.float32, device=device)
    x_num = torch.as_tensor(_standardize(X_num, mean, scale), dtype=torch.float32, device=device)
    w_num = torch.as_tensor(weights_num.astype(np.float32), dtype=torch.float32, device=device)
    ref_train_t = torch.as_tensor(ref_train, dtype=torch.long, device=device)
    ref_valid_t = torch.as_tensor(ref_valid, dtype=torch.long, device=device)
    num_train_t = torch.as_tensor(num_train, dtype=torch.long, device=device)
    num_valid_t = torch.as_tensor(num_valid, dtype=torch.long, device=device)

    history: list[dict[str, float]] = []
    best_state = deepcopy(model.state_dict())
    best_valid = float("inf")
    patience = 0
    updates = 0
    batch_ref = min(int(config.batch_size), max(int(ref_train.size), 1))
    batch_num = min(int(config.batch_size), max(int(num_train.size), 1))
    for step in range(int(config.max_steps)):
        idx_ref = ref_train_t[torch.randint(ref_train_t.numel(), (batch_ref,), device=device)]
        idx_num = num_train_t[torch.randint(num_train_t.numel(), (batch_num,), device=device)]
        if density_ratio_loss == "logistic":
            loss = _weighted_binary_ratio_loss(model(x_ref[idx_ref]), model(x_num[idx_num]), w_num[idx_num])
        else:
            loss = _weighted_source_lsif_loss(
                model(x_ref[idx_ref]),
                model(x_num[idx_num]),
                w_num[idx_num],
                config.normalization_penalty,
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
        opt.step()
        updates += 1

        if step == 0 or step == int(config.max_steps) - 1 or (step + 1) % 10 == 0:
            with torch.no_grad():
                eval_ref = ref_valid_t if ref_valid_t.numel() else ref_train_t
                eval_num = num_valid_t if num_valid_t.numel() else num_train_t
                if density_ratio_loss == "logistic":
                    valid = float(
                        _weighted_binary_ratio_loss(
                            model(x_ref[eval_ref]),
                            model(x_num[eval_num]),
                            w_num[eval_num],
                        ).item()
                    )
                else:
                    valid = float(
                        _weighted_source_lsif_loss(
                            model(x_ref[eval_ref]),
                            model(x_num[eval_num]),
                            w_num[eval_num],
                            config.normalization_penalty,
                        ).item()
                    )
            history.append({"step": float(step + 1), "valid_loss": valid})
            if valid + float(config.min_improvement) < best_valid:
                best_valid = valid
                best_state = deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= int(config.patience):
                    break
    model.load_state_dict(best_state)
    predictor = _RatioPredictor(
        model=_cpu_inference_model(model),
        mean=mean,
        scale=scale,
        prediction_max=config.prediction_max,
        prediction_power=float(config.prediction_power),
        normalize_predictions=bool(config.normalize_predictions),
        output_transform="logistic_ratio" if density_ratio_loss == "logistic" else "identity",
        logistic_logit_clip=config.logistic_logit_clip,
        prior_correction=1.0,
        device="cpu",
    )
    logits_ref = predictor.predict_model_output(X_ref) if density_ratio_loss == "logistic" else None
    raw = predictor.predict(X_ref, postprocess=False)
    _apply_scalar_calibration(
        predictor,
        raw,
        moment_calibration=config.moment_calibration,
        target_mean=1.0,
    )
    source_hat, source_summary = _postprocess_with_summary(predictor, raw)
    return dict(
        predictor=predictor,
        source_hat=source_hat,
        source_hat_raw=raw,
        source_hat_summary=source_summary,
        X_source_ref=X_ref,
        X_source_num=X_num,
        history=history,
        updates=updates,
        best_valid_loss=best_valid,
        prediction_scale=float(predictor.prediction_scale),
        moment_calibration=str(config.moment_calibration),
        calibration=_calibration_dict(config.moment_calibration, predictor.prediction_scale),
        crossfit_folds=int(config.crossfit_folds),
        crossfit_seed=None if config.crossfit_seed is None else int(config.crossfit_seed),
        prediction_offset=0.0,
        prediction_max=config.prediction_max,
        prediction_power=float(config.prediction_power),
        normalize_predictions=bool(config.normalize_predictions),
        density_ratio_loss=density_ratio_loss,
        logistic_logit_clip=config.logistic_logit_clip,
        prior_correction=float(predictor.prior_correction),
        logit_summary=_ratio_summary(logits_ref) if logits_ref is not None else None,
    )


def fit_transition_ratio_neural(
    X_sa: Array,
    S_next: Array,
    S_ref: Array,
    config: NeuralTransitionRatioConfig,
) -> Dict[str, Any]:
    _require_torch()
    X_sa = np.asarray(X_sa, dtype=np.float32)
    S_next = np.asarray(S_next, dtype=np.float32)
    S_ref = np.asarray(S_ref, dtype=np.float32)
    if X_sa.shape[0] != S_next.shape[0] or X_sa.shape[0] != S_ref.shape[0]:
        raise ValueError("X_sa, S_next, and S_ref must have the same number of rows.")
    density_ratio_loss = _normalized_density_ratio_loss(config.density_ratio_loss)

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    train_idx, valid_idx = _split_indices(X_sa.shape[0], config.validation_fraction, rng)
    if density_ratio_loss == "logistic":
        mean, scale = _fit_standardizer(
            np.vstack(
                [
                    np.concatenate([X_sa[train_idx], S_next[train_idx]], axis=1),
                    np.concatenate([X_sa[train_idx], S_ref[train_idx]], axis=1),
                ]
            )
        )
        model = _LogitMLP(
            input_dim=X_sa.shape[1] + S_next.shape[1],
            hidden_dims=config.hidden_dims,
            activation=config.activation,
        ).to(device)
    else:
        mean, scale = _fit_standardizer(np.concatenate([X_sa[train_idx], S_next[train_idx]], axis=1))
        model = _PositiveMLP(
            input_dim=X_sa.shape[1] + S_next.shape[1],
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            initial_output=config.initial_ratio,
        ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    xsa_t = torch.as_tensor(X_sa, dtype=torch.float32, device=device)
    snext_t = torch.as_tensor(S_next, dtype=torch.float32, device=device)
    sref_t = torch.as_tensor(S_ref, dtype=torch.float32, device=device)
    mean_t = torch.as_tensor(mean.reshape(1, -1), dtype=torch.float32, device=device)
    scale_t = torch.as_tensor(scale.reshape(1, -1), dtype=torch.float32, device=device)
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    valid_idx_t = torch.as_tensor(valid_idx, dtype=torch.long, device=device)

    history: list[dict[str, float]] = []
    best_state = deepcopy(model.state_dict())
    best_valid = float("inf")
    patience = 0
    updates = 0
    batch = min(int(config.batch_size), max(int(train_idx.size), 1))
    for step in range(int(config.max_steps)):
        idx = train_idx_t[torch.randint(train_idx_t.numel(), (batch,), device=device)]
        if density_ratio_loss == "logistic":
            ref_src_idx = train_idx_t[torch.randint(train_idx_t.numel(), (batch,), device=device)]
            ref_state_idx = train_idx_t[torch.randint(train_idx_t.numel(), (batch,), device=device)]
            x_obs = torch.cat([xsa_t[idx], snext_t[idx]], dim=1)
            x_ref = torch.cat([xsa_t[ref_src_idx], sref_t[ref_state_idx]], dim=1)
            loss = _binary_ratio_loss(
                model((x_ref - mean_t) / scale_t),
                model((x_obs - mean_t) / scale_t),
            )
        else:
            ref_idx = train_idx_t[
                torch.randint(train_idx_t.numel(), (batch * int(config.permutation_samples),), device=device)
            ]
            src_idx = idx.repeat_interleave(int(config.permutation_samples))
            x_obs = torch.cat([xsa_t[idx], snext_t[idx]], dim=1)
            x_ref = torch.cat([xsa_t[src_idx], sref_t[ref_idx]], dim=1)
            loss = _transition_lsif_loss(
                model((x_obs - mean_t) / scale_t),
                model((x_ref - mean_t) / scale_t),
                config.normalization_penalty,
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
        opt.step()
        updates += 1

        if step == 0 or step == int(config.max_steps) - 1 or (step + 1) % 10 == 0:
            idx_eval = valid_idx_t if valid_idx_t.numel() else train_idx_t
            valid_batch = min(idx_eval.numel(), max(batch, 1))
            eval_idx = idx_eval[torch.randint(idx_eval.numel(), (valid_batch,), device=device)]
            with torch.no_grad():
                if density_ratio_loss == "logistic":
                    eval_ref_src = idx_eval[torch.randint(idx_eval.numel(), (valid_batch,), device=device)]
                    eval_ref_state = idx_eval[torch.randint(idx_eval.numel(), (valid_batch,), device=device)]
                    x_obs_eval = torch.cat([xsa_t[eval_idx], snext_t[eval_idx]], dim=1)
                    x_ref_eval = torch.cat([xsa_t[eval_ref_src], sref_t[eval_ref_state]], dim=1)
                    valid = float(
                        _binary_ratio_loss(
                            model((x_ref_eval - mean_t) / scale_t),
                            model((x_obs_eval - mean_t) / scale_t),
                        ).item()
                    )
                else:
                    eval_ref = idx_eval[
                        torch.randint(idx_eval.numel(), (valid_batch * int(config.permutation_samples),), device=device)
                    ]
                    eval_src = eval_idx.repeat_interleave(int(config.permutation_samples))
                    x_obs_eval = torch.cat([xsa_t[eval_idx], snext_t[eval_idx]], dim=1)
                    x_ref_eval = torch.cat([xsa_t[eval_src], sref_t[eval_ref]], dim=1)
                    valid = float(
                        _transition_lsif_loss(
                            model((x_obs_eval - mean_t) / scale_t),
                            model((x_ref_eval - mean_t) / scale_t),
                            config.normalization_penalty,
                        ).item()
                    )
            history.append({"step": float(step + 1), "valid_loss": valid})
            if valid + float(config.min_improvement) < best_valid:
                best_valid = valid
                best_state = deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= int(config.patience):
                    break
    model.load_state_dict(best_state)
    predictor = _RatioPredictor(
        model=_cpu_inference_model(model),
        mean=mean,
        scale=scale,
        prediction_max=config.prediction_max,
        prediction_power=float(config.prediction_power),
        normalize_predictions=bool(config.normalize_predictions),
        output_transform="logistic_ratio" if density_ratio_loss == "logistic" else "identity",
        logistic_logit_clip=config.logistic_logit_clip,
        prior_correction=1.0,
        device="cpu",
    )
    Xk_beh = np.concatenate([X_sa, S_next], axis=1)
    logits_beh = predictor.predict_model_output(Xk_beh) if density_ratio_loss == "logistic" else None
    k_raw = predictor.predict(Xk_beh, postprocess=False)
    ref_rng = np.random.default_rng(config.seed + 91_919)
    Xk_ref = np.concatenate([X_sa, S_ref[ref_rng.permutation(S_ref.shape[0])]], axis=1)
    ref_raw = predictor.predict(Xk_ref, postprocess=False)
    _apply_scalar_calibration(
        predictor,
        ref_raw,
        moment_calibration=config.moment_calibration,
        target_mean=1.0,
    )
    k_hat, k_summary = _postprocess_with_summary(predictor, k_raw)
    crossfit = (
        _transition_crossfit_diagnostics(X_sa, S_next, S_ref, config)
        if int(config.crossfit_folds) > 1
        else None
    )
    return dict(
        predictor=predictor,
        k_hat=k_hat,
        k_hat_raw=k_raw,
        k_hat_summary=k_summary,
        Xk_beh=Xk_beh,
        X_sa=X_sa,
        S_feat=S_ref,
        S_next_feat=S_next,
        history=history,
        updates=updates,
        best_valid_loss=best_valid,
        prediction_scale=float(predictor.prediction_scale),
        moment_calibration=str(config.moment_calibration),
        calibration=_calibration_dict(config.moment_calibration, predictor.prediction_scale),
        crossfit_folds=int(config.crossfit_folds),
        crossfit_seed=None if config.crossfit_seed is None else int(config.crossfit_seed),
        crossfit=crossfit,
        prediction_offset=0.0,
        prediction_max=config.prediction_max,
        prediction_power=float(config.prediction_power),
        normalize_predictions=bool(config.normalize_predictions),
        density_ratio_loss=density_ratio_loss,
        logistic_logit_clip=config.logistic_logit_clip,
        prior_correction=float(predictor.prior_correction),
        logit_summary=_ratio_summary(logits_beh) if logits_beh is not None else None,
        reference_uses_initial_states=True,
    )


def _fit_neural_crossfit_nuisance_context(
    *,
    X_sa_beh: Array,
    X_sa_pi: Array,
    S_next: Array,
    S_ref: Array,
    action_config: NeuralActionRatioConfig,
    transition_config: NeuralTransitionRatioConfig,
    action_predictor_final: "_RatioPredictor",
    transition_predictor_final: "_RatioPredictor",
    seed: int,
) -> Optional[Dict[str, Any]]:
    action_folds = int(action_config.crossfit_folds)
    transition_folds = int(transition_config.crossfit_folds)
    folds = max(action_folds, transition_folds)
    if folds <= 1:
        return None

    fold_indices = _make_fold_indices(
        X_sa_beh.shape[0],
        folds,
        int(action_config.crossfit_seed or transition_config.crossfit_seed or seed + 31_337),
    )
    action_predictors = []
    transition_predictors = []
    action_oof = np.full(X_sa_beh.shape[0], np.nan, dtype=np.float64)
    transition_oof = np.full(X_sa_beh.shape[0], np.nan, dtype=np.float64)
    action_updates = []
    transition_updates = []

    for fold_id, valid_idx in enumerate(fold_indices):
        train_mask = np.ones(X_sa_beh.shape[0], dtype=bool)
        train_mask[valid_idx] = False
        train_idx = np.flatnonzero(train_mask)
        if train_idx.size == 0:
            action_predictors.append(action_predictor_final)
            transition_predictors.append(transition_predictor_final)
            continue

        if action_folds > 1:
            fold_action_config = replace(
                action_config,
                crossfit_folds=1,
                seed=int(action_config.seed) + 1_003 * (fold_id + 1),
                patience=max(2, min(int(action_config.patience), 6)),
            )
            action_fit = fit_action_ratio_neural(
                X_sa_beh[train_idx],
                X_sa_pi[train_idx],
                fold_action_config,
            )
            action_predictor = action_fit["predictor"]
            action_predictors.append(action_predictor)
            action_oof[valid_idx] = action_predictor.predict(X_sa_beh[valid_idx], postprocess=True)
            action_updates.append(int(action_fit.get("updates", 0)))
        else:
            action_predictors.append(action_predictor_final)

        if transition_folds > 1:
            fold_transition_config = replace(
                transition_config,
                crossfit_folds=1,
                seed=int(transition_config.seed) + 2_003 * (fold_id + 1),
                patience=max(2, min(int(transition_config.patience), 6)),
            )
            transition_fit = fit_transition_ratio_neural(
                X_sa_beh[train_idx],
                S_next[train_idx],
                S_ref[train_idx],
                fold_transition_config,
            )
            transition_predictor = transition_fit["predictor"]
            transition_predictors.append(transition_predictor)
            x_valid = np.concatenate([X_sa_beh[valid_idx], S_next[valid_idx]], axis=1)
            transition_oof[valid_idx] = transition_predictor.predict(x_valid, postprocess=True)
            transition_updates.append(int(transition_fit.get("updates", 0)))
        else:
            transition_predictors.append(transition_predictor_final)

    diagnostics = dict(
        enabled=True,
        target_builder=True,
        folds=int(folds),
        action_crossfit_folds=int(action_folds),
        transition_crossfit_folds=int(transition_folds),
        action_oof_mean=float(np.nanmean(action_oof)) if np.any(np.isfinite(action_oof)) else float("nan"),
        action_oof_min=float(np.nanmin(action_oof)) if np.any(np.isfinite(action_oof)) else float("nan"),
        action_oof_max=float(np.nanmax(action_oof)) if np.any(np.isfinite(action_oof)) else float("nan"),
        transition_oof_mean=float(np.nanmean(transition_oof)) if np.any(np.isfinite(transition_oof)) else float("nan"),
        transition_oof_min=float(np.nanmin(transition_oof)) if np.any(np.isfinite(transition_oof)) else float("nan"),
        transition_oof_max=float(np.nanmax(transition_oof)) if np.any(np.isfinite(transition_oof)) else float("nan"),
        action_fold_updates=float(np.mean(action_updates)) if action_updates else 0.0,
        transition_fold_updates=float(np.mean(transition_updates)) if transition_updates else 0.0,
    )
    return dict(
        folds=fold_indices,
        action_predictors=action_predictors,
        transition_predictors=transition_predictors,
        action_oof=action_oof,
        transition_oof=transition_oof,
        diagnostics=diagnostics,
    )


def _fit_source_state_ratio_neural_for_occupancy(
    *,
    S: Array,
    S_query: Array,
    S_initial: Optional[Array],
    initial_weights: Optional[Array],
    config: NeuralSourceStateRatioConfig,
) -> tuple[Optional[Dict[str, Any]], Optional[Array], Dict[str, Any]]:
    if S_initial is None:
        return None, None, _source_state_ratio_diagnostics(None, None)
    fit = fit_source_state_ratio_neural(
        S,
        S_initial,
        config,
        numerator_weights=initial_weights,
    )
    source_query = fit["predictor"].predict(S_query, postprocess=True)
    return fit, source_query, _source_state_ratio_diagnostics(source_query, fit)


def _fit_initial_ratio_neural_for_occupancy(
    *,
    S: Array,
    X_sa_beh: Array,
    S_query: Array,
    X_sa_query: Array,
    S_initial: Optional[Array],
    X_sa_initial: Optional[Array],
    initial_weights: Optional[Array],
    config: NeuralSourceStateRatioConfig,
    initial_ratio_mode: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Array], Optional[Array], Dict[str, Any]]:
    if initial_ratio_mode == "joint":
        if X_sa_initial is None:
            raise ValueError("initial_ratio_mode='joint' requires initial state-action rows.")
        fit = fit_source_state_ratio_neural(
            X_sa_beh,
            X_sa_initial,
            config,
            numerator_weights=initial_weights,
        )
        source_query = fit["predictor"].predict(X_sa_query, postprocess=True)
        joint_diagnostics = _source_state_ratio_diagnostics(source_query, fit)
        diagnostics = _source_state_ratio_diagnostics(None, None)
        diagnostics.update(
            initial_joint_ratio_enabled=True,
            initial_joint_ratio_mean=joint_diagnostics["source_state_ratio_mean"],
            initial_joint_ratio_max=joint_diagnostics["source_state_ratio_max"],
            initial_joint_ratio_ess_fraction=joint_diagnostics["source_state_ratio_ess_fraction"],
            initial_joint_ratio_loss=joint_diagnostics["source_state_ratio_loss"],
            initial_joint_ratio_updates=joint_diagnostics["source_state_ratio_updates"],
            initial_joint_ratio_density_ratio_loss=joint_diagnostics["source_state_ratio_density_ratio_loss"],
            initial_joint_ratio_clipped_fraction=joint_diagnostics["source_state_ratio_clipped_fraction"],
            initial_joint_ratio_query_clipped_fraction=joint_diagnostics[
                "source_state_ratio_query_clipped_fraction"
            ],
            initial_joint_ratio_prediction_max=joint_diagnostics["source_state_ratio_prediction_max"],
            initial_joint_ratio_prediction_scale=joint_diagnostics["source_state_ratio_prediction_scale"],
        )
        return fit, source_query, None, diagnostics

    fit, source_state_query, diagnostics = _fit_source_state_ratio_neural_for_occupancy(
        S=S,
        S_query=S_query,
        S_initial=S_initial,
        initial_weights=initial_weights,
        config=config,
    )
    diagnostics.update(
        initial_joint_ratio_enabled=False,
        initial_joint_ratio_mean=1.0,
        initial_joint_ratio_max=1.0,
        initial_joint_ratio_ess_fraction=1.0,
        initial_joint_ratio_loss=float("nan"),
        initial_joint_ratio_updates=0.0,
        initial_joint_ratio_density_ratio_loss="none",
        initial_joint_ratio_clipped_fraction=0.0,
        initial_joint_ratio_query_clipped_fraction=0.0,
        initial_joint_ratio_prediction_max=float("nan"),
        initial_joint_ratio_prediction_scale=1.0,
    )
    return fit, None, source_state_query, diagnostics


def _make_neural_factored_initial_source_weights(
    *,
    action_predictor: Optional["_RatioPredictor"],
    X_sa_query: Array,
    source_state_query: Optional[Array],
    known_iw_query: Optional[Array] = None,
) -> Array:
    if known_iw_query is not None:
        out = np.asarray(known_iw_query, dtype=np.float64).reshape(-1)
        if out.shape[0] != np.asarray(X_sa_query).shape[0]:
            raise ValueError("known_iw_query must match X_sa_query rows.")
    else:
        if action_predictor is None:
            raise ValueError("action_predictor is required unless known_iw_query is supplied.")
        out = action_predictor.predict(X_sa_query, postprocess=True).astype(np.float64, copy=False)
    if source_state_query is not None:
        out = out * np.asarray(source_state_query, dtype=np.float64).reshape(-1)
    return np.maximum(out, 0.0)


def _fit_direct_one_step_ratio_neural(
    *,
    X_ref: Array,
    X_next_pi: Optional[Array],
    X_query: Array,
    source_config: NeuralSourceStateRatioConfig,
    occupancy_config: NeuralOccupancyRegressionConfig,
    one_step_ratio_mode: str,
) -> tuple[Optional[Dict[str, Any]], Optional["_RatioPredictor"], Optional[Array], Dict[str, Any]]:
    if one_step_ratio_mode != "direct":
        return None, None, None, _one_step_direct_ratio_diagnostics(None, None)
    if X_next_pi is None:
        raise ValueError("one_step_ratio_mode='direct' requires target_next_actions.")
    config = _direct_one_step_ratio_neural_config(source_config, occupancy_config)
    fit = fit_source_state_ratio_neural(X_ref, X_next_pi, config)
    c_query = fit["predictor"].predict(X_query, postprocess=True)
    return fit, fit["predictor"], c_query, _one_step_direct_ratio_diagnostics(c_query, fit)


def _direct_one_step_ratio_neural_config(
    source_config: NeuralSourceStateRatioConfig,
    occupancy_config: NeuralOccupancyRegressionConfig,
) -> NeuralSourceStateRatioConfig:
    kwargs: Dict[str, Any] = dict(
        density_ratio_loss=str(occupancy_config.direct_one_step_density_ratio_loss),
        prediction_max=occupancy_config.direct_one_step_prediction_max,
        logistic_logit_clip=occupancy_config.direct_one_step_logistic_logit_clip,
        moment_calibration=str(occupancy_config.direct_one_step_moment_calibration),
    )
    if occupancy_config.direct_one_step_max_steps is not None:
        kwargs["max_steps"] = int(occupancy_config.direct_one_step_max_steps)
    if occupancy_config.direct_one_step_hidden_dims is not None:
        kwargs["hidden_dims"] = tuple(int(width) for width in occupancy_config.direct_one_step_hidden_dims)
    return replace(source_config, **kwargs)


def _normalize_fixed_point_initialization(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in {"unit", "gamma_zero", "one_step"}:
        raise ValueError("fixed_point_initialization must be 'unit', 'gamma_zero', or 'one_step'.")
    return normalized


def _make_neural_fixed_point_initial_state(
    *,
    fixed_point_initialization: str,
    initial_ratio: float,
    source_weight_query: Array,
    target_builder: Any,
    n_behavior_rows: int,
    n_query_rows: int,
    normalize_occupancy: bool,
    occupancy_ratio_max: Optional[float],
    occupancy_projection_eps: float,
) -> tuple[Array, Array, Dict[str, Any]]:
    mode = _normalize_fixed_point_initialization(fixed_point_initialization)
    if mode == "unit":
        query_raw = np.full(int(n_query_rows), float(initial_ratio), dtype=np.float64)
    elif mode == "gamma_zero":
        query_raw = np.asarray(source_weight_query, dtype=np.float64).reshape(-1)
        if query_raw.shape[0] != int(n_query_rows):
            raise ValueError("source_weight_query must match query rows.")
    else:
        out = target_builder(w_beh=np.ones(int(n_behavior_rows), dtype=np.float64))
        query_raw = np.asarray(out["y"], dtype=np.float64).reshape(-1)
        if query_raw.shape[0] != int(n_query_rows):
            raise ValueError("one-step initial target must match query rows.")

    beh_raw = query_raw[-int(n_behavior_rows) :]
    query_state = _project_nonnegative_normalized(
        query_raw,
        max_value=occupancy_ratio_max,
        normalize=normalize_occupancy,
        eps=occupancy_projection_eps,
    )
    beh_state = _project_nonnegative_normalized(
        beh_raw,
        max_value=occupancy_ratio_max,
        normalize=normalize_occupancy,
        eps=occupancy_projection_eps,
    )
    return (
        query_state,
        beh_state,
        {
            "fixed_point_initialization": mode,
            "fixed_point_initial_query_mean": float(np.mean(query_state)) if query_state.size else float("nan"),
            "fixed_point_initial_query_max": float(np.max(query_state)) if query_state.size else float("nan"),
            "fixed_point_initial_behavior_mean": float(np.mean(beh_state)) if beh_state.size else float("nan"),
            "fixed_point_initial_behavior_max": float(np.max(beh_state)) if beh_state.size else float("nan"),
        },
    )


def _source_state_ratio_diagnostics(source_query: Optional[Array], fit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if source_query is None:
        return dict(
            source_state_ratio_enabled=False,
            source_state_ratio_mean=1.0,
            source_state_ratio_max=1.0,
            source_state_ratio_ess_fraction=1.0,
            source_state_ratio_loss=float("nan"),
            source_state_ratio_updates=0.0,
            source_state_ratio_density_ratio_loss="none",
            source_state_ratio_clipped_fraction=0.0,
            source_state_ratio_query_clipped_fraction=0.0,
            source_state_ratio_prediction_max=float("nan"),
            source_state_ratio_prediction_scale=1.0,
        )
    values = np.asarray(source_query, dtype=np.float64).reshape(-1)
    summary = fit.get("source_hat_summary", {}) if isinstance(fit, dict) else {}
    query_cap_fraction = _ratio_query_cap_fraction(values, fit)
    return dict(
        source_state_ratio_enabled=True,
        source_state_ratio_mean=float(np.mean(values)) if values.size else float("nan"),
        source_state_ratio_max=float(np.max(values)) if values.size else float("nan"),
        source_state_ratio_ess_fraction=float(_ess(values) / max(values.size, 1)),
        source_state_ratio_loss=float(fit.get("best_valid_loss", float("nan"))) if isinstance(fit, dict) else float("nan"),
        source_state_ratio_updates=float(fit.get("updates", 0.0)) if isinstance(fit, dict) else 0.0,
        source_state_ratio_density_ratio_loss=str(fit.get("density_ratio_loss", "lsif")) if isinstance(fit, dict) else "",
        source_state_ratio_clipped_fraction=float(summary.get("clipped_fraction", 0.0)),
        source_state_ratio_query_clipped_fraction=float(query_cap_fraction),
        source_state_ratio_prediction_max=_fit_prediction_max(fit),
        source_state_ratio_prediction_scale=float(fit.get("prediction_scale", 1.0)) if isinstance(fit, dict) else 1.0,
    )


def _one_step_direct_ratio_diagnostics(c_query: Optional[Array], fit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if c_query is None:
        return dict(
            one_step_direct_ratio_enabled=False,
            one_step_direct_ratio_mean=1.0,
            one_step_direct_ratio_max=1.0,
            one_step_direct_ratio_ess_fraction=1.0,
            one_step_direct_ratio_loss=float("nan"),
            one_step_direct_ratio_updates=0.0,
            one_step_direct_ratio_clipped_fraction=0.0,
            one_step_direct_ratio_query_clipped_fraction=0.0,
            one_step_direct_ratio_density_ratio_loss="none",
            one_step_direct_ratio_prediction_max=float("nan"),
            one_step_direct_ratio_prediction_scale=1.0,
        )
    values = np.asarray(c_query, dtype=np.float64).reshape(-1)
    summary = fit.get("source_hat_summary", {}) if isinstance(fit, dict) else {}
    query_cap_fraction = _ratio_query_cap_fraction(values, fit)
    return dict(
        one_step_direct_ratio_enabled=True,
        one_step_direct_ratio_mean=float(np.mean(values)) if values.size else float("nan"),
        one_step_direct_ratio_max=float(np.max(values)) if values.size else float("nan"),
        one_step_direct_ratio_ess_fraction=float(_ess(values) / max(values.size, 1)),
        one_step_direct_ratio_loss=float(fit.get("best_valid_loss", float("nan"))) if isinstance(fit, dict) else float("nan"),
        one_step_direct_ratio_updates=float(fit.get("updates", 0.0)) if isinstance(fit, dict) else 0.0,
        one_step_direct_ratio_clipped_fraction=float(summary.get("clipped_fraction", 0.0)),
        one_step_direct_ratio_query_clipped_fraction=float(query_cap_fraction),
        one_step_direct_ratio_density_ratio_loss=str(fit.get("density_ratio_loss", "lsif")) if isinstance(fit, dict) else "",
        one_step_direct_ratio_prediction_max=_fit_prediction_max(fit),
        one_step_direct_ratio_prediction_scale=float(fit.get("prediction_scale", 1.0)) if isinstance(fit, dict) else 1.0,
    )


def _summarize_array(values: Array) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return dict(
            mean=float("nan"),
            min=float("nan"),
            p50=float("nan"),
            p90=float("nan"),
            p95=float("nan"),
            p99=float("nan"),
            max=float("nan"),
            ess=float("nan"),
            clipped_fraction=0.0,
        )
    return dict(
        mean=float(np.mean(arr)),
        min=float(np.min(arr)),
        p50=float(np.quantile(arr, 0.50)),
        p90=float(np.quantile(arr, 0.90)),
        p95=float(np.quantile(arr, 0.95)),
        p99=float(np.quantile(arr, 0.99)),
        max=float(np.max(arr)),
        ess=float(_ess(arr)),
        clipped_fraction=0.0,
    )


def _fit_occupancy_neural(
    *,
    X_sa_beh: Array,
    X_sa_query: Array,
    X_s_query: Array,
    gamma: float,
    action_predictor: Optional["_RatioPredictor"],
    transition_predictor: "_RatioPredictor",
    config: NeuralOccupancyRegressionConfig,
    crossfit_context: Optional[Dict[str, Any]] = None,
    source_state_ratio_query: Optional[Array] = None,
    source_weight_query: Optional[Array] = None,
    continuation_query: Optional[Array] = None,
    w_query_override: Optional[Array] = None,
    one_step_ratio_mode: str = "factored",
    X_sa_next_pi: Optional[Array] = None,
    successor_row_index: Optional[Array] = None,
    c_ratio_query: Optional[Array] = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(config.seed + 2029)
    torch.manual_seed(config.seed + 2029)
    device = torch.device(config.device)
    q = X_sa_query.shape[0]
    train_idx, test_idx = _split_indices(q, config.validation_fraction, rng) if config.early_stopping else (
        np.arange(q, dtype=np.int64),
        np.array([], dtype=np.int64),
    )
    mean, scale = _fit_standardizer(X_sa_query[train_idx])
    model = _PositiveMLP(
        input_dim=X_sa_query.shape[1],
        hidden_dims=config.hidden_dims,
        activation=config.activation,
        initial_output=config.initial_ratio,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    predictor = _RatioPredictor(
        model=_cpu_inference_model(model),
        mean=mean,
        scale=scale,
        prediction_max=None,
        prediction_power=1.0,
        normalize_predictions=False,
        device="cpu",
    )

    x_query_t = torch.as_tensor(_standardize(X_sa_query, mean, scale), dtype=torch.float32, device=device)
    x_beh_t = torch.as_tensor(_standardize(X_sa_beh, mean, scale), dtype=torch.float32, device=device)
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    batch = min(int(config.batch_size), max(int(train_idx.size), 1))
    beh_batch = min(int(config.batch_size), max(int(X_sa_beh.shape[0]), 1))

    def sync_predictor() -> None:
        predictor.model = _cpu_inference_model(model)

    refresh_count = 0

    def make_builder(seed_for_builder: int, mcmc_for_builder: int) -> "_NeuralTargetBuilder":
        if one_step_ratio_mode == "direct":
            return _NeuralDirectTargetBuilder(
                X_sa_successor=X_sa_next_pi,
                X_sa_query=X_sa_query,
                c_ratio_query=c_ratio_query,
                w_source_query=source_weight_query,
                continuation_query=continuation_query,
                successor_row_index=successor_row_index,
                gamma=gamma,
                config=config,
                seed=int(seed_for_builder),
            )
        if crossfit_context is not None:
            return _NeuralCrossfitTargetBuilder(
                crossfit_context=crossfit_context,
                X_sa_kernel=X_sa_beh,
                X_s_query=X_s_query,
                X_sa_query_iw=X_sa_query,
                gamma=gamma,
                mcmc_samples=int(mcmc_for_builder),
                seed=int(seed_for_builder),
                batch_query=int(config.batch_size),
                normalize_transition_cache=bool(config.normalize_transition_cache),
                transition_cache_norm_eps=float(config.transition_cache_norm_eps),
                source_state_ratio_query=source_state_ratio_query,
                w_source_query=source_weight_query if source_state_ratio_query is None else None,
                continuation_query=continuation_query,
                w_query_override=w_query_override,
            )
        return _NeuralTargetBuilder(
            transition_predictor=transition_predictor,
            action_predictor=action_predictor,
            X_sa_kernel=X_sa_beh,
            X_s_query=X_s_query,
            X_sa_query_iw=X_sa_query,
            gamma=gamma,
            mcmc_samples=int(mcmc_for_builder),
            seed=int(seed_for_builder),
            batch_query=int(config.batch_size),
            normalize_transition_cache=bool(config.normalize_transition_cache),
            transition_cache_norm_eps=float(config.transition_cache_norm_eps),
            source_state_ratio_query=source_state_ratio_query,
            w_source_query=source_weight_query if source_state_ratio_query is None else None,
            continuation_query=continuation_query,
            w_query_override=w_query_override,
        )

    def make_train_builder() -> "_NeuralTargetBuilder":
        nonlocal refresh_count
        refresh_count += 1
        return make_builder(config.seed + 10_000 * refresh_count, config.mcmc_samples)

    build_train = make_train_builder()
    build_eval = (
        make_builder(config.seed + config.eval_seed_offset, int(config.mcmc_samples * config.eval_mcmc_multiplier))
        if config.early_stopping
        else None
    )
    history: list[dict[str, Any]] = []
    patience = 0
    plateau_streak = 0
    stopped_early = False
    stop_iter: Optional[int] = None
    stop_reason: Optional[str] = None
    fixed_point_stop_streak = 0
    gradient_steps_used = 0
    accepted_count = 0
    validation_warmup_accepts = 0

    sync_predictor()
    pred_query_raw = predictor.predict(X_sa_query, postprocess=False)
    pred_beh_raw = predictor.predict(X_sa_beh, postprocess=False)
    pred_query_state, pred_beh_state, fixed_point_initialization_diag = _make_neural_fixed_point_initial_state(
        fixed_point_initialization=config.fixed_point_initialization,
        initial_ratio=config.initial_ratio,
        source_weight_query=source_weight_query,
        target_builder=build_train,
        n_behavior_rows=X_sa_beh.shape[0],
        n_query_rows=q,
        normalize_occupancy=config.normalize_occupancy,
        occupancy_ratio_max=config.occupancy_ratio_max,
        occupancy_projection_eps=config.occupancy_projection_eps,
    )
    loss_name = config._normalized_loss()

    for iteration in range(int(config.num_iterations)):
        out_train = build_train(w_beh=pred_beh_state)
        target_train, target_train_diag = _make_stabilized_fixed_point_target(
            raw_target=out_train["y"],
            current=pred_query_state,
            eta=1.0,
            normalize=bool(config.normalize_occupancy),
            occupancy_ratio_max=config.occupancy_ratio_max,
            eps=float(config.occupancy_projection_eps),
            clip_pseudo_outcomes=bool(config.clip_pseudo_outcomes),
            pseudo_outcome_max=config.pseudo_outcome_max,
            pseudo_outcome_upper_quantile=float(config.pseudo_outcome_upper_quantile),
            pseudo_outcome_min=float(config.pseudo_outcome_min),
            target_min=config.target_min,
            target_max=config.target_max,
        )
        sample_weights, sample_weight_diag = _make_occupancy_sample_weights(
            mode=config.occupancy_sample_weight_mode,
            action_ratio=out_train.get("w_query"),
            target=target_train,
            max_value=config.occupancy_sample_weight_max,
        )
        train_resid = pred_query_state[train_idx] - target_train[train_idx]
        loss_delta = _resolve_huber_delta(
            train_resid,
            loss=loss_name,
            huber_delta=config.huber_delta,
            huber_delta_scale=float(config.huber_delta_scale),
            huber_delta_quantile_power=config.huber_delta_quantile_power,
            huber_delta_min_quantile=float(config.huber_delta_min_quantile),
        )

        if config.early_stopping:
            out_eval = build_eval(w_beh=pred_beh_state)
            target_eval, target_eval_diag = _make_stabilized_fixed_point_target(
                raw_target=out_eval["y"],
                current=pred_query_state,
                eta=1.0,
                normalize=bool(config.normalize_occupancy),
                occupancy_ratio_max=config.occupancy_ratio_max,
                eps=float(config.occupancy_projection_eps),
                clip_pseudo_outcomes=bool(config.clip_pseudo_outcomes),
                pseudo_outcome_max=config.pseudo_outcome_max,
                pseudo_outcome_upper_quantile=float(config.pseudo_outcome_upper_quantile),
                pseudo_outcome_min=float(config.pseudo_outcome_min),
                target_min=config.target_min,
                target_max=config.target_max,
            )
            risk_old = _occupancy_loss_value(
                pred_query_state[test_idx],
                target_eval[test_idx],
                loss=loss_name,
                huber_delta=loss_delta,
            )
        else:
            out_eval = out_train
            target_eval = target_train
            target_eval_diag = target_train_diag
            risk_old = float("nan")

        before_state = deepcopy(model.state_dict())
        y_t = torch.as_tensor(target_train.astype(np.float32), dtype=torch.float32, device=device)
        sw_t = torch.as_tensor(sample_weights.astype(np.float32), dtype=torch.float32, device=device)
        model.train()
        for _ in range(int(config.gradient_steps_per_iteration)):
            idx = train_idx_t[torch.randint(train_idx_t.numel(), (batch,), device=device)]
            pred = model(x_query_t[idx])
            loss = _torch_regression_loss(pred, y_t[idx], sw_t[idx], loss=loss_name, huber_delta=loss_delta)
            if config.occupancy_ess_penalty > 0.0:
                beh_idx = torch.randint(x_beh_t.shape[0], (beh_batch,), device=device)
                pred_beh_penalty = model(x_beh_t[beh_idx])
                loss = loss + float(config.occupancy_ess_penalty) * _torch_occupancy_ess_penalty(
                    pred_beh_penalty,
                    mean_penalty=float(config.occupancy_ess_mean_penalty),
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
            opt.step()
            gradient_steps_used += 1
        model.eval()
        sync_predictor()
        candidate_query_raw = predictor.predict(X_sa_query, postprocess=False)
        candidate_beh_raw = predictor.predict(X_sa_beh, postprocess=False)
        candidate_query_state, query_projection_diag = _project_nonnegative_normalized(
            candidate_query_raw,
            max_value=config.occupancy_ratio_max,
            normalize=config.normalize_occupancy,
            eps=config.occupancy_projection_eps,
            return_info=True,
        )
        candidate_beh_state, beh_projection_diag = _project_nonnegative_normalized(
            candidate_beh_raw,
            max_value=config.occupancy_ratio_max,
            normalize=config.normalize_occupancy,
            eps=config.occupancy_projection_eps,
            return_info=True,
        )
        candidate_query_damped = _damped_update(pred_query_state, candidate_query_state, config.fixed_point_damping)
        candidate_beh_damped = _damped_update(pred_beh_state, candidate_beh_state, config.fixed_point_damping)

        if config.early_stopping:
            risk_new = _occupancy_loss_value(
                candidate_query_damped[test_idx],
                target_eval[test_idx],
                loss=loss_name,
                huber_delta=loss_delta,
            )
            validation_improved = risk_new <= risk_old - float(config.min_improvement)
            warmup_accept = accepted_count < int(config.validation_warmup_iterations)
            improved = validation_improved or warmup_accept
        else:
            risk_new = float("nan")
            validation_improved = True
            warmup_accept = False
            improved = True

        row = _neural_history_row(
            iteration=iteration,
            risk_old=risk_old,
            risk_new=risk_new,
            improved=improved,
            refresh_count=refresh_count,
            gradient_steps_used=gradient_steps_used,
            out_train=out_train,
            out_eval=out_eval,
            current_query=pred_query_state,
            next_query=candidate_query_damped,
            current_beh=pred_beh_state,
            next_beh=candidate_beh_damped,
            train_idx=train_idx,
            test_idx=test_idx,
            raw_update=candidate_query_raw,
            projected_update=candidate_query_state,
            damped_update=candidate_query_damped,
            target=target_train,
            target_diag=target_train_diag,
            eval_target_diag=target_eval_diag,
            query_projection_diag=query_projection_diag,
            beh_projection_diag=beh_projection_diag,
            sample_weight_diag=sample_weight_diag,
            eta=float(config.fixed_point_damping),
            occupancy_ratio_max=config.occupancy_ratio_max,
            eps=float(config.occupancy_projection_eps),
        )
        row["loss"] = loss_name
        row["occupancy_ess_penalty"] = float(config.occupancy_ess_penalty)
        row["occupancy_ess_mean_penalty"] = float(config.occupancy_ess_mean_penalty)
        row["validation_improved"] = bool(validation_improved)
        row["validation_warmup_accept"] = bool(warmup_accept and not validation_improved)
        row["validation_warmup_iterations"] = int(config.validation_warmup_iterations)
        if loss_delta is not None:
            row["huber_delta"] = float(loss_delta)

        if improved:
            accepted_count += 1
            if warmup_accept and not validation_improved:
                validation_warmup_accepts += 1
            pred_query_raw = candidate_query_raw
            pred_beh_raw = candidate_beh_raw
            pred_query_state = candidate_query_damped
            pred_beh_state = candidate_beh_damped
            patience = 0
            plateau_streak = 0
            row["accepted"] = True
            row["did_refresh"] = False
            fp_rel = row.get("fixed_point_rel_change_eval", row.get("fixed_point_rel_change_train"))
            if config.fixed_point_tol is not None and iteration + 1 >= int(config.min_outer_iterations):
                if np.isfinite(float(fp_rel)) and float(fp_rel) <= float(config.fixed_point_tol):
                    fixed_point_stop_streak += 1
                else:
                    fixed_point_stop_streak = 0
                row["fixed_point_stop_streak"] = int(fixed_point_stop_streak)
                if fixed_point_stop_streak >= int(config.fixed_point_patience):
                    stopped_early = True
                    stop_iter = int(iteration)
                    stop_reason = "fixed_point_tol"
                    history.append(row)
                    break
        else:
            model.load_state_dict(before_state)
            model.eval()
            sync_predictor()
            patience += 1
            plateau_streak += 1
            fixed_point_stop_streak = 0
            row["accepted"] = False
            if config.refresh_on_plateau and plateau_streak >= int(config.refresh_after_plateaus):
                build_train = make_train_builder()
                plateau_streak = 0
                row["did_refresh"] = True
            else:
                row["did_refresh"] = False
            if config.early_stopping and patience >= int(config.patience):
                stopped_early = True
                stop_iter = int(iteration)
                stop_reason = "validation_loss"
                history.append(row)
                break
        history.append(row)

    sync_predictor()
    pred_query_raw = predictor.predict(X_sa_query, postprocess=False)
    pred_beh_raw = predictor.predict(X_sa_beh, postprocess=False)
    return dict(
        predictor=predictor,
        history=history,
        pred_query_raw=pred_query_raw,
        pred_beh_raw=pred_beh_raw,
        pred_query_state=pred_query_state,
        pred_beh_state=pred_beh_state,
        stopped_early=stopped_early,
        stop_iter=stop_iter,
        stop_reason=stop_reason,
        refresh_count=refresh_count,
        fixed_point_initialization_diag=fixed_point_initialization_diag,
        gradient_steps_used=gradient_steps_used,
        accepted_count=accepted_count,
        validation_warmup_accepts=validation_warmup_accepts,
    )


@dataclass
class _RatioPredictor:
    model: Optional[nn.Module]
    mean: Optional[Array]
    scale: Optional[Array]
    prediction_max: Optional[float] = None
    prediction_power: float = 1.0
    normalize_predictions: bool = False
    prediction_scale: float = 1.0
    output_transform: str = "identity"
    logistic_logit_clip: Optional[float] = None
    prior_correction: float = 1.0
    device: str = "cpu"
    constant_value: Optional[float] = None

    @classmethod
    def constant(
        cls,
        value: float,
        *,
        prediction_max: Optional[float],
        prediction_power: float,
        normalize_predictions: bool,
    ) -> "_RatioPredictor":
        return cls(
            model=None,
            mean=None,
            scale=None,
            prediction_max=prediction_max,
            prediction_power=prediction_power,
            normalize_predictions=normalize_predictions,
            prediction_scale=1.0,
            output_transform="identity",
            prior_correction=1.0,
            constant_value=float(value),
        )

    def predict(self, x: Array, *, postprocess: bool = True, batch_size: int = 8192) -> Array:
        raw_model = self.predict_model_output(x, batch_size=batch_size)
        if self.output_transform == "logistic_ratio":
            raw = _logits_to_ratio(
                raw_model,
                logit_clip=self.logistic_logit_clip,
                prior_correction=float(self.prior_correction),
            )
        elif self.output_transform == "identity":
            raw = raw_model
        else:
            raise ValueError("output_transform must be 'identity' or 'logistic_ratio'.")
        if not postprocess:
            return raw
        processed, _ = _postprocess_ratio_predictions(
            raw,
            clip_nonneg=True,
            prediction_max=self.prediction_max,
            prediction_power=float(self.prediction_power),
            normalize_predictions=bool(self.normalize_predictions),
        )
        return processed * float(self.prediction_scale)

    def predict_model_output(self, x: Array, *, batch_size: int = 8192) -> Array:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim != 2:
            raise ValueError("prediction features must be 2D.")
        if self.constant_value is not None:
            return np.full(x_arr.shape[0], float(self.constant_value), dtype=np.float64)
        else:
            if self.model is None or self.mean is None or self.scale is None:
                raise ValueError("non-constant predictor requires model, mean, and scale.")
            _require_torch()
            z = _standardize(x_arr, self.mean, self.scale)
            preds = []
            model = self.model.to(torch.device(self.device))
            model.eval()
            with torch.no_grad():
                for start in range(0, z.shape[0], int(batch_size)):
                    xb = torch.as_tensor(z[start : start + int(batch_size)], dtype=torch.float32, device=self.device)
                    preds.append(model(xb).detach().cpu().numpy())
            raw = np.concatenate(preds).astype(np.float64, copy=False)
            self.model = model.cpu()
            return raw


class _PositiveMLP(nn.Module if nn is not None else object):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
        initial_output: float,
    ) -> None:
        super().__init__()
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }
        act = activations.get(str(activation).lower())
        if act is None:
            raise ValueError("activation must be one of 'relu', 'tanh', 'silu', or 'gelu'.")
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, int(hidden_dim)))
            layers.append(act())
            prev = int(hidden_dim)
        final = nn.Linear(prev, 1)
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, _inverse_softplus(float(initial_output)))
        layers.append(final)
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.net(x)).squeeze(-1) + 1e-8


class _LogitMLP(nn.Module if nn is not None else object):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }
        act = activations.get(str(activation).lower())
        if act is None:
            raise ValueError("activation must be one of 'relu', 'tanh', 'silu', or 'gelu'.")
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, int(hidden_dim)))
            layers.append(act())
            prev = int(hidden_dim)
        final = nn.Linear(prev, 1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _NeuralTargetBuilder:
    def __init__(
        self,
        *,
        transition_predictor: _RatioPredictor,
        action_predictor: Optional[_RatioPredictor],
        X_sa_kernel: Array,
        X_s_query: Array,
        X_sa_query_iw: Array,
        gamma: float,
        mcmc_samples: int,
        seed: int,
        batch_query: int,
        normalize_transition_cache: bool,
        transition_cache_norm_eps: float,
        source_state_ratio_query: Optional[Array] = None,
        w_source_query: Optional[Array] = None,
        continuation_query: Optional[Array] = None,
        w_query_override: Optional[Array] = None,
    ) -> None:
        self.X_sa_kernel = np.asarray(X_sa_kernel, dtype=np.float32)
        self.X_s_query = np.asarray(X_s_query, dtype=np.float32)
        self.X_sa_query_iw = np.asarray(X_sa_query_iw, dtype=np.float32)
        self.gamma = np.float32(gamma)
        self.mcmc_samples = int(mcmc_samples)
        self.batch_query = max(1, int(batch_query))
        self.n = self.X_sa_kernel.shape[0]
        self.q = self.X_s_query.shape[0]
        if w_query_override is not None:
            self.w_query = np.asarray(w_query_override, dtype=np.float64).reshape(-1)
            if self.w_query.shape[0] != self.q:
                raise ValueError("w_query_override must match query rows.")
            self.w_query_summary = _summarize_array(self.w_query)
        else:
            if action_predictor is None:
                raise ValueError("action_predictor is required unless w_query_override is supplied.")
            w_query_raw = action_predictor.predict(self.X_sa_query_iw, postprocess=False)
            self.w_query, self.w_query_summary = _postprocess_with_summary(action_predictor, w_query_raw)
        self.w_query = self.w_query.astype(np.float32)
        self.continuation = (
            np.ones(self.q, dtype=np.float32)
            if continuation_query is None
            else np.asarray(continuation_query, dtype=np.float32).reshape(-1)
        )
        if self.continuation.shape[0] != self.q:
            raise ValueError("continuation_query must match query rows.")
        if source_state_ratio_query is not None and w_source_query is not None:
            raise ValueError("Provide either source_state_ratio_query or w_source_query, not both.")
        self.source_state_ratio_enabled = bool(source_state_ratio_query is not None)
        self.source_state_ratio = _prepare_neural_source_state_ratio(source_state_ratio_query, self.q)
        if w_source_query is None:
            self.w_source = (self.w_query * self.source_state_ratio).astype(np.float32, copy=False)
        else:
            self.w_source = _prepare_neural_source_state_ratio(w_source_query, self.q)
        self.caches = self._build_caches(
            transition_predictor=transition_predictor,
            seed=seed,
            normalize_transition_cache=normalize_transition_cache,
            transition_cache_norm_eps=transition_cache_norm_eps,
        )

    def __call__(self, *, w_beh: Array) -> Dict[str, Any]:
        w_beh32 = np.asarray(w_beh, dtype=np.float32).reshape(-1)
        if w_beh32.shape[0] != self.n:
            raise ValueError(f"w_beh must have length {self.n}.")
        np.maximum(w_beh32, np.float32(0.0), out=w_beh32)
        numer = np.empty(self.q, dtype=np.float32)
        for cache in self.caches:
            taken = w_beh32[cache["idx_flat"]]
            prod = taken * cache["k_flat"]
            numer[cache["j0"] : cache["j1"]] = prod.reshape(cache["mb"], self.mcmc_samples).mean(axis=1)
        y = self.gamma * self.continuation * self.w_query * numer + np.float32(1.0 - self.gamma) * self.w_source
        y64 = y.astype(np.float64, copy=False)
        source_summary = _source_state_ratio_summary(self.source_state_ratio, q=self.q)
        return dict(
            X=self.X_sa_query_iw,
            y=y64,
            w_query=self.w_query.astype(np.float64, copy=True),
            diag=dict(
                mean_target=float(np.mean(y64)),
                min_target=float(np.min(y64)),
                max_target=float(np.max(y64)),
                target_p95=float(np.quantile(y64, 0.95)),
                target_p99=float(np.quantile(y64, 0.99)),
                mean_w_query=float(np.mean(self.w_query)),
                source_state_ratio_enabled=bool(self.source_state_ratio_enabled),
                source_state_ratio_mean=float(source_summary["mean"]),
                source_state_ratio_max=float(source_summary["max"]),
                source_state_ratio_ess_fraction=float(source_summary["ess_fraction"]),
                continuation_mean=float(np.mean(self.continuation)),
                continuation_min=float(np.min(self.continuation)),
                w_query_min=float(self.w_query_summary["min"]),
                w_query_p50=float(self.w_query_summary["p50"]),
                w_query_p90=float(self.w_query_summary["p90"]),
                w_query_p95=float(self.w_query_summary["p95"]),
                w_query_p99=float(self.w_query_summary["p99"]),
                w_query_max=float(self.w_query_summary["max"]),
                w_query_clipped_fraction=float(self.w_query_summary["clipped_fraction"]),
                mean_forward_numer=float(np.mean(numer)),
            ),
        )

    def _build_caches(
        self,
        *,
        transition_predictor: _RatioPredictor,
        seed: int,
        normalize_transition_cache: bool,
        transition_cache_norm_eps: float,
    ) -> list[dict[str, Any]]:
        rng = np.random.default_rng(seed)
        d_sa = self.X_sa_kernel.shape[1]
        d_state = self.X_s_query.shape[1]
        caches: list[dict[str, Any]] = []
        for j0 in range(0, self.q, self.batch_query):
            j1 = min(self.q, j0 + self.batch_query)
            mb = j1 - j0
            n_flat = mb * self.mcmc_samples
            idx_flat = rng.integers(0, self.n, size=n_flat, endpoint=False).astype(np.int32, copy=False)
            xk = np.empty((n_flat, d_sa + d_state), dtype=np.float32)
            xk[:, :d_sa] = self.X_sa_kernel[idx_flat]
            s_batch = self.X_s_query[j0:j1]
            for row in range(mb):
                lo = row * self.mcmc_samples
                hi = lo + self.mcmc_samples
                xk[lo:hi, d_sa:] = s_batch[row]
            k_flat = transition_predictor.predict(xk, postprocess=True).astype(np.float32)
            caches.append(dict(j0=j0, j1=j1, mb=mb, n_flat=n_flat, idx_flat=idx_flat, k_flat=k_flat))
        if normalize_transition_cache:
            sums = np.zeros(self.n, dtype=np.float64)
            counts = np.zeros(self.n, dtype=np.float64)
            for cache in caches:
                np.add.at(sums, cache["idx_flat"], cache["k_flat"].astype(np.float64, copy=False))
                np.add.at(counts, cache["idx_flat"], 1.0)
            means = np.divide(sums, counts, out=np.ones_like(sums), where=counts > 0.0)
            means = np.maximum(means, float(transition_cache_norm_eps))
            for cache in caches:
                cache["k_flat"][:] = (
                    cache["k_flat"].astype(np.float64, copy=False) / means[cache["idx_flat"]]
                ).astype(np.float32, copy=False)
        return caches


class _NeuralDirectTargetBuilder:
    """Direct FORI target builder using c_pi and a neural adjoint regression."""

    def __init__(
        self,
        *,
        X_sa_successor: Array,
        X_sa_query: Array,
        c_ratio_query: Array,
        w_source_query: Array,
        continuation_query: Optional[Array] = None,
        successor_row_index: Optional[Array] = None,
        gamma: float,
        config: NeuralOccupancyRegressionConfig,
        seed: int,
    ) -> None:
        _require_torch()
        if X_sa_successor is None or c_ratio_query is None or w_source_query is None:
            raise ValueError("direct one-step target builder requires successor, c-ratio, and source weights.")
        self.X_sa_successor = np.asarray(X_sa_successor, dtype=np.float32)
        self.X_sa_query = np.asarray(X_sa_query, dtype=np.float32)
        if self.X_sa_successor.ndim != 2 or self.X_sa_query.ndim != 2:
            raise ValueError("X_sa_successor and X_sa_query must be 2D arrays.")
        if self.X_sa_successor.shape[1] != self.X_sa_query.shape[1]:
            raise ValueError("X_sa_successor and X_sa_query must have the same feature dimension.")
        self.n = self.X_sa_successor.shape[0]
        self.q = self.X_sa_query.shape[0]
        self.successor_index = (
            np.arange(self.n, dtype=np.int64)
            if successor_row_index is None
            else np.asarray(successor_row_index, dtype=np.int64).reshape(-1)
        )
        if self.successor_index.shape[0] != self.n:
            raise ValueError("successor_row_index must match X_sa_successor rows.")
        self.c_query = _prepare_neural_source_state_ratio(c_ratio_query, self.q)
        self.w_source = _prepare_neural_source_state_ratio(w_source_query, self.q)
        self.continuation = (
            np.ones(self.q, dtype=np.float32)
            if continuation_query is None
            else np.asarray(continuation_query, dtype=np.float32).reshape(-1)
        )
        if self.continuation.shape[0] != self.q:
            raise ValueError("continuation_query must match query rows.")
        self.gamma = np.float32(gamma)
        self.config = config
        self.seed = int(seed)
        self.call_count = 0
        self.mean, self.scale = _fit_standardizer(np.vstack([self.X_sa_successor, self.X_sa_query]))
        self.x_successor_std = _standardize(self.X_sa_successor, self.mean, self.scale)
        self.x_query_std = _standardize(self.X_sa_query, self.mean, self.scale)
        self.model: Optional[nn.Module] = None
        self.opt: Optional[torch.optim.Optimizer] = None

    def _adjoint_steps(self) -> int:
        configured = self.config.direct_adjoint_steps
        if configured is not None:
            return max(1, int(configured))
        return max(32, 4 * int(self.config.gradient_steps_per_iteration))

    def _adjoint_learning_rate(self) -> float:
        configured = self.config.direct_adjoint_learning_rate
        if configured is not None:
            return float(configured)
        return max(float(self.config.learning_rate), 1e-3)

    def _adjoint_weight_decay(self) -> float:
        configured = self.config.direct_adjoint_weight_decay
        if configured is not None:
            return float(configured)
        return min(float(self.config.weight_decay), 1e-5)

    def _ensure_model(self, *, initial_output: float, device: torch.device) -> None:
        if self.model is not None and self.opt is not None:
            self.model = self.model.to(device)
            return
        self.model = _PositiveMLP(
            input_dim=self.X_sa_successor.shape[1],
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation,
            initial_output=initial_output,
        ).to(device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._adjoint_learning_rate(),
            weight_decay=self._adjoint_weight_decay(),
        )

    def __call__(self, *, w_beh: Array) -> Dict[str, Any]:
        self.call_count += 1
        seed = self.seed + 997 * self.call_count
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        device = torch.device(self.config.device)
        w_beh32 = np.asarray(w_beh, dtype=np.float32).reshape(-1)
        if np.max(self.successor_index, initial=-1) >= w_beh32.shape[0]:
            raise ValueError("successor_row_index references rows outside w_beh.")
        w_beh32 = w_beh32[self.successor_index].astype(np.float32, copy=False)
        np.maximum(w_beh32, np.float32(0.0), out=w_beh32)
        initial_output = max(float(np.mean(w_beh32)), 1e-6)
        self._ensure_model(initial_output=initial_output, device=device)
        model = self.model
        opt = self.opt
        if model is None or opt is None:
            raise RuntimeError("direct adjoint model was not initialized.")
        x_t = torch.as_tensor(self.x_successor_std, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(w_beh32, dtype=torch.float32, device=device)
        sw_t = torch.ones_like(y_t)
        steps = self._adjoint_steps()
        batch = min(int(self.config.batch_size), max(self.n, 1))
        loss_name = self.config._normalized_loss()
        huber_delta = self.config.huber_delta
        if loss_name == "huber" and huber_delta is None:
            huber_delta = _resolve_huber_delta(
                np.full(self.n, initial_output, dtype=np.float64) - w_beh32.astype(np.float64),
                loss=loss_name,
                huber_delta=None,
                huber_delta_scale=float(self.config.huber_delta_scale),
                huber_delta_quantile_power=self.config.huber_delta_quantile_power,
                huber_delta_min_quantile=float(self.config.huber_delta_min_quantile),
            )
        for _ in range(steps):
            idx_np = rng.integers(0, self.n, size=batch, endpoint=False)
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
            pred = model(x_t[idx])
            loss = _torch_regression_loss(
                pred,
                y_t[idx],
                sw_t[idx],
                loss=loss_name,
                huber_delta=huber_delta,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(self.config.grad_clip_norm))
            opt.step()
        model.eval()
        with torch.no_grad():
            m_successor = model(x_t).detach()
            train_loss = _torch_regression_loss(
                m_successor,
                y_t,
                sw_t,
                loss=loss_name,
                huber_delta=huber_delta,
            )
            xq = torch.as_tensor(self.x_query_std, dtype=torch.float32, device=device)
            m_query = model(xq).detach().cpu().numpy().astype(np.float32, copy=False)
            train_loss_value = float(train_loss.item())
        self.model = model.cpu()
        np.maximum(m_query, np.float32(0.0), out=m_query)
        y = np.float32(1.0 - self.gamma) * self.w_source + self.gamma * self.continuation * self.c_query * m_query
        y64 = y.astype(np.float64, copy=False)
        return dict(
            X=self.X_sa_query,
            y=y64,
            w_query=self.c_query.astype(np.float64, copy=True),
            diag=dict(
                one_step_direct_ratio_enabled=True,
                direct_adjoint_steps=float(steps),
                direct_adjoint_learning_rate=float(self._adjoint_learning_rate()),
                direct_adjoint_weight_decay=float(self._adjoint_weight_decay()),
                direct_adjoint_train_loss=train_loss_value,
                initial_joint_or_factored_source_mean=float(np.mean(self.w_source)),
                continuation_mean=float(np.mean(self.continuation)),
                continuation_min=float(np.min(self.continuation)),
                mean_target=float(np.mean(y64)),
                min_target=float(np.min(y64)),
                max_target=float(np.max(y64)),
                target_p95=float(np.quantile(y64, 0.95)),
                target_p99=float(np.quantile(y64, 0.99)),
                mean_w_query=float(np.mean(self.c_query)),
                w_query_min=float(np.min(self.c_query)),
                w_query_p50=float(np.quantile(self.c_query, 0.50)),
                w_query_p90=float(np.quantile(self.c_query, 0.90)),
                w_query_p95=float(np.quantile(self.c_query, 0.95)),
                w_query_p99=float(np.quantile(self.c_query, 0.99)),
                w_query_max=float(np.max(self.c_query)),
                w_query_clipped_fraction=0.0,
                mean_forward_numer=float(np.mean(m_query)),
            ),
        )


class _NeuralCrossfitTargetBuilder:
    """Fold-aware neural target builder using held-out nuisance predictors."""

    def __init__(
        self,
        *,
        crossfit_context: Dict[str, Any],
        X_sa_kernel: Array,
        X_s_query: Array,
        X_sa_query_iw: Array,
        gamma: float,
        mcmc_samples: int,
        seed: int,
        batch_query: int,
        normalize_transition_cache: bool,
        transition_cache_norm_eps: float,
        source_state_ratio_query: Optional[Array] = None,
        w_source_query: Optional[Array] = None,
        continuation_query: Optional[Array] = None,
        w_query_override: Optional[Array] = None,
    ) -> None:
        self.X_sa_query_iw = np.asarray(X_sa_query_iw, dtype=np.float32)
        self.q = self.X_sa_query_iw.shape[0]
        self.y_full = np.empty(self.q, dtype=np.float64)
        self.w_query_full = np.empty(self.q, dtype=np.float64)
        self.builders: list[_NeuralTargetBuilder] = []
        self.query_indices: list[Array] = []
        n = np.asarray(X_sa_kernel).shape[0]
        for fold_id, fold_idx in enumerate(crossfit_context["folds"]):
            query_idx = np.concatenate([fold_idx, n + fold_idx]).astype(np.int64, copy=False)
            self.query_indices.append(query_idx)
            self.builders.append(
                _NeuralTargetBuilder(
                    transition_predictor=crossfit_context["transition_predictors"][fold_id],
                    action_predictor=crossfit_context["action_predictors"][fold_id],
                    X_sa_kernel=X_sa_kernel,
                    X_s_query=np.asarray(X_s_query, dtype=np.float32)[query_idx],
                    X_sa_query_iw=self.X_sa_query_iw[query_idx],
                    gamma=gamma,
                    mcmc_samples=int(mcmc_samples),
                    seed=int(seed) + 10_007 * (fold_id + 1),
                    batch_query=int(batch_query),
                    normalize_transition_cache=bool(normalize_transition_cache),
                    transition_cache_norm_eps=float(transition_cache_norm_eps),
                    source_state_ratio_query=None
                    if source_state_ratio_query is None
                    else np.asarray(source_state_ratio_query, dtype=np.float64)[query_idx],
                    w_source_query=None
                    if w_source_query is None
                    else np.asarray(w_source_query, dtype=np.float64)[query_idx],
                    continuation_query=None
                    if continuation_query is None
                    else np.asarray(continuation_query, dtype=np.float64)[query_idx],
                    w_query_override=None
                    if w_query_override is None
                    else np.asarray(w_query_override, dtype=np.float64)[query_idx],
                )
            )

    def __call__(self, *, w_beh: Array) -> Dict[str, Any]:
        diag_rows = []
        for query_idx, builder in zip(self.query_indices, self.builders):
            out = builder(w_beh=w_beh)
            self.y_full[query_idx] = out["y"]
            self.w_query_full[query_idx] = out["w_query"]
            diag_rows.append(out.get("diag", {}))
        return dict(
            X=self.X_sa_query_iw,
            y=self.y_full.copy(),
            w_query=self.w_query_full.copy(),
            diag=_combine_neural_builder_diags(diag_rows) | {"crossfit_target_builder": True},
        )


def _validate_neural_common(
    *,
    hidden_dims: Sequence[int],
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_steps: int,
    validation_fraction: float,
    patience: int,
    min_improvement: float,
    grad_clip_norm: Optional[float],
) -> None:
    if not tuple(hidden_dims) or any(int(dim) <= 0 for dim in hidden_dims):
        raise ValueError("hidden_dims must contain positive layer widths.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be nonnegative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be in (0, 1).")
    if patience < 0:
        raise ValueError("patience must be nonnegative.")
    if min_improvement < 0.0:
        raise ValueError("min_improvement must be nonnegative.")
    if grad_clip_norm is not None and grad_clip_norm <= 0.0:
        raise ValueError("grad_clip_norm must be positive when supplied.")


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError(
            "PyTorch is required for neural occupancy-ratio estimators. "
            "Install the neural extra with `pip install occupancy-ratio[neural]`."
        )


def _normalized_density_ratio_loss(loss: str) -> str:
    out = str(loss).lower()
    if out not in {"lsif", "logistic"}:
        raise ValueError("density_ratio_loss must be 'lsif' or 'logistic'.")
    return out


def _validate_density_ratio_loss(loss: str, logistic_logit_clip: Optional[float]) -> None:
    _normalized_density_ratio_loss(loss)
    if logistic_logit_clip is not None and float(logistic_logit_clip) <= 0.0:
        raise ValueError("logistic_logit_clip must be positive when supplied.")


def _split_indices(n: int, valid_fraction: float, rng: np.random.Generator) -> tuple[Array, Array]:
    idx = rng.permutation(int(n))
    n_valid = max(1, int(np.floor(float(valid_fraction) * int(n))))
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]
    if train_idx.size == 0:
        train_idx = idx
        valid_idx = np.array([], dtype=np.int64)
    return train_idx.astype(np.int64, copy=False), valid_idx.astype(np.int64, copy=False)


def _fit_standardizer(x: Array) -> tuple[Array, Array]:
    x_arr = np.asarray(x, dtype=np.float64)
    mean = x_arr.mean(axis=0)
    scale = x_arr.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean.astype(np.float64), scale.astype(np.float64)


def _standardize(x: Array, mean: Array, scale: Array) -> Array:
    return ((np.asarray(x, dtype=np.float64) - mean.reshape(1, -1)) / scale.reshape(1, -1)).astype(np.float32)


def _cpu_inference_model(model: nn.Module) -> nn.Module:
    clone = deepcopy(model).cpu()
    clone.eval()
    for param in clone.parameters():
        param.requires_grad_(False)
    return clone


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-8)
    if value > 20.0:
        return value
    return float(np.log(np.expm1(value)))


def _action_lsif_loss(w_beh: torch.Tensor, w_pi: torch.Tensor, normalization_penalty: float) -> torch.Tensor:
    return torch.mean(w_beh.pow(2)) - 2.0 * torch.mean(w_pi) + float(normalization_penalty) * (
        torch.mean(w_beh) - 1.0
    ).pow(2)


def _weighted_source_lsif_loss(
    w_ref: torch.Tensor,
    w_num: torch.Tensor,
    num_weight: torch.Tensor,
    normalization_penalty: float,
) -> torch.Tensor:
    weighted_num = torch.sum(num_weight * w_num) / torch.clamp(torch.sum(num_weight), min=1e-8)
    return torch.mean(w_ref.pow(2)) - 2.0 * weighted_num + float(normalization_penalty) * (
        torch.mean(w_ref) - 1.0
    ).pow(2)


def _transition_lsif_loss(k_obs: torch.Tensor, k_ref: torch.Tensor, normalization_penalty: float) -> torch.Tensor:
    return torch.mean(k_ref.pow(2)) - 2.0 * torch.mean(k_obs) + float(normalization_penalty) * (
        torch.mean(k_ref) - 1.0
    ).pow(2)


def _torch_occupancy_ess_penalty(pred_beh: torch.Tensor, *, mean_penalty: float) -> torch.Tensor:
    return torch.mean(pred_beh.pow(2)) + float(mean_penalty) * (torch.mean(pred_beh) - 1.0).pow(2)


def _binary_ratio_loss(logits_den: torch.Tensor, logits_num: torch.Tensor) -> torch.Tensor:
    loss_den = torch.mean(nn.functional.softplus(logits_den))
    loss_num = torch.mean(nn.functional.softplus(-logits_num))
    return 0.5 * (loss_den + loss_num)


def _weighted_binary_ratio_loss(
    logits_den: torch.Tensor,
    logits_num: torch.Tensor,
    num_weight: torch.Tensor,
) -> torch.Tensor:
    loss_den = torch.mean(nn.functional.softplus(logits_den))
    loss_num_values = nn.functional.softplus(-logits_num)
    loss_num = torch.sum(num_weight * loss_num_values) / torch.clamp(torch.sum(num_weight), min=1e-8)
    return 0.5 * (loss_den + loss_num)


def _normalize_source_weights(weights: Optional[Array], n_rows: int, *, target_sum: float) -> Array:
    n = int(n_rows)
    if n <= 0:
        raise ValueError("n_rows must be positive.")
    if weights is None:
        out = np.ones(n, dtype=np.float64)
    else:
        out = np.asarray(weights, dtype=np.float64).reshape(-1)
        if out.shape[0] != n:
            raise ValueError("numerator_weights must match numerator rows.")
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(out, 0.0, out=out)
    total = float(np.sum(out))
    if not np.isfinite(total) or total <= 0.0:
        out = np.ones(n, dtype=np.float64)
        total = float(n)
    return out * (float(target_sum) / total)


def _prepare_neural_source_state_ratio(source_state_ratio_query: Optional[Array], q: int) -> Array:
    if source_state_ratio_query is None:
        return np.ones(int(q), dtype=np.float32)
    values = np.asarray(source_state_ratio_query, dtype=np.float32).reshape(-1)
    if values.shape[0] != int(q):
        raise ValueError("source_state_ratio_query must match query rows.")
    values = np.nan_to_num(values, nan=0.0, posinf=np.finfo(np.float32).max / 16.0, neginf=0.0)
    np.maximum(values, np.float32(0.0), out=values)
    return values.astype(np.float32, copy=False)


def _checked_optional_neural_query_vector(values: Optional[Array], q: int, name: str) -> Optional[Array]:
    if values is None:
        return None
    out = np.asarray(values, dtype=np.float64).reshape(-1)
    if out.shape[0] != int(q):
        raise ValueError(f"{name} must match query rows.")
    return out.copy()


def _logits_to_ratio(logits: Array, *, logit_clip: Optional[float], prior_correction: float) -> Array:
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    if logit_clip is not None:
        x = np.clip(x, -float(logit_clip), float(logit_clip))
    ratio = np.exp(x) * float(prior_correction)
    return np.nan_to_num(ratio, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)


def _torch_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    loss: str,
    huber_delta: Optional[float],
) -> torch.Tensor:
    resid = pred - target
    if loss == "squared":
        values = resid.pow(2)
    else:
        if huber_delta is None:
            raise ValueError("huber_delta is required for Huber occupancy loss.")
        delta = float(huber_delta)
        abs_resid = torch.abs(resid)
        values = torch.where(abs_resid <= delta, 0.5 * resid.pow(2), delta * (abs_resid - 0.5 * delta))
    return torch.sum(weight * values) / torch.clamp(torch.sum(weight), min=1e-8)


def _postprocess_with_summary(predictor: _RatioPredictor, raw: Array) -> tuple[Array, Dict[str, float]]:
    base, base_summary = _postprocess_ratio_predictions(
        raw,
        clip_nonneg=True,
        prediction_max=predictor.prediction_max,
        prediction_power=float(predictor.prediction_power),
        normalize_predictions=bool(predictor.normalize_predictions),
    )
    processed = np.asarray(base, dtype=np.float64) * float(predictor.prediction_scale)
    summary = _ratio_summary(
        processed,
        clipped_fraction=float(base_summary.get("clipped_fraction", 0.0)),
        normalization_scale=float(base_summary.get("normalization_scale", 1.0)),
        calibration_scale=float(predictor.prediction_scale),
    )
    return processed, summary


def _postprocess_summary(predictor: _RatioPredictor, raw: Array) -> Dict[str, float]:
    return _postprocess_with_summary(predictor, raw)[1]


def _apply_scalar_calibration(
    predictor: _RatioPredictor,
    raw: Array,
    *,
    moment_calibration: str,
    target_mean: float,
    eps: float = 1e-12,
) -> None:
    method = str(moment_calibration)
    if method == "none":
        predictor.prediction_scale = 1.0
        return
    if method != "scalar":
        raise ValueError("moment_calibration must be 'none' or 'scalar'.")
    base, _ = _postprocess_ratio_predictions(
        raw,
        clip_nonneg=True,
        prediction_max=predictor.prediction_max,
        prediction_power=float(predictor.prediction_power),
        normalize_predictions=bool(predictor.normalize_predictions),
    )
    mean = float(np.mean(base)) if np.asarray(base).size else 0.0
    predictor.prediction_scale = float(target_mean) / mean if np.isfinite(mean) and mean > eps else 1.0


def _calibration_dict(moment_calibration: str, scale: float) -> Dict[str, Any]:
    return dict(
        method=str(moment_calibration),
        applied=bool(str(moment_calibration) == "scalar"),
        scale=float(scale),
    )


def _action_crossfit_diagnostics(
    X_sa_beh: Array,
    X_sa_pi: Array,
    config: NeuralActionRatioConfig,
) -> Dict[str, Any]:
    folds = _make_fold_indices(
        X_sa_beh.shape[0],
        int(config.crossfit_folds),
        int(config.crossfit_seed if config.crossfit_seed is not None else config.seed + 31_337),
    )
    oof = np.empty(X_sa_beh.shape[0], dtype=np.float64)
    fold_updates = []
    for fold_id, valid_idx in enumerate(folds):
        if valid_idx.size == 0:
            continue
        train_mask = np.ones(X_sa_beh.shape[0], dtype=bool)
        train_mask[valid_idx] = False
        train_idx = np.flatnonzero(train_mask)
        if train_idx.size == 0:
            oof[valid_idx] = 1.0
            continue
        fold_cfg = replace(
            config,
            crossfit_folds=1,
            seed=int(config.seed) + 1_003 * (fold_id + 1),
            patience=max(2, min(int(config.patience), 6)),
        )
        fit = fit_action_ratio_neural(X_sa_beh[train_idx], X_sa_pi[train_idx], fold_cfg)
        oof[valid_idx] = fit["predictor"].predict(X_sa_beh[valid_idx], postprocess=True)
        fold_updates.append(int(fit.get("updates", 0)))
    return dict(
        enabled=True,
        folds=int(config.crossfit_folds),
        oof_mean=float(np.mean(oof)),
        oof_min=float(np.min(oof)),
        oof_max=float(np.max(oof)),
        fold_updates=float(np.mean(fold_updates)) if fold_updates else 0.0,
    )


def _transition_crossfit_diagnostics(
    X_sa: Array,
    S_next: Array,
    S_ref: Array,
    config: NeuralTransitionRatioConfig,
) -> Dict[str, Any]:
    folds = _make_fold_indices(
        X_sa.shape[0],
        int(config.crossfit_folds),
        int(config.crossfit_seed if config.crossfit_seed is not None else config.seed + 41_337),
    )
    oof = np.empty(X_sa.shape[0], dtype=np.float64)
    fold_updates = []
    for fold_id, valid_idx in enumerate(folds):
        if valid_idx.size == 0:
            continue
        train_mask = np.ones(X_sa.shape[0], dtype=bool)
        train_mask[valid_idx] = False
        train_idx = np.flatnonzero(train_mask)
        if train_idx.size == 0:
            oof[valid_idx] = 1.0
            continue
        fold_cfg = replace(
            config,
            crossfit_folds=1,
            seed=int(config.seed) + 2_003 * (fold_id + 1),
            patience=max(2, min(int(config.patience), 6)),
        )
        fit = fit_transition_ratio_neural(X_sa[train_idx], S_next[train_idx], S_ref[train_idx], fold_cfg)
        x_valid = np.concatenate([X_sa[valid_idx], S_next[valid_idx]], axis=1)
        oof[valid_idx] = fit["predictor"].predict(x_valid, postprocess=True)
        fold_updates.append(int(fit.get("updates", 0)))
    return dict(
        enabled=True,
        folds=int(config.crossfit_folds),
        oof_mean=float(np.mean(oof)),
        oof_min=float(np.min(oof)),
        oof_max=float(np.max(oof)),
        fold_updates=float(np.mean(fold_updates)) if fold_updates else 0.0,
    )


def _make_fold_indices(n_rows: int, n_folds: int, seed: int) -> List[Array]:
    if int(n_folds) < 1:
        raise ValueError("crossfit_folds must be >= 1.")
    rng = np.random.default_rng(seed)
    return [fold.astype(np.int64, copy=False) for fold in np.array_split(rng.permutation(int(n_rows)), int(n_folds))]


def _combine_neural_builder_diags(diags: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = sorted({key for diag in diags for key in diag})
    for key in keys:
        vals = []
        for diag in diags:
            value = diag.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                vals.append(float(value))
        if vals:
            out[key] = float(np.mean(vals))
    return out


def _ratio_summary(
    values: Array,
    *,
    clipped_fraction: float = 0.0,
    normalization_scale: float = 1.0,
    calibration_scale: float = 1.0,
) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(
            min=float("nan"),
            p50=float("nan"),
            p90=float("nan"),
            p95=float("nan"),
            p99=float("nan"),
            max=float("nan"),
            mean=float("nan"),
            clipped_fraction=float(clipped_fraction),
            normalization_scale=float(normalization_scale),
            calibration_scale=float(calibration_scale),
        )
    return dict(
        min=float(np.min(x)),
        p50=float(np.quantile(x, 0.50)),
        p90=float(np.quantile(x, 0.90)),
        p95=float(np.quantile(x, 0.95)),
        p99=float(np.quantile(x, 0.99)),
        max=float(np.max(x)),
        mean=float(np.mean(x)),
        clipped_fraction=float(clipped_fraction),
        normalization_scale=float(normalization_scale),
        calibration_scale=float(calibration_scale),
    )


def _summarize_vector(values: Array) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(mean=float("nan"), std=float("nan"), p95=float("nan"), p99=float("nan"), max=float("nan"))
    return dict(
        mean=float(np.mean(x)),
        std=float(np.std(x)),
        p95=float(np.quantile(x, 0.95)),
        p99=float(np.quantile(x, 0.99)),
        max=float(np.max(x)),
    )


def _neural_history_row(
    *,
    iteration: int,
    risk_old: float,
    risk_new: float,
    improved: bool,
    refresh_count: int,
    gradient_steps_used: int,
    out_train: Dict[str, Any],
    out_eval: Dict[str, Any],
    current_query: Array,
    next_query: Array,
    current_beh: Array,
    next_beh: Array,
    train_idx: Array,
    test_idx: Array,
    raw_update: Array,
    projected_update: Array,
    damped_update: Array,
    target: Array,
    target_diag: Dict[str, Any],
    eval_target_diag: Dict[str, Any],
    query_projection_diag: Dict[str, float],
    beh_projection_diag: Dict[str, float],
    sample_weight_diag: Dict[str, float],
    eta: float,
    occupancy_ratio_max: Optional[float],
    eps: float,
) -> Dict[str, Any]:
    diff = np.abs(np.asarray(next_query, dtype=np.float64) - np.asarray(current_query, dtype=np.float64))
    current_abs = np.abs(np.asarray(current_query, dtype=np.float64))
    out: Dict[str, Any] = dict(
        iter=int(iteration),
        risk_old=float(risk_old),
        risk_new=float(risk_new),
        improved=bool(improved),
        refresh_count=int(refresh_count),
        gradient_steps_used=int(gradient_steps_used),
        fixed_point_abs_change_train=float(np.mean(diff[train_idx])),
        fixed_point_rel_change_train=float(np.mean(diff[train_idx]) / (np.mean(current_abs[train_idx]) + eps)),
        fixed_point_damping=float(eta),
        occupancy_ratio_max=None if occupancy_ratio_max is None else float(occupancy_ratio_max),
        projection_clipped_fraction=float(query_projection_diag.get("projection_clipped_fraction", 0.0)),
        projection_clipped_fraction_beh=float(beh_projection_diag.get("projection_clipped_fraction", 0.0)),
        projection_normalization_scale=float(query_projection_diag.get("projection_normalization_scale", 1.0)),
    )
    if test_idx.size:
        out["fixed_point_abs_change_eval"] = float(np.mean(diff[test_idx]))
        out["fixed_point_rel_change_eval"] = float(np.mean(diff[test_idx]) / (np.mean(current_abs[test_idx]) + eps))
    ess = _ess(next_beh, eps=eps)
    out["ess"] = float(ess)
    out["ess_fraction"] = float(ess / max(np.asarray(next_beh).size, 1))
    out["weight_mean"] = float(np.mean(next_beh)) if np.asarray(next_beh).size else float("nan")
    out["weight_second_moment"] = float(np.mean(np.square(next_beh))) if np.asarray(next_beh).size else float("nan")
    out["weight_max"] = float(np.max(next_beh)) if np.asarray(next_beh).size else float("nan")
    out["weight_p95"] = float(np.quantile(next_beh, 0.95)) if np.asarray(next_beh).size else float("nan")
    out["weight_p99"] = float(np.quantile(next_beh, 0.99)) if np.asarray(next_beh).size else float("nan")
    out.update(out_train.get("diag", {}))
    out.update({f"eval_{key}": val for key, val in out_eval.get("diag", {}).items()})
    out.update({f"raw_update_{key}": val for key, val in _summarize_vector(raw_update).items()})
    out.update({f"projected_update_{key}": val for key, val in _summarize_vector(projected_update).items()})
    out.update({f"damped_update_{key}": val for key, val in _summarize_vector(damped_update).items()})
    out.update({f"target_{key}": val for key, val in _summarize_vector(target).items()})
    out.update(target_diag)
    out.update({f"eval_{key}": val for key, val in eval_target_diag.items()})
    out.update(sample_weight_diag)
    return _json_friendly(out)


def _json_friendly(values: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        elif isinstance(value, (bool, str, int, float)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out
