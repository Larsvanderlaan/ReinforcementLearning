"""KL-projected fitted occupancy-ratio iteration.

This module implements the empirical KL-FORI update from
``main_KLFORI (18).tex``:

    A_hat(h) - (1 - gamma) P0_hat h
    - gamma {n^{-1} sum_i omega_k(X_i) h(X_i^+)}
        / {n^{-1} sum_i omega_k(X_i)}.

The default backend is a convex linear log-ratio model optimized with
full-batch Adam.  LightGBM and Torch backends optimize the same KL projection
objective lazily, without first-stage density ratios or backward adjoint
regressions; those are regression-FORI components.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
import pickle
from typing import Any, Callable, Optional, Sequence

import numpy as np

from occupancy_ratio import _fori_data as _shared_fori_data


Array = np.ndarray
TargetActionSampler = Callable[[Array, np.random.Generator], Array]


@dataclass
class KLFORIConfig:
    """Configuration for KL-projected FORI.

    Parameters
    ----------
    num_iterations:
        Number of outer KL-projection iterations.
    optimizer_steps:
        Optimizer steps per KL projection.  For ``backend="boosting"``, this is
        the number of one-tree functional-gradient steps.
    learning_rate:
        Optimizer learning rate for linear and neural backends.
    backend:
        Log-ratio backend.  ``"linear"`` is deterministic and dependency-light,
        ``"boosting"`` uses LightGBM functional-gradient steps, and
        ``"neural"`` lazily imports Torch and optimizes an MLP log-ratio model.
    l2_penalty:
        Optional ridge penalty on log-ratio coefficients for numerical
        stability in finite samples.
    score_tikhonov_penalty:
        Optional score-level Tikhonov penalty for the KL projection.  This
        regularizes raw model scores on reference, initial-target, and
        successor-target rows.  It is distinct from neural parameter weight
        decay and is not added to held-out validation loss.
    validation_fraction:
        Held-out reference-row fraction used only for diagnostics.
    early_stopping:
        For ``backend="neural"``, accept an outer FORI update only when the
        held-out Bellman projection loss improves after the warmup accepts.  Linear
        and boosting backends currently run the configured iteration count.
    patience:
        Number of consecutive non-improving neural outer updates allowed before
        stopping early.
    min_improvement:
        Minimum held-out KL/projection loss decrease required to accept a neural
        update after the warmup accepts.
    validation_warmup_iterations:
        Number of initial neural outer updates accepted even if the held-out
        loss does not improve.
    logit_clip:
        Optional numerical cap on centered log ratios used between KL-FORI
        iterations and at prediction time.  Capped ratios are re-normalized on
        the reference sample and cap-hit rates are reported; set ``None`` for
        the exact uncapped empirical KL iterate.
    normalize_eps:
        Lower bound used when reporting empirical masses.
    seed:
        Random seed for validation splitting and optional successor-action
        sampling.
    feature_include_quadratic:
        Include squared standardized features in the linear log-ratio model.
    boosting_learning_rate:
        Shrinkage applied to each LightGBM functional-gradient tree.
    boosting_num_leaves:
        Maximum leaves per LightGBM functional-gradient tree.
    boosting_min_data_in_leaf:
        Minimum data per LightGBM leaf.
    boosting_lgb_params:
        Extra LightGBM regressor parameters for the KL boosting backend.
    neural_hidden_dims:
        Hidden-layer widths for the lazy Torch MLP backend.
    neural_log_partition_mode:
        Neural optimizer for the empirical log-partition. ``"exact"`` uses the
        current full-reference ``logsumexp`` objective. ``"variational"`` uses
        the scalar variational form ``a - 1 + E exp(h(X)-a)`` and supports
        stochastic minibatches.
    neural_batch_size:
        Optional minibatch size for ``neural_log_partition_mode="variational"``.
        ``None`` uses the full available objective rows and recovers a
        full-batch variational objective.
    neural_variational_gauge_fix:
        Optional gauge fixing for the variational neural mode. ``"none"``
        leaves the network untouched. ``"batch"`` recenters the final neural
        score on the current reference minibatch after each optimizer step and
        shifts the variational scalar by the same amount. ``"full"`` recenters
        on the full objective reference rows. ``"auto"`` uses batch recentering
        only when successor rows are effectively nonterminal, where the KL
        objective is constant-shift invariant.
    neural_weight_decay:
        L2 penalty applied to neural network parameters.
    neural_grad_clip_norm:
        Optional gradient norm cap for neural optimization.
    min_ess_fraction:
        Optional deployable guardrail for neural early stopping.  When supplied,
        a neural outer update is not accepted if its reference-sample ESS
        fraction falls below this threshold.
    max_logit_cap_fraction:
        Optional deployable guardrail for neural early stopping.  When supplied,
        a neural outer update is not accepted if more than this fraction of
        reference logits hit ``logit_clip`` before re-normalization.
    device:
        Torch device string for the neural backend.
    show_progress:
        Reserved for future progress reporting; kept in the public config so
        callers can silence runs uniformly across estimators.
    selection_objective:
        Optional callback evaluated on the current reference-row ratio after
        each outer iteration.  This is intended for benchmark diagnostics and
        custom stopping analyses; built-in deployable KL-FORI selection does
        not use oracle rewards, target values, or ratio truth.
    selection_objective_name:
        Human-readable name stored with per-iteration callback diagnostics.
    """

    num_iterations: int = 30
    optimizer_steps: int = 300
    learning_rate: float = 5e-2
    backend: str = "linear"
    l2_penalty: float = 1e-4
    score_tikhonov_penalty: float = 0.0
    validation_fraction: float = 0.2
    early_stopping: bool = True
    patience: int = 10
    min_improvement: float = 1e-6
    validation_warmup_iterations: int = 1
    logit_clip: Optional[float] = 30.0
    normalize_eps: float = 1e-12
    seed: int = 123
    feature_include_quadratic: bool = False
    boosting_learning_rate: float = 0.1
    boosting_num_leaves: int = 31
    boosting_min_data_in_leaf: int = 5
    boosting_lgb_params: dict[str, Any] = field(default_factory=dict)
    neural_hidden_dims: Sequence[int] = (64, 64)
    neural_log_partition_mode: str = "exact"
    neural_batch_size: Optional[int] = None
    neural_variational_gauge_fix: str = "none"
    neural_weight_decay: float = 1e-4
    neural_grad_clip_norm: Optional[float] = 10.0
    min_ess_fraction: Optional[float] = None
    max_logit_cap_fraction: Optional[float] = None
    device: str = "cpu"
    show_progress: bool = False
    selection_objective: Optional[Callable[[Array], float]] = None
    selection_objective_name: str = ""

    def __post_init__(self) -> None:
        self.backend = _normalize_backend(self.backend)
        self.neural_hidden_dims = tuple(int(width) for width in self.neural_hidden_dims)
        self.neural_log_partition_mode = str(self.neural_log_partition_mode).lower()
        self.neural_variational_gauge_fix = str(
            self.neural_variational_gauge_fix
        ).lower()
        if self.num_iterations < 0:
            raise ValueError("num_iterations must be nonnegative.")
        if self.optimizer_steps <= 0:
            raise ValueError("optimizer_steps must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.l2_penalty < 0.0:
            raise ValueError("l2_penalty must be nonnegative.")
        if self.score_tikhonov_penalty < 0.0:
            raise ValueError("score_tikhonov_penalty must be nonnegative.")
        if not (0.0 <= float(self.validation_fraction) < 1.0):
            raise ValueError("validation_fraction must be in [0, 1).")
        if int(self.patience) < 0:
            raise ValueError("patience must be nonnegative.")
        if float(self.min_improvement) < 0.0:
            raise ValueError("min_improvement must be nonnegative.")
        if int(self.validation_warmup_iterations) < 0:
            raise ValueError("validation_warmup_iterations must be nonnegative.")
        if self.logit_clip is not None and self.logit_clip <= 0.0:
            raise ValueError("logit_clip must be positive when supplied.")
        if self.normalize_eps <= 0.0:
            raise ValueError("normalize_eps must be positive.")
        if self.boosting_learning_rate <= 0.0:
            raise ValueError("boosting_learning_rate must be positive.")
        if int(self.boosting_num_leaves) <= 1:
            raise ValueError("boosting_num_leaves must be greater than 1.")
        if int(self.boosting_min_data_in_leaf) <= 0:
            raise ValueError("boosting_min_data_in_leaf must be positive.")
        if any(width <= 0 for width in self.neural_hidden_dims):
            raise ValueError("neural_hidden_dims must contain positive widths.")
        if self.neural_log_partition_mode not in {"exact", "variational"}:
            raise ValueError(
                "neural_log_partition_mode must be 'exact' or 'variational'."
            )
        if self.neural_batch_size is not None and int(self.neural_batch_size) <= 0:
            raise ValueError("neural_batch_size must be positive when supplied.")
        if self.neural_variational_gauge_fix not in {"none", "auto", "batch", "full"}:
            raise ValueError(
                "neural_variational_gauge_fix must be 'none', 'auto', 'batch', or 'full'."
            )
        if self.neural_weight_decay < 0.0:
            raise ValueError("neural_weight_decay must be nonnegative.")
        if self.neural_grad_clip_norm is not None and self.neural_grad_clip_norm <= 0.0:
            raise ValueError("neural_grad_clip_norm must be positive when supplied.")
        if self.min_ess_fraction is not None and not (
            0.0 <= self.min_ess_fraction <= 1.0
        ):
            raise ValueError("min_ess_fraction must be in [0, 1] when supplied.")
        if self.max_logit_cap_fraction is not None and not (
            0.0 <= self.max_logit_cap_fraction <= 1.0
        ):
            raise ValueError("max_logit_cap_fraction must be in [0, 1] when supplied.")
        if self.selection_objective is not None and not callable(
            self.selection_objective
        ):
            raise TypeError("selection_objective must be callable when supplied.")
        self.selection_objective_name = str(self.selection_objective_name or "")


@dataclass
class KLFORIModel:
    """Fitted KL-FORI log-ratio model."""

    coef: Array
    mean: Array
    scale: Array
    gamma: float
    state_dim: int
    action_dim: int
    feature_include_quadratic: bool
    log_partition: float
    clip_log_partition_adjustment: float
    logit_clip: Optional[float]
    history: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    fit_payload: dict[str, Any] = field(default_factory=dict)
    backend: str = "linear"
    boosters: tuple[Any, ...] = field(default_factory=tuple)
    boosting_learning_rate: float = 0.1
    neural_hidden_dims: tuple[int, ...] = field(default_factory=tuple)
    neural_state_dict: dict[str, Array] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Serialize the fitted model with pickle."""
        with Path(path).open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "KLFORIModel":
        """Load a pickled KL-FORI model."""
        with Path(path).open("rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, cls):
            raise TypeError(
                f"Serialized object is {type(model).__name__}, not {cls.__name__}."
            )
        return model

    def predict_state_action_log_ratio(
        self, states: Array, actions: Array, *, clip: bool = True
    ) -> Array:
        """Predict normalized log density ratios."""
        raw = self._raw_scores(states, actions)
        centered = raw - float(self.log_partition)
        if clip and self.logit_clip is not None:
            centered = np.clip(
                centered, -float(self.logit_clip), float(self.logit_clip)
            )
            centered = centered - float(self.clip_log_partition_adjustment)
        return centered.astype(np.float64, copy=False)

    def _raw_scores(self, states: Array, actions: Array) -> Array:
        backend = _normalize_backend(self.backend)
        if backend == "linear":
            features = _features_from_state_action(
                states,
                actions,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                mean=self.mean,
                scale=self.scale,
                include_quadratic=self.feature_include_quadratic,
            )
            return features @ np.asarray(self.coef, dtype=np.float64)
        z = _standardized_from_state_action(
            states,
            actions,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            mean=self.mean,
            scale=self.scale,
        )
        if backend == "boosting":
            scores = np.zeros(z.shape[0], dtype=np.float64)
            for booster in self.boosters:
                scores += float(self.boosting_learning_rate) * _predict_lgbm_tree(
                    booster, z
                )
            return scores
        if backend == "neural":
            return _predict_neural_scores(
                z,
                state_dict=self.neural_state_dict,
                hidden_dims=self.neural_hidden_dims,
                device="cpu",
            )
        raise ValueError(f"Unknown KL-FORI backend {self.backend!r}.")

    def predict_state_action_ratio(
        self, states: Array, actions: Array, *, clip: bool = True
    ) -> Array:
        """Predict positive state-action occupancy ratios."""
        return np.exp(self.predict_state_action_log_ratio(states, actions, clip=clip))

    def predict_for_target_actions(
        self,
        states: Array,
        target_actions: Array,
        *,
        observed_actions: Optional[Array] = None,
        clip: bool = True,
    ) -> dict[str, Array]:
        """Predict target-action and optional observed-action ratios."""
        out = {
            "target_state_action_ratio": self.predict_state_action_ratio(
                states, target_actions, clip=clip
            ),
        }
        if observed_actions is not None:
            out["observed_state_action_ratio"] = self.predict_state_action_ratio(
                states,
                observed_actions,
                clip=clip,
            )
        return out


def fit_kl_fori(
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
    config: Optional[KLFORIConfig] = None,
    **unused: Any,
) -> KLFORIModel:
    """Fit the discounted occupancy ratio by KL-projected FORI.

    Parameters follow the package's standard state-action array contract.  The
    primary KL update requires successor target-policy actions
    ``target_next_actions`` or a sampler for ``next_states``.  Current-state
    ``target_actions`` are accepted for API compatibility but are not used by
    the KL objective.
    """
    del target_actions
    if unused:
        ignored = ", ".join(sorted(unused))
        raise TypeError(f"Unsupported KL-FORI arguments: {ignored}")
    if config is not None and not isinstance(config, KLFORIConfig):
        raise TypeError("config must be a KLFORIConfig when supplied.")
    cfg = KLFORIConfig() if config is None else config
    if not (0.0 <= float(gamma) < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    S = _as_2d(states, "states")
    A = _as_2d(actions, "actions")
    S_next = _as_2d(next_states, "next_states")
    if S.shape[0] != A.shape[0] or S.shape[0] != S_next.shape[0]:
        raise ValueError(
            "states, actions, and next_states must have the same number of rows."
        )
    if S.shape[0] == 0:
        raise ValueError("states must be nonempty.")
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
    A_next, successor_row_index, successor_source = _prepare_successor_actions(
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
    continuation_plus = continuation[successor_row_index]

    X_ref = np.concatenate([S, A], axis=1)
    X_init = np.concatenate([S0, A0], axis=1)
    S_plus = S_next[successor_row_index]
    X_plus = np.concatenate([S_plus, A_next], axis=1)
    init_probs = _normalize_weights(
        init_weight, n_rows=X_init.shape[0], name="initial_weights"
    )
    if split_groups is None:
        train_idx, valid_idx = _train_valid_indices(
            X_ref.shape[0], cfg.validation_fraction, cfg.seed
        )
    else:
        train_idx, valid_idx = _train_valid_indices_from_sources(
            split_groups, cfg.validation_fraction, cfg.seed
        )
    initial_split_source = (
        initial_row_index if initial_split_groups is None else initial_split_groups
    )
    init_train_idx, init_valid_idx = _train_valid_indices_from_sources(
        initial_split_source,
        cfg.validation_fraction,
        cfg.seed + 71,
    )
    backend = _normalize_backend(cfg.backend)
    split_standardizer = _can_use_neural_train_validation_split(
        cfg=cfg,
        backend=backend,
        train_idx=train_idx,
        valid_idx=valid_idx,
        init_train_idx=init_train_idx,
        init_valid_idx=init_valid_idx,
        successor_row_index=successor_row_index,
    )
    mean, scale = _fit_standardizer(X_ref[train_idx] if split_standardizer else X_ref)
    fit = _fit_backend_loop(
        cfg=cfg,
        gamma=float(gamma),
        X_ref=X_ref,
        X_init=X_init,
        X_plus=X_plus,
        mean=mean,
        scale=scale,
        init_probs=init_probs,
        successor_row_index=successor_row_index,
        continuation_plus=continuation_plus,
        train_idx=train_idx,
        valid_idx=valid_idx,
        init_train_idx=init_train_idx,
        init_valid_idx=init_valid_idx,
    )
    fit["fit_payload"]["reference_split_grouped"] = split_groups is not None
    fit["fit_payload"]["initial_split_grouped"] = initial_split_groups is not None
    model_mean = np.asarray(fit.get("feature_mean", mean), dtype=np.float64)
    model_scale = np.asarray(fit.get("feature_scale", scale), dtype=np.float64)

    diagnostics = _diagnostics(
        cfg=cfg,
        gamma=float(gamma),
        history=fit["history"],
        weights_ref=fit["weights_ref"],
        log_ratio_ref=fit["log_ratio_ref"],
        raw_log_ratio_ref=fit["log_ratio_ref_raw"],
        clip_log_partition_adjustment=float(fit["clip_log_partition_adjustment"]),
        successor_source=successor_source,
        initial_action_source=initial_action_source,
        successor_row_index=successor_row_index,
        continuation_plus=continuation_plus,
        initial_rows=X_init.shape[0],
        reference_rows=X_ref.shape[0],
        feature_dim=int(fit["feature_dim"]),
    )
    for key in (
        "refit_after_validation",
        "refit_num_iterations",
        "selection_accepted_count",
        "selection_iterations_completed",
        "selection_stopped_early",
        "selection_stop_iter",
        "selection_stop_reason",
        "selection_best_valid_loss",
        "selection_split_training",
        "selection_objective_train_rows",
        "selection_objective_init_rows",
        "selection_objective_successor_rows",
        "selection_validation_init_rows",
        "reference_split_grouped",
        "initial_split_grouped",
    ):
        if key in fit["fit_payload"]:
            diagnostics[key] = fit["fit_payload"][key]
    if bool(fit["fit_payload"].get("refit_after_validation", False)):
        diagnostics["iterations_completed"] = int(
            fit["fit_payload"].get("refit_num_iterations", 0)
        )
        diagnostics["final_stage"] = "refit"
        diagnostics["validation_loss"] = float(
            fit["fit_payload"].get("selection_best_valid_loss", float("nan"))
        )
    elif np.isfinite(float(diagnostics.get("best_valid_loss", float("nan")))):
        diagnostics["validation_loss"] = float(diagnostics["best_valid_loss"])
    fit_payload = {
        "algorithm": "kl_fori",
        "backend": cfg.backend,
        "coef": fit["coef"],
        "feature_mean": model_mean,
        "feature_scale": model_scale,
        "log_partition": float(fit["log_partition"]),
        "clip_log_partition_adjustment": float(fit["clip_log_partition_adjustment"]),
        "history": fit["history"],
        "pred_state_action_ratio_beh": fit["weights_ref"],
        "pred_state_action_ratio_beh_raw": np.exp(fit["log_ratio_ref_raw"]),
        "initial_action_source": initial_action_source,
        "successor_action_source": successor_source,
        "X_ref": X_ref,
        "X_plus": X_plus,
        "successor_row_index": successor_row_index,
    }
    fit_payload.update(fit["fit_payload"])
    return KLFORIModel(
        coef=fit["coef"],
        mean=model_mean,
        scale=model_scale,
        gamma=float(gamma),
        state_dim=S.shape[1],
        action_dim=A.shape[1],
        feature_include_quadratic=bool(cfg.feature_include_quadratic),
        log_partition=float(fit["log_partition"]),
        clip_log_partition_adjustment=float(fit["clip_log_partition_adjustment"]),
        logit_clip=cfg.logit_clip,
        history=fit["history"],
        diagnostics=diagnostics,
        fit_payload=fit_payload,
        backend=cfg.backend,
        boosters=tuple(fit["boosters"]),
        boosting_learning_rate=float(cfg.boosting_learning_rate),
        neural_hidden_dims=tuple(int(width) for width in cfg.neural_hidden_dims),
        neural_state_dict=fit["neural_state_dict"],
    )


def fit_kl_fori_boosting(**kwargs: Any) -> KLFORIModel:
    """Fit KL-FORI with the LightGBM functional-gradient backend."""
    kwargs["config"] = _config_with_backend(kwargs.get("config"), "boosting")
    return fit_kl_fori(**kwargs)


def fit_kl_fori_neural(**kwargs: Any) -> KLFORIModel:
    """Fit KL-FORI with the lazy Torch neural backend."""
    kwargs["config"] = _config_with_backend(kwargs.get("config"), "neural")
    return fit_kl_fori(**kwargs)


def _config_with_backend(config: Optional[KLFORIConfig], backend: str) -> KLFORIConfig:
    if config is None:
        return KLFORIConfig(backend=backend)
    if not isinstance(config, KLFORIConfig):
        raise TypeError("config must be a KLFORIConfig when supplied.")
    return replace(config, backend=backend)


# Standard and clipped KL-FORI intentionally share one input contract.
_as_2d = _shared_fori_data.as_2d
_prepare_initial_rows = _shared_fori_data.prepare_initial_rows
_prepare_successor_actions = _shared_fori_data.prepare_successor_actions
_sample_policy_actions = _shared_fori_data.sample_policy_actions
_resolve_continuation = _shared_fori_data.resolve_continuation
_fit_standardizer = _shared_fori_data.fit_standardizer
_standardize = _shared_fori_data.standardize
_make_features = _shared_fori_data.make_features
_features_from_state_action = _shared_fori_data.features_from_state_action
_standardized_from_state_action = _shared_fori_data.standardized_from_state_action
_normalize_weights = _shared_fori_data.normalize_weights
_optional_split_groups = _shared_fori_data.optional_split_groups
_train_valid_indices = _shared_fori_data.train_valid_indices
_train_valid_indices_from_sources = _shared_fori_data.train_valid_indices_from_sources


def _probability_subset(probabilities: Array, indices: Array) -> Array:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    subset = probs[idx].astype(np.float64, copy=True)
    total = float(np.sum(subset))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("probability subset must have positive finite mass.")
    subset /= total
    return subset


def _sample_batch_indices(
    pool: Array, *, batch_size: Optional[int], rng: np.random.Generator
) -> Array:
    values = np.asarray(pool, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot sample a minibatch from an empty index pool.")
    if batch_size is None or int(batch_size) >= values.size:
        return values
    chosen = rng.integers(0, values.size, size=int(batch_size))
    return values[chosen].astype(np.int64, copy=False)


def _subset_normalized_weight_vector(
    scores: Array, cfg: KLFORIConfig, indices: Array
) -> Array:
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        raise ValueError("normalization subset must be nonempty.")
    out = np.zeros(arr.shape[0], dtype=np.float64)
    _, _, _, log_ratio = _normalized_log_ratios(arr[idx], cfg)
    out[idx] = np.exp(log_ratio)
    return out


def _can_use_neural_train_validation_split(
    *,
    cfg: KLFORIConfig,
    backend: str,
    train_idx: Array,
    valid_idx: Array,
    init_train_idx: Array,
    init_valid_idx: Array,
    successor_row_index: Array,
) -> bool:
    if backend != "neural" or not bool(cfg.early_stopping):
        return False
    if (
        train_idx.size == 0
        or valid_idx.size == 0
        or init_train_idx.size == 0
        or init_valid_idx.size == 0
    ):
        return False
    train_mask = np.zeros(
        int(
            max(
                np.max(successor_row_index, initial=-1) + 1,
                train_idx.size + valid_idx.size,
            )
        ),
        dtype=bool,
    )
    valid_mask = np.zeros_like(train_mask)
    train_mask[np.asarray(train_idx, dtype=np.int64)] = True
    valid_mask[np.asarray(valid_idx, dtype=np.int64)] = True
    source = np.asarray(successor_row_index, dtype=np.int64).reshape(-1)
    return bool(np.any(train_mask[source]) and np.any(valid_mask[source]))


def _normalize_backend(backend: str) -> str:
    value = str(backend).strip().lower()
    aliases = {
        "linear": "linear",
        "lin": "linear",
        "boosting": "boosting",
        "boosted": "boosting",
        "lgbm": "boosting",
        "lightgbm": "boosting",
        "neural": "neural",
        "torch": "neural",
        "mlp": "neural",
    }
    if value not in aliases:
        raise ValueError("backend must be one of 'linear', 'boosting', or 'neural'.")
    return aliases[value]


def _fit_backend_loop(
    *,
    cfg: KLFORIConfig,
    gamma: float,
    X_ref: Array,
    X_init: Array,
    X_plus: Array,
    mean: Array,
    scale: Array,
    init_probs: Array,
    successor_row_index: Array,
    continuation_plus: Array,
    train_idx: Array,
    valid_idx: Array,
    init_train_idx: Array,
    init_valid_idx: Array,
) -> dict[str, Any]:
    backend = _normalize_backend(cfg.backend)
    if backend == "linear":
        return _fit_linear_backend_loop(
            cfg=cfg,
            gamma=gamma,
            X_ref=X_ref,
            X_init=X_init,
            X_plus=X_plus,
            mean=mean,
            scale=scale,
            init_probs=init_probs,
            successor_row_index=successor_row_index,
            continuation_plus=continuation_plus,
            train_idx=train_idx,
            valid_idx=valid_idx,
            init_train_idx=init_train_idx,
            init_valid_idx=init_valid_idx,
        )
    if backend == "boosting":
        return _fit_boosting_backend_loop(
            cfg=cfg,
            gamma=gamma,
            X_ref=X_ref,
            X_init=X_init,
            X_plus=X_plus,
            mean=mean,
            scale=scale,
            init_probs=init_probs,
            successor_row_index=successor_row_index,
            continuation_plus=continuation_plus,
            train_idx=train_idx,
            valid_idx=valid_idx,
            init_train_idx=init_train_idx,
            init_valid_idx=init_valid_idx,
        )
    return _fit_neural_backend_loop(
        cfg=cfg,
        gamma=gamma,
        X_ref=X_ref,
        X_init=X_init,
        X_plus=X_plus,
        mean=mean,
        scale=scale,
        init_probs=init_probs,
        successor_row_index=successor_row_index,
        continuation_plus=continuation_plus,
        train_idx=train_idx,
        valid_idx=valid_idx,
        init_train_idx=init_train_idx,
        init_valid_idx=init_valid_idx,
    )


def _fit_linear_backend_loop(
    *,
    cfg: KLFORIConfig,
    gamma: float,
    X_ref: Array,
    X_init: Array,
    X_plus: Array,
    mean: Array,
    scale: Array,
    init_probs: Array,
    successor_row_index: Array,
    continuation_plus: Array,
    train_idx: Array,
    valid_idx: Array,
    init_train_idx: Array,
    init_valid_idx: Array,
) -> dict[str, Any]:
    del init_train_idx, init_valid_idx
    Phi_ref = _make_features(
        X_ref, mean, scale, include_quadratic=cfg.feature_include_quadratic
    )
    Phi_init = _make_features(
        X_init, mean, scale, include_quadratic=cfg.feature_include_quadratic
    )
    Phi_plus = _make_features(
        X_plus, mean, scale, include_quadratic=cfg.feature_include_quadratic
    )
    init_phi = init_probs @ Phi_init

    coef = np.zeros(Phi_ref.shape[1], dtype=np.float64)
    ref_scores = Phi_ref @ coef
    log_partition, clip_adjustment, log_ratio_ref_raw, log_ratio_ref = (
        _normalized_log_ratios(ref_scores, cfg)
    )
    weights_ref = np.exp(log_ratio_ref)
    history: list[dict[str, Any]] = []

    for iteration in range(int(cfg.num_iterations)):
        previous_weights_ref = weights_ref.copy()
        target_weights_ref = previous_weights_ref
        weighted_plus_phi = _successor_weighted_features(
            Phi_plus=Phi_plus,
            weights_ref=target_weights_ref,
            successor_row_index=successor_row_index,
            continuation_plus=continuation_plus,
        )
        target_phi = (1.0 - gamma) * init_phi + gamma * weighted_plus_phi
        coef, training_objective, grad_norm = _fit_projection_adam(
            coef=coef,
            Phi_ref=Phi_ref,
            target_phi=target_phi,
            learning_rate=float(cfg.learning_rate),
            steps=int(cfg.optimizer_steps),
            l2_penalty=float(cfg.l2_penalty),
            score_tikhonov_penalty=float(cfg.score_tikhonov_penalty),
            tikhonov_blocks=((Phi_ref, None), (Phi_init, init_probs), (Phi_plus, None)),
        )
        ref_scores = Phi_ref @ coef
        init_scores = Phi_init @ coef
        plus_scores = Phi_plus @ coef
        objective = _projection_loss_from_scores(
            scores_ref=ref_scores,
            scores_init=init_scores,
            scores_plus=plus_scores,
            init_probs=init_probs,
            weights_ref=target_weights_ref,
            successor_row_index=successor_row_index,
            continuation_plus=continuation_plus,
            gamma=gamma,
        )
        tikhonov_value = _score_tikhonov_value_from_scores(
            ((ref_scores, None), (init_scores, init_probs), (plus_scores, None))
        )
        log_partition, clip_adjustment, log_ratio_ref_raw, log_ratio_ref = (
            _normalized_log_ratios(ref_scores, cfg)
        )
        weights_ref = np.exp(log_ratio_ref)
        history.append(
            _history_row(
                iteration=iteration,
                objective=objective,
                score_tikhonov_value=tikhonov_value,
                score_regularized_objective=objective
                + float(cfg.score_tikhonov_penalty) * tikhonov_value,
                training_objective=training_objective,
                grad_norm=grad_norm,
                log_partition=log_partition,
                clip_log_partition_adjustment=clip_adjustment,
                weights_ref=weights_ref,
                previous_weights_ref=previous_weights_ref,
                log_ratio_ref=log_ratio_ref,
                raw_log_ratio_ref=log_ratio_ref_raw,
                train_idx=train_idx,
                valid_idx=valid_idx,
                target_phi=target_phi,
                cfg=cfg,
            )
        )

    return {
        "coef": coef,
        "boosters": (),
        "neural_state_dict": {},
        "feature_dim": Phi_ref.shape[1],
        "log_partition": log_partition,
        "clip_log_partition_adjustment": clip_adjustment,
        "log_ratio_ref_raw": log_ratio_ref_raw,
        "log_ratio_ref": log_ratio_ref,
        "weights_ref": weights_ref,
        "history": history,
        "fit_payload": {"linear_feature_dim": Phi_ref.shape[1]},
    }


def _fit_boosting_backend_loop(
    *,
    cfg: KLFORIConfig,
    gamma: float,
    X_ref: Array,
    X_init: Array,
    X_plus: Array,
    mean: Array,
    scale: Array,
    init_probs: Array,
    successor_row_index: Array,
    continuation_plus: Array,
    train_idx: Array,
    valid_idx: Array,
    init_train_idx: Array,
    init_valid_idx: Array,
) -> dict[str, Any]:
    del init_train_idx, init_valid_idx
    try:
        import lightgbm as lgb
    except Exception as exc:  # pragma: no cover - depends on optional install state
        raise ImportError("backend='boosting' requires lightgbm.") from exc

    Z_ref = _standardize(X_ref, mean, scale)
    Z_init = _standardize(X_init, mean, scale)
    Z_plus = _standardize(X_plus, mean, scale)
    scores_ref = np.zeros(Z_ref.shape[0], dtype=np.float64)
    scores_init = np.zeros(Z_init.shape[0], dtype=np.float64)
    scores_plus = np.zeros(Z_plus.shape[0], dtype=np.float64)
    log_partition, clip_adjustment, log_ratio_ref_raw, log_ratio_ref = (
        _normalized_log_ratios(scores_ref, cfg)
    )
    weights_ref = np.exp(log_ratio_ref)
    boosters: list[Any] = []
    history: list[dict[str, Any]] = []
    fit_X = np.concatenate([Z_ref, Z_init, Z_plus], axis=0)
    sample_weight = np.concatenate(
        [
            np.full(Z_ref.shape[0], 1.0 / max(Z_ref.shape[0], 1), dtype=np.float64),
            init_probs,
            np.full(Z_plus.shape[0], 1.0 / max(Z_plus.shape[0], 1), dtype=np.float64),
        ]
    )
    base_params: dict[str, Any] = {
        "n_estimators": 1,
        "learning_rate": 1.0,
        "objective": "regression",
        "num_leaves": int(cfg.boosting_num_leaves),
        "min_data_in_leaf": int(cfg.boosting_min_data_in_leaf),
        "verbose": -1,
        "num_threads": 1,
        "random_state": int(cfg.seed),
    }
    base_params.update(dict(cfg.boosting_lgb_params))

    for iteration in range(int(cfg.num_iterations)):
        previous_weights_ref = weights_ref.copy()
        grad_norm = float("nan")
        for step in range(int(cfg.optimizer_steps)):
            candidate_ref_ratio = Z_ref.shape[0] * _softmax_mean(scores_ref)
            successor_weight = weights_ref[successor_row_index]
            successor_weight_mean = float(np.mean(successor_weight))
            if not np.isfinite(successor_weight_mean) or successor_weight_mean <= 0.0:
                raise FloatingPointError(
                    "successor weights must have positive finite empirical mean."
                )
            residual = np.concatenate(
                [
                    -candidate_ref_ratio,
                    np.full(Z_init.shape[0], 1.0 - gamma, dtype=np.float64),
                    gamma
                    * successor_weight
                    * continuation_plus
                    / successor_weight_mean,
                ]
            )
            if float(cfg.score_tikhonov_penalty) > 0.0:
                residual = residual - float(
                    cfg.score_tikhonov_penalty
                ) * np.concatenate([scores_ref, scores_init, scores_plus])
            grad_norm = _weighted_norm(residual, sample_weight)
            params = dict(base_params)
            params["random_state"] = int(cfg.seed) + 10_003 * iteration + step
            tree = lgb.LGBMRegressor(**params)
            tree.fit(fit_X, residual, sample_weight=sample_weight)
            boosters.append(tree)
            step_scale = float(cfg.boosting_learning_rate)
            scores_ref += step_scale * _predict_lgbm_tree(tree, Z_ref)
            scores_init += step_scale * _predict_lgbm_tree(tree, Z_init)
            scores_plus += step_scale * _predict_lgbm_tree(tree, Z_plus)
        objective = _projection_loss_from_scores(
            scores_ref=scores_ref,
            scores_init=scores_init,
            scores_plus=scores_plus,
            init_probs=init_probs,
            weights_ref=weights_ref,
            successor_row_index=successor_row_index,
            continuation_plus=continuation_plus,
            gamma=gamma,
        )
        tikhonov_value = _score_tikhonov_value_from_scores(
            ((scores_ref, None), (scores_init, init_probs), (scores_plus, None))
        )
        log_partition, clip_adjustment, log_ratio_ref_raw, log_ratio_ref = (
            _normalized_log_ratios(scores_ref, cfg)
        )
        weights_ref = np.exp(log_ratio_ref)
        history.append(
            _history_row(
                iteration=iteration,
                objective=objective,
                score_tikhonov_value=tikhonov_value,
                score_regularized_objective=objective
                + float(cfg.score_tikhonov_penalty) * tikhonov_value,
                grad_norm=grad_norm,
                log_partition=log_partition,
                clip_log_partition_adjustment=clip_adjustment,
                weights_ref=weights_ref,
                previous_weights_ref=previous_weights_ref,
                log_ratio_ref=log_ratio_ref,
                raw_log_ratio_ref=log_ratio_ref_raw,
                train_idx=train_idx,
                valid_idx=valid_idx,
                target_phi=np.empty(0, dtype=np.float64),
                cfg=cfg,
            )
        )

    return {
        "coef": np.empty(0, dtype=np.float64),
        "boosters": tuple(boosters),
        "neural_state_dict": {},
        "feature_dim": Z_ref.shape[1],
        "log_partition": log_partition,
        "clip_log_partition_adjustment": clip_adjustment,
        "log_ratio_ref_raw": log_ratio_ref_raw,
        "log_ratio_ref": log_ratio_ref,
        "weights_ref": weights_ref,
        "history": history,
        "fit_payload": {"boosting_num_trees": len(boosters)},
    }


def _fit_neural_backend_loop(
    *,
    cfg: KLFORIConfig,
    gamma: float,
    X_ref: Array,
    X_init: Array,
    X_plus: Array,
    mean: Array,
    scale: Array,
    init_probs: Array,
    successor_row_index: Array,
    continuation_plus: Array,
    train_idx: Array,
    valid_idx: Array,
    init_train_idx: Array,
    init_valid_idx: Array,
) -> dict[str, Any]:
    torch = _import_torch_for_kl_fori()
    device = torch.device(str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    Z_ref_np = _standardize(X_ref, mean, scale).astype(np.float32, copy=False)
    Z_init_np = _standardize(X_init, mean, scale).astype(np.float32, copy=False)
    Z_plus_np = _standardize(X_plus, mean, scale).astype(np.float32, copy=False)
    Z_ref = torch.as_tensor(Z_ref_np, dtype=torch.float32, device=device)
    Z_init = torch.as_tensor(Z_init_np, dtype=torch.float32, device=device)
    Z_plus = torch.as_tensor(Z_plus_np, dtype=torch.float32, device=device)
    successor_idx = torch.as_tensor(
        successor_row_index.astype(np.int64, copy=False),
        dtype=torch.long,
        device=device,
    )
    continuation_t = torch.as_tensor(
        continuation_plus.astype(np.float32, copy=False),
        dtype=torch.float32,
        device=device,
    )
    model = _build_torch_mlp(
        torch, Z_ref.shape[1], tuple(int(width) for width in cfg.neural_hidden_dims)
    ).to(device)
    variational_log_partition = str(cfg.neural_log_partition_mode) == "variational"
    variational_a = torch.nn.Parameter(
        torch.zeros((), dtype=torch.float32, device=device)
    )
    optimizer_params = list(model.parameters())
    if variational_log_partition:
        optimizer_params.append(variational_a)
    optimizer = torch.optim.Adam(
        optimizer_params, lr=float(cfg.learning_rate), weight_decay=0.0
    )
    grad_parameters = list(model.parameters())
    if variational_log_partition:
        grad_parameters.append(variational_a)

    scores_ref_np = np.zeros(Z_ref_np.shape[0], dtype=np.float64)
    scores_init_np = np.zeros(Z_init_np.shape[0], dtype=np.float64)
    scores_plus_np = np.zeros(Z_plus_np.shape[0], dtype=np.float64)
    log_partition, clip_adjustment, log_ratio_ref_raw, log_ratio_ref = (
        _normalized_log_ratios(scores_ref_np, cfg)
    )
    weights_ref = np.exp(log_ratio_ref)
    history: list[dict[str, Any]] = []
    patience = 0
    stopped_early = False
    stop_iter: int | None = None
    stop_reason: str | None = None
    accepted_count = 0
    validation_warmup_accepts = 0
    valid_source_mask = np.zeros(Z_ref_np.shape[0], dtype=bool)
    valid_source_mask[np.asarray(valid_idx, dtype=np.int64)] = True
    valid_successor_idx = np.flatnonzero(valid_source_mask[successor_row_index])
    init_train_idx_np = np.asarray(init_train_idx, dtype=np.int64)
    init_valid_idx_np = np.asarray(init_valid_idx, dtype=np.int64)
    early_stop_enabled = bool(
        cfg.early_stopping
        and valid_idx.size > 0
        and valid_successor_idx.size > 0
        and init_valid_idx_np.size > 0
    )
    train_ref_idx = np.asarray(train_idx, dtype=np.int64)
    train_source_mask = np.zeros(Z_ref_np.shape[0], dtype=bool)
    train_source_mask[train_ref_idx] = True
    train_successor_idx = np.flatnonzero(train_source_mask[successor_row_index])
    split_training = bool(
        early_stop_enabled
        and train_ref_idx.size > 0
        and train_successor_idx.size > 0
        and init_train_idx_np.size > 0
    )
    objective_ref_idx_np = (
        train_ref_idx
        if split_training
        else np.arange(Z_ref_np.shape[0], dtype=np.int64)
    )
    objective_init_idx_np = (
        init_train_idx_np
        if split_training
        else np.arange(Z_init_np.shape[0], dtype=np.int64)
    )
    objective_successor_idx_np = (
        train_successor_idx
        if split_training
        else np.arange(successor_row_index.shape[0], dtype=np.int64)
    )
    objective_init_probs_np = _probability_subset(init_probs, objective_init_idx_np)
    valid_init_probs_np = (
        _probability_subset(init_probs, init_valid_idx_np)
        if init_valid_idx_np.size
        else init_probs
    )
    objective_ref_idx = torch.as_tensor(
        objective_ref_idx_np, dtype=torch.long, device=device
    )
    objective_init_idx = torch.as_tensor(
        objective_init_idx_np, dtype=torch.long, device=device
    )
    objective_init_w = torch.as_tensor(
        objective_init_probs_np.astype(np.float32, copy=False),
        dtype=torch.float32,
        device=device,
    )
    objective_successor_pos = torch.as_tensor(
        objective_successor_idx_np, dtype=torch.long, device=device
    )
    rng = np.random.default_rng(int(cfg.seed) + 91_337)
    neural_batch_size = (
        None if cfg.neural_batch_size is None else int(cfg.neural_batch_size)
    )
    terminal_sensitive_gauge = bool(
        np.any(
            np.abs(np.asarray(continuation_plus, dtype=np.float64).reshape(-1) - 1.0)
            > 1e-12
        )
    )
    requested_gauge_fix = str(cfg.neural_variational_gauge_fix)
    active_gauge_fix = requested_gauge_fix
    if requested_gauge_fix == "auto":
        active_gauge_fix = "none" if terminal_sensitive_gauge else "batch"
    if not variational_log_partition:
        active_gauge_fix = "none"
    elif active_gauge_fix not in {"none", "batch", "full"}:
        active_gauge_fix = "none"
    final_linear_layer = _torch_final_linear_layer(torch, model)

    def validation_loss(
        scores_ref: Array,
        scores_init: Array,
        scores_plus: Array,
        *,
        target_weights_ref: Array,
    ) -> float:
        if not early_stop_enabled:
            return float("nan")
        return _projection_loss_from_scores(
            scores_ref=np.asarray(scores_ref, dtype=np.float64).reshape(-1)[valid_idx],
            scores_init=np.asarray(scores_init, dtype=np.float64).reshape(-1)[
                init_valid_idx_np
            ],
            scores_plus=np.asarray(scores_plus, dtype=np.float64).reshape(-1)[
                valid_successor_idx
            ],
            init_probs=valid_init_probs_np,
            weights_ref=target_weights_ref,
            successor_row_index=successor_row_index[valid_successor_idx],
            continuation_plus=continuation_plus[valid_successor_idx],
            gamma=gamma,
        )

    def sample_uniform(pool: Array) -> Array:
        return _sample_batch_indices(pool, batch_size=neural_batch_size, rng=rng)

    def sample_initial() -> tuple[Array, Optional[Array]]:
        if neural_batch_size is None or neural_batch_size >= objective_init_idx_np.size:
            return objective_init_idx_np, objective_init_probs_np
        chosen = rng.choice(
            objective_init_idx_np,
            size=int(neural_batch_size),
            replace=True,
            p=objective_init_probs_np,
        )
        return np.asarray(chosen, dtype=np.int64), None

    for iteration in range(int(cfg.num_iterations)):
        previous_weights_ref = weights_ref.copy()
        target_weights_train = (
            _subset_normalized_weight_vector(scores_ref_np, cfg, objective_ref_idx_np)
            if split_training
            else weights_ref
        )
        target_weights_valid = (
            _subset_normalized_weight_vector(
                scores_ref_np, cfg, np.asarray(valid_idx, dtype=np.int64)
            )
            if early_stop_enabled
            else weights_ref
        )
        current_w = torch.as_tensor(
            target_weights_train.astype(np.float32, copy=False),
            dtype=torch.float32,
            device=device,
        )
        old_objective = _projection_loss_from_scores(
            scores_ref=scores_ref_np,
            scores_init=scores_init_np,
            scores_plus=scores_plus_np,
            init_probs=init_probs,
            weights_ref=weights_ref,
            successor_row_index=successor_row_index,
            continuation_plus=continuation_plus,
            gamma=gamma,
        )
        old_tikhonov_value = _score_tikhonov_value_from_scores(
            (
                (scores_ref_np, None),
                (scores_init_np, init_probs),
                (scores_plus_np, None),
            )
        )
        early_stop_active = bool(early_stop_enabled)
        old_validation_loss = validation_loss(
            scores_ref_np,
            scores_init_np,
            scores_plus_np,
            target_weights_ref=target_weights_valid,
        )
        before_state = deepcopy(model.state_dict())
        before_optimizer_state = deepcopy(optimizer.state_dict())
        before_variational_a = variational_a.detach().clone()
        candidate_objective = float("nan")
        grad_norm = float("nan")
        gauge_shift_last = 0.0
        for _step in range(int(cfg.optimizer_steps)):
            optimizer.zero_grad(set_to_none=True)
            if variational_log_partition:
                ref_batch_np = sample_uniform(objective_ref_idx_np)
                init_batch_np, init_batch_probs_np = sample_initial()
                successor_batch_np = sample_uniform(objective_successor_idx_np)
                ref_batch = torch.as_tensor(
                    ref_batch_np, dtype=torch.long, device=device
                )
                init_batch = torch.as_tensor(
                    init_batch_np, dtype=torch.long, device=device
                )
                successor_pos = torch.as_tensor(
                    successor_batch_np, dtype=torch.long, device=device
                )
                scores_ref_obj = model(Z_ref[ref_batch]).reshape(-1)
                scores_init_obj = model(Z_init[init_batch]).reshape(-1)
                scores_plus_obj = model(Z_plus[successor_pos]).reshape(-1)
                if init_batch_probs_np is None:
                    init_term = torch.mean(scores_init_obj)
                else:
                    init_batch_w = torch.as_tensor(
                        init_batch_probs_np.astype(np.float32, copy=False),
                        dtype=torch.float32,
                        device=device,
                    )
                    init_term = torch.sum(init_batch_w * scores_init_obj)
                successor_w = current_w[successor_idx[successor_pos]]
                successor_denom = torch.clamp(torch.sum(successor_w), min=1e-12)
                successor_term = (
                    torch.sum(
                        successor_w * continuation_t[successor_pos] * scores_plus_obj
                    )
                    / successor_denom
                )
                centered_for_exp = torch.clamp(
                    scores_ref_obj - variational_a, min=-60.0, max=60.0
                )
                objective = (
                    variational_a
                    - 1.0
                    + torch.mean(torch.exp(centered_for_exp))
                    - (1.0 - gamma) * init_term
                    - gamma * successor_term
                )
            else:
                scores_ref = model(Z_ref).reshape(-1)
                scores_init = model(Z_init).reshape(-1)
                scores_plus = model(Z_plus).reshape(-1)
                scores_ref_obj = scores_ref[objective_ref_idx]
                scores_init_obj = scores_init[objective_init_idx]
                successor_pos = objective_successor_pos
                successor_w = current_w[successor_idx[successor_pos]]
                successor_denom = torch.clamp(torch.mean(successor_w), min=1e-12)
                successor_term = (
                    torch.mean(
                        successor_w
                        * continuation_t[successor_pos]
                        * scores_plus[successor_pos]
                    )
                    / successor_denom
                )
                objective = (
                    torch.logsumexp(scores_ref_obj, dim=0)
                    - np.log(max(int(scores_ref_obj.numel()), 1))
                    - (1.0 - gamma) * torch.sum(objective_init_w * scores_init_obj)
                    - gamma * successor_term
                )
            if float(cfg.score_tikhonov_penalty) > 0.0:
                if variational_log_partition:
                    if init_batch_probs_np is None:
                        init_penalty = torch.mean(scores_init_obj * scores_init_obj)
                    else:
                        init_penalty = torch.sum(
                            init_batch_w * scores_init_obj * scores_init_obj
                        )
                    score_penalty = (
                        torch.mean(scores_ref_obj * scores_ref_obj)
                        + init_penalty
                        + torch.mean(scores_plus_obj * scores_plus_obj)
                    )
                else:
                    score_penalty = (
                        torch.mean(scores_ref_obj * scores_ref_obj)
                        + torch.sum(
                            objective_init_w * scores_init_obj * scores_init_obj
                        )
                        + torch.mean(
                            scores_plus[successor_pos] * scores_plus[successor_pos]
                        )
                    )
                objective = (
                    objective + 0.5 * float(cfg.score_tikhonov_penalty) * score_penalty
                )
            if float(cfg.neural_weight_decay) > 0.0:
                penalty = torch.zeros((), dtype=torch.float32, device=device)
                for param in model.parameters():
                    penalty = penalty + torch.sum(param * param)
                objective = objective + 0.5 * float(cfg.neural_weight_decay) * penalty
            objective.backward()
            grad_norm = _torch_grad_norm(torch, grad_parameters)
            if cfg.neural_grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    grad_parameters, float(cfg.neural_grad_clip_norm)
                )
            optimizer.step()
            if variational_log_partition and active_gauge_fix != "none":
                center_idx = (
                    ref_batch if active_gauge_fix == "batch" else objective_ref_idx
                )
                gauge_shift_last = _recenter_torch_score_gauge(
                    torch=torch,
                    model=model,
                    final_linear_layer=final_linear_layer,
                    variational_a=variational_a,
                    reference_batch=Z_ref[center_idx],
                )
            candidate_objective = float(objective.detach().cpu().item())

        with torch.no_grad():
            candidate_scores_ref = (
                model(Z_ref).reshape(-1).detach().cpu().numpy().astype(np.float64)
            )
            candidate_scores_init = (
                model(Z_init).reshape(-1).detach().cpu().numpy().astype(np.float64)
            )
            candidate_scores_plus = (
                model(Z_plus).reshape(-1).detach().cpu().numpy().astype(np.float64)
            )
        candidate_objective = _projection_loss_from_scores(
            scores_ref=candidate_scores_ref,
            scores_init=candidate_scores_init,
            scores_plus=candidate_scores_plus,
            init_probs=init_probs,
            weights_ref=weights_ref,
            successor_row_index=successor_row_index,
            continuation_plus=continuation_plus,
            gamma=gamma,
        )
        candidate_tikhonov_value = _score_tikhonov_value_from_scores(
            (
                (candidate_scores_ref, None),
                (candidate_scores_init, init_probs),
                (candidate_scores_plus, None),
            )
        )
        candidate_validation_loss = validation_loss(
            candidate_scores_ref,
            candidate_scores_init,
            candidate_scores_plus,
            target_weights_ref=target_weights_valid,
        )
        validation_improved = (
            not early_stop_active
            or candidate_validation_loss
            <= old_validation_loss - float(cfg.min_improvement)
        )
        (
            _,
            candidate_clip_adjustment,
            candidate_raw_log_ratio_ref,
            candidate_log_ratio_ref,
        ) = _normalized_log_ratios(
            candidate_scores_ref,
            cfg,
        )
        candidate_weights_ref = np.exp(candidate_log_ratio_ref)
        candidate_cap_fraction = (
            0.0
            if cfg.logit_clip is None
            else float(
                np.mean(np.abs(candidate_raw_log_ratio_ref) >= float(cfg.logit_clip))
            )
        )
        candidate_ess_fraction = _ess_fraction(candidate_weights_ref)
        candidate_step_kl = _empirical_weight_kl(
            candidate_weights_ref, previous_weights_ref, cfg
        )
        candidate_step_reverse_kl = _empirical_weight_kl(
            previous_weights_ref, candidate_weights_ref, cfg
        )
        candidate_step_l1 = float(
            np.mean(np.abs(candidate_weights_ref - previous_weights_ref))
        )
        candidate_valid_kl_to_uniform = float("nan")
        candidate_valid_step_kl = float("nan")
        candidate_valid_step_reverse_kl = float("nan")
        candidate_valid_step_l1 = float("nan")
        if early_stop_active:
            _, _, candidate_valid_raw_log_ratio, candidate_valid_log_ratio = (
                _normalized_log_ratios(
                    np.asarray(candidate_scores_ref, dtype=np.float64).reshape(-1)[
                        valid_idx
                    ],
                    cfg,
                )
            )
            candidate_valid_weights = np.exp(candidate_valid_log_ratio)
            guardrail_ess_fraction = _ess_fraction(candidate_valid_weights)
            guardrail_cap_fraction = (
                0.0
                if cfg.logit_clip is None
                else float(
                    np.mean(
                        np.abs(candidate_valid_raw_log_ratio) >= float(cfg.logit_clip)
                    )
                )
            )
            candidate_valid_kl_to_uniform = _empirical_weight_kl(
                candidate_valid_weights,
                np.ones_like(candidate_valid_weights),
                cfg,
            )
            previous_valid_weights = target_weights_valid[
                np.asarray(valid_idx, dtype=np.int64)
            ]
            candidate_valid_step_kl = _empirical_weight_kl(
                candidate_valid_weights, previous_valid_weights, cfg
            )
            candidate_valid_step_reverse_kl = _empirical_weight_kl(
                previous_valid_weights, candidate_valid_weights, cfg
            )
            candidate_valid_step_l1 = float(
                np.mean(
                    np.abs(
                        _normalize_mean_one(candidate_valid_weights, cfg)
                        - _normalize_mean_one(previous_valid_weights, cfg)
                    )
                )
            )
        else:
            guardrail_ess_fraction = candidate_ess_fraction
            guardrail_cap_fraction = candidate_cap_fraction
        ess_guardrail_pass = cfg.min_ess_fraction is None or (
            np.isfinite(guardrail_ess_fraction)
            and guardrail_ess_fraction >= float(cfg.min_ess_fraction)
        )
        cap_guardrail_pass = (
            cfg.max_logit_cap_fraction is None
            or guardrail_cap_fraction <= float(cfg.max_logit_cap_fraction)
        )
        guardrail_pass = bool(ess_guardrail_pass and cap_guardrail_pass)
        warmup_accept = bool(
            early_stop_active and accepted_count < int(cfg.validation_warmup_iterations)
        )
        accepted = bool((validation_improved or warmup_accept) and guardrail_pass)
        if accepted:
            scores_ref_np = candidate_scores_ref
            scores_init_np = candidate_scores_init
            scores_plus_np = candidate_scores_plus
            log_partition = _logmeanexp(scores_ref_np)
            clip_adjustment = candidate_clip_adjustment
            log_ratio_ref_raw = candidate_raw_log_ratio_ref
            log_ratio_ref = candidate_log_ratio_ref
            weights_ref = candidate_weights_ref
            accepted_count += 1
            if warmup_accept and not validation_improved:
                validation_warmup_accepts += 1
            patience = 0
            selected_objective = candidate_objective
            selected_tikhonov_value = candidate_tikhonov_value
            selected_validation_loss = candidate_validation_loss
        else:
            model.load_state_dict(before_state)
            with torch.no_grad():
                variational_a.copy_(before_variational_a)
            optimizer.load_state_dict(before_optimizer_state)
            model.eval()
            patience += 1
            selected_objective = old_objective
            selected_tikhonov_value = old_tikhonov_value
            selected_validation_loss = old_validation_loss

        row = _history_row(
            iteration=iteration,
            objective=selected_objective,
            score_tikhonov_value=selected_tikhonov_value,
            score_regularized_objective=selected_objective
            + float(cfg.score_tikhonov_penalty) * selected_tikhonov_value,
            grad_norm=grad_norm,
            log_partition=log_partition,
            clip_log_partition_adjustment=clip_adjustment,
            weights_ref=weights_ref,
            previous_weights_ref=previous_weights_ref,
            log_ratio_ref=log_ratio_ref,
            raw_log_ratio_ref=log_ratio_ref_raw,
            train_idx=train_idx,
            valid_idx=valid_idx,
            target_phi=np.empty(0, dtype=np.float64),
            cfg=cfg,
        )
        row.update(
            {
                "early_stopping": bool(early_stop_active),
                "split_training": bool(split_training),
                "split_target_normalization": bool(split_training),
                "objective_train_rows": int(objective_ref_idx_np.shape[0]),
                "objective_init_rows": int(objective_init_idx_np.shape[0]),
                "objective_successor_rows": int(objective_successor_idx_np.shape[0]),
                "validation_init_rows": int(init_valid_idx_np.shape[0])
                if early_stop_active
                else 0,
                "accepted": bool(accepted),
                "candidate_objective": float(candidate_objective),
                "validation_loss_old": float(old_validation_loss),
                "candidate_validation_loss": float(candidate_validation_loss),
                "selected_validation_loss": float(selected_validation_loss),
                "validation_loss": float(selected_validation_loss),
                "validation_improved": bool(validation_improved),
                "validation_warmup_accept": bool(
                    warmup_accept and not validation_improved
                ),
                "validation_warmup_iterations": int(cfg.validation_warmup_iterations),
                "guardrail_pass": bool(guardrail_pass),
                "ess_guardrail_pass": bool(ess_guardrail_pass),
                "cap_guardrail_pass": bool(cap_guardrail_pass),
                "candidate_ess_fraction": float(candidate_ess_fraction),
                "candidate_logit_cap_fraction": float(candidate_cap_fraction),
                "candidate_weight_step_kl": float(candidate_step_kl),
                "candidate_weight_step_reverse_kl": float(candidate_step_reverse_kl),
                "candidate_weight_step_l1": float(candidate_step_l1),
                "candidate_valid_weight_kl_to_uniform": float(
                    candidate_valid_kl_to_uniform
                ),
                "candidate_valid_weight_step_kl": float(candidate_valid_step_kl),
                "candidate_valid_weight_step_reverse_kl": float(
                    candidate_valid_step_reverse_kl
                ),
                "candidate_valid_weight_step_l1": float(candidate_valid_step_l1),
                "guardrail_ess_fraction": float(guardrail_ess_fraction),
                "guardrail_logit_cap_fraction": float(guardrail_cap_fraction),
                "candidate_weight_max": float(np.max(candidate_weights_ref)),
                "neural_log_partition_mode": str(cfg.neural_log_partition_mode),
                "neural_batch_size": ""
                if cfg.neural_batch_size is None
                else int(cfg.neural_batch_size),
                "neural_variational_a": float(variational_a.detach().cpu().item()),
                "neural_variational_gauge_fix": str(cfg.neural_variational_gauge_fix),
                "neural_variational_gauge_active": bool(active_gauge_fix != "none"),
                "neural_variational_gauge_mode": str(active_gauge_fix),
                "neural_variational_gauge_terminal_sensitive": bool(
                    terminal_sensitive_gauge
                ),
                "neural_variational_gauge_shift": float(gauge_shift_last),
                "patience": int(patience),
                "accepted_count": int(accepted_count),
            }
        )
        if not accepted and early_stop_active and patience >= int(cfg.patience):
            stopped_early = True
            stop_iter = int(iteration)
            stop_reason = "validation_projection_loss"
            row["stopped_early"] = True
            row["stop_reason"] = stop_reason
            history.append(row)
            break
        row["stopped_early"] = False
        row["stop_reason"] = ""
        history.append(row)

    if split_training:
        selection_history = list(history)
        for row in selection_history:
            row["stage"] = "selection"
        selected_iterations = int(accepted_count)
        full_mean, full_scale = _fit_standardizer(X_ref)
        full_ref_idx = np.arange(X_ref.shape[0], dtype=np.int64)
        full_init_idx = np.arange(X_init.shape[0], dtype=np.int64)
        refit_cfg = replace(
            cfg,
            num_iterations=selected_iterations,
            early_stopping=False,
            validation_fraction=0.0,
        )
        refit = _fit_neural_backend_loop(
            cfg=refit_cfg,
            gamma=gamma,
            X_ref=X_ref,
            X_init=X_init,
            X_plus=X_plus,
            mean=full_mean,
            scale=full_scale,
            init_probs=init_probs,
            successor_row_index=successor_row_index,
            continuation_plus=continuation_plus,
            train_idx=full_ref_idx,
            valid_idx=np.array([], dtype=np.int64),
            init_train_idx=full_init_idx,
            init_valid_idx=np.array([], dtype=np.int64),
        )
        refit_history = list(refit["history"])
        for row in refit_history:
            row["stage"] = "refit"
            row["refit_after_validation"] = True
            row["refit_num_iterations"] = selected_iterations
            row["selection_accepted_count"] = int(accepted_count)
            row["selection_iterations_completed"] = int(len(selection_history))
            row["selection_stopped_early"] = bool(stopped_early)
            row["selection_stop_iter"] = "" if stop_iter is None else int(stop_iter)
            row["selection_stop_reason"] = stop_reason or ""
        selection_valid_losses = [
            float(row["validation_loss"])
            for row in selection_history
            if "validation_loss" in row and np.isfinite(float(row["validation_loss"]))
        ]
        selection_final = selection_history[-1] if selection_history else {}
        refit["history"] = selection_history + refit_history
        refit["fit_payload"].update(
            {
                "refit_after_validation": True,
                "refit_num_iterations": int(selected_iterations),
                "selection_accepted_count": int(accepted_count),
                "selection_iterations_completed": int(len(selection_history)),
                "selection_stopped_early": bool(stopped_early),
                "selection_stop_iter": "" if stop_iter is None else int(stop_iter),
                "selection_stop_reason": stop_reason or "",
                "selection_best_valid_loss": (
                    float(np.min(selection_valid_losses))
                    if selection_valid_losses
                    else float("nan")
                ),
                "selection_split_training": bool(
                    selection_final.get("split_training", False)
                ),
                "selection_objective_train_rows": int(
                    selection_final.get("objective_train_rows", 0)
                ),
                "selection_objective_init_rows": int(
                    selection_final.get("objective_init_rows", 0)
                ),
                "selection_objective_successor_rows": int(
                    selection_final.get("objective_successor_rows", 0)
                ),
                "selection_validation_init_rows": int(
                    selection_final.get("validation_init_rows", 0)
                ),
                "selection_history": selection_history,
            }
        )
        return refit

    state_dict = {
        name: tensor.detach().cpu().numpy().astype(np.float32, copy=True)
        for name, tensor in model.state_dict().items()
    }
    return {
        "coef": np.empty(0, dtype=np.float64),
        "boosters": (),
        "neural_state_dict": state_dict,
        "feature_dim": Z_ref_np.shape[1],
        "log_partition": log_partition,
        "clip_log_partition_adjustment": clip_adjustment,
        "log_ratio_ref_raw": log_ratio_ref_raw,
        "log_ratio_ref": log_ratio_ref,
        "weights_ref": weights_ref,
        "history": history,
        "feature_mean": mean,
        "feature_scale": scale,
        "fit_payload": {
            "neural_hidden_dims": tuple(int(width) for width in cfg.neural_hidden_dims),
            "neural_log_partition_mode": str(cfg.neural_log_partition_mode),
            "neural_batch_size": ""
            if cfg.neural_batch_size is None
            else int(cfg.neural_batch_size),
            "neural_variational_a": float(variational_a.detach().cpu().item()),
            "neural_variational_gauge_fix": str(cfg.neural_variational_gauge_fix),
            "neural_variational_gauge_active": bool(active_gauge_fix != "none"),
            "neural_variational_gauge_mode": str(active_gauge_fix),
            "neural_variational_gauge_terminal_sensitive": bool(
                terminal_sensitive_gauge
            ),
            "stopped_early": bool(stopped_early),
            "stop_iter": stop_iter,
            "stop_reason": stop_reason or "",
            "accepted_count": int(accepted_count),
            "validation_warmup_accepts": int(validation_warmup_accepts),
        },
    }


def _successor_weighted_features(
    *,
    Phi_plus: Array,
    weights_ref: Array,
    successor_row_index: Array,
    continuation_plus: Array,
) -> Array:
    w = np.asarray(weights_ref, dtype=np.float64).reshape(-1)
    idx = np.asarray(successor_row_index, dtype=np.int64).reshape(-1)
    if idx.shape[0] != Phi_plus.shape[0]:
        raise ValueError("successor_row_index must match successor features.")
    successor_weight = w[idx]
    denom = float(np.mean(successor_weight))
    if not np.isfinite(denom) or denom <= 0.0:
        raise FloatingPointError(
            "successor weights must have positive finite empirical mean."
        )
    weighted = successor_weight * np.asarray(
        continuation_plus, dtype=np.float64
    ).reshape(-1)
    return np.mean(Phi_plus * weighted.reshape(-1, 1), axis=0) / denom


def _fit_projection_adam(
    *,
    coef: Array,
    Phi_ref: Array,
    target_phi: Array,
    learning_rate: float,
    steps: int,
    l2_penalty: float,
    score_tikhonov_penalty: float = 0.0,
    tikhonov_blocks: Sequence[tuple[Array, Optional[Array]]] = (),
) -> tuple[Array, float, float]:
    theta = np.asarray(coef, dtype=np.float64).copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    objective = float("nan")
    grad_norm = float("nan")
    for step in range(1, int(steps) + 1):
        objective, grad = _projection_objective_and_grad(
            theta,
            Phi_ref=Phi_ref,
            target_phi=target_phi,
            l2_penalty=l2_penalty,
            score_tikhonov_penalty=score_tikhonov_penalty,
            tikhonov_blocks=tikhonov_blocks,
        )
        grad_norm = float(np.linalg.norm(grad))
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        theta -= float(learning_rate) * m_hat / (np.sqrt(v_hat) + eps)
        if grad_norm <= 1e-8:
            break
    objective, grad = _projection_objective_and_grad(
        theta,
        Phi_ref=Phi_ref,
        target_phi=target_phi,
        l2_penalty=l2_penalty,
        score_tikhonov_penalty=score_tikhonov_penalty,
        tikhonov_blocks=tikhonov_blocks,
    )
    grad_norm = float(np.linalg.norm(grad))
    return theta, objective, grad_norm


def _projection_objective_and_grad(
    coef: Array,
    *,
    Phi_ref: Array,
    target_phi: Array,
    l2_penalty: float,
    score_tikhonov_penalty: float = 0.0,
    tikhonov_blocks: Sequence[tuple[Array, Optional[Array]]] = (),
) -> tuple[float, Array]:
    theta = np.asarray(coef, dtype=np.float64).reshape(-1)
    scores = Phi_ref @ theta
    log_a = _logmeanexp(scores)
    probs = _softmax_mean(scores)
    grad = probs @ Phi_ref - target_phi
    if l2_penalty > 0.0:
        grad = grad + float(l2_penalty) * theta
    objective = float(
        log_a
        - np.dot(theta, target_phi)
        + 0.5 * float(l2_penalty) * np.dot(theta, theta)
    )
    if score_tikhonov_penalty > 0.0:
        penalty, penalty_grad = _score_tikhonov_value_grad(theta, tikhonov_blocks)
        objective += float(score_tikhonov_penalty) * penalty
        grad = grad + float(score_tikhonov_penalty) * penalty_grad
    return objective, grad


def _score_tikhonov_value_grad(
    theta: Array,
    blocks: Sequence[tuple[Array, Optional[Array]]],
) -> tuple[float, Array]:
    coef = np.asarray(theta, dtype=np.float64).reshape(-1)
    value = 0.0
    grad = np.zeros_like(coef)
    for phi_raw, weight_raw in blocks:
        phi = np.asarray(phi_raw, dtype=np.float64)
        if phi.ndim != 2 or phi.shape[1] != coef.shape[0] or phi.shape[0] == 0:
            raise ValueError(
                "Tikhonov feature blocks must be nonempty 2D arrays matching coef."
            )
        scores = phi @ coef
        if weight_raw is None:
            weights = np.full(phi.shape[0], 1.0 / phi.shape[0], dtype=np.float64)
        else:
            weights = _normalize_weights(
                weight_raw, n_rows=phi.shape[0], name="score_tikhonov_weights"
            )
        value += 0.5 * float(np.sum(weights * scores * scores))
        grad += phi.T @ (weights * scores)
    return float(value), grad


def _score_tikhonov_value_from_scores(
    blocks: Sequence[tuple[Array, Optional[Array]]],
) -> float:
    value = 0.0
    for scores_raw, weight_raw in blocks:
        scores = np.asarray(scores_raw, dtype=np.float64).reshape(-1)
        if scores.size == 0:
            raise ValueError("Tikhonov score blocks must be nonempty.")
        if weight_raw is None:
            weights = np.full(scores.shape[0], 1.0 / scores.shape[0], dtype=np.float64)
        else:
            weights = _normalize_weights(
                weight_raw, n_rows=scores.shape[0], name="score_tikhonov_weights"
            )
        value += 0.5 * float(np.sum(weights * scores * scores))
    return float(value)


def _projection_loss_from_scores(
    *,
    scores_ref: Array,
    scores_init: Array,
    scores_plus: Array,
    init_probs: Array,
    weights_ref: Array,
    successor_row_index: Array,
    continuation_plus: Array,
    gamma: float,
) -> float:
    ref = np.asarray(scores_ref, dtype=np.float64).reshape(-1)
    init = np.asarray(scores_init, dtype=np.float64).reshape(-1)
    plus = np.asarray(scores_plus, dtype=np.float64).reshape(-1)
    init_w = np.asarray(init_probs, dtype=np.float64).reshape(-1)
    if init.shape[0] != init_w.shape[0]:
        raise ValueError("scores_init and init_probs must have the same length.")
    idx = np.asarray(successor_row_index, dtype=np.int64).reshape(-1)
    if plus.shape[0] != idx.shape[0]:
        raise ValueError(
            "scores_plus and successor_row_index must have the same length."
        )
    successor_weight = np.asarray(weights_ref, dtype=np.float64).reshape(-1)[idx]
    successor_weight_mean = float(np.mean(successor_weight))
    if not np.isfinite(successor_weight_mean) or successor_weight_mean <= 0.0:
        raise FloatingPointError(
            "successor weights must have positive finite empirical mean."
        )
    successor_weighted_score = (
        successor_weight
        * np.asarray(continuation_plus, dtype=np.float64).reshape(-1)
        * plus
    )
    return float(
        _logmeanexp(ref)
        - (1.0 - float(gamma)) * np.dot(init_w, init)
        - float(gamma) * np.mean(successor_weighted_score) / successor_weight_mean
    )


def _logmeanexp(scores: Array) -> float:
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("scores must be nonempty.")
    m = float(np.max(arr))
    return m + float(np.log(np.mean(np.exp(arr - m))))


def _normalized_log_ratios(
    scores: Array, cfg: KLFORIConfig
) -> tuple[float, float, Array, Array]:
    log_partition = _logmeanexp(scores)
    raw_log_ratio = np.asarray(scores, dtype=np.float64).reshape(-1) - float(
        log_partition
    )
    if cfg.logit_clip is None:
        return float(log_partition), 0.0, raw_log_ratio, raw_log_ratio
    clipped = np.clip(raw_log_ratio, -float(cfg.logit_clip), float(cfg.logit_clip))
    clip_adjustment = _logmeanexp(clipped)
    return (
        float(log_partition),
        float(clip_adjustment),
        raw_log_ratio,
        clipped - float(clip_adjustment),
    )


def _softmax_mean(scores: Array) -> Array:
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    m = float(np.max(arr))
    exp = np.exp(arr - m)
    denom = float(np.sum(exp))
    if not np.isfinite(denom) or denom <= 0.0:
        raise FloatingPointError("invalid log-partition normalization.")
    return exp / denom


def _weighted_norm(values: Array, sample_weight: Array) -> float:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if value.shape[0] != weight.shape[0]:
        raise ValueError("values and sample_weight must have the same length.")
    mass = float(np.sum(weight))
    if mass <= 0.0:
        return float("nan")
    return float(np.sqrt(np.sum(weight * value * value) / mass))


def _predict_lgbm_tree(tree: Any, x: Array) -> Array:
    if hasattr(tree, "booster_"):
        return np.asarray(
            tree.booster_.predict(np.asarray(x, dtype=np.float64)), dtype=np.float64
        )
    return np.asarray(tree.predict(np.asarray(x, dtype=np.float64)), dtype=np.float64)


def _ess_fraction(weights: Array) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    denom = float(np.sum(w * w))
    if w.size == 0 or denom <= 0.0:
        return float("nan")
    return float(np.sum(w) ** 2 / (w.size * denom))


def _normalize_mean_one(weights: Array, cfg: KLFORIConfig) -> Array:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size == 0:
        return w
    eps = float(cfg.normalize_eps)
    w = np.maximum(w, eps)
    mean = float(np.mean(w))
    if not np.isfinite(mean) or mean <= 0.0:
        return np.full(w.shape, 1.0, dtype=np.float64)
    return w / mean


def _empirical_weight_kl(
    numerator_weights: Array, denominator_weights: Array, cfg: KLFORIConfig
) -> float:
    p = _normalize_mean_one(numerator_weights, cfg)
    q = _normalize_mean_one(denominator_weights, cfg)
    if p.shape != q.shape or p.size == 0:
        return float("nan")
    eps = float(cfg.normalize_eps)
    return float(np.mean(p * (np.log(np.maximum(p, eps)) - np.log(np.maximum(q, eps)))))


def _history_row(
    *,
    iteration: int,
    objective: float,
    grad_norm: float,
    log_partition: float,
    clip_log_partition_adjustment: float,
    weights_ref: Array,
    log_ratio_ref: Array,
    raw_log_ratio_ref: Array,
    train_idx: Array,
    valid_idx: Array,
    target_phi: Array,
    cfg: KLFORIConfig,
    score_tikhonov_value: Optional[float] = None,
    score_regularized_objective: Optional[float] = None,
    training_objective: Optional[float] = None,
    previous_weights_ref: Optional[Array] = None,
) -> dict[str, Any]:
    del target_phi
    cap = cfg.logit_clip
    cap_hits = (
        0.0 if cap is None else float(np.mean(np.abs(raw_log_ratio_ref) >= float(cap)))
    )
    ones = np.ones_like(weights_ref, dtype=np.float64)
    row = {
        "iteration": int(iteration),
        "backend": cfg.backend,
        "objective": float(objective),
        "grad_norm": float(grad_norm),
        "log_partition": float(log_partition),
        "clip_log_partition_adjustment": float(clip_log_partition_adjustment),
        "empirical_mass": float(np.mean(weights_ref)),
        "ess_fraction": _ess_fraction(weights_ref),
        "weight_mean": float(np.mean(weights_ref)),
        "weight_min": float(np.min(weights_ref)),
        "weight_max": float(np.max(weights_ref)),
        "weight_q95": float(np.quantile(weights_ref, 0.95)),
        "weight_q99": float(np.quantile(weights_ref, 0.99)),
        "logit_cap_fraction": cap_hits,
        "weight_kl_to_uniform": _empirical_weight_kl(weights_ref, ones, cfg),
        "weight_reverse_kl_to_uniform": _empirical_weight_kl(ones, weights_ref, cfg),
        "weight_step_kl": float("nan"),
        "weight_step_reverse_kl": float("nan"),
        "weight_step_l1": float("nan"),
    }
    if previous_weights_ref is not None:
        row["weight_step_kl"] = _empirical_weight_kl(
            weights_ref, previous_weights_ref, cfg
        )
        row["weight_step_reverse_kl"] = _empirical_weight_kl(
            previous_weights_ref, weights_ref, cfg
        )
        row["weight_step_l1"] = float(
            np.mean(
                np.abs(
                    _normalize_mean_one(weights_ref, cfg)
                    - _normalize_mean_one(previous_weights_ref, cfg)
                )
            )
        )
    if score_tikhonov_value is not None:
        row["score_tikhonov_value"] = float(score_tikhonov_value)
    if score_regularized_objective is not None:
        row["score_regularized_objective"] = float(score_regularized_objective)
    if training_objective is not None:
        row["training_objective"] = float(training_objective)
    if valid_idx.size:
        valid_weights = weights_ref[valid_idx]
        row["train_weight_mean"] = float(np.mean(weights_ref[train_idx]))
        row["valid_weight_mean"] = float(np.mean(valid_weights))
        row["heldout_kl_loss_proxy"] = float(_logmeanexp(log_ratio_ref[valid_idx]))
        row["valid_weight_kl_to_uniform"] = _empirical_weight_kl(
            valid_weights,
            np.ones_like(valid_weights),
            cfg,
        )
        row["valid_weight_reverse_kl_to_uniform"] = _empirical_weight_kl(
            np.ones_like(valid_weights),
            valid_weights,
            cfg,
        )
        row["valid_weight_step_kl"] = float("nan")
        row["valid_weight_step_reverse_kl"] = float("nan")
        row["valid_weight_step_l1"] = float("nan")
        if previous_weights_ref is not None:
            previous_valid_weights = previous_weights_ref[valid_idx]
            row["valid_weight_step_kl"] = _empirical_weight_kl(
                valid_weights, previous_valid_weights, cfg
            )
            row["valid_weight_step_reverse_kl"] = _empirical_weight_kl(
                previous_valid_weights,
                valid_weights,
                cfg,
            )
            row["valid_weight_step_l1"] = float(
                np.mean(
                    np.abs(
                        _normalize_mean_one(valid_weights, cfg)
                        - _normalize_mean_one(previous_valid_weights, cfg)
                    )
                )
            )
    else:
        row["heldout_kl_loss_proxy"] = float("nan")
        row["valid_weight_kl_to_uniform"] = float("nan")
        row["valid_weight_reverse_kl_to_uniform"] = float("nan")
        row["valid_weight_step_kl"] = float("nan")
        row["valid_weight_step_reverse_kl"] = float("nan")
        row["valid_weight_step_l1"] = float("nan")
    if cfg.selection_objective is not None:
        try:
            selection_value = float(
                cfg.selection_objective(
                    np.asarray(weights_ref, dtype=np.float64).reshape(-1)
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"KL-FORI selection_objective failed at outer iteration {int(iteration)}."
            ) from exc
        row["selection_objective_value"] = selection_value
        row["selection_objective_name"] = str(cfg.selection_objective_name or "custom")
    return row


def _diagnostics(
    *,
    cfg: KLFORIConfig,
    gamma: float,
    history: list[dict[str, Any]],
    weights_ref: Array,
    log_ratio_ref: Array,
    raw_log_ratio_ref: Array,
    clip_log_partition_adjustment: float,
    successor_source: str,
    initial_action_source: str,
    successor_row_index: Array,
    continuation_plus: Array,
    initial_rows: int,
    reference_rows: int,
    feature_dim: int,
) -> dict[str, Any]:
    final = history[-1] if history else {}
    final_stage = str(final.get("stage", ""))
    refit_rows = [row for row in history if str(row.get("stage", "")) == "refit"]
    selection_rows = [
        row for row in history if str(row.get("stage", "")) == "selection"
    ]
    valid_losses = [
        float(row["validation_loss"])
        for row in history
        if "validation_loss" in row and np.isfinite(float(row["validation_loss"]))
    ]
    selection_values = [
        (int(row["iteration"]), float(row["selection_objective_value"]))
        for row in history
        if "selection_objective_value" in row
        and np.isfinite(float(row["selection_objective_value"]))
    ]
    best_selection_iteration = ""
    best_selection_value = float("nan")
    if selection_values:
        best_selection_iteration, best_selection_value = min(
            selection_values, key=lambda item: item[1]
        )
    final_stage_rows = refit_rows if refit_rows else history
    accepted_rows = [
        row for row in final_stage_rows if bool(row.get("accepted", False))
    ]
    stop_rows = [row for row in history if bool(row.get("stopped_early", False))]
    early_stop_rows = [row for row in history if bool(row.get("early_stopping", False))]
    cap = cfg.logit_clip
    return {
        "algorithm": "kl_fori",
        "backend": cfg.backend,
        "gamma": float(gamma),
        "num_iterations": int(cfg.num_iterations),
        "iterations_completed": int(len(final_stage_rows)),
        "history_rows": int(len(history)),
        "selection_history_rows": int(len(selection_rows)),
        "refit_history_rows": int(len(refit_rows)),
        "final_stage": final_stage,
        "optimizer_steps": int(cfg.optimizer_steps),
        "learning_rate": float(cfg.learning_rate),
        "neural_log_partition_mode": str(cfg.neural_log_partition_mode),
        "neural_batch_size": ""
        if cfg.neural_batch_size is None
        else int(cfg.neural_batch_size),
        "neural_variational_a": float(final.get("neural_variational_a", float("nan"))),
        "neural_variational_gauge_fix": str(cfg.neural_variational_gauge_fix),
        "neural_variational_gauge_active": bool(
            final.get("neural_variational_gauge_active", False)
        ),
        "neural_variational_gauge_mode": str(
            final.get("neural_variational_gauge_mode", "none")
        ),
        "neural_variational_gauge_terminal_sensitive": bool(
            final.get("neural_variational_gauge_terminal_sensitive", False)
        ),
        "neural_variational_gauge_shift": float(
            final.get("neural_variational_gauge_shift", float("nan"))
        ),
        "early_stopping": bool(early_stop_rows),
        "early_stopping_requested": bool(
            cfg.early_stopping and cfg.backend == "neural"
        ),
        "patience": int(cfg.patience),
        "min_improvement": float(cfg.min_improvement),
        "validation_warmup_iterations": int(cfg.validation_warmup_iterations),
        "split_training": bool(final.get("split_training", False)),
        "objective_train_rows": int(final.get("objective_train_rows", reference_rows)),
        "objective_init_rows": int(final.get("objective_init_rows", initial_rows)),
        "objective_successor_rows": int(
            final.get("objective_successor_rows", successor_row_index.shape[0])
        ),
        "validation_init_rows": int(final.get("validation_init_rows", 0)),
        "min_ess_fraction": ""
        if cfg.min_ess_fraction is None
        else float(cfg.min_ess_fraction),
        "max_logit_cap_fraction": (
            ""
            if cfg.max_logit_cap_fraction is None
            else float(cfg.max_logit_cap_fraction)
        ),
        "accepted_count": int(final.get("accepted_count", len(accepted_rows))),
        "validation_warmup_accepts": int(
            sum(
                1 for row in history if bool(row.get("validation_warmup_accept", False))
            )
        ),
        "stopped_early": bool(stop_rows),
        "stop_iter": int(stop_rows[-1]["iteration"]) if stop_rows else "",
        "stop_reason": str(stop_rows[-1].get("stop_reason", "")) if stop_rows else "",
        "best_valid_loss": float(np.min(valid_losses))
        if valid_losses
        else float("nan"),
        "final_valid_loss": float(valid_losses[-1]) if valid_losses else float("nan"),
        "boosting_learning_rate": float(cfg.boosting_learning_rate),
        "l2_penalty": float(cfg.l2_penalty),
        "score_tikhonov_penalty": float(cfg.score_tikhonov_penalty),
        "selection_objective_name": str(cfg.selection_objective_name or ""),
        "selection_objective_final": float(
            final.get("selection_objective_value", float("nan"))
        ),
        "selection_objective_best": float(best_selection_value),
        "selection_objective_best_iteration": best_selection_iteration,
        "feature_dim": int(feature_dim),
        "reference_rows": int(reference_rows),
        "initial_rows": int(initial_rows),
        "successor_rows": int(successor_row_index.shape[0]),
        "initial_action_source": str(initial_action_source),
        "successor_action_source": str(successor_source),
        "sample_reuse": True,
        "log_partition": float(final.get("log_partition", _logmeanexp(log_ratio_ref))),
        "clip_log_partition_adjustment": float(clip_log_partition_adjustment),
        "empirical_mass": float(np.mean(weights_ref)),
        "empirical_mass_abs_error": float(abs(np.mean(weights_ref) - 1.0)),
        "ess_fraction": _ess_fraction(weights_ref),
        "weight_mean": float(np.mean(weights_ref)),
        "weight_min": float(np.min(weights_ref)),
        "weight_max": float(np.max(weights_ref)),
        "weight_q95": float(np.quantile(weights_ref, 0.95)),
        "weight_q99": float(np.quantile(weights_ref, 0.99)),
        "weight_kl_to_uniform": _empirical_weight_kl(
            weights_ref, np.ones_like(weights_ref), cfg
        ),
        "weight_reverse_kl_to_uniform": _empirical_weight_kl(
            np.ones_like(weights_ref), weights_ref, cfg
        ),
        "weight_step_kl_final": float(final.get("weight_step_kl", float("nan"))),
        "weight_step_reverse_kl_final": float(
            final.get("weight_step_reverse_kl", float("nan"))
        ),
        "weight_step_l1_final": float(final.get("weight_step_l1", float("nan"))),
        "valid_weight_kl_to_uniform": float(
            final.get("valid_weight_kl_to_uniform", float("nan"))
        ),
        "valid_weight_step_kl_final": float(
            final.get("valid_weight_step_kl", float("nan"))
        ),
        "valid_weight_step_reverse_kl_final": float(
            final.get("valid_weight_step_reverse_kl", float("nan"))
        ),
        "valid_weight_step_l1_final": float(
            final.get("valid_weight_step_l1", float("nan"))
        ),
        "logit_clip": cap,
        "logit_clip_applied_to_iteration": cap is not None,
        "logit_cap_fraction": 0.0
        if cap is None
        else float(np.mean(np.abs(raw_log_ratio_ref) >= float(cap))),
        "continuation_mean": float(np.mean(continuation_plus)),
        "continuation_min": float(np.min(continuation_plus)),
        "objective_final": float(final.get("objective", float("nan"))),
        "grad_norm_final": float(final.get("grad_norm", float("nan"))),
    }


def _import_torch_for_kl_fori() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional install state
        raise ImportError(
            "backend='neural' requires the optional torch dependency."
        ) from exc
    return torch


def _build_torch_mlp(torch: Any, input_dim: int, hidden_dims: Sequence[int]) -> Any:
    layers: list[Any] = []
    in_dim = int(input_dim)
    for width in tuple(int(width) for width in hidden_dims):
        layer = torch.nn.Linear(in_dim, width)
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
        layers.extend([layer, torch.nn.ReLU()])
        in_dim = width
    out = torch.nn.Linear(in_dim, 1)
    torch.nn.init.zeros_(out.weight)
    torch.nn.init.zeros_(out.bias)
    layers.append(out)
    return torch.nn.Sequential(*layers)


def _torch_final_linear_layer(torch: Any, model: Any) -> Any:
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Linear):
            return module
    raise TypeError("neural KL-FORI model must contain a final linear layer.")


def _recenter_torch_score_gauge(
    *,
    torch: Any,
    model: Any,
    final_linear_layer: Any,
    variational_a: Any,
    reference_batch: Any,
) -> float:
    # The fitted ratio uses h(x) - log E exp(h(X)), so a common score shift is
    # only an optimizer gauge.  Adjusting the final bias and variational scalar
    # together preserves exp(h-a) exactly.
    if getattr(final_linear_layer, "bias", None) is None:
        return 0.0
    with torch.no_grad():
        shift = model(reference_batch).reshape(-1).mean()
        shift_value = float(shift.detach().cpu().item())
        final_linear_layer.bias.sub_(shift)
        variational_a.sub_(shift)
    return shift_value


def _torch_grad_norm(torch: Any, parameters: Any) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            value = float(
                torch.sum(param.grad.detach() * param.grad.detach()).cpu().item()
            )
            total += value
    return float(np.sqrt(total))


def _predict_neural_scores(
    z: Array,
    *,
    state_dict: dict[str, Array],
    hidden_dims: Sequence[int],
    device: str,
) -> Array:
    if not state_dict:
        raise ValueError("Neural KL-FORI model has no stored state_dict.")
    torch = _import_torch_for_kl_fori()
    dev = torch.device(str(device))
    model = _build_torch_mlp(
        torch, np.asarray(z).shape[1], tuple(int(width) for width in hidden_dims)
    ).to(dev)
    tensor_state = {
        name: torch.as_tensor(value, dtype=torch.float32, device=dev)
        for name, value in state_dict.items()
    }
    model.load_state_dict(tensor_state)
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(
            np.asarray(z, dtype=np.float32), dtype=torch.float32, device=dev
        )
        return (
            model(x).reshape(-1).detach().cpu().numpy().astype(np.float64, copy=False)
        )


__all__ = [
    "KLFORIConfig",
    "KLFORIModel",
    "fit_kl_fori",
    "fit_kl_fori_boosting",
    "fit_kl_fori_neural",
]
