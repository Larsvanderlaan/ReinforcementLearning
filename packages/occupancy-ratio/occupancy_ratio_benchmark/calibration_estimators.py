"""Fold-level estimator adapters for the occupancy calibration benchmark.

Each adapter fits exactly once on a grouped training fold, evaluates the same
uncapped and unnormalized ratio estimate on all current, successor, and
initial rows, and returns only NumPy arrays plus JSON-friendly diagnostics.
This keeps TensorFlow and SCOPE-RL model objects out of resumable artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from occupancy_ratio.google_dualdice import (
    GoogleDualDICEConfig,
    fit_google_dualdice_occupancy_ratio,
)
from occupancy_ratio.kl_fori import KLFORIConfig, fit_kl_fori_neural
from occupancy_ratio.minimax_weight import (
    GoogleDICERLConfig,
    MinimaxWeightConfig,
    ScopeRLMinimaxWeightConfig,
    fit_minimax_weight,
)
from occupancy_ratio_benchmark.data import BenchmarkDataset


Array = np.ndarray
LEARNED_ESTIMATORS = ("neural_fori", "google_dualdice", "scope_mwl", "bestdice")


@dataclass(frozen=True)
class EstimatorPaths:
    """Optional baseline source checkouts used by fold workers."""

    google_research: Path = Path("/tmp/google-research")
    dice_rl: Path = Path("/tmp/dice_rl")
    scope_rl: Path | None = None


@dataclass(frozen=True)
class FoldPredictionResult:
    """All-row predictions emitted by one atomic estimator-fold fit."""

    estimator_id: str
    fold_index: int
    fit_seed: int
    source_q: Array
    next_q: Array
    initial_q: Array
    fit_runtime_sec: float
    prediction_runtime_sec: float
    diagnostics: dict[str, Any]


def fit_fold_predictions(
    *,
    estimator_id: str,
    dataset: BenchmarkDataset,
    train_source_indices: Array,
    train_initial_indices: Array,
    fold_index: int,
    fit_seed: int,
    registry_entry: Mapping[str, Any],
    paths: EstimatorPaths | None = None,
    prediction_chunk_size: int = 65_536,
    negative_tolerance: float = 1e-10,
) -> FoldPredictionResult:
    """Fit one frozen baseline and predict q on every cell row.

    Estimator-side normalization and upper caps are always disabled.  Because
    DualDICE and minimax learners can emit a signed unconstrained critic, the
    shared ratio score is its positive-part projection.  This lower projection
    is applied once for native, scalar, and PAVA candidates and fully recorded;
    nonfinite values remain structured numerical failures.
    """

    estimator = str(estimator_id)
    if estimator not in LEARNED_ESTIMATORS:
        raise ValueError(f"unsupported calibration estimator {estimator!r}")
    thread_diagnostics = configure_calibration_worker_threads(estimator)
    if int(prediction_chunk_size) <= 0:
        raise ValueError("prediction_chunk_size must be positive")
    if not np.isfinite(negative_tolerance) or negative_tolerance < 0.0:
        raise ValueError("negative_tolerance must be finite and nonnegative")
    source_indices = _validated_indices(train_source_indices, dataset.n, "train_source_indices")
    initial_n = int(np.asarray(dataset.initial_states).shape[0])
    initial_indices = _validated_indices(train_initial_indices, initial_n, "train_initial_indices")
    if source_indices.size == 0 or initial_indices.size == 0:
        raise ValueError("fold training source and initial samples must be nonempty")
    schedule = registry_entry.get("schedule", {})
    if not isinstance(schedule, Mapping):
        raise ValueError("estimator registry schedule must be an object")
    resolved_paths = EstimatorPaths() if paths is None else paths
    train = _slice_training_dataset(dataset, source_indices, initial_indices)

    fit_started = time.perf_counter()
    model = _fit_model(
        estimator,
        train,
        schedule=schedule,
        fit_seed=int(fit_seed),
        paths=resolved_paths,
    )
    fit_runtime = time.perf_counter() - fit_started

    prediction_started = time.perf_counter()
    source_q, source_projection = _predict_q(
        model,
        dataset.states,
        dataset.actions,
        chunk_size=int(prediction_chunk_size),
        negative_tolerance=float(negative_tolerance),
        role="source_q",
    )
    next_q, next_projection = _predict_q(
        model,
        dataset.next_states,
        dataset.next_target_actions,
        chunk_size=int(prediction_chunk_size),
        negative_tolerance=float(negative_tolerance),
        role="next_q",
    )
    initial_q, initial_projection = _predict_q(
        model,
        dataset.initial_states,
        dataset.initial_actions,
        chunk_size=int(prediction_chunk_size),
        negative_tolerance=float(negative_tolerance),
        role="initial_q",
    )
    prediction_runtime = time.perf_counter() - prediction_started
    projection_count = int(
        source_projection["count"] + next_projection["count"] + initial_projection["count"]
    )
    projection_mass = float(
        source_projection["mass"] + next_projection["mass"] + initial_projection["mass"]
    )
    diagnostics = {
        "estimator_id": estimator,
        "fold_index": int(fold_index),
        "fit_seed": int(fit_seed),
        "train_source_rows": int(source_indices.size),
        "train_initial_rows": int(initial_indices.size),
        "fit_runtime_sec": float(fit_runtime),
        "prediction_runtime_sec": float(prediction_runtime),
        "base_upper_cap_enabled": False,
        "base_query_normalization_enabled": False,
        "negative_projection_count": projection_count,
        "negative_projection_mass": projection_mass,
        "material_negative_projection_count": int(
            source_projection["material_count"]
            + next_projection["material_count"]
            + initial_projection["material_count"]
        ),
        "tiny_negative_projection_count": int(
            source_projection["tiny_count"]
            + next_projection["tiny_count"]
            + initial_projection["tiny_count"]
        ),
        "raw_prediction_minimum": float(
            min(
                source_projection["minimum"],
                next_projection["minimum"],
                initial_projection["minimum"],
            )
        ),
        "source_q_mean": float(np.mean(source_q)),
        "source_q_max": float(np.max(source_q)),
        "next_q_mean": float(np.mean(next_q)),
        "initial_q_mean": float(np.mean(initial_q)),
        "model_diagnostics": _json_safe(getattr(model, "diagnostics", {})),
        "thread_limits": thread_diagnostics,
    }
    return FoldPredictionResult(
        estimator_id=estimator,
        fold_index=int(fold_index),
        fit_seed=int(fit_seed),
        source_q=source_q,
        next_q=next_q,
        initial_q=initial_q,
        fit_runtime_sec=float(fit_runtime),
        prediction_runtime_sec=float(prediction_runtime),
        diagnostics=diagnostics,
    )


def configure_calibration_worker_threads(estimator_id: str) -> dict[str, Any]:
    """Force one native thread in a dedicated atomic fold process.

    On macOS, unrestricted OpenMP/BLAS pools combined with the neural backend
    can terminate the interpreter with SIGSEGV (exit 139).  The manifest
    runner also exports these values before process startup; this function is
    a defense for direct adapter calls and records what was applied.
    """

    variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for name in variables:
        os.environ[name] = "1"
    threadpoolctl_applied = False
    try:
        from threadpoolctl import threadpool_limits  # noqa: PLC0415
    except ImportError:
        pass
    else:
        threadpool_limits(limits=1)
        threadpoolctl_applied = True

    torch_threads_applied = False
    if str(estimator_id) in {"neural_fori", "scope_mwl"}:
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            pass
        else:
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # PyTorch permits setting inter-op threads only before its
                # first parallel region; the process-level env still applies.
                pass
            torch_threads_applied = True
    return {
        "environment": {name: os.environ[name] for name in variables},
        "threadpoolctl_applied": threadpoolctl_applied,
        "torch_threads_applied": torch_threads_applied,
    }


def _fit_model(
    estimator: str,
    dataset: BenchmarkDataset,
    *,
    schedule: Mapping[str, Any],
    fit_seed: int,
    paths: EstimatorPaths,
) -> Any:
    common = {
        "states": dataset.states,
        "actions": dataset.actions,
        "next_states": dataset.next_states,
        "target_actions": dataset.target_actions,
        "gamma": float(dataset.gamma),
        "initial_states": dataset.initial_states,
        "initial_actions": dataset.initial_actions,
        "target_next_actions": dataset.next_target_actions,
        "terminals": np.asarray(dataset.masks, dtype=np.float64).reshape(-1) <= 0.0,
        "initial_weights": dataset.initial_weights,
    }
    if estimator == "neural_fori":
        hidden = tuple(int(value) for value in schedule.get("hidden_sizes", (128, 128)))
        return fit_kl_fori_neural(
            **common,
            groups=dataset.episode_ids,
            initial_groups=dataset.initial_episode_ids,
            config=KLFORIConfig(
                backend="neural",
                num_iterations=int(schedule.get("outer_iterations", 300)),
                optimizer_steps=int(schedule.get("variational_steps", 30)),
                learning_rate=float(schedule.get("learning_rate", 1e-3)),
                l2_penalty=0.0,
                score_tikhonov_penalty=0.0,
                validation_fraction=0.2,
                early_stopping=True,
                patience=int(schedule.get("early_stopping_patience", 20)),
                validation_warmup_iterations=int(schedule.get("early_stopping_warmup", 10)),
                logit_clip=None,
                seed=int(fit_seed),
                neural_hidden_dims=hidden,
                neural_log_partition_mode="variational",
                neural_batch_size=int(schedule.get("batch_size", 8192)),
                neural_variational_gauge_fix="auto",
                neural_weight_decay=float(schedule.get("weight_decay", 0.1)),
                neural_grad_clip_norm=5.0,
                device=str(schedule.get("device", "cpu")),
            ),
        )
    if estimator == "google_dualdice":
        return fit_google_dualdice_occupancy_ratio(
            **common,
            config=GoogleDualDICEConfig(
                google_research_path=paths.google_research,
                num_updates=int(schedule.get("updates", 1_000)),
                batch_size=int(schedule.get("batch_size", 256)),
                seed=int(fit_seed),
                prediction_max=None,
                normalize_predictions=False,
                limit_tf_threads=True,
            ),
        )
    if estimator == "bestdice":
        return fit_minimax_weight(
            **common,
            rewards=dataset.rewards,
            method="google_dice_rl_recommended",
            config=MinimaxWeightConfig(
                method="google_dice_rl_recommended",
                google_dice_rl=GoogleDICERLConfig(
                    dice_rl_repo_path=paths.dice_rl,
                    num_steps=int(schedule.get("updates", 5_000)),
                    batch_size=int(schedule.get("batch_size", 256)),
                    learning_rate=float(schedule.get("learning_rate", 1e-4)),
                    hidden_dims=tuple(int(value) for value in schedule.get("hidden_sizes", (64, 64))),
                    seed=int(fit_seed),
                    prediction_max=None,
                    normalize_predictions=False,
                    limit_tf_threads=True,
                ),
            ),
        )
    if estimator == "scope_mwl":
        return fit_minimax_weight(
            **common,
            rewards=dataset.rewards,
            method="scope_rl_minimax_state_action",
            config=MinimaxWeightConfig(
                method="scope_rl_minimax_state_action",
                scope_rl=ScopeRLMinimaxWeightConfig(
                    scope_rl_repo_path=paths.scope_rl,
                    n_steps=int(schedule.get("updates", 10_000)),
                    n_steps_per_epoch=int(schedule.get("updates", 10_000)),
                    batch_size=int(schedule.get("batch_size", 128)),
                    learning_rate=float(schedule.get("learning_rate", 1e-4)),
                    hidden_dim=int(schedule.get("hidden_size", 128)),
                    bandwidth=1.0,
                    bandwidth_selection="median",
                    bandwidth_num_pairs=4_096,
                    standardize_inputs=True,
                    regularization_weight=1.0,
                    seed=int(fit_seed),
                    device=str(schedule.get("device", "cpu")),
                    prediction_max=None,
                    normalize_predictions=False,
                    limit_torch_threads=True,
                ),
            ),
            episode_ids=np.arange(dataset.n, dtype=np.int64),
            timesteps=np.zeros(dataset.n, dtype=np.int64),
            step_per_trajectory=1,
        )
    raise AssertionError(f"unreachable estimator {estimator}")


def _predict_q(
    model: Any,
    states: Array,
    actions: Array,
    *,
    chunk_size: int,
    negative_tolerance: float,
    role: str,
) -> tuple[Array, dict[str, float | int]]:
    states_array = np.asarray(states)
    actions_array = np.asarray(actions)
    if states_array.shape[0] != actions_array.shape[0]:
        raise ValueError(f"{role} states/actions row mismatch")
    chunks: list[Array] = []
    for start in range(0, states_array.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), states_array.shape[0])
        prediction = np.asarray(
            model.predict_state_action_ratio(
                states_array[start:stop],
                actions_array[start:stop],
                clip=False,
            ),
            dtype=np.float64,
        ).reshape(-1)
        if prediction.shape[0] != stop - start:
            raise ValueError(f"{role} predictor returned the wrong number of rows")
        chunks.append(prediction)
    values = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{role} contains nonfinite predictions")
    negative = values < 0.0
    material = values < -float(negative_tolerance)
    tiny = negative & ~material
    count = int(np.sum(negative))
    mass = float(-np.sum(values[negative]))
    minimum = float(np.min(values)) if values.size else 0.0
    if count:
        values = values.copy()
        values[negative] = 0.0
    return values, {
        "count": count,
        "mass": mass,
        "material_count": int(np.sum(material)),
        "tiny_count": int(np.sum(tiny)),
        "minimum": minimum,
    }


def _slice_training_dataset(
    dataset: BenchmarkDataset,
    source_indices: Array,
    initial_indices: Array,
) -> BenchmarkDataset:
    def source_optional(value: Array | None) -> Array | None:
        return None if value is None else np.asarray(value)[source_indices]

    def initial_optional(value: Array | None) -> Array | None:
        return None if value is None else np.asarray(value)[initial_indices]

    return BenchmarkDataset(
        setting=dataset.setting,
        states=np.asarray(dataset.states)[source_indices],
        actions=np.asarray(dataset.actions)[source_indices],
        next_states=np.asarray(dataset.next_states)[source_indices],
        target_actions=np.asarray(dataset.target_actions)[source_indices],
        next_target_actions=np.asarray(dataset.next_target_actions)[source_indices],
        rewards=np.asarray(dataset.rewards)[source_indices],
        true_ratio=source_optional(dataset.true_ratio),
        initial_states=np.asarray(dataset.initial_states)[initial_indices],
        initial_actions=np.asarray(dataset.initial_actions)[initial_indices],
        initial_weights=np.asarray(dataset.initial_weights)[initial_indices],
        masks=np.asarray(dataset.masks)[source_indices],
        gamma=float(dataset.gamma),
        seed=int(dataset.seed),
        sample_size=int(source_indices.size),
        true_action_ratio=source_optional(dataset.true_action_ratio),
        true_transition_ratio=source_optional(dataset.true_transition_ratio),
        reference_weights=source_optional(dataset.reference_weights),
        episode_ids=source_optional(dataset.episode_ids),
        timesteps=source_optional(dataset.timesteps),
        initial_episode_ids=initial_optional(dataset.initial_episode_ids),
        is_absorbing=source_optional(dataset.is_absorbing),
        target_policy_value=dataset.target_policy_value,
        target_policy_value_se=dataset.target_policy_value_se,
        target_policy_value_kind=dataset.target_policy_value_kind,
        metadata={**dataset.metadata, "parent_sample_size": int(dataset.sample_size)},
    )


def _validated_indices(value: Array, upper: int, name: str) -> Array:
    raw = np.asarray(value)
    indices = np.asarray(raw, dtype=np.int64).reshape(-1)
    if raw.ndim != 1 or not np.all(raw == indices):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(indices < 0) or np.any(indices >= int(upper)):
        raise ValueError(f"{name} contains an out-of-range index")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} must not contain duplicate indices")
    return indices


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return str(value)


__all__ = [
    "EstimatorPaths",
    "FoldPredictionResult",
    "LEARNED_ESTIMATORS",
    "configure_calibration_worker_threads",
    "fit_fold_predictions",
]
