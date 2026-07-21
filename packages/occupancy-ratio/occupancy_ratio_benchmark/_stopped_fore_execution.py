"""Train-once execution and external-test metrics for stopped FORE."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from occupancy_ratio.clipped_kl_fori import ClippedKLFORIConfig, fit_clipped_kl_fori
from occupancy_ratio.kl_fori import KLFORIConfig, fit_kl_fori
from occupancy_ratio_benchmark._stopped_fore_data import (
    StoppedFOREExternalConfig,
    StructuralStoppedDataset,
    context_target_rows,
)


Array = np.ndarray


def run_external_test_cell(
    train: StructuralStoppedDataset,
    test: StructuralStoppedDataset,
    config: StoppedFOREExternalConfig,
    *,
    backend: str,
    repetition: int,
    train_seed: int,
    test_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Array]]:
    """Fit each estimator once on training data and score one external test set."""
    _validate_train_test_truth(train, test)
    common = _common_fields(
        train,
        test,
        config,
        backend=backend,
        repetition=repetition,
        train_seed=train_seed,
        test_seed=test_seed,
    )
    fit_kwargs = {
        "states": train.states,
        "actions": train.actions,
        "next_states": train.next_states,
        "target_next_actions": train.target_next_actions,
        "initial_states": train.initial_states,
        "initial_actions": train.initial_actions,
        "gamma": config.gamma,
    }
    rows: list[dict[str, Any]] = []
    arrays: dict[str, Array] = {
        "test_context_id": test.context_id,
        "test_stage": test.stage,
        "test_action_id": test.action_id,
        "test_stopped_ratio": test.stopped_ratio,
        **{f"test_reward_{key}": value for key, value in test.rewards.items()},
    }

    if "stopped_fori_learned_gate" in config.methods:
        start = time.perf_counter()
        try:
            model = fit_clipped_kl_fori(
                **fit_kwargs,
                config=ClippedKLFORIConfig(
                    tau_lower=config.tau_lower,
                    tau_upper=config.tau_upper,
                    backend=backend,
                    num_iterations=config.stopped_num_iterations,
                    min_iterations=min(5, config.stopped_num_iterations),
                    gate_optimizer_steps=config.stopped_gate_steps,
                    ratio_optimizer_steps=config.stopped_ratio_steps,
                    gate_learning_rate=config.stopped_gate_learning_rate,
                    ratio_learning_rate=config.stopped_ratio_learning_rate,
                    neural_hidden_dims=tuple(config.stopped_hidden_dims),
                    validation_fraction=config.validation_fraction,
                    device=config.device,
                    seed=train_seed,
                ),
            )
            runtime = time.perf_counter() - start
            prediction = model.predict_state_action_ratio(
                test.states, test.actions
            )
            row = _metric_row(
                common,
                method="stopped_fori_learned_gate",
                prediction=prediction,
                test=test,
                runtime_sec=runtime,
                model=model,
                tau_upper=config.tau_upper,
                diagnostics=model.diagnostics,
            )
            arrays["prediction_stopped_fori_learned_gate"] = prediction
            rows.append(row)
        except Exception as exc:  # benchmark failures are result rows
            rows.append(
                _error_row(
                    common,
                    method="stopped_fori_learned_gate",
                    runtime_sec=time.perf_counter() - start,
                    exc=exc,
                )
            )

    standard_prediction: Array | None = None
    standard_model: Any = None
    standard_runtime = 0.0
    standard_diagnostics: dict[str, Any] = {}
    if "standard_fori" in config.methods:
        start = time.perf_counter()
        try:
            standard_model = fit_kl_fori(
                **fit_kwargs,
                config=KLFORIConfig(
                    backend=backend,
                    num_iterations=config.standard_num_iterations,
                    optimizer_steps=config.standard_optimizer_steps,
                    neural_hidden_dims=tuple(config.standard_hidden_dims),
                    validation_fraction=config.validation_fraction,
                    early_stopping=False,
                    device=config.device,
                    seed=train_seed + 911,
                ),
            )
            standard_runtime = time.perf_counter() - start
            standard_prediction = standard_model.predict_state_action_ratio(
                test.states, test.actions
            )
            standard_diagnostics = dict(standard_model.diagnostics)
            rows.append(
                _metric_row(
                    common,
                    method="standard_fori",
                    prediction=standard_prediction,
                    test=test,
                    runtime_sec=standard_runtime,
                    model=standard_model,
                    tau_upper=config.tau_upper,
                    diagnostics=standard_diagnostics,
                )
            )
            arrays["prediction_standard_fori"] = standard_prediction
        except Exception as exc:
            standard_runtime = time.perf_counter() - start
            rows.append(
                _error_row(
                    common,
                    method="standard_fori",
                    runtime_sec=standard_runtime,
                    exc=exc,
                )
            )

    if "posthoc_winsorized" in config.methods:
        if standard_prediction is None:
            rows.append(
                common
                | {
                    "method": "posthoc_winsorized",
                    "status": "dependency_error",
                    "failure_type": "standard_fori_failed",
                    "error": "posthoc_winsorized requires a successful standard_fori fit",
                    "runtime_sec": float(standard_runtime),
                }
            )
        else:
            prediction = np.minimum(standard_prediction, config.tau_upper)
            rows.append(
                _metric_row(
                    common,
                    method="posthoc_winsorized",
                    prediction=prediction,
                    test=test,
                    runtime_sec=standard_runtime,
                    model=standard_model,
                    tau_upper=config.tau_upper,
                    diagnostics=standard_diagnostics,
                )
                | {"fit_reused_from": "standard_fori"}
            )
            arrays["prediction_posthoc_winsorized"] = prediction
    return rows, arrays


def _metric_row(
    common: dict[str, Any],
    *,
    method: str,
    prediction: Array,
    test: StructuralStoppedDataset,
    runtime_sec: float,
    model: Any,
    tau_upper: float,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    ratio = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if ratio.shape[0] != test.n:
        raise ValueError("prediction length does not match the external test set.")
    if not np.all(np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise FloatingPointError("external-test ratios must be finite and nonnegative.")
    truth = test.stopped_ratio
    error = ratio - truth
    estimated_mass = float(np.mean(ratio))
    values = test.truth.values()
    row = common | {
        "method": method,
        "status": "ok",
        "failure_type": "",
        "error": "",
        "runtime_sec": float(runtime_sec),
        "ratio_l1": float(np.mean(np.abs(error))),
        "ratio_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "ratio_bias": float(np.mean(error)),
        "estimated_mass": estimated_mass,
        "mass_abs_error": abs(estimated_mass - test.truth.retained_mass),
        "weight_mean": estimated_mass,
        "weight_q99": float(np.quantile(ratio, 0.99)),
        "weight_max": float(np.max(ratio)),
        "ess_fraction": _ess_fraction(ratio),
        "top_1pct_mass_fraction": _top_mass_fraction(ratio, fraction=0.01),
        "upper_envelope_hit_rate": float(
            np.mean(ratio >= float(tau_upper) * (1.0 - 1e-8))
        ),
        "diagnostics_json": json.dumps(
            _json_safe(diagnostics), sort_keys=True, allow_nan=False
        ),
    }
    value_errors: list[float] = []
    for reward_name, reward in test.rewards.items():
        estimate = float(np.mean(ratio * reward))
        target = float(values[reward_name])
        row[f"value_{reward_name}_estimate"] = estimate
        row[f"value_{reward_name}_truth"] = target
        row[f"value_{reward_name}_abs_error"] = abs(estimate - target)
        value_errors.append(abs(estimate - target))
    row["value_mean_abs_error"] = float(np.mean(value_errors))
    row.update(_external_bellman_metrics(model, ratio, test, method=method))
    row.update(_gate_metrics(model, test, method=method))
    return row


def _external_bellman_metrics(
    model: Any,
    ratio: Array,
    test: StructuralStoppedDataset,
    *,
    method: str,
) -> dict[str, float]:
    source_features = _moment_features(test.states, test.actions)
    initial_features = _moment_features(test.initial_states, test.initial_actions)
    successor_features = _moment_features(
        test.next_states, test.target_next_actions
    )
    if method == "stopped_fori_learned_gate":
        source_gate = model.predict_gate_indicator(test.states, test.actions)
        initial_gate = model.predict_gate_indicator(
            test.initial_states, test.initial_actions
        )
        successor_gate = model.predict_gate_indicator(
            test.next_states, test.target_next_actions
        )
        clipped_source = float(model.tau_upper) * np.mean(
            (1.0 - source_gate)[:, None] * source_features, axis=0
        )
    else:
        initial_gate = np.ones(test.initial_states.shape[0], dtype=np.float64)
        successor_gate = np.ones(test.n, dtype=np.float64)
        clipped_source = np.zeros(source_features.shape[1], dtype=np.float64)
    left = np.mean(ratio[:, None] * source_features, axis=0)
    right = (
        (1.0 - test.truth.gamma)
        * np.mean(initial_gate[:, None] * initial_features, axis=0)
        + test.truth.gamma
        * np.mean(
            (ratio * successor_gate)[:, None] * successor_features, axis=0
        )
        + clipped_source
    )
    residual = left - right

    initial_context = np.argmax(
        test.initial_states[:, : 2 * test.truth.context_probability.shape[0]],
        axis=1,
    ) % test.truth.context_probability.shape[0]
    oracle_initial_gate = test.truth.initial_supported[initial_context].astype(float)
    oracle_successor_gate = test.truth.hub_supported[test.context_id].astype(float)
    oracle_left = np.mean(
        test.stopped_ratio[:, None] * source_features, axis=0
    )
    oracle_right = (
        (1.0 - test.truth.gamma)
        * np.mean(oracle_initial_gate[:, None] * initial_features, axis=0)
        + test.truth.gamma
        * np.mean(
            (test.stopped_ratio * oracle_successor_gate)[:, None]
            * successor_features,
            axis=0,
        )
    )
    oracle_residual = oracle_left - oracle_right
    return {
        "external_bellman_l2": float(np.linalg.norm(residual)),
        "external_bellman_max_abs": float(np.max(np.abs(residual))),
        "oracle_sampling_bellman_l2": float(np.linalg.norm(oracle_residual)),
        "oracle_sampling_bellman_max_abs": float(
            np.max(np.abs(oracle_residual))
        ),
    }


def _gate_metrics(
    model: Any,
    test: StructuralStoppedDataset,
    *,
    method: str,
) -> dict[str, float]:
    if method != "stopped_fori_learned_gate":
        return {}
    grid = context_target_rows(
        test.truth,
        irrelevant_features=(
            test.states.shape[1]
            - 2 * test.truth.context_probability.shape[0]
        ),
    )
    initial_prediction = model.predict_gate_indicator(
        grid["initial_states"], grid["target_actions"]
    )
    successor_prediction = model.predict_gate_indicator(
        grid["successor_states"], grid["target_actions"]
    )
    initial_truth = grid["initial_retained"].astype(bool)
    successor_truth = grid["successor_retained"].astype(bool)
    successor_relevant = grid["successor_relevant"].astype(bool)
    result = {
        "gate_source_reject_rate": float(
            np.mean(1.0 - model.predict_gate_indicator(test.states, test.actions))
        ),
    }
    masks_and_values = {
        "gate_initial_singular_accept_rate": (
            initial_prediction,
            ~initial_truth,
        ),
        "gate_initial_covered_reject_rate": (
            1.0 - initial_prediction,
            initial_truth,
        ),
        "gate_successor_singular_accept_rate": (
            successor_prediction,
            successor_relevant & ~successor_truth,
        ),
        "gate_successor_covered_reject_rate": (
            1.0 - successor_prediction,
            successor_relevant & successor_truth,
        ),
    }
    for name, (values, mask) in masks_and_values.items():
        if np.any(mask):
            result[name] = float(np.mean(np.asarray(values)[mask]))
    return result


def _common_fields(
    train: StructuralStoppedDataset,
    test: StructuralStoppedDataset,
    config: StoppedFOREExternalConfig,
    *,
    backend: str,
    repetition: int,
    train_seed: int,
    test_seed: int,
) -> dict[str, Any]:
    return {
        "repetition": int(repetition),
        "train_seed": int(train_seed),
        "test_seed": int(test_seed),
        "n_train": int(train.n),
        "n_test": int(test.n),
        "fit_scope": "single_full_training_sample",
        "evaluation_scope": "independent_external_test",
        "crossfit_folds": 0,
        "calibration_method": "none",
        "backend": str(backend),
        "failure_mode": train.truth.failure_mode,
        "support_fraction": float(train.truth.support_fraction),
        "gamma": float(config.gamma),
        "contexts": int(config.contexts),
        "behavior_probability": float(config.behavior_probability),
        "tau_lower": float(config.tau_lower),
        "tau_upper": float(config.tau_upper),
        "true_ratio_max": float(train.truth.max_positive_ratio),
        "upper_clipping_bias": 0.0,
        "oracle_mass": float(train.truth.retained_mass),
        "target_estimand": "unclipped_coverage_stopped_occupancy",
        "oracle_used_for_fitting_or_selection": False,
    }


def _error_row(
    common: dict[str, Any],
    *,
    method: str,
    runtime_sec: float,
    exc: Exception,
) -> dict[str, Any]:
    return common | {
        "method": method,
        "status": "error",
        "failure_type": "fit_exception",
        "error": f"{type(exc).__name__}: {exc}",
        "runtime_sec": float(runtime_sec),
    }


def _validate_train_test_truth(
    train: StructuralStoppedDataset, test: StructuralStoppedDataset
) -> None:
    left = train.truth
    right = test.truth
    scalar_fields = ("gamma", "failure_mode", "support_fraction")
    if any(getattr(left, name) != getattr(right, name) for name in scalar_fields):
        raise ValueError("training and test samples do not share the same estimand.")
    array_fields = (
        "context_probability",
        "initial_behavior_probability",
        "hub_behavior_probability",
        "context_reward",
    )
    if any(
        not np.array_equal(getattr(left, name), getattr(right, name))
        for name in array_fields
    ):
        raise ValueError("training and test truth arrays do not match.")


def _moment_features(states: Array, actions: Array) -> Array:
    states_array = np.asarray(states, dtype=np.float64)
    actions_array = np.asarray(actions, dtype=np.float64)
    if states_array.shape[0] != actions_array.shape[0]:
        raise ValueError("states and actions must have the same number of rows.")
    return np.column_stack(
        [np.ones(states_array.shape[0]), states_array, actions_array]
    )


def _ess_fraction(weights: Array) -> float:
    total = float(np.sum(weights))
    denominator = float(np.sum(np.square(weights)))
    if denominator <= 0.0:
        return 0.0
    return float(total * total / (weights.shape[0] * denominator))


def _top_mass_fraction(weights: Array, *, fraction: float) -> float:
    if weights.size == 0:
        return float("nan")
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    count = max(1, int(np.ceil(float(fraction) * weights.shape[0])))
    return float(np.sum(np.partition(weights, -count)[-count:]) / total)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else str(scalar)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


__all__ = ["run_external_test_cell"]
