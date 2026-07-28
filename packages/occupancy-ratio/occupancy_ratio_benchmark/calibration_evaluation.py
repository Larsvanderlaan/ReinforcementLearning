"""Submission endpoints for normalized occupancy cross-calibration.

This module turns one fitted cross-calibrated prediction matrix into flat,
auditable candidate rows.  It deliberately contains no fitting or candidate
selection logic: rewards and oracle ratios are first touched here, after the
native, scalar, and PAVA candidates have all been constructed.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from occupancy_ratio_benchmark.calibration_crossfit import CrossCalibratedMatrixResult
from occupancy_ratio_benchmark.calibration_metrics import (
    controlled_ratio_errors,
    estimate_bellman_cross_moment_error,
    estimate_multi_reward_occupancy_functional_error,
    oracle_floor_kl_sensitivity,
)
from occupancy_ratio_benchmark.calibration_truth import has_exact_finite_support
from occupancy_ratio_benchmark.data import BenchmarkDataset


Array = np.ndarray
CANDIDATE_IDS = (
    "native_pointwise_median",
    "scalar_normalized_pointwise_median",
    "pava_pointwise_median",
)
FULL_DATA_CANDIDATE_ID = "full_data_raw"


def evaluate_cross_calibrated_result(
    *,
    dataset: BenchmarkDataset,
    result: CrossCalibratedMatrixResult,
    source_groups: Array,
    initial_groups: Array,
    identity: Mapping[str, Any],
    estimator_id: str,
    score_distortion: str | None = None,
    fold_runtime_sec: Sequence[float] = (),
    aggregation_runtime_sec: float = 0.0,
    retry_count: int = 0,
    full_data_predictions: Mapping[str, Array] | None = None,
    full_data_fit_runtime_sec: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Array]]]:
    """Evaluate all frozen candidates and return flat rows plus predictions.

    The cross-moment basis is the candidate transform of the pooled diagonal
    OOF score, while the audited current/successor/initial weights are the
    deployed pointwise fold medians.  The two audit halves are used only by the
    metric and never as calibration-fit splits.
    """

    _validate_normalized_dataset(dataset)
    source_group = _group_vector(source_groups, dataset.n, "source_groups")
    initial_n = int(np.asarray(dataset.initial_states).shape[0])
    initial_group = _group_vector(initial_groups, initial_n, "initial_groups")
    if result.assignment.source_fold_ids.shape[0] != dataset.n:
        raise ValueError("cross-calibration source assignment does not match dataset")
    if result.assignment.initial_fold_ids.shape[0] != initial_n:
        raise ValueError("cross-calibration initial assignment does not match dataset")
    if not np.allclose(dataset.initial_weights, dataset.initial_weights[0]):
        raise ValueError("cross-moment evaluation currently requires equal initial-row weights")

    candidate_arrays = _candidate_arrays(result)
    pooled_basis = _pooled_basis_arrays(result)
    candidate_ids = list(CANDIDATE_IDS)
    if full_data_predictions is not None:
        full_data = _validated_prediction_roles(
            full_data_predictions,
            source_n=dataset.n,
            next_n=dataset.n,
            initial_n=initial_n,
        )
        candidate_arrays[FULL_DATA_CANDIDATE_ID] = full_data
        pooled_basis[FULL_DATA_CANDIDATE_ID] = full_data["current"]
        candidate_ids.append(FULL_DATA_CANDIDATE_ID)
    split_seed = _stable_seed(identity)
    common = {
        **{str(key): value for key, value in identity.items()},
        "estimator_id": str(estimator_id),
        "benchmark_family": str(identity.get("benchmark_family", dataset.setting)),
        "setting": str(dataset.setting),
        "sample_size": int(dataset.n),
        "gamma": float(dataset.gamma),
        "seed": int(dataset.seed),
        "score_distortion": None if score_distortion is None else str(score_distortion),
        "status": "ok",
        "failure_reason": None,
        "occupancy_estimand": "normalized_discounted",
        "coverage_stopping": False,
        "base_upper_cap_enabled": False,
        "base_query_normalization_enabled": False,
        "cross_moment_split_seed": int(split_seed),
        "cross_moment_halves_used_for_fit": False,
        "cross_moment_debiasing_basis": "pooled_oof_candidate_predictions",
        "cross_moment_dependence_justification": "grouped_cross_calibration_theorem",
        "policy_value_truth_precision_met": dataset.metadata.get(
            "target_truth_precision_met"
        ),
        "policy_value_truth_ci95_half_width": _optional_float(
            dataset.metadata.get("target_truth_ci95_half_width")
        ),
        "policy_value_truth_target_half_width": _optional_float(
            dataset.metadata.get("target_truth_target_half_width")
        ),
        "fold_runtime_sec": [float(value) for value in fold_runtime_sec],
        "fold_runtime_total_sec": float(np.sum(fold_runtime_sec)),
        "aggregation_runtime_sec": float(aggregation_runtime_sec),
        "retry_count": int(retry_count),
        "full_data_fit_runtime_sec": _optional_float(full_data_fit_runtime_sec),
    }
    rows: list[dict[str, Any]] = []
    rewards = np.asarray(dataset.rewards, dtype=np.float64).reshape(-1)
    if rewards.size != dataset.n or not np.all(np.isfinite(rewards)):
        raise ValueError("dataset rewards must be finite and row aligned")

    for candidate_id in candidate_ids:
        is_full_data = candidate_id == FULL_DATA_CANDIDATE_ID
        prediction = candidate_arrays[candidate_id]
        current = prediction["current"]
        successor = prediction["next"]
        initial = prediction["initial"]
        cross_moment = estimate_bellman_cross_moment_error(
            basis_candidate_weights=pooled_basis[candidate_id],
            audit_current_weights=current,
            audit_next_weights=successor,
            audit_initial_weights=initial,
            audit_transition_group_ids=source_group,
            audit_initial_group_ids=initial_group,
            gamma=float(dataset.gamma),
            basis_group_ids=source_group,
            split_seed=int(split_seed),
        )
        value_estimate = float(np.mean(current * rewards))
        value_fields = _value_error_fields(dataset, value_estimate)
        diagnostics = weight_diagnostics(current)
        functional_fields, functional_arrays = _occupancy_functional_fields(
            dataset,
            current,
            panel_seed=_stable_reward_panel_seed(identity),
        )
        prediction.update(functional_arrays)
        row: dict[str, Any] = {
            **common,
            "candidate_id": candidate_id,
            "fit_scope": (
                "full_data"
                if is_full_data
                else f"oof{int(result.assignment.num_folds)}"
            ),
            "num_base_fits": (
                1 if is_full_data else int(result.assignment.num_folds)
            ),
            "calibration_fit": (
                "none"
                if is_full_data
                else "single_map_on_pooled_oof_scores"
            ),
            "pointwise_aggregation": (
                "none" if is_full_data else "median"
            ),
            "cross_moment_dependence_justification": (
                "descriptive_full_data_fit"
                if is_full_data
                else "grouped_cross_calibration_theorem"
            ),
            "policy_value_estimate": value_estimate,
            **value_fields,
            "cross_moment_signed_squared_error": float(
                cross_moment.base.signed_squared_error
            ),
            "cross_moment_positive_part_root_error": float(
                cross_moment.base.positive_part_root_error
            ),
            "cross_moment_half_bins_signed_squared_error": float(
                cross_moment.half_bins.signed_squared_error
            ),
            "cross_moment_double_bins_signed_squared_error": float(
                cross_moment.double_bins.signed_squared_error
            ),
            "cross_moment_mass_signed_squared_error": float(
                cross_moment.mass.signed_squared_error
            ),
            "cross_moment_requested_bins": int(cross_moment.base.requested_bins),
            "cross_moment_effective_bins": int(cross_moment.base.effective_bins),
            "cross_moment_basis_near_constant": bool(cross_moment.basis_near_constant),
            "cross_moment_basis_audit_groups_disjoint": (
                cross_moment.basis_audit_groups_disjoint
            ),
            "cross_moment_n_union_groups": int(cross_moment.n_union_groups),
            **diagnostics,
            **functional_fields,
            **_controlled_fields(dataset, current),
        }
        if candidate_id == "pava_pointwise_median":
            pava_diagnostics = getattr(result.calibrator, "diagnostics", {})
            next_fraction = _optional_float(
                pava_diagnostics.get("constant_extrapolation_next_fraction")
            )
            initial_mass = _optional_float(
                pava_diagnostics.get("constant_extrapolation_initial_mass")
            )
            row.update(
                {
                    "pava_status": str(getattr(result.calibrator, "status", "unknown")),
                    "pava_converged": bool(pava_diagnostics.get("converged", False)),
                    "pava_iterations": int(pava_diagnostics.get("iterations", 0)),
                    "pava_relative_change": _optional_float(
                        pava_diagnostics.get("relative_change")
                    ),
                    "pava_boundary_extrapolation_next_fraction": next_fraction,
                    "pava_boundary_extrapolation_initial_mass": initial_mass,
                    "pava_boundary_extrapolation_count": (
                        int(round(next_fraction * dataset.n))
                        + int(round(initial_mass * initial_n))
                        if next_fraction is not None and initial_mass is not None
                        else None
                    ),
                    "pava_boundary_extrapolation_bellman_mass": (
                        float(dataset.gamma) * next_fraction
                        + (1.0 - float(dataset.gamma)) * initial_mass
                        if next_fraction is not None and initial_mass is not None
                        else None
                    ),
                }
            )
        else:
            row.update(
                {
                    "pava_status": None,
                    "pava_converged": None,
                    "pava_iterations": None,
                    "pava_relative_change": None,
                    "pava_boundary_extrapolation_next_fraction": None,
                    "pava_boundary_extrapolation_initial_mass": None,
                    "pava_boundary_extrapolation_count": None,
                    "pava_boundary_extrapolation_bellman_mass": None,
                }
            )
        rows.append(row)
    return rows, candidate_arrays


def weight_diagnostics(weights: Array) -> dict[str, float | int]:
    """Return prespecified mass, tail, and effective-sample-size diagnostics."""

    value = np.asarray(weights, dtype=np.float64).reshape(-1)
    if value.size == 0:
        raise ValueError("weights must be nonempty")
    finite = np.isfinite(value)
    nonfinite_rate = float(1.0 - np.mean(finite))
    if not np.all(finite) or np.any(value < 0.0):
        raise ValueError("candidate weights must be finite and nonnegative")
    total = float(np.sum(value))
    squared_total = float(np.sum(value**2))
    positive_mass = max(total, np.finfo(np.float64).tiny)
    top_count = max(1, int(np.ceil(0.01 * value.size)))
    top_mass = float(np.sum(np.partition(value, value.size - top_count)[-top_count:]))
    mean = float(np.mean(value))
    return {
        "empirical_mass": mean,
        "effective_sample_size": (
            total**2 / squared_total if squared_total > 0.0 else 0.0
        ),
        "effective_sample_size_fraction": (
            total**2 / (value.size * squared_total) if squared_total > 0.0 else 0.0
        ),
        "weight_coefficient_of_variation": (
            float(np.std(value)) / mean if mean > 0.0 else float("inf")
        ),
        "weight_q95": float(np.quantile(value, 0.95)),
        "weight_q99": float(np.quantile(value, 0.99)),
        "weight_maximum": float(np.max(value)),
        "top_one_percent_weight_mass": top_mass / positive_mass,
        "nonfinite_rate": nonfinite_rate,
    }


def _candidate_arrays(
    result: CrossCalibratedMatrixResult,
) -> dict[str, dict[str, Array]]:
    return {
        "native_pointwise_median": {
            "current": result.source.raw,
            "next": result.next.raw,
            "initial": result.initial.raw,
        },
        "scalar_normalized_pointwise_median": {
            "current": result.source.scalar,
            "next": result.next.scalar,
            "initial": result.initial.scalar,
        },
        "pava_pointwise_median": {
            "current": result.source.pava,
            "next": result.next.pava,
            "initial": result.initial.pava,
        },
    }


def _pooled_basis_arrays(result: CrossCalibratedMatrixResult) -> dict[str, Array]:
    raw = np.asarray(result.pooled_oof.source_q, dtype=np.float64)
    scalar = float(result.pooled_oof.scalar_scale) * raw
    pava = np.asarray(result.calibrator.predict(raw), dtype=np.float64).reshape(-1)
    return {
        "native_pointwise_median": raw,
        "scalar_normalized_pointwise_median": scalar,
        "pava_pointwise_median": pava,
    }


def _validated_prediction_roles(
    predictions: Mapping[str, Array],
    *,
    source_n: int,
    next_n: int,
    initial_n: int,
) -> dict[str, Array]:
    expected = {
        "current": int(source_n),
        "next": int(next_n),
        "initial": int(initial_n),
    }
    output: dict[str, Array] = {}
    for role, size in expected.items():
        if role not in predictions:
            raise ValueError(f"full_data_predictions is missing {role!r}")
        value = np.asarray(predictions[role], dtype=np.float64).reshape(-1)
        if value.size != size:
            raise ValueError(
                f"full_data_predictions[{role!r}] must have {size} rows"
            )
        if not np.all(np.isfinite(value)) or np.any(value < 0.0):
            raise ValueError(
                f"full_data_predictions[{role!r}] must be finite and nonnegative"
            )
        output[role] = value
    return output


def _occupancy_functional_fields(
    dataset: BenchmarkDataset,
    current: Array,
    *,
    panel_seed: int,
) -> tuple[dict[str, Any], dict[str, Array]]:
    names = (
        "occupancy_functional_reward_count",
        "occupancy_functional_cross_pool_signed_mse",
        "occupancy_functional_positive_part_root_mse",
        "occupancy_functional_pooled_rmse",
        "occupancy_functional_pooled_mae",
        "occupancy_functional_pooled_max_absolute_error",
        "occupancy_functional_target_pool_a_rows",
        "occupancy_functional_target_pool_b_rows",
        "occupancy_functional_panel_sha256",
        "occupancy_functional_panel_seed",
        "occupancy_functional_panel_bandwidths",
    )
    target_arrays = (
        dataset.target_occupancy_states,
        dataset.target_occupancy_actions,
        dataset.target_occupancy_pool_ids,
    )
    if any(value is None for value in target_arrays):
        return {name: None for name in names}, {}

    target_states = np.asarray(dataset.target_occupancy_states)
    target_actions = np.asarray(dataset.target_occupancy_actions)
    pool_ids = np.asarray(dataset.target_occupancy_pool_ids).reshape(-1)
    pool_values = np.unique(pool_ids)
    if not np.array_equal(pool_values, np.asarray([0, 1])):
        raise ValueError("functional evaluation requires target pool ids {0, 1}")
    in_a = pool_ids == 0
    in_b = pool_ids == 1
    metric = estimate_multi_reward_occupancy_functional_error(
        source_states=np.asarray(dataset.states),
        source_actions=np.asarray(dataset.actions),
        candidate_weights=np.asarray(current),
        target_states_a=target_states[in_a],
        target_actions_a=target_actions[in_a],
        target_states_b=target_states[in_b],
        target_actions_b=target_actions[in_b],
        reward_count=256,
        bandwidths=(0.5, 1.0, 2.0),
        seed=int(panel_seed),
    )
    fields = {
        "occupancy_functional_reward_count": int(metric.panel.reward_count),
        "occupancy_functional_cross_pool_signed_mse": float(
            metric.signed_cross_pool_mse
        ),
        "occupancy_functional_positive_part_root_mse": float(
            metric.positive_part_root_mse
        ),
        "occupancy_functional_pooled_rmse": float(metric.pooled_rmse),
        "occupancy_functional_pooled_mae": float(metric.pooled_mae),
        "occupancy_functional_pooled_max_absolute_error": float(
            metric.pooled_max_absolute_error
        ),
        "occupancy_functional_target_pool_a_rows": int(metric.n_target_a),
        "occupancy_functional_target_pool_b_rows": int(metric.n_target_b),
        "occupancy_functional_panel_sha256": metric.panel.panel_sha256,
        "occupancy_functional_panel_seed": int(metric.panel.seed),
        "occupancy_functional_panel_bandwidths": list(metric.panel.bandwidths),
    }
    arrays = {
        "functional_source": np.asarray(metric.source_functionals),
        "functional_target_a": np.asarray(metric.target_functionals_a),
        "functional_target_b": np.asarray(metric.target_functionals_b),
        "functional_error_a": np.asarray(metric.error_a),
        "functional_error_b": np.asarray(metric.error_b),
        "functional_pooled_error": np.asarray(metric.pooled_error),
    }
    return fields, arrays


def _value_error_fields(dataset: BenchmarkDataset, estimate: float) -> dict[str, Any]:
    truth = dataset.target_policy_value
    truth_se = dataset.target_policy_value_se
    if truth is None:
        return {
            "policy_value_truth": None,
            "policy_value_truth_se": None,
            "policy_value_truth_kind": dataset.target_policy_value_kind or None,
            "policy_value_signed_error": None,
            "policy_value_absolute_error": None,
            "policy_value_squared_error": None,
            "policy_value_error_in_truth_se_units": None,
            "policy_value_safety_margin": None,
        }
    target = float(truth)
    signed = float(estimate - target)
    se = 0.0 if truth_se is None else float(truth_se)
    margin = max(0.02 * abs(target), 2.0 * se)
    return {
        "policy_value_truth": target,
        "policy_value_truth_se": se,
        "policy_value_truth_kind": dataset.target_policy_value_kind or None,
        "policy_value_signed_error": signed,
        "policy_value_absolute_error": abs(signed),
        "policy_value_squared_error": signed**2,
        "policy_value_error_in_truth_se_units": abs(signed) / se if se > 0.0 else None,
        "policy_value_safety_margin": margin,
    }


def _controlled_fields(dataset: BenchmarkDataset, current: Array) -> dict[str, Any]:
    if dataset.true_ratio is None:
        return {
            "ratio_mse_untruncated": None,
            "ratio_relative_mse_untruncated": None,
            "ratio_rmse_untruncated": None,
            "ratio_l1_untruncated": None,
            "ratio_generalized_kl_extended": None,
            "oracle_floor_smallest_positive": None,
            "oracle_floor_0p1x_kl": None,
            "oracle_floor_1x_kl": None,
            "oracle_floor_10x_kl": None,
        }
    metrics = controlled_ratio_errors(current, np.asarray(dataset.true_ratio, dtype=np.float64))
    fields: dict[str, Any] = {
        "ratio_mse_untruncated": float(metrics.mse),
        "ratio_relative_mse_untruncated": float(metrics.relative_mse),
        "ratio_rmse_untruncated": float(metrics.rmse),
        "ratio_l1_untruncated": float(metrics.l1),
        "ratio_generalized_kl_extended": float(metrics.generalized_kl),
        "oracle_floor_smallest_positive": None,
        "oracle_floor_0p1x_kl": None,
        "oracle_floor_1x_kl": None,
        "oracle_floor_10x_kl": None,
    }
    if has_exact_finite_support(dataset):
        sensitivity = oracle_floor_kl_sensitivity(
            current,
            np.asarray(dataset.true_ratio, dtype=np.float64),
            oracle_is_exact_finite_support=True,
        )
        fields["oracle_floor_smallest_positive"] = float(
            sensitivity.smallest_positive_oracle
        )
        for point in sensitivity.points:
            key = {0.1: "oracle_floor_0p1x_kl", 1.0: "oracle_floor_1x_kl", 10.0: "oracle_floor_10x_kl"}[
                point.multiplier
            ]
            fields[key] = float(point.generalized_kl)
    return fields


def _validate_normalized_dataset(dataset: BenchmarkDataset) -> None:
    if dataset.next_retention is not None and not np.allclose(dataset.next_retention, 1.0):
        raise ValueError("coverage-stopped next retention is forbidden in calibration experiments")
    if dataset.initial_retention is not None and not np.allclose(dataset.initial_retention, 1.0):
        raise ValueError("coverage-stopped initial retention is forbidden in calibration experiments")
    if bool(float(dataset.metadata.get("coverage_stopped_target", 0.0))):
        raise ValueError("coverage-stopped datasets are outside the normalized calibration protocol")


def _group_vector(value: Array, expected: int, name: str) -> Array:
    array = np.asarray(value)
    if array.ndim != 1 or array.size != expected:
        raise ValueError(f"{name} must be one-dimensional with {expected} rows")
    return array


def _stable_seed(identity: Mapping[str, Any]) -> int:
    payload = "|".join(f"{key}={identity[key]}" for key in sorted(identity))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _stable_reward_panel_seed(identity: Mapping[str, Any]) -> int:
    keys = ("benchmark_family", "cell_id", "gamma")
    payload = "|".join(
        f"{key}={identity[key]}" for key in keys if key in identity
    )
    digest = hashlib.sha256(
        f"occupancy-functional-reward-panel-v1|{payload}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


__all__: Sequence[str] = (
    "CANDIDATE_IDS",
    "FULL_DATA_CANDIDATE_ID",
    "evaluate_cross_calibrated_result",
    "weight_diagnostics",
)
