"""Pure unit operations for the manifest-driven calibration runner."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping, Sequence

import numpy as np

from occupancy_ratio_benchmark.calibration_crossfit import (
    CrossCalibrationConfig,
    GroupedFoldAssignment,
    apply_fitted_cross_calibration,
    fit_cross_calibrated_matrices,
    make_grouped_fold_assignment,
)
from occupancy_ratio_benchmark.calibration_estimators import (
    EstimatorPaths,
    FoldPredictionResult,
    fit_fold_predictions,
)
from occupancy_ratio_benchmark.calibration_evaluation import evaluate_cross_calibrated_result
from occupancy_ratio_benchmark.calibration_protocol import dataset_id, stable_uint32
from occupancy_ratio_benchmark.calibration_truth import OracleScoreMatrices, oracle_score_matrices
from occupancy_ratio_benchmark.data import BenchmarkDataset


Array = np.ndarray


@dataclass(frozen=True)
class AggregationOutput:
    """Serializable output of one pooled-OOF calibration aggregation."""

    rows: tuple[dict[str, Any], ...]
    candidate_arrays: dict[str, dict[str, Array]]
    calibration_arrays: dict[str, Array]
    diagnostics: dict[str, Any]
    aggregation_runtime_sec: float


def cross_calibration_config(manifest: Mapping[str, Any], *, assignment_seed: int) -> CrossCalibrationConfig:
    config = _config(manifest)
    cross = _mapping(config, "cross_calibration")
    pava = _mapping(config, "pava")
    boundary = str(pava.get("boundary_rule", "constant_endpoint_extrapolation"))
    support_policy = {
        "constant_endpoint_extrapolation": "constant_extrapolation",
        "error": "error",
    }.get(boundary)
    if support_policy is None:
        raise ValueError(f"unsupported PAVA boundary rule {boundary!r}")
    return CrossCalibrationConfig(
        num_folds=int(cross["folds"]),
        seed=int(assignment_seed),
        pava_num_iterations=int(pava["maximum_iterations"]),
        pava_tolerance=float(pava["relative_tolerance"]),
        pava_direction=str(pava["direction"]),
        pava_fixed_point_damping=float(pava["damping"]),
        pava_support_policy=support_policy,
        pava_minimum_boundary_block_observations=int(
            pava.get("minimum_boundary_block_observations", 1)
        ),
    )


def grouped_assignment(
    *,
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
    source_groups: Array,
    initial_groups: Array,
    audit_dataset: BenchmarkDataset | None = None,
) -> GroupedFoldAssignment:
    seed = stable_uint32(dataset_id(unit), "grouped-fold-assignment")
    folds = int(_mapping(_config(manifest), "cross_calibration")["folds"])
    return make_grouped_fold_assignment(
        source_groups,
        initial_groups,
        num_folds=folds,
        seed=seed,
    )


def execute_learned_fold(
    *,
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
    dataset: BenchmarkDataset,
    source_groups: Array,
    initial_groups: Array,
    paths: EstimatorPaths | None = None,
    prediction_chunk_size: int = 65_536,
    audit_dataset: BenchmarkDataset | None = None,
) -> FoldPredictionResult:
    """Fit exactly one held-out fold and score every dataset row."""

    identity = _mapping(unit, "identity")
    estimator_id = str(identity["estimator_id"])
    registry = _mapping(_config(manifest), "estimator_registry")
    registry_entry = _mapping(registry, estimator_id)
    if registry_entry.get("crossfit_base_fit_required") is not True:
        raise ValueError(f"estimator {estimator_id!r} is not a learned fold estimator")
    assignment = grouped_assignment(
        manifest=manifest,
        unit=unit,
        source_groups=source_groups,
        initial_groups=initial_groups,
    )
    held_out = int(unit["held_out_fold"])
    if held_out < 0 or held_out >= assignment.num_folds:
        raise ValueError("held_out_fold is outside the configured fold range")
    expected_train = [fold for fold in range(assignment.num_folds) if fold != held_out]
    if list(unit.get("train_folds", [])) != expected_train:
        raise ValueError("unit train_folds do not match held_out_fold")
    source_train = np.flatnonzero(assignment.source_fold_ids != held_out)
    initial_train = np.flatnonzero(assignment.initial_fold_ids != held_out)
    source_calibration = np.flatnonzero(assignment.source_fold_ids == held_out)
    fit_seed = stable_uint32(manifest["run_id"], unit["unit_id"], "base-fit")
    return fit_fold_predictions(
        estimator_id=estimator_id,
        dataset=dataset,
        train_source_indices=source_train,
        train_initial_indices=initial_train,
        fold_index=held_out,
        fit_seed=fit_seed,
        registry_entry=registry_entry,
        paths=paths,
        calibration_source_indices=source_calibration,
        prediction_chunk_size=int(prediction_chunk_size),
        audit_dataset=audit_dataset,
    )


def execute_deterministic_score(
    *,
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
    dataset: BenchmarkDataset,
    audit_dataset: BenchmarkDataset | None = None,
) -> OracleScoreMatrices:
    """Create one deterministic oracle-distortion score artifact."""

    identity = _mapping(unit, "identity")
    axes = _mapping(identity, "axis_values")
    distortion = axes.get("score_distortion")
    if not isinstance(distortion, str) or not distortion:
        raise ValueError("deterministic score unit lacks score_distortion")
    folds = int(_mapping(_config(manifest), "cross_calibration")["folds"])
    if int(unit.get("conceptual_fold_count", -1)) != folds:
        raise ValueError("deterministic conceptual fold count has drifted")
    return oracle_score_matrices(
        dataset,
        distortion=distortion,
        num_folds=folds,
        audit_dataset=audit_dataset,
    )


def execute_aggregation(
    *,
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
    dataset: BenchmarkDataset,
    source_groups: Array,
    initial_groups: Array,
    source_q_by_fold: Array,
    next_q_by_fold: Array,
    initial_q_by_fold: Array,
    source_log_score_by_fold: Array | None = None,
    next_log_score_by_fold: Array | None = None,
    initial_log_score_by_fold: Array | None = None,
    fold_runtime_sec: Sequence[float] = (),
    retry_count: int = 0,
    paths: EstimatorPaths | None = None,
    cached_full_data_predictions: Mapping[str, Array] | None = None,
    cached_full_data_fit_runtime_sec: float | None = None,
    cached_full_data_fit_diagnostics: Mapping[str, Any] | None = None,
    audit_dataset: BenchmarkDataset | None = None,
    audit_source_groups: Array | None = None,
    audit_initial_groups: Array | None = None,
    audit_source_q_by_fold: Array | None = None,
    audit_next_q_by_fold: Array | None = None,
    audit_initial_q_by_fold: Array | None = None,
    audit_source_log_score_by_fold: Array | None = None,
    audit_next_log_score_by_fold: Array | None = None,
    audit_initial_log_score_by_fold: Array | None = None,
    cached_full_data_audit_predictions: Mapping[str, Array] | None = None,
) -> AggregationOutput:
    """Fit one pooled OOF map, apply it foldwise, median, and evaluate.

    Previously frozen full-data predictions may be supplied when reaggregating
    a calibration map. This avoids retraining the comparison arm and preserves
    its original provenance.
    """

    started = time.perf_counter()
    assignment = grouped_assignment(
        manifest=manifest,
        unit=unit,
        source_groups=source_groups,
        initial_groups=initial_groups,
    )
    config = cross_calibration_config(manifest, assignment_seed=int(assignment.seed))
    result = fit_cross_calibrated_matrices(
        source_q_by_fold=source_q_by_fold,
        next_q_by_fold=next_q_by_fold,
        initial_q_by_fold=initial_q_by_fold,
        source_log_score_by_fold=source_log_score_by_fold,
        next_log_score_by_fold=next_log_score_by_fold,
        initial_log_score_by_fold=initial_log_score_by_fold,
        assignment=assignment,
        gamma=float(dataset.gamma),
        initial_weights=dataset.initial_weights,
        config=config,
    )
    audit_inputs = (
        audit_source_q_by_fold,
        audit_next_q_by_fold,
        audit_initial_q_by_fold,
    )
    if any(value is not None for value in audit_inputs) and not all(
        value is not None for value in audit_inputs
    ):
        raise ValueError("all three audit fold-prediction matrices are required")
    audit_candidate_predictions: dict[str, dict[str, Array]] | None = None
    if all(value is not None for value in audit_inputs):
        audit_source, audit_next, audit_initial = apply_fitted_cross_calibration(
            result,
            source_q_by_fold=audit_source_q_by_fold,
            next_q_by_fold=audit_next_q_by_fold,
            initial_q_by_fold=audit_initial_q_by_fold,
            source_log_score_by_fold=audit_source_log_score_by_fold,
            next_log_score_by_fold=audit_next_log_score_by_fold,
            initial_log_score_by_fold=audit_initial_log_score_by_fold,
        )
        audit_candidate_predictions = {
            candidate_id: {
                "current": getattr(audit_source, role),
                "next": getattr(audit_next, role),
                "initial": getattr(audit_initial, role),
            }
            for candidate_id, role in (
                ("native_pointwise_median", "raw"),
                ("scalar_normalized_pointwise_median", "scalar"),
                ("pava_pointwise_median", "pava"),
            )
        }
    calibrator_status = str(getattr(result.calibrator, "status", "unknown"))
    calibrator_diagnostics = dict(getattr(result.calibrator, "diagnostics", {}))
    if calibrator_status != "ok" or calibrator_diagnostics.get("converged") is not True:
        raise RuntimeError(
            "PAVA failed the convergence gate: "
            f"status={calibrator_status}, iterations={calibrator_diagnostics.get('iterations')}"
        )
    runtime = time.perf_counter() - started
    identity = _flat_evaluation_identity(unit, dataset)
    axes = _mapping(_mapping(unit, "identity"), "axis_values")
    estimator_id = str(_mapping(unit, "identity")["estimator_id"])
    full_data_result: FoldPredictionResult | None = None
    full_data_predictions = (
        None
        if cached_full_data_predictions is None
        else {
            role: np.asarray(cached_full_data_predictions[role])
            for role in ("current", "next", "initial")
        }
    )
    full_data_runtime = (
        None
        if cached_full_data_fit_runtime_sec is None
        else float(cached_full_data_fit_runtime_sec)
    )
    full_data_diagnostics = (
        None
        if cached_full_data_fit_diagnostics is None
        else dict(cached_full_data_fit_diagnostics)
    )
    full_data_audit_predictions = (
        None
        if cached_full_data_audit_predictions is None
        else {
            role: np.asarray(cached_full_data_audit_predictions[role])
            for role in ("current", "next", "initial")
        }
    )
    registry_entry = _mapping(
        _mapping(_config(manifest), "estimator_registry"),
        estimator_id,
    )
    if (
        registry_entry.get("crossfit_base_fit_required") is True
        and full_data_predictions is None
    ):
        full_data_result = fit_fold_predictions(
            estimator_id=estimator_id,
            dataset=dataset,
            train_source_indices=np.arange(dataset.n, dtype=np.int64),
            train_initial_indices=np.arange(
                np.asarray(dataset.initial_states).shape[0],
                dtype=np.int64,
            ),
            fold_index=-1,
            fit_seed=stable_uint32(
                manifest["run_id"],
                unit["unit_id"],
                "full-data-base-fit",
            ),
            registry_entry=registry_entry,
            paths=paths,
            calibration_source_indices=np.arange(dataset.n, dtype=np.int64),
            audit_dataset=audit_dataset,
        )
        full_data_predictions = {
            "current": full_data_result.source_q,
            "next": full_data_result.next_q,
            "initial": full_data_result.initial_q,
        }
        full_data_runtime = float(full_data_result.fit_runtime_sec)
        full_data_diagnostics = dict(full_data_result.diagnostics)
        if full_data_result.audit_source_q is not None:
            full_data_audit_predictions = {
                "current": full_data_result.audit_source_q,
                "next": full_data_result.audit_next_q,
                "initial": full_data_result.audit_initial_q,
            }
    rows, arrays = evaluate_cross_calibrated_result(
        dataset=dataset,
        result=result,
        source_groups=source_groups,
        initial_groups=initial_groups,
        identity=identity,
        estimator_id=estimator_id,
        score_distortion=(
            str(axes["score_distortion"]) if "score_distortion" in axes else None
        ),
        fold_runtime_sec=fold_runtime_sec,
        aggregation_runtime_sec=runtime,
        retry_count=int(retry_count),
        full_data_predictions=full_data_predictions,
        full_data_fit_runtime_sec=full_data_runtime,
        audit_dataset=audit_dataset,
        audit_source_groups=audit_source_groups,
        audit_initial_groups=audit_initial_groups,
        audit_candidate_predictions=audit_candidate_predictions,
        full_data_audit_predictions=full_data_audit_predictions,
    )
    diagnostics = {
        **result.diagnostics,
        "pava_status": calibrator_status,
        "pava_diagnostics": calibrator_diagnostics,
        "aggregation_runtime_sec": runtime,
        "dependency_count": len(unit.get("depends_on", [])),
        "full_data_fit_diagnostics": full_data_diagnostics,
        "independent_behavior_audit": audit_dataset is not None,
    }
    calibration_arrays = {
        "pooled_oof_source_raw": np.asarray(result.pooled_oof.source_q),
        "pooled_oof_next_raw": np.asarray(result.pooled_oof.next_q),
        "pooled_oof_initial_raw": np.asarray(result.pooled_oof.initial_q),
        "scalar_scale": np.asarray([result.pooled_oof.scalar_scale], dtype=np.float64),
    }
    if result.pooled_oof.score_space == "log_ratio":
        calibration_arrays.update(
            {
                "pooled_oof_source_score": np.asarray(result.pooled_oof.source_score),
                "pooled_oof_next_score": np.asarray(result.pooled_oof.next_score),
                "pooled_oof_initial_score": np.asarray(result.pooled_oof.initial_score),
                "scalar_log_shift": np.asarray(
                    [result.pooled_oof.scalar_log_shift], dtype=np.float64
                ),
            }
        )
    if hasattr(result.calibrator, "grid") and hasattr(result.calibrator, "fitted_grid_values"):
        calibration_arrays["pava_grid"] = np.asarray(result.calibrator.grid, dtype=np.float64)
        calibration_arrays["pava_fitted_grid_values"] = np.asarray(
            result.calibrator.fitted_grid_values, dtype=np.float64
        )
    if audit_candidate_predictions is not None:
        for candidate_id, values in audit_candidate_predictions.items():
            for role, value in values.items():
                calibration_arrays[f"audit_candidate__{candidate_id}__{role}"] = np.asarray(value)
    if full_data_audit_predictions is not None:
        for role, value in full_data_audit_predictions.items():
            calibration_arrays[f"audit_candidate__full_data_raw__{role}"] = np.asarray(value)
    return AggregationOutput(
        rows=tuple(rows),
        candidate_arrays=arrays,
        calibration_arrays=calibration_arrays,
        diagnostics=diagnostics,
        aggregation_runtime_sec=runtime,
    )


def _flat_evaluation_identity(
    unit: Mapping[str, Any], dataset: BenchmarkDataset
) -> dict[str, Any]:
    identity = _mapping(unit, "identity")
    axes = _mapping(identity, "axis_values")
    cell = _mapping(unit, "cell")
    return {
        "study_id": str(identity["study_id"]),
        "cell_id": str(identity["cell_id"]),
        "benchmark_family": str(cell["benchmark_family"]),
        "sample_size": int(axes["sample_size"]),
        "gamma": float(axes["gamma"]),
        "seed": int(axes["seed"]),
        "dataset_id": dataset_id(unit),
    }


def _config(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(manifest, "resolved_config")


def _mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


__all__ = [
    "AggregationOutput",
    "cross_calibration_config",
    "execute_aggregation",
    "execute_deterministic_score",
    "execute_learned_fold",
    "grouped_assignment",
]
