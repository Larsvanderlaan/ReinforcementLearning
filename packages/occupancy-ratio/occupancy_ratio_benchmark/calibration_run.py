"""Manifest-driven runner for normalized occupancy cross-calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from occupancy_ratio_benchmark.calibration_artifacts import (
    ArtifactConflictError,
    artifact_is_ready,
    read_unit_arrays,
    read_unit_artifact,
    write_unit_artifact,
    write_unit_state,
)
from occupancy_ratio_benchmark.calibration_data import (
    DatasetPaths,
    build_calibration_audit_dataset,
    build_calibration_dataset,
    read_dataset_bundle,
    validate_train_audit_independence,
    write_dataset_bundle,
)
from occupancy_ratio_benchmark.calibration_estimators import EstimatorPaths
from occupancy_ratio_benchmark.calibration_eta import (
    build_eta_gate,
    eta_execution_limits,
)
from occupancy_ratio_benchmark.calibration_pipeline import (
    execute_aggregation,
    execute_deterministic_score,
    execute_learned_fold,
    grouped_assignment,
)
from occupancy_ratio_benchmark.calibration_protocol import (
    CalibrationManifestError,
    content_sha256,
    dataset_id,
    load_calibration_manifest,
    unit_index,
)
from occupancy_ratio_benchmark.calibration_supervisor import (
    UnitOutcome,
    run_supervised_units,
    summarize_outcomes,
)
from occupancy_ratio_benchmark.tabular import OptionalDatasetUnavailable


def prepare_datasets(
    *,
    manifest: Mapping[str, Any],
    run_root: Path,
    dataset_paths: DatasetPaths,
    units: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Materialize every unique dataset before launching estimator workers."""

    selected = list(units) if units is not None else [
        *manifest["atomic_fold_units"],
        *manifest["deterministic_score_units"],
    ]
    representative: dict[str, Mapping[str, Any]] = {}
    for unit in selected:
        representative.setdefault(dataset_id(unit), unit)
    existing_runtime: dict[str, float] = {}
    index_path = run_root / "dataset_index.json"
    if index_path.exists():
        try:
            existing_payload = json.loads(index_path.read_text(encoding="utf-8"))
            existing_runtime = {
                str(row["dataset_id"]): float(row["dataset_runtime_sec"])
                for row in existing_payload.get("datasets", [])
                if row.get("dataset_runtime_sec") is not None
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            existing_runtime = {}
    rows = []
    for data_id, unit in sorted(representative.items()):
        dataset_started = time.perf_counter()
        path = run_root / "datasets" / f"{data_id}.json"
        if path.exists():
            bundle = read_dataset_bundle(path)
            status = "resumed"
            dataset_runtime = existing_runtime.get(data_id)
        else:
            identity = _mapping(unit, "identity")
            bundle = build_calibration_dataset(
                cell=_mapping(unit, "cell"),
                axis_values=_mapping(identity, "axis_values"),
                resolved_config=_mapping(manifest, "resolved_config"),
                paths=dataset_paths,
            )
            write_dataset_bundle(path, bundle)
            status = "created"
            dataset_runtime = time.perf_counter() - dataset_started
        audit_bundle = None
        audit_digests: dict[str, str] = {}
        if _independent_audit_enabled(manifest):
            audit_path = run_root / "audit_datasets" / f"{data_id}.json"
            if audit_path.exists():
                audit_bundle = read_dataset_bundle(audit_path)
            else:
                identity = _mapping(unit, "identity")
                audit_bundle = build_calibration_audit_dataset(
                    cell=_mapping(unit, "cell"),
                    axis_values=_mapping(identity, "axis_values"),
                    resolved_config=_mapping(manifest, "resolved_config"),
                    paths=dataset_paths,
                )
                write_dataset_bundle(audit_path, audit_bundle)
            validate_train_audit_independence(bundle, audit_bundle)
            audit_digests = {
                f"audit_{key}": value
                for key, value in _dataset_digest_fields(audit_path).items()
            }
        digests = _dataset_digest_fields(path)
        assignment = grouped_assignment(
            manifest=manifest,
            unit=unit,
            source_groups=bundle.source_groups,
            initial_groups=bundle.initial_groups,
        )
        rows.append(
            {
                "dataset_id": data_id,
                "status": status,
                "benchmark_family": _mapping(unit, "cell")["benchmark_family"],
                "setting": bundle.dataset.setting,
                "sample_size": bundle.dataset.n,
                "source_groups": int(np.unique(bundle.source_groups).size),
                "initial_groups": int(np.unique(bundle.initial_groups).size),
                "folds": int(assignment.num_folds),
                "dataset_runtime_sec": dataset_runtime,
                "truth_precision_met": bundle.dataset.metadata.get("target_truth_precision_met"),
                **digests,
                **audit_digests,
                "audit_sample_size": (
                    None if audit_bundle is None else int(audit_bundle.dataset.n)
                ),
                "path": str(path),
            }
        )
    _atomic_json(index_path, {"datasets": rows})
    return rows


def execute_worker(
    *,
    manifest_path: Path,
    run_root: Path,
    unit_id: str,
    attempt: int,
    google_research: Path,
    dice_rl: Path,
    scope_rl: Path | None,
) -> int:
    """Execute exactly one fold, deterministic score, or aggregation unit."""

    started = time.time()
    try:
        manifest = load_calibration_manifest(manifest_path)
        units = unit_index(manifest)
        if unit_id not in units:
            raise ValueError(f"unit_id {unit_id!r} is not in the manifest")
        unit = units[unit_id]
        dataset_path = run_root / "datasets" / f"{dataset_id(unit)}.json"
        bundle = read_dataset_bundle(dataset_path)
        dataset_digests = _dataset_digest_fields(dataset_path)
        audit_bundle = None
        audit_digests: dict[str, str] = {}
        if _independent_audit_enabled(manifest):
            audit_path = run_root / "audit_datasets" / f"{dataset_id(unit)}.json"
            audit_bundle = read_dataset_bundle(audit_path)
            validate_train_audit_independence(bundle, audit_bundle)
            audit_digests = {
                f"audit_{key}": value
                for key, value in _dataset_digest_fields(audit_path).items()
            }
        kind = str(_mapping(unit, "identity").get("kind", ""))
        if kind == "cross_calibration_fold":
            result = execute_learned_fold(
                manifest=manifest,
                unit=unit,
                dataset=bundle.dataset,
                source_groups=bundle.source_groups,
                initial_groups=bundle.initial_groups,
                paths=EstimatorPaths(
                    google_research=google_research,
                    dice_rl=dice_rl,
                    scope_rl=scope_rl,
                ),
                audit_dataset=(None if audit_bundle is None else audit_bundle.dataset),
            )
            arrays = {
                "source_q": result.source_q,
                "next_q": result.next_q,
                "initial_q": result.initial_q,
            }
            if result.source_log_score is not None:
                arrays.update(
                    {
                        "source_log_score": result.source_log_score,
                        "next_log_score": result.next_log_score,
                        "initial_log_score": result.initial_log_score,
                    }
                )
            if result.audit_source_q is not None:
                arrays.update(
                    {
                        "audit_source_q": result.audit_source_q,
                        "audit_next_q": result.audit_next_q,
                        "audit_initial_q": result.audit_initial_q,
                    }
                )
            if result.audit_source_log_score is not None:
                arrays.update(
                    {
                        "audit_source_log_score": result.audit_source_log_score,
                        "audit_next_log_score": result.audit_next_log_score,
                        "audit_initial_log_score": result.audit_initial_log_score,
                    }
                )
            identity = _mapping(unit, "identity")
            axes = _mapping(identity, "axis_values")
            metadata = {
                **result.diagnostics,
                "dataset_id": dataset_id(unit),
                **dataset_digests,
                **audit_digests,
                "benchmark_family": _mapping(unit, "cell")["benchmark_family"],
                "sample_size": int(axes["sample_size"]),
                "gamma": float(axes["gamma"]),
                "seed": int(axes["seed"]),
                "held_out_fold": int(unit["held_out_fold"]),
                "estimator_id": str(identity["estimator_id"]),
            }
        elif kind == "deterministic_score":
            result = execute_deterministic_score(
                manifest=manifest,
                unit=unit,
                dataset=bundle.dataset,
                audit_dataset=(None if audit_bundle is None else audit_bundle.dataset),
            )
            arrays = {
                "source_q_by_fold": result.source_q_by_fold,
                "next_q_by_fold": result.next_q_by_fold,
                "initial_q_by_fold": result.initial_q_by_fold,
                "source_oracle": result.source_oracle,
                "next_oracle": result.next_oracle,
                "initial_oracle": result.initial_oracle,
            }
            if result.audit_source_q_by_fold is not None:
                arrays.update(
                    {
                        "audit_source_q_by_fold": result.audit_source_q_by_fold,
                        "audit_next_q_by_fold": result.audit_next_q_by_fold,
                        "audit_initial_q_by_fold": result.audit_initial_q_by_fold,
                    }
                )
            metadata = {
                "dataset_id": dataset_id(unit),
                **dataset_digests,
                **audit_digests,
                "distortion": result.distortion,
                "conceptual_fold_count": int(result.source_q_by_fold.shape[0]),
                "fit_runtime_sec": 0.0,
                "truth_used_for_learned_fit": False,
            }
        elif kind == "pooled_oof_cross_calibration":
            (
                source_matrix,
                next_matrix,
                initial_matrix,
                source_log_matrix,
                next_log_matrix,
                initial_log_matrix,
                fold_runtime,
                dependency_metadata,
            ) = _dependency_matrices(manifest, run_root, unit)
            audit_matrices = _dependency_audit_matrices(
                manifest, run_root, unit
            ) if audit_bundle is not None else (None,) * 6
            retry_count = int(
                sum(int(item.get("attempt", 0)) for item in dependency_metadata)
            )
            result = execute_aggregation(
                manifest=manifest,
                unit=unit,
                dataset=bundle.dataset,
                source_groups=bundle.source_groups,
                initial_groups=bundle.initial_groups,
                source_q_by_fold=source_matrix,
                next_q_by_fold=next_matrix,
                initial_q_by_fold=initial_matrix,
                source_log_score_by_fold=source_log_matrix,
                next_log_score_by_fold=next_log_matrix,
                initial_log_score_by_fold=initial_log_matrix,
                fold_runtime_sec=fold_runtime,
                retry_count=retry_count,
                paths=EstimatorPaths(
                    google_research=google_research,
                    dice_rl=dice_rl,
                    scope_rl=scope_rl,
                ),
                audit_dataset=(None if audit_bundle is None else audit_bundle.dataset),
                audit_source_groups=(None if audit_bundle is None else audit_bundle.source_groups),
                audit_initial_groups=(None if audit_bundle is None else audit_bundle.initial_groups),
                audit_source_q_by_fold=audit_matrices[0],
                audit_next_q_by_fold=audit_matrices[1],
                audit_initial_q_by_fold=audit_matrices[2],
                audit_source_log_score_by_fold=audit_matrices[3],
                audit_next_log_score_by_fold=audit_matrices[4],
                audit_initial_log_score_by_fold=audit_matrices[5],
            )
            arrays = dict(result.calibration_arrays)
            for candidate_id, values in result.candidate_arrays.items():
                for role, value in values.items():
                    arrays[f"candidate__{candidate_id}__{role}"] = value
            projection_count = int(
                sum(
                    int(item.get("metadata", {}).get("negative_projection_count", 0))
                    for item in dependency_metadata
                )
            )
            projection_mass = float(
                sum(
                    float(item.get("metadata", {}).get("negative_projection_mass", 0.0))
                    for item in dependency_metadata
                )
            )
            material_projection_count = int(
                sum(
                    int(
                        item.get("metadata", {}).get(
                            "material_negative_projection_count", 0
                        )
                    )
                    for item in dependency_metadata
                )
            )
            raw_minima = [
                float(value)
                for item in dependency_metadata
                if (
                    value := item.get("metadata", {}).get(
                        "raw_prediction_minimum"
                    )
                )
                is not None
            ]
            raw_minimum = min(raw_minima) if raw_minima else None
            fold_peak_memory = max(
                (
                    float(
                        item.get("metadata", {}).get(
                            "worker_peak_memory_mb", 0.0
                        )
                    )
                    for item in dependency_metadata
                ),
                default=0.0,
            )
            clamp_diagnostics = _fold_clamp_diagnostics(
                dependency_metadata,
                source_rows=int(bundle.dataset.n),
                initial_rows=int(np.asarray(bundle.dataset.initial_states).shape[0]),
            )
            rows = [dict(row) for row in result.rows]
            for row in rows:
                row["base_negative_projection_count_across_folds"] = projection_count
                row["base_negative_projection_mass_across_folds"] = projection_mass
                row[
                    "base_material_negative_projection_count_across_folds"
                ] = material_projection_count
                row["base_raw_prediction_minimum_across_folds"] = raw_minimum
                row["fold_peak_memory_mb"] = fold_peak_memory
                row.update(clamp_diagnostics)
            metadata = {
                **result.diagnostics,
                "dataset_id": dataset_id(unit),
                **dataset_digests,
                **audit_digests,
                "rows": rows,
                "depends_on": list(unit["depends_on"]),
                "dependency_payload_sha256": [item["payload_sha256"] for item in dependency_metadata],
                "negative_projection_count_across_folds": projection_count,
                "negative_projection_mass_across_folds": projection_mass,
                "material_negative_projection_count_across_folds": (
                    material_projection_count
                ),
                "raw_prediction_minimum_across_folds": raw_minimum,
                "fold_peak_memory_mb": fold_peak_memory,
                **clamp_diagnostics,
            }
        else:
            raise ValueError(f"unsupported unit kind {kind!r}")
        metadata = dict(metadata)
        worker_peak_memory = _peak_memory_mb()
        metadata["worker_peak_memory_mb"] = worker_peak_memory
        if kind == "pooled_oof_cross_calibration":
            for row in metadata["rows"]:
                row["aggregation_peak_memory_mb"] = worker_peak_memory
                row["peak_memory_mb"] = max(
                    float(row.get("fold_peak_memory_mb", 0.0)),
                    worker_peak_memory,
                )
        write_unit_artifact(
            run_root=run_root,
            manifest=manifest,
            unit=unit,
            arrays=arrays,
            metadata=metadata,
            attempt=int(attempt),
            started_at=started,
        )
        return 0
    except (CalibrationManifestError, ArtifactConflictError, ValueError, TypeError) as error:
        _worker_error(run_root, unit_id, "validation", error)
        return 2
    except (OptionalDatasetUnavailable, ImportError, ModuleNotFoundError, FileNotFoundError) as error:
        _worker_error(run_root, unit_id, "missing_backend", error)
        return 3
    except (FloatingPointError, OverflowError, RuntimeError) as error:
        _worker_error(run_root, unit_id, "numerical", error)
        return 4
    except OSError as error:
        _worker_error(run_root, unit_id, "transient_io", error)
        return 75
    except Exception as error:  # retryable unknown worker failure
        _worker_error(run_root, unit_id, "worker_exception", error)
        return 1


def run_manifest(
    *,
    manifest_path: Path,
    run_root: Path,
    python: Path,
    asset_cache: Path,
    google_research: Path,
    dice_rl: Path,
    scope_rl: Path | None,
    install_assets: bool,
    eta_gate_path: Path | None,
    shard_count: int,
    shard_index: int,
) -> dict[str, Any]:
    manifest = load_calibration_manifest(manifest_path)
    config = _mapping(manifest, "resolved_config")
    _validate_prepared_contract(
        manifest=manifest,
        manifest_path=manifest_path,
        python=python,
        asset_cache=asset_cache,
        external_roots=(google_research, dice_rl, scope_rl),
    )
    _validate_launch_gate(config, eta_gate_path, run_id=str(manifest["run_id"]))
    if config.get("evidence_tier") == "confirmatory" and install_assets:
        raise ValueError(
            "confirmatory runs require a pre-frozen asset cache; --install-assets is forbidden"
        )
    execution = _mapping(config, "execution")
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("invalid shard_count/shard_index")
    all_folds = list(manifest["atomic_fold_units"])
    fold_units = [
        unit
        for unit in all_folds
        if _shard_for(str(unit["unit_id"]), shard_count) == shard_index
    ]
    deterministic_units = list(manifest["deterministic_score_units"]) if shard_index == 0 else []
    prepare_datasets(
        manifest=manifest,
        run_root=run_root,
        dataset_paths=DatasetPaths(
            asset_cache=asset_cache,
            dice_rl=dice_rl,
            install_assets=install_assets,
        ),
        units=[*fold_units, *deterministic_units],
    )
    common_args = [
        "--manifest",
        str(manifest_path),
        "--run-root",
        str(run_root),
        "--google-research",
        str(google_research),
        "--dice-rl",
        str(dice_rl),
    ]
    if scope_rl is not None:
        common_args.extend(["--scope-rl", str(scope_rl)])

    def command(unit: Mapping[str, Any], attempt: int) -> list[str]:
        return [
            str(python),
            "-m",
            "occupancy_ratio_benchmark.calibration_run",
            "worker",
            *common_args,
            "--unit-id",
            str(unit["unit_id"]),
            "--attempt",
            str(attempt),
        ]

    def complete(unit: Mapping[str, Any]) -> bool:
        if not artifact_is_ready(run_root=run_root, manifest=manifest, unit=unit):
            return False
        payload = read_unit_artifact(run_root=run_root, manifest=manifest, unit=unit)
        dataset_path = run_root / "datasets" / f"{dataset_id(unit)}.json"
        expected = _dataset_digest_fields(dataset_path)
        metadata = payload.get("metadata", {})
        audit_expected: dict[str, str] = {}
        if _independent_audit_enabled(manifest):
            audit_path = run_root / "audit_datasets" / f"{dataset_id(unit)}.json"
            audit_expected = {
                f"audit_{key}": value
                for key, value in _dataset_digest_fields(audit_path).items()
            }
        return all(
            metadata.get(key) == value
            for key, value in {**expected, **audit_expected}.items()
        )

    outcomes: list[UnitOutcome] = []
    if deterministic_units:
        outcomes.extend(
            run_supervised_units(
                units=deterministic_units,
                manifest=manifest,
                run_root=run_root,
                command_builder=command,
                completion_check=complete,
                maximum_concurrency=1,
                timeout_sec=float(execution["timeout_sec"]),
                heartbeat_interval_sec=float(execution["heartbeat_interval_sec"]),
                stale_after_sec=float(execution["stale_after_sec"]),
                maximum_transient_retries=int(execution["maximum_transient_retries"]),
            )
        )
    if fold_units:
        outcomes.extend(
            run_supervised_units(
                units=fold_units,
                manifest=manifest,
                run_root=run_root,
                command_builder=command,
                completion_check=complete,
                maximum_concurrency=int(execution["maximum_concurrent_estimator_processes"]),
                timeout_sec=float(execution["timeout_sec"]),
                heartbeat_interval_sec=float(execution["heartbeat_interval_sec"]),
                stale_after_sec=float(execution["stale_after_sec"]),
                maximum_transient_retries=int(execution["maximum_transient_retries"]),
            )
        )

    aggregate_outcomes: list[UnitOutcome] = []
    if shard_count == 1:
        ready_aggregations = []
        for unit in manifest["aggregation_units"]:
            dependencies = [unit_index(manifest)[name] for name in unit["depends_on"]]
            if all(complete(dependency) for dependency in dependencies):
                ready_aggregations.append(unit)
            else:
                write_unit_state(
                    run_root=run_root,
                    manifest=manifest,
                    unit=unit,
                    status="failed",
                    attempt=0,
                    failure_type="dependency",
                    error="one or more fold artifacts are unavailable",
                )
        if ready_aggregations:
            aggregate_outcomes = run_supervised_units(
                units=ready_aggregations,
                manifest=manifest,
                run_root=run_root,
                command_builder=command,
                completion_check=complete,
                maximum_concurrency=int(
                    execution["maximum_concurrent_aggregation_processes"]
                ),
                timeout_sec=float(execution["timeout_sec"]),
                heartbeat_interval_sec=float(execution["heartbeat_interval_sec"]),
                stale_after_sec=float(execution["stale_after_sec"]),
                maximum_transient_retries=int(execution["maximum_transient_retries"]),
            )
            outcomes.extend(aggregate_outcomes)
    summary = summarize_outcomes(outcomes)
    summary.update(
        {
            "run_id": manifest["run_id"],
            "manifest": str(manifest_path),
            "run_root": str(run_root),
            "shard_count": shard_count,
            "shard_index": shard_index,
            "planned_fold_units_this_shard": len(fold_units),
            "planned_deterministic_units_this_shard": len(deterministic_units),
            "aggregation_units_run": len(aggregate_outcomes),
        }
    )
    _atomic_json(run_root / f"run_summary_shard_{shard_index}_of_{shard_count}.json", summary)
    write_flat_rows(manifest=manifest, run_root=run_root)
    return summary


def write_flat_rows(*, manifest: Mapping[str, Any], run_root: Path) -> Path:
    rows = []
    failures = []
    for unit in manifest["aggregation_units"]:
        try:
            payload = read_unit_artifact(run_root=run_root, manifest=manifest, unit=unit)
        except (ArtifactConflictError, OSError) as error:
            identity = _mapping(unit, "identity")
            axes = _mapping(identity, "axis_values")
            state_path = run_root / "unit_state" / f"{unit['unit_id']}.json"
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = {}
            failures.append(
                {
                    "unit_id": unit["unit_id"],
                    "study_id": identity.get("study_id"),
                    "cell_id": identity.get("cell_id"),
                    "estimator_id": identity.get("estimator_id"),
                    "score_distortion": axes.get("score_distortion"),
                    "sample_size": axes.get("sample_size"),
                    "gamma": axes.get("gamma"),
                    "seed": axes.get("seed"),
                    "status": state.get("status", "missing"),
                    "failure_type": state.get("failure_type", type(error).__name__),
                    "error": state.get("error", str(error)),
                    "attempt": state.get("attempt"),
                }
            )
            continue
        rows.extend(payload.get("metadata", {}).get("rows", []))
    path = run_root / "candidate_rows.json"
    _atomic_json(
        path,
        {
            "rows": rows,
            "row_count": len(rows),
            "failures": failures,
            "failure_count": len(failures),
        },
    )
    if rows:
        columns = sorted({key for row in rows for key in row})
        csv_path = run_root / "candidate_rows.csv"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{csv_path.name}.", suffix=".tmp", dir=csv_path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(
                descriptor, "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            key: json.dumps(value, sort_keys=True)
                            if isinstance(value, (list, dict))
                            else value
                            for key, value in row.items()
                        }
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, csv_path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return path


def status_payload(*, manifest: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    index = unit_index(manifest)
    ready = 0
    failed = 0
    running = 0
    by_kind: dict[str, dict[str, int]] = {}
    for unit_id, unit in index.items():
        kind = str(_mapping(unit, "identity").get("kind", "unknown"))
        counts = by_kind.setdefault(kind, {"planned": 0, "ok": 0, "failed": 0, "running": 0})
        counts["planned"] += 1
        try:
            is_ready = artifact_is_ready(run_root=run_root, manifest=manifest, unit=unit)
            if is_ready:
                payload = read_unit_artifact(
                    run_root=run_root, manifest=manifest, unit=unit
                )
                expected = _dataset_digest_fields(
                    run_root / "datasets" / f"{dataset_id(unit)}.json"
                )
                metadata = payload.get("metadata", {})
                is_ready = all(
                    metadata.get(key) == value for key, value in expected.items()
                )
        except ArtifactConflictError:
            is_ready = False
            counts["failed"] += 1
            failed += 1
            continue
        if is_ready:
            ready += 1
            counts["ok"] += 1
            continue
        state_path = run_root / "unit_state" / f"{unit_id}.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))["status"]
            except Exception:
                state = "failed"
            if state == "running":
                running += 1
                counts["running"] += 1
            elif state == "failed":
                failed += 1
                counts["failed"] += 1
    return {
        "run_id": manifest["run_id"],
        "planned_units": len(index),
        "ok_units": ready,
        "failed_units": failed,
        "running_units": running,
        "pending_units": len(index) - ready - failed - running,
        "by_kind": by_kind,
    }


def _dependency_matrices(
    manifest: Mapping[str, Any],
    run_root: Path,
    unit: Mapping[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    list[float],
    list[dict[str, Any]],
]:
    index = unit_index(manifest)
    dependencies = [index[name] for name in unit["depends_on"]]
    payloads = [
        read_unit_artifact(run_root=run_root, manifest=manifest, unit=dependency)
        for dependency in dependencies
    ]
    arrays = [
        read_unit_arrays(run_root=run_root, manifest=manifest, unit=dependency)
        for dependency in dependencies
    ]
    dataset_digests = _dataset_digest_fields(
        run_root / "datasets" / f"{dataset_id(unit)}.json"
    )
    for payload in payloads:
        metadata = payload.get("metadata", {})
        if not all(
            metadata.get(key) == value for key, value in dataset_digests.items()
        ):
            raise ArtifactConflictError(
                "dependency artifact was produced from a different dataset bundle"
            )
    base_fit_required = bool(_mapping(unit, "operation")["base_fit_required"])
    if base_fit_required:
        order = np.argsort(
            [int(dependency["held_out_fold"]) for dependency in dependencies]
        )
        source = np.stack([arrays[index]["source_q"] for index in order])
        next_q = np.stack([arrays[index]["next_q"] for index in order])
        initial = np.stack([arrays[index]["initial_q"] for index in order])
        log_presence = ["source_log_score" in arrays[index] for index in order]
        if any(log_presence) and not all(log_presence):
            raise ArtifactConflictError(
                "learned fold dependencies mix ratio and log-score payloads"
            )
        if all(log_presence):
            source_log = np.stack(
                [arrays[index]["source_log_score"] for index in order]
            )
            next_log = np.stack(
                [arrays[index]["next_log_score"] for index in order]
            )
            initial_log = np.stack(
                [arrays[index]["initial_log_score"] for index in order]
            )
        else:
            source_log = next_log = initial_log = None
        runtimes = [float(payloads[index]["metadata"]["fit_runtime_sec"]) for index in order]
        ordered_payloads = [payloads[index] for index in order]
        return (
            source,
            next_q,
            initial,
            source_log,
            next_log,
            initial_log,
            runtimes,
            ordered_payloads,
        )
    if len(arrays) != 1:
        raise ValueError("deterministic aggregation requires exactly one score dependency")
    return (
        arrays[0]["source_q_by_fold"],
        arrays[0]["next_q_by_fold"],
        arrays[0]["initial_q_by_fold"],
        None,
        None,
        None,
        [0.0],
        payloads,
    )


def _dependency_audit_matrices(
    manifest: Mapping[str, Any],
    run_root: Path,
    unit: Mapping[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Load external-audit predictions without touching fitted models."""

    index = unit_index(manifest)
    dependencies = [index[name] for name in unit["depends_on"]]
    payloads = [
        read_unit_artifact(run_root=run_root, manifest=manifest, unit=dependency)
        for dependency in dependencies
    ]
    arrays = [
        read_unit_arrays(run_root=run_root, manifest=manifest, unit=dependency)
        for dependency in dependencies
    ]
    audit_path = run_root / "audit_datasets" / f"{dataset_id(unit)}.json"
    expected = {
        f"audit_{key}": value
        for key, value in _dataset_digest_fields(audit_path).items()
    }
    for payload in payloads:
        metadata = payload.get("metadata", {})
        if not all(metadata.get(key) == value for key, value in expected.items()):
            raise ArtifactConflictError(
                "dependency artifact was produced from a different audit bundle"
            )
    base_fit_required = bool(_mapping(unit, "operation")["base_fit_required"])
    if base_fit_required:
        order = np.argsort(
            [int(dependency["held_out_fold"]) for dependency in dependencies]
        )
        required = ("audit_source_q", "audit_next_q", "audit_initial_q")
        if any(any(name not in arrays[index] for name in required) for index in order):
            raise ArtifactConflictError("learned fold dependency lacks audit predictions")
        source = np.stack([arrays[index]["audit_source_q"] for index in order])
        next_q = np.stack([arrays[index]["audit_next_q"] for index in order])
        initial = np.stack([arrays[index]["audit_initial_q"] for index in order])
        log_presence = ["audit_source_log_score" in arrays[index] for index in order]
        if any(log_presence) and not all(log_presence):
            raise ArtifactConflictError("learned folds mix audit ratio and log-score payloads")
        if all(log_presence):
            source_log = np.stack(
                [arrays[index]["audit_source_log_score"] for index in order]
            )
            next_log = np.stack(
                [arrays[index]["audit_next_log_score"] for index in order]
            )
            initial_log = np.stack(
                [arrays[index]["audit_initial_log_score"] for index in order]
            )
        else:
            source_log = next_log = initial_log = None
        return source, next_q, initial, source_log, next_log, initial_log
    if len(arrays) != 1:
        raise ValueError("deterministic audit aggregation requires one dependency")
    return (
        arrays[0]["audit_source_q_by_fold"],
        arrays[0]["audit_next_q_by_fold"],
        arrays[0]["audit_initial_q_by_fold"],
        None,
        None,
        None,
    )


def _independent_audit_enabled(manifest: Mapping[str, Any]) -> bool:
    config = _mapping(manifest, "resolved_config")
    evaluation = _mapping(config, "evaluation")
    calibration_error = _mapping(evaluation, "calibration_error")
    audit = calibration_error.get("independent_behavior_audit")
    return isinstance(audit, Mapping) and audit.get("enabled") is True


def _validate_launch_gate(
    config: Mapping[str, Any], path: Path | None, *, run_id: str | None = None
) -> None:
    if config.get("evidence_tier") != "confirmatory":
        return
    if path is None:
        raise ValueError("confirmatory runs require --eta-gate from all three runtime pilots")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise ValueError("confirmatory ETA gate is not passing")
    if payload.get("full_learned_fold_units") != 24_000:
        raise ValueError("ETA gate does not cover all 24,000 learned fold units")
    if payload.get("full_data_fit_units") != 2_400:
        raise ValueError("ETA gate does not cover all 2,400 full-data raw fits")
    if float(payload.get("guarded_projected_walltime_hours", float("inf"))) > 84.0:
        raise ValueError("ETA gate exceeds the frozen 84-hour ceiling")
    covered = payload.get("confirmatory_run_ids")
    if run_id is not None and isinstance(covered, list) and run_id not in covered:
        raise ValueError("ETA gate was computed for different confirmatory manifests")


def _collect_runtime_rows(manifest: Mapping[str, Any], run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for unit in manifest["atomic_fold_units"]:
        try:
            payload = read_unit_artifact(run_root=run_root, manifest=manifest, unit=unit)
        except (ArtifactConflictError, OSError):
            state_path = run_root / "unit_state" / f"{unit['unit_id']}.json"
            rows.append(
                {
                    "status": "failed",
                    "unit_id": unit["unit_id"],
                    "failure_state_present": state_path.exists(),
                }
            )
            continue
        metadata = payload["metadata"]
        rows.append(
            {
                "status": "ok",
                "unit_id": unit["unit_id"],
                "benchmark_family": metadata["benchmark_family"],
                "estimator_id": metadata["estimator_id"],
                "sample_size": metadata["sample_size"],
                "fit_runtime_sec": metadata["fit_runtime_sec"],
            }
        )
    return rows


def _collect_dataset_runtime_rows(run_root: Path) -> list[dict[str, Any]]:
    path = run_root / "dataset_index.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [
            {
                "status": "failed",
                "failure_type": "missing_dataset_index",
                "error": str(error),
            }
        ]
    rows = payload.get("datasets")
    if not isinstance(rows, list):
        return [{"status": "failed", "failure_type": "invalid_dataset_index"}]
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _eta_compatibility_reasons(
    pilot_manifests: Sequence[Mapping[str, Any]],
    confirmatory_manifests: Sequence[Mapping[str, Any]],
) -> list[str]:
    manifests = [*pilot_manifests, *confirmatory_manifests]
    reasons = []
    if len(pilot_manifests) != 3:
        reasons.append(
            f"ETA gate requires exactly three pilot manifests, received {len(pilot_manifests)}"
        )
    if len(confirmatory_manifests) != 3:
        reasons.append(
            "ETA gate requires exactly three confirmatory manifests, received "
            f"{len(confirmatory_manifests)}"
        )
    source_environment = {
        (
            manifest.get("provenance", {}).get("source_tree_sha256"),
            manifest.get("provenance", {}).get("environment_sha256"),
        )
        for manifest in manifests
    }
    if len(source_environment) != 1:
        reasons.append(
            "pilot and confirmatory manifests do not share source/environment digests"
        )
    registries = {
        hashlib.sha256(
            json.dumps(
                manifest.get("resolved_config", {}).get("estimator_registry"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        for manifest in manifests
    }
    if len(registries) != 1:
        reasons.append("pilot and confirmatory estimator schedules differ")
    if any(
        manifest.get("resolved_config", {}).get("evidence_tier") != "pilot"
        for manifest in pilot_manifests
    ):
        reasons.append("every runtime manifest supplied as a pilot must have tier 'pilot'")
    if any(
        manifest.get("resolved_config", {}).get("evidence_tier") != "confirmatory"
        for manifest in confirmatory_manifests
    ):
        reasons.append("every full manifest must have tier 'confirmatory'")
    return reasons


def _worker_error(run_root: Path, unit_id: str, failure_type: str, error: BaseException) -> None:
    path = run_root / "worker_errors"
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "unit_id": unit_id,
        "failure_type": failure_type,
        "exception_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "time_unix": time.time(),
    }
    _atomic_json(path / f"{unit_id}.{os.getpid()}.json", payload)


def _dataset_digest_fields(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactConflictError(f"cannot read dataset sidecar {path}: {error}") from error
    metadata_digest = payload.get("payload_sha256")
    array_digest = payload.get("array_sha256")
    if not isinstance(metadata_digest, str) or not isinstance(array_digest, str):
        raise ArtifactConflictError(f"dataset sidecar lacks content digests: {path}")
    return {
        "dataset_payload_sha256": metadata_digest,
        "dataset_array_sha256": array_digest,
    }


def _validate_prepared_contract(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    python: Path,
    asset_cache: Path,
    external_roots: Sequence[Path | None],
) -> None:
    _validate_source_and_environment(
        manifest=manifest,
        manifest_path=manifest_path,
        python=python,
    )
    path = manifest_path.parent / "dataset_contract.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"prepared dataset contract is unavailable: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("prepared dataset contract must be an object")
    recorded = payload.pop("sha256", None)
    expected = _mapping(manifest, "provenance").get("dataset_manifest_sha256")
    if recorded != expected or recorded != content_sha256(payload):
        raise ValueError("prepared dataset contract digest does not match the manifest")
    expected_cache = payload.get("asset_cache_root")
    if expected_cache is not None and Path(str(expected_cache)).resolve() != asset_cache.resolve():
        raise ValueError("runtime asset cache differs from the prepared dataset contract")
    for record in payload.get("asset_files", []):
        if not isinstance(record, Mapping):
            raise ValueError("dataset contract contains an invalid asset record")
        asset = asset_cache / str(record.get("path", ""))
        if (
            not asset.is_file()
            or int(asset.stat().st_size) != int(record.get("size", -1))
            or _sha256_file(asset) != record.get("sha256")
        ):
            raise ValueError(f"frozen asset changed or is missing: {asset}")
    expected_repositories = {
        str(Path(str(record["path"])).resolve()): record
        for record in payload.get("external_repositories", [])
        if isinstance(record, Mapping)
        and record.get("status") == "ok"
        and record.get("path")
    }
    supplied = {
        str(path.resolve()) for path in external_roots if path is not None and path.exists()
    }
    if set(expected_repositories) != supplied:
        raise ValueError("runtime external repository paths differ from the prepared contract")
    for repository, record in expected_repositories.items():
        state = _repository_state(Path(repository))
        for key in ("git_commit", "git_dirty", "git_dirty_digest"):
            if state.get(key) != record.get(key):
                raise ValueError(
                    f"external repository provenance changed for {repository}: {key}"
                )


def _validate_source_and_environment(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    python: Path,
) -> None:
    path = manifest_path.parent / "source_tree.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"prepared source-tree record is unavailable: {error}") from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("files"), list):
        raise ValueError("prepared source-tree record is invalid")
    roots = {
        "rltools": Path(str(payload.get("rltools_root", ""))).resolve(),
        "manuscript": Path(str(payload.get("manuscript_root", ""))).resolve(),
    }
    aggregate = hashlib.sha256()
    for record in payload["files"]:
        if not isinstance(record, Mapping):
            raise ValueError("source-tree record contains a non-object file entry")
        name = str(record.get("path", ""))
        prefix, separator, relative = name.partition("/")
        if not separator or prefix not in roots or not relative:
            raise ValueError(f"invalid frozen source path {name!r}")
        source = roots[prefix] / relative
        digest = _sha256_file(source) if source.is_file() else None
        if (
            digest != record.get("sha256")
            or int(source.stat().st_size if source.is_file() else -1)
            != int(record.get("size", -2))
        ):
            raise ValueError(f"frozen calibration source changed or is missing: {source}")
        aggregate.update(
            name.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\0"
        )
    observed = aggregate.hexdigest()
    provenance = _mapping(manifest, "provenance")
    if payload.get("sha256") != observed or provenance.get("source_tree_sha256") != observed:
        raise ValueError("source-tree aggregate digest does not match the manifest")
    try:
        result = subprocess.run(
            [str(python), "-m", "pip", "freeze", "--all"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot verify the frozen Python environment: {error}") from error
    lock = result.stdout.replace(b"\r\n", b"\n")
    if hashlib.sha256(lock).hexdigest() != provenance.get("environment_sha256"):
        raise ValueError("runtime Python environment differs from the prepared lock")


def _repository_state(path: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
                "-z",
            ],
            cwd=path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot verify external repository {path}: {error}") from error
    return {
        "git_commit": head.decode("ascii"),
        "git_dirty": bool(status),
        "git_dirty_digest": hashlib.sha256(status + diff).hexdigest()
        if status
        else None,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shard_for(unit_id: str, count: int) -> int:
    digest = hashlib.sha256(unit_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % int(count)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _peak_memory_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0**2 if sys.platform == "darwin" else 1024.0
    return peak / divisor


def _mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _fold_clamp_diagnostics(
    dependency_metadata: Sequence[Mapping[str, Any]],
    *,
    source_rows: int,
    initial_rows: int,
) -> dict[str, Any]:
    """Aggregate deployable Neural FORE clamp telemetry across fold artifacts."""

    fold_metadata = [
        item.get("metadata", {})
        for item in dependency_metadata
        if isinstance(item.get("metadata"), Mapping)
    ]
    enabled = [
        metadata
        for metadata in fold_metadata
        if metadata.get("base_upper_cap_enabled") is True
    ]
    lower_count = sum(
        int(metadata.get("base_prediction_lower_clamp_count", 0))
        for metadata in enabled
    )
    upper_count = sum(
        int(metadata.get("base_prediction_upper_clamp_count", 0))
        for metadata in enabled
    )
    query_count = len(enabled) * (2 * int(source_rows) + int(initial_rows))

    def bounds(name: str) -> list[float]:
        values = []
        for metadata in enabled:
            value = metadata.get(name)
            if value is not None and np.isfinite(float(value)):
                values.append(float(value))
        return values

    lower_bounds = bounds("base_prediction_log_lower_bound")
    upper_bounds = bounds("base_prediction_log_upper_bound")
    return {
        "base_prediction_clamp_enabled_fold_count": len(enabled),
        "base_prediction_lower_clamp_count_across_folds": lower_count,
        "base_prediction_upper_clamp_count_across_folds": upper_count,
        "base_prediction_clamp_fraction_across_folds": (
            float((lower_count + upper_count) / query_count)
            if query_count > 0
            else 0.0
        ),
        "base_prediction_log_lower_bound_min_across_folds": (
            min(lower_bounds) if lower_bounds else None
        ),
        "base_prediction_log_lower_bound_max_across_folds": (
            max(lower_bounds) if lower_bounds else None
        ),
        "base_prediction_log_upper_bound_min_across_folds": (
            min(upper_bounds) if upper_bounds else None
        ),
        "base_prediction_log_upper_bound_max_across_folds": (
            max(upper_bounds) if upper_bounds else None
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=Path, required=True)
    common.add_argument("--run-root", type=Path)

    worker = sub.add_parser("worker", parents=[common])
    worker.add_argument("--unit-id", required=True)
    worker.add_argument("--attempt", type=int, default=0)
    worker.add_argument("--google-research", type=Path, default=Path("/Users/larsvanderlaan/repos/google-research"))
    worker.add_argument("--dice-rl", type=Path, default=Path("/Users/larsvanderlaan/repos/dice_rl"))
    worker.add_argument("--scope-rl", type=Path)

    run = sub.add_parser("run", parents=[common])
    run.add_argument("--python", type=Path, default=Path(sys.executable))
    run.add_argument("--asset-cache", type=Path)
    run.add_argument("--google-research", type=Path, default=Path("/Users/larsvanderlaan/repos/google-research"))
    run.add_argument("--dice-rl", type=Path, default=Path("/Users/larsvanderlaan/repos/dice_rl"))
    run.add_argument("--scope-rl", type=Path)
    run.add_argument("--install-assets", action="store_true")
    run.add_argument("--eta-gate", type=Path)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)

    sub.add_parser("status", parents=[common])

    eta = sub.add_parser("eta")
    eta.add_argument("--pilot-run-root", action="append", type=Path, required=True)
    eta.add_argument("--confirmatory-manifest", action="append", type=Path, required=True)
    eta.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "eta":
        pilot_rows = []
        pilot_dataset_rows = []
        pilot_manifests = []
        for root in args.pilot_run_root:
            manifest = load_calibration_manifest(root / "manifest.json")
            pilot_manifests.append(manifest)
            pilot_rows.extend(_collect_runtime_rows(manifest, root))
            pilot_dataset_rows.extend(_collect_dataset_runtime_rows(root))
        full = [load_calibration_manifest(path) for path in args.confirmatory_manifest]
        (
            maximum_concurrency,
            maximum_aggregation_concurrency,
            ceiling_hours,
        ) = eta_execution_limits(full)
        gate = build_eta_gate(
            pilot_fold_rows=pilot_rows,
            pilot_dataset_rows=pilot_dataset_rows,
            confirmatory_manifests=full,
            maximum_concurrency=maximum_concurrency,
            maximum_aggregation_concurrency=maximum_aggregation_concurrency,
            ceiling_hours=ceiling_hours,
            expected_full_data_fit_units=2_400,
        )
        provenance_reasons = _eta_compatibility_reasons(pilot_manifests, full)
        if provenance_reasons:
            gate["blocking_reasons"].extend(provenance_reasons)
            gate["status"] = "blocked"
        gate["pilot_run_roots"] = [str(path) for path in args.pilot_run_root]
        gate["confirmatory_run_ids"] = [manifest["run_id"] for manifest in full]
        _atomic_json(args.output, gate)
        print(json.dumps(gate, indent=2, sort_keys=True))
        return 0 if gate["status"] == "pass" else 2

    manifest_path = args.manifest.resolve()
    run_root = manifest_path.parent if args.run_root is None else args.run_root.resolve()
    if args.command == "worker":
        return execute_worker(
            manifest_path=manifest_path,
            run_root=run_root,
            unit_id=args.unit_id,
            attempt=args.attempt,
            google_research=args.google_research,
            dice_rl=args.dice_rl,
            scope_rl=args.scope_rl,
        )
    manifest = load_calibration_manifest(manifest_path)
    if args.command == "status":
        payload = status_payload(manifest=manifest, run_root=run_root)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["failed_units"] == 0 else 2
    try:
        summary = run_manifest(
            manifest_path=manifest_path,
            run_root=run_root,
            python=args.python,
            asset_cache=(run_root / "asset_cache" if args.asset_cache is None else args.asset_cache),
            google_research=args.google_research,
            dice_rl=args.dice_rl,
            scope_rl=args.scope_rl,
            install_assets=args.install_assets,
            eta_gate_path=args.eta_gate,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )
    except Exception as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    failures = int(summary.get("status_counts", {}).get("failed", 0))
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
