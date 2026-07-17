"""Versioned configuration freeze for paper-ready clipped coverage runs."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from occupancy_ratio_benchmark._clipped_coverage_artifacts import atomic_write_text
from occupancy_ratio_benchmark._clipped_coverage_data import CoverageRunConfig


FREEZE_SCHEMA = "clipped-coverage-freeze-v1"
LINEAR_PILOT_SHA256 = (
    "f47d5d6f0751248b2306d1711594d0654c21740fefd091e99a57b4a8ffc5eece"
)
CLIPPED_FIELDS = {
    "clipped_gate_steps",
    "clipped_ratio_steps",
    "clipped_gate_learning_rate",
    "clipped_ratio_learning_rate",
    "clipped_inner_relative_tolerance",
    "clipped_inner_gradient_tolerance",
    "clipped_inner_patience",
}
STANDARD_FIELDS = {
    "standard_num_iterations",
    "standard_optimizer_steps",
    "standard_outer_tolerance",
    "standard_objective_tolerance",
    "standard_mass_tolerance",
    "standard_require_convergence",
}


def load_freeze_manifest(
    path: str | Path, *, expected_revision: str | None = None
) -> dict[str, Any]:
    """Load and strictly validate a backend-specific experiment freeze."""
    freeze_path = Path(path).resolve()
    payload = json.loads(freeze_path.read_text())
    if payload.get("schema") != FREEZE_SCHEMA:
        raise ValueError("freeze manifest has an unknown schema")
    if bool(payload.get("oracle_used_for_selection", True)):
        raise ValueError("freeze manifest selection must be oracle-free")
    revision = payload.get("source_revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("freeze manifest has no source revision")
    if expected_revision is not None and revision != expected_revision:
        raise ValueError(
            f"freeze revision {revision} does not match checkout {expected_revision}"
        )
    optimizers = payload.get("optimizer_configs")
    if not isinstance(optimizers, dict):
        raise ValueError("freeze manifest has no optimizer configurations")
    _validate_fields(optimizers.get("clipped_linear"), CLIPPED_FIELDS, "linear")
    _validate_fields(optimizers.get("clipped_neural"), CLIPPED_FIELDS, "neural")
    _validate_fields(optimizers.get("standard_linear"), STANDARD_FIELDS, "standard")
    pilot_artifacts = payload.get("pilot_artifacts")
    if not isinstance(pilot_artifacts, dict):
        raise ValueError("freeze manifest has no pilot artifact audit")
    linear = pilot_artifacts.get("clipped_linear")
    if not isinstance(linear, dict) or linear.get("sha256") != LINEAR_PILOT_SHA256:
        raise ValueError("freeze manifest has the wrong clipped-linear pilot hash")
    return {
        **payload,
        "path": str(freeze_path),
        "sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
    }


def apply_freeze(
    config: CoverageRunConfig, manifest: Mapping[str, Any]
) -> CoverageRunConfig:
    """Apply the frozen optimizer settings for a single-backend config."""
    if len(config.backends) != 1:
        raise ValueError("frozen coverage configs must contain one backend")
    backend = str(config.backends[0])
    optimizers = manifest["optimizer_configs"]
    updates = dict(optimizers[f"clipped_{backend}"])
    if backend == "linear" and "standard_fori" in config.methods:
        updates.update(optimizers["standard_linear"])
    return replace(config, **updates)


def write_freeze_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a validated freeze manifest atomically."""
    output = Path(path)
    candidate = {"schema": FREEZE_SCHEMA, **dict(payload)}
    atomic_write_text(output, json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    load_freeze_manifest(output)
    return output


def selection_artifact(path: str | Path) -> dict[str, str]:
    """Return the portable audit fields for a pilot-selection artifact."""
    value = Path(path).resolve()
    return {
        "path": str(value),
        "sha256": hashlib.sha256(value.read_bytes()).hexdigest(),
    }


def compare_linear_reproduction(
    reference_cells: str | Path,
    reproduced_cells: str | Path,
    *,
    prediction_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compare the selected pilot's fold predictions across source checkouts."""
    reference = _prediction_records(Path(reference_cells))
    reproduced = _prediction_records(Path(reproduced_cells))
    if set(reference) != set(reproduced):
        raise ValueError("linear verification cells do not match the reference design")
    maximum_prediction_error = 0.0
    gate_disagreements = 0
    gate_rows = 0
    for key in sorted(reference):
        old = reference[key]
        new = reproduced[key]
        if old["status"] != new["status"]:
            raise ValueError(f"linear verification status mismatch for cell {key}")
        prediction_error = np.max(
            np.abs(old["prediction"] - new["prediction"]), initial=0.0
        )
        maximum_prediction_error = max(
            maximum_prediction_error, float(prediction_error)
        )
        gate_disagreements += int(np.sum(old["gate"] != new["gate"]))
        gate_rows += int(old["gate"].size)
    matched = bool(
        maximum_prediction_error <= float(prediction_tolerance)
        and gate_disagreements == 0
    )
    return {
        "reference_cells": str(Path(reference_cells).resolve()),
        "reproduced_cells": str(Path(reproduced_cells).resolve()),
        "fold_cells": len(reference),
        "prediction_tolerance": float(prediction_tolerance),
        "maximum_prediction_abs_error": maximum_prediction_error,
        "gate_disagreements": gate_disagreements,
        "gate_rows": gate_rows,
        "reference_prediction_hash": _prediction_hash(reference),
        "reproduced_prediction_hash": _prediction_hash(reproduced),
        "matched": matched,
    }


def _prediction_records(root: Path) -> dict[tuple[int, int, float], dict[str, Any]]:
    records: dict[tuple[int, int, float], dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text())
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError(f"malformed fold cell {path}")
        row = rows[0]
        key = (
            int(row["fit_seed"]),
            int(row["fold"]),
            float(row["requested_mass"]),
        )
        if key in records:
            raise ValueError(f"duplicate linear verification cell {key}")
        clipped = row.get("clipped")
        if not isinstance(clipped, dict):
            raise ValueError(f"fold cell {path} has no clipped result")
        prediction = np.asarray(clipped.get("prediction", []), dtype=np.float64)
        gate = np.asarray(clipped.get("gate_prediction", []), dtype=np.float64)
        if prediction.ndim != 1 or gate.shape != prediction.shape:
            raise ValueError(f"fold cell {path} has malformed predictions")
        if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(gate)):
            raise ValueError(f"fold cell {path} has nonfinite predictions")
        records[key] = {
            "status": str(clipped.get("status", "")),
            "prediction": prediction,
            "gate": gate,
        }
    if not records:
        raise ValueError(f"no fold cells found in {root}")
    return records


def _prediction_hash(
    records: Mapping[tuple[int, int, float], Mapping[str, Any]],
) -> str:
    digest = hashlib.sha256()
    for key in sorted(records):
        record = records[key]
        digest.update(json.dumps(key, separators=(",", ":")).encode())
        digest.update(str(record["status"]).encode())
        digest.update(np.asarray(record["prediction"], dtype="<f8").tobytes())
        digest.update(np.asarray(record["gate"], dtype="<f8").tobytes())
    return digest.hexdigest()


def _validate_fields(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"freeze manifest has malformed {label} optimizer settings")


__all__ = [
    "CLIPPED_FIELDS",
    "FREEZE_SCHEMA",
    "LINEAR_PILOT_SHA256",
    "STANDARD_FIELDS",
    "apply_freeze",
    "compare_linear_reproduction",
    "load_freeze_manifest",
    "selection_artifact",
    "write_freeze_manifest",
]
