"""Runtime validation for immutable normalized-calibration manifests."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_ID = "normalized_occupancy_cross_calibration_v1"
UNIT_COLLECTIONS = (
    "atomic_fold_units",
    "deterministic_score_units",
    "aggregation_units",
)


class CalibrationManifestError(ValueError):
    """Raised when a run manifest is modified or violates the fit contract."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_calibration_manifest(path: str | Path) -> dict[str, Any]:
    """Read and fully validate a content-addressed run manifest."""

    manifest_path = Path(path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationManifestError(f"cannot read manifest {manifest_path}: {error}") from error
    if not isinstance(payload, dict):
        raise CalibrationManifestError("manifest must contain a JSON object")
    _validate_manifest_payload(payload)
    return payload


def unit_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the unique unit-id index after collection/type validation."""

    index: dict[str, dict[str, Any]] = {}
    for collection in UNIT_COLLECTIONS:
        units = manifest.get(collection)
        if not isinstance(units, list):
            raise CalibrationManifestError(f"manifest.{collection} must be an array")
        for raw in units:
            if not isinstance(raw, dict):
                raise CalibrationManifestError(f"manifest.{collection} contains a non-object")
            unit_id = raw.get("unit_id")
            if not isinstance(unit_id, str) or not unit_id:
                raise CalibrationManifestError(f"manifest.{collection} has an invalid unit_id")
            if unit_id in index:
                raise CalibrationManifestError(f"duplicate unit_id {unit_id!r}")
            index[unit_id] = raw
    return index


def dataset_identity(unit: Mapping[str, Any]) -> dict[str, Any]:
    """Return the estimator/fold-free identity of a shared sampled dataset."""

    identity = unit.get("identity")
    cell = unit.get("cell")
    if not isinstance(identity, Mapping) or not isinstance(cell, Mapping):
        raise CalibrationManifestError("unit must contain identity and cell objects")
    axes = identity.get("axis_values")
    if not isinstance(axes, Mapping):
        raise CalibrationManifestError("unit identity.axis_values must be an object")
    sampling_axes = {
        str(key): value
        for key, value in axes.items()
        if key not in {"score_distortion"}
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "config_sha256": identity.get("config_sha256"),
        "cell": dict(cell),
        "sampling_axes": sampling_axes,
    }


def dataset_id(unit: Mapping[str, Any]) -> str:
    return f"dataset-{content_sha256(dataset_identity(unit))[:24]}"


def stable_uint32(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little", signed=False)


def _validate_manifest_payload(manifest: Mapping[str, Any]) -> None:
    if manifest.get("manifest_schema_version") != 1:
        raise CalibrationManifestError("manifest_schema_version must be 1")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.startswith("occ-cal-"):
        raise CalibrationManifestError("manifest has an invalid run_id")
    config = manifest.get("resolved_config")
    if not isinstance(config, Mapping):
        raise CalibrationManifestError("manifest.resolved_config must be an object")
    if config.get("protocol_id") != PROTOCOL_ID:
        raise CalibrationManifestError(f"protocol_id must be {PROTOCOL_ID!r}")
    estimand = config.get("estimand")
    if not isinstance(estimand, Mapping):
        raise CalibrationManifestError("manifest estimand is missing")
    if (
        estimand.get("normalized") is not True
        or estimand.get("target_mass") != 1.0
        or estimand.get("coverage_stopping") is not False
        or estimand.get("coverage_gate") is not None
    ):
        raise CalibrationManifestError("runtime supports normalized occupancy without coverage stopping only")
    cross = config.get("cross_calibration")
    if not isinstance(cross, Mapping):
        raise CalibrationManifestError("manifest cross_calibration is missing")
    expected = {
        "base_score_ratio_constraint": "neural_fori_held_out_current_finite_range_clamp_otherwise_positive_part_no_upper_cap",
        "base_query_normalization": False,
        "negative_projection_audit": "count_mass_and_raw_minimum",
        "calibration_fit": "single_map_on_pooled_oof_scores",
        "pointwise_aggregation": "median_of_calibrated_fold_predictors",
        "honest_three_way_split": False,
        "cross_moment_halves_are_fit_splits": False,
        "post_median_refit": False,
    }
    for key, value in expected.items():
        if cross.get(key) != value:
            raise CalibrationManifestError(f"cross_calibration.{key} has drifted")
    evaluation = config.get("evaluation")
    calibration_error = (
        evaluation.get("calibration_error") if isinstance(evaluation, Mapping) else None
    )
    audit = (
        calibration_error.get("independent_behavior_audit")
        if isinstance(calibration_error, Mapping)
        else None
    )
    if isinstance(audit, Mapping) and audit.get("enabled") is True:
        expected_audit = {
            "design": "external_group_disjoint_c_a_b",
            "basis_role": "c_deployed_candidate_quantile_bins_and_gram",
            "moment_roles": "a_b_cross_product",
            "fit_use": "none",
        }
        for key, value in expected_audit.items():
            if audit.get(key) != value:
                raise CalibrationManifestError(
                    f"independent_behavior_audit.{key} has drifted"
                )
        fraction = audit.get("d4rl_raw_episode_fraction")
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0.0 < float(fraction) < 1.0:
            raise CalibrationManifestError(
                "independent behavior audit fraction must lie in (0,1)"
            )
    folds = cross.get("folds")
    if not isinstance(folds, int) or isinstance(folds, bool) or folds < 2:
        raise CalibrationManifestError("cross_calibration.folds must be an integer >=2")
    pava = config.get("pava")
    if not isinstance(pava, Mapping) or pava.get("normalize_each_iteration") is not True:
        raise CalibrationManifestError("PAVA must normalize each iteration")
    if pava.get("ratio_upper_cap") is not None:
        raise CalibrationManifestError("PAVA ratio caps are forbidden")
    minimum_block_observations = pava.get(
        "minimum_boundary_block_observations", 1
    )
    if (
        not isinstance(minimum_block_observations, int)
        or isinstance(minimum_block_observations, bool)
        or minimum_block_observations <= 0
    ):
        raise CalibrationManifestError(
            "pava.minimum_boundary_block_observations must be a positive integer"
        )
    acceptance = config.get("acceptance")
    scientific = acceptance.get("scientific") if isinstance(acceptance, Mapping) else None
    if not isinstance(scientific, Mapping):
        raise CalibrationManifestError("scientific acceptance gates are missing")
    for endpoint in (
        "calibration_benefit",
        "value_safety",
        "controlled_ratio_corroboration",
    ):
        gate = scientific.get(endpoint)
        if not isinstance(gate, Mapping) or gate.get("scope") != "learned_estimators_only":
            raise CalibrationManifestError(
                f"scientific {endpoint} must use learned estimators only"
            )
    if (
        scientific["calibration_benefit"].get(
            "mechanism_scores_reported_separately"
        )
        is not True
    ):
        raise CalibrationManifestError("exact-score mechanisms must be reported separately")
    execution = config.get("execution")
    if not isinstance(execution, Mapping) or execution.get("timeout_scope") != "atomic_fold_unit":
        raise CalibrationManifestError("timeouts must apply to individual fold units")
    if execution.get("parent_estimator_timeout_sec") is not None:
        raise CalibrationManifestError("parent estimator timeout is forbidden")
    expected_digest = manifest.get("manifest_payload_sha256")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise CalibrationManifestError("manifest payload digest is missing")
    immutable = deepcopy(dict(manifest))
    immutable.pop("manifest_payload_sha256", None)
    provenance = immutable.get("provenance")
    if not isinstance(provenance, dict):
        raise CalibrationManifestError("manifest provenance is missing")
    provenance.pop("manifest_host", None)
    provenance.pop("python_version", None)
    observed_digest = content_sha256(immutable)
    if observed_digest != expected_digest:
        raise CalibrationManifestError(
            "manifest payload digest mismatch; refusing modified run definition"
        )
    index = unit_index(manifest)
    aggregation_ids = {
        unit["unit_id"] for unit in manifest["aggregation_units"]
    }
    dependency_ids = set(index) - aggregation_ids
    for unit in manifest["aggregation_units"]:
        dependencies = unit.get("depends_on")
        if not isinstance(dependencies, list) or not dependencies:
            raise CalibrationManifestError("aggregation unit has no dependencies")
        if not set(dependencies).issubset(dependency_ids):
            raise CalibrationManifestError("aggregation unit references an unknown dependency")


__all__: Sequence[str] = (
    "CalibrationManifestError",
    "PROTOCOL_ID",
    "UNIT_COLLECTIONS",
    "canonical_bytes",
    "content_sha256",
    "dataset_id",
    "dataset_identity",
    "load_calibration_manifest",
    "stable_uint32",
    "unit_index",
)
