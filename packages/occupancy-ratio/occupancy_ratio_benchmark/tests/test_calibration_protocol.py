from __future__ import annotations

import json

import pytest

from occupancy_ratio_benchmark.calibration_protocol import (
    CalibrationManifestError,
    content_sha256,
    dataset_id,
    load_calibration_manifest,
)


def _manifest():
    fold = {
        "unit_id": "fold-a",
        "cell": {"cell_id": "cell", "benchmark_family": "random_tabular"},
        "identity": {
            "axis_values": {"sample_size": 10, "gamma": 0.9, "seed": 0},
            "config_sha256": "0" * 64,
            "estimator_id": "neural_fori",
        },
    }
    aggregate = {
        "unit_id": "aggregate-a",
        "cell": fold["cell"],
        "depends_on": ["fold-a"],
        "identity": fold["identity"],
    }
    payload = {
        "manifest_schema_version": 1,
        "run_id": "occ-cal-fixture",
        "resolved_config": {
            "protocol_id": "normalized_occupancy_cross_calibration_v1",
            "estimand": {
                "normalized": True,
                "target_mass": 1.0,
                "coverage_stopping": False,
                "coverage_gate": None,
            },
            "cross_calibration": {
                "folds": 2,
                "base_score_ratio_constraint": "neural_fori_held_out_current_finite_range_clamp_otherwise_positive_part_no_upper_cap",
                "base_query_normalization": False,
                "negative_projection_audit": "count_mass_and_raw_minimum",
                "calibration_fit": "single_map_on_pooled_oof_scores",
                "pointwise_aggregation": "median_of_calibrated_fold_predictors",
                "honest_three_way_split": False,
                "cross_moment_halves_are_fit_splits": False,
                "post_median_refit": False,
            },
            "pava": {"normalize_each_iteration": True, "ratio_upper_cap": None},
            "acceptance": {
                "scientific": {
                    "calibration_benefit": {
                        "scope": "learned_estimators_only",
                        "mechanism_scores_reported_separately": True,
                    },
                    "value_safety": {"scope": "learned_estimators_only"},
                    "controlled_ratio_corroboration": {
                        "scope": "learned_estimators_only"
                    },
                }
            },
            "execution": {
                "timeout_scope": "atomic_fold_unit",
                "parent_estimator_timeout_sec": None,
            },
        },
        "provenance": {"manifest_host": "fixture", "python_version": "fixture"},
        "atomic_fold_units": [fold],
        "deterministic_score_units": [],
        "aggregation_units": [aggregate],
    }
    immutable = json.loads(json.dumps(payload))
    immutable["provenance"].pop("manifest_host")
    immutable["provenance"].pop("python_version")
    payload["manifest_payload_sha256"] = content_sha256(immutable)
    return payload


def test_generated_smoke_manifest_is_accepted_and_dataset_id_is_shared(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    payload = _manifest()
    payload["atomic_fold_units"].append(
        {**payload["atomic_fold_units"][0], "unit_id": "fold-b"}
    )
    payload["aggregation_units"][0]["depends_on"].append("fold-b")
    immutable = json.loads(json.dumps(payload))
    immutable.pop("manifest_payload_sha256")
    immutable["provenance"].pop("manifest_host")
    immutable["provenance"].pop("python_version")
    payload["manifest_payload_sha256"] = content_sha256(immutable)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_calibration_manifest(manifest_path)
    learned = manifest["atomic_fold_units"]
    assert dataset_id(learned[0]) == dataset_id(learned[1])


def test_modified_manifest_is_rejected(tmp_path) -> None:
    payload = _manifest()
    payload["resolved_config"]["cross_calibration"]["honest_three_way_split"] = True
    path = tmp_path / "modified.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CalibrationManifestError):
        load_calibration_manifest(path)
