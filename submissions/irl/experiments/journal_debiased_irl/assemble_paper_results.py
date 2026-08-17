"""Assemble provenance-verified JASA confirmatory results.

The assembler consumes complete run directories, not free-standing summaries.
It recomputes every summary from the locked raw cells and reports coverage
calibration diagnostically.  Coverage is never used to select or suppress a
completed, predeclared cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from argparse import Namespace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from data_fusion_simulation import DataFusionRunResult, summarize_data_fusion_results
from jrssb_simulation import SingleRunResult, summarize_results
from run_jrssb_simulation import (
    EXPERIMENT_SCHEMA_VERSION,
    PAPER_PROTOCOL_ID,
    _configuration_identity,
    _file_sha256,
    _load_data_fusion_cell,
    _load_single_run_cell,
    write_data_fusion_latex_table,
    write_main_inference_latex_table,
)

PAPER_REPETITIONS = 300
FROZEN_FORE_SHA256 = "ba5810fa12774b558eecb7fba2c7c8b9ce575e0b1a796179fa3124a7fb87d862"
FROZEN_G_SHA256 = "fcb34bff7c0909100e8a6120bd1e2349bdffdd538067933a2efc721ed9275b0e"
FUSION_PILOT_MANIFEST_SHA256 = (
    "97030d7d631894f113e0b2dc35032583c7818ed7379095fcfe079ef30e09b760"
)
FUSION_PILOT_MEDIAN_SE = 0.05109142519375903
FORE_BUDGETS = (30, 100, 300)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    artifacts = script_dir / "artifacts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example1a-run-dir",
        type=Path,
        default=artifacts / "paper_confirmatory_v2_example1a",
    )
    parser.add_argument(
        "--example1b-run-dir",
        type=Path,
        default=artifacts / "paper_confirmatory_v2_example1b",
    )
    parser.add_argument(
        "--data-fusion-run-dir",
        type=Path,
        default=artifacts / "paper_confirmatory_v2_data_fusion",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=artifacts / "paper_confirmatory_v2_tables",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing locked artifact: {path}")
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _require_config(
    config: dict[str, object], expected: dict[str, object], path: Path
) -> None:
    mismatches = {
        key: {"observed": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Locked configuration mismatch in {path}: {mismatches}")


def _verify_artifact_manifest(run_dir: Path, required_paths: Sequence[Path]) -> str:
    manifest_path = run_dir / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("validated") is not True:
        raise ValueError(f"Unvalidated artifact manifest: {manifest_path}")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise ValueError(f"Malformed artifact manifest: {manifest_path}")

    def portable_path(raw_path: Path) -> Path:
        path = raw_path
        if raw_path.is_absolute():
            # Locked runs may have been produced in another checkout.  Rebase
            # paths below the experiment's artifacts directory while retaining
            # the recorded byte size and SHA-256 as the source of truth.
            parts = raw_path.parts
            if "artifacts" in parts:
                artifact_index = parts.index("artifacts")
                path = run_dir.parent.joinpath(*parts[artifact_index + 1 :])
        else:
            path = manifest_path.parent / raw_path
        return path.resolve()

    recorded: dict[Path, dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
            raise ValueError(f"Malformed artifact entry in {manifest_path}")
        path = portable_path(Path(str(entry["path"])))
        if path in recorded:
            raise ValueError(f"Duplicate artifact path in {manifest_path}: {path}")
        recorded[path] = entry
        if not path.is_file() or path.stat().st_size != int(entry["size_bytes"]):
            raise RuntimeError(f"Missing or size-mismatched artifact: {path}")
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Artifact SHA-256 mismatch: {path}")
    missing = [path for path in required_paths if portable_path(path) not in recorded]
    if missing:
        raise RuntimeError(
            f"Run manifest omits required locked artifacts: {', '.join(map(str, missing))}"
        )
    return _sha256(manifest_path)


def _parse_fold_values(
    encoded: str,
    *,
    field: str,
    n_folds: int,
    allowed: set[int] | None = None,
) -> list[float]:
    try:
        values = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid fold telemetry in {field}.") from exc
    if not isinstance(values, list) or len(values) != n_folds:
        raise ValueError(f"{field} must contain exactly {n_folds} fold values.")
    if any(value is None or not np.isfinite(float(value)) for value in values):
        raise ValueError(f"{field} contains a nonfinite fold value.")
    if allowed is not None and any(int(value) not in allowed for value in values):
        raise ValueError(f"{field} contains a non-predeclared stopping budget.")
    return [float(value) for value in values]


def _require_finite(row: object, fields: Sequence[str], *, cell: str) -> None:
    bad = [field for field in fields if not np.isfinite(float(getattr(row, field)))]
    if bad:
        raise ValueError(f"Nonfinite required fields in {cell}: {bad}")


def _validate_main_rows(
    rows: Sequence[SingleRunResult],
    *,
    example_id: str,
    n: int,
    seed: int,
    seed_offset: int,
    repetitions: int = PAPER_REPETITIONS,
) -> None:
    effective_offset = int(seed) + int(seed_offset)
    example_offset = {"1a": 101, "1b": 202}[example_id]
    expected_seeds = {
        10_000 * (rep + 1) + 97 * n + example_offset + effective_offset
        for rep in range(repetitions)
    }
    observed_seeds = [row.seed for row in rows]
    if len(rows) != repetitions or len(set(observed_seeds)) != repetitions:
        raise RuntimeError(f"{example_id}, n={n} does not contain {repetitions} unique rows.")
    if set(observed_seeds) != expected_seeds:
        raise RuntimeError(f"{example_id}, n={n} has the wrong deterministic seed set.")
    finite_fields = (
        "plugin_estimate",
        "if_estimate",
        "truth",
        "estimated_se",
        "ci_lower",
        "ci_upper",
        "covered",
        "plugin_error",
        "if_error",
        "reward_rmse",
        "reward_rmse_grid",
        "reward_rmse_stationary",
        "reward_rmse_rho_weighted",
        "bellman_residual_rmse",
        "ratio_q01",
        "ratio_q50",
        "ratio_q99",
        "weight_ess",
        "fore_selected_iterations",
        "fore_apbv_score",
        "fore_fit_seconds",
        "fore_training_rows",
        "fore_normalized_mass",
        "fore_logit_cap_fraction",
        "iid_estimated_se",
        "crossfit_fold_count",
        "ci_critical_value",
    )
    signed_fields = (
        "signed_positive_iterations",
        "signed_negative_iterations",
        "signed_positive_mass",
        "signed_negative_mass",
        "signed_positive_apbv_score",
        "signed_negative_apbv_score",
        "signed_positive_fit_seconds",
        "signed_negative_fit_seconds",
        "signed_positive_normalized_mass",
        "signed_negative_normalized_mass",
        "signed_positive_ess",
        "signed_negative_ess",
        "signed_positive_ratio_q99",
        "signed_negative_ratio_q99",
        "signed_positive_logit_cap_fraction",
        "signed_negative_logit_cap_fraction",
    )
    for row in rows:
        cell = f"{example_id}, n={n}, seed={row.seed}"
        if (
            row.example_id != example_id
            or row.n != n
            or row.ratio_mode != "neural-fore"
            or row.ratio_failure != 0.0
            or row.failure_message
        ):
            raise RuntimeError(f"Invalid or failed locked row in {cell}.")
        _require_finite(row, finite_fields, cell=cell)
        if row.estimated_se <= 0.0 or row.weight_ess <= 0.0:
            raise ValueError(f"Nonpositive inference scale or ESS in {cell}.")
        if row.covered not in (0.0, 1.0) or not (0.0 <= row.fore_logit_cap_fraction <= 1.0):
            raise ValueError(f"Invalid coverage or clipping telemetry in {cell}.")
        if row.crossfit_fold_count != 5.0:
            raise ValueError(f"Incorrect fold count in {cell}.")
        _parse_fold_values(
            row.fore_selected_iterations_by_fold,
            field="fore_selected_iterations_by_fold",
            n_folds=5,
            allowed=set(FORE_BUDGETS),
        )
        _parse_fold_values(
            row.fore_apbv_scores_by_fold,
            field="fore_apbv_scores_by_fold",
            n_folds=5,
        )
        if example_id == "1b":
            _require_finite(row, signed_fields, cell=cell)
            for field in (
                "signed_positive_iterations_by_fold",
                "signed_negative_iterations_by_fold",
            ):
                _parse_fold_values(
                    getattr(row, field), field=field, n_folds=5, allowed=set(FORE_BUDGETS)
                )
            for field in (
                "signed_positive_apbv_scores_by_fold",
                "signed_negative_apbv_scores_by_fold",
            ):
                _parse_fold_values(getattr(row, field), field=field, n_folds=5)


def _validate_fusion_rows(
    rows: Sequence[DataFusionRunResult],
    *,
    n: int,
    seed: int,
    repetitions: int = PAPER_REPETITIONS,
) -> None:
    expected_seeds = {seed + 100_003 * rep + 97 * n for rep in range(repetitions)}
    observed_seeds = [row.seed for row in rows]
    if len(rows) != repetitions or len(set(observed_seeds)) != repetitions:
        raise RuntimeError(f"Data fusion n={n} does not contain {repetitions} unique rows.")
    if set(observed_seeds) != expected_seeds:
        raise RuntimeError(f"Data fusion n={n} has the wrong deterministic seed set.")
    finite_fields = (
        "truth",
        "plugin_estimate",
        "if_estimate",
        "plugin_error",
        "if_error",
        "estimated_se",
        "ci_lower",
        "ci_upper",
        "covered",
        "ci_length",
        "reward_rmse",
        "g_rmse",
        "g_estimand_shift",
        "ratio_q99",
        "ratio_max",
        "ratio_ess",
        "fore_selected_iterations",
        "fore_apbv_score",
        "fore_fit_seconds",
        "fore_normalized_mass",
        "fore_logit_cap_fraction",
        "transition_selected_alpha",
        "transition_validation_mse",
        "transition_residual_sd_x",
        "transition_residual_sd_z",
        "iid_estimated_se",
        "crossfit_fold_count",
        "ci_critical_value",
        "behavior_probability_floor",
        "behavior_probability_clipping_fraction",
    )
    for row in rows:
        cell = f"data fusion, n={n}, seed={row.seed}"
        if (
            row.n != n
            or row.ratio_failure != 0.0
            or row.failure_message
            or row.data_fusion_policy_mode != "known-logging"
            or row.data_fusion_transition_mode != "sieve"
            or row.data_fusion_g_mode != "frozen"
            or row.data_fusion_ratio_mode != "neural-fore"
            or row.normalization_policy_mode != "known-uniform"
        ):
            raise RuntimeError(f"Invalid, failed, or nonlocked row in {cell}.")
        _require_finite(row, finite_fields, cell=cell)
        if row.estimated_se <= 0.0 or row.ratio_ess <= 0.0:
            raise ValueError(f"Nonpositive inference scale or ESS in {cell}.")
        if row.covered not in (0.0, 1.0) or row.crossfit_fold_count != 5.0:
            raise ValueError(f"Invalid coverage or fold count in {cell}.")
        if not (0.0 <= row.fore_logit_cap_fraction <= 1.0):
            raise ValueError(f"Invalid clipping telemetry in {cell}.")
        _parse_fold_values(
            row.fore_selected_iterations_by_fold,
            field="fore_selected_iterations_by_fold",
            n_folds=5,
            allowed=set(FORE_BUDGETS),
        )
        _parse_fold_values(
            row.fore_apbv_scores_by_fold,
            field="fore_apbv_scores_by_fold",
            n_folds=5,
        )


def _check_stored_summary(stored_path: Path, recomputed: pd.DataFrame) -> None:
    if not stored_path.is_file():
        raise FileNotFoundError(f"Missing stored summary: {stored_path}")
    stored = pd.read_csv(stored_path)
    if set(stored.columns) != set(recomputed.columns):
        raise RuntimeError(f"Stored summary schema differs from recomputation: {stored_path}")
    sort_columns = [column for column in ("example_id", "n") if column in stored.columns]
    stored = stored[recomputed.columns].sort_values(sort_columns).reset_index(drop=True)
    expected = recomputed.sort_values(sort_columns).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(stored, expected, check_exact=False, rtol=1e-12, atol=1e-12)
    except AssertionError as exc:
        raise RuntimeError(f"Stored summary does not match raw cells: {stored_path}") from exc


def _validate_selection_manifest(path: Path) -> None:
    payload = _read_json(path)
    if payload.get("uses_oracle_truth") is not False:
        raise ValueError(f"Non-truth-blind selector manifest: {path}")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or [
        candidate.get("num_iterations") for candidate in candidates
    ] != list(FORE_BUDGETS):
        raise ValueError(f"Unexpected locked FORE candidate library: {path}")


def _validate_run_identity(
    run_dir: Path,
    *,
    scope: str,
    expected_cells: Sequence[dict[str, object]],
    expected_config: dict[str, object],
) -> tuple[dict[str, object], Path, list[Path]]:
    run_config_path = run_dir / "run_config.json"
    run_payload = _read_json(run_config_path)
    config = run_payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Malformed run configuration: {run_config_path}")
    _require_config(config, expected_config, run_config_path)
    if run_payload.get("experiment_schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(f"Wrong experiment schema in {run_config_path}.")
    identity = _configuration_identity(Namespace(**config), scope)
    if run_payload.get("result_source_sha256") != identity["result_source_sha256"]:
        raise RuntimeError(f"Run source identity differs from live locked source: {run_dir}")
    cell_dir = run_dir / "cells" / str(identity["fingerprint"])
    identity_paths: list[Path] = []
    for cell in expected_cells:
        stem = (
            f"example_{cell['example_id']}_n{cell['n']}"
            if "example_id" in cell
            else f"data_fusion_n{cell['n']}"
        )
        identity_path = cell_dir / f"{stem}.identity.json"
        observed = _read_json(identity_path)
        if observed != identity | {"cell": cell}:
            raise RuntimeError(f"Cell identity mismatch: {identity_path}")
        identity_paths.append(identity_path)
    return config, cell_dir, identity_paths


def _validate_main_run(
    run_dir: Path, *, example_id: str, sample_sizes: Sequence[int]
) -> tuple[pd.DataFrame, dict[str, object]]:
    expected_config = {
        "paper_protocol_id": PAPER_PROTOCOL_ID,
        "mode": "monte-carlo",
        "examples": [example_id],
        "sample_sizes": list(sample_sizes),
        "repetitions": PAPER_REPETITIONS,
        "ratio_mode": "neural-fore",
        "allow_oracle_diagnostics": False,
        "seed": 404,
        "replication_seed_offset": 0,
        "crossfit_folds": 5,
        "crossfit_se_method": "iid",
        "crossfit_ci_method": "normal",
        "nuisance_sample_mode": "crossfit",
        "behavior_policy_design": "quadratic-logit",
        "main_sieve_mode": "fixed-quadratic",
        "main_sieve_degree": 2,
        "main_sieve_c": 10.0,
        "main_grid_points": 41,
        "fore_hidden_sizes": [64, 64],
        "fore_learning_rate": 0.0003,
        "fore_weight_decay": 0.001,
        "fore_iteration_budgets": list(FORE_BUDGETS),
        "fore_batch_size": 512,
        "fore_optimizer_steps": 5,
        "fore_target_action_draws": 4,
        "fore_logit_clip": 10.0,
        "fore_grad_clip_norm": 10.0,
        "fore_device": "cpu",
        "fore_max_training_rows": 20_000,
        "expected_fore_sha256": FROZEN_FORE_SHA256,
        f"example{example_id}_policy_estimator": "sieve-logit",
    }
    cells = [{"example_id": example_id, "n": int(n)} for n in sample_sizes]
    config, cell_dir, identities = _validate_run_identity(
        run_dir,
        scope="main-inference",
        expected_cells=cells,
        expected_config=expected_config,
    )
    frozen_path = Path(str(config["fore_frozen_config"]))
    if _file_sha256(frozen_path) != FROZEN_FORE_SHA256:
        raise RuntimeError("The main run does not use the pinned FORE configuration.")
    selection_path = run_dir / "selection_manifest.json"
    _validate_selection_manifest(selection_path)
    rows: list[SingleRunResult] = []
    cell_paths: list[Path] = []
    for n in sample_sizes:
        cell_path = cell_dir / f"example_{example_id}_n{n}.csv"
        cell_rows = _load_single_run_cell(cell_path)
        _validate_main_rows(
            cell_rows,
            example_id=example_id,
            n=int(n),
            seed=int(config["seed"]),
            seed_offset=int(config["replication_seed_offset"]),
        )
        rows.extend(cell_rows)
        cell_paths.append(cell_path)
    recomputed = pd.DataFrame(summarize_results(rows))
    _check_stored_summary(run_dir / "summary.csv", recomputed)
    required = [
        run_dir / "run_config.json",
        run_dir / "environment.json",
        run_dir / "results.csv",
        run_dir / "summary.csv",
        run_dir / "main_inference_table.tex",
        selection_path,
        frozen_path,
        *cell_paths,
        *identities,
    ]
    manifest_sha = _verify_artifact_manifest(run_dir, required)
    return recomputed, {"artifact_manifest_sha256": manifest_sha, "cells": len(rows)}


def _validate_fusion_run(run_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    sample_sizes = (2500, 5000, 10000)
    expected_config = {
        "paper_protocol_id": PAPER_PROTOCOL_ID,
        "mode": "data-fusion-confirmatory",
        "sample_sizes": list(sample_sizes),
        "repetitions": PAPER_REPETITIONS,
        "allow_oracle_diagnostics": False,
        "seed": 404,
        "crossfit_folds": 5,
        "crossfit_se_method": "iid",
        "crossfit_ci_method": "normal",
        "data_fusion_policy_mode": "known-logging",
        "data_fusion_transition_mode": "sieve",
        "data_fusion_g_mode": "frozen",
        "data_fusion_ratio_mode": "neural-fore",
        "data_fusion_target_gamma": 0.8,
        "data_fusion_repeated_splits": 1,
        "fore_hidden_sizes": [64, 64],
        "fore_learning_rate": 0.0003,
        "fore_weight_decay": 0.001,
        "fore_iteration_budgets": list(FORE_BUDGETS),
        "fore_batch_size": 512,
        "fore_optimizer_steps": 5,
        "fore_target_action_draws": 4,
        "fore_logit_clip": 10.0,
        "fore_grad_clip_norm": 10.0,
        "fore_device": "cpu",
        "fore_max_training_rows": 20_000,
        "expected_fore_sha256": FROZEN_FORE_SHA256,
        "expected_fusion_g_sha256": FROZEN_G_SHA256,
        "expected_fusion_pilot_manifest_sha256": FUSION_PILOT_MANIFEST_SHA256,
        "fusion_pilot_median_se": FUSION_PILOT_MEDIAN_SE,
    }
    cells = [{"n": int(n)} for n in sample_sizes]
    config, cell_dir, identities = _validate_run_identity(
        run_dir,
        scope="data-fusion",
        expected_cells=cells,
        expected_config=expected_config,
    )
    frozen_fore = Path(str(config["fore_frozen_config"]))
    frozen_g = Path(str(config["fusion_g_cache"]))
    pilot_manifest = Path(str(config["fusion_pilot_manifest"]))
    for path, expected in (
        (frozen_fore, FROZEN_FORE_SHA256),
        (frozen_g, FROZEN_G_SHA256),
        (pilot_manifest, FUSION_PILOT_MANIFEST_SHA256),
    ):
        if _file_sha256(path) != expected:
            raise RuntimeError(f"Pinned data-fusion input changed: {path}")
    fusion_manifest_path = run_dir / "data_fusion_manifest.json"
    fusion_manifest = _read_json(fusion_manifest_path)
    readiness = fusion_manifest.get("readiness")
    pilot_readiness = _read_json(pilot_manifest).get("readiness")
    if (
        fusion_manifest.get("mode") != "data-fusion-confirmatory"
        or not isinstance(readiness, dict)
        or readiness.get("passed") is not True
        or float(readiness.get("pilot_median_se", math.nan)) != FUSION_PILOT_MEDIAN_SE
        or readiness != pilot_readiness
    ):
        raise RuntimeError("The confirmatory data-fusion readiness lineage is invalid.")
    selection_path = run_dir / "selection_manifest.json"
    _validate_selection_manifest(selection_path)
    rows: list[DataFusionRunResult] = []
    cell_paths: list[Path] = []
    for n in sample_sizes:
        cell_path = cell_dir / f"data_fusion_n{n}.csv"
        cell_rows = _load_data_fusion_cell(cell_path)
        _validate_fusion_rows(cell_rows, n=n, seed=int(config["seed"]))
        rows.extend(cell_rows)
        cell_paths.append(cell_path)
    recomputed = pd.DataFrame(summarize_data_fusion_results(rows))
    _check_stored_summary(run_dir / "data_fusion_summary.csv", recomputed)
    required = [
        run_dir / "run_config.json",
        run_dir / "environment.json",
        run_dir / "data_fusion_results.csv",
        run_dir / "data_fusion_summary.csv",
        run_dir / "data_fusion_table.tex",
        fusion_manifest_path,
        selection_path,
        frozen_fore,
        frozen_g,
        pilot_manifest,
        *cell_paths,
        *identities,
    ]
    manifest_sha = _verify_artifact_manifest(run_dir, required)
    return recomputed, {"artifact_manifest_sha256": manifest_sha, "cells": len(rows)}


def _wilson_interval(
    successes: int, repetitions: int, z: float = 1.96
) -> tuple[float, float]:
    proportion = successes / repetitions
    denominator = 1.0 + z**2 / repetitions
    center = (proportion + z**2 / (2.0 * repetitions)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / repetitions
            + z**2 / (4.0 * repetitions**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def _coverage_calibration_rows(
    main: pd.DataFrame, fusion: pd.DataFrame
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for study, frame in (("main", main), ("data-fusion", fusion)):
        for record in frame.to_dict("records"):
            repetitions = int(record["repetitions"])
            successes = int(round(float(record["coverage_95"]) * repetitions))
            lower, upper = _wilson_interval(successes, repetitions)
            rows.append(
                {
                    "study": study,
                    "example_id": record.get("example_id", "data-fusion"),
                    "n": int(record["n"]),
                    "repetitions": repetitions,
                    "coverage_95": float(record["coverage_95"]),
                    "coverage_wilson_lower": lower,
                    "coverage_wilson_upper": upper,
                    "nominal_0.95_inside_wilson_interval": bool(lower <= 0.95 <= upper),
                    "mc_sd_to_mean_se": float(record["if_sd"])
                    / float(record["avg_estimated_se"]),
                    "absolute_bias_to_mean_se": abs(float(record["if_bias"]))
                    / float(record["avg_estimated_se"]),
                    "diagnostic_only": True,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    example1a, provenance_1a = _validate_main_run(
        args.example1a_run_dir, example_id="1a", sample_sizes=(2500, 5000, 10000)
    )
    example1b, provenance_1b = _validate_main_run(
        args.example1b_run_dir, example_id="1b", sample_sizes=(25000, 50000, 100000)
    )
    fusion, provenance_fusion = _validate_fusion_run(args.data_fusion_run_dir)
    main_summary = pd.concat([example1a, example1b], ignore_index=True)
    calibration = pd.DataFrame(_coverage_calibration_rows(main_summary, fusion))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_summary_path = args.output_dir / "main_summary.csv"
    fusion_summary_path = args.output_dir / "data_fusion_summary.csv"
    calibration_path = args.output_dir / "coverage_calibration_audit.csv"
    main_table_path = args.output_dir / "main_inference_table.tex"
    fusion_table_path = args.output_dir / "data_fusion_table.tex"
    main_summary.to_csv(main_summary_path, index=False)
    fusion.to_csv(fusion_summary_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    write_main_inference_latex_table(main_summary.to_dict("records"), main_table_path)
    write_data_fusion_latex_table(fusion.to_dict("records"), fusion_table_path)

    outputs = [
        main_summary_path,
        fusion_summary_path,
        calibration_path,
        main_table_path,
        fusion_table_path,
    ]
    manifest = {
        "paper_protocol_id": PAPER_PROTOCOL_ID,
        "expected_repetitions_per_cell": PAPER_REPETITIONS,
        "protocol_complete": True,
        "all_locked_cells_reported": True,
        "coverage_calibration_is_diagnostic_only": True,
        "truth_used_for_selection_or_suppression": False,
        "input_runs": {
            str(args.example1a_run_dir.resolve()): provenance_1a,
            str(args.example1b_run_dir.resolve()): provenance_1b,
            str(args.data_fusion_run_dir.resolve()): provenance_fusion,
        },
        "outputs": {str(path.resolve()): _sha256(path) for path in outputs},
    }
    with (args.output_dir / "artifact_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
