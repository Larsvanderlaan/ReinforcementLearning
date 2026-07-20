"""Submission tables, figures, and claim gates for calibration runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from occupancy_ratio_benchmark.calibration_protocol import load_calibration_manifest
from occupancy_ratio_benchmark.calibration_run import status_payload


PAIR_KEYS = (
    "study_id",
    "cell_id",
    "sample_size",
    "gamma",
    "seed",
    "estimator_id",
    "score_distortion",
)
SCALAR_ID = "scalar_normalized_pointwise_median"
PAVA_ID = "pava_pointwise_median"
TRACK_BY_FAMILY = {
    "random_tabular": "controlled",
    "linear_gaussian": "controlled",
    "d4rl_matched": "d4rl",
    "dice_rl_cartpole": "cartpole",
}


def build_report(
    *,
    run_roots: Sequence[str | Path],
    output_dir: str | Path,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 20_260_720,
) -> dict[str, Any]:
    """Build a suite report without converting incomplete output into claims."""

    roots = [Path(path).resolve() for path in run_roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("run_roots must be a nonempty unique list")
    repetitions = int(bootstrap_repetitions)
    if repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive")
    manifests = [load_calibration_manifest(root / "manifest.json") for root in roots]
    rows = []
    statuses = []
    for root, manifest in zip(roots, manifests, strict=True):
        candidate_path = root / "candidate_rows.json"
        if candidate_path.exists():
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            rows.extend(payload.get("rows", []))
        statuses.append(status_payload(manifest=manifest, run_root=root))
    pairs, duplicate_candidates = pair_candidate_rows(rows)
    planned_pairs = int(sum(len(manifest["aggregation_units"]) for manifest in manifests))
    finite_pairs = len(pairs)
    evidence_tiers = {str(manifest["resolved_config"].get("evidence_tier")) for manifest in manifests}
    smoke_only = evidence_tiers == {"smoke"}

    learned_pairs = [row for row in pairs if row["is_learned_estimator"]]
    calibration_rows = [
        row
        for row in learned_pairs
        if np.isfinite(row["calibration_delta"])
    ]
    calibration_ci = cluster_bootstrap_interval(
        calibration_rows,
        value_key="calibration_delta",
        repetitions=repetitions,
        seed=_mixed_seed(bootstrap_seed, "calibration-overall"),
    )
    track_summaries = []
    for track in ("controlled", "d4rl", "cartpole"):
        all_selected = [row for row in pairs if row["track"] == track]
        selected = [row for row in all_selected if row["is_learned_estimator"]]
        track_summaries.append(
            {
                "track": track,
                "planned_pairs": _planned_track_pairs(
                    manifests, track, learned_only=True
                ),
                "finite_pairs": len(selected),
                "planned_pairs_including_mechanisms": _planned_track_pairs(
                    manifests, track, learned_only=False
                ),
                "finite_pairs_including_mechanisms": len(all_selected),
                "calibration_scalar": _interval(
                    selected,
                    "calibration_scalar",
                    repetitions,
                    bootstrap_seed,
                    f"calibration-scalar-{track}",
                ),
                "calibration_pava": _interval(
                    selected,
                    "calibration_pava",
                    repetitions,
                    bootstrap_seed,
                    f"calibration-pava-{track}",
                ),
                "calibration": cluster_bootstrap_interval(
                    selected,
                    value_key="calibration_delta",
                    repetitions=repetitions,
                    seed=_mixed_seed(bootstrap_seed, f"calibration-{track}"),
                ),
                "policy_value_scalar": _interval(
                    selected,
                    "value_abs_error_scalar",
                    repetitions,
                    bootstrap_seed,
                    f"value-scalar-{track}",
                ),
                "policy_value_pava": _interval(
                    selected,
                    "value_abs_error_pava",
                    repetitions,
                    bootstrap_seed,
                    f"value-pava-{track}",
                ),
                "policy_value": _interval(
                    selected,
                    "value_delta",
                    repetitions,
                    bootstrap_seed,
                    f"value-delta-{track}",
                ),
                "value_safety": cluster_bootstrap_interval(
                    selected,
                    value_key="value_delta_minus_margin",
                    repetitions=repetitions,
                    seed=_mixed_seed(bootstrap_seed, f"value-{track}"),
                ),
                "ratio_mse_scalar": _interval(
                    selected,
                    "ratio_mse_scalar",
                    repetitions,
                    bootstrap_seed,
                    f"mse-scalar-{track}",
                ),
                "ratio_mse_pava": _interval(
                    selected,
                    "ratio_mse_pava",
                    repetitions,
                    bootstrap_seed,
                    f"mse-pava-{track}",
                ),
                "ratio_mse": cluster_bootstrap_interval(
                    selected,
                    value_key="ratio_mse_delta",
                    repetitions=repetitions,
                    seed=_mixed_seed(bootstrap_seed, f"mse-{track}"),
                ),
                "ratio_kl_scalar": _interval(
                    selected,
                    "ratio_kl_scalar",
                    repetitions,
                    bootstrap_seed,
                    f"kl-scalar-{track}",
                ),
                "ratio_kl_pava": _interval(
                    selected,
                    "ratio_kl_pava",
                    repetitions,
                    bootstrap_seed,
                    f"kl-pava-{track}",
                ),
                "ratio_kl": cluster_bootstrap_interval(
                    selected,
                    value_key="ratio_kl_delta",
                    repetitions=repetitions,
                    seed=_mixed_seed(bootstrap_seed, f"kl-{track}"),
                ),
            }
        )
    mechanism_summaries = _mechanism_summaries(
        pairs,
        repetitions=repetitions,
        bootstrap_seed=bootstrap_seed,
    )

    all_units_complete = all(
        status["ok_units"] == status["planned_units"] and status["failed_units"] == 0
        for status in statuses
    )
    all_pairs_complete = finite_pairs == planned_pairs and duplicate_candidates == 0
    pava_converged = all(
        row.get("pava_converged") is True
        for row in rows
        if row.get("candidate_id") == PAVA_ID and row.get("status") == "ok"
    ) and bool(pairs)
    shared_source = len(
        {
            (
                manifest["provenance"].get("git_commit"),
                manifest["provenance"].get("source_tree_sha256"),
                manifest["provenance"].get("environment_sha256"),
            )
            for manifest in manifests
        }
    ) == 1
    clean_source_for_evidence = smoke_only or all(
        manifest["provenance"].get("git_dirty") is False for manifest in manifests
    )
    infrastructure_pass = bool(
        all_units_complete
        and all_pairs_complete
        and pava_converged
        and shared_source
        and clean_source_for_evidence
    )
    calibration_pass = bool(
        calibration_ci["status"] == "ok" and calibration_ci["ci95_high"] < 0.0
    )
    value_track_gates = {
        row["track"]: bool(
            row["value_safety"]["status"] == "ok"
            and row["value_safety"]["ci95_high"] <= 0.0
        )
        for row in track_summaries
    }
    required_tracks_present = all(row["planned_pairs"] > 0 for row in track_summaries)
    value_pass = required_tracks_present and all(value_track_gates.values())
    scientific_enabled = not smoke_only and all(
        manifest["resolved_config"].get("acceptance", {})
        .get("scientific", {})
        .get("enabled", True)
        is not False
        for manifest in manifests
    )
    if smoke_only:
        readiness = "smoke_only_not_evidence"
    elif not infrastructure_pass:
        readiness = "incomplete_not_paper_ready"
    elif scientific_enabled and calibration_pass and value_pass:
        readiness = "paper_ready_claim_gates_pass"
    else:
        readiness = "complete_report_claim_gates_not_all_passed"

    result = {
        "schema_version": 1,
        "paper_readiness_status": readiness,
        "run_roots": [str(root) for root in roots],
        "run_ids": [manifest["run_id"] for manifest in manifests],
        "evidence_tiers": sorted(evidence_tiers),
        "planned_pairs": planned_pairs,
        "finite_pairs": finite_pairs,
        "failed_or_missing_pairs": planned_pairs - finite_pairs,
        "duplicate_candidate_rows": duplicate_candidates,
        "infrastructure": {
            "pass": infrastructure_pass,
            "all_units_complete": all_units_complete,
            "all_pairs_complete": all_pairs_complete,
            "pava_converged": pava_converged,
            "shared_source_and_environment": shared_source,
            "clean_source_for_evidence": clean_source_for_evidence,
            "run_statuses": statuses,
        },
        "scientific": {
            "enabled": scientific_enabled,
            "primary_scope": "learned_estimators_only",
            "calibration_benefit_pass": calibration_pass if scientific_enabled else None,
            "calibration_overall": calibration_ci,
            "value_safety_pass": value_pass if scientific_enabled else None,
            "value_track_gates": value_track_gates,
            "required_tracks_present": required_tracks_present,
            "negative_control_excluded_from_benefit_gate": True,
        },
        "track_summaries": track_summaries,
        "mechanism_summaries": mechanism_summaries,
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": int(bootstrap_seed),
        "truth_based_selection_count": 0,
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "paper_readiness.json", result)
    _write_csv(output / "paired_candidate_deltas.csv", pairs)
    _write_csv(output / "track_summary.csv", _flatten_track_summaries(track_summaries))
    _write_csv(
        output / "mechanism_summary.csv",
        _flatten_mechanism_summaries(mechanism_summaries),
    )
    _write_markdown(output / "paper_readiness.md", result)
    _write_latex(output / "track_summary.tex", track_summaries)
    _write_controlled_latex(output / "controlled_accuracy_summary.tex", track_summaries)
    _write_mechanism_latex(output / "mechanism_summary.tex", mechanism_summaries)
    figure_status = _write_figures(output, track_summaries)
    result["figure_status"] = figure_status
    _atomic_json(output / "paper_readiness.json", result)
    return result


def pair_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Pair PAVA and scalar rows and compute predeclared deltas."""

    lookup: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    duplicates = 0
    for row in rows:
        candidate = str(row.get("candidate_id", ""))
        if candidate not in {SCALAR_ID, PAVA_ID} or row.get("status") != "ok":
            continue
        key = tuple(row.get(name) for name in PAIR_KEYS)
        slot = lookup.setdefault(key, {})
        if candidate in slot:
            duplicates += 1
        else:
            slot[candidate] = row
    paired = []
    for key, candidates in lookup.items():
        if set(candidates) != {SCALAR_ID, PAVA_ID}:
            continue
        scalar = candidates[SCALAR_ID]
        pava = candidates[PAVA_ID]
        family = str(pava.get("benchmark_family", ""))
        if family not in TRACK_BY_FAMILY:
            raise ValueError(f"unknown benchmark family {family!r}")
        margin = _number(pava.get("policy_value_safety_margin"))
        value_delta = _difference(
            pava.get("policy_value_absolute_error"),
            scalar.get("policy_value_absolute_error"),
        )
        record = {name: value for name, value in zip(PAIR_KEYS, key, strict=True)}
        record.update(
            {
                "benchmark_family": family,
                "track": TRACK_BY_FAMILY[family],
                "cluster_id": f"{family}|{pava.get('cell_id')}|{pava.get('seed')}",
                "is_rank_permuted_negative_control": (
                    pava.get("score_distortion") == "rank_permuted_oracle"
                ),
                "is_learned_estimator": pava.get("score_distortion") is None,
                "calibration_scalar": _number(
                    scalar.get("cross_moment_signed_squared_error")
                ),
                "calibration_pava": _number(
                    pava.get("cross_moment_signed_squared_error")
                ),
                "calibration_delta": _difference(
                    pava.get("cross_moment_signed_squared_error"),
                    scalar.get("cross_moment_signed_squared_error"),
                ),
                "value_abs_error_scalar": _number(
                    scalar.get("policy_value_absolute_error")
                ),
                "value_abs_error_pava": _number(
                    pava.get("policy_value_absolute_error")
                ),
                "value_delta": value_delta,
                "value_margin": margin,
                "value_delta_minus_margin": (
                    value_delta - margin
                    if np.isfinite(value_delta) and np.isfinite(margin)
                    else float("nan")
                ),
                "ratio_mse_scalar": _number(scalar.get("ratio_mse_untruncated")),
                "ratio_mse_pava": _number(pava.get("ratio_mse_untruncated")),
                "ratio_mse_delta": _difference(
                    pava.get("ratio_mse_untruncated"),
                    scalar.get("ratio_mse_untruncated"),
                ),
                "ratio_kl_scalar": _number(
                    scalar.get("ratio_generalized_kl_extended")
                ),
                "ratio_kl_pava": _number(
                    pava.get("ratio_generalized_kl_extended")
                ),
                "ratio_kl_delta": _difference(
                    pava.get("ratio_generalized_kl_extended"),
                    scalar.get("ratio_generalized_kl_extended"),
                ),
                "pava_converged": pava.get("pava_converged"),
            }
        )
        paired.append(record)
    return paired, duplicates


def cluster_bootstrap_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Return a paired cluster bootstrap CI over cell/seed cluster means."""

    clusters: dict[str, list[float]] = {}
    for row in rows:
        value = _number(row.get(value_key))
        if np.isfinite(value):
            clusters.setdefault(str(row["cluster_id"]), []).append(value)
    means = np.asarray(
        [float(np.mean(values)) for _, values in sorted(clusters.items())],
        dtype=np.float64,
    )
    if means.size < 2:
        return {
            "status": "insufficient_clusters",
            "finite_pairs": int(sum(len(values) for values in clusters.values())),
            "clusters": int(means.size),
            "mean": float(np.mean(means)) if means.size else None,
            "median": float(np.median(means)) if means.size else None,
            "win_rate": float(np.mean(means < 0.0)) if means.size else None,
            "ci95_low": None,
            "ci95_high": None,
        }
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, means.size, size=(int(repetitions), means.size))
    bootstrap = np.mean(means[draws], axis=1)
    return {
        "status": "ok",
        "finite_pairs": int(sum(len(values) for values in clusters.values())),
        "clusters": int(means.size),
        "mean": float(np.mean(means)),
        "median": float(np.median(means)),
        "win_rate": float(np.mean(means < 0.0)),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
    }


def _interval(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    repetitions: int,
    bootstrap_seed: int,
    label: str,
) -> dict[str, Any]:
    return cluster_bootstrap_interval(
        rows,
        value_key=value_key,
        repetitions=repetitions,
        seed=_mixed_seed(bootstrap_seed, label),
    )


def _mechanism_summaries(
    pairs: Sequence[Mapping[str, Any]],
    *,
    repetitions: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    distortions = sorted(
        {
            str(row["score_distortion"])
            for row in pairs
            if not row["is_learned_estimator"]
        }
    )
    output = []
    for distortion in distortions:
        selected = [
            row for row in pairs if str(row["score_distortion"]) == distortion
        ]
        summary: dict[str, Any] = {
            "score_distortion": distortion,
            "is_rank_permuted_negative_control": distortion
            == "rank_permuted_oracle",
            "finite_pairs": len(selected),
        }
        for endpoint in (
            "calibration_delta",
            "value_delta",
            "ratio_mse_delta",
            "ratio_kl_delta",
        ):
            summary[endpoint] = _interval(
                selected,
                endpoint,
                repetitions,
                bootstrap_seed,
                f"mechanism-{distortion}-{endpoint}",
            )
        output.append(summary)
    return output


def _planned_track_pairs(
    manifests: Sequence[Mapping[str, Any]],
    track: str,
    *,
    learned_only: bool,
) -> int:
    count = 0
    for manifest in manifests:
        for unit in manifest["aggregation_units"]:
            family = str(unit["cell"]["benchmark_family"])
            operation = unit.get("operation", {})
            if (
                TRACK_BY_FAMILY.get(family) == track
                and (
                    not learned_only
                    or operation.get("base_fit_required") is True
                )
            ):
                count += 1
    return count


def _flatten_track_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        flat = {
            "track": row["track"],
            "planned_pairs": row["planned_pairs"],
            "finite_pairs": row["finite_pairs"],
        }
        for endpoint in (
            "calibration_scalar",
            "calibration_pava",
            "calibration",
            "policy_value_scalar",
            "policy_value_pava",
            "policy_value",
            "value_safety",
            "ratio_mse_scalar",
            "ratio_mse_pava",
            "ratio_mse",
            "ratio_kl_scalar",
            "ratio_kl_pava",
            "ratio_kl",
        ):
            for key, value in row[endpoint].items():
                flat[f"{endpoint}_{key}"] = value
        output.append(flat)
    return output


def _flatten_mechanism_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        flat = {
            "score_distortion": row["score_distortion"],
            "is_rank_permuted_negative_control": row[
                "is_rank_permuted_negative_control"
            ],
            "finite_pairs": row["finite_pairs"],
        }
        for endpoint in (
            "calibration_delta",
            "value_delta",
            "ratio_mse_delta",
            "ratio_kl_delta",
        ):
            for key, value in row[endpoint].items():
                flat[f"{endpoint}_{key}"] = value
        output.append(flat)
    return output


def _write_markdown(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Occupancy calibration experiment readiness",
        "",
        f"- Status: `{result['paper_readiness_status']}`",
        f"- Planned / finite pairs: {result['planned_pairs']} / {result['finite_pairs']}",
        f"- Infrastructure pass: {result['infrastructure']['pass']}",
        f"- Scientific gates enabled: {result['scientific']['enabled']}",
        f"- Calibration benefit gate: {result['scientific']['calibration_benefit_pass']}",
        f"- Value safety gate: {result['scientific']['value_safety_pass']}",
        f"- Primary scope: `{result['scientific']['primary_scope']}`",
        "",
        "| Track | Planned | Finite | Calibration scalar | Calibration PAVA | PAVA - scalar [95% CI] | Value scalar | Value PAVA | PAVA - scalar [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["track_summaries"]:
        lines.append(
            f"| {row['track']} | {row['planned_pairs']} | {row['finite_pairs']} | "
            f"{_mean_text(row['calibration_scalar'])} | {_mean_text(row['calibration_pava'])} | "
            f"{_ci_text(row['calibration'])} | {_mean_text(row['policy_value_scalar'])} | "
            f"{_mean_text(row['policy_value_pava'])} | {_ci_text(row['policy_value'])} |"
        )
    lines.extend(
        [
            "",
            "### Controlled occupancy-ratio accuracy",
            "",
            "| Track | MSE scalar | MSE PAVA | MSE delta [95% CI] | KL scalar | KL PAVA | KL delta [95% CI] |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["track_summaries"]:
        if row["track"] == "controlled":
            lines.append(
                f"| controlled | {_mean_text(row['ratio_mse_scalar'])} | "
                f"{_mean_text(row['ratio_mse_pava'])} | {_ci_text(row['ratio_mse'])} | "
                f"{_mean_text(row['ratio_kl_scalar'])} | {_mean_text(row['ratio_kl_pava'])} | "
                f"{_ci_text(row['ratio_kl'])} |"
            )
    lines.extend(
        [
            "",
            "### Predeclared exact-score mechanisms",
            "",
            "| Score distortion | Negative control | Finite | Calibration delta | Value delta | MSE delta | KL delta |",
            "|---|:---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["mechanism_summaries"]:
        lines.append(
            f"| {row['score_distortion']} | {row['is_rank_permuted_negative_control']} | "
            f"{row['finite_pairs']} | {_ci_text(row['calibration_delta'])} | "
            f"{_ci_text(row['value_delta'])} | {_ci_text(row['ratio_mse_delta'])} | "
            f"{_ci_text(row['ratio_kl_delta'])} |"
        )
    lines.extend(
        [
            "",
            "Signed cross-moment estimates are retained. Primary claim gates use learned estimators only. Exact-score mechanisms and the rank-permuted negative control are reported separately. Missing or failed pairs remain in the planned denominator.",
        ]
    )
    _atomic_text(path, "\n".join(lines) + "\n")


def _write_latex(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Track & Planned & Finite & Scalar calibration & PAVA calibration & $\\Delta$ (95\\% CI) \\\\",
        "\\midrule",
    ]
    for row in rows:
        track_label = str(row["track"]).replace("_", "\\_")
        lines.append(
            f"{track_label} & {row['planned_pairs']} & {row['finite_pairs']} & "
            f"{_mean_latex(row['calibration_scalar'])} & "
            f"{_mean_latex(row['calibration_pava'])} & "
            f"{_ci_latex(row['calibration'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _atomic_text(path, "\n".join(lines) + "\n")


def _write_controlled_latex(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    controlled = next(row for row in rows if row["track"] == "controlled")
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Endpoint & Scalar & PAVA & $\\Delta$ (95\\% CI) \\\\",
        "\\midrule",
        f"Policy value absolute error & {_mean_latex(controlled['policy_value_scalar'])} & {_mean_latex(controlled['policy_value_pava'])} & {_ci_latex(controlled['policy_value'])} \\\\",
        f"Untruncated ratio MSE & {_mean_latex(controlled['ratio_mse_scalar'])} & {_mean_latex(controlled['ratio_mse_pava'])} & {_ci_latex(controlled['ratio_mse'])} \\\\",
        f"Extended generalized KL & {_mean_latex(controlled['ratio_kl_scalar'])} & {_mean_latex(controlled['ratio_kl_pava'])} & {_ci_latex(controlled['ratio_kl'])} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    _atomic_text(path, "\n".join(lines) + "\n")


def _write_mechanism_latex(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Score distortion & Calibration $\\Delta$ & Value $\\Delta$ & MSE $\\Delta$ & KL $\\Delta$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        label = str(row["score_distortion"]).replace("_", "\\_")
        lines.append(
            f"{label} & {_ci_latex(row['calibration_delta'])} & "
            f"{_ci_latex(row['value_delta'])} & {_ci_latex(row['ratio_mse_delta'])} & "
            f"{_ci_latex(row['ratio_kl_delta'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _atomic_text(path, "\n".join(lines) + "\n")


def _write_figures(output: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as error:
        return f"skipped: {type(error).__name__}: {error}"
    generated = []
    for endpoint, filename, label in (
        ("calibration", "calibration_delta_by_track", "PAVA - scalar signed cross-moment error"),
        ("policy_value", "policy_value_delta_by_track", "PAVA - scalar absolute policy-value error"),
        ("value_safety", "value_safety_delta_by_track", "PAVA - scalar absolute policy-value error - margin"),
        ("ratio_mse", "ratio_mse_delta_by_track", "PAVA - scalar untruncated ratio MSE"),
        ("ratio_kl", "ratio_kl_delta_by_track", "PAVA - scalar generalized KL"),
    ):
        usable = [row for row in rows if row[endpoint]["status"] == "ok"]
        if not usable:
            continue
        means = np.asarray([row[endpoint]["mean"] for row in usable], dtype=float)
        lower = np.asarray([row[endpoint]["ci95_low"] for row in usable], dtype=float)
        upper = np.asarray([row[endpoint]["ci95_high"] for row in usable], dtype=float)
        y = np.arange(len(usable))
        fig, ax = plt.subplots(figsize=(5.4, 1.4 + 0.55 * len(usable)))
        ax.errorbar(means, y, xerr=np.vstack([means - lower, upper - means]), fmt="o", color="#1f4e79", capsize=3)
        ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(y, [row["track"] for row in usable])
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        for suffix in ("pdf", "svg"):
            final_path = output / f"{filename}.{suffix}"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{filename}.", suffix=f".{suffix}.tmp", dir=output
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                fig.savefig(temporary, format=suffix, bbox_inches="tight")
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, final_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            generated.append(final_path.name)
        plt.close(fig)
    return "generated:" + ",".join(generated) if generated else "no_eligible_intervals"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        _atomic_text(path, "")
        return
    columns = sorted({key for row in rows for key in row})
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _number(value: object) -> float:
    if value is None or value == "":
        return float("nan")
    if value in {"+inf", "inf"}:
        return float("inf")
    if value == "-inf":
        return float("-inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _difference(left: object, right: object) -> float:
    a, b = _number(left), _number(right)
    if np.isnan(a) or np.isnan(b):
        return float("nan")
    if np.isinf(a) and np.isinf(b) and np.sign(a) == np.sign(b):
        return float("nan")
    return float(a - b)


def _mixed_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _ci_text(row: Mapping[str, Any]) -> str:
    if row["status"] != "ok":
        return str(row["status"])
    return f"{row['mean']:.4g} [{row['ci95_low']:.4g}, {row['ci95_high']:.4g}]"


def _mean_text(row: Mapping[str, Any]) -> str:
    value = row.get("mean")
    return "--" if value is None else f"{float(value):.4g}"


def _ci_latex(row: Mapping[str, Any]) -> str:
    if row["status"] != "ok":
        return "--"
    return f"{row['mean']:.3g} [{row['ci95_low']:.3g}, {row['ci95_high']:.3g}]"


def _mean_latex(row: Mapping[str, Any]) -> str:
    value = row.get("mean")
    return "--" if value is None else f"{float(value):.3g}"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_720)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = build_report(
        run_roots=args.run_root,
        output_dir=args.output,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["paper_readiness_status"] == "paper_ready_claim_gates_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_report",
    "cluster_bootstrap_interval",
    "pair_candidate_rows",
]
