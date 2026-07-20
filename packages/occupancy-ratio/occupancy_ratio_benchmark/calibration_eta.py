"""Pilot-based wall-time gate for the three-day calibration suite."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from occupancy_ratio_benchmark.calibration_protocol import dataset_id


TRACK_BY_FAMILY = {
    "random_tabular": "controlled",
    "linear_gaussian": "controlled",
    "d4rl_matched": "d4rl",
    "dice_rl_cartpole": "cartpole",
}


def build_eta_gate(
    *,
    pilot_fold_rows: Sequence[Mapping[str, Any]],
    confirmatory_manifests: Sequence[Mapping[str, Any]],
    pilot_dataset_rows: Sequence[Mapping[str, Any]] | None = None,
    maximum_concurrency: int = 2,
    ceiling_hours: float = 84.0,
    contingency_fraction: float = 0.10,
    expected_full_fold_units: int = 24_000,
) -> dict[str, Any]:
    """Project the full suite from p95 pilot costs by track/estimator/size.

    Pilot truth or endpoint values are never consumed.  Failed or missing
    runtime strata close the gate instead of being silently dropped.
    """

    concurrency = int(maximum_concurrency)
    if concurrency <= 0:
        raise ValueError("maximum_concurrency must be positive")
    ceiling = float(ceiling_hours)
    contingency = float(contingency_fraction)
    if not np.isfinite(ceiling) or ceiling <= 0.0:
        raise ValueError("ceiling_hours must be positive and finite")
    if not np.isfinite(contingency) or contingency < 0.0:
        raise ValueError("contingency_fraction must be nonnegative and finite")

    full_rows = _full_fold_rows(confirmatory_manifests)
    full_total = len(full_rows)
    size_limits = _size_limits(full_rows)
    full_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in full_rows:
        track = _track(str(row["benchmark_family"]))
        band = _size_band(int(row["sample_size"]), size_limits[track])
        full_counts[(track, str(row["estimator_id"]), band)] += 1

    pilot_runtime: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    failed_pilot_rows = 0
    invalid_pilot_rows = 0
    for row in pilot_fold_rows:
        if str(row.get("status", "")) != "ok":
            failed_pilot_rows += 1
            continue
        try:
            track = _track(str(row["benchmark_family"]))
            estimator = str(row["estimator_id"])
            size = int(row["sample_size"])
            runtime = float(row["fit_runtime_sec"])
        except (KeyError, TypeError, ValueError):
            invalid_pilot_rows += 1
            continue
        if track not in size_limits or not np.isfinite(runtime) or runtime <= 0.0:
            invalid_pilot_rows += 1
            continue
        pilot_runtime[(track, estimator, _size_band(size, size_limits[track]))].append(runtime)

    strata: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    serial_seconds = 0.0
    for key in sorted(full_counts):
        track, estimator, band = key
        count = int(full_counts[key])
        runtimes = pilot_runtime.get(key, [])
        if not runtimes:
            missing.append(
                {
                    "unit_kind": "fold_fit",
                    "track": track,
                    "estimator_id": estimator,
                    "size_band": band,
                    "full_fold_units": count,
                }
            )
            continue
        p50 = float(np.quantile(runtimes, 0.50))
        p95 = float(np.quantile(runtimes, 0.95))
        projected = count * p95
        serial_seconds += projected
        strata.append(
            {
                "unit_kind": "fold_fit",
                "track": track,
                "estimator_id": estimator,
                "size_band": band,
                "pilot_ok_units": len(runtimes),
                "pilot_p50_sec": p50,
                "pilot_p95_sec": p95,
                "full_fold_units": count,
                "projected_serial_hours": projected / 3600.0,
            }
        )
    dataset_strata: list[dict[str, Any]] = []
    dataset_seconds = 0.0
    dataset_failed_rows = 0
    dataset_invalid_rows = 0
    if pilot_dataset_rows is not None:
        full_dataset_rows = _full_dataset_rows(confirmatory_manifests)
        full_dataset_counts: dict[tuple[str, str], int] = defaultdict(int)
        for row in full_dataset_rows:
            track = _track(str(row["benchmark_family"]))
            band = _size_band(int(row["sample_size"]), size_limits[track])
            full_dataset_counts[(track, band)] += 1
        pilot_dataset_runtime: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in pilot_dataset_rows:
            if str(row.get("status", "")) not in {"created", "resumed", "ok"}:
                dataset_failed_rows += 1
                continue
            try:
                track = _track(str(row["benchmark_family"]))
                size = int(row["sample_size"])
                runtime = float(row["dataset_runtime_sec"])
            except (KeyError, TypeError, ValueError):
                dataset_invalid_rows += 1
                continue
            if track not in size_limits or not np.isfinite(runtime) or runtime <= 0.0:
                dataset_invalid_rows += 1
                continue
            pilot_dataset_runtime[(track, _size_band(size, size_limits[track]))].append(
                runtime
            )
        for key in sorted(full_dataset_counts):
            track, band = key
            count = int(full_dataset_counts[key])
            runtimes = pilot_dataset_runtime.get(key, [])
            if not runtimes:
                missing.append(
                    {
                        "unit_kind": "dataset_build",
                        "track": track,
                        "size_band": band,
                        "full_dataset_units": count,
                    }
                )
                continue
            p50 = float(np.quantile(runtimes, 0.50))
            p95 = float(np.quantile(runtimes, 0.95))
            projected = count * p95
            dataset_seconds += projected
            dataset_strata.append(
                {
                    "unit_kind": "dataset_build",
                    "track": track,
                    "size_band": band,
                    "pilot_ok_units": len(runtimes),
                    "pilot_p50_sec": p50,
                    "pilot_p95_sec": p95,
                    "full_dataset_units": count,
                    "projected_serial_hours": projected / 3600.0,
                }
            )
    projected_fit_hours = serial_seconds / concurrency / 3600.0
    projected_dataset_hours = dataset_seconds / 3600.0
    projected_hours = projected_fit_hours + projected_dataset_hours
    guarded_hours = projected_hours * (1.0 + contingency)
    reasons = []
    if full_total != int(expected_full_fold_units):
        reasons.append(
            f"full learned-fold count is {full_total}, expected {int(expected_full_fold_units)}"
        )
    if missing:
        reasons.append(f"{len(missing)} required pilot runtime strata are missing")
    if failed_pilot_rows:
        reasons.append(f"{failed_pilot_rows} pilot fold units failed")
    if invalid_pilot_rows:
        reasons.append(f"{invalid_pilot_rows} pilot runtime rows are invalid")
    if dataset_failed_rows:
        reasons.append(f"{dataset_failed_rows} pilot dataset builds failed")
    if dataset_invalid_rows:
        reasons.append(f"{dataset_invalid_rows} pilot dataset runtimes are invalid")
    if guarded_hours > ceiling:
        reasons.append(
            f"guarded p95 projection {guarded_hours:.2f}h exceeds {ceiling:.2f}h"
        )
    return {
        "schema_version": 1,
        "status": "pass" if not reasons else "blocked",
        "scientific_selection_performed": False,
        "projection_quantile": 0.95,
        "maximum_concurrency": concurrency,
        "contingency_fraction": contingency,
        "ceiling_hours": ceiling,
        "full_learned_fold_units": full_total,
        "expected_full_learned_fold_units": int(expected_full_fold_units),
        "pilot_ok_runtime_rows": int(sum(len(values) for values in pilot_runtime.values())),
        "pilot_failed_rows": failed_pilot_rows,
        "pilot_invalid_rows": invalid_pilot_rows,
        "pilot_dataset_failed_rows": dataset_failed_rows,
        "pilot_dataset_invalid_rows": dataset_invalid_rows,
        "projected_p95_fold_walltime_hours": projected_fit_hours,
        "projected_p95_dataset_walltime_hours": projected_dataset_hours,
        "projected_p95_walltime_hours": projected_hours,
        "guarded_projected_walltime_hours": guarded_hours,
        "missing_strata": missing,
        "strata": strata,
        "dataset_strata": dataset_strata,
        "blocking_reasons": reasons,
    }


def _full_fold_rows(manifests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for manifest in manifests:
        units = manifest.get("atomic_fold_units")
        if not isinstance(units, list):
            raise ValueError("confirmatory manifest lacks atomic_fold_units")
        for unit in units:
            identity = unit.get("identity", {})
            axes = identity.get("axis_values", {})
            cell = unit.get("cell", {})
            try:
                rows.append(
                    {
                        "benchmark_family": str(cell["benchmark_family"]),
                        "estimator_id": str(identity["estimator_id"]),
                        "sample_size": int(axes["sample_size"]),
                    }
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid confirmatory fold unit") from error
    return rows


def _full_dataset_rows(
    manifests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for manifest in manifests:
        units = [
            *manifest.get("atomic_fold_units", []),
            *manifest.get("deterministic_score_units", []),
        ]
        for unit in units:
            data_id = dataset_id(unit)
            if data_id in seen:
                continue
            seen.add(data_id)
            identity = unit.get("identity", {})
            axes = identity.get("axis_values", {})
            cell = unit.get("cell", {})
            try:
                rows.append(
                    {
                        "benchmark_family": str(cell["benchmark_family"]),
                        "sample_size": int(axes["sample_size"]),
                    }
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid confirmatory dataset unit") from error
    return rows


def _size_limits(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, int]]:
    sizes: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        sizes[_track(str(row["benchmark_family"]))].append(int(row["sample_size"]))
    return {track: (min(values), max(values)) for track, values in sizes.items()}


def _size_band(size: int, limits: tuple[int, int]) -> str:
    low, high = limits
    if low == high:
        return "high"
    midpoint = 0.5 * (low + high)
    return "low" if int(size) <= midpoint else "high"


def _track(family: str) -> str:
    try:
        return TRACK_BY_FAMILY[family]
    except KeyError as error:
        raise ValueError(f"unknown calibration benchmark family {family!r}") from error


__all__ = ["TRACK_BY_FAMILY", "build_eta_gate"]
