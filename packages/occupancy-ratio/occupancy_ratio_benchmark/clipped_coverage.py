"""Shared-hub coverage benchmark for recursively clipped KL-FORI.

The benchmark is deliberately separate from the normalized occupancy-ratio
benchmark schema.  Standard FORE, recursively clipped FORE, and one-shot
winsorization target different objects once coverage fails.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

from occupancy_ratio_benchmark._clipped_coverage_artifacts import (
    SCHEMA_VERSION,
    atomic_write_text,
    benchmark_cell_id,
    canonical_json,
    configuration_hash,
    environment_metadata,
    read_cell,
    repository_metadata,
    write_cell,
)
from occupancy_ratio_benchmark._clipped_coverage_data import (
    DEFAULT_MASS_GRID,
    CoverageRunConfig,
    SharedHubDataset,
    SharedHubTruth,
    analytic_shared_hub_rows,
    make_shared_hub_dataset,
    solve_context_coverage_scale,
)
from occupancy_ratio_benchmark._clipped_coverage_execution import (
    combine_backend_folds as _combine_backend_folds,
    run_backend_cell as _run_backend_cell,
    run_backend_fold as _run_backend_fold,
)
from occupancy_ratio_benchmark._clipped_coverage_metrics import summarize_rows
from occupancy_ratio_benchmark._clipped_coverage_freeze import (
    apply_freeze,
    load_freeze_manifest,
)
from occupancy_ratio_benchmark._clipped_coverage_pilots import (
    linear_verification_config,
    neural_finalist_configs,
    neural_screen_configs,
    promoted_neural_ids,
    select_standard_pilot,
    standard_candidate_id,
    standard_extension_config,
    standard_pilot_configs,
)


Array = np.ndarray


def run_coverage_benchmark(
    config: CoverageRunConfig,
    *,
    cell_dir: str | Path | None = None,
    resume: bool = True,
    fail_fast: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
    progress_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run auditable cross-fitted FORE comparisons.

    When ``cell_dir`` is supplied, each logical cell is written atomically and
    compatible completed cells are reused.  Sharding is a deterministic hash
    partition and therefore does not change seeds or fitted results.
    """
    if num_shards <= 0 or not (0 <= shard_index < num_shards):
        raise ValueError("shard_index must lie in [0, num_shards)")
    rows: list[dict[str, Any]] = []
    config_digest = configuration_hash(config)
    cells_root = None if cell_dir is None else Path(cell_dir)
    progress_file = None if progress_path is None else Path(progress_path)
    assigned_cells = _assigned_result_count(
        config, shard_index=shard_index, num_shards=num_shards
    )
    completed_cells = 0
    started = time.time()
    for repetition in range(int(config.repetitions)):
        for mass_index, requested_mass in enumerate(config.mass_grid):
            seed = int(
                config.seed
                + 100_003 * repetition
                + 997 * mass_index
                + 37 * config.contexts
            )
            dataset = make_shared_hub_dataset(
                n=config.n,
                gamma=config.gamma,
                tau_upper=config.tau_upper,
                oracle_mass=float(requested_mass),
                contexts=config.contexts,
                seed=seed,
                sampling_mode=config.sampling_mode,
                irrelevant_features=config.irrelevant_features,
                termination_encoding=config.termination_encoding,
            )
            for backend in config.backends:
                fold_cell_ids = [
                    benchmark_cell_id(
                        config=config,
                        repetition=repetition,
                        requested_mass=float(requested_mass),
                        backend=str(backend),
                        seed=seed,
                        fold=fold,
                    )
                    for fold in (0, 1)
                ]
                result_id = hashlib.sha256(
                    canonical_json({"fold_cell_ids": fold_cell_ids}).encode()
                ).hexdigest()[:24]
                if int(result_id, 16) % int(num_shards) != int(shard_index):
                    continue
                if cells_root is None:
                    cell_rows = _run_backend_cell(
                        dataset,
                        config,
                        backend=str(backend),
                        repetition=repetition,
                        seed=seed,
                    )
                else:
                    fold_payloads: list[dict[str, Any]] = []
                    for fold, fold_cell_id in enumerate(fold_cell_ids):
                        cell_path = cells_root / f"{fold_cell_id}.json"
                        if resume and cell_path.exists():
                            stored = read_cell(
                                cell_path,
                                cell_id=fold_cell_id,
                                config_hash=config_digest,
                            )
                            if len(stored) != 1:
                                raise ValueError(
                                    f"fold cell {fold_cell_id} must contain one payload"
                                )
                            payload = stored[0]
                        else:
                            payload = _run_backend_fold(
                                dataset,
                                config,
                                backend=str(backend),
                                repetition=repetition,
                                seed=seed,
                                heldout=fold,
                            )
                            payload.update(
                                {
                                    "schema": SCHEMA_VERSION,
                                    "fold_cell_id": fold_cell_id,
                                    "config_hash": config_digest,
                                    "requested_mass": float(requested_mass),
                                }
                            )
                            write_cell(
                                cell_path,
                                cell_id=fold_cell_id,
                                config_hash=config_digest,
                                rows=[payload],
                            )
                        if (
                            payload.get("fold_cell_id") != fold_cell_id
                            or int(payload.get("fold", -1)) != fold
                        ):
                            raise ValueError(
                                f"fold payload identity mismatch in {cell_path}"
                            )
                        fold_payloads.append(payload)
                    cell_rows = _combine_backend_folds(
                        fold_payloads,
                        dataset,
                        config,
                        backend=str(backend),
                        repetition=repetition,
                        seed=seed,
                    )
                for row in cell_rows:
                    row.update(
                        {
                            "schema": SCHEMA_VERSION,
                            "cell_id": result_id,
                            "fold_cell_ids": json.dumps(fold_cell_ids),
                            "config_hash": config_digest,
                            "requested_mass": float(requested_mass),
                        }
                    )
                rows.extend(cell_rows)
                completed_cells += 1
                if progress_file is not None:
                    elapsed = time.time() - started
                    rate = completed_cells / elapsed if elapsed > 0.0 else 0.0
                    eta = (
                        (assigned_cells - completed_cells) / rate
                        if rate > 0.0
                        else None
                    )
                    atomic_write_text(
                        progress_file,
                        json.dumps(
                            {
                                "schema": SCHEMA_VERSION,
                                "config_hash": config_digest,
                                "shard_index": shard_index,
                                "num_shards": num_shards,
                                "assigned_result_cells": assigned_cells,
                                "completed_result_cells": completed_cells,
                                "progress_fraction": (
                                    completed_cells / assigned_cells
                                    if assigned_cells
                                    else 1.0
                                ),
                                "elapsed_sec": elapsed,
                                "eta_sec": eta,
                                "last_cell_id": result_id,
                                "last_statuses": {
                                    str(row["method"]): str(row["status"])
                                    for row in cell_rows
                                },
                                "updated_unix": time.time(),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                    )
                if fail_fast and any(
                    row.get("status") in {"error", "nonconverged", "optimizer_unstable"}
                    for row in cell_rows
                ):
                    raise RuntimeError(
                        f"benchmark result {result_id} failed a fail-fast guardrail"
                    )
    return rows


def _assigned_result_count(
    config: CoverageRunConfig, *, shard_index: int, num_shards: int
) -> int:
    count = 0
    for repetition in range(int(config.repetitions)):
        for mass_index, requested_mass in enumerate(config.mass_grid):
            seed = int(
                config.seed
                + 100_003 * repetition
                + 997 * mass_index
                + 37 * config.contexts
            )
            for backend in config.backends:
                fold_ids = [
                    benchmark_cell_id(
                        config=config,
                        repetition=repetition,
                        requested_mass=float(requested_mass),
                        backend=str(backend),
                        seed=seed,
                        fold=fold,
                    )
                    for fold in (0, 1)
                ]
                result_id = hashlib.sha256(
                    canonical_json({"fold_cell_ids": fold_ids}).encode()
                ).hexdigest()[:24]
                count += int(
                    int(result_id, 16) % int(num_shards) == int(shard_index)
                )
    return count


def write_benchmark_artifacts(
    output_dir: str | Path,
    *,
    rows: Sequence[dict[str, Any]],
    config: CoverageRunConfig | Sequence[CoverageRunConfig],
    make_plots: bool = True,
    allow_dirty: bool = False,
    require_complete: bool = True,
    selected_candidate_id: str | None = None,
) -> dict[str, Path]:
    """Write raw rows, summaries, manifest, and optional figures."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results_path = root / "results.csv"
    summary_path = root / "summary.csv"
    selected_summary_path = root / "selected_summary.csv"
    figure_inputs_path = root / "figure_inputs.csv"
    table_path = root / "paper_table.csv"
    manifest_path = root / "manifest.json"
    summary = summarize_rows(rows)
    _validate_result_rows(rows)
    configs = [config] if isinstance(config, CoverageRunConfig) else list(config)
    if require_complete:
        validate_benchmark_completeness(rows, configs)
    _write_csv(results_path, rows)
    _write_csv(summary_path, summary)
    _write_csv(figure_inputs_path, _paper_figure_inputs(summary))
    _write_csv(table_path, _paper_table_rows(rows))
    if selected_candidate_id is not None:
        selected_rows = [
            row
            for row in rows
            if str(row.get("candidate_id", "")) == str(selected_candidate_id)
        ]
        if not selected_rows:
            raise ValueError("selected candidate has no benchmark rows")
        _write_csv(selected_summary_path, summarize_rows(selected_rows))
    config_payload = [asdict(item) for item in configs]
    repo = repository_metadata(Path(__file__).resolve())
    include_torch = any("neural" in item.backends for item in configs)
    deterministic_rows = sorted(
        (
            {
                key: value
                for key, value in row.items()
                if key not in {"runtime_sec", "created_unix"}
            }
            for row in rows
        ),
        key=lambda row: (str(row.get("cell_id", "")), str(row.get("method", ""))),
    )
    manifest = {
        "schema": SCHEMA_VERSION,
        "configs": config_payload,
        "config_hashes": [configuration_hash(item) for item in configs],
        "repository": repo,
        "environment": environment_metadata(include_torch=include_torch),
        "command": list(sys.argv),
        "seeds": sorted({int(row["seed"]) for row in rows if "seed" in row}),
        "cell_ids": sorted(_row_fold_cell_ids(rows)),
        "result_ids": sorted({str(row["cell_id"]) for row in rows if "cell_id" in row}),
        "fold_seeds": sorted(_row_fold_seeds(rows)),
        "deterministic_results_hash": hashlib.sha256(
            canonical_json(deterministic_rows).encode()
        ).hexdigest(),
        "allow_dirty": bool(allow_dirty),
        "oracle_used_for_selection": False,
        "created_unix": time.time(),
    }
    atomic_write_text(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    artifacts = {
        "results": results_path,
        "summary": summary_path,
        "manifest": manifest_path,
        "figure_inputs": figure_inputs_path,
        "paper_table": table_path,
    }
    if selected_candidate_id is not None:
        artifacts["selected_summary"] = selected_summary_path
    if make_plots:
        artifacts.update(plot_coverage_summary(summary, output_dir=root))
    return artifacts


def validate_benchmark_completeness(
    rows: Sequence[dict[str, Any]],
    configs: Sequence[CoverageRunConfig],
) -> None:
    """Require every configured fold and method exactly once before aggregation."""
    expected_cells: set[str] = set()
    expected_configs = {configuration_hash(config) for config in configs}
    methods_by_config = {
        configuration_hash(config): set(str(method) for method in config.methods)
        for config in configs
    }
    for config in configs:
        for repetition in range(int(config.repetitions)):
            for mass_index, requested_mass in enumerate(config.mass_grid):
                seed = int(
                    config.seed
                    + 100_003 * repetition
                    + 997 * mass_index
                    + 37 * config.contexts
                )
                for backend in config.backends:
                    expected_cells.update(
                        benchmark_cell_id(
                            config=config,
                            repetition=repetition,
                            requested_mass=float(requested_mass),
                            backend=str(backend),
                            seed=seed,
                            fold=fold,
                        )
                        for fold in (0, 1)
                    )
    actual_cells = _row_fold_cell_ids(rows)
    missing = sorted(expected_cells - actual_cells)
    unexpected = sorted(actual_cells - expected_cells)
    if missing or unexpected:
        raise ValueError(
            "incomplete benchmark fold cells: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    methods_by_result: dict[str, set[str]] = {}
    for row in rows:
        if row.get("schema") != SCHEMA_VERSION:
            raise ValueError("benchmark rows have an incompatible schema")
        if row.get("config_hash") not in expected_configs:
            raise ValueError("benchmark rows have an incompatible configuration hash")
        methods_by_result.setdefault(str(row["cell_id"]), set()).add(
            str(row.get("method"))
        )
    for result_id, methods in methods_by_result.items():
        result_rows = [row for row in rows if str(row.get("cell_id")) == result_id]
        config_hash = str(result_rows[0].get("config_hash"))
        if methods != methods_by_config[config_hash]:
            raise ValueError(
                f"benchmark result {result_id} has methods {sorted(methods)}, "
                f"expected {sorted(methods_by_config[config_hash])}"
            )


def merge_benchmark_shards(
    shards: Sequence[Sequence[dict[str, Any]]],
    *,
    configs: Sequence[CoverageRunConfig],
) -> list[dict[str, Any]]:
    """Merge compatible shard rows only after duplicate and completeness checks."""
    rows = [row for shard in shards for row in shard]
    _validate_result_rows(rows)
    validate_benchmark_completeness(rows, configs)
    return rows


def plot_coverage_summary(
    summary: Sequence[dict[str, Any]], *, output_dir: str | Path
) -> dict[str, Path]:
    """Plot the three predeclared paper outcomes for the shared-hub design."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        raise ImportError("coverage plotting requires matplotlib.") from exc
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), constrained_layout=True)
    main = [row for row in summary if int(row.get("contexts", 0)) == 1]
    sample_sizes = sorted({int(row["n"]) for row in main})
    for n in sample_sizes:
        subset = sorted(
            (
                row
                for row in main
                if row.get("method") == "clipped_fori" and int(row["n"]) == n
            ),
            key=lambda row: float(row["oracle_mass"]),
        )
        if subset:
            axes[0].plot(
                [row["oracle_mass"] for row in subset],
                [row["crossfit_mass_mean"] for row in subset],
                marker="o",
                label=f"n={n:,}",
            )
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Recursive oracle")
    if sample_sizes:
        largest_n = max(sample_sizes)
        for method, label, marker in (
            ("clipped_fori", "Clipped FORE", "o"),
            ("posthoc_winsorized", "Post-hoc cap", "s"),
        ):
            subset = sorted(
                (
                    row
                    for row in main
                    if row.get("method") == method and int(row["n"]) == largest_n
                ),
                key=lambda row: float(row["oracle_mass"]),
            )
            if subset:
                axes[1].plot(
                    [row["oracle_mass"] for row in subset],
                    [row.get("hub_value_mean", np.nan) for row in subset],
                    marker=marker,
                    label=label,
                )
        gamma = float(main[0]["gamma"])
        grid = np.linspace(0.0, 1.0, 101)
        axes[1].plot(grid, gamma * grid, "k--", linewidth=1, label="Recursive oracle")
        axes[1].plot(grid, np.full_like(grid, gamma), "k:", linewidth=1,
                     label="Post-hoc oracle")
    for mass in (0.25, 0.5, 0.9):
        subset = sorted(
            (
                row
                for row in main
                if row.get("method") == "clipped_fori"
                and np.isclose(float(row["oracle_mass"]), mass)
            ),
            key=lambda row: int(row["n"]),
        )
        if subset:
            axes[2].plot(
                [row["n"] for row in subset],
                [row["mass_own_error_rmse"] for row in subset],
                marker="o",
                label=rf"$\alpha={mass:g}$",
            )
    axes[0].set_ylabel("Estimated retained mass")
    axes[1].set_ylabel("Estimated hub stopped value")
    axes[2].set_ylabel("Retained-mass RMSE")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].set_xlabel("Oracle retained mass")
    axes[1].set_xlabel("Oracle retained mass")
    axes[2].set_xlabel("Sample size")
    axes[2].set_xscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=8)
    pdf = root / "clipped_coverage_summary.pdf"
    png = root / "clipped_coverage_summary.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=180)
    plt.close(fig)
    return {"figure_pdf": pdf, "figure_png": png}


def _paper_figure_inputs(
    summary: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the immutable summary rows consumed by the manuscript figure."""
    return [dict(row) for row in summary if int(row.get("contexts", 0)) in {1, 64}]


def _paper_table_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate method-specific accuracy and plug-in diagnostics for the paper."""
    keys = ("method", "backend", "n", "contexts", "tau_lower", "tau_upper")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    for values, group in groups.items():
        analyzable = [
            row
            for row in group
            if row.get("status") in {"ok", "nonconverged", "optimizer_unstable"}
        ]
        result = dict(zip(keys, values))
        result.update(
            {
                "rows": len(group),
                "mass_rmse": _row_rmse(analyzable, "mass_own_error"),
                "hub_value_rmse": _row_rmse(analyzable, "hub_own_error"),
                "ratio_l1_mean": _row_mean(analyzable, "ratio_l1_own_target"),
                "gate_error_mean": _row_mean(analyzable, "gate_error"),
                "converged_fraction": _row_mean(analyzable, "converged_fraction"),
                "failure_rate": sum(
                    row.get("status")
                    in {"error", "nonconverged", "optimizer_unstable"}
                    for row in group
                )
                / len(group),
                "out_of_regime_rate": sum(
                    row.get("status") == "out_of_regime" for row in group
                )
                / len(group),
                "runtime_sec_mean": _row_mean(analyzable, "runtime_sec"),
                "hub_plugin_overshoot_rate": _row_mean(
                    analyzable, "hub_plugin_overshoot"
                ),
                "hub_plugin_containment_rate": _row_mean(
                    analyzable, "hub_plugin_interval_contains_full"
                ),
            }
        )
        output.append(result)
    return output


def _row_mean(rows: Sequence[dict[str, Any]], name: str) -> float:
    values = np.asarray([row.get(name, np.nan) for row in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _row_rmse(rows: Sequence[dict[str, Any]], name: str) -> float:
    values = np.asarray([row.get(name, np.nan) for row in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite * finite))) if finite.size else float("nan")


def smoke_config(*, seed: int = 12_000) -> CoverageRunConfig:
    """Return a CI-sized deterministic benchmark configuration."""
    return CoverageRunConfig(
        n=240,
        repetitions=1,
        mass_grid=(0.0, 0.5, 1.0),
        contexts=1,
        backends=("linear",),
        seed=seed,
        clipped_num_iterations=3,
        clipped_gate_steps=8,
        clipped_ratio_steps=12,
        standard_num_iterations=3,
        standard_optimizer_steps=12,
        validation_fraction=0.0,
    )


def pilot_optimizer_configs(
    *, backend: str = "linear", seed: int = 2_000
) -> list[CoverageRunConfig]:
    """Return paired deployable-loss optimizer candidates for a ten-seed pilot."""
    candidates = (
        ((5e-2, 1e-8), (2e-2, 1e-8), (5e-2, 1e-10), (2e-2, 1e-10))
        if backend == "linear"
        else ((1e-3, 1e-8), (5e-4, 1e-8), (1e-3, 1e-10), (5e-4, 1e-10))
    )
    return [
        CoverageRunConfig(
            n=10_000,
            repetitions=10,
            mass_grid=(0.0, 0.25, 0.5, 0.9, 1.0),
            contexts=1,
            backends=(backend,),
            seed=seed,
            clipped_gate_steps=200,
            clipped_ratio_steps=300,
            clipped_gate_learning_rate=5e-2 if backend == "linear" else 1e-3,
            clipped_ratio_learning_rate=ratio_lr,
            clipped_inner_relative_tolerance=inner_tolerance,
            standard_num_iterations=1,
            standard_optimizer_steps=1,
            optimizer_stability_restarts=3,
        )
        for ratio_lr, inner_tolerance in candidates
    ]


def optimizer_candidate_id(config: CoverageRunConfig) -> str:
    """Return a readable identifier covering every tuned optimizer field."""
    backend = str(config.backends[0]) if len(config.backends) == 1 else "mixed"
    gate_lr = _resolved_learning_rate(config.clipped_gate_learning_rate, backend)
    ratio_lr = _resolved_learning_rate(config.clipped_ratio_learning_rate, backend)
    return (
        f"g{config.clipped_gate_steps}_r{config.clipped_ratio_steps}"
        f"_glr{gate_lr:.0e}_rlr{ratio_lr:.0e}"
        f"_irt{config.clipped_inner_relative_tolerance:.0e}"
    )


def select_optimizer_pilot(
    candidate_rows: Sequence[tuple[str, Sequence[dict[str, Any]]]],
    *,
    objective_stability_threshold: float = 1e-4,
    ratio_stability_threshold: float = 1e-3,
    gate_stability_threshold: float = 1e-3,
) -> dict[str, Any]:
    """Select the fastest zero-failure candidate using deployable diagnostics.

    Oracle mass, ratios, and values are intentionally absent from the selection
    rule.  They remain available only in the candidate result files for audit.
    """
    audit: list[dict[str, Any]] = []
    for candidate_id, rows in candidate_rows:
        deployable_rows = [_pilot_deployable_row(row) for row in rows]
        clipped = [
            row for row in deployable_rows if row.get("method") == "clipped_fori"
        ]
        ok = [row for row in clipped if row.get("status") == "ok"]
        stability = np.asarray(
            [row.get("objective_stability", np.nan) for row in clipped], dtype=float
        )
        stability = stability[np.isfinite(stability)]
        restart_gap = np.asarray(
            [row.get("optimizer_restart_objective_gap", np.nan) for row in clipped],
            dtype=float,
        )
        restart_gap = restart_gap[np.isfinite(restart_gap)]
        ratio_gap = np.asarray(
            [row.get("optimizer_restart_ratio_l1", np.nan) for row in clipped],
            dtype=float,
        )
        ratio_gap = ratio_gap[np.isfinite(ratio_gap)]
        gate_gap = np.asarray(
            [
                row.get("optimizer_restart_gate_disagreement", np.nan)
                for row in clipped
            ],
            dtype=float,
        )
        gate_gap = gate_gap[np.isfinite(gate_gap)]
        runtime = np.asarray(
            [row.get("runtime_sec", np.nan) for row in clipped], dtype=float
        )
        runtime = runtime[np.isfinite(runtime)]
        failure_rate = 1.0 - len(ok) / max(len(clipped), 1)
        maximum_stability = float(np.max(stability)) if stability.size else float("inf")
        maximum_restart_gap = (
            float(np.max(restart_gap)) if restart_gap.size else float("inf")
        )
        maximum_ratio_gap = float(np.max(ratio_gap)) if ratio_gap.size else float("inf")
        maximum_gate_gap = float(np.max(gate_gap)) if gate_gap.size else float("inf")
        runtime_mean = float(np.mean(runtime)) if runtime.size else float("inf")
        optimizer_config = _pilot_optimizer_config(clipped)
        eligible = bool(
            clipped
            and failure_rate == 0.0
            and stability.size == len(clipped)
            and restart_gap.size == len(clipped)
            and ratio_gap.size == len(clipped)
            and gate_gap.size == len(clipped)
            and maximum_stability <= float(objective_stability_threshold)
            and maximum_restart_gap <= float(objective_stability_threshold)
            and maximum_ratio_gap <= float(ratio_stability_threshold)
            and maximum_gate_gap <= float(gate_stability_threshold)
        )
        audit.append(
            {
                "candidate_id": candidate_id,
                "rows": len(clipped),
                "failure_rate": failure_rate,
                "objective_stability_max": maximum_stability,
                "optimizer_restart_objective_gap_max": maximum_restart_gap,
                "optimizer_restart_ratio_l1_max": maximum_ratio_gap,
                "optimizer_restart_gate_disagreement_max": maximum_gate_gap,
                "runtime_sec_mean": runtime_mean,
                "eligible": eligible,
                "optimizer_config": optimizer_config,
                "selector_uses_oracle_truth": False,
            }
        )
    eligible = [row for row in audit if row["eligible"]]
    selected = (
        min(eligible, key=lambda row: (row["runtime_sec_mean"], row["candidate_id"]))
        if eligible
        else None
    )
    return {
        "selected_candidate_id": None if selected is None else selected["candidate_id"],
        "objective_stability_threshold": float(objective_stability_threshold),
        "ratio_stability_threshold": float(ratio_stability_threshold),
        "gate_stability_threshold": float(gate_stability_threshold),
        "selector_uses_oracle_truth": False,
        "candidates": audit,
    }


def _pilot_deployable_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return the strict deployable-only view consumed by pilot selection."""
    allowed = (
        "method",
        "status",
        "objective_stability",
        "optimizer_restart_objective_gap",
        "optimizer_restart_ratio_l1",
        "optimizer_restart_gate_disagreement",
        "runtime_sec",
        "converged_fraction",
        "fixed_point_residual_bound",
        "error",
        "failure_type",
        "clipped_gate_steps",
        "clipped_ratio_steps",
        "clipped_gate_learning_rate",
        "clipped_ratio_learning_rate",
        "clipped_inner_relative_tolerance",
        "clipped_inner_gradient_tolerance",
        "clipped_inner_patience",
    )
    return {name: row.get(name) for name in allowed}


def _pilot_optimizer_config(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "clipped_gate_steps",
        "clipped_ratio_steps",
        "clipped_gate_learning_rate",
        "clipped_ratio_learning_rate",
        "clipped_inner_relative_tolerance",
        "clipped_inner_gradient_tolerance",
        "clipped_inner_patience",
    )
    if not rows:
        return {}
    config = {name: rows[0].get(name) for name in names}
    if any(any(row.get(name) != value for row in rows) for name, value in config.items()):
        raise ValueError("optimizer candidate rows contain inconsistent configuration")
    return config


def _resolved_learning_rate(value: float | None, backend: str) -> float:
    if value is not None:
        return float(value)
    return 5e-2 if backend == "linear" else 1e-3


def confirmatory_configs(*, seed: int = 12_000) -> list[CoverageRunConfig]:
    """Return the linear confirmatory workload after the neural pilot failed."""
    main_seed = 20_000_000 + int(seed)
    return [
        CoverageRunConfig(
            n=n,
            repetitions=repetitions,
            contexts=1,
            backends=("linear",),
            methods=("clipped_fori", "standard_fori", "posthoc_winsorized"),
            seed=main_seed,
        )
        for n, repetitions in ((2_000, 100), (10_000, 100), (50_000, 50))
    ]


def sensitivity_configs(*, seed: int = 22_000) -> list[CoverageRunConfig]:
    """Return the frozen clipping-level and lower-envelope appendix screens."""
    clipping_seed = 40_000_000 + int(seed)
    floor_seed = 41_000_000 + int(seed)
    configs = [
        CoverageRunConfig(
            n=10_000,
            repetitions=30,
            contexts=1,
            backends=("linear",),
            seed=clipping_seed + 1_000_000 * index,
            tau_upper=tau_upper,
            mass_grid=(0.0, 0.05, 0.5, 0.95, 1.0),
            methods=("clipped_fori", "standard_fori", "posthoc_winsorized"),
        )
        for index, tau_upper in enumerate((5.0, 20.0))
    ]
    configs.extend(
        CoverageRunConfig(
            n=10_000,
            repetitions=30,
            contexts=1,
            backends=("linear",),
            seed=floor_seed + 1_000_000 * index,
            tau_lower=tau_lower,
            mass_grid=(0.0, 0.05, 0.5, 1.0),
            methods=("clipped_fori",),
        )
        for index, tau_lower in enumerate((1e-6, 1e-3))
    )
    return configs


def main(argv: Sequence[str] | None = None) -> int:
    """Run a pilot, frozen paper experiment, or smoke benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "smoke",
            "pilot",
            "linear-verify",
            "standard-pilot",
            "standard-extension",
            "neural-screen",
            "neural-final",
            "confirmatory",
            "sensitivity",
        ),
        default="smoke",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=12_000)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--screen-selection", type=Path)
    args = parser.parse_args(argv)
    if args.num_shards <= 0 or not (0 <= args.shard_index < args.num_shards):
        parser.error("--shard-index must lie in [0, --num-shards)")
    pilot_stages = {
        "pilot",
        "linear-verify",
        "standard-pilot",
        "standard-extension",
        "neural-screen",
        "neural-final",
    }
    if args.stage in pilot_stages and args.num_shards != 1:
        parser.error("pilot selection must run unsharded so eligibility is complete")
    repo = repository_metadata(Path(__file__).resolve())
    if (
        args.stage in {"confirmatory", "sensitivity"}
        and repo.get("git_dirty")
        and not args.allow_dirty
    ):
        parser.error(
            "confirmatory and sensitivity runs require a clean worktree; "
            "use --allow-dirty only for an explicitly recorded exception"
        )
    freeze: dict[str, Any] | None = None
    if args.stage in {"confirmatory", "sensitivity"}:
        if args.freeze_manifest is None:
            parser.error("confirmatory and sensitivity runs require --freeze-manifest")
        freeze = load_freeze_manifest(
            args.freeze_manifest,
            expected_revision=str(repo.get("git_revision")),
        )
    if args.stage == "smoke":
        configs = [smoke_config(seed=args.seed)]
    elif args.stage == "pilot":
        configs = pilot_optimizer_configs(seed=args.seed)
    elif args.stage == "linear-verify":
        configs = [linear_verification_config()]
    elif args.stage == "standard-pilot":
        configs = standard_pilot_configs(seed=12_000_000 + args.seed)
    elif args.stage == "standard-extension":
        configs = [standard_extension_config(seed=12_000_000 + args.seed)]
    elif args.stage == "neural-screen":
        configs = neural_screen_configs(seed=10_000_000 + args.seed)
    elif args.stage == "neural-final":
        if args.screen_selection is None:
            parser.error("neural-final requires --screen-selection")
        screen_selection = json.loads(args.screen_selection.read_text())
        promoted = promoted_neural_ids(screen_selection)
        if len(promoted) != 2:
            parser.error("neural screen did not yield two eligible candidates")
        configs = neural_finalist_configs(
            promoted, seed=11_000_000 + args.seed
        )
    elif args.stage == "confirmatory":
        configs = confirmatory_configs(seed=args.seed)
    else:
        configs = sensitivity_configs(seed=args.seed)
    if freeze is not None:
        configs = [apply_freeze(config, freeze) for config in configs]
        frozen_grid = freeze.get("grids", {}).get(args.stage)
        if canonical_json([asdict(config) for config in configs]) != canonical_json(
            frozen_grid
        ):
            parser.error(
                f"{args.stage} workload does not match the committed freeze grid"
            )
    all_rows: list[dict[str, Any]] = []
    pilot_rows: list[tuple[str, Sequence[dict[str, Any]]]] = []
    aggregate_root = (
        args.output_dir
        if args.num_shards == 1
        else args.output_dir / f"shard_{args.shard_index:03d}"
    )
    for index, config in enumerate(configs):
        rows = run_coverage_benchmark(
            config,
            cell_dir=args.output_dir / "cells" / f"config_{index:02d}",
            resume=not args.no_resume,
            fail_fast=args.fail_fast,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            progress_path=aggregate_root / f"progress_config_{index:02d}.json",
        )
        if args.stage in {"standard-pilot", "standard-extension"}:
            candidate_id = standard_candidate_id(config)
        else:
            candidate_id = optimizer_candidate_id(config)
        for row in rows:
            row["candidate_id"] = candidate_id
        if args.stage in pilot_stages:
            pilot_rows.append((candidate_id, rows))
        if len(configs) == 1:
            all_rows.extend(rows)
        else:
            subdir = aggregate_root / f"config_{index:02d}"
            write_benchmark_artifacts(
                subdir,
                rows=rows,
                config=config,
                make_plots=False,
                allow_dirty=args.allow_dirty,
                require_complete=args.num_shards == 1,
            )
            all_rows.extend(rows)
    selection: dict[str, Any] | None = None
    if args.stage in pilot_stages:
        if args.stage in {"standard-pilot", "standard-extension"}:
            selection = select_standard_pilot(pilot_rows)
        else:
            selection = select_optimizer_pilot(pilot_rows)
        if args.stage == "neural-screen":
            selection["promoted_candidate_ids"] = promoted_neural_ids(selection)
        selection_path = aggregate_root / "pilot_selection.json"
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            selection_path, json.dumps(selection, indent=2, sort_keys=True) + "\n"
        )
        selected = selection["selected_candidate_id"]
        if args.stage == "neural-screen":
            eligible = len(selection["promoted_candidate_ids"]) == 2
        else:
            eligible = selected is not None
        if not eligible:
            raise RuntimeError(
                "no optimizer candidate met the predeclared stability guardrails"
            )
    workload_path = aggregate_root / "workload.json"
    workload_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        workload_path,
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "stage": args.stage,
                "configs": [asdict(config) for config in configs],
                "config_hashes": [configuration_hash(config) for config in configs],
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "resume": not args.no_resume,
                "allow_dirty": args.allow_dirty,
                "freeze_manifest": freeze,
                "oracle_used_for_selection": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    artifacts = write_benchmark_artifacts(
        aggregate_root,
        rows=all_rows,
        config=configs,
        make_plots=not args.no_plot and args.num_shards == 1,
        allow_dirty=args.allow_dirty,
        require_complete=args.num_shards == 1,
        selected_candidate_id=(
            None if selection is None else selection.get("selected_candidate_id")
        ),
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    print(f"workload: {workload_path}")
    if args.stage in pilot_stages:
        print(f"pilot_selection: {selection_path}")
    return 0


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, handle.getvalue())


def _validate_result_rows(rows: Sequence[dict[str, Any]]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        cell_id = row.get("cell_id")
        method = row.get("method")
        if cell_id is None:
            continue
        key = (str(cell_id), str(method))
        if key in seen:
            raise ValueError(f"duplicate method row for benchmark cell {key}")
        seen.add(key)


def _row_fold_cell_ids(rows: Sequence[dict[str, Any]]) -> set[str]:
    cell_ids: set[str] = set()
    for row in rows:
        value = row.get("fold_cell_ids", [])
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError as exc:
            raise ValueError("benchmark row has malformed fold_cell_ids") from exc
        if not isinstance(parsed, list) or len(parsed) != 2:
            raise ValueError("benchmark row must identify exactly two fold cells")
        cell_ids.update(str(item) for item in parsed)
    return cell_ids


def _row_fold_seeds(rows: Sequence[dict[str, Any]]) -> set[int]:
    seeds: set[int] = set()
    for row in rows:
        value = row.get("fold_seeds", [])
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError as exc:
            raise ValueError("benchmark row has malformed fold_seeds") from exc
        if not isinstance(parsed, list) or len(parsed) != 2:
            raise ValueError("benchmark row must identify exactly two fold seeds")
        seeds.update(int(item) for item in parsed)
    return seeds


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CoverageRunConfig",
    "DEFAULT_MASS_GRID",
    "SharedHubDataset",
    "SharedHubTruth",
    "analytic_shared_hub_rows",
    "confirmatory_configs",
    "make_shared_hub_dataset",
    "merge_benchmark_shards",
    "optimizer_candidate_id",
    "plot_coverage_summary",
    "pilot_optimizer_configs",
    "run_coverage_benchmark",
    "smoke_config",
    "select_optimizer_pilot",
    "sensitivity_configs",
    "solve_context_coverage_scale",
    "summarize_rows",
    "validate_benchmark_completeness",
    "write_benchmark_artifacts",
]
