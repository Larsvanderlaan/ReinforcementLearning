"""CLI for the structural-support, external-test stopped-FORE benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np

from occupancy_ratio_benchmark._stopped_fore_data import (
    StoppedFOREExternalConfig,
    make_structural_stopped_dataset,
)
from occupancy_ratio_benchmark._stopped_fore_execution import run_external_test_cell


SCHEMA = "stopped-fore-external-v1"


def run_benchmark(
    config: StoppedFOREExternalConfig,
    *,
    output_dir: str | Path,
    shard_index: int = 0,
    num_shards: int = 1,
    resume: bool = True,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    """Run deterministic cells assigned to one process-safe hash shard."""
    if num_shards <= 0 or not (0 <= shard_index < num_shards):
        raise ValueError("shard_index must lie in [0, num_shards).")
    root = Path(output_dir)
    cells_dir = root / "cells"
    arrays_dir = root / "arrays"
    cells_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _config_hash(config)
    assigned = [
        cell
        for cell in _expected_cells(config)
        if int(cell["cell_id"], 16) % num_shards == shard_index
    ]
    rows: list[dict[str, Any]] = []
    started = time.time()
    for completed, cell in enumerate(assigned, start=1):
        cell_path = cells_dir / f"{cell['cell_id']}.json"
        if resume and cell_path.exists():
            payload = _read_cell(
                cell_path, cell_id=cell["cell_id"], config_hash=config_hash
            )
            cell_rows = payload["rows"]
        else:
            train = make_structural_stopped_dataset(
                n=config.n_train,
                gamma=config.gamma,
                contexts=config.contexts,
                behavior_probability=config.behavior_probability,
                support_fraction=cell["support_fraction"],
                failure_mode=cell["failure_mode"],
                seed=cell["train_seed"],
                irrelevant_features=config.irrelevant_features,
            )
            test = make_structural_stopped_dataset(
                n=config.n_test,
                gamma=config.gamma,
                contexts=config.contexts,
                behavior_probability=config.behavior_probability,
                support_fraction=cell["support_fraction"],
                failure_mode=cell["failure_mode"],
                seed=cell["test_seed"],
                irrelevant_features=config.irrelevant_features,
            )
            cell_rows, arrays = run_external_test_cell(
                train,
                test,
                config,
                backend=cell["backend"],
                repetition=cell["repetition"],
                train_seed=cell["train_seed"],
                test_seed=cell["test_seed"],
            )
            for row in cell_rows:
                row.update(
                    {
                        "schema": SCHEMA,
                        "config_hash": config_hash,
                        "cell_id": cell["cell_id"],
                    }
                )
            arrays_path = arrays_dir / f"{cell['cell_id']}.npz"
            _atomic_write_npz(arrays_path, arrays)
            _atomic_write_json(
                cell_path,
                {
                    "schema": SCHEMA,
                    "config_hash": config_hash,
                    "cell_id": cell["cell_id"],
                    "cell": cell,
                    "arrays": str(arrays_path.relative_to(root)),
                    "rows": cell_rows,
                    "created_unix": time.time(),
                },
            )
        rows.extend(cell_rows)
        elapsed = time.time() - started
        rate = completed / elapsed if elapsed > 0.0 else 0.0
        _atomic_write_json(
            root / f"progress_shard_{shard_index:02d}.json",
            {
                "schema": SCHEMA,
                "config_hash": config_hash,
                "shard_index": shard_index,
                "num_shards": num_shards,
                "assigned_cells": len(assigned),
                "completed_cells": completed,
                "elapsed_sec": elapsed,
                "eta_sec": (
                    (len(assigned) - completed) / rate if rate > 0.0 else None
                ),
                "last_cell_id": cell["cell_id"],
                "last_statuses": {
                    row["method"]: row["status"] for row in cell_rows
                },
                "updated_unix": time.time(),
            },
        )
        if fail_fast and any(row.get("status") != "ok" for row in cell_rows):
            raise RuntimeError(
                f"cell {cell['cell_id']} failed a stopped-FORE fail-fast guardrail"
            )
    _write_csv(root / f"results_shard_{shard_index:02d}.csv", rows)
    return rows


def merge_benchmark(
    config: StoppedFOREExternalConfig,
    *,
    output_dir: str | Path,
    require_complete: bool = True,
) -> dict[str, Path]:
    """Merge atomic cells and write publication-facing results and summaries."""
    root = Path(output_dir)
    config_hash = _config_hash(config)
    expected = _expected_cells(config)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for cell in expected:
        path = root / "cells" / f"{cell['cell_id']}.json"
        if not path.exists():
            missing.append(cell["cell_id"])
            continue
        rows.extend(
            _read_cell(
                path, cell_id=cell["cell_id"], config_hash=config_hash
            )["rows"]
        )
    if require_complete and missing:
        raise ValueError(
            f"benchmark is missing {len(missing)} cells; first ids: {missing[:5]}"
        )
    _validate_rows(rows, config)
    results_path = root / "results.csv"
    summary_path = root / "summary.csv"
    manifest_path = root / "manifest.json"
    _write_csv(results_path, rows)
    _write_csv(summary_path, summarize_rows(rows))
    _atomic_write_json(
        manifest_path,
        {
            "schema": SCHEMA,
            "config": asdict(config),
            "config_hash": config_hash,
            "repository": _repository_metadata(),
            "environment": _environment_metadata(),
            "command": list(sys.argv),
            "expected_cells": len(expected),
            "completed_cells": len(expected) - len(missing),
            "missing_cell_ids": missing,
            "target_estimand": "unclipped_coverage_stopped_occupancy",
            "fit_scope": "single_full_training_sample",
            "evaluation_scope": "independent_external_test",
            "crossfit_folds": 0,
            "calibration_method": "none",
            "oracle_used_for_fitting_or_selection": False,
            "created_unix": time.time(),
        },
    )
    return {
        "results": results_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate accuracy, gate, tail, and runtime diagnostics by study cell."""
    metric_names = (
        "ratio_l1",
        "ratio_rmse",
        "mass_abs_error",
        "value_mean_abs_error",
        "value_constant_abs_error",
        "value_initial_abs_error",
        "value_hub_abs_error",
        "value_context_abs_error",
        "external_bellman_l2",
        "gate_initial_singular_accept_rate",
        "gate_successor_singular_accept_rate",
        "ess_fraction",
        "top_1pct_mass_fraction",
        "weight_max",
        "runtime_sec",
    )
    keys = (
        "method",
        "backend",
        "failure_mode",
        "support_fraction",
        "n_train",
        "n_test",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    summary: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        result = dict(zip(keys, group_key))
        result["replications"] = len(group_rows)
        result["successful_replications"] = sum(
            row.get("status") == "ok" for row in group_rows
        )
        for metric in metric_names:
            values = np.asarray(
                [
                    float(row[metric])
                    for row in group_rows
                    if row.get("status") == "ok"
                    and metric in row
                    and _is_finite_number(row[metric])
                ],
                dtype=np.float64,
            )
            result[f"{metric}_mean"] = (
                float(np.mean(values)) if values.size else float("nan")
            )
            result[f"{metric}_median"] = (
                float(np.median(values)) if values.size else float("nan")
            )
        summary.append(result)
    return summary


def load_config(path: str | Path) -> StoppedFOREExternalConfig:
    """Load a strict JSON config for the stopped-FORE benchmark."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("benchmark config must be a JSON object.")
    return StoppedFOREExternalConfig(**payload)


def _expected_cells(config: StoppedFOREExternalConfig) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    config_hash = _config_hash(config)
    for repetition in range(config.repetitions):
        for mode_index, failure_mode in enumerate(config.failure_modes):
            for support_index, support_fraction in enumerate(config.support_fractions):
                train_seed = int(
                    config.seed
                    + 100_003 * repetition
                    + 10_007 * mode_index
                    + 997 * support_index
                )
                test_seed = int(train_seed + 50_000_003)
                for backend in config.backends:
                    identity = {
                        "config_hash": config_hash,
                        "repetition": repetition,
                        "failure_mode": str(failure_mode),
                        "support_fraction": float(support_fraction),
                        "backend": str(backend),
                        "train_seed": train_seed,
                        "test_seed": test_seed,
                    }
                    cells.append(
                        identity
                        | {
                            "cell_id": hashlib.sha256(
                                _canonical_json(identity).encode()
                            ).hexdigest()[:24]
                        }
                    )
    return cells


def _validate_rows(
    rows: Sequence[dict[str, Any]], config: StoppedFOREExternalConfig
) -> None:
    expected_methods = set(config.methods)
    by_cell: dict[str, set[str]] = {}
    for row in rows:
        if row.get("schema") != SCHEMA:
            raise ValueError("row has an incompatible schema.")
        if row.get("config_hash") != _config_hash(config):
            raise ValueError("row has an incompatible config hash.")
        if row.get("calibration_method") != "none" or row.get("crossfit_folds") != 0:
            raise ValueError("stopped-FORE rows must not use calibration or cross-fitting.")
        if row.get("target_estimand") != "unclipped_coverage_stopped_occupancy":
            raise ValueError("row targets the wrong occupancy estimand.")
        by_cell.setdefault(str(row["cell_id"]), set()).add(str(row["method"]))
    for cell_id, methods in by_cell.items():
        if methods != expected_methods:
            raise ValueError(
                f"cell {cell_id} has methods {methods}, expected {expected_methods}."
            )


def _read_cell(path: Path, *, cell_id: str, config_hash: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"cell {path} has an incompatible schema.")
    if payload.get("cell_id") != cell_id:
        raise ValueError(f"cell {path} has an incompatible identity.")
    if payload.get("config_hash") != config_hash:
        raise ValueError(f"cell {path} has an incompatible config hash.")
    if not isinstance(payload.get("rows"), list):
        raise ValueError(f"cell {path} is missing result rows.")
    return payload


def _config_hash(config: StoppedFOREExternalConfig) -> str:
    return hashlib.sha256(_canonical_json(asdict(config)).encode()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _repository_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True
        ).strip()
    except Exception:
        return {"commit": "unknown", "dirty": None}
    return {"commit": commit, "dirty": bool(status), "status": status.splitlines()}


def _environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch

        metadata["torch"] = torch.__version__
        metadata["torch_threads"] = torch.get_num_threads()
    except Exception as exc:
        metadata["torch"] = f"unavailable: {type(exc).__name__}: {exc}"
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--num-shards", type=int, default=1)
    run_parser.add_argument("--no-resume", action="store_true")
    run_parser.add_argument("--fail-fast", action="store_true")
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--config", required=True)
    merge_parser.add_argument("--output-dir", required=True)
    merge_parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "run":
        rows = run_benchmark(
            config,
            output_dir=args.output_dir,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            resume=not args.no_resume,
            fail_fast=args.fail_fast,
        )
        print(json.dumps({"completed_rows": len(rows)}, sort_keys=True))
        return 0
    artifacts = merge_benchmark(
        config,
        output_dir=args.output_dir,
        require_complete=not args.allow_incomplete,
    )
    print(json.dumps({key: str(value) for key, value in artifacts.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA",
    "load_config",
    "main",
    "merge_benchmark",
    "run_benchmark",
    "summarize_rows",
]
