"""Strict merger for sharded clipped-coverage benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from occupancy_ratio_benchmark._clipped_coverage_artifacts import (
    SCHEMA_VERSION,
    canonical_json,
    configuration_hash,
)
from occupancy_ratio_benchmark._clipped_coverage_data import CoverageRunConfig
from occupancy_ratio_benchmark.clipped_coverage import (
    merge_benchmark_shards,
    write_benchmark_artifacts,
)


def merge_shard_directories(
    shard_dirs: Sequence[str | Path],
    *,
    output_dir: str | Path,
    make_plots: bool = True,
) -> dict[str, Path]:
    """Validate and merge a complete set of compatible deterministic shards."""
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")
    workloads: list[dict[str, Any]] = []
    shards: list[list[dict[str, Any]]] = []
    for value in shard_dirs:
        root = Path(value)
        workload_path = root / "workload.json"
        results_path = root / "results.csv"
        if not workload_path.exists() or not results_path.exists():
            raise ValueError(f"shard directory {root} is missing required artifacts")
        workloads.append(json.loads(workload_path.read_text()))
        with results_path.open(newline="") as handle:
            shards.append(list(csv.DictReader(handle)))
    reference = workloads[0]
    if reference.get("schema") != SCHEMA_VERSION:
        raise ValueError("shard workload has an incompatible schema")
    num_shards = int(reference.get("num_shards", 0))
    if num_shards <= 0 or len(workloads) != num_shards:
        raise ValueError("all declared shards must be present before merging")
    indices = {int(workload.get("shard_index", -1)) for workload in workloads}
    if indices != set(range(num_shards)):
        raise ValueError("shard indices are incomplete or duplicated")
    compatibility_keys = (
        "schema",
        "stage",
        "configs",
        "config_hashes",
        "num_shards",
        "freeze_manifest",
        "oracle_used_for_selection",
    )
    reference_payload = {
        key: reference.get(key) for key in compatibility_keys
    }
    for workload in workloads[1:]:
        payload = {key: workload.get(key) for key in compatibility_keys}
        if canonical_json(payload) != canonical_json(reference_payload):
            raise ValueError("shard workloads are configuration-incompatible")
    configs = [CoverageRunConfig(**item) for item in reference["configs"]]
    if [configuration_hash(config) for config in configs] != list(
        reference["config_hashes"]
    ):
        raise ValueError("workload configuration hashes do not match payloads")
    rows = merge_benchmark_shards(shards, configs=configs)
    return write_benchmark_artifacts(
        output_dir,
        rows=rows,
        config=configs,
        make_plots=make_plots,
        allow_dirty=bool(reference.get("allow_dirty", False)),
        require_complete=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    artifacts = merge_shard_directories(
        args.input,
        output_dir=args.output_dir,
        make_plots=not args.no_plot,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "merge_shard_directories"]
