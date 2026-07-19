"""Create the versioned optimizer freeze for clipped-coverage experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

from occupancy_ratio_benchmark._clipped_coverage_artifacts import (
    environment_metadata,
    repository_metadata,
)
from occupancy_ratio_benchmark._clipped_coverage_freeze import (
    CLIPPED_FIELDS,
    LINEAR_PILOT_SHA256,
    STANDARD_FIELDS,
    compare_linear_reproduction,
    apply_freeze,
    selection_artifact,
    write_freeze_manifest,
)
from occupancy_ratio_benchmark.clipped_coverage import (
    confirmatory_configs,
    sensitivity_configs,
)


def build_freeze_payload(
    *,
    source_revision: str,
    environment: dict[str, Any],
    linear_selection_path: str | Path,
    linear_verification_path: str | Path,
    standard_selection_path: str | Path,
    neural_exclusion_audit_path: str | Path,
    reproduction_audit: dict[str, Any],
    source_compatibility: dict[str, Any],
) -> dict[str, Any]:
    """Build a strict oracle-free freeze payload from completed pilot audits."""
    linear_selection = _load_selection(linear_selection_path)
    linear_verification = _load_selection(linear_verification_path)
    standard_selection = _load_selection(standard_selection_path)
    neural_exclusion = _load_neural_exclusion(neural_exclusion_audit_path)
    linear_artifact = selection_artifact(linear_selection_path)
    if linear_artifact["sha256"] != LINEAR_PILOT_SHA256:
        raise ValueError("clipped-linear pilot artifact does not match the fixed hash")
    linear_config = _selected_config(linear_selection, CLIPPED_FIELDS)
    verification_config = _selected_config(linear_verification, CLIPPED_FIELDS)
    if linear_config != verification_config:
        raise ValueError("clean-checkout linear verification changed optimizer settings")
    if not bool(reproduction_audit.get("matched", False)):
        raise ValueError("clean-checkout linear predictions do not match the pilot")
    if not bool(source_compatibility.get("reproduction_critical_files_unchanged")):
        raise ValueError("reproduction-critical source files changed after verification")
    standard_config = _selected_config(standard_selection, STANDARD_FIELDS)
    optimizer_configs = {
        "clipped_linear": linear_config,
        "standard_linear": standard_config,
    }
    freeze_view = {"optimizer_configs": optimizer_configs}
    confirmatory = [
        apply_freeze(config, freeze_view) for config in confirmatory_configs()
    ]
    sensitivities = [
        apply_freeze(config, freeze_view) for config in sensitivity_configs()
    ]
    return {
        "source_revision": source_revision,
        "oracle_used_for_selection": False,
        "optimizer_configs": optimizer_configs,
        "pilot_artifacts": {
            "clipped_linear": {
                **linear_artifact,
                "verification": selection_artifact(linear_verification_path),
                "reproduction": reproduction_audit,
            },
            "standard_linear": selection_artifact(standard_selection_path),
        },
        "excluded_components": {
            "contextual_neural_appendix": {
                "reason": str(neural_exclusion["reason"]),
                "completed_candidates": int(
                    neural_exclusion["completed_candidates"]
                ),
                "eligible_candidates": int(neural_exclusion["eligible_candidates"]),
                "audit_path": str(Path(neural_exclusion_audit_path).resolve()),
                "audit_sha256": selection_artifact(neural_exclusion_audit_path)[
                    "sha256"
                ],
            }
        },
        "source_compatibility": source_compatibility,
        "grids": {
            "confirmatory": [asdict(config) for config in confirmatory],
            "sensitivity": [asdict(config) for config in sensitivities],
        },
        "seed_namespaces": {
            "linear_verification": 41_000,
            "standard_pilot": 12_000_000,
            "confirmatory_main": int(confirmatory[0].seed),
            "sensitivity_clipping": int(sensitivities[0].seed),
            "sensitivity_floor": int(sensitivities[2].seed),
        },
        "expected_counts": {
            "confirmatory_dataset_cells": 2_500,
            "confirmatory_fold_artifacts": 5_000,
            "confirmatory_method_rows": 7_500,
            "sensitivity_dataset_cells": 540,
            "sensitivity_fold_artifacts": 1_080,
            "sensitivity_method_rows": 1_140,
        },
        "environment": environment,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-selection", type=Path, required=True)
    parser.add_argument("--linear-verification", type=Path, required=True)
    parser.add_argument("--reference-cells", type=Path, required=True)
    parser.add_argument("--reproduced-cells", type=Path, required=True)
    parser.add_argument("--standard-selection", type=Path, required=True)
    parser.add_argument("--neural-exclusion-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = repository_metadata(Path(__file__).resolve())
    if repo.get("git_dirty"):
        parser.error("freeze creation requires a clean committed worktree")
    revision = repo.get("git_revision")
    if not isinstance(revision, str) or not revision:
        parser.error("freeze creation requires a Git revision")
    reproduction = compare_linear_reproduction(
        args.reference_cells, args.reproduced_cells
    )
    verification_manifest = json.loads(
        (args.linear_verification.parent / "manifest.json").read_text()
    )
    verification_revision = verification_manifest["repository"]["git_revision"]
    compatibility = source_compatibility_audit(
        repository_root=Path(str(repo["git_root"])),
        verification_revision=str(verification_revision),
        current_revision=revision,
    )
    payload = build_freeze_payload(
        source_revision=revision,
        environment=environment_metadata(include_torch=True),
        linear_selection_path=args.linear_selection,
        linear_verification_path=args.linear_verification,
        standard_selection_path=args.standard_selection,
        neural_exclusion_audit_path=args.neural_exclusion_audit,
        reproduction_audit=reproduction,
        source_compatibility=compatibility,
    )
    path = write_freeze_manifest(args.output, payload)
    print(f"freeze_manifest: {path}")
    return 0


def _load_selection(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if bool(payload.get("selector_uses_oracle_truth", True)):
        raise ValueError("pilot selection is not truth-blind")
    return payload


def _load_neural_exclusion(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != "clipped-neural-exclusion-v1":
        raise ValueError("neural exclusion audit has an unknown schema")
    if bool(payload.get("oracle_used_for_selection", True)):
        raise ValueError("neural exclusion decision is not truth blind")
    if payload.get("decision") != "exclude_contextual_neural_appendix":
        raise ValueError("neural exclusion audit has the wrong decision")
    if int(payload.get("completed_candidates", 0)) < 1:
        raise ValueError("neural exclusion audit has no completed candidates")
    if int(payload.get("eligible_candidates", -1)) != 0:
        raise ValueError("neural exclusion audit contradicts the pilot gate")
    return payload


REPRODUCTION_CRITICAL_PATHS = (
    "packages/occupancy-ratio/occupancy_ratio/_clipped_kl_fori_backend_common.py",
    "packages/occupancy-ratio/occupancy_ratio/_clipped_kl_fori_diagnostics.py",
    "packages/occupancy-ratio/occupancy_ratio/_clipped_kl_fori_impl.py",
    "packages/occupancy-ratio/occupancy_ratio/_clipped_kl_fori_linear.py",
    "packages/occupancy-ratio/occupancy_ratio/_clipped_kl_fori_neural.py",
    "packages/occupancy-ratio/occupancy_ratio/_clipped_kl_fori_objectives.py",
    "packages/occupancy-ratio/occupancy_ratio/_clipped_kl_fori_types.py",
    "packages/occupancy-ratio/occupancy_ratio/_fori_data.py",
    "packages/occupancy-ratio/occupancy_ratio/_kl_fori_impl.py",
    "packages/occupancy-ratio/occupancy_ratio_benchmark/_clipped_coverage_artifacts.py",
    "packages/occupancy-ratio/occupancy_ratio_benchmark/_clipped_coverage_data.py",
    "packages/occupancy-ratio/occupancy_ratio_benchmark/_clipped_coverage_execution.py",
    "packages/occupancy-ratio/occupancy_ratio_benchmark/_clipped_coverage_metrics.py",
    "packages/occupancy-ratio/occupancy_ratio_benchmark/_clipped_coverage_oracle.py",
)


def source_compatibility_audit(
    *,
    repository_root: Path,
    verification_revision: str,
    current_revision: str,
) -> dict[str, Any]:
    """Verify that the fitted-estimator execution path is byte-identical."""
    old = hashlib.sha256()
    current = hashlib.sha256()
    for relative in REPRODUCTION_CRITICAL_PATHS:
        historical = subprocess.run(
            ["git", "show", f"{verification_revision}:{relative}"],
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        present = (repository_root / relative).read_bytes()
        old.update(relative.encode())
        old.update(historical)
        current.update(relative.encode())
        current.update(present)
    historical_hash = old.hexdigest()
    current_hash = current.hexdigest()
    return {
        "verification_revision": verification_revision,
        "current_revision": current_revision,
        "critical_paths": list(REPRODUCTION_CRITICAL_PATHS),
        "verification_source_hash": historical_hash,
        "current_source_hash": current_hash,
        "reproduction_critical_files_unchanged": historical_hash == current_hash,
    }


def _selected_config(
    selection: dict[str, Any], expected_fields: set[str]
) -> dict[str, Any]:
    selected = selection.get("selected_candidate_id")
    candidates = selection.get("candidates")
    if not isinstance(selected, str) or not isinstance(candidates, list):
        raise ValueError("pilot selection is malformed")
    rows = [row for row in candidates if row.get("candidate_id") == selected]
    if len(rows) != 1 or not bool(rows[0].get("eligible", False)):
        raise ValueError("pilot selection did not choose one eligible candidate")
    config = rows[0].get("optimizer_config")
    if not isinstance(config, dict) or set(config) != expected_fields:
        raise ValueError("pilot optimizer configuration has malformed fields")
    return dict(config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "REPRODUCTION_CRITICAL_PATHS",
    "build_freeze_payload",
    "main",
    "source_compatibility_audit",
]
