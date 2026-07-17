"""Truth-blind optimizer pilots for paper-ready clipped coverage experiments."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from occupancy_ratio_benchmark._clipped_coverage_data import CoverageRunConfig


LINEAR_SELECTED_ID = "g200_r300_glr5e-02_rlr2e-02_irt1e-10"


def linear_verification_config() -> CoverageRunConfig:
    """Reproduce the selected candidate on the original paired pilot cells."""
    return CoverageRunConfig(
        n=10_000,
        repetitions=10,
        mass_grid=(0.0, 0.25, 0.5, 0.9, 1.0),
        contexts=1,
        backends=("linear",),
        seed=41_000,
        clipped_gate_steps=200,
        clipped_ratio_steps=300,
        clipped_gate_learning_rate=0.05,
        clipped_ratio_learning_rate=0.02,
        clipped_inner_relative_tolerance=1e-10,
        standard_num_iterations=1,
        standard_optimizer_steps=1,
        optimizer_stability_restarts=3,
    )


def standard_pilot_configs(*, seed: int = 12_000_000) -> list[CoverageRunConfig]:
    """Return paired standard-FORE outer-budget candidates."""
    return [
        CoverageRunConfig(
            n=10_000,
            repetitions=10,
            mass_grid=(0.25, 0.5, 0.9, 1.0),
            contexts=1,
            backends=("linear",),
            methods=("standard_fori",),
            seed=seed,
            standard_num_iterations=iterations,
            standard_optimizer_steps=300,
            standard_outer_tolerance=1e-4,
            standard_objective_tolerance=1e-5,
            standard_mass_tolerance=1e-10,
            standard_require_convergence=True,
        )
        for iterations in (30, 100, 300)
    ]


def standard_extension_config(*, seed: int = 12_000_000) -> CoverageRunConfig:
    """Return the predeclared 600-iteration fallback candidate."""
    return replace(standard_pilot_configs(seed=seed)[-1], standard_num_iterations=600)


def standard_candidate_id(config: CoverageRunConfig) -> str:
    return f"standard_i{config.standard_num_iterations}_s{config.standard_optimizer_steps}"


def select_standard_pilot(
    candidate_rows: Sequence[tuple[str, Sequence[dict[str, Any]]]],
) -> dict[str, Any]:
    """Select the fastest converged standard budget without oracle fields."""
    audit: list[dict[str, Any]] = []
    for candidate_id, rows in candidate_rows:
        standard = [
            _standard_deployable_row(row)
            for row in rows
            if row.get("method") == "standard_fori"
        ]
        failures = [row for row in standard if row.get("status") != "ok"]
        weight_step = _finite_values(standard, "standard_weight_step_l1")
        objective = _finite_values(
            standard, "standard_objective_relative_change"
        )
        mass_error = _finite_values(standard, "standard_mass_abs_error")
        runtime = _finite_values(standard, "runtime_sec")
        optimizer_config = _standard_optimizer_config(standard)
        eligible = bool(
            standard
            and not failures
            and weight_step.size == len(standard)
            and objective.size == len(standard)
            and mass_error.size == len(standard)
            and np.max(weight_step) <= 1e-4
            and np.max(objective) <= 1e-5
            and np.max(mass_error) <= 1e-10
        )
        audit.append(
            {
                "candidate_id": candidate_id,
                "rows": len(standard),
                "failure_rate": len(failures) / max(1, len(standard)),
                "standard_weight_step_l1_max": _max_or_inf(weight_step),
                "standard_objective_relative_change_max": _max_or_inf(objective),
                "standard_mass_abs_error_max": _max_or_inf(mass_error),
                "runtime_sec_mean": (
                    float(np.mean(runtime)) if runtime.size else float("inf")
                ),
                "eligible": eligible,
                "optimizer_config": optimizer_config,
                "selector_uses_oracle_truth": False,
            }
        )
    eligible_rows = [row for row in audit if row["eligible"]]
    selected = (
        min(
            eligible_rows,
            key=lambda row: (row["runtime_sec_mean"], row["candidate_id"]),
        )
        if eligible_rows
        else None
    )
    return {
        "selected_candidate_id": None if selected is None else selected["candidate_id"],
        "standard_weight_step_l1_threshold": 1e-4,
        "standard_objective_relative_change_threshold": 1e-5,
        "standard_mass_abs_error_threshold": 1e-10,
        "selector_uses_oracle_truth": False,
        "candidates": audit,
    }


def neural_screen_configs(*, seed: int = 10_000_000) -> list[CoverageRunConfig]:
    """Return the four inexpensive contextual-neural screen candidates."""
    candidates = ((1e-3, 1e-8), (5e-4, 1e-8), (1e-3, 1e-10), (5e-4, 1e-10))
    return [
        CoverageRunConfig(
            n=10_000,
            repetitions=3,
            mass_grid=(0.0, 0.5, 1.0),
            contexts=64,
            backends=("neural",),
            methods=("clipped_fori",),
            seed=seed,
            clipped_gate_steps=200,
            clipped_ratio_steps=300,
            clipped_gate_learning_rate=1e-3,
            clipped_ratio_learning_rate=ratio_lr,
            clipped_inner_relative_tolerance=inner_tolerance,
            optimizer_stability_restarts=1,
        )
        for ratio_lr, inner_tolerance in candidates
    ]


def neural_finalist_configs(
    promoted_ids: Sequence[str], *, seed: int = 11_000_000
) -> list[CoverageRunConfig]:
    """Return paired three-restart configs for the two promoted candidates."""
    by_id = {
        _clipped_candidate_id(config): config for config in neural_screen_configs()
    }
    if len(promoted_ids) != 2 or any(value not in by_id for value in promoted_ids):
        raise ValueError("neural final requires two valid promoted candidate ids")
    return [
        replace(
            by_id[candidate_id],
            repetitions=5,
            mass_grid=(0.1, 0.5, 0.9),
            seed=seed,
            optimizer_stability_restarts=3,
        )
        for candidate_id in promoted_ids
    ]


def promoted_neural_ids(selection: dict[str, Any]) -> list[str]:
    """Promote the fastest two zero-failure screen candidates."""
    eligible = [row for row in selection["candidates"] if bool(row["eligible"])]
    return [
        str(row["candidate_id"])
        for row in sorted(
            eligible, key=lambda row: (row["runtime_sec_mean"], row["candidate_id"])
        )[:2]
    ]


def _clipped_candidate_id(config: CoverageRunConfig) -> str:
    return (
        f"g{config.clipped_gate_steps}_r{config.clipped_ratio_steps}"
        f"_glr{float(config.clipped_gate_learning_rate):.0e}"
        f"_rlr{float(config.clipped_ratio_learning_rate):.0e}"
        f"_irt{config.clipped_inner_relative_tolerance:.0e}"
    )


def _standard_deployable_row(row: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "method",
        "status",
        "failure_type",
        "error",
        "runtime_sec",
        "standard_num_iterations",
        "standard_optimizer_steps",
        "standard_outer_tolerance",
        "standard_objective_tolerance",
        "standard_mass_tolerance",
        "standard_weight_step_l1",
        "standard_objective_relative_change",
        "standard_mass_abs_error",
    }
    return {name: row.get(name) for name in allowed}


def _standard_optimizer_config(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "standard_num_iterations",
        "standard_optimizer_steps",
        "standard_outer_tolerance",
        "standard_objective_tolerance",
        "standard_mass_tolerance",
    )
    if not rows:
        return {}
    config = {name: rows[0].get(name) for name in names}
    config["standard_require_convergence"] = True
    if any(any(row.get(name) != value for row in rows) for name, value in config.items() if name != "standard_require_convergence"):
        raise ValueError("standard candidate rows contain inconsistent configuration")
    return config


def _finite_values(rows: Sequence[dict[str, Any]], name: str) -> np.ndarray:
    values = np.asarray([row.get(name, np.nan) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def _max_or_inf(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else float("inf")


__all__ = [
    "LINEAR_SELECTED_ID",
    "linear_verification_config",
    "neural_finalist_configs",
    "neural_screen_configs",
    "promoted_neural_ids",
    "select_standard_pilot",
    "standard_candidate_id",
    "standard_extension_config",
    "standard_pilot_configs",
]
