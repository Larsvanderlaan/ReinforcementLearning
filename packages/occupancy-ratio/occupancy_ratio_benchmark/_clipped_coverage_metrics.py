"""Metric construction and aggregation for the clipped coverage benchmark."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from occupancy_ratio_benchmark._clipped_coverage_data import SharedHubDataset


Array = np.ndarray


def metric_row(
    *,
    common: dict[str, Any],
    method: str,
    prediction: Array,
    own_ratio: Array,
    recursive_ratio: Array,
    dataset: SharedHubDataset,
    own_target: str,
    train_mass: float,
    runtime_sec: float,
    gate_prediction: Array | None,
    diagnostics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build one auditable estimator row against method-specific truth."""
    mass = float(np.mean(prediction))
    own_mass = {
        "full": 1.0,
        "clipped": dataset.truth.projected_mass(float(common["tau_lower"])),
        "posthoc": dataset.truth.posthoc_mass,
    }[own_target]
    row = common | {
        "method": method,
        "status": "ok",
        "failure_type": "",
        "error": "",
        "crossfit_mass": mass,
        "training_mass": float(train_mass),
        "mass_own_target": float(own_mass),
        "mass_own_error": mass - float(own_mass),
        "mass_own_abs_error": abs(mass - float(own_mass)),
        "mass_bias_vs_recursive": mass - dataset.truth.retained_mass,
        "mass_abs_error_vs_recursive": abs(mass - dataset.truth.retained_mass),
        "mass_error_vs_projected": mass
        - dataset.truth.projected_mass(float(common["tau_lower"])),
        "mass_abs_error_vs_projected": abs(
            mass - dataset.truth.projected_mass(float(common["tau_lower"]))
        ),
        "ratio_l1_own_target": float(np.mean(np.abs(prediction - own_ratio))),
        "ratio_l1_recursive_target": float(
            np.mean(np.abs(prediction - recursive_ratio))
        ),
        "ess_fraction": ess_fraction(prediction),
        "weight_max": float(np.max(prediction)),
        "weight_p95": float(np.quantile(prediction, 0.95)),
        "weight_p99": float(np.quantile(prediction, 0.99)),
        "runtime_sec": float(runtime_sec),
        "converged_fraction": diagnostic_mean(diagnostics, "converged"),
        "fixed_point_residual_bound": diagnostic_mean(
            diagnostics, "fixed_point_residual_bound"
        ),
        "objective_stability": diagnostic_max(
            diagnostics, "objective_stability_final"
        ),
        "lower_envelope_fraction": diagnostic_mean(
            diagnostics, "lower_envelope_fraction"
        ),
        "upper_envelope_fraction": diagnostic_mean(
            diagnostics, "upper_envelope_fraction"
        ),
        "rejected_bellman_mass_fraction": diagnostic_mean(
            diagnostics, "rejected_bellman_mass_fraction"
        ),
        "standard_weight_step_l1": diagnostic_mean(
            diagnostics, "standard_weight_step_l1"
        ),
        "standard_objective_relative_change": diagnostic_mean(
            diagnostics, "standard_objective_relative_change"
        ),
        "standard_mass_abs_error": diagnostic_mean(
            diagnostics, "standard_mass_abs_error"
        ),
    }
    empirical_mass = float(common.get("empirical_projected_oracle_mass", np.nan))
    if method == "clipped_fori" and np.isfinite(empirical_mass):
        row["mass_fitting_error_vs_empirical_support"] = mass - empirical_mass
        row["mass_sampling_error_empirical_vs_population"] = (
            empirical_mass - float(common["projected_oracle_mass"])
        )
    else:
        row["mass_fitting_error_vs_empirical_support"] = float("nan")
        row["mass_sampling_error_empirical_vs_population"] = float("nan")
    row["gate_error"] = (
        float(np.mean(gate_prediction != dataset.gate_indicator))
        if gate_prediction is not None
        else float("nan")
    )
    for reward, reward_values in dataset.rewards.items():
        estimated = float(np.mean(prediction * reward_values))
        truth = dataset.truth.values(reward)
        full = float(truth["full"])
        own = float(truth[own_target])
        upper = estimated + max(0.0, 1.0 - mass)
        row[f"{reward}_value"] = estimated
        row[f"{reward}_own_target"] = own
        row[f"{reward}_own_error"] = estimated - own
        row[f"{reward}_own_abs_error"] = abs(estimated - own)
        row[f"{reward}_recursive_target"] = float(truth["clipped"])
        row[f"{reward}_recursive_error"] = estimated - float(truth["clipped"])
        row[f"{reward}_recursive_abs_error"] = abs(estimated - float(truth["clipped"]))
        row[f"{reward}_plugin_overshoot"] = float(estimated > full)
        row[f"{reward}_plugin_interval_contains_full"] = float(
            estimated <= full <= upper
        )
        row[f"{reward}_plugin_interval_width"] = max(0.0, 1.0 - mass)
    return row


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate analyzable rows without using truth for method selection."""
    metric_names = (
        "crossfit_mass",
        "mass_bias_vs_recursive",
        "mass_own_error",
        "mass_own_abs_error",
        "mass_abs_error_vs_recursive",
        "ratio_l1_own_target",
        "ratio_l1_recursive_target",
        "ess_fraction",
        "weight_max",
        "weight_p95",
        "weight_p99",
        "runtime_sec",
        "converged_fraction",
        "fixed_point_residual_bound",
        "objective_stability",
        "gate_error",
        "hub_value",
        "hub_own_target",
        "constant_own_abs_error",
        "hub_own_abs_error",
        "constant_plugin_overshoot",
        "hub_plugin_overshoot",
        "constant_plugin_interval_contains_full",
        "hub_plugin_interval_contains_full",
        "mass_error_vs_projected",
        "mass_abs_error_vs_projected",
        "lower_envelope_fraction",
        "upper_envelope_fraction",
        "rejected_bellman_mass_fraction",
        "optimizer_restart_objective_gap",
        "optimizer_restart_self_objective_gap",
        "optimizer_restart_ratio_l1",
        "optimizer_restart_gate_disagreement",
        "optimizer_restart_mass_gap",
        "mass_fitting_error_vs_empirical_support",
        "mass_sampling_error_empirical_vs_population",
        "standard_weight_step_l1",
        "standard_objective_relative_change",
        "standard_mass_abs_error",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = (
        "candidate_id",
        "config_hash",
        "method",
        "backend",
        "n",
        "contexts",
        "sampling_mode",
        "irrelevant_features",
        "termination_encoding",
        "oracle_mass",
        "gamma",
        "tau_lower",
        "tau_upper",
    )
    for row in rows:
        key = tuple(row.get(name) for name in keys)
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for key, group in groups.items():
        ok = [row for row in group if row.get("status") == "ok"]
        analyzable = [
            row
            for row in group
            if row.get("status") in {"ok", "nonconverged", "optimizer_unstable"}
        ]
        out = {name: value for name, value in zip(keys, key)}
        out["rows"] = len(group)
        out["ok_rows"] = len(ok)
        out["error_rows"] = sum(row.get("status") == "error" for row in group)
        out["out_of_regime_rows"] = sum(
            row.get("status") == "out_of_regime" for row in group
        )
        out["nonconverged_rows"] = sum(
            row.get("status") == "nonconverged" for row in group
        )
        out["optimizer_unstable_rows"] = sum(
            row.get("status") == "optimizer_unstable" for row in group
        )
        out["failure_rate"] = out["error_rows"] / len(group)
        out["fit_failure_rate"] = (
            out["error_rows"]
            + out["nonconverged_rows"]
            + out["optimizer_unstable_rows"]
        ) / len(group)
        out["out_of_regime_rate"] = out["out_of_regime_rows"] / len(group)
        for metric in metric_names:
            values = np.asarray(
                [row.get(metric, np.nan) for row in analyzable], dtype=np.float64
            )
            finite = values[np.isfinite(values)]
            out[f"{metric}_mean"] = (
                float(np.mean(finite)) if finite.size else float("nan")
            )
            out[f"{metric}_se"] = (
                float(np.std(finite, ddof=1) / np.sqrt(finite.size))
                if finite.size > 1
                else float("nan")
            )
        for error_name in (
            "mass_bias_vs_recursive",
            "mass_own_error",
            "mass_error_vs_projected",
            "mass_fitting_error_vs_empirical_support",
            "mass_sampling_error_empirical_vs_population",
            "constant_own_error",
            "hub_own_error",
            "gate_own_error",
            "constant_recursive_error",
            "hub_recursive_error",
            "gate_recursive_error",
        ):
            values = np.asarray(
                [row.get(error_name, np.nan) for row in analyzable], dtype=np.float64
            )
            finite = values[np.isfinite(values)]
            out[f"{error_name}_rmse"] = (
                float(np.sqrt(np.mean(finite * finite)))
                if finite.size
                else float("nan")
            )
        summary.append(out)
    return summary


def finite_mean(values: Iterable[float]) -> float:
    """Return a mean over finite values or NaN when none exist."""
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def diagnostic_max(diagnostics: Sequence[dict[str, Any]], name: str) -> float:
    """Return the maximum finite fold diagnostic or NaN when unavailable."""
    values = np.asarray([item.get(name, np.nan) for item in diagnostics], dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def ess_fraction(weight: Array) -> float:
    """Return effective sample size divided by row count."""
    value = np.asarray(weight, dtype=np.float64)
    denominator = float(value @ value)
    return (
        float(np.sum(value) ** 2 / (value.size * denominator)) if denominator else 0.0
    )


def diagnostic_mean(diagnostics: Sequence[dict[str, Any]], key: str) -> float:
    """Average a finite scalar diagnostic across folds."""
    values = []
    for item in diagnostics:
        try:
            values.append(float(item.get(key, np.nan)))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return finite_mean(values)


__all__ = [
    "diagnostic_mean",
    "ess_fraction",
    "finite_mean",
    "metric_row",
    "summarize_rows",
]
