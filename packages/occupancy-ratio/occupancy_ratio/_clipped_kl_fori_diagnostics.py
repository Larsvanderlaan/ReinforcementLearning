"""Iteration and final diagnostics for recursively clipped KL-FORI."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


Array = np.ndarray


def history_row(
    *,
    iteration: int,
    weights_ref: Array,
    previous_weights: Array,
    gate_ref: Array,
    gate_init: Array,
    gate_plus: Array,
    init_probs: Array,
    successor_row_index: Array,
    continuation_plus: Array,
    gamma: float,
    train_gate_loss: float,
    train_projection_loss: float,
    valid: dict[str, float],
    gate_result: Any,
    ratio_result: Any,
    ratio_residual: float,
    gate_change: float,
    residual: float,
    cfg: Any,
) -> dict[str, Any]:
    """Create one JSON-compatible deployable-iterate record."""
    target_total = (1.0 - gamma) + gamma * float(
        np.mean(previous_weights[successor_row_index] * continuation_plus)
    )
    rejected = (1.0 - gamma) * float(np.sum(init_probs * (1.0 - gate_init)))
    rejected += gamma * float(
        np.mean(
            previous_weights[successor_row_index]
            * continuation_plus
            * (1.0 - gate_plus)
        )
    )
    return {
        "iteration": int(iteration + 1),
        "gate_objective": float(train_gate_loss),
        "projection_objective": float(train_projection_loss),
        "valid_gate_objective": float(valid["valid_gate_loss"]),
        "valid_projection_objective": float(valid["valid_projection_loss"]),
        "gate_grad_norm": float(gate_result.gradient_norm),
        "ratio_grad_norm": float(ratio_result.gradient_norm),
        "gate_optimizer_steps": int(gate_result.steps_completed),
        "ratio_optimizer_steps": int(ratio_result.steps_completed),
        "gate_termination_reason": gate_result.termination_reason,
        "ratio_termination_reason": ratio_result.termination_reason,
        "gate_optimizer_objective": float(gate_result.objective),
        "ratio_optimizer_objective": float(ratio_result.objective),
        "outer_ratio_relative_change": float(ratio_residual),
        "outer_gate_change_fraction": float(gate_change),
        "outer_relative_change": float(residual),
        "empirical_mass": float(np.mean(weights_ref)),
        "ess_fraction": ess_fraction(weights_ref),
        "weight_min": float(np.min(weights_ref)),
        "weight_max": float(np.max(weights_ref)),
        "lower_envelope_fraction": float(
            np.mean(weights_ref <= cfg.tau_lower * (1.0 + 1e-4))
        ),
        "upper_envelope_fraction": float(
            np.mean(weights_ref >= cfg.tau_upper * (1.0 - 1e-4))
        ),
        "gate_rejected_reference_fraction": float(np.mean(1.0 - gate_ref)),
        "rejected_bellman_mass": float(rejected),
        "rejected_bellman_mass_fraction": float(
            rejected / max(target_total, cfg.normalize_eps)
        ),
        "nonfinite": False,
    }


def final_diagnostics(
    *,
    cfg: Any,
    gamma: float,
    fit: dict[str, Any],
    continuation_plus: Array,
    selection: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize selection, refit, convergence, envelope, and failure telemetry."""
    final = fit["history"][-1] if fit["history"] else {}
    refit_rows = [row for row in fit["history"] if row.get("stage") == "refit"]
    if refit_rows:
        final = refit_rows[-1]
    residual = float(final.get("outer_relative_change", float("nan")))
    objective_rows = refit_rows if refit_rows else fit["history"]
    objective_stability = _objective_stability(objective_rows)
    converged = _deployable_converged(objective_rows, cfg)
    selection_final = (
        selection["history"][-1]
        if selection is not None and selection.get("history")
        else {}
    )
    selection_converged = bool(
        selection is not None
        and _deployable_converged(selection.get("history", []), cfg)
    )
    out = {
        "algorithm": "clipped_kl_fori",
        "backend": cfg.backend,
        "tau_lower": float(cfg.tau_lower),
        "tau_upper": float(cfg.tau_upper),
        "iterations_completed": int(fit["iterations_completed"]),
        "converged": converged,
        "selection_converged": selection_converged,
        "refit_completed": bool(selection is not None),
        "outer_termination_reason": (
            "deployable_tolerance" if converged else "configured_iterations"
        ),
        "outer_relative_change_final": residual,
        "outer_ratio_relative_change_final": float(
            final.get("outer_ratio_relative_change", float("nan"))
        ),
        "outer_gate_change_fraction_final": float(
            final.get("outer_gate_change_fraction", float("nan"))
        ),
        "fixed_point_residual_bound": (
            residual / max(1.0 - gamma, cfg.normalize_eps)
            if np.isfinite(residual)
            else float("nan")
        ),
        "objective_stability_final": objective_stability,
        "empirical_mass": float(
            final.get("empirical_mass", np.mean(fit["weights_ref"]))
        ),
        "ess_fraction": float(
            final.get("ess_fraction", ess_fraction(fit["weights_ref"]))
        ),
        "weight_min": float(np.min(fit["weights_ref"])),
        "weight_max": float(np.max(fit["weights_ref"])),
        "gate_objective_final": float(final.get("gate_objective", float("nan"))),
        "projection_objective_final": float(
            final.get("projection_objective", float("nan"))
        ),
        "valid_gate_objective_final": float(
            final.get("valid_gate_objective", float("nan"))
        ),
        "valid_projection_objective_final": float(
            final.get("valid_projection_objective", float("nan"))
        ),
        "gate_grad_norm_final": float(final.get("gate_grad_norm", float("nan"))),
        "ratio_grad_norm_final": float(final.get("ratio_grad_norm", float("nan"))),
        "gate_termination_reason_final": str(
            final.get("gate_termination_reason", "not_run")
        ),
        "ratio_termination_reason_final": str(
            final.get("ratio_termination_reason", "not_run")
        ),
        "gate_optimizer_steps_final": int(final.get("gate_optimizer_steps", 0)),
        "ratio_optimizer_steps_final": int(final.get("ratio_optimizer_steps", 0)),
        "lower_envelope_fraction": float(
            final.get("lower_envelope_fraction", float("nan"))
        ),
        "upper_envelope_fraction": float(
            final.get("upper_envelope_fraction", float("nan"))
        ),
        "gate_rejected_reference_fraction": float(np.mean(1.0 - fit["gate_ref"])),
        "rejected_bellman_mass": float(
            final.get("rejected_bellman_mass", float("nan"))
        ),
        "rejected_bellman_mass_fraction": float(
            final.get("rejected_bellman_mass_fraction", float("nan"))
        ),
        "continuation_mean": float(np.mean(continuation_plus)),
        "nonfinite": False,
        "neural_deterministic_requested": bool(cfg.neural_deterministic),
        "neural_deterministic_enabled": bool(fit["neural_deterministic_enabled"]),
        "neural_deterministic_error": str(fit["neural_deterministic_error"]),
        "selector_uses_oracle_truth": 0.0,
        "selector_uses_target_value": 0.0,
        "selector_uses_final_ope_error": 0.0,
    }
    if selection is not None:
        out.update(
            {
                "selection_iterations_completed": int(
                    selection["iterations_completed"]
                ),
                "selection_valid_gate_objective": float(
                    selection_final.get("valid_gate_objective", float("nan"))
                ),
                "selection_valid_projection_objective": float(
                    selection_final.get("valid_projection_objective", float("nan"))
                ),
            }
        )
    return out


def ess_fraction(weight: Array) -> float:
    """Return effective sample size divided by row count."""
    value = np.asarray(weight, dtype=np.float64).reshape(-1)
    denominator = float(value.size * np.sum(value * value))
    return float(np.sum(value) ** 2 / denominator) if denominator > 0.0 else 0.0


def _objective_stability(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return float("nan")
    return float(
        max(
            abs(float(rows[-1]["gate_objective"]) - float(rows[-2]["gate_objective"])),
            abs(
                float(rows[-1]["projection_objective"])
                - float(rows[-2]["projection_objective"])
            ),
        )
    )


def _deployable_converged(rows: list[dict[str, Any]], cfg: Any) -> bool:
    """Require the configured number of consecutive deployable residuals."""
    if cfg.outer_tolerance is None or not rows:
        return False
    patience = int(cfg.outer_patience)
    if len(rows) < patience:
        return False
    final_iteration = int(rows[-1].get("iteration", len(rows)))
    if final_iteration < int(cfg.min_iterations):
        return False
    residuals = np.asarray(
        [row.get("outer_relative_change", np.nan) for row in rows[-patience:]],
        dtype=np.float64,
    )
    return bool(
        np.all(np.isfinite(residuals))
        and np.all(residuals <= float(cfg.outer_tolerance))
    )


__all__ = ["ess_fraction", "final_diagnostics", "history_row"]
