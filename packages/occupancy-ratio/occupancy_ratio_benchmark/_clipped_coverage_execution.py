"""Cross-fitted fold execution for the clipped coverage benchmark."""

from __future__ import annotations

from dataclasses import replace
import json
import time
from typing import Any, Sequence

import numpy as np

from occupancy_ratio import _clipped_kl_fori_objectives as clipped_objectives
from occupancy_ratio._fori_data import resolve_continuation
from occupancy_ratio.clipped_kl_fori import ClippedKLFORIConfig, fit_clipped_kl_fori
from occupancy_ratio.kl_fori import KLFORIConfig, fit_kl_fori
from occupancy_ratio_benchmark._clipped_coverage_data import (
    CoverageRunConfig,
    SharedHubDataset,
)
from occupancy_ratio_benchmark._clipped_coverage_metrics import finite_mean, metric_row
from occupancy_ratio_benchmark._clipped_coverage_oracle import shared_hub_box_oracle


Array = np.ndarray


def run_backend_fold(
    dataset: SharedHubDataset,
    config: CoverageRunConfig,
    *,
    backend: str,
    repetition: int,
    seed: int,
    heldout: int,
) -> dict[str, Any]:
    """Fit one held-out fold and return a JSON-compatible fit payload."""
    if heldout not in {0, 1}:
        raise ValueError("heldout must be 0 or 1")
    train, test, init_train = _fold_indices(dataset, seed=seed, heldout=heldout)
    fit_kwargs = {
        "states": dataset.states[train],
        "actions": dataset.actions[train],
        "next_states": dataset.next_states[train],
        "target_next_actions": dataset.target_next_actions[train],
        "initial_states": dataset.initial_states[init_train],
        "initial_actions": dataset.initial_actions[init_train],
        "terminals": (None if dataset.terminals is None else dataset.terminals[train]),
        "timeouts": None if dataset.timeouts is None else dataset.timeouts[train],
        "handle_timeouts": dataset.handle_timeouts,
        "absorbing_state": dataset.absorbing_state,
        "gamma": config.gamma,
    }
    run_clipped = "clipped_fori" in config.methods
    run_standard = "standard_fori" in config.methods
    clipped_cfg = ClippedKLFORIConfig(
        tau_lower=config.tau_lower,
        tau_upper=config.tau_upper,
        backend=backend,
        num_iterations=config.clipped_num_iterations,
        min_iterations=min(5, config.clipped_num_iterations),
        gate_optimizer_steps=config.clipped_gate_steps,
        ratio_optimizer_steps=config.clipped_ratio_steps,
        gate_learning_rate=config.clipped_gate_learning_rate,
        ratio_learning_rate=config.clipped_ratio_learning_rate,
        inner_relative_tolerance=config.clipped_inner_relative_tolerance,
        inner_gradient_tolerance=config.clipped_inner_gradient_tolerance,
        inner_patience=config.clipped_inner_patience,
        validation_fraction=config.validation_fraction,
        seed=seed + heldout,
    )
    clipped: dict[str, Any] = {
        "status": "skipped",
        "error": "",
        "prediction": None,
        "gate_prediction": None,
        "training_mass": None,
        "diagnostics": None,
        "runtime_sec": 0.0,
        "optimizer_restart_objective_gap": None,
        "optimizer_restart_self_objective_gap": None,
        "optimizer_restart_ratio_l1": None,
        "optimizer_restart_gate_disagreement": None,
        "optimizer_restart_mass_gap": None,
        "optimizer_restart_all_converged": None,
        "termination_reason": None,
    }
    if run_clipped:
        start = time.perf_counter()
        try:
            model = fit_clipped_kl_fori(**fit_kwargs, config=clipped_cfg)
            clipped.update(
                {
                    "prediction": model.predict_state_action_ratio(
                        dataset.states[test], dataset.actions[test]
                    ).tolist(),
                    "gate_prediction": model.predict_gate_indicator(
                        dataset.states[test], dataset.actions[test]
                    ).tolist(),
                    "training_mass": float(model.diagnostics["empirical_mass"]),
                    "diagnostics": model.diagnostics,
                    "termination_reason": model.diagnostics.get(
                        "outer_termination_reason"
                    ),
                }
            )
            models = [model]
            self_objectives = [
                (
                    float(model.diagnostics["gate_objective_final"]),
                    float(model.diagnostics["projection_objective_final"]),
                )
            ]
            for restart in range(1, int(config.optimizer_stability_restarts)):
                audit_model = fit_clipped_kl_fori(
                    **fit_kwargs,
                    config=replace(
                        clipped_cfg,
                        seed=seed + heldout + 10_007 * restart,
                        initialization_perturbation_scale=float(
                            config.optimizer_perturbation_scale
                        ),
                    ),
                )
                models.append(audit_model)
                self_objectives.append(
                    (
                        float(audit_model.diagnostics["gate_objective_final"]),
                        float(audit_model.diagnostics["projection_objective_final"]),
                    )
                )
            common_objectives = _common_restart_objectives(
                models, fit_kwargs, clipped_cfg
            )
            stability = _deployable_restart_stability(
                models,
                states=dataset.states[test],
                actions=dataset.actions[test],
            )
            clipped.update(stability)
            clipped["optimizer_restart_objective_gap"] = _objective_gap(
                common_objectives
            )
            clipped["optimizer_restart_self_objective_gap"] = _objective_gap(
                self_objectives
            )
            clipped["optimizer_restart_all_converged"] = all(
                bool(item.diagnostics.get("converged", False)) for item in models
            )
            clipped["status"] = (
                "ok"
                if clipped["optimizer_restart_all_converged"]
                else "nonconverged"
            )
            if clipped["status"] == "nonconverged":
                clipped["termination_reason"] = "restart_nonconvergence"
        except Exception as exc:  # benchmark failures remain structured rows
            clipped["error"] = f"{type(exc).__name__}: {exc}"
            clipped["status"] = "error"
        clipped["runtime_sec"] = float(time.perf_counter() - start)

    standard: dict[str, Any] = {
        "status": "out_of_regime",
        "error": "full occupancy ratio is not identified at q=0",
        "prediction": None,
        "training_mass": None,
        "diagnostics": None,
        "runtime_sec": 0.0,
    }
    if run_standard and dataset.truth.retained_mass > 0.0:
        standard_cfg = KLFORIConfig(
            backend=backend,
            num_iterations=config.standard_num_iterations,
            optimizer_steps=config.standard_optimizer_steps,
            validation_fraction=config.validation_fraction,
            early_stopping=False,
            seed=seed + heldout,
        )
        start = time.perf_counter()
        try:
            model = fit_kl_fori(**fit_kwargs, config=standard_cfg)
            convergence = _standard_convergence_diagnostics(model, config)
            standard.update(
                {
                    "status": (
                        "ok"
                        if convergence["converged"]
                        or not config.standard_require_convergence
                        else "nonconverged"
                    ),
                    "error": "",
                    "prediction": model.predict_state_action_ratio(
                        dataset.states[test], dataset.actions[test]
                    ).tolist(),
                    "training_mass": float(
                        model.diagnostics.get("empirical_mass", 1.0)
                    ),
                    "diagnostics": dict(model.diagnostics, **convergence),
                }
            )
        except Exception as exc:
            standard.update(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        standard["runtime_sec"] = float(time.perf_counter() - start)
    elif not run_standard:
        standard.update({"status": "skipped", "error": ""})

    return {
        "fold": int(heldout),
        "fit_seed": int(seed + heldout),
        "repetition": int(repetition),
        "backend": str(backend),
        "test_index": test.tolist(),
        "clipped": clipped,
        "standard": standard,
    }


def combine_backend_folds(
    fold_payloads: Sequence[dict[str, Any]],
    dataset: SharedHubDataset,
    config: CoverageRunConfig,
    *,
    backend: str,
    repetition: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Validate and combine two fold artifacts into method-level metric rows."""
    payloads = _validated_fold_payloads(fold_payloads, dataset.n)
    clipped_prediction = np.full(dataset.n, np.nan, dtype=np.float64)
    standard_prediction = np.full(dataset.n, np.nan, dtype=np.float64)
    gate_prediction = np.full(dataset.n, np.nan, dtype=np.float64)
    clipped_train_mass: list[float] = []
    standard_train_mass: list[float] = []
    clipped_diagnostics: list[dict[str, Any]] = []
    standard_diagnostics: list[dict[str, Any]] = []
    clipped_runtime = 0.0
    standard_runtime = 0.0
    restart_gaps: list[float] = []
    restart_self_gaps: list[float] = []
    restart_ratio_l1: list[float] = []
    restart_gate_disagreement: list[float] = []
    restart_mass_gaps: list[float] = []
    fold_statuses: list[dict[str, Any]] = []

    for payload in payloads:
        test = np.asarray(payload["test_index"], dtype=np.int64)
        clipped = payload["clipped"]
        standard = payload["standard"]
        if "clipped_fori" in config.methods:
            fold_statuses.append(
                {
                    "fold": int(payload["fold"]),
                    "method": "clipped_fori",
                    "status": clipped["status"],
                    "termination_reason": clipped.get("termination_reason"),
                    "error": clipped.get("error", ""),
                }
            )
        if "standard_fori" in config.methods:
            fold_statuses.append(
                {
                    "fold": int(payload["fold"]),
                    "method": "standard_fori",
                    "status": standard["status"],
                    "error": standard.get("error", ""),
                }
            )
        if clipped.get("prediction") is not None:
            clipped_prediction[test] = np.asarray(
                clipped["prediction"], dtype=np.float64
            )
        if clipped.get("gate_prediction") is not None:
            gate_prediction[test] = np.asarray(
                clipped["gate_prediction"], dtype=np.float64
            )
        if clipped.get("training_mass") is not None:
            clipped_train_mass.append(float(clipped["training_mass"]))
        if isinstance(clipped.get("diagnostics"), dict):
            clipped_diagnostics.append(clipped["diagnostics"])
        if clipped.get("optimizer_restart_objective_gap") is not None:
            restart_gaps.append(float(clipped["optimizer_restart_objective_gap"]))
        if clipped.get("optimizer_restart_self_objective_gap") is not None:
            restart_self_gaps.append(
                float(clipped["optimizer_restart_self_objective_gap"])
            )
        if clipped.get("optimizer_restart_ratio_l1") is not None:
            restart_ratio_l1.append(float(clipped["optimizer_restart_ratio_l1"]))
        if clipped.get("optimizer_restart_gate_disagreement") is not None:
            restart_gate_disagreement.append(
                float(clipped["optimizer_restart_gate_disagreement"])
            )
        if clipped.get("optimizer_restart_mass_gap") is not None:
            restart_mass_gaps.append(float(clipped["optimizer_restart_mass_gap"]))
        clipped_runtime += float(clipped.get("runtime_sec", 0.0))

        if standard.get("prediction") is not None:
            standard_prediction[test] = np.asarray(
                standard["prediction"], dtype=np.float64
            )
        if standard.get("training_mass") is not None:
            standard_train_mass.append(float(standard["training_mass"]))
        if isinstance(standard.get("diagnostics"), dict):
            standard_diagnostics.append(standard["diagnostics"])
        standard_runtime += float(standard.get("runtime_sec", 0.0))

    clipped_statuses = [payload["clipped"]["status"] for payload in payloads]
    standard_statuses = [payload["standard"]["status"] for payload in payloads]
    clipped_error = _first_error(payloads, "clipped")
    standard_error = _first_error(payloads, "standard")
    sampling = _sampling_diagnostics(dataset, config)
    common = {
        "repetition": int(repetition),
        "seed": int(seed),
        "fold_seeds": json.dumps([int(payload["fit_seed"]) for payload in payloads]),
        "n": int(config.n),
        "contexts": int(config.contexts),
        "backend": backend,
        "gamma": float(config.gamma),
        "tau_lower": float(config.tau_lower),
        "tau_upper": float(config.tau_upper),
        "oracle_mass": float(dataset.truth.retained_mass),
        "projected_oracle_mass": float(dataset.truth.projected_mass(config.tau_lower)),
        "floor_approximation_bias": float(
            dataset.truth.projected_mass(config.tau_lower) - dataset.truth.retained_mass
        ),
        "posthoc_oracle_mass": float(dataset.truth.posthoc_mass),
        "q_min": float(np.min(dataset.truth.q_by_context)),
        "q_max": float(np.max(dataset.truth.q_by_context)),
        "sampling_mode": config.sampling_mode,
        "irrelevant_features": int(config.irrelevant_features),
        "termination_encoding": config.termination_encoding,
        **sampling,
        "clipped_gate_steps": int(config.clipped_gate_steps),
        "clipped_ratio_steps": int(config.clipped_ratio_steps),
        "clipped_gate_learning_rate": float(
            0.05
            if config.clipped_gate_learning_rate is None and backend == "linear"
            else 1e-3
            if config.clipped_gate_learning_rate is None
            else config.clipped_gate_learning_rate
        ),
        "clipped_ratio_learning_rate": float(
            0.05
            if config.clipped_ratio_learning_rate is None and backend == "linear"
            else 1e-3
            if config.clipped_ratio_learning_rate is None
            else config.clipped_ratio_learning_rate
        ),
        "clipped_inner_relative_tolerance": float(
            config.clipped_inner_relative_tolerance
        ),
        "clipped_inner_gradient_tolerance": float(
            config.clipped_inner_gradient_tolerance
        ),
        "clipped_inner_patience": int(config.clipped_inner_patience),
        "standard_num_iterations": int(config.standard_num_iterations),
        "standard_optimizer_steps": int(config.standard_optimizer_steps),
        "standard_outer_tolerance": float(config.standard_outer_tolerance),
        "standard_objective_tolerance": float(
            config.standard_objective_tolerance
        ),
        "standard_mass_tolerance": float(config.standard_mass_tolerance),
        "fold_statuses": json.dumps(fold_statuses, sort_keys=True),
        "optimizer_restart_objective_gap": (
            float(np.max(restart_gaps)) if restart_gaps else float("nan")
        ),
        "optimizer_restart_self_objective_gap": (
            float(np.max(restart_self_gaps)) if restart_self_gaps else float("nan")
        ),
        "optimizer_restart_ratio_l1": (
            float(np.max(restart_ratio_l1)) if restart_ratio_l1 else float("nan")
        ),
        "optimizer_restart_gate_disagreement": (
            float(np.max(restart_gate_disagreement))
            if restart_gate_disagreement
            else float("nan")
        ),
        "optimizer_restart_mass_gap": (
            float(np.max(restart_mass_gaps)) if restart_mass_gaps else float("nan")
        ),
    }
    rows: list[dict[str, Any]] = []
    if "clipped_fori" in config.methods:
        if "error" in clipped_statuses or not np.all(
            np.isfinite(clipped_prediction)
        ):
            rows.append(
                common
                | {
                    "method": "clipped_fori",
                    "status": "error",
                    "failure_type": "fit_exception",
                    "error": clipped_error,
                }
            )
        else:
            clipped_row = metric_row(
                common=common,
                method="clipped_fori",
                prediction=clipped_prediction,
                own_ratio=dataset.recursive_ratio,
                recursive_ratio=dataset.recursive_ratio,
                dataset=dataset,
                own_target="clipped",
                train_mass=finite_mean(clipped_train_mass),
                runtime_sec=clipped_runtime,
                gate_prediction=gate_prediction,
                diagnostics=clipped_diagnostics,
            )
            stability_failures = []
            if float(common["optimizer_restart_objective_gap"]) > 1e-4:
                stability_failures.append("common_objective_gap")
            if float(common["optimizer_restart_ratio_l1"]) > 1e-3:
                stability_failures.append("ratio_l1")
            if float(common["optimizer_restart_gate_disagreement"]) > 1e-3:
                stability_failures.append("gate_disagreement")
            clipped_row["optimizer_stability_failures"] = json.dumps(
                stability_failures
            )
            if "nonconverged" in clipped_statuses:
                clipped_row["status"] = "nonconverged"
                clipped_row["failure_type"] = "outer_convergence"
            elif stability_failures:
                clipped_row["status"] = "optimizer_unstable"
                clipped_row["failure_type"] = "optimizer_restart_stability"
            rows.append(clipped_row)

    if dataset.truth.retained_mass <= 0.0:
        for method in config.methods:
            if method == "clipped_fori":
                continue
            rows.append(
                common
                | {
                    "method": method,
                    "status": "out_of_regime",
                    "failure_type": "identification",
                    "error": "full occupancy ratio is not identified at q=0",
                }
            )
        return rows
    if "standard_fori" not in config.methods:
        return rows
    if "error" in standard_statuses or not np.all(np.isfinite(standard_prediction)):
        for method in config.methods:
            if method == "clipped_fori":
                continue
            rows.append(
                common
                | {
                    "method": method,
                    "status": "error",
                    "failure_type": "fit_exception",
                    "error": standard_error,
                }
            )
        return rows
    standard_row = metric_row(
        common=common,
        method="standard_fori",
        prediction=standard_prediction,
        own_ratio=dataset.full_ratio,
        recursive_ratio=dataset.recursive_ratio,
        dataset=dataset,
        own_target="full",
        train_mass=finite_mean(standard_train_mass),
        runtime_sec=standard_runtime,
        gate_prediction=None,
        diagnostics=standard_diagnostics,
    )
    if "nonconverged" in standard_statuses:
        standard_row["status"] = "nonconverged"
        standard_row["failure_type"] = "standard_convergence"
    rows.append(standard_row)
    if "posthoc_winsorized" not in config.methods:
        return rows
    posthoc = np.minimum(standard_prediction, config.tau_upper)
    posthoc_row = metric_row(
        common=common,
        method="posthoc_winsorized",
        prediction=posthoc,
        own_ratio=dataset.posthoc_ratio,
        recursive_ratio=dataset.recursive_ratio,
        dataset=dataset,
        own_target="posthoc",
        train_mass=float("nan"),
        runtime_sec=standard_runtime,
        gate_prediction=None,
        diagnostics=standard_diagnostics,
    )
    if "nonconverged" in standard_statuses:
        posthoc_row["status"] = "nonconverged"
        posthoc_row["failure_type"] = "standard_convergence"
    rows.append(posthoc_row)
    return rows


def run_backend_cell(
    dataset: SharedHubDataset,
    config: CoverageRunConfig,
    *,
    backend: str,
    repetition: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run both folds without persistence; retained for direct programmatic use."""
    payloads = [
        run_backend_fold(
            dataset,
            config,
            backend=backend,
            repetition=repetition,
            seed=seed,
            heldout=heldout,
        )
        for heldout in (0, 1)
    ]
    return combine_backend_folds(
        payloads,
        dataset,
        config,
        backend=backend,
        repetition=repetition,
        seed=seed,
    )


def _fold_indices(
    dataset: SharedHubDataset, *, seed: int, heldout: int
) -> tuple[Array, Array, Array]:
    fold = np.arange(dataset.n, dtype=np.int64) % 2
    rng = np.random.default_rng(seed + 311)
    rng.shuffle(fold)
    initial_fold = np.arange(dataset.initial_states.shape[0], dtype=np.int64) % 2
    rng.shuffle(initial_fold)
    return (
        np.flatnonzero(fold != heldout),
        np.flatnonzero(fold == heldout),
        np.flatnonzero(initial_fold != heldout),
    )


def _validated_fold_payloads(
    fold_payloads: Sequence[dict[str, Any]], n_rows: int
) -> list[dict[str, Any]]:
    if len(fold_payloads) != 2:
        raise ValueError("cross-fit aggregation requires exactly two fold artifacts")
    payloads = sorted(fold_payloads, key=lambda payload: int(payload.get("fold", -1)))
    if [int(payload.get("fold", -1)) for payload in payloads] != [0, 1]:
        raise ValueError("fold artifacts must contain folds 0 and 1 exactly once")
    indices = [
        np.asarray(payload.get("test_index", []), dtype=np.int64)
        for payload in payloads
    ]
    combined = np.concatenate(indices)
    if (
        combined.size != int(n_rows)
        or np.unique(combined).size != int(n_rows)
        or np.any(combined < 0)
        or np.any(combined >= int(n_rows))
    ):
        raise ValueError("fold artifacts do not partition all reference rows")
    for payload, index in zip(payloads, indices):
        for method in ("clipped", "standard"):
            value = payload.get(method)
            if not isinstance(value, dict) or "status" not in value:
                raise ValueError(f"fold artifact is missing {method} status")
            prediction = value.get("prediction")
            if prediction is not None and len(prediction) != index.size:
                raise ValueError(f"fold artifact has malformed {method} predictions")
    return payloads


def _first_error(payloads: Sequence[dict[str, Any]], method: str) -> str:
    return next(
        (
            str(payload[method].get("error", ""))
            for payload in payloads
            if payload[method].get("status") == "error"
        ),
        "",
    )


def _common_restart_objectives(
    models: Sequence[Any], fit_kwargs: dict[str, Any], cfg: ClippedKLFORIConfig
) -> list[tuple[float, float]]:
    """Evaluate every restart on one frozen, penalized empirical objective."""
    states = np.asarray(fit_kwargs["states"], dtype=np.float64)
    actions = np.asarray(fit_kwargs["actions"], dtype=np.float64)
    next_states = np.asarray(fit_kwargs["next_states"], dtype=np.float64)
    next_actions = np.asarray(fit_kwargs["target_next_actions"], dtype=np.float64)
    initial_states = np.asarray(fit_kwargs["initial_states"], dtype=np.float64)
    initial_actions = np.asarray(fit_kwargs["initial_actions"], dtype=np.float64)
    continuation = resolve_continuation(
        n_rows=states.shape[0],
        terminals=fit_kwargs.get("terminals"),
        timeouts=fit_kwargs.get("timeouts"),
        handle_timeouts=str(fit_kwargs.get("handle_timeouts", "nonterminal")),
        absorbing_state=bool(fit_kwargs.get("absorbing_state", False)),
    )
    initial_probability = np.full(
        initial_states.shape[0], 1.0 / initial_states.shape[0], dtype=np.float64
    )
    canonical = models[0]
    source_weights = canonical.predict_state_action_ratio(states, actions)
    gate_ref = canonical.predict_gate_indicator(states, actions)
    gate_init = canonical.predict_gate_indicator(initial_states, initial_actions)
    gate_plus = canonical.predict_gate_indicator(next_states, next_actions)
    out: list[tuple[float, float]] = []
    for model in models:
        gate_loss = clipped_objectives.gate_loss_from_scores(
            scores_ref=model.predict_gate_score(states, actions),
            scores_init=model.predict_gate_score(initial_states, initial_actions),
            scores_plus=model.predict_gate_score(next_states, next_actions),
            init_probs=initial_probability,
            source_weights=source_weights,
            continuation=continuation,
            gamma=float(fit_kwargs["gamma"]),
            tau_upper=cfg.tau_upper,
        ) + _model_l2_penalty(model, "gate", cfg.gate_l2_penalty)
        ratio_loss = clipped_objectives.projection_loss_from_log_ratios(
            log_ratio_ref=model.predict_state_action_log_ratio(states, actions),
            log_ratio_init=model.predict_state_action_log_ratio(
                initial_states, initial_actions
            ),
            log_ratio_plus=model.predict_state_action_log_ratio(
                next_states, next_actions
            ),
            init_probs=initial_probability,
            source_weights=source_weights,
            continuation=continuation,
            gate_ref=gate_ref,
            gate_init=gate_init,
            gate_plus=gate_plus,
            gamma=float(fit_kwargs["gamma"]),
            tau_upper=cfg.tau_upper,
        ) + _model_l2_penalty(model, "ratio", cfg.ratio_l2_penalty)
        out.append((float(gate_loss), float(ratio_loss)))
    return out


def _model_l2_penalty(model: Any, component: str, strength: float) -> float:
    if model.backend == "linear":
        coefficient = np.asarray(
            model.gate_coef if component == "gate" else model.ratio_coef,
            dtype=np.float64,
        ).copy()
        coefficient[0] = 0.0
        squared_norm = float(coefficient @ coefficient)
    else:
        state = (
            model.gate_neural_state_dict
            if component == "gate"
            else model.ratio_neural_state_dict
        )
        squared_norm = float(
            sum(
                np.sum(np.asarray(value, dtype=np.float64) ** 2)
                for name, value in state.items()
                if "weight" in name
            )
        )
    return 0.5 * float(strength) * squared_norm


def _deployable_restart_stability(
    models: Sequence[Any], *, states: Array, actions: Array
) -> dict[str, float]:
    ratios = [model.predict_state_action_ratio(states, actions) for model in models]
    gates = [model.predict_gate_indicator(states, actions) for model in models]
    ratio_l1 = 0.0
    gate_disagreement = 0.0
    for left in range(len(models)):
        for right in range(left + 1, len(models)):
            ratio_l1 = max(
                ratio_l1, float(np.mean(np.abs(ratios[left] - ratios[right])))
            )
            gate_disagreement = max(
                gate_disagreement, float(np.mean(gates[left] != gates[right]))
            )
    masses = np.asarray([np.mean(ratio) for ratio in ratios], dtype=np.float64)
    return {
        "optimizer_restart_ratio_l1": ratio_l1,
        "optimizer_restart_gate_disagreement": gate_disagreement,
        "optimizer_restart_mass_gap": float(np.ptp(masses)) if masses.size > 1 else 0.0,
    }


def _objective_gap(objectives: Sequence[tuple[float, float]]) -> float:
    array = np.asarray(objectives, dtype=np.float64)
    return float(np.max(np.ptp(array, axis=0))) if array.shape[0] > 1 else 0.0


def _standard_convergence_diagnostics(
    model: Any, config: CoverageRunConfig
) -> dict[str, float | bool]:
    history = list(getattr(model, "history", ()))
    objectives = np.asarray(
        [row.get("objective", np.nan) for row in history], dtype=np.float64
    )
    objective_relative_change = float("inf")
    if objectives.size >= 2 and np.all(np.isfinite(objectives[-2:])):
        objective_relative_change = float(
            abs(objectives[-1] - objectives[-2])
            / max(1.0, abs(float(objectives[-2])))
        )
    weight_step_l1 = float(
        model.diagnostics.get("weight_step_l1_final", float("inf"))
    )
    mass_abs_error = float(
        model.diagnostics.get("empirical_mass_abs_error", float("inf"))
    )
    finite = bool(
        np.isfinite(objective_relative_change)
        and np.isfinite(weight_step_l1)
        and np.isfinite(mass_abs_error)
    )
    converged = bool(
        finite
        and weight_step_l1 <= config.standard_outer_tolerance
        and objective_relative_change <= config.standard_objective_tolerance
        and mass_abs_error <= config.standard_mass_tolerance
    )
    return {
        "converged": converged,
        "standard_objective_relative_change": objective_relative_change,
        "standard_weight_step_l1": weight_step_l1,
        "standard_mass_abs_error": mass_abs_error,
        "objective_stability_final": objective_relative_change,
        "fixed_point_residual_bound": weight_step_l1,
    }


def _sampling_diagnostics(
    dataset: SharedHubDataset, config: CoverageRunConfig
) -> dict[str, float | int]:
    category = np.asarray(dataset.category, dtype=np.int64)
    initial_rows = int(np.sum(category != 2))
    target_rows = int(np.sum(category == 0))
    expected_target_rows = float(
        config.n
        * (1.0 - config.gamma)
        * np.sum(dataset.truth.context_probability * dataset.truth.q_by_context)
    )
    empirical_q = (
        float(target_rows / initial_rows) if initial_rows > 0 else float("nan")
    )
    empirical_projected_mass = float("nan")
    if config.contexts == 1 and np.isfinite(empirical_q):
        empirical_projected_mass = shared_hub_box_oracle(
            q=np.asarray([empirical_q]),
            gamma=config.gamma,
            tau_lower=config.tau_lower,
            tau_upper=config.tau_upper,
        ).projected_mass
    return {
        "observed_initial_rows": initial_rows,
        "observed_target_branch_rows": target_rows,
        "expected_target_branch_rows": expected_target_rows,
        "empirical_q": empirical_q,
        "empirical_projected_oracle_mass": empirical_projected_mass,
    }


__all__ = ["combine_backend_folds", "run_backend_cell", "run_backend_fold"]
