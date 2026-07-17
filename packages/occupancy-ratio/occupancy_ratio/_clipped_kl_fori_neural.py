"""Lazy PyTorch backend for recursively clipped KL-FORI."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from occupancy_ratio import _clipped_kl_fori_backend_common as common
from occupancy_ratio import _clipped_kl_fori_diagnostics as diagnostics
from occupancy_ratio import _clipped_kl_fori_objectives as objectives
from occupancy_ratio._clipped_kl_fori_types import (
    FitResult,
    InnerOptimizerResult,
    IterationRecord,
)


Array = np.ndarray


def import_torch() -> Any:
    """Import PyTorch only when the neural backend is requested."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "backend='neural' requires the optional torch dependency."
        ) from exc
    return torch


def enable_torch_determinism(torch: Any, cfg: Any) -> tuple[bool, str]:
    """Enable deterministic algorithms and report whether the request held."""
    if not cfg.neural_deterministic:
        return False, "disabled_by_config"
    try:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        return True, ""
    except Exception as exc:  # pragma: no cover - backend/version dependent
        return False, f"{type(exc).__name__}: {exc}"


def build_torch_mlp(torch: Any, input_dim: int, hidden_dims: Sequence[int]) -> Any:
    """Build the shared deterministic-initialization MLP architecture."""
    layers: list[Any] = []
    width_in = int(input_dim)
    for width in tuple(int(value) for value in hidden_dims):
        layer = torch.nn.Linear(width_in, width)
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
        layers.extend([layer, torch.nn.ReLU()])
        width_in = width
    output = torch.nn.Linear(width_in, 1)
    torch.nn.init.zeros_(output.weight)
    torch.nn.init.zeros_(output.bias)
    layers.append(output)
    return torch.nn.Sequential(*layers)


def predict_neural_scores(
    z: Array,
    *,
    state_dict: dict[str, Array],
    hidden_dims: Sequence[int],
    device: str,
) -> Array:
    """Restore a neural backend state dictionary and predict raw scores."""
    if not state_dict:
        raise ValueError("neural clipped KL-FORI model has no stored state_dict")
    torch = import_torch()
    target_device = torch.device(str(device))
    model = build_torch_mlp(
        torch, np.asarray(z).shape[1], tuple(int(width) for width in hidden_dims)
    ).to(target_device)
    tensor_state = {
        name: torch.as_tensor(value, dtype=torch.float32, device=target_device)
        for name, value in state_dict.items()
    }
    model.load_state_dict(tensor_state)
    model.eval()
    with torch.no_grad():
        inputs = torch.as_tensor(
            np.asarray(z, dtype=np.float32),
            dtype=torch.float32,
            device=target_device,
        )
        return (
            model(inputs)
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )


def fit_neural_backend(
    *,
    cfg: Any,
    gamma: float,
    X_ref: Array,
    X_init: Array,
    X_plus: Array,
    mean: Array,
    scale: Array,
    init_probs: Array,
    successor_row_index: Array,
    continuation_plus: Array,
    objective_ref_idx: Array,
    objective_init_idx: Array,
    valid_ref_idx: Array,
    valid_init_idx: Array,
) -> FitResult:
    """Run deployable outer iteration for the lazy neural backend."""
    torch = import_torch()
    torch.manual_seed(int(cfg.seed))
    deterministic_enabled, deterministic_error = enable_torch_determinism(torch, cfg)
    device = torch.device(str(cfg.device))
    z_ref = (X_ref - mean.reshape(1, -1)) / scale.reshape(1, -1)
    z_init = (X_init - mean.reshape(1, -1)) / scale.reshape(1, -1)
    z_plus = (X_plus - mean.reshape(1, -1)) / scale.reshape(1, -1)
    t_ref = torch.as_tensor(z_ref, dtype=torch.float32, device=device)
    t_init = torch.as_tensor(z_init, dtype=torch.float32, device=device)
    t_plus = torch.as_tensor(z_plus, dtype=torch.float32, device=device)
    gate_model = build_torch_mlp(torch, z_ref.shape[1], cfg.neural_hidden_dims).to(
        device
    )
    ratio_model = build_torch_mlp(torch, z_ref.shape[1], cfg.neural_hidden_dims).to(
        device
    )
    with torch.no_grad():
        _final_linear(torch, ratio_model).bias.fill_(
            objectives.uniform_raw_score(cfg.tau_lower, cfg.tau_upper)
        )
        if cfg.initialization_perturbation_scale > 0.0:
            for parameter in list(gate_model.parameters()) + list(
                ratio_model.parameters()
            ):
                parameter.add_(
                    float(cfg.initialization_perturbation_scale)
                    * torch.randn_like(parameter)
                )

    train_plus_idx = common.plus_indices_for_sources(
        successor_row_index, objective_ref_idx
    )
    valid_plus_idx = common.plus_indices_for_sources(successor_row_index, valid_ref_idx)
    train_init_probs = common.probability_subset(init_probs, objective_init_idx)
    valid_init_probs = (
        common.probability_subset(init_probs, valid_init_idx)
        if valid_init_idx.size
        else None
    )
    idx_ref = torch.as_tensor(objective_ref_idx, dtype=torch.long, device=device)
    idx_init = torch.as_tensor(objective_init_idx, dtype=torch.long, device=device)
    idx_plus = torch.as_tensor(train_plus_idx, dtype=torch.long, device=device)
    t_ref_objective = t_ref[idx_ref]
    t_init_objective = t_init[idx_init]
    t_plus_objective = t_plus[idx_plus]
    t_init_probs = torch.as_tensor(train_init_probs, dtype=torch.float32, device=device)
    t_continuation = torch.as_tensor(
        continuation_plus[train_plus_idx], dtype=torch.float32, device=device
    )

    weights_ref = np.ones(X_ref.shape[0], dtype=np.float64)
    gate_ref = np.ones(X_ref.shape[0], dtype=np.float64)
    history: list[dict[str, Any]] = []
    convergence_count = 0

    for iteration in range(int(cfg.num_iterations)):
        previous_weights = weights_ref.copy()
        previous_gate = gate_ref.copy()
        source_weight = previous_weights[successor_row_index[train_plus_idx]]
        t_source_weight = torch.as_tensor(
            source_weight, dtype=torch.float32, device=device
        )

        gate_optimizer = torch.optim.Adam(
            gate_model.parameters(),
            lr=cfg.resolved_gate_learning_rate,
            weight_decay=float(cfg.neural_weight_decay),
        )
        gate_result = _fit_gate_inner(
            torch=torch,
            model=gate_model,
            optimizer=gate_optimizer,
            t_ref=t_ref_objective,
            t_init=t_init_objective,
            t_plus=t_plus_objective,
            init_probs=t_init_probs,
            source_weight=t_source_weight,
            continuation=t_continuation,
            gamma=gamma,
            cfg=cfg,
        )

        with torch.no_grad():
            gate_score_ref = (
                gate_model(t_ref).reshape(-1).cpu().numpy().astype(np.float64)
            )
            gate_score_init = (
                gate_model(t_init).reshape(-1).cpu().numpy().astype(np.float64)
            )
            gate_score_plus = (
                gate_model(t_plus).reshape(-1).cpu().numpy().astype(np.float64)
            )
        gate_ref = (gate_score_ref >= 0.0).astype(np.float64)
        gate_init = (gate_score_init >= 0.0).astype(np.float64)
        gate_plus = (gate_score_plus >= 0.0).astype(np.float64)

        # These tensors are copied once and remain frozen for the entire ratio fit.
        t_gate_ref = torch.as_tensor(
            gate_ref[objective_ref_idx], dtype=torch.float32, device=device
        )
        t_gate_init = torch.as_tensor(
            gate_init[objective_init_idx], dtype=torch.float32, device=device
        )
        t_gate_plus = torch.as_tensor(
            gate_plus[train_plus_idx], dtype=torch.float32, device=device
        )
        ratio_optimizer = torch.optim.Adam(
            ratio_model.parameters(),
            lr=cfg.resolved_ratio_learning_rate,
            weight_decay=float(cfg.neural_weight_decay),
        )
        ratio_result = _fit_ratio_inner(
            torch=torch,
            model=ratio_model,
            optimizer=ratio_optimizer,
            t_ref=t_ref_objective,
            t_init=t_init_objective,
            t_plus=t_plus_objective,
            init_probs=t_init_probs,
            source_weight=t_source_weight,
            continuation=t_continuation,
            gate_ref=t_gate_ref,
            gate_init=t_gate_init,
            gate_plus=t_gate_plus,
            gamma=gamma,
            cfg=cfg,
        )

        with torch.no_grad():
            raw_ref = ratio_model(t_ref).reshape(-1).cpu().numpy().astype(np.float64)
            raw_init = ratio_model(t_init).reshape(-1).cpu().numpy().astype(np.float64)
            raw_plus = ratio_model(t_plus).reshape(-1).cpu().numpy().astype(np.float64)
        h_ref = objectives.bounded_log_ratio(
            raw_ref, tau_lower=cfg.tau_lower, tau_upper=cfg.tau_upper
        )[0]
        h_init = objectives.bounded_log_ratio(
            raw_init, tau_lower=cfg.tau_lower, tau_upper=cfg.tau_upper
        )[0]
        h_plus = objectives.bounded_log_ratio(
            raw_plus, tau_lower=cfg.tau_lower, tau_upper=cfg.tau_upper
        )[0]
        weights_ref = np.exp(h_ref)
        common.require_finite(weights_ref, "clipped neural ratio iterate")
        ratio_residual = common.relative_weight_change(
            weights_ref, previous_weights, cfg.normalize_eps
        )
        gate_change = float(np.mean(gate_ref != previous_gate))
        residual = max(ratio_residual, gate_change)
        convergence_count = common.updated_convergence_count(
            cfg, iteration, residual, convergence_count
        )
        train_gate_loss = objectives.gate_loss_from_scores(
            scores_ref=gate_score_ref[objective_ref_idx],
            scores_init=gate_score_init[objective_init_idx],
            scores_plus=gate_score_plus[train_plus_idx],
            init_probs=train_init_probs,
            source_weights=source_weight,
            continuation=continuation_plus[train_plus_idx],
            gamma=gamma,
            tau_upper=cfg.tau_upper,
        )
        train_projection_loss = objectives.projection_loss_from_log_ratios(
            log_ratio_ref=h_ref[objective_ref_idx],
            log_ratio_init=h_init[objective_init_idx],
            log_ratio_plus=h_plus[train_plus_idx],
            init_probs=train_init_probs,
            source_weights=source_weight,
            continuation=continuation_plus[train_plus_idx],
            gate_ref=gate_ref[objective_ref_idx],
            gate_init=gate_init[objective_init_idx],
            gate_plus=gate_plus[train_plus_idx],
            gamma=gamma,
            tau_upper=cfg.tau_upper,
        )
        valid = common.validation_losses(
            valid_ref_idx=valid_ref_idx,
            valid_init_idx=valid_init_idx,
            valid_plus_idx=valid_plus_idx,
            init_probs=valid_init_probs,
            successor_row_index=successor_row_index,
            continuation_plus=continuation_plus,
            previous_weights=previous_weights,
            gate_score_ref=gate_score_ref,
            gate_score_init=gate_score_init,
            gate_score_plus=gate_score_plus,
            gate_ref=gate_ref,
            gate_init=gate_init,
            gate_plus=gate_plus,
            h_ref=h_ref,
            h_init=h_init,
            h_plus=h_plus,
            gamma=gamma,
            tau_upper=cfg.tau_upper,
        )
        history.append(
            IterationRecord(
                diagnostics.history_row(
                    iteration=iteration,
                    weights_ref=weights_ref,
                    previous_weights=previous_weights,
                    gate_ref=gate_ref,
                    gate_init=gate_init,
                    gate_plus=gate_plus,
                    init_probs=init_probs,
                    successor_row_index=successor_row_index,
                    continuation_plus=continuation_plus,
                    gamma=gamma,
                    train_gate_loss=train_gate_loss,
                    train_projection_loss=train_projection_loss,
                    valid=valid,
                    gate_result=gate_result,
                    ratio_result=ratio_result,
                    ratio_residual=ratio_residual,
                    gate_change=gate_change,
                    residual=residual,
                    cfg=cfg,
                )
            ).to_dict()
        )
        if convergence_count >= int(cfg.outer_patience):
            break

    ratio_state = {
        name: tensor.detach().cpu().numpy().astype(np.float32, copy=True)
        for name, tensor in ratio_model.state_dict().items()
    }
    gate_state = {
        name: tensor.detach().cpu().numpy().astype(np.float32, copy=True)
        for name, tensor in gate_model.state_dict().items()
    }
    return FitResult(
        ratio_coef=np.empty(0, dtype=np.float64),
        gate_coef=np.empty(0, dtype=np.float64),
        ratio_neural_state_dict=ratio_state,
        gate_neural_state_dict=gate_state,
        history=history,
        weights_ref=weights_ref,
        gate_ref=gate_ref,
        iterations_completed=len(history),
        neural_deterministic_enabled=deterministic_enabled,
        neural_deterministic_error=deterministic_error,
    )


def _fit_gate_inner(**kwargs: Any) -> InnerOptimizerResult:
    torch = kwargs["torch"]
    model = kwargs["model"]
    optimizer = kwargs["optimizer"]
    cfg = kwargs["cfg"]
    previous_objective: Optional[float] = None
    stable_count = 0
    reason = "max_steps"
    objective = float("nan")
    gradient_norm = float("nan")
    for step in range(1, int(cfg.gate_optimizer_steps) + 1):
        optimizer.zero_grad()
        score_ref = model(kwargs["t_ref"]).reshape(-1)
        score_init = model(kwargs["t_init"]).reshape(-1)
        score_plus = model(kwargs["t_plus"]).reshape(-1)
        loss = (
            float(cfg.tau_upper) * torch.nn.functional.softplus(-score_ref).mean()
            + (1.0 - kwargs["gamma"])
            * torch.sum(kwargs["init_probs"] * torch.nn.functional.softplus(score_init))
            + kwargs["gamma"]
            * torch.mean(
                kwargs["source_weight"]
                * kwargs["continuation"]
                * torch.nn.functional.softplus(score_plus)
            )
            + float(cfg.gate_l2_penalty) * _torch_squared_weight_penalty(torch, model)
        )
        objective = float(loss.detach().cpu().item())
        loss.backward()
        gradient_norm = _clip_or_measure_grad(torch, model, cfg.neural_grad_clip_norm)
        common.require_finite_optimizer_state(
            objective,
            np.asarray([gradient_norm]),
            np.asarray([0.0]),
            "neural gate",
        )
        if gradient_norm <= float(cfg.inner_gradient_tolerance):
            reason = "gradient_tolerance"
            break
        stable_count = common.updated_inner_stability_count(
            objective,
            previous_objective,
            relative_tolerance=cfg.inner_relative_tolerance,
            stable_count=stable_count,
        )
        if stable_count >= int(cfg.inner_patience):
            reason = "objective_stability"
            break
        optimizer.step()
        _require_finite_torch_parameters(torch, model, "neural gate")
        previous_objective = objective
    return InnerOptimizerResult(None, objective, gradient_norm, step, reason)


def _fit_ratio_inner(**kwargs: Any) -> InnerOptimizerResult:
    torch = kwargs["torch"]
    model = kwargs["model"]
    optimizer = kwargs["optimizer"]
    cfg = kwargs["cfg"]
    previous_objective: Optional[float] = None
    stable_count = 0
    reason = "max_steps"
    objective = float("nan")
    gradient_norm = float("nan")
    for step in range(1, int(cfg.ratio_optimizer_steps) + 1):
        optimizer.zero_grad()
        h_ref = _bounded_log_ratio_torch(
            torch,
            model(kwargs["t_ref"]).reshape(-1),
            cfg.tau_lower,
            cfg.tau_upper,
        )
        h_init = _bounded_log_ratio_torch(
            torch,
            model(kwargs["t_init"]).reshape(-1),
            cfg.tau_lower,
            cfg.tau_upper,
        )
        h_plus = _bounded_log_ratio_torch(
            torch,
            model(kwargs["t_plus"]).reshape(-1),
            cfg.tau_lower,
            cfg.tau_upper,
        )
        loss = (
            torch.exp(h_ref).mean()
            - (1.0 - kwargs["gamma"])
            * torch.sum(kwargs["init_probs"] * kwargs["gate_init"] * h_init)
            - kwargs["gamma"]
            * torch.mean(
                kwargs["source_weight"]
                * kwargs["continuation"]
                * kwargs["gate_plus"]
                * h_plus
            )
            - float(cfg.tau_upper) * torch.mean((1.0 - kwargs["gate_ref"]) * h_ref)
            + float(cfg.ratio_l2_penalty) * _torch_squared_weight_penalty(torch, model)
        )
        objective = float(loss.detach().cpu().item())
        loss.backward()
        gradient_norm = _clip_or_measure_grad(torch, model, cfg.neural_grad_clip_norm)
        common.require_finite_optimizer_state(
            objective,
            np.asarray([gradient_norm]),
            np.asarray([0.0]),
            "neural ratio",
        )
        if gradient_norm <= float(cfg.inner_gradient_tolerance):
            reason = "gradient_tolerance"
            break
        stable_count = common.updated_inner_stability_count(
            objective,
            previous_objective,
            relative_tolerance=cfg.inner_relative_tolerance,
            stable_count=stable_count,
        )
        if stable_count >= int(cfg.inner_patience):
            reason = "objective_stability"
            break
        optimizer.step()
        _require_finite_torch_parameters(torch, model, "neural ratio")
        previous_objective = objective
    return InnerOptimizerResult(None, objective, gradient_norm, step, reason)


def _bounded_log_ratio_torch(
    torch: Any, raw: Any, tau_lower: float, tau_upper: float
) -> Any:
    lower = float(np.log(tau_lower))
    width = float(np.log(tau_upper) - lower)
    return lower + width * torch.sigmoid(raw)


def _final_linear(torch: Any, model: Any) -> Any:
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Linear):
            return module
    raise TypeError("neural clipped KL-FORI model has no linear output layer.")


def _torch_squared_weight_penalty(torch: Any, model: Any) -> Any:
    terms = [
        torch.sum(parameter * parameter)
        for name, parameter in model.named_parameters()
        if "weight" in name
    ]
    return 0.5 * sum(terms) if terms else torch.tensor(0.0)


def _clip_or_measure_grad(torch: Any, model: Any, cap: Optional[float]) -> float:
    if cap is not None:
        return float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cap))
            .detach()
            .cpu()
            .item()
        )
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(torch.sum(parameter.grad.detach() ** 2).cpu().item())
    return float(np.sqrt(total))


def _require_finite_torch_parameters(torch: Any, model: Any, name: str) -> None:
    for parameter in model.parameters():
        if not bool(torch.all(torch.isfinite(parameter.detach())).cpu().item()):
            raise FloatingPointError(f"{name} parameters contain nonfinite values.")


__all__ = [
    "build_torch_mlp",
    "enable_torch_determinism",
    "fit_neural_backend",
    "import_torch",
    "predict_neural_scores",
]
