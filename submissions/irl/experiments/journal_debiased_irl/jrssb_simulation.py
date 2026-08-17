from __future__ import annotations

import csv
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import pickle
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
IRL_NEURIPS_DIR = ROOT / "submissions" / "irl" / "experiments" / "conference_genpqr" / "repro"
if str(IRL_NEURIPS_DIR) not in sys.path:
    sys.path.insert(0, str(IRL_NEURIPS_DIR))


def _load_alias_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_utils_path = IRL_NEURIPS_DIR / "utils.py"
if "utils" not in sys.modules or Path(getattr(sys.modules["utils"], "__file__", "")).resolve() != _utils_path.resolve():
    _load_alias_module("utils", _utils_path)

from policy_estimation import fit_behavior_cloning_policy, fit_maxent_irl_policy
from q_evaluation import fit_fqe_neural

try:
    from .fore_ratio import (
        FOREFitOptions,
        deterministic_three_way_split,
        fit_selected_fore_ratio,
        fit_selected_signed_fore_ratio,
        paper_early_stopping_candidates,
        ratio_diagnostics,
    )
except ImportError:  # pragma: no cover - direct script execution
    from fore_ratio import (
        FOREFitOptions,
        deterministic_three_way_split,
        fit_selected_fore_ratio,
        fit_selected_signed_fore_ratio,
        paper_early_stopping_candidates,
        ratio_diagnostics,
    )


EPS = 1e-8
EXAMPLE1A_POLICY_ESTIMATORS = (
    "bc",
    "maxent",
    "sieve-logit",
    "coarse",
    "blend",
    "structural-linear",
)
EXAMPLE1B_POLICY_ESTIMATORS = (
    "bc",
    "maxent",
    "sieve-logit",
    "structural",
    "structural-linear",
)
EXAMPLE2_POLICY_ESTIMATORS = ("bc", "maxent", "structural-linear")
SUPPORTED_EXAMPLE1A_POLICY_ESTIMATORS = EXAMPLE1A_POLICY_ESTIMATORS + ("oracle",)
SUPPORTED_EXAMPLE1B_POLICY_ESTIMATORS = EXAMPLE1B_POLICY_ESTIMATORS + ("oracle",)
POLICY_ESTIMATORS = ("bc", "maxent")
NUISANCE_SAMPLE_MODES = ("crossfit", "independent")
CROSSFIT_SE_METHODS = ("iid", "fold-cluster", "fold-cluster-max")
CROSSFIT_CI_METHODS = ("normal", "fold-t")
DATA_FUSION_POLICY_MODES = ("known-logging", "sieve-logit", "oracle")
DATA_FUSION_TRANSITION_MODES = ("sieve", "oracle")
DATA_FUSION_G_MODES = ("frozen", "oracle")
DATA_FUSION_RATIO_MODES = ("neural-fore", "oracle-adaptive")
BEHAVIOR_POLICY_DESIGNS = ("quadratic-logit", "soft-mdp-stress")
_MONTE_CARLO_WORKER_ORACLE: Optional["JRSSBOracle"] = None


def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.clip(np.sum(exp_shifted, axis=axis, keepdims=True), EPS, None)


def logsumexp(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    max_logits = np.max(logits, axis=axis, keepdims=True)
    shifted = logits - max_logits
    out = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True)) + max_logits
    return np.squeeze(out, axis=axis)


def normal_cdf(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    erf_vec = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + erf_vec(values / math.sqrt(2.0)))


def clip_and_normalize(probs: np.ndarray, lo: float = EPS, hi: float = 1.0) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), lo, hi)
    return probs / np.clip(probs.sum(axis=1, keepdims=True), EPS, None)


def weighted_quantiles(values: np.ndarray, quantiles: Sequence[float], weights: Optional[np.ndarray] = None) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.full(len(tuple(quantiles)), np.nan, dtype=float)
    if weights is None:
        return np.quantile(values, quantiles)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cumulative = np.cumsum(weights_sorted)
    cumulative = cumulative / np.clip(cumulative[-1], EPS, None)
    return np.interp(np.asarray(list(quantiles), dtype=float), cumulative, values_sorted)


def safe_nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def combine_repeated_split_se(split_estimates: Sequence[float], split_ses: Sequence[float]) -> float:
    split_estimates = np.asarray(split_estimates, dtype=float)
    split_ses = np.asarray(split_ses, dtype=float)
    if split_estimates.size == 0 or split_ses.size == 0:
        return float("nan")
    if split_estimates.size == 1:
        return float(split_ses[0])
    return float(
        math.sqrt(
            np.mean(np.square(split_ses))
            + np.var(split_estimates, ddof=1)
        )
    )


def crossfit_se_diagnostics(
    contributions: np.ndarray,
    fold_indices: Optional[Sequence[np.ndarray]],
) -> Dict[str, float]:
    """Return iid and fold-cluster standard errors for one cross-fit estimate.

    The cluster calculation treats each held-out fold as one cluster and is a
    finite-sample diagnostic for fold-specific first-stage drift. It is not
    used for ordinary iid inference unless explicitly requested in the config.
    """
    contributions = np.asarray(contributions, dtype=float).reshape(-1)
    n = contributions.size
    if n < 2 or not np.all(np.isfinite(contributions)):
        raise ValueError("contributions must contain at least two finite values.")
    centered = contributions - float(np.mean(contributions))
    iid_se = float(np.std(centered, ddof=1) / math.sqrt(n))
    folds = [] if fold_indices is None else [np.asarray(fold, dtype=int) for fold in fold_indices]
    folds = [fold for fold in folds if fold.size > 0]
    if len(folds) < 2:
        cluster_se = float("nan")
    else:
        covered = np.concatenate(folds)
        if covered.size != n or np.unique(covered).size != n or np.any(np.sort(covered) != np.arange(n)):
            raise ValueError("fold_indices must partition all contribution rows exactly once.")
        cluster_scores = np.asarray(
            [float(np.sum(centered[fold])) for fold in folds], dtype=float
        )
        cluster_variance = (
            len(folds)
            / (len(folds) - 1.0)
            * float(np.sum(np.square(cluster_scores)))
            / float(n**2)
        )
        cluster_se = float(math.sqrt(max(cluster_variance, 0.0)))
    return {
        "iid_se": iid_se,
        "fold_cluster_se": cluster_se,
        "fold_count": float(len(folds)),
    }


def selected_crossfit_se(
    diagnostics: Dict[str, float],
    method: str,
) -> float:
    """Select a predeclared SE from cross-fit diagnostics."""
    if method not in CROSSFIT_SE_METHODS:
        raise ValueError(f"Unknown cross-fit SE method: {method}")
    iid_se = float(diagnostics["iid_se"])
    cluster_se = float(diagnostics["fold_cluster_se"])
    if method == "iid" or not np.isfinite(cluster_se):
        return iid_se
    if method == "fold-cluster":
        return cluster_se
    return max(iid_se, cluster_se)


def crossfit_critical_value(
    base_critical_value: float,
    fold_count: int,
    method: str,
) -> float:
    """Return a normal or fold-level Student critical value."""
    if method not in CROSSFIT_CI_METHODS:
        raise ValueError(f"Unknown cross-fit CI method: {method}")
    base = float(base_critical_value)
    if method == "normal" or fold_count < 2:
        return base
    from scipy.stats import t

    return max(base, float(t.ppf(0.975, df=fold_count - 1)))


def mean_action_kl(reference_probs: np.ndarray, estimated_probs: np.ndarray) -> float:
    reference_probs = clip_and_normalize(reference_probs)
    estimated_probs = clip_and_normalize(estimated_probs)
    return float(
        np.mean(
            np.sum(
                reference_probs
                * (
                    np.log(np.clip(reference_probs, EPS, None))
                    - np.log(np.clip(estimated_probs, EPS, None))
                ),
                axis=1,
            )
        )
    )


def ordinary_ratio_if_contribution(
    value: np.ndarray,
    state_action_ratio: np.ndarray,
    state_ratio: np.ndarray,
    bellman_residual: np.ndarray,
) -> np.ndarray:
    """Evaluate the ordinary EIF using stable state-action ratio algebra."""
    return value + state_action_ratio * bellman_residual + state_action_ratio - state_ratio


def signed_ratio_if_contribution(
    value: np.ndarray,
    state_action_ratio: np.ndarray,
    state_ratio: np.ndarray,
    signed_state_action_ratio: np.ndarray,
    signed_state_ratio: np.ndarray,
    bellman_residual: np.ndarray,
    policy_residual: np.ndarray,
    *,
    gamma: float,
    temperature: float,
) -> np.ndarray:
    """Evaluate the policy-dependent EIF without explicit propensity division."""
    return (
        value
        + state_action_ratio * bellman_residual
        + (gamma / temperature) * signed_state_action_ratio * policy_residual
        + signed_state_action_ratio / temperature
        + state_action_ratio
        - signed_state_ratio / temperature
        - state_ratio
    )


def reconstruct_total_signed_ratios(
    ordinary_all_action_ratios: np.ndarray,
    future_signed_all_action_ratios: np.ndarray,
    advantages: np.ndarray,
    behavior_probs: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Add the current score to the future Jordan/FORE signed ratio.

    If ``z=d(q-V)``, the signed state-action representer is
    ``(I-gamma P_pi^dagger)^(-1) z``.  Successor-initialized FORE estimates
    only the strictly future part, so the observed-action ratio must include
    ``z`` explicitly.  The state ratio is always reconstructed by averaging
    the total action-specific ratio under the fitted behavior policy; this
    immediate state average need not vanish for estimated nuisances.
    """
    ordinary = np.asarray(ordinary_all_action_ratios, dtype=float)
    future = np.asarray(future_signed_all_action_ratios, dtype=float)
    advantage = np.asarray(advantages, dtype=float)
    probs = np.asarray(behavior_probs, dtype=float)
    action_index = np.asarray(actions, dtype=int).reshape(-1)
    if not (
        ordinary.ndim == 2
        and ordinary.shape == future.shape == advantage.shape == probs.shape
        and ordinary.shape[0] == action_index.size
    ):
        raise ValueError("Signed-ratio reconstruction arrays have incompatible shapes.")
    if np.any(action_index < 0) or np.any(action_index >= ordinary.shape[1]):
        raise ValueError("Signed-ratio reconstruction received an invalid action.")
    total = ordinary * advantage + future
    if not np.all(np.isfinite(total)):
        raise FloatingPointError("Signed-ratio reconstruction is nonfinite.")
    rows = np.arange(action_index.size)
    return total[rows, action_index], np.sum(probs * total, axis=1)


def calibrate_policy_temperature(
    policy,
    states: np.ndarray,
    actions: np.ndarray,
    temperature_grid: Sequence[float],
):
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    if states.shape[0] == 0:
        return policy
    logits = np.asarray(policy.predict_logits(states), dtype=float)
    best_temperature = float(policy.parameters.get("temperature", 1.0))
    best_nll = float("inf")
    for temperature in temperature_grid:
        scaled = logits / max(float(temperature), EPS)
        scaled = scaled - np.max(scaled, axis=1, keepdims=True)
        log_probs = scaled - np.log(np.clip(np.sum(np.exp(scaled), axis=1, keepdims=True), EPS, None))
        nll = -float(np.mean(log_probs[np.arange(actions.shape[0]), actions]))
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
    parameters = dict(policy.parameters)
    parameters["temperature"] = best_temperature
    return type(policy)(n_actions=policy.n_actions, kind=policy.kind, parameters=parameters)


def _calibrate_fitted_policy(
    oracle: "JRSSBOracle",
    policy_hat,
    states: np.ndarray,
    actions: np.ndarray,
    seed: int,
    enabled: bool,
):
    if not enabled:
        return policy_hat
    n = states.shape[0]
    calib_n = max(1, int(round(oracle.config.bc_calibration_fraction * n)))
    if calib_n >= n:
        calib_n = max(1, n // 5)
    if calib_n <= 0 or calib_n >= n:
        return policy_hat
    rng = np.random.default_rng(seed + 7919)
    calib_idx = rng.choice(n, size=calib_n, replace=False)
    calibrated = calibrate_policy_temperature(
        policy_hat,
        states=states[calib_idx],
        actions=actions[calib_idx],
        temperature_grid=oracle.config.bc_temperature_grid,
    )
    return calibrated


def log_linear_policy_blend(base_probs: np.ndarray, coarse_probs: np.ndarray, alpha: float) -> np.ndarray:
    log_blend = alpha * np.log(np.clip(base_probs, EPS, None)) + (1.0 - alpha) * np.log(np.clip(coarse_probs, EPS, None))
    return clip_and_normalize(np.exp(log_blend))


class BlendedProbabilityPolicy:
    def __init__(self, oracle: "JRSSBOracle", base_policy, coarse_policy: np.ndarray, alpha: float) -> None:
        self.oracle = oracle
        self.base_policy = base_policy
        self.coarse_policy = np.asarray(coarse_policy, dtype=float)
        self.alpha = float(alpha)

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        base_probs = np.asarray(self.base_policy.predict_proba(states), dtype=float)
        coarse_probs = clip_and_normalize(self.oracle.coarse_grid.interpolate(self.coarse_policy, states))
        return log_linear_policy_blend(base_probs, coarse_probs, self.alpha)


def calibrate_example1b_policy_blend(
    oracle: "JRSSBOracle",
    base_policy,
    states: np.ndarray,
    actions: np.ndarray,
    seed: int,
):
    if not oracle.config.example1b_use_coarse_blend:
        return base_policy
    coarse_policy = estimate_coarse_behavior_policy(
        oracle=oracle,
        states=states,
        actions=actions,
        alpha=oracle.config.coarse_policy_alpha,
    )
    n = states.shape[0]
    if n == 0:
        return base_policy
    calib_n = max(1, int(round(oracle.config.bc_calibration_fraction * n)))
    calib_n = min(calib_n, n)
    rng = np.random.default_rng(seed + 12_371)
    calib_idx = rng.choice(n, size=calib_n, replace=False)
    calib_states = states[calib_idx]
    calib_actions = actions[calib_idx]
    base_probs = np.asarray(base_policy.predict_proba(calib_states), dtype=float)
    coarse_probs = clip_and_normalize(oracle.coarse_grid.interpolate(coarse_policy, calib_states))
    best_alpha = 1.0
    best_nll = float("inf")
    for alpha in oracle.config.example1b_blend_grid:
        probs = log_linear_policy_blend(base_probs, coarse_probs, float(alpha))
        nll = -float(np.mean(np.log(np.clip(probs[np.arange(calib_actions.shape[0]), calib_actions], EPS, None))))
        if nll < best_nll:
            best_nll = nll
            best_alpha = float(alpha)
    if best_alpha >= 1.0 - 1e-12:
        return base_policy
    return BlendedProbabilityPolicy(oracle=oracle, base_policy=base_policy, coarse_policy=coarse_policy, alpha=best_alpha)


class GridProbabilityPolicy:
    def __init__(self, oracle: "JRSSBOracle", policy_grid: np.ndarray) -> None:
        self.oracle = oracle
        self.policy_grid = clip_and_normalize(np.asarray(policy_grid, dtype=float))

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        return self.oracle.policy_probs(states, self.policy_grid)

    def predict_logits(self, states: np.ndarray) -> np.ndarray:
        return np.log(np.clip(self.predict_proba(states), EPS, None))


def fit_structural_behavior_policy_example1b(
    oracle: "JRSSBOracle",
    states: np.ndarray,
    actions: np.ndarray,
    seed: int,
):
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is available in the working environment
        raise RuntimeError("Structural example 1b learner requires torch.") from exc

    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    device = torch.device("cpu")
    dtype = torch.float32

    coarse_grid = oracle.coarse_grid
    n_actions = oracle.config.n_actions
    n_cells = coarse_grid.n_states
    n_points = coarse_grid.n_points
    action_index = torch.as_tensor(actions, dtype=torch.long, device=device)
    transitions = torch.as_tensor(oracle.coarse_transition_matrices(), dtype=dtype, device=device)

    ix0, iz0, wx, wz = coarse_grid._fractional_indices(states)
    ix1 = ix0 + 1
    iz1 = iz0 + 1
    idx00 = torch.as_tensor(ix0 * n_points + iz0, dtype=torch.long, device=device)
    idx01 = torch.as_tensor(ix0 * n_points + iz1, dtype=torch.long, device=device)
    idx10 = torch.as_tensor(ix1 * n_points + iz0, dtype=torch.long, device=device)
    idx11 = torch.as_tensor(ix1 * n_points + iz1, dtype=torch.long, device=device)
    w00 = torch.as_tensor((1.0 - wx) * (1.0 - wz), dtype=dtype, device=device).unsqueeze(1)
    w01 = torch.as_tensor((1.0 - wx) * wz, dtype=dtype, device=device).unsqueeze(1)
    w10 = torch.as_tensor(wx * (1.0 - wz), dtype=dtype, device=device).unsqueeze(1)
    w11 = torch.as_tensor(wx * wz, dtype=dtype, device=device).unsqueeze(1)

    raw_reward = torch.nn.Parameter(torch.zeros((n_cells, n_actions - 1), dtype=dtype, device=device))
    optimizer = torch.optim.Adam(
        [raw_reward],
        lr=oracle.config.example1b_structural_learning_rate,
        weight_decay=oracle.config.example1b_structural_weight_decay,
    )

    tau = float(oracle.config.tau_behavior)
    gamma = float(oracle.config.gamma_behavior)
    prob_lo = float(oracle.config.example1b_probability_clip_min)
    prob_hi = float(oracle.config.example1b_probability_clip_max)
    reward_scale = float(oracle.config.example1b_structural_reward_scale)
    smoothness_penalty_weight = float(oracle.config.example1b_structural_smoothness_penalty)

    coarse_empirical = estimate_coarse_behavior_policy(
        oracle=oracle,
        states=states,
        actions=actions,
        alpha=oracle.config.coarse_policy_alpha,
    )
    coarse_empirical_t = torch.as_tensor(coarse_empirical, dtype=dtype, device=device)

    linear_policy = fit_structural_linear_behavior_policy_example1b(
        oracle=oracle,
        states=states,
        actions=actions,
        seed=seed,
    )
    linear_reward_main = np.log(np.clip(linear_policy.predict_proba(oracle.main_grid.states), EPS, None))
    linear_reward_coarse = oracle.main_grid.interpolate(linear_reward_main, oracle.coarse_grid.states)
    init_reward = np.clip(linear_reward_coarse[:, 1:] / max(reward_scale, EPS), -0.95, 0.95)
    with torch.no_grad():
        raw_reward.copy_(torch.as_tensor(np.arctanh(init_reward), dtype=dtype, device=device))

    def reward_from_params() -> torch.Tensor:
        reward = torch.zeros((n_cells, n_actions), dtype=dtype, device=device)
        reward[:, 1:] = reward_scale * torch.tanh(raw_reward)
        return reward

    def solve_coarse_policy(reward_grid: torch.Tensor) -> torch.Tensor:
        q = reward_grid
        for _ in range(oracle.config.example1b_structural_bellman_iters):
            logits = q / max(tau, EPS)
            v_soft = max(tau, EPS) * torch.logsumexp(logits, dim=1)
            continuation = torch.einsum("ask,k->sa", transitions, v_soft)
            q = reward_grid + gamma * continuation
        return torch.softmax(q / max(tau, EPS), dim=1)

    def interpolate_probs(policy_grid: torch.Tensor) -> torch.Tensor:
        return (
            w00 * policy_grid[idx00]
            + w01 * policy_grid[idx01]
            + w10 * policy_grid[idx10]
            + w11 * policy_grid[idx11]
        )

    def smoothness_penalty(reward_grid: torch.Tensor) -> torch.Tensor:
        reward_cube = reward_grid[:, 1:].reshape(n_points, n_points, n_actions - 1)
        penalty_x = (reward_cube[1:, :, :] - reward_cube[:-1, :, :]).pow(2).mean()
        penalty_z = (reward_cube[:, 1:, :] - reward_cube[:, :-1, :]).pow(2).mean()
        return penalty_x + penalty_z

    torch.manual_seed(int(seed))
    for _ in range(oracle.config.example1b_structural_iters):
        reward_coarse = reward_from_params()
        policy_coarse = solve_coarse_policy(reward_coarse)
        probs = interpolate_probs(policy_coarse)
        probs = probs / torch.clamp(probs.sum(dim=1, keepdim=True), min=EPS)
        probs = torch.clamp(probs, min=prob_lo, max=prob_hi)
        probs = probs / torch.clamp(probs.sum(dim=1, keepdim=True), min=EPS)
        sample_nll = -torch.log(torch.clamp(probs[torch.arange(action_index.shape[0]), action_index], min=EPS)).mean()
        cell_cross_entropy = -torch.sum(
            coarse_empirical_t * torch.log(torch.clamp(policy_coarse, min=EPS)),
            dim=1,
        ).mean()
        objective = (
            0.5 * sample_nll
            + 0.5 * cell_cross_entropy
            + smoothness_penalty_weight * smoothness_penalty(reward_coarse)
        )
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()

    with torch.no_grad():
        reward_coarse = reward_from_params().cpu().numpy()
    reward_main = interpolate_coarse_to_main_grid(oracle, reward_coarse)
    _, _, policy_main = oracle.solve_soft_optimal_policy(
        reward_grid=reward_main,
        tau=oracle.config.tau_behavior,
        allowed_mask=np.ones(oracle.config.n_actions, dtype=bool),
    )
    policy_main = clip_and_normalize(
        np.clip(
            policy_main,
            oracle.config.example1b_probability_clip_min,
            oracle.config.example1b_probability_clip_max,
        )
    )
    return GridProbabilityPolicy(oracle=oracle, policy_grid=policy_main)


def fit_structural_linear_behavior_policy_example1b(
    oracle: "JRSSBOracle",
    states: np.ndarray,
    actions: np.ndarray,
    seed: int,
):
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is available in the working environment
        raise RuntimeError("Structural example 1b learner requires torch.") from exc

    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    device = torch.device("cpu")
    dtype = torch.float32

    n_points = oracle.main_grid.n_points
    n_actions = oracle.config.n_actions
    main_states = oracle.main_grid.states
    grid_x = torch.as_tensor(main_states[:, 0], dtype=dtype, device=device)
    grid_z = torch.as_tensor(main_states[:, 1], dtype=dtype, device=device)
    px = torch.as_tensor(oracle._transition_factors_main["px"], dtype=dtype, device=device)
    pz = torch.as_tensor(oracle._transition_factors_main["pz"], dtype=dtype, device=device)
    action_index = torch.as_tensor(actions, dtype=torch.long, device=device)

    ix0, iz0, wx, wz = oracle.main_grid._fractional_indices(states)
    ix1 = ix0 + 1
    iz1 = iz0 + 1
    idx00 = torch.as_tensor(ix0 * n_points + iz0, dtype=torch.long, device=device)
    idx01 = torch.as_tensor(ix0 * n_points + iz1, dtype=torch.long, device=device)
    idx10 = torch.as_tensor(ix1 * n_points + iz0, dtype=torch.long, device=device)
    idx11 = torch.as_tensor(ix1 * n_points + iz1, dtype=torch.long, device=device)
    w00 = torch.as_tensor((1.0 - wx) * (1.0 - wz), dtype=dtype, device=device).unsqueeze(1)
    w01 = torch.as_tensor((1.0 - wx) * wz, dtype=dtype, device=device).unsqueeze(1)
    w10 = torch.as_tensor(wx * (1.0 - wz), dtype=dtype, device=device).unsqueeze(1)
    w11 = torch.as_tensor(wx * wz, dtype=dtype, device=device).unsqueeze(1)

    # Reward family nests the true simulation DGP while remaining low-dimensional.
    beta = torch.nn.Parameter(torch.zeros((3, 5), dtype=dtype, device=device))
    raw_scale = torch.nn.Parameter(torch.full((3,), 0.5, dtype=dtype, device=device))
    optimizer = torch.optim.Adam(
        [beta, raw_scale],
        lr=oracle.config.example1b_structural_learning_rate,
        weight_decay=oracle.config.example1b_structural_weight_decay,
    )

    tau = float(oracle.config.tau_behavior)
    gamma = float(oracle.config.gamma_behavior)
    prob_lo = float(oracle.config.example1b_probability_clip_min)
    prob_hi = float(oracle.config.example1b_probability_clip_max)

    def reward_from_params() -> torch.Tensor:
        features = torch.stack(
            [
                torch.ones_like(grid_x),
                grid_x,
                grid_z,
                grid_x * grid_z,
                grid_x**2,
            ],
            dim=1,
        )
        reward = torch.zeros((grid_x.shape[0], n_actions), dtype=dtype, device=device)
        for action in range(1, n_actions):
            linear = features @ beta[action - 1]
            scale = torch.nn.functional.softplus(raw_scale[action - 1]) + 1e-3
            reward[:, action] = scale * torch.tanh(linear)
        return reward

    def solve_policy(reward_grid: torch.Tensor) -> torch.Tensor:
        q = reward_grid
        for _ in range(oracle.config.example1b_structural_bellman_iters):
            logits = q / max(tau, EPS)
            v_soft = max(tau, EPS) * torch.logsumexp(logits, dim=1)
            value_grid = v_soft.reshape(n_points, n_points)
            tmp = torch.einsum("asx,xz->asz", px, value_grid)
            continuation = torch.einsum("asz,asz->sa", tmp, pz)
            q = reward_grid + gamma * continuation
        return torch.softmax(q / max(tau, EPS), dim=1)

    def interpolate_probs(policy_grid: torch.Tensor) -> torch.Tensor:
        return (
            w00 * policy_grid[idx00]
            + w01 * policy_grid[idx01]
            + w10 * policy_grid[idx10]
            + w11 * policy_grid[idx11]
        )

    torch.manual_seed(int(seed))
    for _ in range(oracle.config.example1b_structural_iters):
        reward_grid = reward_from_params()
        policy_grid = solve_policy(reward_grid)
        probs = interpolate_probs(policy_grid)
        probs = probs / torch.clamp(probs.sum(dim=1, keepdim=True), min=EPS)
        probs = torch.clamp(probs, min=prob_lo, max=prob_hi)
        probs = probs / torch.clamp(probs.sum(dim=1, keepdim=True), min=EPS)
        nll = -torch.log(torch.clamp(probs[torch.arange(action_index.shape[0]), action_index], min=EPS)).mean()
        optimizer.zero_grad(set_to_none=True)
        nll.backward()
        optimizer.step()

    with torch.no_grad():
        reward_grid = reward_from_params()
        policy_grid = solve_policy(reward_grid).cpu().numpy()
    return GridProbabilityPolicy(oracle=oracle, policy_grid=policy_grid)


def _fit_policy_estimator(
    oracle: "JRSSBOracle",
    states: np.ndarray,
    actions: np.ndarray,
    seed: int,
    estimator: str,
    hidden_sizes: Sequence[int],
    n_epochs: int,
    prob_clip_min: float,
    prob_clip_max: float,
    temperature_calibration: bool,
    maxent_temperature: Optional[float] = None,
):
    estimator = str(estimator)
    if estimator == "bc":
        policy_hat = fit_behavior_cloning_policy(
            states=states,
            actions=actions,
            n_actions=oracle.config.n_actions,
            hidden_sizes=hidden_sizes,
            n_epochs=n_epochs,
            prob_clip_min=prob_clip_min,
            prob_clip_max=prob_clip_max,
            seed=seed,
        )
    elif estimator == "maxent":
        policy_hat = fit_maxent_irl_policy(
            states=states,
            actions=actions,
            n_actions=oracle.config.n_actions,
            hidden_sizes=hidden_sizes,
            n_iters=oracle.config.maxent_iters,
            temperature=oracle.config.tau_behavior if maxent_temperature is None else float(maxent_temperature),
            prob_clip_min=prob_clip_min,
            prob_clip_max=prob_clip_max,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown policy estimator: {estimator}")
    return _calibrate_fitted_policy(
        oracle=oracle,
        policy_hat=policy_hat,
        states=states,
        actions=actions,
        seed=seed,
        enabled=temperature_calibration,
    )


def fit_behavior_policy(
    oracle: "JRSSBOracle",
    states: np.ndarray,
    actions: np.ndarray,
    seed: int,
):
    if oracle.config.example2_policy_estimator == "structural-linear":
        return fit_structural_linear_behavior_policy_example1b(
            oracle=oracle,
            states=states,
            actions=actions,
            seed=seed,
        )
    return _fit_policy_estimator(
        oracle=oracle,
        states=states,
        actions=actions,
        seed=seed,
        estimator=oracle.config.example2_policy_estimator,
        hidden_sizes=oracle.config.bc_hidden_sizes,
        n_epochs=oracle.config.bc_epochs,
        prob_clip_min=oracle.config.probability_clip_min,
        prob_clip_max=oracle.config.probability_clip_max,
        temperature_calibration=oracle.config.bc_temperature_calibration,
    )


def fit_coarse_behavior_policy_grid(
    oracle: "JRSSBOracle",
    states: np.ndarray,
    actions: np.ndarray,
    prob_clip_min: float,
    prob_clip_max: float,
) -> GridProbabilityPolicy:
    coarse_policy = estimate_coarse_behavior_policy(
        oracle=oracle,
        states=states,
        actions=actions,
        alpha=oracle.config.coarse_policy_alpha,
    )
    main_policy = clip_and_normalize(
        np.clip(
            interpolate_coarse_to_main_grid(oracle, coarse_policy),
            prob_clip_min,
            prob_clip_max,
        )
    )
    return GridProbabilityPolicy(oracle=oracle, policy_grid=main_policy)


def fit_blended_behavior_policy_example1a(
    oracle: "JRSSBOracle",
    states: np.ndarray,
    actions: np.ndarray,
    seed: int,
):
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    n = states.shape[0]
    if n <= 1:
        return fit_coarse_behavior_policy_grid(
            oracle=oracle,
            states=states,
            actions=actions,
            prob_clip_min=oracle.config.example1a_probability_clip_min,
            prob_clip_max=oracle.config.example1a_probability_clip_max,
        )

    calib_n = max(1, int(round(oracle.config.example1a_blend_validation_fraction * n)))
    calib_n = min(calib_n, max(1, n - 1))
    rng = np.random.default_rng(seed + 17_021)
    calib_idx = np.sort(rng.choice(n, size=calib_n, replace=False))
    train_mask = np.ones(n, dtype=bool)
    train_mask[calib_idx] = False
    train_idx = np.flatnonzero(train_mask)
    if train_idx.size == 0:
        train_idx = np.arange(n, dtype=int)
        calib_idx = np.arange(n, dtype=int)

    maxent_train = _fit_policy_estimator(
        oracle=oracle,
        states=states[train_idx],
        actions=actions[train_idx],
        seed=seed,
        estimator="maxent",
        hidden_sizes=oracle.config.bc_hidden_sizes,
        n_epochs=oracle.config.bc_epochs,
        prob_clip_min=oracle.config.example1a_probability_clip_min,
        prob_clip_max=oracle.config.example1a_probability_clip_max,
        temperature_calibration=oracle.config.example1a_temperature_calibration,
    )
    coarse_train = estimate_coarse_behavior_policy(
        oracle=oracle,
        states=states[train_idx],
        actions=actions[train_idx],
        alpha=oracle.config.coarse_policy_alpha,
    )
    base_probs = np.asarray(maxent_train.predict_proba(states[calib_idx]), dtype=float)
    coarse_probs = clip_and_normalize(oracle.coarse_grid.interpolate(coarse_train, states[calib_idx]))
    calib_actions = actions[calib_idx]
    best_alpha = 1.0
    best_nll = float("inf")
    for alpha in oracle.config.example1a_blend_grid:
        probs = log_linear_policy_blend(base_probs, coarse_probs, float(alpha))
        nll = -float(np.mean(np.log(np.clip(probs[np.arange(calib_actions.shape[0]), calib_actions], EPS, None))))
        if nll < best_nll:
            best_nll = nll
            best_alpha = float(alpha)

    maxent_full = _fit_policy_estimator(
        oracle=oracle,
        states=states,
        actions=actions,
        seed=seed,
        estimator="maxent",
        hidden_sizes=oracle.config.bc_hidden_sizes,
        n_epochs=oracle.config.bc_epochs,
        prob_clip_min=oracle.config.example1a_probability_clip_min,
        prob_clip_max=oracle.config.example1a_probability_clip_max,
        temperature_calibration=oracle.config.example1a_temperature_calibration,
    )
    coarse_full = estimate_coarse_behavior_policy(
        oracle=oracle,
        states=states,
        actions=actions,
        alpha=oracle.config.coarse_policy_alpha,
    )
    if best_alpha >= 1.0 - 1e-12:
        return maxent_full
    if best_alpha <= 1e-12:
        return fit_coarse_behavior_policy_grid(
            oracle=oracle,
            states=states,
            actions=actions,
            prob_clip_min=oracle.config.example1a_probability_clip_min,
            prob_clip_max=oracle.config.example1a_probability_clip_max,
        )
    return BlendedProbabilityPolicy(
        oracle=oracle,
        base_policy=maxent_full,
        coarse_policy=coarse_full,
        alpha=best_alpha,
    )


def fit_behavior_policy_example1a(
    oracle: "JRSSBOracle",
    states: np.ndarray,
    actions: np.ndarray,
    seed: int,
):
    if oracle.config.example1a_policy_estimator == "oracle":
        return GridProbabilityPolicy(oracle=oracle, policy_grid=oracle.pi0)
    if oracle.config.example1a_policy_estimator == "sieve-logit":
        from data_fusion_simulation import fit_sieve_logit_behavior_policy

        sieve_kwargs = (
            {
                "degree": oracle.config.main_sieve_degree,
                "candidate_c": (oracle.config.main_sieve_c,),
            }
            if oracle.config.main_sieve_mode == "fixed-quadratic"
            else {}
        )
        return fit_sieve_logit_behavior_policy(
            states=states,
            actions=actions,
            n_actions=oracle.config.n_actions,
            seed=seed,
            **sieve_kwargs,
        ).policy
    if oracle.config.example1a_policy_estimator == "structural-linear":
        return fit_structural_linear_behavior_policy_example1b(
            oracle=oracle,
            states=states,
            actions=actions,
            seed=seed,
        )
    if oracle.config.example1a_policy_estimator == "blend":
        return fit_blended_behavior_policy_example1a(
            oracle=oracle,
            states=states,
            actions=actions,
            seed=seed,
        )
    if oracle.config.example1a_policy_estimator == "coarse":
        return fit_coarse_behavior_policy_grid(
            oracle=oracle,
            states=states,
            actions=actions,
            prob_clip_min=oracle.config.example1a_probability_clip_min,
            prob_clip_max=oracle.config.example1a_probability_clip_max,
        )
    return _fit_policy_estimator(
        oracle=oracle,
        states=states,
        actions=actions,
        seed=seed,
        estimator=oracle.config.example1a_policy_estimator,
        hidden_sizes=oracle.config.bc_hidden_sizes,
        n_epochs=oracle.config.bc_epochs,
        prob_clip_min=oracle.config.example1a_probability_clip_min,
        prob_clip_max=oracle.config.example1a_probability_clip_max,
        temperature_calibration=oracle.config.example1a_temperature_calibration,
    )


def fit_behavior_policy_example1b(
    oracle: "JRSSBOracle",
    states: np.ndarray,
    actions: np.ndarray,
    seed: int,
):
    if oracle.config.example1b_policy_estimator == "oracle":
        return GridProbabilityPolicy(oracle=oracle, policy_grid=oracle.pi0)
    if oracle.config.example1b_policy_estimator == "sieve-logit":
        from data_fusion_simulation import fit_sieve_logit_behavior_policy

        sieve_kwargs = (
            {
                "degree": oracle.config.main_sieve_degree,
                "candidate_c": (oracle.config.main_sieve_c,),
            }
            if oracle.config.main_sieve_mode == "fixed-quadratic"
            else {}
        )
        return fit_sieve_logit_behavior_policy(
            states=states,
            actions=actions,
            n_actions=oracle.config.n_actions,
            seed=seed,
            **sieve_kwargs,
        ).policy
    if oracle.config.example1b_policy_estimator == "structural":
        return fit_structural_behavior_policy_example1b(
            oracle=oracle,
            states=states,
            actions=actions,
            seed=seed,
        )
    if oracle.config.example1b_policy_estimator == "structural-linear":
        return fit_structural_linear_behavior_policy_example1b(
            oracle=oracle,
            states=states,
            actions=actions,
            seed=seed,
        )
    policy_hat = _fit_policy_estimator(
        oracle=oracle,
        states=states,
        actions=actions,
        seed=seed,
        estimator=oracle.config.example1b_policy_estimator,
        hidden_sizes=oracle.config.example1b_bc_hidden_sizes,
        n_epochs=oracle.config.example1b_bc_epochs,
        prob_clip_min=oracle.config.example1b_probability_clip_min,
        prob_clip_max=oracle.config.example1b_probability_clip_max,
        temperature_calibration=oracle.config.example1b_temperature_calibration,
    )
    return calibrate_example1b_policy_blend(
        oracle=oracle,
        base_policy=policy_hat,
        states=states,
        actions=actions,
        seed=seed,
    )


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    numerator = np.sum(weights) ** 2
    denominator = np.sum(weights**2)
    return float(numerator / max(denominator, EPS))


def _init_monte_carlo_worker(config: "JRSSBConfig") -> None:
    global _MONTE_CARLO_WORKER_ORACLE
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
    _MONTE_CARLO_WORKER_ORACLE = JRSSBOracle(config)


def _run_single_replication_worker(task: tuple[int, int, str, str]) -> "SingleRunResult":
    global _MONTE_CARLO_WORKER_ORACLE
    if _MONTE_CARLO_WORKER_ORACLE is None:
        raise RuntimeError("Monte Carlo worker oracle was not initialized.")
    n, seed, example_id, ratio_mode = task
    return run_single_replication_safe(
        oracle=_MONTE_CARLO_WORKER_ORACLE,
        n=n,
        seed=seed,
        example_id=example_id,
        ratio_mode=ratio_mode,
    )


def _run_single_replication_thread(
    task: tuple["JRSSBOracle", int, int, str, str]
) -> "SingleRunResult":
    oracle, n, seed, example_id, ratio_mode = task
    return run_single_replication_safe(
        oracle=oracle,
        n=n,
        seed=seed,
        example_id=example_id,
        ratio_mode=ratio_mode,
    )


@dataclass
class Grid2D:
    low: float
    high: float
    n_points: int
    x: np.ndarray = field(init=False)
    z: np.ndarray = field(init=False)
    dx: float = field(init=False)
    dz: float = field(init=False)
    x_bounds: np.ndarray = field(init=False)
    z_bounds: np.ndarray = field(init=False)
    states: np.ndarray = field(init=False)
    states_xz: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.x = np.linspace(self.low, self.high, self.n_points)
        self.z = np.linspace(self.low, self.high, self.n_points)
        self.dx = float(self.x[1] - self.x[0])
        self.dz = float(self.z[1] - self.z[0])
        self.x_bounds = self._make_bounds(self.x)
        self.z_bounds = self._make_bounds(self.z)
        xx, zz = np.meshgrid(self.x, self.z, indexing="ij")
        self.states_xz = np.stack([xx, zz], axis=-1)
        self.states = self.states_xz.reshape(-1, 2)

    @staticmethod
    def _make_bounds(points: np.ndarray) -> np.ndarray:
        mids = 0.5 * (points[1:] + points[:-1])
        return np.concatenate([[points[0]], mids, [points[-1]]])

    @property
    def n_states(self) -> int:
        return int(self.n_points * self.n_points)

    def reshape_state_values(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            return values.reshape(self.n_points, self.n_points)
        if values.ndim == 2:
            if values.shape[0] == self.n_states:
                return values.reshape(self.n_points, self.n_points, values.shape[1])
            if values.shape[1] == self.n_states:
                return values.reshape(values.shape[0], self.n_points, self.n_points)
        raise ValueError("Unexpected shape for grid reshape.")

    def _fractional_indices(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = np.asarray(states, dtype=float)
        x_scaled = (states[:, 0] - self.low) / self.dx
        z_scaled = (states[:, 1] - self.low) / self.dz
        ix0 = np.floor(x_scaled).astype(int)
        iz0 = np.floor(z_scaled).astype(int)
        ix0 = np.clip(ix0, 0, self.n_points - 2)
        iz0 = np.clip(iz0, 0, self.n_points - 2)
        wx = np.clip(x_scaled - ix0, 0.0, 1.0)
        wz = np.clip(z_scaled - iz0, 0.0, 1.0)
        return ix0, iz0, wx, wz

    def interpolate(self, values: np.ndarray, states: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        states = np.asarray(states, dtype=float)
        ix0, iz0, wx, wz = self._fractional_indices(states)
        ix1 = ix0 + 1
        iz1 = iz0 + 1
        if values.ndim == 1:
            grid_vals = values.reshape(self.n_points, self.n_points)
            v00 = grid_vals[ix0, iz0]
            v01 = grid_vals[ix0, iz1]
            v10 = grid_vals[ix1, iz0]
            v11 = grid_vals[ix1, iz1]
            return (
                (1.0 - wx) * (1.0 - wz) * v00
                + (1.0 - wx) * wz * v01
                + wx * (1.0 - wz) * v10
                + wx * wz * v11
            )
        elif values.ndim == 2:
            if values.shape[0] != self.n_states:
                raise ValueError("Expected flattened state-action values with shape (n_states, k).")
            grid_vals = values.reshape(self.n_points, self.n_points, values.shape[1])
            v00 = grid_vals[ix0, iz0]
            v01 = grid_vals[ix0, iz1]
            v10 = grid_vals[ix1, iz0]
            v11 = grid_vals[ix1, iz1]
        else:
            raise ValueError("Unsupported interpolation rank.")
        out = (
            (1.0 - wx)[:, None] * (1.0 - wz)[:, None] * v00
            + (1.0 - wx)[:, None] * wz[:, None] * v01
            + wx[:, None] * (1.0 - wz)[:, None] * v10
            + wx[:, None] * wz[:, None] * v11
        )
        return out

    def nearest_index(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        ix = np.clip(np.round((states[:, 0] - self.low) / self.dx).astype(int), 0, self.n_points - 1)
        iz = np.clip(np.round((states[:, 1] - self.low) / self.dz).astype(int), 0, self.n_points - 1)
        return ix * self.n_points + iz

    def cell_index(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        ix = np.clip(np.searchsorted(self.x_bounds, states[:, 0], side="right") - 1, 0, self.n_points - 1)
        iz = np.clip(np.searchsorted(self.z_bounds, states[:, 1], side="right") - 1, 0, self.n_points - 1)
        return ix * self.n_points + iz

    def sample_states(
        self,
        weights: np.ndarray,
        rng: np.random.Generator,
        n_samples: int,
        jitter: bool = True,
    ) -> np.ndarray:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        indices = rng.choice(self.n_states, size=n_samples, p=weights / np.clip(weights.sum(), EPS, None))
        ix = indices // self.n_points
        iz = indices % self.n_points
        if not jitter:
            return self.states[indices].copy()
        x_low = self.x_bounds[ix]
        x_high = self.x_bounds[ix + 1]
        z_low = self.z_bounds[iz]
        z_high = self.z_bounds[iz + 1]
        sampled = np.empty((n_samples, 2), dtype=float)
        sampled[:, 0] = rng.uniform(x_low, x_high)
        sampled[:, 1] = rng.uniform(z_low, z_high)
        return sampled


@dataclass
class JRSSBConfig:
    state_low: float = -2.5
    state_high: float = 2.5
    # The 41x41 discretization changes the adaptive estimand by 0.0017 versus
    # a 61x61 reference and keeps stationary-weighted ratio error near 1% or
    # below, while materially reducing repeated structural-policy fit time.
    main_grid_points: int = 41
    coarse_grid_points: int = 31
    n_actions: int = 4
    behavior_policy_design: str = "soft-mdp-stress"
    gamma_behavior: float = 0.90
    gamma_example2: float = 0.92
    data_fusion_target_gamma: float = 0.80
    tau_behavior: float = 0.80
    tau_star: float = 0.60
    noise_scale: float = 0.20
    action_subset: tuple[int, ...] = (0, 1, 3)
    stationary_tol: float = 1e-10
    bellman_tol: float = 1e-8
    nuisance_bellman_tol: float = 1e-5
    max_iterations: int = 2000
    nuisance_max_iterations: int = 100
    ratio_tol: float = 1e-10
    bc_epochs: int = 80
    bc_hidden_sizes: tuple[int, ...] = (64, 64)
    bc_temperature_calibration: bool = True
    bc_calibration_fraction: float = 0.20
    bc_temperature_grid: tuple[float, ...] = (0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0, 2.5, 3.0)
    maxent_iters: int = 150
    fqe_iters: int = 12
    fqe_epochs_per_iter: int = 8
    fqe_hidden_sizes: tuple[int, ...] = (128, 128)
    fqe_learning_rate: float = 5e-3
    example2_stage1_iters_multiplier: int = 1
    example2_stage1_epochs_multiplier: int = 1
    example2_stage2_iters_multiplier: int = 2
    example2_stage2_epochs_multiplier: int = 2
    fixed_policy_temperature: float = 0.80
    state_jitter: bool = True
    probability_clip_min: float = 0.02
    probability_clip_max: float = 0.98
    ratio_smoothing: float = 1.0
    coarse_policy_alpha: float = 2.0
    crossfit_folds: int = 5
    crossfit_se_method: str = "iid"
    crossfit_ci_method: str = "normal"
    nuisance_sample_mode: str = "crossfit"
    example1a_nuisance_method: str = "neural-main-oracle-bellman"
    example1a_policy_estimator: str = "sieve-logit"
    main_sieve_mode: str = "fixed-quadratic"
    main_sieve_degree: int = 2
    main_sieve_c: float = 10.0
    example1a_probability_clip_min: float = 1e-3
    example1a_probability_clip_max: float = 0.999
    example1a_temperature_calibration: bool = False
    example1a_blend_validation_fraction: float = 0.20
    example1a_blend_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    example1a_stabilized_lambda: float = 0.60
    example1a_repeated_splits: int = 1
    example1a_ci_critical_value: float = 1.96
    example1b_nuisance_method: str = "neural-main-oracle-bellman"
    example1b_bc_epochs: int = 80
    example1b_bc_hidden_sizes: tuple[int, ...] = (64, 64)
    example1b_probability_clip_min: float = 0.02
    example1b_probability_clip_max: float = 0.98
    # Selected by held-out action log likelihood in the policy-nuisance pilot;
    # the structural inverse fit is slower and does not improve validation.
    example1b_policy_estimator: str = "sieve-logit"
    example1b_temperature_calibration: bool = True
    example1b_use_coarse_blend: bool = False
    example1b_blend_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    # Held-out action log likelihood plateaus by 300 iterations in the
    # truth-blind behavior-policy pilot; 600 iterations doubles runtime with
    # no material validation improvement.
    example1b_structural_iters: int = 300
    # Twenty differentiable Bellman steps and 40/80-step variants have
    # indistinguishable held-out action log likelihood in the locked pilot.
    example1b_structural_bellman_iters: int = 20
    example1b_structural_learning_rate: float = 0.05
    example1b_structural_weight_decay: float = 1e-4
    example1b_structural_smoothness_penalty: float = 0.5
    example1b_structural_reward_scale: float = 4.0
    example1b_repeated_splits: int = 1
    example1b_ci_critical_value: float = 1.96
    example2_nuisance_method: str = "neural-main-oracle-bellman"
    example2_policy_estimator: str = "maxent"
    example2_repeated_splits: int = 1
    example2_ci_critical_value: float = 1.96
    fore_hidden_sizes: tuple[int, ...] = (64, 64)
    fore_learning_rate: float = 1e-3
    fore_weight_decay: float = 1e-3
    fore_iteration_budgets: tuple[int, ...] = (30, 100, 300)
    fore_batch_size: int = 512
    fore_optimizer_steps: int = 5
    fore_target_action_draws: int = 4
    fore_logit_clip: float = 10.0
    fore_grad_clip_norm: float = 10.0
    fore_device: str = "cpu"
    # Deterministic neural optimization budget. The paired full-data audit at
    # n=100,000 changes the estimate by only 0.0017 RMSE while reducing ratio
    # fit time about 3.4-fold. Policy fitting and outer-fold inference use all
    # assigned rows. Zero remains available as a full-data diagnostic.
    fore_max_training_rows: int = 20_000
    fore_signed_mass_tolerance: float = 1e-10
    data_fusion_policy_mode: str = "known-logging"
    data_fusion_transition_mode: str = "sieve"
    data_fusion_g_mode: str = "frozen"
    data_fusion_ratio_mode: str = "neural-fore"
    data_fusion_repeated_splits: int = 1
    data_fusion_probability_floor: float = 0.02
    mc_sample_sizes: tuple[int, ...] = (2500, 5000, 10000)
    mc_repetitions: int = 300
    use_oracle_cache: bool = True
    oracle_cache_dir: str = "/tmp/rl_evaluation_suite_jrssb_cache"


@dataclass
class SingleRunResult:
    example_id: str
    n: int
    seed: int
    ratio_mode: str
    nuisance_method: str
    plugin_estimate: float
    if_estimate: float
    truth: float
    estimated_se: float
    ci_lower: float
    ci_upper: float
    covered: float
    plugin_error: float
    if_error: float
    reward_rmse: float
    reward_rmse_grid: float
    reward_rmse_stationary: float
    reward_rmse_rho_weighted: float
    bellman_residual_rmse: float
    q_nu_bellman_rmse: float
    q_eval_bellman_rmse: float
    ratio_q01: float
    ratio_q50: float
    ratio_q99: float
    weight_ess: float
    pi0_action0_q01: float
    pi0_action0_q50: float
    pi0_action0_q99: float
    pi_ratio_q01: float
    pi_ratio_q50: float
    pi_ratio_q99: float
    nu_ratio_q01: float
    nu_ratio_q50: float
    nu_ratio_q99: float
    fore_selected_iterations: float = float("nan")
    fore_apbv_score: float = float("nan")
    fore_fit_seconds: float = float("nan")
    fore_training_rows: float = float("nan")
    fore_normalized_mass: float = float("nan")
    fore_logit_cap_fraction: float = float("nan")
    signed_positive_iterations: float = float("nan")
    signed_negative_iterations: float = float("nan")
    signed_positive_mass: float = float("nan")
    signed_negative_mass: float = float("nan")
    signed_positive_apbv_score: float = float("nan")
    signed_negative_apbv_score: float = float("nan")
    signed_positive_fit_seconds: float = float("nan")
    signed_negative_fit_seconds: float = float("nan")
    signed_positive_normalized_mass: float = float("nan")
    signed_negative_normalized_mass: float = float("nan")
    signed_positive_ess: float = float("nan")
    signed_negative_ess: float = float("nan")
    signed_positive_ratio_q99: float = float("nan")
    signed_negative_ratio_q99: float = float("nan")
    signed_positive_logit_cap_fraction: float = float("nan")
    signed_negative_logit_cap_fraction: float = float("nan")
    ratio_failure: float = 0.0
    fore_selected_iterations_by_fold: str = ""
    fore_apbv_scores_by_fold: str = ""
    signed_positive_iterations_by_fold: str = ""
    signed_negative_iterations_by_fold: str = ""
    signed_positive_apbv_scores_by_fold: str = ""
    signed_negative_apbv_scores_by_fold: str = ""
    failure_message: str = ""
    iid_estimated_se: float = float("nan")
    fold_cluster_estimated_se: float = float("nan")
    crossfit_fold_count: float = float("nan")
    ci_critical_value: float = float("nan")

    def as_dict(self) -> Dict[str, float | str | int]:
        return asdict(self)


class ProbabilityPolicyAdapter:
    def __init__(self, predict_proba_fn: Callable[[np.ndarray], np.ndarray], n_actions: int) -> None:
        self.predict_proba_fn = predict_proba_fn
        self.n_actions = n_actions

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        return clip_and_normalize(self.predict_proba_fn(states))

    def sample_actions(self, states: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(0 if seed is None else seed)
        probs = self.predict_proba(states)
        return np.array([rng.choice(self.n_actions, p=row) for row in probs], dtype=int)


class JRSSBOracle:
    def __init__(self, config: Optional[JRSSBConfig] = None) -> None:
        self.config = JRSSBConfig() if config is None else config
        if self.config.example1a_policy_estimator not in SUPPORTED_EXAMPLE1A_POLICY_ESTIMATORS:
            raise ValueError(f"Unknown example 1a policy estimator: {self.config.example1a_policy_estimator}")
        if self.config.example1b_policy_estimator not in SUPPORTED_EXAMPLE1B_POLICY_ESTIMATORS:
            raise ValueError(f"Unknown example 1b policy estimator: {self.config.example1b_policy_estimator}")
        if self.config.example2_policy_estimator not in EXAMPLE2_POLICY_ESTIMATORS:
            raise ValueError(f"Unknown example 2 policy estimator: {self.config.example2_policy_estimator}")
        if self.config.main_sieve_mode not in ("fixed-quadratic", "selected"):
            raise ValueError(f"Unknown main sieve mode: {self.config.main_sieve_mode}")
        if self.config.main_sieve_degree < 1 or self.config.main_sieve_c <= 0.0:
            raise ValueError("Main sieve degree and C must be positive.")
        if self.config.fore_max_training_rows < 0:
            raise ValueError("fore_max_training_rows must be nonnegative.")
        if self.config.nuisance_sample_mode not in NUISANCE_SAMPLE_MODES:
            raise ValueError(f"Unknown nuisance sample mode: {self.config.nuisance_sample_mode}")
        if self.config.crossfit_se_method not in CROSSFIT_SE_METHODS:
            raise ValueError(f"Unknown cross-fit SE method: {self.config.crossfit_se_method}")
        if self.config.crossfit_ci_method not in CROSSFIT_CI_METHODS:
            raise ValueError(f"Unknown cross-fit CI method: {self.config.crossfit_ci_method}")
        if self.config.data_fusion_policy_mode not in DATA_FUSION_POLICY_MODES:
            raise ValueError(
                f"Unknown data-fusion policy mode: {self.config.data_fusion_policy_mode}"
            )
        if self.config.data_fusion_transition_mode not in DATA_FUSION_TRANSITION_MODES:
            raise ValueError(
                "Unknown data-fusion transition mode: "
                f"{self.config.data_fusion_transition_mode}"
            )
        if self.config.data_fusion_g_mode not in DATA_FUSION_G_MODES:
            raise ValueError(f"Unknown data-fusion g mode: {self.config.data_fusion_g_mode}")
        if self.config.data_fusion_ratio_mode not in DATA_FUSION_RATIO_MODES:
            raise ValueError(
                f"Unknown data-fusion ratio mode: {self.config.data_fusion_ratio_mode}"
            )
        if not 0.0 <= self.config.data_fusion_target_gamma < 1.0:
            raise ValueError("data_fusion_target_gamma must lie in [0, 1).")
        if self.config.data_fusion_repeated_splits < 1:
            raise ValueError("data_fusion_repeated_splits must be positive.")
        if not 0.0 < self.config.data_fusion_probability_floor < 0.25:
            raise ValueError("data_fusion_probability_floor must lie in (0, 0.25).")
        if self.config.behavior_policy_design not in BEHAVIOR_POLICY_DESIGNS:
            raise ValueError(
                f"Unknown behavior-policy design: {self.config.behavior_policy_design}"
            )
        self.main_grid = Grid2D(self.config.state_low, self.config.state_high, self.config.main_grid_points)
        self.coarse_grid = Grid2D(self.config.state_low, self.config.state_high, self.config.coarse_grid_points)
        self.allowed_mask = np.zeros(self.config.n_actions, dtype=bool)
        self.allowed_mask[list(self.config.action_subset)] = True
        if not self._load_truth_bundle_from_cache():
            self._compute_truth_bundle()
            self._save_truth_bundle_to_cache()

    def _oracle_cache_file(self) -> Path:
        relevant = {
            "state_low": self.config.state_low,
            "state_high": self.config.state_high,
            "main_grid_points": self.config.main_grid_points,
            "coarse_grid_points": self.config.coarse_grid_points,
            "n_actions": self.config.n_actions,
            "behavior_policy_design": self.config.behavior_policy_design,
            "gamma_behavior": self.config.gamma_behavior,
            "gamma_example2": self.config.gamma_example2,
            "data_fusion_target_gamma": self.config.data_fusion_target_gamma,
            "tau_behavior": self.config.tau_behavior,
            "tau_star": self.config.tau_star,
            "noise_scale": self.config.noise_scale,
            "action_subset": tuple(self.config.action_subset),
            "stationary_tol": self.config.stationary_tol,
            "bellman_tol": self.config.bellman_tol,
            "max_iterations": self.config.max_iterations,
            "ratio_tol": self.config.ratio_tol,
            "fixed_policy_temperature": self.config.fixed_policy_temperature,
            # Version 7 truth arrays remain valid. The corrected signed
            # state-action objects are deterministic transforms of cached
            # eta/Q/policy arrays and are reconstructed on load below.
            "version": 7,
        }
        key = hashlib.sha256(json.dumps(relevant, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return ROOT / self.config.oracle_cache_dir / f"oracle_{key}.pkl"

    def _save_truth_bundle_to_cache(self) -> None:
        if not self.config.use_oracle_cache:
            return
        cache_file = self._oracle_cache_file()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "transition_factors_main": self._transition_factors_main,
            "transition_factors_coarse": self._transition_factors_coarse,
            "reward_dagger": self.reward_dagger,
            "q_behavior_soft": self.q_behavior_soft,
            "v_behavior_soft": self.v_behavior_soft,
            "pi0": self.pi0,
            "r0": self.r0,
            "pi_fix": self.pi_fix,
            "nu": self.nu,
            "stationary_behavior": self.stationary_behavior,
            "q_1a": self.q_1a,
            "v_1a": self.v_1a,
            "psi_1a": self.psi_1a,
            "eta_fix": self.eta_fix,
            "rho_fix": self.rho_fix,
            "q_nu": self.q_nu,
            "v_nu": self.v_nu,
            "reward_norm": self.reward_norm,
            "q_2": self.q_2,
            "v_2": self.v_2,
            "psi_2": self.psi_2,
            "eta_fix_gamma_prime": self.eta_fix_gamma_prime,
            "rho_fix_gamma_prime": self.rho_fix_gamma_prime,
            "eta_fix_data_fusion": self.eta_fix_data_fusion,
            "rho_fix_data_fusion": self.rho_fix_data_fusion,
            "q_star_soft": self.q_star_soft,
            "v_star_soft_state": self.v_star_soft_state,
            "pi_star": self.pi_star,
            "v_star": self.v_star,
            "q_1b": self.q_1b,
            "v_1b": self.v_1b,
            "psi_1b": self.psi_1b,
            "eta_star": self.eta_star,
            "rho_star": self.rho_star,
            "d_star": self.d_star,
            "tilde_eta_star": self.tilde_eta_star,
            "tilde_rho_star": self.tilde_rho_star,
            "tilde_d_star": self.tilde_d_star,
        }
        with cache_file.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_truth_bundle_from_cache(self) -> bool:
        if not self.config.use_oracle_cache:
            return False
        cache_file = self._oracle_cache_file()
        if not cache_file.exists():
            return False
        with cache_file.open("rb") as handle:
            payload = pickle.load(handle)
        self._transition_factors_main = payload["transition_factors_main"]
        self._transition_factors_coarse = payload["transition_factors_coarse"]
        for key, value in payload.items():
            if key.startswith("transition_factors"):
                continue
            setattr(self, key, value)
        behavior_joint = (
            np.clip(self.stationary_behavior, EPS, None)[:, None]
            * np.clip(self.pi0, EPS, None)
        )
        self.d_star = self.eta_star[:, None] * self.pi_star / behavior_joint
        self.tilde_d_star = self._compute_tilde_d_star()
        self.tilde_rho_star = np.sum(self.pi0 * self.tilde_d_star, axis=1)
        return True

    def _compute_truth_bundle(self) -> None:
        self._transition_factors_main = self._build_transition_factors(self.main_grid)
        self._transition_factors_coarse = self._build_transition_factors(self.coarse_grid)

        self.reward_dagger = self._reward_dagger(self.main_grid.states)
        if self.config.behavior_policy_design == "soft-mdp-stress":
            (
                self.q_behavior_soft,
                self.v_behavior_soft,
                self.pi0,
            ) = self.solve_soft_optimal_policy(
                reward_grid=self.reward_dagger,
                tau=self.config.tau_behavior,
                allowed_mask=np.ones(self.config.n_actions, dtype=bool),
            )
        else:
            self.pi0 = self._quadratic_logit_behavior_policy(self.main_grid.states)
            # These fields are retained for cache compatibility; inference uses
            # pi0 and its normalized reward r0 below.
            self.q_behavior_soft = np.log(np.clip(self.pi0, EPS, None))
            self.v_behavior_soft = np.zeros(self.main_grid.n_states, dtype=float)
        self.r0 = np.log(np.clip(self.pi0, EPS, None))
        self.pi_fix = self._fixed_policy_probs(self.main_grid.states)
        self.nu = self._reference_policy_probs(self.main_grid.states)
        self.stationary_behavior = self.stationary_distribution(self.pi0)

        self.q_1a, self.v_1a = self.evaluate_policy(self.r0, self.pi_fix, self.config.gamma_behavior)
        self.psi_1a = float(np.dot(self.stationary_behavior, self.v_1a))
        self.eta_fix = self.discounted_state_visitation(self.stationary_behavior, self.pi_fix, self.config.gamma_behavior)
        self.rho_fix = self.eta_fix / np.clip(self.stationary_behavior, EPS, None)

        self.q_nu, self.v_nu = self.evaluate_policy(self.r0, self.nu, self.config.gamma_behavior)
        self.reward_norm = self.q_nu - self.v_nu[:, None]
        self.q_2, self.v_2 = self.evaluate_policy(self.reward_norm, self.pi_fix, self.config.gamma_example2)
        self.psi_2 = float(np.dot(self.stationary_behavior, self.v_2))
        self.eta_fix_gamma_prime = self.discounted_state_visitation(
            self.stationary_behavior,
            self.pi_fix,
            self.config.gamma_example2,
        )
        self.rho_fix_gamma_prime = self.eta_fix_gamma_prime / np.clip(self.stationary_behavior, EPS, None)
        self.eta_fix_data_fusion = self.discounted_state_visitation(
            self.stationary_behavior,
            self.pi_fix,
            self.config.data_fusion_target_gamma,
        )
        self.rho_fix_data_fusion = self.eta_fix_data_fusion / np.clip(
            self.stationary_behavior, EPS, None
        )

        self.q_star_soft, self.v_star_soft_state, self.pi_star = self.solve_soft_optimal_policy(
            reward_grid=self.r0,
            tau=self.config.tau_star,
            allowed_mask=self.allowed_mask,
        )
        self.v_star = (self.q_star_soft - self.r0) / max(self.config.gamma_behavior, EPS)
        self.q_1b, self.v_1b = self.evaluate_policy(self.r0, self.pi_star, self.config.gamma_behavior)
        self.psi_1b = float(np.dot(self.stationary_behavior, self.v_1b))
        self.eta_star = self.discounted_state_visitation(self.stationary_behavior, self.pi_star, self.config.gamma_behavior)
        self.rho_star = self.eta_star / np.clip(self.stationary_behavior, EPS, None)
        behavior_joint = (
            np.clip(self.stationary_behavior, EPS, None)[:, None]
            * np.clip(self.pi0, EPS, None)
        )
        self.d_star = self.eta_star[:, None] * self.pi_star / behavior_joint
        self.tilde_eta_star = self._compute_tilde_eta_star()
        self.tilde_d_star = self._compute_tilde_d_star()
        self.tilde_rho_star = np.sum(self.pi0 * self.tilde_d_star, axis=1)

    def _build_transition_factors(self, grid: Grid2D) -> Dict[str, np.ndarray]:
        n_states = grid.n_states
        n_x = grid.n_points
        n_z = grid.n_points
        px = np.empty((self.config.n_actions, n_states, n_x), dtype=float)
        pz = np.empty((self.config.n_actions, n_states, n_z), dtype=float)

        x_bounds = grid.x_bounds.copy()
        z_bounds = grid.z_bounds.copy()
        x_bounds[0] = -np.inf
        x_bounds[-1] = np.inf
        z_bounds[0] = -np.inf
        z_bounds[-1] = np.inf

        for action in range(self.config.n_actions):
            means = self.transition_mean(grid.states, np.full(n_states, action, dtype=int))
            mean_x = means[:, 0][:, None]
            mean_z = means[:, 1][:, None]
            x_hi = normal_cdf((x_bounds[1:][None, :] - mean_x) / self.config.noise_scale)
            x_lo = normal_cdf((x_bounds[:-1][None, :] - mean_x) / self.config.noise_scale)
            z_hi = normal_cdf((z_bounds[1:][None, :] - mean_z) / self.config.noise_scale)
            z_lo = normal_cdf((z_bounds[:-1][None, :] - mean_z) / self.config.noise_scale)
            px[action] = np.clip(x_hi - x_lo, 0.0, 1.0)
            pz[action] = np.clip(z_hi - z_lo, 0.0, 1.0)
            px[action] /= np.clip(px[action].sum(axis=1, keepdims=True), EPS, None)
            pz[action] /= np.clip(pz[action].sum(axis=1, keepdims=True), EPS, None)
        return {"px": px, "pz": pz}

    def coarse_transition_matrices(self) -> np.ndarray:
        cached = getattr(self, "_coarse_transition_matrices_cache", None)
        if cached is not None:
            return cached
        matrices = np.empty(
            (self.config.n_actions, self.coarse_grid.n_states, self.coarse_grid.n_states),
            dtype=float,
        )
        for action in range(self.config.n_actions):
            matrices[action] = np.einsum(
                "sx,sz->sxz",
                self._transition_factors_coarse["px"][action],
                self._transition_factors_coarse["pz"][action],
                optimize=True,
            ).reshape(self.coarse_grid.n_states, self.coarse_grid.n_states)
        self._coarse_transition_matrices_cache = matrices
        return matrices

    def _reward_dagger(self, states: np.ndarray) -> np.ndarray:
        x = states[:, 0]
        z = states[:, 1]
        reward = np.zeros((states.shape[0], self.config.n_actions), dtype=float)
        reward[:, 1] = 1.2 * np.tanh(0.55 * x - 0.35 * z - 0.10 * x * z - 0.15)
        reward[:, 2] = 1.2 * np.tanh(0.90 * x - 0.80 * z - 0.15 * x * z - 0.60)
        reward[:, 3] = 1.2 * np.tanh(-0.45 * x + 0.30 * z - 0.05 * x**2 - 0.10)
        return reward

    def _quadratic_logit_behavior_policy(self, states: np.ndarray) -> np.ndarray:
        """Well-overlapped behavior policy in the correctly specified sieve."""
        states = np.asarray(states, dtype=float)
        x = states[:, 0]
        z = states[:, 1]
        logits = np.zeros((states.shape[0], self.config.n_actions), dtype=float)
        # Action 2 is intentionally uncommon because the adaptive target is
        # restricted to actions (0, 1, 3); the remaining logits retain smooth,
        # nontrivial state dependence without creating an artificial overlap
        # failure in the main finite-sample experiment.
        logits[:, 1] = 0.12 + 0.08 * x - 0.05 * z + 0.02 * x * z
        logits[:, 2] = -1.80 - 0.03 * x + 0.04 * z - 0.01 * x**2
        logits[:, 3] = -0.05 + 0.04 * x + 0.06 * z + 0.015 * x * z
        logits -= np.max(logits, axis=1, keepdims=True)
        return clip_and_normalize(np.exp(logits))

    def transition_mean(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=int).reshape(-1)
        x = states[:, 0]
        z = states[:, 1]
        out = np.empty_like(states)
        for action in range(self.config.n_actions):
            mask = actions == action
            if not np.any(mask):
                continue
            x_sel = x[mask]
            z_sel = z[mask]
            if action == 0:
                out[mask, 0] = 0.80 * x_sel + 0.15 * np.sin(z_sel)
                out[mask, 1] = 0.85 * z_sel + 0.10 * np.tanh(x_sel)
            elif action == 1:
                out[mask, 0] = 0.65 * x_sel + 0.10 * np.sin(z_sel) - 0.45
                out[mask, 1] = 0.85 * z_sel + 0.15 * np.tanh(x_sel) + 0.20
            elif action == 2:
                out[mask, 0] = 0.55 * x_sel + 0.10 * np.sin(z_sel) - 0.80
                out[mask, 1] = 0.80 * z_sel + 0.20 * np.tanh(x_sel) + 0.45
            else:
                out[mask, 0] = 0.90 * x_sel + 0.10 * np.sin(z_sel) + 0.35
                out[mask, 1] = 0.75 * z_sel + 0.10 * np.tanh(x_sel) - 0.30
        return np.clip(out, self.config.state_low, self.config.state_high)

    def sample_next_states(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        mean = self.transition_mean(states, actions)
        noise = rng.normal(scale=self.config.noise_scale, size=mean.shape)
        return np.clip(mean + noise, self.config.state_low, self.config.state_high)

    def _fixed_policy_probs(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        x = states[:, 0]
        z = states[:, 1]
        logits = np.stack(
            [
                0.2 - 0.2 * np.abs(x) - 0.1 * np.abs(z),
                1.4 * (x - 0.6) - 1.0 * (z - 0.7),
                -1.5 + 0.3 * x - 0.3 * z,
                1.1 * (-x - 0.3) + 1.0 * (z - 1.0),
            ],
            axis=1,
        )
        return stable_softmax(logits / max(self.config.fixed_policy_temperature, EPS), axis=1)

    def _reference_policy_probs(self, states: np.ndarray) -> np.ndarray:
        probs = np.zeros((states.shape[0], self.config.n_actions), dtype=float)
        probs[:, 0] = 1.0
        return probs

    def _expected_next_value(self, values: np.ndarray, transition_factors: Dict[str, np.ndarray]) -> np.ndarray:
        grid = self.main_grid if transition_factors is self._transition_factors_main else self.coarse_grid
        value_grid = values.reshape(grid.n_points, grid.n_points)
        tmp = np.einsum("asx,xz->asz", transition_factors["px"], value_grid, optimize=True)
        return np.einsum("asz,asz->as", tmp, transition_factors["pz"], optimize=True).transpose(1, 0)

    def solve_soft_optimal_policy(
        self,
        reward_grid: np.ndarray,
        tau: float,
        allowed_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reward_grid = np.asarray(reward_grid, dtype=float)
        q = reward_grid.copy()
        masked_fill = np.where(allowed_mask[None, :], 0.0, -1e12)
        for _ in range(self.config.max_iterations):
            logits = q / max(tau, EPS) + masked_fill
            v_soft = tau * logsumexp(logits, axis=1)
            q_new = reward_grid + self.config.gamma_behavior * self._expected_next_value(
                v_soft,
                self._transition_factors_main,
            )
            if np.max(np.abs(q_new - q)) < self.config.bellman_tol:
                q = q_new
                break
            q = q_new
        logits = q / max(tau, EPS) + masked_fill
        v_soft = tau * logsumexp(logits, axis=1)
        pi = stable_softmax(logits, axis=1)
        pi[:, ~allowed_mask] = 0.0
        pi = clip_and_normalize(pi)
        return q, v_soft, pi

    def evaluate_policy(
        self,
        reward_grid: np.ndarray,
        policy_probs: np.ndarray,
        gamma: float,
        init_q: Optional[np.ndarray] = None,
        tol: Optional[float] = None,
        max_iterations: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        reward_grid = np.asarray(reward_grid, dtype=float)
        policy_probs = clip_and_normalize(policy_probs)
        q = reward_grid.copy() if init_q is None else np.asarray(init_q, dtype=float).copy()
        tol = self.config.bellman_tol if tol is None else tol
        max_iterations = self.config.max_iterations if max_iterations is None else max_iterations
        for _ in range(max_iterations):
            v = np.sum(policy_probs * q, axis=1)
            q_new = reward_grid + gamma * self._expected_next_value(v, self._transition_factors_main)
            if np.max(np.abs(q_new - q)) < tol:
                q = q_new
                break
            q = q_new
        v = np.sum(policy_probs * q, axis=1)
        return q, v

    def pushforward_distribution(self, state_weights: np.ndarray, policy_probs: np.ndarray) -> np.ndarray:
        state_weights = np.asarray(state_weights, dtype=float).reshape(-1)
        policy_probs = clip_and_normalize(policy_probs)
        weighted_actions = state_weights[:, None] * policy_probs
        out = np.zeros((self.main_grid.n_points, self.main_grid.n_points), dtype=float)
        for action in range(self.config.n_actions):
            out += np.einsum(
                "s,sx,sz->xz",
                weighted_actions[:, action],
                self._transition_factors_main["px"][action],
                self._transition_factors_main["pz"][action],
                optimize=True,
            )
        return out.reshape(-1)

    def pushforward_weighted_actions(self, action_weights: np.ndarray) -> np.ndarray:
        action_weights = np.asarray(action_weights, dtype=float)
        out = np.zeros((self.main_grid.n_points, self.main_grid.n_points), dtype=float)
        for action in range(self.config.n_actions):
            out += np.einsum(
                "s,sx,sz->xz",
                action_weights[:, action],
                self._transition_factors_main["px"][action],
                self._transition_factors_main["pz"][action],
                optimize=True,
            )
        return out.reshape(-1)

    def stationary_distribution(self, policy_probs: np.ndarray) -> np.ndarray:
        dist = np.full(self.main_grid.n_states, 1.0 / self.main_grid.n_states, dtype=float)
        for _ in range(self.config.max_iterations):
            dist_next = self.pushforward_distribution(dist, policy_probs)
            if np.max(np.abs(dist_next - dist)) < self.config.stationary_tol:
                dist = dist_next
                break
            dist = dist_next
        return dist / np.clip(dist.sum(), EPS, None)

    def discounted_state_visitation(
        self,
        initial_state_weights: np.ndarray,
        policy_probs: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        eta = np.zeros_like(initial_state_weights, dtype=float)
        current = np.asarray(initial_state_weights, dtype=float).reshape(-1)
        discount = 1.0
        for _ in range(self.config.max_iterations):
            eta += discount * current
            current = self.pushforward_distribution(current, policy_probs)
            discount *= gamma
            if discount < self.config.ratio_tol:
                break
        return eta

    def generalized_discounted_pushforward(
        self,
        initial_state_weights: np.ndarray,
        policy_probs: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        total = np.zeros_like(initial_state_weights, dtype=float)
        current = np.asarray(initial_state_weights, dtype=float).reshape(-1)
        discount = 1.0
        for _ in range(self.config.max_iterations):
            total += discount * current
            current = self.pushforward_distribution(current, policy_probs)
            discount *= gamma
            if discount < self.config.ratio_tol:
                break
        return total

    def conditional_rho_star(self, start_state_index: int, action: int) -> np.ndarray:
        start = np.zeros(self.main_grid.n_states, dtype=float)
        start[start_state_index] = 1.0
        action_weights = np.zeros((self.main_grid.n_states, self.config.n_actions), dtype=float)
        action_weights[start_state_index, action] = 1.0
        first_step = self.pushforward_weighted_actions(action_weights)
        visitation = start + self.config.gamma_behavior * self.generalized_discounted_pushforward(
            first_step,
            self.pi_star,
            self.config.gamma_behavior,
        )
        return visitation / np.clip(self.stationary_behavior, EPS, None)

    def _compute_tilde_eta_star(self) -> np.ndarray:
        """Return the strictly future signed state measure."""
        advantage = self.q_1b - self.v_1b[:, None]
        weights = self.eta_star[:, None] * self.pi_star * advantage
        source = self.pushforward_weighted_actions(weights)
        tilde_eta = self.config.gamma_behavior * self.generalized_discounted_pushforward(
            source,
            self.pi_star,
            self.config.gamma_behavior,
        )
        return tilde_eta

    def _compute_tilde_d_star(self) -> np.ndarray:
        """Return the full current-plus-future signed state-action ratio."""
        behavior_joint = (
            np.clip(self.stationary_behavior, EPS, None)[:, None]
            * np.clip(self.pi0, EPS, None)
        )
        advantage = self.q_1b - self.v_1b[:, None]
        current = self.eta_star[:, None] * self.pi_star * advantage
        future = self.tilde_eta_star[:, None] * self.pi_star
        return (current + future) / behavior_joint

    def signed_jordan_oracle_diagnostic(self) -> Dict[str, float]:
        """Check the signed-FORE Jordan convention against the oracle grid.

        Each positive component below is the population target of the
        corresponding positive FORE fit. Recombining the two normalized
        ratios with ``gamma / (1 - gamma)`` recovers the strictly future
        component; adding the current ``d * (q - V)`` term must recover the
        full signed state--action representer used by the Example 1b EIF.
        """
        gamma = self.config.gamma_behavior
        # Match the simulator's stable factorized ratio convention exactly;
        # clipping the joint product would define a different tail ratio.
        behavior_joint = (
            np.clip(self.stationary_behavior, EPS, None)[:, None]
            * np.clip(self.pi0, EPS, None)
        )
        advantage = self.q_1b - self.v_1b[:, None]
        signed_action_source = self.eta_star[:, None] * self.pi_star * advantage
        positive_source = np.maximum(signed_action_source, 0.0)
        negative_source = np.maximum(-signed_action_source, 0.0)

        def normalized_component(action_source: np.ndarray) -> tuple[float, np.ndarray]:
            mass = float(np.sum(action_source))
            if mass <= self.config.fore_signed_mass_tolerance:
                return mass, np.zeros_like(action_source)
            pushed = self.pushforward_weighted_actions(action_source / mass)
            visitation = self.generalized_discounted_pushforward(
                pushed,
                self.pi_star,
                gamma,
            )
            normalized_ratio = (
                (1.0 - gamma)
                * visitation[:, None]
                * self.pi_star
                / behavior_joint
            )
            return mass, normalized_ratio

        positive_mass, positive_ratio = normalized_component(positive_source)
        negative_mass, negative_ratio = normalized_component(negative_source)
        reconstructed_future = (gamma / (1.0 - gamma)) * (
            positive_mass * positive_ratio - negative_mass * negative_ratio
        )
        oracle_future = (
            self.tilde_eta_star[:, None] * self.pi_star / behavior_joint
        )
        reconstructed = signed_action_source / behavior_joint + reconstructed_future
        future_error = reconstructed_future - oracle_future
        error = reconstructed - self.tilde_d_star
        reconstructed_state = np.sum(self.pi0 * reconstructed, axis=1)
        return {
            "signed_jordan_positive_mass": positive_mass,
            "signed_jordan_negative_mass": negative_mass,
            "signed_jordan_future_ratio_rmse": float(
                np.sqrt(np.mean(np.square(future_error)))
            ),
            "signed_jordan_future_ratio_sup_error": float(
                np.max(np.abs(future_error))
            ),
            "signed_jordan_ratio_rmse": float(np.sqrt(np.mean(np.square(error)))),
            "signed_jordan_ratio_sup_error": float(np.max(np.abs(error))),
            "signed_jordan_state_marginal_sup_error": float(
                np.max(np.abs(reconstructed_state - self.tilde_rho_star))
            ),
        }

    def truth_for_example(self, example_id: str) -> float:
        if example_id == "1a":
            return self.psi_1a
        if example_id == "1b":
            return self.psi_1b
        if example_id == "2":
            return self.psi_2
        raise ValueError(f"Unknown example id: {example_id}")

    def sample_stationary_transitions(self, n: int, seed: int) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        states = self.main_grid.sample_states(
            self.stationary_behavior,
            rng=rng,
            n_samples=n,
            jitter=self.config.state_jitter,
        )
        probs = self.policy_probs(states, self.pi0)
        actions = np.array([rng.choice(self.config.n_actions, p=row) for row in probs], dtype=int)
        next_states = self.sample_next_states(states, actions, rng)
        return {
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "behavior_probs": probs,
        }

    def sample_oracle_grid_transitions(self, n: int, seed: int) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        state_index = rng.choice(self.main_grid.n_states, size=n, p=self.stationary_behavior)
        behavior_probs = self.pi0[state_index]
        actions = np.array([rng.choice(self.config.n_actions, p=row) for row in behavior_probs], dtype=int)
        next_index = np.empty(n, dtype=int)
        for action in range(self.config.n_actions):
            mask = actions == action
            if not np.any(mask):
                continue
            probs_x = self._transition_factors_main["px"][action, state_index[mask]]
            probs_z = self._transition_factors_main["pz"][action, state_index[mask]]
            ix = np.array([rng.choice(self.main_grid.n_points, p=row) for row in probs_x], dtype=int)
            iz = np.array([rng.choice(self.main_grid.n_points, p=row) for row in probs_z], dtype=int)
            next_index[mask] = ix * self.main_grid.n_points + iz
        return {
            "states": self.main_grid.states[state_index].copy(),
            "actions": actions,
            "next_states": self.main_grid.states[next_index].copy(),
            "behavior_probs": behavior_probs.copy(),
            "state_index": state_index,
            "next_state_index": next_index,
        }

    def policy_probs(self, states: np.ndarray, policy_grid: np.ndarray) -> np.ndarray:
        probs = self.main_grid.interpolate(policy_grid, states)
        return clip_and_normalize(probs)

    def state_values(self, states: np.ndarray, values_grid: np.ndarray) -> np.ndarray:
        return self.main_grid.interpolate(values_grid, states)

    def action_values(self, states: np.ndarray, action_values_grid: np.ndarray) -> np.ndarray:
        return self.main_grid.interpolate(action_values_grid, states)

    def estimate_coarse_ratio_bundle(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        target_policy_probs: np.ndarray,
        gamma: float,
        q_grid: Optional[np.ndarray] = None,
        v_grid: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        grid = self.coarse_grid
        state_cells = grid.cell_index(states)
        next_cells = grid.cell_index(next_states)
        n_cells = grid.n_states
        alpha = self.config.ratio_smoothing

        state_counts = np.bincount(state_cells, minlength=n_cells).astype(float)
        behavior_state = (state_counts + alpha) / np.clip(state_counts.sum() + alpha * n_cells, EPS, None)

        transition = np.zeros((self.config.n_actions, n_cells, n_cells), dtype=float)
        sa_counts = np.zeros((n_cells, self.config.n_actions), dtype=float)
        for action in range(self.config.n_actions):
            mask = actions == action
            if not np.any(mask):
                transition[action] += 1.0 / n_cells
                continue
            sa = state_cells[mask]
            sp = next_cells[mask]
            np.add.at(sa_counts[:, action], sa, 1.0)
            np.add.at(transition[action], (sa, sp), 1.0)
        transition += alpha
        transition /= np.clip(transition.sum(axis=2, keepdims=True), EPS, None)

        cell_policy_sums = np.zeros((n_cells, self.config.n_actions), dtype=float)
        np.add.at(cell_policy_sums, state_cells, target_policy_probs)
        target_policy_cell = cell_policy_sums / np.clip(state_counts[:, None], 1.0, None)
        empty_cells = state_counts < 1.0
        if np.any(empty_cells):
            target_policy_cell[empty_cells] = self.policy_probs(grid.states[empty_cells], target_policy_probs_on_main_grid(target_policy_probs, states, self))
        target_policy_cell = clip_and_normalize(target_policy_cell)

        eta = np.zeros(n_cells, dtype=float)
        current = behavior_state.copy()
        discount = 1.0
        k_target = np.zeros((n_cells, n_cells), dtype=float)
        for action in range(self.config.n_actions):
            k_target += target_policy_cell[:, action][:, None] * transition[action]
        for _ in range(self.config.max_iterations):
            eta += discount * current
            current = k_target.T @ current
            discount *= gamma
            if discount < self.config.ratio_tol:
                break
        rho = eta / np.clip(behavior_state, EPS, None)

        output = {
            "behavior_state": behavior_state,
            "target_policy": target_policy_cell,
            "transition": transition,
            "rho": rho,
        }
        if q_grid is not None and v_grid is not None:
            q_cell = self.main_grid.interpolate(q_grid, grid.states)
            v_cell = self.main_grid.interpolate(v_grid, grid.states)
            advantage = q_cell - v_cell[:, None]
            action_weights = eta[:, None] * target_policy_cell * advantage
            source = np.zeros(n_cells, dtype=float)
            for action in range(self.config.n_actions):
                source += transition[action].T @ action_weights[:, action]
            tilde_eta = np.zeros(n_cells, dtype=float)
            current_signed = source.copy()
            discount = gamma
            for _ in range(self.config.max_iterations):
                tilde_eta += discount * current_signed
                current_signed = k_target.T @ current_signed
                discount *= gamma
                if discount < self.config.ratio_tol:
                    break
            output["tilde_rho"] = tilde_eta / np.clip(behavior_state, EPS, None)
        return output

    def run_oracle_diagnostics(
        self,
        large_n: int = 20000,
        seed: int = 123,
        sampler: str = "oracle-grid",
    ) -> Dict[str, float]:
        if sampler == "oracle-grid":
            sample = self.sample_oracle_grid_transitions(large_n, seed)
        elif sampler == "continuous":
            sample = self.sample_stationary_transitions(large_n, seed)
        else:
            raise ValueError(f"Unknown oracle diagnostic sampler: {sampler}")
        states = sample["states"]
        actions = sample["actions"]
        next_states = sample["next_states"]
        pi0 = self.policy_probs(states, self.pi0)
        pi_fix = self._fixed_policy_probs(states)
        pi_star = self.policy_probs(states, self.pi_star)
        nu = self._reference_policy_probs(states)

        r0_obs = np.log(np.clip(pi0[np.arange(large_n), actions], EPS, None))
        q_1a = self.action_values(states, self.q_1a)
        v_1a = self.state_values(states, self.v_1a)
        v_1a_next = self.state_values(next_states, self.v_1a)
        rho_1a = self.state_values(states, self.rho_fix)
        contrib_1a = (
            v_1a
            + rho_1a * (pi_fix[np.arange(large_n), actions] / np.clip(pi0[np.arange(large_n), actions], EPS, None))
            * (r0_obs + self.config.gamma_behavior * v_1a_next - q_1a[np.arange(large_n), actions])
            + rho_1a * (pi_fix[np.arange(large_n), actions] / np.clip(pi0[np.arange(large_n), actions], EPS, None) - 1.0)
        )

        q_1b = self.action_values(states, self.q_1b)
        v_1b = self.state_values(states, self.v_1b)
        v_1b_next = self.state_values(next_states, self.v_1b)
        v_star = self.action_values(states, self.v_star)
        v_star_next = self.action_values(next_states, self.v_star)
        rho_star = self.state_values(states, self.rho_star)
        pi_star_ratio = pi_star[np.arange(large_n), actions] / np.clip(pi0[np.arange(large_n), actions], EPS, None)
        ordinary_all = (
            rho_star[:, None] * pi_star / np.clip(pi0, EPS, None)
        )
        future_tilde_rho_star = self.state_values(
            states,
            self.tilde_eta_star / np.clip(self.stationary_behavior, EPS, None),
        )
        future_signed_all = (
            future_tilde_rho_star[:, None]
            * pi_star
            / np.clip(pi0, EPS, None)
        )
        tilde_d_star, tilde_rho_star = reconstruct_total_signed_ratios(
            ordinary_all,
            future_signed_all,
            q_1b - v_1b[:, None],
            pi0,
            actions,
        )
        d_star = ordinary_all[np.arange(large_n), actions]
        soft_next = self.config.tau_star * logsumexp(
            (self.action_values(next_states, self.r0) + self.config.gamma_behavior * v_star_next)
            / max(self.config.tau_star, EPS)
            + np.where(self.allowed_mask[None, :], 0.0, -1e12),
            axis=1,
        )
        contrib_1b = signed_ratio_if_contribution(
            v_1b,
            d_star,
            rho_star,
            tilde_d_star,
            tilde_rho_star,
            r0_obs + self.config.gamma_behavior * v_1b_next - q_1b[
                np.arange(large_n), actions
            ],
            soft_next - v_star[np.arange(large_n), actions],
            gamma=self.config.gamma_behavior,
            temperature=self.config.tau_star,
        )

        q_nu = self.action_values(states, self.q_nu)
        q_2 = self.action_values(states, self.q_2)
        v_nu_next = self.state_values(next_states, self.v_nu)
        v_2 = self.state_values(states, self.v_2)
        v_2_next = self.state_values(next_states, self.v_2)
        reward_norm = self.action_values(states, self.reward_norm)
        rho_2 = self.state_values(states, self.rho_fix_gamma_prime)
        pi_fix_ratio = pi_fix[np.arange(large_n), actions] / np.clip(pi0[np.arange(large_n), actions], EPS, None)
        nu_ratio = nu[np.arange(large_n), actions] / np.clip(pi0[np.arange(large_n), actions], EPS, None)
        contrib_2 = (
            v_2
            + rho_2 * pi_fix_ratio * (reward_norm[np.arange(large_n), actions] + self.config.gamma_example2 * v_2_next - q_2[np.arange(large_n), actions])
            + rho_2 * (pi_fix_ratio - nu_ratio) * (r0_obs + self.config.gamma_behavior * v_nu_next - q_nu[np.arange(large_n), actions])
            + rho_2 * (pi_fix_ratio - nu_ratio)
        )

        reward_norm_identity = np.max(np.abs(self.reward_norm - (self.q_nu - self.v_nu[:, None])))
        stationary_check = np.max(
            np.abs(self.pushforward_distribution(self.stationary_behavior, self.pi0) - self.stationary_behavior)
        )
        subset_mass = float(np.max(self.pi_star[:, ~self.allowed_mask])) if np.any(~self.allowed_mask) else 0.0

        return {
            "zero_action_reward_sup": float(np.max(np.abs(self.reward_dagger[:, 0]))),
            "r0_matches_log_pi0_sup": float(np.max(np.abs(self.r0 - np.log(np.clip(self.pi0, EPS, None))))),
            "stationary_residual_sup": float(stationary_check),
            "reward_norm_identity_sup": float(reward_norm_identity),
            "restricted_policy_outside_mass_sup": subset_mass,
            "oracle_if_mean_1a": float(np.mean(contrib_1a - self.psi_1a)),
            "oracle_if_mean_1b": float(np.mean(contrib_1b - self.psi_1b)),
            "oracle_if_mean_2": float(np.mean(contrib_2 - self.psi_2)),
        }


def target_policy_probs_on_main_grid(target_policy_probs: np.ndarray, states: np.ndarray, oracle: JRSSBOracle) -> np.ndarray:
    del states
    if target_policy_probs.shape[0] == oracle.main_grid.n_states:
        return target_policy_probs
    raise ValueError("Expected target policy probabilities on the main grid.")


def fold_splits(n: int, seed: int, n_folds: int = 2) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_folds = max(2, min(int(n_folds), int(n)))
    return [fold.astype(int, copy=False) for fold in np.array_split(perm, n_folds) if fold.size > 0]


def fore_candidates_from_config(oracle: JRSSBOracle):
    """Return the frozen hyperparameters and predeclared stopping budgets."""
    return paper_early_stopping_candidates(
        hidden_dims=tuple(oracle.config.fore_hidden_sizes),
        learning_rate=float(oracle.config.fore_learning_rate),
        weight_decay=float(oracle.config.fore_weight_decay),
        iteration_budgets=tuple(oracle.config.fore_iteration_budgets),
    )


def fore_options_from_config(oracle: JRSSBOracle) -> FOREFitOptions:
    """Return fixed minibatch FORE numerical settings."""
    return FOREFitOptions(
        batch_size=int(oracle.config.fore_batch_size),
        optimizer_steps=int(oracle.config.fore_optimizer_steps),
        target_action_draws=int(oracle.config.fore_target_action_draws),
        logit_clip=float(oracle.config.fore_logit_clip),
        grad_clip_norm=float(oracle.config.fore_grad_clip_norm),
        device=str(oracle.config.fore_device),
    )


def deterministic_fore_training_indices(
    n_rows: int,
    *,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    """Return a truth-blind fixed-size FORE training view.

    The cap affects only neural ratio fitting. Policy/value nuisances and the
    untouched outer evaluation fold continue to use their full assigned rows.
    """
    n_rows = int(n_rows)
    max_rows = int(max_rows)
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")
    if max_rows < 0:
        raise ValueError("max_rows must be nonnegative.")
    if max_rows == 0 or n_rows <= max_rows:
        return np.arange(n_rows, dtype=int)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n_rows, size=max_rows, replace=False)).astype(
        int, copy=False
    )


def exact_adaptive_ratio_bundle(
    oracle: JRSSBOracle,
    target_policy_grid: np.ndarray,
    gamma: float,
    *,
    q_grid: Optional[np.ndarray] = None,
    v_grid: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute diagnostic exact ratios for a fold-estimated target policy.

    This helper uses the simulation transition kernel and true behavior law and
    is therefore forbidden as an ordinary estimator. It provides a coherent
    oracle upper bound when the target policy itself depends on an estimated
    behavior policy.
    """
    target_policy_grid = clip_and_normalize(target_policy_grid)
    eta = oracle.discounted_state_visitation(
        oracle.stationary_behavior,
        target_policy_grid,
        gamma,
    )
    behavior_joint = (
        np.clip(oracle.stationary_behavior, EPS, None)[:, None]
        * np.clip(oracle.pi0, EPS, None)
    )
    output = {
        "eta": eta,
        "rho": eta / np.clip(oracle.stationary_behavior, EPS, None),
        "d": eta[:, None] * target_policy_grid / behavior_joint,
    }
    if (q_grid is None) != (v_grid is None):
        raise ValueError("q_grid and v_grid must be supplied together.")
    if q_grid is None or v_grid is None:
        return output
    q_grid = np.asarray(q_grid, dtype=float)
    v_grid = np.asarray(v_grid, dtype=float).reshape(-1)
    if q_grid.shape != target_policy_grid.shape or v_grid.size != q_grid.shape[0]:
        raise ValueError("Adaptive signed-ratio Q/V arrays have invalid shapes.")
    advantage = q_grid - v_grid[:, None]
    action_source = eta[:, None] * target_policy_grid * advantage
    pushed_source = oracle.pushforward_weighted_actions(action_source)
    tilde_eta = gamma * oracle.generalized_discounted_pushforward(
        pushed_source,
        target_policy_grid,
        gamma,
    )
    future_tilde_d = tilde_eta[:, None] * target_policy_grid / behavior_joint
    tilde_d = action_source / behavior_joint + future_tilde_d
    output["tilde_eta"] = tilde_eta
    output["future_tilde_d"] = future_tilde_d
    output["tilde_rho_future"] = tilde_eta / np.clip(
        oracle.stationary_behavior, EPS, None
    )
    output["tilde_d"] = tilde_d
    output["tilde_rho"] = np.sum(oracle.pi0 * tilde_d, axis=1)
    return output


def combine_transition_samples(
    train_data: Dict[str, np.ndarray],
    eval_data: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    keys = ("states", "actions", "next_states", "behavior_probs")
    return {key: np.concatenate([np.asarray(train_data[key]), np.asarray(eval_data[key])], axis=0) for key in keys}


def nuisance_method_name(oracle: JRSSBOracle, example_id: str) -> str:
    if example_id == "1a":
        return oracle.config.example1a_nuisance_method
    if example_id == "1b":
        return oracle.config.example1b_nuisance_method
    if example_id == "2":
        return oracle.config.example2_nuisance_method
    return "default"


def ci_critical_value_for_example(oracle: JRSSBOracle, example_id: str) -> float:
    if example_id == "1a":
        return float(oracle.config.example1a_ci_critical_value)
    if example_id == "1b":
        return float(oracle.config.example1b_ci_critical_value)
    if example_id == "2":
        return float(oracle.config.example2_ci_critical_value)
    return 1.96


def finalize_single_run_result(
    oracle: JRSSBOracle,
    example_id: str,
    n: int,
    seed: int,
    ratio_mode: str,
    contributions_plugin: np.ndarray,
    contributions_if: np.ndarray,
    reward_rmse_grid_parts: Sequence[float],
    reward_rmse_stationary_parts: Sequence[float],
    reward_rmse_rho_weighted_parts: Sequence[float],
    bellman_all: np.ndarray,
    bellman_q_nu_all: np.ndarray,
    bellman_q_eval_all: np.ndarray,
    ratio_parts: Sequence[np.ndarray],
    pi0_action0_parts: Sequence[np.ndarray],
    pi_ratio_parts: Sequence[np.ndarray],
    nu_ratio_parts: Sequence[np.ndarray],
    fold_indices: Optional[Sequence[np.ndarray]] = None,
) -> SingleRunResult:
    plugin_estimate = float(np.mean(contributions_plugin))
    if_estimate = float(np.mean(contributions_if))
    se_diagnostics = crossfit_se_diagnostics(contributions_if, fold_indices)
    estimated_se = selected_crossfit_se(
        se_diagnostics,
        oracle.config.crossfit_se_method,
    )
    critical_value = crossfit_critical_value(
        ci_critical_value_for_example(oracle, example_id),
        int(se_diagnostics["fold_count"]),
        oracle.config.crossfit_ci_method,
    )
    ci_halfwidth = critical_value * estimated_se
    ci_lower = if_estimate - ci_halfwidth
    ci_upper = if_estimate + ci_halfwidth
    truth = oracle.truth_for_example(example_id)
    ratio_values = np.concatenate(ratio_parts) if ratio_parts else np.array([np.nan])
    ratio_quantiles = weighted_quantiles(ratio_values, [0.01, 0.50, 0.99])
    pi0_action0_values = np.concatenate(pi0_action0_parts) if pi0_action0_parts else np.array([np.nan])
    pi_ratio_values = np.concatenate(pi_ratio_parts) if pi_ratio_parts else np.array([np.nan])
    nu_ratio_values = np.concatenate(nu_ratio_parts) if nu_ratio_parts else np.array([np.nan])
    pi0_quantiles = weighted_quantiles(pi0_action0_values, [0.01, 0.50, 0.99])
    pi_ratio_quantiles = weighted_quantiles(pi_ratio_values, [0.01, 0.50, 0.99])
    nu_ratio_quantiles = weighted_quantiles(nu_ratio_values, [0.01, 0.50, 0.99])
    return SingleRunResult(
        example_id=example_id,
        n=n,
        seed=seed,
        ratio_mode=ratio_mode,
        nuisance_method=nuisance_method_name(oracle, example_id),
        plugin_estimate=plugin_estimate,
        if_estimate=if_estimate,
        truth=truth,
        estimated_se=estimated_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        covered=float(ci_lower <= truth <= ci_upper),
        plugin_error=plugin_estimate - truth,
        if_error=if_estimate - truth,
        reward_rmse=float(np.mean(reward_rmse_grid_parts)),
        reward_rmse_grid=float(np.mean(reward_rmse_grid_parts)),
        reward_rmse_stationary=float(np.mean(reward_rmse_stationary_parts)),
        reward_rmse_rho_weighted=float(np.mean(reward_rmse_rho_weighted_parts)),
        bellman_residual_rmse=float(np.sqrt(np.mean(bellman_all**2))),
        q_nu_bellman_rmse=float(np.sqrt(np.nanmean(bellman_q_nu_all**2))) if np.any(~np.isnan(bellman_q_nu_all)) else float("nan"),
        q_eval_bellman_rmse=float(np.sqrt(np.nanmean(bellman_q_eval_all**2))) if np.any(~np.isnan(bellman_q_eval_all)) else float("nan"),
        ratio_q01=float(ratio_quantiles[0]),
        ratio_q50=float(ratio_quantiles[1]),
        ratio_q99=float(ratio_quantiles[2]),
        weight_ess=effective_sample_size(ratio_values),
        pi0_action0_q01=float(pi0_quantiles[0]),
        pi0_action0_q50=float(pi0_quantiles[1]),
        pi0_action0_q99=float(pi0_quantiles[2]),
        pi_ratio_q01=float(pi_ratio_quantiles[0]),
        pi_ratio_q50=float(pi_ratio_quantiles[1]),
        pi_ratio_q99=float(pi_ratio_quantiles[2]),
        nu_ratio_q01=float(nu_ratio_quantiles[0]),
        nu_ratio_q50=float(nu_ratio_quantiles[1]),
        nu_ratio_q99=float(nu_ratio_quantiles[2]),
        iid_estimated_se=float(se_diagnostics["iid_se"]),
        fold_cluster_estimated_se=float(se_diagnostics["fold_cluster_se"]),
        crossfit_fold_count=float(se_diagnostics["fold_count"]),
        ci_critical_value=critical_value,
    )


def attach_fore_diagnostics(
    result: SingleRunResult,
    diagnostics_parts: Sequence[Dict[str, float]],
) -> SingleRunResult:
    """Attach fold-averaged FORE telemetry to a simulation row."""
    if not diagnostics_parts:
        return result

    def average(key: str) -> float:
        values = np.asarray(
            [part.get(key, np.nan) for part in diagnostics_parts], dtype=float
        )
        return safe_nanmean(values)

    result.fore_selected_iterations = average("selected_iterations")
    result.fore_apbv_score = average("apbv_score")
    result.fore_fit_seconds = average("fit_seconds")
    result.fore_training_rows = average("training_rows")
    result.fore_normalized_mass = average("normalized_mass")
    result.fore_logit_cap_fraction = average("logit_cap_fraction")
    result.signed_positive_iterations = average("signed_positive_iterations")
    result.signed_negative_iterations = average("signed_negative_iterations")
    result.signed_positive_mass = average("signed_positive_mass")
    result.signed_negative_mass = average("signed_negative_mass")
    result.signed_positive_apbv_score = average("signed_positive_apbv_score")
    result.signed_negative_apbv_score = average("signed_negative_apbv_score")
    result.signed_positive_fit_seconds = average("signed_positive_fit_seconds")
    result.signed_negative_fit_seconds = average("signed_negative_fit_seconds")
    result.signed_positive_normalized_mass = average(
        "signed_positive_normalized_mass"
    )
    result.signed_negative_normalized_mass = average(
        "signed_negative_normalized_mass"
    )
    result.signed_positive_ess = average("signed_positive_ess")
    result.signed_negative_ess = average("signed_negative_ess")
    result.signed_positive_ratio_q99 = average("signed_positive_ratio_q99")
    result.signed_negative_ratio_q99 = average("signed_negative_ratio_q99")
    result.signed_positive_logit_cap_fraction = average(
        "signed_positive_logit_cap_fraction"
    )
    result.signed_negative_logit_cap_fraction = average(
        "signed_negative_logit_cap_fraction"
    )
    result.ratio_failure = float(any(part.get("nonfinite", False) for part in diagnostics_parts))
    result.fore_selected_iterations_by_fold = json.dumps(
        [int(part["selected_iterations"]) for part in diagnostics_parts]
    )
    result.fore_apbv_scores_by_fold = json.dumps(
        [float(part["apbv_score"]) for part in diagnostics_parts]
    )
    result.signed_positive_iterations_by_fold = json.dumps(
        [
            None
            if not np.isfinite(float(part.get("signed_positive_iterations", np.nan)))
            else int(part["signed_positive_iterations"])
            for part in diagnostics_parts
        ]
    )
    result.signed_negative_iterations_by_fold = json.dumps(
        [
            None
            if not np.isfinite(float(part.get("signed_negative_iterations", np.nan)))
            else int(part["signed_negative_iterations"])
            for part in diagnostics_parts
        ]
    )
    result.signed_positive_apbv_scores_by_fold = json.dumps(
        [
            None
            if not np.isfinite(float(part.get("signed_positive_apbv_score", np.nan)))
            else float(part["signed_positive_apbv_score"])
            for part in diagnostics_parts
        ]
    )
    result.signed_negative_apbv_scores_by_fold = json.dumps(
        [
            None
            if not np.isfinite(float(part.get("signed_negative_apbv_score", np.nan)))
            else float(part["signed_negative_apbv_score"])
            for part in diagnostics_parts
        ]
    )
    return result


def reward_rmse(states: np.ndarray, reward_hat: np.ndarray, oracle: JRSSBOracle, example_id: str) -> float:
    if example_id == "2":
        reward_true = oracle.action_values(states, oracle.reward_norm)
    else:
        reward_true = oracle.action_values(states, oracle.r0)
    diff = reward_hat - reward_true
    return float(np.sqrt(np.mean(diff**2)))


def deterministic_policy_from_probs(probs: np.ndarray) -> np.ndarray:
    actions = np.argmax(probs, axis=1)
    out = np.zeros_like(probs)
    out[np.arange(probs.shape[0]), actions] = 1.0
    return out


def evaluate_piecewise_constant(grid: Grid2D, values: np.ndarray, states: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    cell_index = grid.cell_index(states)
    if values.ndim == 1:
        return values[cell_index]
    return values[cell_index]


def fit_soft_value_from_reward(
    oracle: JRSSBOracle,
    reward_grid: np.ndarray,
    tau: float,
    allowed_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_soft, v_soft_state, pi_star = oracle.solve_soft_optimal_policy(reward_grid, tau=tau, allowed_mask=allowed_mask)
    continuation = (q_soft - reward_grid) / max(oracle.config.gamma_behavior, EPS)
    return continuation, v_soft_state, pi_star


def oracle_soft_policy_self_check(oracle: JRSSBOracle) -> Dict[str, float | bool]:
    continuation_grid, _, pi_star_grid = fit_soft_value_from_reward(
        oracle,
        reward_grid=oracle.r0,
        tau=oracle.config.tau_star,
        allowed_mask=oracle.allowed_mask,
    )
    pi_star_max_abs_diff = float(np.max(np.abs(pi_star_grid - oracle.pi_star)))
    continuation_max_abs_diff = float(np.max(np.abs(continuation_grid - oracle.v_star)))
    tol = 1e-10
    return {
        "pi_star_max_abs_diff": pi_star_max_abs_diff,
        "continuation_max_abs_diff": continuation_max_abs_diff,
        "passes": bool(pi_star_max_abs_diff <= tol and continuation_max_abs_diff <= tol),
    }


def make_policy_adapter(
    oracle: JRSSBOracle,
    policy_grid: Optional[np.ndarray] = None,
    deterministic_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> ProbabilityPolicyAdapter:
    if policy_grid is not None:
        return ProbabilityPolicyAdapter(lambda states: oracle.policy_probs(states, policy_grid), oracle.config.n_actions)
    if deterministic_fn is None:
        raise ValueError("Need either a policy grid or deterministic function.")
    return ProbabilityPolicyAdapter(deterministic_fn, oracle.config.n_actions)


def estimate_coarse_behavior_policy(
    oracle: JRSSBOracle,
    states: np.ndarray,
    actions: np.ndarray,
    alpha: Optional[float] = None,
) -> np.ndarray:
    grid = oracle.coarse_grid
    alpha = oracle.config.coarse_policy_alpha if alpha is None else float(alpha)
    state_cells = grid.cell_index(states)
    state_counts = np.bincount(state_cells, minlength=grid.n_states).astype(float)
    sa_counts = np.zeros((grid.n_states, oracle.config.n_actions), dtype=float)
    np.add.at(sa_counts, (state_cells, actions), 1.0)
    global_action_counts = np.bincount(actions, minlength=oracle.config.n_actions).astype(float)
    global_policy = clip_and_normalize(global_action_counts[None, :] + alpha)[0]
    policy_cell = (sa_counts + alpha * global_policy) / np.clip(state_counts[:, None] + alpha, EPS, None)
    empty_cells = state_counts < 1.0
    if np.any(empty_cells):
        policy_cell[empty_cells] = global_policy
    return clip_and_normalize(
        np.clip(
            policy_cell,
            oracle.config.probability_clip_min,
            oracle.config.probability_clip_max,
        )
    )


def expand_coarse_to_main_grid(oracle: JRSSBOracle, coarse_values: np.ndarray) -> np.ndarray:
    coarse_values = np.asarray(coarse_values, dtype=float)
    coarse_idx = oracle.coarse_grid.cell_index(oracle.main_grid.states)
    return coarse_values[coarse_idx]


def interpolate_coarse_to_main_grid(oracle: JRSSBOracle, coarse_values: np.ndarray) -> np.ndarray:
    coarse_values = np.asarray(coarse_values, dtype=float)
    return oracle.coarse_grid.interpolate(coarse_values, oracle.main_grid.states)


def reward_grid_diagnostics(oracle: "JRSSBOracle", reward_grid: np.ndarray, example_id: str) -> Dict[str, float]:
    reward_grid = np.asarray(reward_grid, dtype=float)
    if example_id == "2":
        reward_true = oracle.reward_norm
        target_weights = oracle.eta_fix_gamma_prime
    elif example_id == "1b":
        reward_true = oracle.r0
        target_weights = oracle.eta_star
    else:
        reward_true = oracle.r0
        target_weights = oracle.eta_fix
    state_weights = oracle.stationary_behavior / np.clip(np.sum(oracle.stationary_behavior), EPS, None)
    target_weights = np.asarray(target_weights, dtype=float).reshape(-1)
    target_weights = target_weights / np.clip(np.sum(target_weights), EPS, None)
    squared_error = np.mean((reward_grid - reward_true) ** 2, axis=1)
    return {
        "reward_rmse_grid": float(np.sqrt(np.mean(squared_error))),
        "reward_rmse_stationary": float(np.sqrt(np.sum(state_weights * squared_error))),
        "reward_rmse_rho_weighted": float(np.sqrt(np.sum(target_weights * squared_error))),
    }


def evaluate_policy_coarse(
    oracle: JRSSBOracle,
    reward_cell: np.ndarray,
    policy_cell: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    reward_cell = np.asarray(reward_cell, dtype=float)
    policy_cell = clip_and_normalize(policy_cell)
    transitions = oracle.coarse_transition_matrices()
    q = reward_cell.copy()
    for _ in range(oracle.config.max_iterations):
        v = np.sum(policy_cell * q, axis=1)
        continuation = np.einsum("ask,k->sa", transitions, v, optimize=True)
        q_new = reward_cell + gamma * continuation
        if np.max(np.abs(q_new - q)) < oracle.config.bellman_tol:
            q = q_new
            break
        q = q_new
    v = np.sum(policy_cell * q, axis=1)
    return q, v


def evaluate_fold_example_1a(
    oracle: JRSSBOracle,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    data: Dict[str, np.ndarray],
    seed: int,
    ratio_mode: str,
) -> Dict[str, np.ndarray | float]:
    if oracle.config.example1a_nuisance_method == "neural-main-oracle-bellman":
        policy_hat = fit_behavior_policy_example1a(
            oracle=oracle,
            states=data["states"][train_idx],
            actions=data["actions"][train_idx],
            seed=seed,
        )
        states_eval = data["states"][eval_idx]
        next_states_eval = data["next_states"][eval_idx]
        actions_eval = data["actions"][eval_idx]
        probs_eval = policy_hat.predict_proba(states_eval)
        reward_grid = np.log(np.clip(policy_hat.predict_proba(oracle.main_grid.states), EPS, None))
        q_init = oracle.q_1a + (reward_grid - oracle.r0)
        q_grid, v_grid = oracle.evaluate_policy(
            reward_grid,
            oracle.pi_fix,
            oracle.config.gamma_behavior,
            init_q=q_init,
            tol=oracle.config.nuisance_bellman_tol,
            max_iterations=oracle.config.nuisance_max_iterations,
        )
        q_all_eval = oracle.action_values(states_eval, q_grid)
        q_sa_eval = q_all_eval[np.arange(eval_idx.shape[0]), actions_eval]
        v_eval = oracle.state_values(states_eval, v_grid)
        v_next = oracle.state_values(next_states_eval, v_grid)
        r_obs_eval = np.log(np.clip(probs_eval[np.arange(eval_idx.shape[0]), actions_eval], EPS, None))

        pi_fix_eval = oracle._fixed_policy_probs(states_eval)
        pi_ratio = pi_fix_eval[np.arange(eval_idx.shape[0]), actions_eval] / np.clip(
            probs_eval[np.arange(eval_idx.shape[0]), actions_eval],
            EPS,
            None,
        )
        fore_diag: Dict[str, float] = {}
        if ratio_mode == "oracle":
            rho_eval = oracle.state_values(states_eval, oracle.rho_fix)
            d_eval = rho_eval * pi_ratio
        elif ratio_mode == "oracle-adaptive":
            rho_eval = oracle.state_values(states_eval, oracle.rho_fix)
            # Evaluate the estimated EIF: the target occupancy is exact, while
            # the action denominator remains the fitted behavior policy.  This
            # score term is what removes first-order policy-estimation error.
            d_eval = rho_eval * pi_ratio
        elif ratio_mode == "neural-fore":
            target_policy = make_policy_adapter(oracle, deterministic_fn=oracle._fixed_policy_probs)
            fore_idx = deterministic_fore_training_indices(
                train_idx.shape[0],
                max_rows=oracle.config.fore_max_training_rows,
                seed=seed + 677,
            )
            fore_states = data["states"][train_idx][fore_idx]
            fore_actions = data["actions"][train_idx][fore_idx]
            fore_next_states = data["next_states"][train_idx][fore_idx]
            local_split = deterministic_three_way_split(fore_idx.shape[0], seed + 701)
            selected_fore = fit_selected_fore_ratio(
                states=fore_states,
                actions=fore_actions,
                next_states=fore_next_states,
                target_policy=target_policy,
                gamma=oracle.config.gamma_behavior,
                n_actions=oracle.config.n_actions,
                candidates=fore_candidates_from_config(oracle),
                seed=seed + 809,
                options=fore_options_from_config(oracle),
                split=local_split,
            )
            d_eval = selected_fore.ratio.predict_unnormalized(states_eval, actions_eval)
            rho_eval = selected_fore.ratio.predict_state_ratio(states_eval, probs_eval)
            fore_diag = ratio_diagnostics(
                selected_fore.ratio,
                data["states"][train_idx],
                data["actions"][train_idx],
            )
            fore_diag["apbv_score"] = float(
                selected_fore.selection.worst_case_scores[selected_fore.selection.selected_index]
            )
            fore_diag["fit_seconds"] = selected_fore.total_fit_seconds
            fore_diag["training_rows"] = float(fore_idx.shape[0])
        elif ratio_mode == "coarse-estimated":
            coarse_bundle = oracle.estimate_coarse_ratio_bundle(
                states=data["states"][train_idx],
                actions=data["actions"][train_idx],
                next_states=data["next_states"][train_idx],
                target_policy_probs=oracle.pi_fix,
                gamma=oracle.config.gamma_behavior,
            )
            rho_eval = oracle.coarse_grid.interpolate(coarse_bundle["rho"], states_eval)
            d_eval = rho_eval * pi_ratio
        else:
            raise ValueError(f"Unknown ratio mode: {ratio_mode}")
        contributions_plugin = v_eval
        contributions_if = ordinary_ratio_if_contribution(
            v_eval,
            d_eval,
            rho_eval,
            r_obs_eval + oracle.config.gamma_behavior * v_next - q_sa_eval,
        )
        bellman = r_obs_eval + oracle.config.gamma_behavior * v_next - q_sa_eval
        return {
            "plugin": contributions_plugin,
            "if": contributions_if,
            "reward_grid": reward_grid,
            "reward_eval": np.log(np.clip(probs_eval, EPS, None)),
            "bellman": bellman,
            "ratio": d_eval,
            "fore_diagnostics": fore_diag,
        }

    if oracle.config.example1a_nuisance_method != "neural-fqe":
        raise ValueError(f"Unknown Example 1a nuisance method: {oracle.config.example1a_nuisance_method}")

    policy_hat = fit_behavior_policy_example1a(
        oracle=oracle,
        states=data["states"][train_idx],
        actions=data["actions"][train_idx],
        seed=seed,
    )
    eval_policy = make_policy_adapter(oracle, deterministic_fn=oracle._fixed_policy_probs)
    probs_eval = policy_hat.predict_proba(data["states"][eval_idx])
    rewards_train = np.log(np.clip(policy_hat.predict_proba(data["states"][train_idx]), EPS, None))
    r_obs_train = rewards_train[np.arange(train_idx.shape[0]), data["actions"][train_idx]]
    q_hat = fit_fqe_neural(
        states=data["states"][train_idx],
        actions=data["actions"][train_idx],
        rewards=r_obs_train,
        next_states=data["next_states"][train_idx],
        dones=np.zeros(train_idx.shape[0], dtype=float),
        policy=eval_policy,
        n_actions=oracle.config.n_actions,
        gamma=oracle.config.gamma_behavior,
        hidden_sizes=oracle.config.fqe_hidden_sizes,
        learning_rate=oracle.config.fqe_learning_rate,
        n_fqe_iters=oracle.config.fqe_iters,
        epochs_per_iter=oracle.config.fqe_epochs_per_iter,
        seed=seed,
    )

    states_eval = data["states"][eval_idx]
    next_states_eval = data["next_states"][eval_idx]
    actions_eval = data["actions"][eval_idx]
    q_all_eval = q_hat.predict_all_actions(states_eval)
    q_all_next = q_hat.predict_all_actions(next_states_eval)
    q_sa_eval = q_all_eval[np.arange(eval_idx.shape[0]), actions_eval]
    pi_fix_eval = oracle._fixed_policy_probs(states_eval)
    pi_fix_next = oracle._fixed_policy_probs(next_states_eval)
    v_eval = np.sum(pi_fix_eval * q_all_eval, axis=1)
    v_next = np.sum(pi_fix_next * q_all_next, axis=1)
    r_obs_eval = np.log(np.clip(probs_eval[np.arange(eval_idx.shape[0]), actions_eval], EPS, None))

    pi_ratio = pi_fix_eval[np.arange(eval_idx.shape[0]), actions_eval] / np.clip(
        probs_eval[np.arange(eval_idx.shape[0]), actions_eval],
        EPS,
        None,
    )
    fore_diag: Dict[str, float] = {}
    if ratio_mode == "oracle":
        rho_eval = oracle.state_values(states_eval, oracle.rho_fix)
        d_eval = rho_eval * pi_ratio
    elif ratio_mode == "oracle-adaptive":
        rho_eval = oracle.state_values(states_eval, oracle.rho_fix)
        d_eval = rho_eval * pi_ratio
    elif ratio_mode == "neural-fore":
        target_policy = make_policy_adapter(oracle, deterministic_fn=oracle._fixed_policy_probs)
        fore_idx = deterministic_fore_training_indices(
            train_idx.shape[0],
            max_rows=oracle.config.fore_max_training_rows,
            seed=seed + 677,
        )
        fore_states = data["states"][train_idx][fore_idx]
        fore_actions = data["actions"][train_idx][fore_idx]
        fore_next_states = data["next_states"][train_idx][fore_idx]
        local_split = deterministic_three_way_split(fore_idx.shape[0], seed + 701)
        selected_fore = fit_selected_fore_ratio(
            states=fore_states,
            actions=fore_actions,
            next_states=fore_next_states,
            target_policy=target_policy,
            gamma=oracle.config.gamma_behavior,
            n_actions=oracle.config.n_actions,
            candidates=fore_candidates_from_config(oracle),
            seed=seed + 809,
            options=fore_options_from_config(oracle),
            split=local_split,
        )
        d_eval = selected_fore.ratio.predict_unnormalized(states_eval, actions_eval)
        rho_eval = selected_fore.ratio.predict_state_ratio(states_eval, probs_eval)
        fore_diag = ratio_diagnostics(
            selected_fore.ratio,
            data["states"][train_idx],
            data["actions"][train_idx],
        )
        fore_diag["apbv_score"] = float(
            selected_fore.selection.worst_case_scores[selected_fore.selection.selected_index]
        )
        fore_diag["fit_seconds"] = selected_fore.total_fit_seconds
        fore_diag["training_rows"] = float(fore_idx.shape[0])
    elif ratio_mode == "coarse-estimated":
        coarse_bundle = oracle.estimate_coarse_ratio_bundle(
            states=data["states"][train_idx],
            actions=data["actions"][train_idx],
            next_states=data["next_states"][train_idx],
            target_policy_probs=oracle.pi_fix,
            gamma=oracle.config.gamma_behavior,
        )
        rho_eval = oracle.coarse_grid.interpolate(coarse_bundle["rho"], states_eval)
        d_eval = rho_eval * pi_ratio
    else:
        raise ValueError(f"Unknown ratio mode: {ratio_mode}")
    contributions_plugin = v_eval
    contributions_if = ordinary_ratio_if_contribution(
        v_eval,
        d_eval,
        rho_eval,
        r_obs_eval + oracle.config.gamma_behavior * v_next - q_sa_eval,
    )
    bellman = r_obs_eval + oracle.config.gamma_behavior * v_next - q_sa_eval
    return {
        "plugin": contributions_plugin,
        "if": contributions_if,
        "reward_grid": np.log(np.clip(policy_hat.predict_proba(oracle.main_grid.states), EPS, None)),
        "reward_eval": np.log(np.clip(probs_eval, EPS, None)),
        "bellman": bellman,
        "ratio": d_eval,
        "fore_diagnostics": fore_diag,
    }


def evaluate_fold_example_1b(
    oracle: JRSSBOracle,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    data: Dict[str, np.ndarray],
    seed: int,
    ratio_mode: str,
) -> Dict[str, np.ndarray | float]:
    policy_hat = fit_behavior_policy_example1b(
        oracle=oracle,
        states=data["states"][train_idx],
        actions=data["actions"][train_idx],
        seed=seed,
    )
    reward_grid = np.log(np.clip(policy_hat.predict_proba(oracle.main_grid.states), EPS, None))
    continuation_grid, _, pi_star_grid = fit_soft_value_from_reward(
        oracle,
        reward_grid=reward_grid,
        tau=oracle.config.tau_star,
        allowed_mask=oracle.allowed_mask,
    )
    probs_eval = policy_hat.predict_proba(data["states"][eval_idx])
    if oracle.config.example1b_nuisance_method == "neural-main-oracle-bellman":
        q_hat_grid, v_hat_grid = oracle.evaluate_policy(
            reward_grid,
            pi_star_grid,
            gamma=oracle.config.gamma_behavior,
            init_q=oracle.q_1b,
            tol=oracle.config.nuisance_bellman_tol,
            max_iterations=oracle.config.nuisance_max_iterations,
        )
        q_hat = None
    elif oracle.config.example1b_nuisance_method == "neural-fqe":
        policy_star_adapter = make_policy_adapter(oracle, policy_grid=pi_star_grid)
        r_obs_train_matrix = np.log(np.clip(policy_hat.predict_proba(data["states"][train_idx]), EPS, None))
        r_obs_train = r_obs_train_matrix[np.arange(train_idx.shape[0]), data["actions"][train_idx]]
        q_hat = fit_fqe_neural(
            states=data["states"][train_idx],
            actions=data["actions"][train_idx],
            rewards=r_obs_train,
            next_states=data["next_states"][train_idx],
            dones=np.zeros(train_idx.shape[0], dtype=float),
            policy=policy_star_adapter,
            n_actions=oracle.config.n_actions,
            gamma=oracle.config.gamma_behavior,
            hidden_sizes=oracle.config.fqe_hidden_sizes,
            learning_rate=oracle.config.fqe_learning_rate,
            n_fqe_iters=oracle.config.fqe_iters,
            epochs_per_iter=oracle.config.fqe_epochs_per_iter,
            seed=seed,
        )
        q_hat_grid = None
        v_hat_grid = None
    else:
        raise ValueError(f"Unknown Example 1b nuisance method: {oracle.config.example1b_nuisance_method}")

    states_eval = data["states"][eval_idx]
    next_states_eval = data["next_states"][eval_idx]
    actions_eval = data["actions"][eval_idx]
    pi_star_eval = oracle.policy_probs(states_eval, pi_star_grid)
    pi_star_next = oracle.policy_probs(next_states_eval, pi_star_grid)
    if q_hat_grid is not None and v_hat_grid is not None:
        q_all_eval = oracle.action_values(states_eval, q_hat_grid)
        q_sa_eval = q_all_eval[np.arange(eval_idx.shape[0]), actions_eval]
        v_eval = oracle.state_values(states_eval, v_hat_grid)
        v_next = oracle.state_values(next_states_eval, v_hat_grid)
    else:
        q_all_eval = q_hat.predict_all_actions(states_eval)
        q_all_next = q_hat.predict_all_actions(next_states_eval)
        q_sa_eval = q_all_eval[np.arange(eval_idx.shape[0]), actions_eval]
        v_eval = np.sum(pi_star_eval * q_all_eval, axis=1)
        v_next = np.sum(pi_star_next * q_all_next, axis=1)
    r_obs_eval = np.log(np.clip(probs_eval[np.arange(eval_idx.shape[0]), actions_eval], EPS, None))
    continuation_eval = oracle.action_values(states_eval, continuation_grid)
    continuation_next = oracle.action_values(next_states_eval, continuation_grid)
    reward_next = oracle.action_values(next_states_eval, reward_grid)
    soft_next = oracle.config.tau_star * logsumexp(
        (reward_next + oracle.config.gamma_behavior * continuation_next) / max(oracle.config.tau_star, EPS)
        + np.where(oracle.allowed_mask[None, :], 0.0, -1e12),
        axis=1,
    )

    pi_ratio = pi_star_eval[np.arange(eval_idx.shape[0]), actions_eval] / np.clip(
        probs_eval[np.arange(eval_idx.shape[0]), actions_eval],
        EPS,
        None,
    )
    fore_diag: Dict[str, float] = {}
    if ratio_mode == "oracle":
        rho_eval = oracle.state_values(states_eval, oracle.rho_star)
        future_tilde_rho_eval = oracle.state_values(
            states_eval, oracle.tilde_eta_star / np.clip(oracle.stationary_behavior, EPS, None)
        )
        ordinary_all = (
            rho_eval[:, None]
            * pi_star_eval
            / np.clip(probs_eval, EPS, None)
        )
        future_signed_all = (
            future_tilde_rho_eval[:, None]
            * pi_star_eval
            / np.clip(probs_eval, EPS, None)
        )
        d_eval = ordinary_all[np.arange(eval_idx.shape[0]), actions_eval]
        tilde_d_eval, tilde_rho_eval = reconstruct_total_signed_ratios(
            ordinary_all,
            future_signed_all,
            q_all_eval - v_eval[:, None],
            probs_eval,
            actions_eval,
        )
    elif ratio_mode == "oracle-adaptive":
        if q_hat_grid is None or v_hat_grid is None:
            raise ValueError(
                "oracle-adaptive signed ratios require the grid Bellman nuisance."
            )
        exact_bundle = exact_adaptive_ratio_bundle(
            oracle,
            pi_star_grid,
            oracle.config.gamma_behavior,
            q_grid=q_hat_grid,
            v_grid=v_hat_grid,
        )
        rho_eval = oracle.state_values(states_eval, exact_bundle["rho"])
        future_tilde_rho_eval = oracle.state_values(
            states_eval, exact_bundle["tilde_rho_future"]
        )
        ordinary_all = (
            rho_eval[:, None]
            * pi_star_eval
            / np.clip(probs_eval, EPS, None)
        )
        future_signed_all = (
            future_tilde_rho_eval[:, None]
            * pi_star_eval
            / np.clip(probs_eval, EPS, None)
        )
        d_eval = ordinary_all[np.arange(eval_idx.shape[0]), actions_eval]
        tilde_d_eval, tilde_rho_eval = reconstruct_total_signed_ratios(
            ordinary_all,
            future_signed_all,
            q_all_eval - v_eval[:, None],
            probs_eval,
            actions_eval,
        )
    elif ratio_mode == "neural-fore":
        target_policy = make_policy_adapter(oracle, policy_grid=pi_star_grid)
        train_states = data["states"][train_idx]
        train_actions = data["actions"][train_idx]
        train_next_states = data["next_states"][train_idx]
        fore_idx = deterministic_fore_training_indices(
            train_idx.shape[0],
            max_rows=oracle.config.fore_max_training_rows,
            seed=seed + 677,
        )
        fore_states = train_states[fore_idx]
        fore_actions = train_actions[fore_idx]
        fore_next_states = train_next_states[fore_idx]
        local_split = deterministic_three_way_split(fore_idx.shape[0], seed + 701)
        fore_candidates = fore_candidates_from_config(oracle)
        fore_options = fore_options_from_config(oracle)
        selected_fore = fit_selected_fore_ratio(
            states=fore_states,
            actions=fore_actions,
            next_states=fore_next_states,
            target_policy=target_policy,
            gamma=oracle.config.gamma_behavior,
            n_actions=oracle.config.n_actions,
            candidates=fore_candidates,
            seed=seed + 809,
            options=fore_options,
            split=local_split,
        )
        if q_hat_grid is not None and v_hat_grid is not None:
            q_all_fore = oracle.action_values(fore_states, q_hat_grid)
            v_fore = oracle.state_values(fore_states, v_hat_grid)
        else:
            q_all_fore = q_hat.predict_all_actions(fore_states)
            pi_star_fore = oracle.policy_probs(fore_states, pi_star_grid)
            v_fore = np.sum(pi_star_fore * q_all_fore, axis=1)
        q_sa_fore = q_all_fore[np.arange(fore_idx.shape[0]), fore_actions]
        preliminary_d_fore = selected_fore.selection_ratio.predict_unnormalized(
            fore_states, fore_actions
        )
        final_d_fore = selected_fore.ratio.predict_unnormalized(
            fore_states, fore_actions
        )
        signed_fore = fit_selected_signed_fore_ratio(
            states=fore_states,
            actions=fore_actions,
            next_states=fore_next_states,
            source_weights=preliminary_d_fore * (q_sa_fore - v_fore),
            refit_source_weights=final_d_fore * (q_sa_fore - v_fore),
            target_policy=target_policy,
            gamma=oracle.config.gamma_behavior,
            n_actions=oracle.config.n_actions,
            candidates=fore_candidates,
            seed=seed + 1_809,
            split=local_split,
            options=fore_options,
            mass_tolerance=oracle.config.fore_signed_mass_tolerance,
        )
        d_eval = selected_fore.ratio.predict_unnormalized(states_eval, actions_eval)
        rho_eval = selected_fore.ratio.predict_state_ratio(states_eval, probs_eval)
        repeated_states = np.repeat(states_eval, oracle.config.n_actions, axis=0)
        tiled_actions = np.tile(
            np.arange(oracle.config.n_actions), states_eval.shape[0]
        )
        ordinary_all = selected_fore.ratio.predict_unnormalized(
            repeated_states, tiled_actions
        ).reshape(states_eval.shape[0], oracle.config.n_actions)
        future_signed_all = signed_fore.predict_future_unnormalized(
            repeated_states, tiled_actions
        ).reshape(states_eval.shape[0], oracle.config.n_actions)
        tilde_d_eval, tilde_rho_eval = reconstruct_total_signed_ratios(
            ordinary_all,
            future_signed_all,
            q_all_eval - v_eval[:, None],
            probs_eval,
            actions_eval,
        )
        fore_diag = ratio_diagnostics(selected_fore.ratio, train_states, train_actions)
        positive_score = (
            float(
                signed_fore.positive_selection.worst_case_scores[
                    signed_fore.positive_selection.selected_index
                ]
            )
            if signed_fore.positive_selection is not None
            else np.nan
        )
        negative_score = (
            float(
                signed_fore.negative_selection.worst_case_scores[
                    signed_fore.negative_selection.selected_index
                ]
            )
            if signed_fore.negative_selection is not None
            else np.nan
        )
        positive_diagnostics = (
            ratio_diagnostics(signed_fore.positive, train_states, train_actions)
            if signed_fore.positive is not None
            else {}
        )
        negative_diagnostics = (
            ratio_diagnostics(signed_fore.negative, train_states, train_actions)
            if signed_fore.negative is not None
            else {}
        )
        fore_diag.update(
            {
                "apbv_score": float(
                    selected_fore.selection.worst_case_scores[selected_fore.selection.selected_index]
                ),
                "fit_seconds": selected_fore.total_fit_seconds,
                "training_rows": float(fore_idx.shape[0]),
                "signed_positive_iterations": float(
                    signed_fore.positive.candidate.num_iterations
                    if signed_fore.positive is not None
                    else np.nan
                ),
                "signed_negative_iterations": float(
                    signed_fore.negative.candidate.num_iterations
                    if signed_fore.negative is not None
                    else np.nan
                ),
                "signed_positive_mass": float(signed_fore.positive_mass),
                "signed_negative_mass": float(signed_fore.negative_mass),
                "signed_positive_apbv_score": positive_score,
                "signed_negative_apbv_score": negative_score,
                "signed_positive_fit_seconds": signed_fore.positive_total_fit_seconds,
                "signed_negative_fit_seconds": signed_fore.negative_total_fit_seconds,
                "signed_positive_normalized_mass": positive_diagnostics.get(
                    "normalized_mass", np.nan
                ),
                "signed_negative_normalized_mass": negative_diagnostics.get(
                    "normalized_mass", np.nan
                ),
                "signed_positive_ess": positive_diagnostics.get("ess", np.nan),
                "signed_negative_ess": negative_diagnostics.get("ess", np.nan),
                "signed_positive_ratio_q99": positive_diagnostics.get(
                    "ratio_q99", np.nan
                ),
                "signed_negative_ratio_q99": negative_diagnostics.get(
                    "ratio_q99", np.nan
                ),
                "signed_positive_logit_cap_fraction": positive_diagnostics.get(
                    "logit_cap_fraction", np.nan
                ),
                "signed_negative_logit_cap_fraction": negative_diagnostics.get(
                    "logit_cap_fraction", np.nan
                ),
            }
        )
    elif ratio_mode == "coarse-estimated":
        coarse_bundle = oracle.estimate_coarse_ratio_bundle(
            states=data["states"][train_idx],
            actions=data["actions"][train_idx],
            next_states=data["next_states"][train_idx],
            target_policy_probs=pi_star_grid,
            gamma=oracle.config.gamma_behavior,
            q_grid=oracle.q_1b,
            v_grid=oracle.v_1b,
        )
        rho_eval = oracle.coarse_grid.interpolate(coarse_bundle["rho"], states_eval)
        future_tilde_rho_eval = oracle.coarse_grid.interpolate(
            coarse_bundle.get("tilde_rho", coarse_bundle["rho"]), states_eval
        )
        ordinary_all = (
            rho_eval[:, None]
            * pi_star_eval
            / np.clip(probs_eval, EPS, None)
        )
        future_signed_all = (
            future_tilde_rho_eval[:, None]
            * pi_star_eval
            / np.clip(probs_eval, EPS, None)
        )
        d_eval = ordinary_all[np.arange(eval_idx.shape[0]), actions_eval]
        tilde_d_eval, tilde_rho_eval = reconstruct_total_signed_ratios(
            ordinary_all,
            future_signed_all,
            q_all_eval - v_eval[:, None],
            probs_eval,
            actions_eval,
        )
    else:
        raise ValueError(f"Unknown ratio mode: {ratio_mode}")
    contributions_plugin = v_eval
    contributions_if = signed_ratio_if_contribution(
        v_eval,
        d_eval,
        rho_eval,
        tilde_d_eval,
        tilde_rho_eval,
        r_obs_eval + oracle.config.gamma_behavior * v_next - q_sa_eval,
        soft_next - continuation_eval[np.arange(eval_idx.shape[0]), actions_eval],
        gamma=oracle.config.gamma_behavior,
        temperature=oracle.config.tau_star,
    )
    bellman = r_obs_eval + oracle.config.gamma_behavior * v_next - q_sa_eval
    return {
        "plugin": contributions_plugin,
        "if": contributions_if,
        "reward_grid": reward_grid,
        "reward_eval": np.log(np.clip(probs_eval, EPS, None)),
        "bellman": bellman,
        "ratio": d_eval,
        "fore_diagnostics": fore_diag,
    }


def evaluate_fold_example_2(
    oracle: JRSSBOracle,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    data: Dict[str, np.ndarray],
    seed: int,
    ratio_mode: str,
) -> Dict[str, np.ndarray | float]:
    if oracle.config.example2_nuisance_method == "oracle-all":
        states_eval = data["states"][eval_idx]
        next_states_eval = data["next_states"][eval_idx]
        actions_eval = data["actions"][eval_idx]
        probs_eval = oracle.policy_probs(states_eval, oracle.pi0)
        pi_fix_eval = oracle._fixed_policy_probs(states_eval)
        nu_eval = oracle._reference_policy_probs(states_eval)
        q_nu_all_eval = oracle.action_values(states_eval, oracle.q_nu)
        q_eval_all = oracle.action_values(states_eval, oracle.q_2)
        q_nu_sa = q_nu_all_eval[np.arange(eval_idx.shape[0]), actions_eval]
        q_eval_sa = q_eval_all[np.arange(eval_idx.shape[0]), actions_eval]
        reward_norm_eval = oracle.action_values(states_eval, oracle.reward_norm)
        reward_norm_sa = reward_norm_eval[np.arange(eval_idx.shape[0]), actions_eval]
        v_nu_next = oracle.state_values(next_states_eval, oracle.v_nu)
        v_eval = oracle.state_values(states_eval, oracle.v_2)
        v_eval_next = oracle.state_values(next_states_eval, oracle.v_2)
        r_obs_eval = np.log(np.clip(probs_eval[np.arange(eval_idx.shape[0]), actions_eval], EPS, None))
        rho_eval = oracle.state_values(states_eval, oracle.rho_fix_gamma_prime)
        pi_ratio = pi_fix_eval[np.arange(eval_idx.shape[0]), actions_eval] / np.clip(
            probs_eval[np.arange(eval_idx.shape[0]), actions_eval],
            EPS,
            None,
        )
        nu_ratio = nu_eval[np.arange(eval_idx.shape[0]), actions_eval] / np.clip(
            probs_eval[np.arange(eval_idx.shape[0]), actions_eval],
            EPS,
            None,
        )
        contributions_plugin = v_eval
        contributions_if = (
            v_eval
            + rho_eval * pi_ratio * (reward_norm_sa + oracle.config.gamma_example2 * v_eval_next - q_eval_sa)
            + rho_eval * (pi_ratio - nu_ratio) * (r_obs_eval + oracle.config.gamma_behavior * v_nu_next - q_nu_sa)
            + rho_eval * (pi_ratio - nu_ratio)
        )
        bellman = reward_norm_sa + oracle.config.gamma_example2 * v_eval_next - q_eval_sa
        bellman_nu = r_obs_eval + oracle.config.gamma_behavior * v_nu_next - q_nu_sa
        return {
            "plugin": contributions_plugin,
            "if": contributions_if,
            "reward_grid": oracle.r0,
            "reward_eval": reward_norm_eval,
            "bellman": bellman,
            "bellman_nu": bellman_nu,
            "bellman_eval": bellman,
            "ratio": rho_eval * pi_ratio,
            "pi0_action0": probs_eval[:, 0],
            "pi_ratio": pi_ratio,
            "nu_ratio": nu_ratio,
        }

    if oracle.config.example2_nuisance_method == "oracle-q":
        policy_hat = fit_behavior_policy(
            oracle=oracle,
            states=data["states"][train_idx],
            actions=data["actions"][train_idx],
            seed=seed,
        )
        states_eval = data["states"][eval_idx]
        next_states_eval = data["next_states"][eval_idx]
        actions_eval = data["actions"][eval_idx]
        probs_eval = policy_hat.predict_proba(states_eval)
        pi_fix_eval = oracle._fixed_policy_probs(states_eval)
        nu_eval = oracle._reference_policy_probs(states_eval)
        q_nu_all_eval = oracle.action_values(states_eval, oracle.q_nu)
        q_eval_all = oracle.action_values(states_eval, oracle.q_2)
        q_nu_sa = q_nu_all_eval[np.arange(eval_idx.shape[0]), actions_eval]
        q_eval_sa = q_eval_all[np.arange(eval_idx.shape[0]), actions_eval]
        reward_norm_eval = oracle.action_values(states_eval, oracle.reward_norm)
        reward_norm_sa = reward_norm_eval[np.arange(eval_idx.shape[0]), actions_eval]
        v_nu_next = oracle.state_values(next_states_eval, oracle.v_nu)
        v_eval = oracle.state_values(states_eval, oracle.v_2)
        v_eval_next = oracle.state_values(next_states_eval, oracle.v_2)
        r_obs_eval = np.log(np.clip(probs_eval[np.arange(eval_idx.shape[0]), actions_eval], EPS, None))
        rho_eval = oracle.state_values(states_eval, oracle.rho_fix_gamma_prime)
        pi_ratio = pi_fix_eval[np.arange(eval_idx.shape[0]), actions_eval] / np.clip(
            probs_eval[np.arange(eval_idx.shape[0]), actions_eval],
            EPS,
            None,
        )
        nu_ratio = nu_eval[np.arange(eval_idx.shape[0]), actions_eval] / np.clip(
            probs_eval[np.arange(eval_idx.shape[0]), actions_eval],
            EPS,
            None,
        )
        contributions_plugin = v_eval
        contributions_if = (
            v_eval
            + rho_eval * pi_ratio * (reward_norm_sa + oracle.config.gamma_example2 * v_eval_next - q_eval_sa)
            + rho_eval * (pi_ratio - nu_ratio) * (r_obs_eval + oracle.config.gamma_behavior * v_nu_next - q_nu_sa)
            + rho_eval * (pi_ratio - nu_ratio)
        )
        bellman = reward_norm_sa + oracle.config.gamma_example2 * v_eval_next - q_eval_sa
        bellman_nu = r_obs_eval + oracle.config.gamma_behavior * v_nu_next - q_nu_sa
        return {
            "plugin": contributions_plugin,
            "if": contributions_if,
            "reward_grid": oracle.r0,
            "reward_eval": reward_norm_eval,
            "bellman": bellman,
            "bellman_nu": bellman_nu,
            "bellman_eval": bellman,
            "ratio": rho_eval * pi_ratio,
            "pi0_action0": probs_eval[:, 0],
            "pi_ratio": pi_ratio,
            "nu_ratio": nu_ratio,
        }

    if oracle.config.example2_nuisance_method == "neural-main-oracle-bellman":
        policy_hat = fit_behavior_policy(
            oracle=oracle,
            states=data["states"][train_idx],
            actions=data["actions"][train_idx],
            seed=seed,
        )
        reward_grid = np.log(np.clip(policy_hat.predict_proba(oracle.main_grid.states), EPS, None))
        q_nu_init = oracle.q_nu + (reward_grid - oracle.r0)
        q_nu_grid, v_nu_grid = oracle.evaluate_policy(
            reward_grid,
            oracle.nu,
            oracle.config.gamma_behavior,
            init_q=q_nu_init,
            tol=oracle.config.nuisance_bellman_tol,
            max_iterations=oracle.config.nuisance_max_iterations,
        )
        reward_norm_grid = q_nu_grid - v_nu_grid[:, None]
        q_eval_init = oracle.q_2 + (reward_norm_grid - oracle.reward_norm)
        q_eval_grid, v_eval_grid = oracle.evaluate_policy(
            reward_norm_grid,
            oracle.pi_fix,
            oracle.config.gamma_example2,
            init_q=q_eval_init,
            tol=oracle.config.nuisance_bellman_tol,
            max_iterations=oracle.config.nuisance_max_iterations,
        )

        states_eval = data["states"][eval_idx]
        next_states_eval = data["next_states"][eval_idx]
        actions_eval = data["actions"][eval_idx]
        probs_eval = policy_hat.predict_proba(states_eval)
        pi_fix_eval = oracle._fixed_policy_probs(states_eval)
        nu_eval = oracle._reference_policy_probs(states_eval)

        q_nu_all_eval = oracle.action_values(states_eval, q_nu_grid)
        q_eval_all = oracle.action_values(states_eval, q_eval_grid)
        q_nu_sa = q_nu_all_eval[np.arange(eval_idx.shape[0]), actions_eval]
        q_eval_sa = q_eval_all[np.arange(eval_idx.shape[0]), actions_eval]
        reward_norm_eval = oracle.action_values(states_eval, reward_norm_grid)
        reward_norm_sa = reward_norm_eval[np.arange(eval_idx.shape[0]), actions_eval]
        v_nu_next = oracle.state_values(next_states_eval, v_nu_grid)
        v_eval = oracle.state_values(states_eval, v_eval_grid)
        v_eval_next = oracle.state_values(next_states_eval, v_eval_grid)
        r_obs_eval = np.log(np.clip(probs_eval[np.arange(eval_idx.shape[0]), actions_eval], EPS, None))

        if ratio_mode == "oracle":
            rho_eval = oracle.state_values(states_eval, oracle.rho_fix_gamma_prime)
        else:
            coarse_bundle = oracle.estimate_coarse_ratio_bundle(
                states=data["states"][train_idx],
                actions=data["actions"][train_idx],
                next_states=data["next_states"][train_idx],
                target_policy_probs=oracle.pi_fix,
                gamma=oracle.config.gamma_example2,
            )
            rho_eval = oracle.coarse_grid.interpolate(coarse_bundle["rho"], states_eval)

        pi_ratio = pi_fix_eval[np.arange(eval_idx.shape[0]), actions_eval] / np.clip(
            probs_eval[np.arange(eval_idx.shape[0]), actions_eval],
            EPS,
            None,
        )
        nu_ratio = nu_eval[np.arange(eval_idx.shape[0]), actions_eval] / np.clip(
            probs_eval[np.arange(eval_idx.shape[0]), actions_eval],
            EPS,
            None,
        )
        contributions_plugin = v_eval
        contributions_if = (
            v_eval
            + rho_eval * pi_ratio * (reward_norm_sa + oracle.config.gamma_example2 * v_eval_next - q_eval_sa)
            + rho_eval * (pi_ratio - nu_ratio) * (r_obs_eval + oracle.config.gamma_behavior * v_nu_next - q_nu_sa)
            + rho_eval * (pi_ratio - nu_ratio)
        )
        bellman = reward_norm_sa + oracle.config.gamma_example2 * v_eval_next - q_eval_sa
        bellman_nu = r_obs_eval + oracle.config.gamma_behavior * v_nu_next - q_nu_sa
        return {
            "plugin": contributions_plugin,
            "if": contributions_if,
            "reward_grid": reward_grid,
            "reward_eval": reward_norm_eval,
            "bellman": bellman,
            "bellman_nu": bellman_nu,
            "bellman_eval": bellman,
            "ratio": rho_eval * pi_ratio,
            "pi0_action0": probs_eval[:, 0],
            "pi_ratio": pi_ratio,
            "nu_ratio": nu_ratio,
        }

    if oracle.config.example2_nuisance_method in {"coarse-oracle-bellman", "neural-coarse-bellman"}:
        if oracle.config.example2_nuisance_method == "neural-coarse-bellman":
            policy_hat = fit_behavior_policy(
                oracle=oracle,
                states=data["states"][train_idx],
                actions=data["actions"][train_idx],
                seed=seed,
            )
            coarse_policy = clip_and_normalize(
                np.clip(
                    policy_hat.predict_proba(oracle.coarse_grid.states),
                    oracle.config.probability_clip_min,
                    oracle.config.probability_clip_max,
                )
            )
        else:
            coarse_policy = estimate_coarse_behavior_policy(
                oracle,
                states=data["states"][train_idx],
                actions=data["actions"][train_idx],
            )
        reward_coarse = np.log(np.clip(coarse_policy, EPS, None))
        pi_fix_coarse = oracle._fixed_policy_probs(oracle.coarse_grid.states)
        nu_coarse = oracle._reference_policy_probs(oracle.coarse_grid.states)
        q_nu_coarse, v_nu_coarse = evaluate_policy_coarse(
            oracle,
            reward_coarse,
            nu_coarse,
            oracle.config.gamma_behavior,
        )
        reward_norm_coarse = q_nu_coarse - v_nu_coarse[:, None]
        q_eval_coarse, v_eval_coarse = evaluate_policy_coarse(
            oracle,
            reward_norm_coarse,
            pi_fix_coarse,
            oracle.config.gamma_example2,
        )

        states_eval = data["states"][eval_idx]
        next_states_eval = data["next_states"][eval_idx]
        actions_eval = data["actions"][eval_idx]
        coarse_eval = evaluate_piecewise_constant(oracle.coarse_grid, coarse_policy, states_eval)
        pi_fix_eval = oracle._fixed_policy_probs(states_eval)
        nu_eval = oracle._reference_policy_probs(states_eval)

        q_nu_all_eval = evaluate_piecewise_constant(oracle.coarse_grid, q_nu_coarse, states_eval)
        q_eval_all = evaluate_piecewise_constant(oracle.coarse_grid, q_eval_coarse, states_eval)
        q_nu_sa = q_nu_all_eval[np.arange(eval_idx.shape[0]), actions_eval]
        q_eval_sa = q_eval_all[np.arange(eval_idx.shape[0]), actions_eval]
        reward_norm_eval = evaluate_piecewise_constant(oracle.coarse_grid, reward_norm_coarse, states_eval)
        reward_norm_sa = reward_norm_eval[np.arange(eval_idx.shape[0]), actions_eval]
        v_nu_next = evaluate_piecewise_constant(oracle.coarse_grid, v_nu_coarse, next_states_eval)
        v_eval = evaluate_piecewise_constant(oracle.coarse_grid, v_eval_coarse, states_eval)
        v_eval_next = evaluate_piecewise_constant(oracle.coarse_grid, v_eval_coarse, next_states_eval)
        r_obs_eval = np.log(np.clip(coarse_eval[np.arange(eval_idx.shape[0]), actions_eval], EPS, None))

        if ratio_mode == "oracle":
            rho_eval = oracle.state_values(states_eval, oracle.rho_fix_gamma_prime)
        else:
            coarse_bundle = oracle.estimate_coarse_ratio_bundle(
                states=data["states"][train_idx],
                actions=data["actions"][train_idx],
                next_states=data["next_states"][train_idx],
                target_policy_probs=oracle.pi_fix,
                gamma=oracle.config.gamma_example2,
            )
            rho_eval = oracle.coarse_grid.interpolate(coarse_bundle["rho"], states_eval)

        pi_ratio = pi_fix_eval[np.arange(eval_idx.shape[0]), actions_eval] / np.clip(
            coarse_eval[np.arange(eval_idx.shape[0]), actions_eval],
            EPS,
            None,
        )
        nu_ratio = nu_eval[np.arange(eval_idx.shape[0]), actions_eval] / np.clip(
            coarse_eval[np.arange(eval_idx.shape[0]), actions_eval],
            EPS,
            None,
        )
        contributions_plugin = v_eval
        contributions_if = (
            v_eval
            + rho_eval * pi_ratio * (reward_norm_sa + oracle.config.gamma_example2 * v_eval_next - q_eval_sa)
            + rho_eval * (pi_ratio - nu_ratio) * (r_obs_eval + oracle.config.gamma_behavior * v_nu_next - q_nu_sa)
            + rho_eval * (pi_ratio - nu_ratio)
        )
        bellman = reward_norm_sa + oracle.config.gamma_example2 * v_eval_next - q_eval_sa
        bellman_nu = r_obs_eval + oracle.config.gamma_behavior * v_nu_next - q_nu_sa
        return {
            "plugin": contributions_plugin,
            "if": contributions_if,
            "reward_grid": expand_coarse_to_main_grid(oracle, reward_coarse),
            "reward_eval": reward_norm_eval,
            "bellman": bellman,
            "bellman_nu": bellman_nu,
            "bellman_eval": bellman,
            "ratio": rho_eval * pi_ratio,
            "pi0_action0": coarse_eval[:, 0],
            "pi_ratio": pi_ratio,
            "nu_ratio": nu_ratio,
        }

    policy_hat = fit_behavior_policy(
        oracle=oracle,
        states=data["states"][train_idx],
        actions=data["actions"][train_idx],
        seed=seed,
    )
    nu_adapter = make_policy_adapter(oracle, deterministic_fn=oracle._reference_policy_probs)
    pi_fix_adapter = make_policy_adapter(oracle, deterministic_fn=oracle._fixed_policy_probs)

    train_probs = policy_hat.predict_proba(data["states"][train_idx])
    r_obs_train = np.log(np.clip(train_probs[np.arange(train_idx.shape[0]), data["actions"][train_idx]], EPS, None))
    q_nu_hat = fit_fqe_neural(
        states=data["states"][train_idx],
        actions=data["actions"][train_idx],
        rewards=r_obs_train,
        next_states=data["next_states"][train_idx],
        dones=np.zeros(train_idx.shape[0], dtype=float),
        policy=nu_adapter,
        n_actions=oracle.config.n_actions,
        gamma=oracle.config.gamma_behavior,
        hidden_sizes=oracle.config.fqe_hidden_sizes,
        learning_rate=oracle.config.fqe_learning_rate,
        n_fqe_iters=oracle.config.fqe_iters * oracle.config.example2_stage1_iters_multiplier,
        epochs_per_iter=oracle.config.fqe_epochs_per_iter * oracle.config.example2_stage1_epochs_multiplier,
        seed=seed,
    )
    q_nu_all_train = q_nu_hat.predict_all_actions(data["states"][train_idx])
    reward_norm_train_matrix = q_nu_all_train - q_nu_all_train[:, [0]]
    reward_norm_train = reward_norm_train_matrix[np.arange(train_idx.shape[0]), data["actions"][train_idx]]

    q_eval_hat = fit_fqe_neural(
        states=data["states"][train_idx],
        actions=data["actions"][train_idx],
        rewards=reward_norm_train,
        next_states=data["next_states"][train_idx],
        dones=np.zeros(train_idx.shape[0], dtype=float),
        policy=pi_fix_adapter,
        n_actions=oracle.config.n_actions,
        gamma=oracle.config.gamma_example2,
        hidden_sizes=oracle.config.fqe_hidden_sizes,
        learning_rate=oracle.config.fqe_learning_rate,
        n_fqe_iters=oracle.config.fqe_iters * oracle.config.example2_stage2_iters_multiplier,
        epochs_per_iter=oracle.config.fqe_epochs_per_iter * oracle.config.example2_stage2_epochs_multiplier,
        seed=seed + 17,
    )

    states_eval = data["states"][eval_idx]
    next_states_eval = data["next_states"][eval_idx]
    actions_eval = data["actions"][eval_idx]
    probs_eval = policy_hat.predict_proba(states_eval)
    pi_fix_eval = oracle._fixed_policy_probs(states_eval)
    nu_eval = oracle._reference_policy_probs(states_eval)

    q_nu_all_eval = q_nu_hat.predict_all_actions(states_eval)
    q_eval_all = q_eval_hat.predict_all_actions(states_eval)
    q_nu_all_next = q_nu_hat.predict_all_actions(next_states_eval)
    q_eval_all_next = q_eval_hat.predict_all_actions(next_states_eval)
    q_nu_sa = q_nu_all_eval[np.arange(eval_idx.shape[0]), actions_eval]
    q_eval_sa = q_eval_all[np.arange(eval_idx.shape[0]), actions_eval]
    reward_norm_eval = q_nu_all_eval - q_nu_all_eval[:, [0]]
    reward_norm_sa = reward_norm_eval[np.arange(eval_idx.shape[0]), actions_eval]
    v_nu_next = q_nu_all_next[:, 0]
    v_eval = np.sum(pi_fix_eval * q_eval_all, axis=1)
    v_eval_next = np.sum(oracle._fixed_policy_probs(next_states_eval) * q_eval_all_next, axis=1)
    r_obs_eval = np.log(np.clip(probs_eval[np.arange(eval_idx.shape[0]), actions_eval], EPS, None))

    if ratio_mode == "oracle":
        rho_eval = oracle.state_values(states_eval, oracle.rho_fix_gamma_prime)
    else:
        coarse_bundle = oracle.estimate_coarse_ratio_bundle(
            states=data["states"][train_idx],
            actions=data["actions"][train_idx],
            next_states=data["next_states"][train_idx],
            target_policy_probs=oracle.pi_fix,
            gamma=oracle.config.gamma_example2,
        )
        rho_eval = oracle.coarse_grid.interpolate(coarse_bundle["rho"], states_eval)

    pi_ratio = pi_fix_eval[np.arange(eval_idx.shape[0]), actions_eval] / np.clip(
        probs_eval[np.arange(eval_idx.shape[0]), actions_eval],
        EPS,
        None,
    )
    nu_ratio = nu_eval[np.arange(eval_idx.shape[0]), actions_eval] / np.clip(
        probs_eval[np.arange(eval_idx.shape[0]), actions_eval],
        EPS,
        None,
    )
    contributions_plugin = v_eval
    contributions_if = (
        v_eval
        + rho_eval * pi_ratio * (reward_norm_sa + oracle.config.gamma_example2 * v_eval_next - q_eval_sa)
        + rho_eval * (pi_ratio - nu_ratio) * (r_obs_eval + oracle.config.gamma_behavior * v_nu_next - q_nu_sa)
        + rho_eval * (pi_ratio - nu_ratio)
    )
    bellman = reward_norm_sa + oracle.config.gamma_example2 * v_eval_next - q_eval_sa
    bellman_nu = r_obs_eval + oracle.config.gamma_behavior * v_nu_next - q_nu_sa
    return {
        "plugin": contributions_plugin,
        "if": contributions_if,
        "reward_grid": reward_norm_eval,
        "reward_eval": reward_norm_eval,
        "bellman": bellman,
        "bellman_nu": bellman_nu,
        "bellman_eval": bellman,
        "ratio": rho_eval * pi_ratio,
        "pi0_action0": probs_eval[:, 0],
        "pi_ratio": pi_ratio,
        "nu_ratio": nu_ratio,
    }


def estimate_example_1a_stabilized(
    oracle: JRSSBOracle,
    data: Dict[str, np.ndarray],
    seed: int,
    ratio_mode: str = "oracle",
) -> SingleRunResult:
    n = data["states"].shape[0]
    lam = oracle.config.example1a_stabilized_lambda
    repeated_splits = max(1, int(oracle.config.example1a_repeated_splits))
    plugin_split_estimates: List[float] = []
    stabilized_split_estimates: List[float] = []
    stabilized_split_ses: List[float] = []
    reward_rmse_grid_parts: List[float] = []
    reward_rmse_stationary_parts: List[float] = []
    reward_rmse_rho_weighted_parts: List[float] = []
    bellman_values: List[np.ndarray] = []
    ratio_parts: List[np.ndarray] = []
    fore_diagnostics_parts: List[Dict[str, float]] = []

    for split_number in range(repeated_splits):
        split_seed = seed + 100_003 * split_number
        fold_indices = fold_splits(n, split_seed, n_folds=oracle.config.crossfit_folds)
        all_idx = np.arange(n)
        contributions_plugin = np.zeros(n, dtype=float)
        contributions_if = np.zeros(n, dtype=float)
        bellman_all = np.zeros(n, dtype=float)
        for fold_number, eval_idx in enumerate(fold_indices):
            train_mask = np.ones(n, dtype=bool)
            train_mask[eval_idx] = False
            train_idx = all_idx[train_mask]
            fold_seed = split_seed * 10 + fold_number + 1
            fold_result = evaluate_fold_example_1a(oracle, train_idx, eval_idx, data, fold_seed, ratio_mode)
            contributions_plugin[eval_idx] = np.asarray(fold_result["plugin"], dtype=float)
            contributions_if[eval_idx] = np.asarray(fold_result["if"], dtype=float)
            bellman_all[eval_idx] = np.asarray(fold_result["bellman"], dtype=float)
            reward_metrics = reward_grid_diagnostics(
                oracle=oracle,
                reward_grid=np.asarray(fold_result["reward_grid"], dtype=float),
                example_id="1a",
            )
            reward_rmse_grid_parts.append(reward_metrics["reward_rmse_grid"])
            reward_rmse_stationary_parts.append(reward_metrics["reward_rmse_stationary"])
            reward_rmse_rho_weighted_parts.append(reward_metrics["reward_rmse_rho_weighted"])
            ratio_parts.append(np.asarray(fold_result["ratio"], dtype=float))
            fold_fore = fold_result.get("fore_diagnostics", {})
            if fold_fore:
                fore_diagnostics_parts.append(dict(fold_fore))
        stabilized = (1.0 - lam) * contributions_plugin + lam * contributions_if
        plugin_split_estimates.append(float(np.mean(contributions_plugin)))
        stabilized_split_estimates.append(float(np.mean(stabilized)))
        centered = stabilized - np.mean(stabilized)
        stabilized_split_ses.append(float(np.std(centered, ddof=1) / math.sqrt(n)))
        bellman_values.append(bellman_all)

    plugin_estimate = float(np.mean(plugin_split_estimates))
    if_estimate = float(np.mean(stabilized_split_estimates))
    estimated_se = combine_repeated_split_se(stabilized_split_estimates, stabilized_split_ses)
    truth = oracle.truth_for_example("1a")
    ci_halfwidth = oracle.config.example1a_ci_critical_value * estimated_se
    ci_lower = if_estimate - ci_halfwidth
    ci_upper = if_estimate + ci_halfwidth
    ratio_values = np.concatenate(ratio_parts) if ratio_parts else np.array([np.nan])
    ratio_quantiles = weighted_quantiles(ratio_values, [0.01, 0.50, 0.99])
    bellman_concat = np.concatenate(bellman_values) if bellman_values else np.array([np.nan])
    result = SingleRunResult(
        example_id="1a",
        n=n,
        seed=seed,
        ratio_mode=ratio_mode,
        nuisance_method=f"{oracle.config.example1a_nuisance_method}-stabilized",
        plugin_estimate=plugin_estimate,
        if_estimate=if_estimate,
        truth=truth,
        estimated_se=estimated_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        covered=float(ci_lower <= truth <= ci_upper),
        plugin_error=plugin_estimate - truth,
        if_error=if_estimate - truth,
        reward_rmse=float(np.mean(reward_rmse_grid_parts)),
        reward_rmse_grid=float(np.mean(reward_rmse_grid_parts)),
        reward_rmse_stationary=float(np.mean(reward_rmse_stationary_parts)),
        reward_rmse_rho_weighted=float(np.mean(reward_rmse_rho_weighted_parts)),
        bellman_residual_rmse=float(np.sqrt(np.mean(bellman_concat**2))),
        q_nu_bellman_rmse=float("nan"),
        q_eval_bellman_rmse=float("nan"),
        ratio_q01=float(ratio_quantiles[0]),
        ratio_q50=float(ratio_quantiles[1]),
        ratio_q99=float(ratio_quantiles[2]),
        weight_ess=effective_sample_size(ratio_values),
        pi0_action0_q01=float("nan"),
        pi0_action0_q50=float("nan"),
        pi0_action0_q99=float("nan"),
        pi_ratio_q01=float("nan"),
        pi_ratio_q50=float("nan"),
        pi_ratio_q99=float("nan"),
        nu_ratio_q01=float("nan"),
        nu_ratio_q50=float("nan"),
        nu_ratio_q99=float("nan"),
    )
    return attach_fore_diagnostics(result, fore_diagnostics_parts)


def estimate_example_1b_stabilized(
    oracle: JRSSBOracle,
    data: Dict[str, np.ndarray],
    seed: int,
    ratio_mode: str = "oracle",
) -> SingleRunResult:
    n = data["states"].shape[0]
    repeated_splits = max(1, int(oracle.config.example1b_repeated_splits))
    plugin_split_estimates: List[float] = []
    if_split_estimates: List[float] = []
    if_split_ses: List[float] = []
    reward_rmse_grid_parts: List[float] = []
    reward_rmse_stationary_parts: List[float] = []
    reward_rmse_rho_weighted_parts: List[float] = []
    bellman_values: List[np.ndarray] = []
    ratio_parts: List[np.ndarray] = []
    fore_diagnostics_parts: List[Dict[str, float]] = []

    for split_number in range(repeated_splits):
        split_seed = seed + 100_003 * split_number
        fold_indices = fold_splits(n, split_seed, n_folds=oracle.config.crossfit_folds)
        all_idx = np.arange(n)
        contributions_plugin = np.zeros(n, dtype=float)
        contributions_if = np.zeros(n, dtype=float)
        bellman_all = np.zeros(n, dtype=float)
        for fold_number, eval_idx in enumerate(fold_indices):
            train_mask = np.ones(n, dtype=bool)
            train_mask[eval_idx] = False
            train_idx = all_idx[train_mask]
            fold_seed = split_seed * 10 + fold_number + 1
            fold_result = evaluate_fold_example_1b(oracle, train_idx, eval_idx, data, fold_seed, ratio_mode)
            contributions_plugin[eval_idx] = np.asarray(fold_result["plugin"], dtype=float)
            contributions_if[eval_idx] = np.asarray(fold_result["if"], dtype=float)
            bellman_all[eval_idx] = np.asarray(fold_result["bellman"], dtype=float)
            reward_metrics = reward_grid_diagnostics(
                oracle=oracle,
                reward_grid=np.asarray(fold_result["reward_grid"], dtype=float),
                example_id="1b",
            )
            reward_rmse_grid_parts.append(reward_metrics["reward_rmse_grid"])
            reward_rmse_stationary_parts.append(reward_metrics["reward_rmse_stationary"])
            reward_rmse_rho_weighted_parts.append(reward_metrics["reward_rmse_rho_weighted"])
            ratio_parts.append(np.asarray(fold_result["ratio"], dtype=float))
            fold_fore = fold_result.get("fore_diagnostics", {})
            if fold_fore:
                fore_diagnostics_parts.append(dict(fold_fore))
        plugin_split_estimates.append(float(np.mean(contributions_plugin)))
        if_split_estimates.append(float(np.mean(contributions_if)))
        centered = contributions_if - np.mean(contributions_if)
        if_split_ses.append(float(np.std(centered, ddof=1) / math.sqrt(n)))
        bellman_values.append(bellman_all)

    plugin_estimate = float(np.mean(plugin_split_estimates))
    if_estimate = float(np.mean(if_split_estimates))
    estimated_se = combine_repeated_split_se(if_split_estimates, if_split_ses)
    truth = oracle.truth_for_example("1b")
    ci_halfwidth = oracle.config.example1b_ci_critical_value * estimated_se
    ci_lower = if_estimate - ci_halfwidth
    ci_upper = if_estimate + ci_halfwidth
    ratio_values = np.concatenate(ratio_parts) if ratio_parts else np.array([np.nan])
    ratio_quantiles = weighted_quantiles(ratio_values, [0.01, 0.50, 0.99])
    bellman_concat = np.concatenate(bellman_values) if bellman_values else np.array([np.nan])
    result = SingleRunResult(
        example_id="1b",
        n=n,
        seed=seed,
        ratio_mode=ratio_mode,
        nuisance_method=f"{oracle.config.example1b_nuisance_method}-stabilized",
        plugin_estimate=plugin_estimate,
        if_estimate=if_estimate,
        truth=truth,
        estimated_se=estimated_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        covered=float(ci_lower <= truth <= ci_upper),
        plugin_error=plugin_estimate - truth,
        if_error=if_estimate - truth,
        reward_rmse=float(np.mean(reward_rmse_grid_parts)),
        reward_rmse_grid=float(np.mean(reward_rmse_grid_parts)),
        reward_rmse_stationary=float(np.mean(reward_rmse_stationary_parts)),
        reward_rmse_rho_weighted=float(np.mean(reward_rmse_rho_weighted_parts)),
        bellman_residual_rmse=float(np.sqrt(np.mean(bellman_concat**2))),
        q_nu_bellman_rmse=float("nan"),
        q_eval_bellman_rmse=float("nan"),
        ratio_q01=float(ratio_quantiles[0]),
        ratio_q50=float(ratio_quantiles[1]),
        ratio_q99=float(ratio_quantiles[2]),
        weight_ess=effective_sample_size(ratio_values),
        pi0_action0_q01=float("nan"),
        pi0_action0_q50=float("nan"),
        pi0_action0_q99=float("nan"),
        pi_ratio_q01=float("nan"),
        pi_ratio_q50=float("nan"),
        pi_ratio_q99=float("nan"),
        nu_ratio_q01=float("nan"),
        nu_ratio_q50=float("nan"),
        nu_ratio_q99=float("nan"),
    )
    return attach_fore_diagnostics(result, fore_diagnostics_parts)


def estimate_example(
    oracle: JRSSBOracle,
    data: Dict[str, np.ndarray],
    example_id: str,
    seed: int,
    ratio_mode: str = "oracle",
    nuisance_data: Optional[Dict[str, np.ndarray]] = None,
) -> SingleRunResult:
    if example_id == "1a" and oracle.config.example1a_repeated_splits > 1:
        return estimate_example_1a_stabilized(oracle=oracle, data=data, seed=seed, ratio_mode=ratio_mode)
    if example_id == "1b" and oracle.config.example1b_repeated_splits > 1:
        return estimate_example_1b_stabilized(oracle=oracle, data=data, seed=seed, ratio_mode=ratio_mode)
    if example_id == "2" and oracle.config.example2_repeated_splits > 1:
        return estimate_example_2_repeated(oracle=oracle, data=data, seed=seed, ratio_mode=ratio_mode)

    n = data["states"].shape[0]
    if nuisance_data is not None:
        combined = combine_transition_samples(nuisance_data, data)
        train_idx = np.arange(nuisance_data["states"].shape[0], dtype=int)
        eval_idx = np.arange(nuisance_data["states"].shape[0], nuisance_data["states"].shape[0] + n, dtype=int)
        fold_seed = seed * 10 + 1
        if example_id == "1a":
            fold_result = evaluate_fold_example_1a(oracle, train_idx, eval_idx, combined, fold_seed, ratio_mode)
        elif example_id == "1b":
            fold_result = evaluate_fold_example_1b(oracle, train_idx, eval_idx, combined, fold_seed, ratio_mode)
        elif example_id == "2":
            fold_result = evaluate_fold_example_2(oracle, train_idx, eval_idx, combined, fold_seed, ratio_mode)
        else:
            raise ValueError(f"Unknown example id: {example_id}")
        bellman_q_nu = (
            np.asarray(fold_result["bellman_nu"], dtype=float)
            if "bellman_nu" in fold_result
            else np.full(n, np.nan, dtype=float)
        )
        bellman_q_eval = (
            np.asarray(fold_result["bellman_eval"], dtype=float)
            if "bellman_eval" in fold_result
            else np.full(n, np.nan, dtype=float)
        )
        result = finalize_single_run_result(
            oracle=oracle,
            example_id=example_id,
            n=n,
            seed=seed,
            ratio_mode=ratio_mode,
            contributions_plugin=np.asarray(fold_result["plugin"], dtype=float),
            contributions_if=np.asarray(fold_result["if"], dtype=float),
            reward_rmse_grid_parts=[reward_grid_diagnostics(oracle, np.asarray(fold_result["reward_grid"], dtype=float), example_id)["reward_rmse_grid"]],
            reward_rmse_stationary_parts=[reward_grid_diagnostics(oracle, np.asarray(fold_result["reward_grid"], dtype=float), example_id)["reward_rmse_stationary"]],
            reward_rmse_rho_weighted_parts=[reward_grid_diagnostics(oracle, np.asarray(fold_result["reward_grid"], dtype=float), example_id)["reward_rmse_rho_weighted"]],
            bellman_all=np.asarray(fold_result["bellman"], dtype=float),
            bellman_q_nu_all=bellman_q_nu,
            bellman_q_eval_all=bellman_q_eval,
            ratio_parts=[np.asarray(fold_result["ratio"], dtype=float)],
            pi0_action0_parts=(
                [np.asarray(fold_result["pi0_action0"], dtype=float)] if "pi0_action0" in fold_result else []
            ),
            pi_ratio_parts=(
                [np.asarray(fold_result["pi_ratio"], dtype=float)] if "pi_ratio" in fold_result else []
            ),
            nu_ratio_parts=(
                [np.asarray(fold_result["nu_ratio"], dtype=float)] if "nu_ratio" in fold_result else []
            ),
            fold_indices=[np.arange(n, dtype=int)],
        )
        fold_fore = fold_result.get("fore_diagnostics", {})
        return attach_fore_diagnostics(result, [dict(fold_fore)] if fold_fore else [])

    contributions_plugin = np.zeros(n, dtype=float)
    contributions_if = np.zeros(n, dtype=float)
    bellman_all = np.zeros(n, dtype=float)
    bellman_q_nu_all = np.full(n, np.nan, dtype=float)
    bellman_q_eval_all = np.full(n, np.nan, dtype=float)
    reward_rmse_grid_parts: List[float] = []
    reward_rmse_stationary_parts: List[float] = []
    reward_rmse_rho_weighted_parts: List[float] = []
    ratio_parts: List[np.ndarray] = []
    pi0_action0_parts: List[np.ndarray] = []
    pi_ratio_parts: List[np.ndarray] = []
    nu_ratio_parts: List[np.ndarray] = []
    fore_diagnostics_parts: List[Dict[str, float]] = []
    fold_indices = fold_splits(n, seed, n_folds=oracle.config.crossfit_folds)
    all_idx = np.arange(n)

    for fold_number, eval_idx in enumerate(fold_indices):
        train_mask = np.ones(n, dtype=bool)
        train_mask[eval_idx] = False
        train_idx = all_idx[train_mask]
        fold_seed = seed * 10 + fold_number + 1
        if example_id == "1a":
            fold_result = evaluate_fold_example_1a(oracle, train_idx, eval_idx, data, fold_seed, ratio_mode)
        elif example_id == "1b":
            fold_result = evaluate_fold_example_1b(oracle, train_idx, eval_idx, data, fold_seed, ratio_mode)
        elif example_id == "2":
            fold_result = evaluate_fold_example_2(oracle, train_idx, eval_idx, data, fold_seed, ratio_mode)
        else:
            raise ValueError(f"Unknown example id: {example_id}")
        contributions_plugin[eval_idx] = np.asarray(fold_result["plugin"], dtype=float)
        contributions_if[eval_idx] = np.asarray(fold_result["if"], dtype=float)
        bellman_all[eval_idx] = np.asarray(fold_result["bellman"], dtype=float)
        if "bellman_nu" in fold_result:
            bellman_q_nu_all[eval_idx] = np.asarray(fold_result["bellman_nu"], dtype=float)
        if "bellman_eval" in fold_result:
            bellman_q_eval_all[eval_idx] = np.asarray(fold_result["bellman_eval"], dtype=float)
        reward_metrics = reward_grid_diagnostics(
            oracle=oracle,
            reward_grid=np.asarray(fold_result["reward_grid"], dtype=float),
            example_id=example_id,
        )
        reward_rmse_grid_parts.append(reward_metrics["reward_rmse_grid"])
        reward_rmse_stationary_parts.append(reward_metrics["reward_rmse_stationary"])
        reward_rmse_rho_weighted_parts.append(reward_metrics["reward_rmse_rho_weighted"])
        ratio_parts.append(np.asarray(fold_result["ratio"], dtype=float))
        if "pi0_action0" in fold_result:
            pi0_action0_parts.append(np.asarray(fold_result["pi0_action0"], dtype=float))
        if "pi_ratio" in fold_result:
            pi_ratio_parts.append(np.asarray(fold_result["pi_ratio"], dtype=float))
        if "nu_ratio" in fold_result:
            nu_ratio_parts.append(np.asarray(fold_result["nu_ratio"], dtype=float))
        fold_fore = fold_result.get("fore_diagnostics", {})
        if fold_fore:
            fore_diagnostics_parts.append(dict(fold_fore))
    result = finalize_single_run_result(
        oracle=oracle,
        example_id=example_id,
        n=n,
        seed=seed,
        ratio_mode=ratio_mode,
        contributions_plugin=contributions_plugin,
        contributions_if=contributions_if,
        reward_rmse_grid_parts=reward_rmse_grid_parts,
        reward_rmse_stationary_parts=reward_rmse_stationary_parts,
        reward_rmse_rho_weighted_parts=reward_rmse_rho_weighted_parts,
        bellman_all=bellman_all,
        bellman_q_nu_all=bellman_q_nu_all,
        bellman_q_eval_all=bellman_q_eval_all,
        ratio_parts=ratio_parts,
        pi0_action0_parts=pi0_action0_parts,
        pi_ratio_parts=pi_ratio_parts,
        nu_ratio_parts=nu_ratio_parts,
        fold_indices=fold_indices,
    )
    return attach_fore_diagnostics(result, fore_diagnostics_parts)


def estimate_example_2_repeated(
    oracle: JRSSBOracle,
    data: Dict[str, np.ndarray],
    seed: int,
    ratio_mode: str = "oracle",
) -> SingleRunResult:
    n = data["states"].shape[0]
    repeated_splits = max(1, int(oracle.config.example2_repeated_splits))
    plugin_split_estimates: List[float] = []
    if_split_estimates: List[float] = []
    if_split_ses: List[float] = []
    reward_rmse_grid_parts: List[float] = []
    reward_rmse_stationary_parts: List[float] = []
    reward_rmse_rho_weighted_parts: List[float] = []
    bellman_values: List[np.ndarray] = []
    bellman_q_nu_values: List[np.ndarray] = []
    bellman_q_eval_values: List[np.ndarray] = []
    ratio_parts: List[np.ndarray] = []
    pi0_action0_parts: List[np.ndarray] = []
    pi_ratio_parts: List[np.ndarray] = []
    nu_ratio_parts: List[np.ndarray] = []

    for split_number in range(repeated_splits):
        split_seed = seed + 100_003 * split_number
        fold_indices = fold_splits(n, split_seed, n_folds=oracle.config.crossfit_folds)
        all_idx = np.arange(n)
        contributions_plugin = np.zeros(n, dtype=float)
        contributions_if = np.zeros(n, dtype=float)
        bellman_all = np.zeros(n, dtype=float)
        bellman_q_nu_all = np.full(n, np.nan, dtype=float)
        bellman_q_eval_all = np.full(n, np.nan, dtype=float)
        for fold_number, eval_idx in enumerate(fold_indices):
            train_mask = np.ones(n, dtype=bool)
            train_mask[eval_idx] = False
            train_idx = all_idx[train_mask]
            fold_seed = split_seed * 10 + fold_number + 1
            fold_result = evaluate_fold_example_2(oracle, train_idx, eval_idx, data, fold_seed, ratio_mode)
            contributions_plugin[eval_idx] = np.asarray(fold_result["plugin"], dtype=float)
            contributions_if[eval_idx] = np.asarray(fold_result["if"], dtype=float)
            bellman_all[eval_idx] = np.asarray(fold_result["bellman"], dtype=float)
            if "bellman_nu" in fold_result:
                bellman_q_nu_all[eval_idx] = np.asarray(fold_result["bellman_nu"], dtype=float)
            if "bellman_eval" in fold_result:
                bellman_q_eval_all[eval_idx] = np.asarray(fold_result["bellman_eval"], dtype=float)
            reward_metrics = reward_grid_diagnostics(
                oracle=oracle,
                reward_grid=np.asarray(fold_result["reward_grid"], dtype=float),
                example_id="2",
            )
            reward_rmse_grid_parts.append(reward_metrics["reward_rmse_grid"])
            reward_rmse_stationary_parts.append(reward_metrics["reward_rmse_stationary"])
            reward_rmse_rho_weighted_parts.append(reward_metrics["reward_rmse_rho_weighted"])
            ratio_parts.append(np.asarray(fold_result["ratio"], dtype=float))
            pi0_action0_parts.append(np.asarray(fold_result["pi0_action0"], dtype=float))
            pi_ratio_parts.append(np.asarray(fold_result["pi_ratio"], dtype=float))
            nu_ratio_parts.append(np.asarray(fold_result["nu_ratio"], dtype=float))
        plugin_split_estimates.append(float(np.mean(contributions_plugin)))
        if_split_estimates.append(float(np.mean(contributions_if)))
        centered = contributions_if - np.mean(contributions_if)
        if_split_ses.append(float(np.std(centered, ddof=1) / math.sqrt(n)))
        bellman_values.append(bellman_all)
        bellman_q_nu_values.append(bellman_q_nu_all)
        bellman_q_eval_values.append(bellman_q_eval_all)

    plugin_estimate = float(np.mean(plugin_split_estimates))
    if_estimate = float(np.mean(if_split_estimates))
    estimated_se = combine_repeated_split_se(if_split_estimates, if_split_ses)
    ci_halfwidth = oracle.config.example2_ci_critical_value * estimated_se
    ci_lower = if_estimate - ci_halfwidth
    ci_upper = if_estimate + ci_halfwidth
    truth = oracle.truth_for_example("2")
    ratio_values = np.concatenate(ratio_parts) if ratio_parts else np.array([np.nan])
    ratio_quantiles = weighted_quantiles(ratio_values, [0.01, 0.50, 0.99])
    pi0_action0_values = np.concatenate(pi0_action0_parts) if pi0_action0_parts else np.array([np.nan])
    pi_ratio_values = np.concatenate(pi_ratio_parts) if pi_ratio_parts else np.array([np.nan])
    nu_ratio_values = np.concatenate(nu_ratio_parts) if nu_ratio_parts else np.array([np.nan])
    pi0_quantiles = weighted_quantiles(pi0_action0_values, [0.01, 0.50, 0.99])
    pi_ratio_quantiles = weighted_quantiles(pi_ratio_values, [0.01, 0.50, 0.99])
    nu_ratio_quantiles = weighted_quantiles(nu_ratio_values, [0.01, 0.50, 0.99])
    bellman_concat = np.concatenate(bellman_values) if bellman_values else np.array([np.nan])
    bellman_q_nu_concat = np.concatenate(bellman_q_nu_values) if bellman_q_nu_values else np.array([np.nan])
    bellman_q_eval_concat = np.concatenate(bellman_q_eval_values) if bellman_q_eval_values else np.array([np.nan])
    return SingleRunResult(
        example_id="2",
        n=n,
        seed=seed,
        ratio_mode=ratio_mode,
        nuisance_method=oracle.config.example2_nuisance_method,
        plugin_estimate=plugin_estimate,
        if_estimate=if_estimate,
        truth=truth,
        estimated_se=estimated_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        covered=float(ci_lower <= truth <= ci_upper),
        plugin_error=plugin_estimate - truth,
        if_error=if_estimate - truth,
        reward_rmse=float(np.mean(reward_rmse_grid_parts)),
        reward_rmse_grid=float(np.mean(reward_rmse_grid_parts)),
        reward_rmse_stationary=float(np.mean(reward_rmse_stationary_parts)),
        reward_rmse_rho_weighted=float(np.mean(reward_rmse_rho_weighted_parts)),
        bellman_residual_rmse=float(np.sqrt(np.mean(bellman_concat**2))),
        q_nu_bellman_rmse=float(np.sqrt(np.nanmean(bellman_q_nu_concat**2))) if np.any(~np.isnan(bellman_q_nu_concat)) else float("nan"),
        q_eval_bellman_rmse=float(np.sqrt(np.nanmean(bellman_q_eval_concat**2))) if np.any(~np.isnan(bellman_q_eval_concat)) else float("nan"),
        ratio_q01=float(ratio_quantiles[0]),
        ratio_q50=float(ratio_quantiles[1]),
        ratio_q99=float(ratio_quantiles[2]),
        weight_ess=effective_sample_size(ratio_values),
        pi0_action0_q01=float(pi0_quantiles[0]),
        pi0_action0_q50=float(pi0_quantiles[1]),
        pi0_action0_q99=float(pi0_quantiles[2]),
        pi_ratio_q01=float(pi_ratio_quantiles[0]),
        pi_ratio_q50=float(pi_ratio_quantiles[1]),
        pi_ratio_q99=float(pi_ratio_quantiles[2]),
        nu_ratio_q01=float(nu_ratio_quantiles[0]),
        nu_ratio_q50=float(nu_ratio_quantiles[1]),
        nu_ratio_q99=float(nu_ratio_quantiles[2]),
    )


def run_single_replication(
    oracle: JRSSBOracle,
    n: int,
    seed: int,
    example_id: str,
    ratio_mode: str = "neural-fore",
) -> SingleRunResult:
    data = oracle.sample_stationary_transitions(n=n, seed=seed)
    nuisance_data = None
    if oracle.config.nuisance_sample_mode == "independent":
        nuisance_data = oracle.sample_stationary_transitions(n=n, seed=seed + 1_000_003)
    return estimate_example(
        oracle=oracle,
        data=data,
        example_id=example_id,
        seed=seed,
        ratio_mode=ratio_mode,
        nuisance_data=nuisance_data,
    )


def _recordable_estimation_failure(exc: Exception) -> bool:
    """Distinguish numerical fit failures from programming/configuration errors."""
    if isinstance(exc, (FloatingPointError, OverflowError, np.linalg.LinAlgError)):
        return True
    message = str(exc).lower()
    numerical_markers = (
        "nonfinite",
        "non-finite",
        "nan",
        "infinite",
        "numerical",
        "singular",
    )
    return isinstance(exc, (RuntimeError, ValueError)) and any(
        marker in message for marker in numerical_markers
    )


def failed_single_run_result(
    *,
    oracle: JRSSBOracle,
    n: int,
    seed: int,
    example_id: str,
    ratio_mode: str,
    exc: Exception,
) -> SingleRunResult:
    """Return a labeled failure row so a confirmatory cell can continue."""
    truth = {
        "1a": oracle.psi_1a,
        "1b": oracle.psi_1b,
        "2": oracle.psi_2,
    }[example_id]
    nan = float("nan")
    return SingleRunResult(
        example_id=example_id,
        n=n,
        seed=seed,
        ratio_mode=ratio_mode,
        nuisance_method=nuisance_method_name(oracle, example_id),
        plugin_estimate=nan,
        if_estimate=nan,
        truth=float(truth),
        estimated_se=nan,
        ci_lower=nan,
        ci_upper=nan,
        covered=nan,
        plugin_error=nan,
        if_error=nan,
        reward_rmse=nan,
        reward_rmse_grid=nan,
        reward_rmse_stationary=nan,
        reward_rmse_rho_weighted=nan,
        bellman_residual_rmse=nan,
        q_nu_bellman_rmse=nan,
        q_eval_bellman_rmse=nan,
        ratio_q01=nan,
        ratio_q50=nan,
        ratio_q99=nan,
        weight_ess=nan,
        pi0_action0_q01=nan,
        pi0_action0_q50=nan,
        pi0_action0_q99=nan,
        pi_ratio_q01=nan,
        pi_ratio_q50=nan,
        pi_ratio_q99=nan,
        nu_ratio_q01=nan,
        nu_ratio_q50=nan,
        nu_ratio_q99=nan,
        ratio_failure=1.0,
        failure_message=f"{type(exc).__name__}: {exc}",
    )


def run_single_replication_safe(
    oracle: JRSSBOracle,
    n: int,
    seed: int,
    example_id: str,
    ratio_mode: str,
) -> SingleRunResult:
    """Run one replication while recording only genuine numerical failures."""
    try:
        return run_single_replication(
            oracle=oracle,
            n=n,
            seed=seed,
            example_id=example_id,
            ratio_mode=ratio_mode,
        )
    except Exception as exc:
        if not _recordable_estimation_failure(exc):
            raise
        return failed_single_run_result(
            oracle=oracle,
            n=n,
            seed=seed,
            example_id=example_id,
            ratio_mode=ratio_mode,
            exc=exc,
        )


def run_monte_carlo(
    oracle: JRSSBOracle,
    sample_sizes: Optional[Sequence[int]] = None,
    repetitions: Optional[int] = None,
    example_ids: Sequence[str] = ("1a", "1b"),
    ratio_mode: str = "neural-fore",
    jobs: int = 1,
    seed_offset: int = 0,
    replication_indices: Optional[Sequence[int]] = None,
    on_result: Optional[Callable[[SingleRunResult], None]] = None,
) -> List[SingleRunResult]:
    """Run requested replications, optionally checkpointing each parent-side result."""
    sample_sizes = tuple(oracle.config.mc_sample_sizes if sample_sizes is None else sample_sizes)
    repetitions = oracle.config.mc_repetitions if repetitions is None else repetitions
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    if jobs <= 0:
        raise ValueError("jobs must be positive.")
    if replication_indices is None:
        requested_replications = tuple(range(repetitions))
    else:
        requested_replications = tuple(int(index) for index in replication_indices)
        if len(set(requested_replications)) != len(requested_replications):
            raise ValueError("replication_indices must be unique.")
        if any(index < 0 or index >= repetitions for index in requested_replications):
            raise ValueError("replication_indices must lie in [0, repetitions).")
    example_offsets = {"1a": 101, "1b": 202, "2": 303}
    tasks: List[tuple[int, int, str, str]] = []
    for example_id in example_ids:
        for n in sample_sizes:
            for rep in requested_replications:
                # Keep the generated dataset fixed across ratio estimators.
                # Estimator-specific seed offsets invalidate paired nuisance
                # audits and can badly distort small Monte Carlo comparisons.
                seed = (
                    10_000 * (rep + 1)
                    + 97 * n
                    + example_offsets[example_id]
                    + int(seed_offset)
                )
                tasks.append((n, seed, example_id, ratio_mode))

    def collect(iterator: Iterable[SingleRunResult]) -> List[SingleRunResult]:
        collected: List[SingleRunResult] = []
        for result in iterator:
            collected.append(result)
            if on_result is not None:
                on_result(result)
        return collected

    if jobs <= 1:
        return collect(
            run_single_replication_safe(oracle=oracle, n=n, seed=seed, example_id=example_id, ratio_mode=ratio_mode)
            for n, seed, example_id, ratio_mode in tasks
        )
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_init_monte_carlo_worker,
            initargs=(oracle.config,),
        ) as executor:
            return collect(executor.map(_run_single_replication_worker, tasks))
    except (OSError, PermissionError):
        return collect(
            run_single_replication_safe(
                oracle=oracle,
                n=n,
                seed=seed,
                example_id=example_id,
                ratio_mode=ratio_mode,
            )
            for n, seed, example_id, ratio_mode in tasks
        )


def summarize_results(results: Sequence[SingleRunResult]) -> List[Dict[str, float | str | int]]:
    grouped: Dict[tuple[str, int, str, str], List[SingleRunResult]] = {}
    for result in results:
        key = (result.example_id, result.n, result.ratio_mode, result.nuisance_method)
        grouped.setdefault(key, []).append(result)

    summary: List[Dict[str, float | str | int]] = []
    for (example_id, n, ratio_mode, nuisance_method), all_rows in sorted(grouped.items()):
        reps = len(all_rows)
        rows = [
            row
            for row in all_rows
            if row.ratio_failure == 0.0
            and np.isfinite(row.plugin_estimate)
            and np.isfinite(row.if_estimate)
        ]
        successful_reps = len(rows)
        plugin_errors = np.array([row.plugin_error for row in rows], dtype=float)
        if_errors = np.array([row.if_error for row in rows], dtype=float)
        estimated_ses = np.array([row.estimated_se for row in rows], dtype=float)
        iid_ses = np.array([row.iid_estimated_se for row in rows], dtype=float)
        fold_cluster_ses = np.array(
            [row.fold_cluster_estimated_se for row in rows], dtype=float
        )
        critical_values = np.array(
            [row.ci_critical_value for row in rows], dtype=float
        )
        coverages = np.array([row.covered for row in rows], dtype=float)
        reward_rmses = np.array([row.reward_rmse for row in rows], dtype=float)
        reward_rmses_grid = np.array([row.reward_rmse_grid for row in rows], dtype=float)
        reward_rmses_stationary = np.array([row.reward_rmse_stationary for row in rows], dtype=float)
        reward_rmses_rho_weighted = np.array([row.reward_rmse_rho_weighted for row in rows], dtype=float)
        bellman_rmses = np.array([row.bellman_residual_rmse for row in rows], dtype=float)
        bellman_nu_rmses = np.array([row.q_nu_bellman_rmse for row in rows], dtype=float)
        bellman_eval_rmses = np.array([row.q_eval_bellman_rmse for row in rows], dtype=float)
        ci_lengths = np.array([row.ci_upper - row.ci_lower for row in rows], dtype=float)
        plugin_estimates = np.array([row.plugin_estimate for row in rows], dtype=float)
        if_estimates = np.array([row.if_estimate for row in rows], dtype=float)
        pi0_q01 = np.array([row.pi0_action0_q01 for row in rows], dtype=float)
        pi0_q50 = np.array([row.pi0_action0_q50 for row in rows], dtype=float)
        pi0_q99 = np.array([row.pi0_action0_q99 for row in rows], dtype=float)
        pi_ratio_q99 = np.array([row.pi_ratio_q99 for row in rows], dtype=float)
        nu_ratio_q99 = np.array([row.nu_ratio_q99 for row in rows], dtype=float)
        fore_iterations = np.array([row.fore_selected_iterations for row in rows], dtype=float)
        fore_scores = np.array([row.fore_apbv_score for row in rows], dtype=float)
        fore_seconds = np.array([row.fore_fit_seconds for row in rows], dtype=float)
        fore_training_rows = np.array([row.fore_training_rows for row in rows], dtype=float)
        fore_mass = np.array([row.fore_normalized_mass for row in rows], dtype=float)
        fore_cap = np.array([row.fore_logit_cap_fraction for row in rows], dtype=float)
        signed_positive_iterations = np.array([row.signed_positive_iterations for row in rows], dtype=float)
        signed_negative_iterations = np.array([row.signed_negative_iterations for row in rows], dtype=float)
        signed_positive_scores = np.array([row.signed_positive_apbv_score for row in rows], dtype=float)
        signed_negative_scores = np.array([row.signed_negative_apbv_score for row in rows], dtype=float)
        signed_positive_seconds = np.array([row.signed_positive_fit_seconds for row in rows], dtype=float)
        signed_negative_seconds = np.array([row.signed_negative_fit_seconds for row in rows], dtype=float)
        signed_positive_normalized_mass = np.array(
            [row.signed_positive_normalized_mass for row in rows], dtype=float
        )
        signed_negative_normalized_mass = np.array(
            [row.signed_negative_normalized_mass for row in rows], dtype=float
        )
        signed_positive_ess = np.array(
            [row.signed_positive_ess for row in rows], dtype=float
        )
        signed_negative_ess = np.array(
            [row.signed_negative_ess for row in rows], dtype=float
        )
        signed_positive_q99 = np.array(
            [row.signed_positive_ratio_q99 for row in rows], dtype=float
        )
        signed_negative_q99 = np.array(
            [row.signed_negative_ratio_q99 for row in rows], dtype=float
        )
        signed_positive_cap = np.array(
            [row.signed_positive_logit_cap_fraction for row in rows], dtype=float
        )
        signed_negative_cap = np.array(
            [row.signed_negative_logit_cap_fraction for row in rows], dtype=float
        )
        plugin_sd = float(np.std(plugin_estimates, ddof=1)) if successful_reps > 1 else float("nan")
        if_sd = float(np.std(if_estimates, ddof=1)) if successful_reps > 1 else float("nan")
        summary.append(
            {
                "example_id": example_id,
                "n": n,
                "ratio_mode": ratio_mode,
                "nuisance_method": nuisance_method,
                "repetitions": reps,
                "successful_repetitions": successful_reps,
                "truth": all_rows[0].truth,
                "plugin_bias": safe_nanmean(plugin_errors),
                "plugin_sd": plugin_sd,
                "plugin_rmse": float(np.sqrt(safe_nanmean(plugin_errors**2))),
                "if_bias": safe_nanmean(if_errors),
                "if_sd": if_sd,
                "if_rmse": float(np.sqrt(safe_nanmean(if_errors**2))),
                "avg_estimated_se": safe_nanmean(estimated_ses),
                "avg_iid_estimated_se": safe_nanmean(iid_ses),
                "avg_fold_cluster_estimated_se": safe_nanmean(fold_cluster_ses),
                "avg_ci_critical_value": safe_nanmean(critical_values),
                "coverage_95": safe_nanmean(coverages),
                "avg_ci_length": safe_nanmean(ci_lengths),
                "avg_reward_rmse": safe_nanmean(reward_rmses),
                "avg_reward_rmse_grid": safe_nanmean(reward_rmses_grid),
                "avg_reward_rmse_stationary": safe_nanmean(reward_rmses_stationary),
                "avg_reward_rmse_rho_weighted": safe_nanmean(reward_rmses_rho_weighted),
                "avg_bellman_rmse": safe_nanmean(bellman_rmses),
                "avg_q_nu_bellman_rmse": safe_nanmean(bellman_nu_rmses),
                "avg_q_eval_bellman_rmse": safe_nanmean(bellman_eval_rmses),
                "avg_pi0_action0_q01": safe_nanmean(pi0_q01),
                "avg_pi0_action0_q50": safe_nanmean(pi0_q50),
                "avg_pi0_action0_q99": safe_nanmean(pi0_q99),
                "avg_pi_ratio_q99": safe_nanmean(pi_ratio_q99),
                "avg_nu_ratio_q99": safe_nanmean(nu_ratio_q99),
                "avg_fore_selected_iterations": safe_nanmean(fore_iterations),
                "avg_fore_apbv_score": safe_nanmean(fore_scores),
                "avg_fore_fit_seconds": safe_nanmean(fore_seconds),
                "avg_fore_training_rows": safe_nanmean(fore_training_rows),
                "avg_fore_normalized_mass": safe_nanmean(fore_mass),
                "avg_fore_logit_cap_fraction": safe_nanmean(fore_cap),
                "avg_signed_positive_iterations": safe_nanmean(signed_positive_iterations),
                "avg_signed_negative_iterations": safe_nanmean(signed_negative_iterations),
                "avg_signed_positive_apbv_score": safe_nanmean(signed_positive_scores),
                "avg_signed_negative_apbv_score": safe_nanmean(signed_negative_scores),
                "avg_signed_positive_fit_seconds": safe_nanmean(signed_positive_seconds),
                "avg_signed_negative_fit_seconds": safe_nanmean(signed_negative_seconds),
                "avg_signed_positive_normalized_mass": safe_nanmean(
                    signed_positive_normalized_mass
                ),
                "avg_signed_negative_normalized_mass": safe_nanmean(
                    signed_negative_normalized_mass
                ),
                "avg_signed_positive_ess": safe_nanmean(signed_positive_ess),
                "avg_signed_negative_ess": safe_nanmean(signed_negative_ess),
                "avg_signed_positive_ratio_q99": safe_nanmean(signed_positive_q99),
                "avg_signed_negative_ratio_q99": safe_nanmean(signed_negative_q99),
                "avg_signed_positive_logit_cap_fraction": safe_nanmean(
                    signed_positive_cap
                ),
                "avg_signed_negative_logit_cap_fraction": safe_nanmean(
                    signed_negative_cap
                ),
                "ratio_failure_rate": float(
                    np.mean([row.ratio_failure for row in all_rows])
                ),
            }
        )
    return summary


def save_results_csv(results: Sequence[SingleRunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row.as_dict() for row in results]
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(summary_rows: Sequence[Dict[str, float | str | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(summary_rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def summarize_audit_rows(
    rows: Sequence[Dict[str, float | str | int]],
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> List[Dict[str, float | str | int]]:
    grouped: Dict[tuple[object, ...], List[Dict[str, float | str | int]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(dict(row))
    summary_rows: List[Dict[str, float | str | int]] = []
    for key, group_rows in sorted(grouped.items()):
        summary_row: Dict[str, float | str | int] = {
            group_key: group_value for group_key, group_value in zip(group_keys, key)
        }
        summary_row["repetitions"] = len(group_rows)
        for metric_key in metric_keys:
            values = np.array([float(row[metric_key]) for row in group_rows], dtype=float)
            summary_row[metric_key] = float(np.mean(values))
            summary_row[f"{metric_key}_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
        summary_rows.append(summary_row)
    return summary_rows


def run_example1a_policy_audit(
    base_config: Optional[JRSSBConfig] = None,
    estimators: Sequence[str] = EXAMPLE1A_POLICY_ESTIMATORS,
    sample_sizes: Sequence[int] = (5_000, 10_000),
    repetitions: int = 10,
) -> Dict[str, object]:
    base_config = JRSSBConfig() if base_config is None else base_config
    rows: List[Dict[str, float | str | int]] = []
    for estimator in estimators:
        oracle = JRSSBOracle(replace(base_config, example1a_policy_estimator=str(estimator)))
        for n in sample_sizes:
            for rep in range(repetitions):
                seed = 60_000 * (rep + 1) + 89 * int(n) + 503
                data = oracle.sample_stationary_transitions(n=int(n), seed=seed)
                policy_hat = fit_behavior_policy_example1a(
                    oracle=oracle,
                    states=data["states"],
                    actions=data["actions"],
                    seed=seed,
                )
                reward_grid = np.log(np.clip(policy_hat.predict_proba(oracle.main_grid.states), EPS, None))
                reward_metrics = reward_grid_diagnostics(oracle=oracle, reward_grid=reward_grid, example_id="1a")
                rows.append(
                    {
                        "policy_estimator": str(estimator),
                        "n": int(n),
                        "seed": seed,
                        **reward_metrics,
                    }
                )
    summary_rows = summarize_audit_rows(
        rows,
        group_keys=("policy_estimator", "n"),
        metric_keys=("reward_rmse_grid", "reward_rmse_stationary", "reward_rmse_rho_weighted"),
    )
    return {"rows": rows, "summary": summary_rows}


def run_example1b_policy_audit(
    base_config: Optional[JRSSBConfig] = None,
    estimators: Sequence[str] = EXAMPLE1B_POLICY_ESTIMATORS,
    sample_sizes: Sequence[int] = (5_000, 10_000),
    repetitions: int = 10,
) -> Dict[str, object]:
    base_config = JRSSBConfig() if base_config is None else base_config
    soft_policy_self_check = oracle_soft_policy_self_check(JRSSBOracle(base_config))
    rows: List[Dict[str, float | str | int]] = []
    for estimator in estimators:
        oracle = JRSSBOracle(replace(base_config, example1b_policy_estimator=str(estimator)))
        for n in sample_sizes:
            for rep in range(repetitions):
                seed = 70_000 * (rep + 1) + 97 * int(n) + 1_001
                data = oracle.sample_stationary_transitions(n=int(n), seed=seed)
                policy_hat = fit_behavior_policy_example1b(
                    oracle=oracle,
                    states=data["states"],
                    actions=data["actions"],
                    seed=seed,
                )
                reward_grid = np.log(np.clip(policy_hat.predict_proba(oracle.main_grid.states), EPS, None))
                _, _, pi_star_grid = fit_soft_value_from_reward(
                    oracle,
                    reward_grid=reward_grid,
                    tau=oracle.config.tau_star,
                    allowed_mask=oracle.allowed_mask,
                )
                _, v_true_policy = oracle.evaluate_policy(oracle.r0, pi_star_grid, oracle.config.gamma_behavior)
                reward_metrics = reward_grid_diagnostics(oracle=oracle, reward_grid=reward_grid, example_id="1b")
                rows.append(
                    {
                        "policy_estimator": str(estimator),
                        "n": int(n),
                        "seed": seed,
                        **reward_metrics,
                        "pi_star_l1": float(np.mean(np.sum(np.abs(pi_star_grid - oracle.pi_star), axis=1))),
                        "pi_star_kl": mean_action_kl(oracle.pi_star, pi_star_grid),
                        "psi_target_shift": float(np.dot(oracle.stationary_behavior, v_true_policy) - oracle.psi_1b),
                    }
                )
    summary_rows = summarize_audit_rows(
        rows,
        group_keys=("policy_estimator", "n"),
        metric_keys=(
            "reward_rmse_grid",
            "reward_rmse_stationary",
            "reward_rmse_rho_weighted",
            "pi_star_l1",
            "pi_star_kl",
            "psi_target_shift",
        ),
    )
    return {"rows": rows, "summary": summary_rows, "soft_policy_self_check": soft_policy_self_check}


def run_example1a_large_sample_decomposition(
    base_config: Optional[JRSSBConfig] = None,
    estimators: Sequence[str] = ("maxent", "blend"),
    sample_sizes: Sequence[int] = (5_000, 10_000),
    repetitions: int = 5,
    eval_n: int = 50_000,
) -> Dict[str, object]:
    base_config = JRSSBConfig() if base_config is None else base_config
    rows: List[Dict[str, float | str | int]] = []
    for estimator in estimators:
        oracle = JRSSBOracle(replace(base_config, example1a_policy_estimator=str(estimator)))
        for n in sample_sizes:
            for rep in range(repetitions):
                seed = 81_000 * (rep + 1) + 173 * int(n) + 211
                nuisance = oracle.sample_stationary_transitions(n=int(n), seed=seed)
                eval_data = oracle.sample_stationary_transitions(n=int(eval_n), seed=seed + 1_000_003)
                policy_hat = fit_behavior_policy_example1a(
                    oracle=oracle,
                    states=nuisance["states"],
                    actions=nuisance["actions"],
                    seed=seed,
                )
                reward_grid = np.log(np.clip(policy_hat.predict_proba(oracle.main_grid.states), EPS, None))
                reward_metrics = reward_grid_diagnostics(oracle=oracle, reward_grid=reward_grid, example_id="1a")
                q_hat_grid, v_hat_grid = oracle.evaluate_policy(
                    reward_grid,
                    oracle.pi_fix,
                    gamma=oracle.config.gamma_behavior,
                    init_q=oracle.q_1a + (reward_grid - oracle.r0),
                    tol=oracle.config.nuisance_bellman_tol,
                    max_iterations=oracle.config.nuisance_max_iterations,
                )
                states_eval = eval_data["states"]
                next_states_eval = eval_data["next_states"]
                actions_eval = eval_data["actions"]
                idx = np.arange(actions_eval.shape[0], dtype=int)
                probs_eval = policy_hat.predict_proba(states_eval)
                q_sa_hat = oracle.action_values(states_eval, q_hat_grid)[idx, actions_eval]
                v_hat = oracle.state_values(states_eval, v_hat_grid)
                v_next_hat = oracle.state_values(next_states_eval, v_hat_grid)
                q_sa_oracle = oracle.action_values(states_eval, oracle.q_1a)[idx, actions_eval]
                v_oracle = oracle.state_values(states_eval, oracle.v_1a)
                v_next_oracle = oracle.state_values(next_states_eval, oracle.v_1a)
                r_obs = np.log(np.clip(probs_eval[idx, actions_eval], EPS, None))
                rho_eval = oracle.state_values(states_eval, oracle.rho_fix)
                pi_fix_eval = oracle._fixed_policy_probs(states_eval)
                pi_ratio = pi_fix_eval[idx, actions_eval] / np.clip(probs_eval[idx, actions_eval], EPS, None)
                plugin_est = float(np.mean(v_hat))
                if_est = float(
                    np.mean(
                        v_hat
                        + rho_eval * pi_ratio * (r_obs + oracle.config.gamma_behavior * v_next_hat - q_sa_hat)
                        + rho_eval * (pi_ratio - 1.0)
                    )
                )
                if_oracle_q_est = float(
                    np.mean(
                        v_oracle
                        + rho_eval * pi_ratio * (r_obs + oracle.config.gamma_behavior * v_next_oracle - q_sa_oracle)
                        + rho_eval * (pi_ratio - 1.0)
                    )
                )
                rows.append(
                    {
                        "policy_estimator": str(estimator),
                        "n": int(n),
                        "seed": seed,
                        **reward_metrics,
                        "plugin_bias_large_n": plugin_est - oracle.psi_1a,
                        "if_bias_large_n": if_est - oracle.psi_1a,
                        "if_bias_oracle_q_large_n": if_oracle_q_est - oracle.psi_1a,
                    }
                )
    summary_rows = summarize_audit_rows(
        rows,
        group_keys=("policy_estimator", "n"),
        metric_keys=(
            "reward_rmse_grid",
            "reward_rmse_stationary",
            "reward_rmse_rho_weighted",
            "plugin_bias_large_n",
            "if_bias_large_n",
            "if_bias_oracle_q_large_n",
        ),
    )
    return {"rows": rows, "summary": summary_rows}


def run_example1b_large_sample_decomposition(
    base_config: Optional[JRSSBConfig] = None,
    estimators: Sequence[str] = ("structural-linear", "structural"),
    sample_sizes: Sequence[int] = (5_000, 10_000),
    repetitions: int = 5,
    eval_n: int = 50_000,
) -> Dict[str, object]:
    base_config = JRSSBConfig() if base_config is None else base_config
    rows: List[Dict[str, float | str | int]] = []
    for estimator in estimators:
        oracle = JRSSBOracle(replace(base_config, example1b_policy_estimator=str(estimator)))
        for n in sample_sizes:
            for rep in range(repetitions):
                seed = 91_000 * (rep + 1) + 179 * int(n) + 307
                nuisance = oracle.sample_stationary_transitions(n=int(n), seed=seed)
                eval_data = oracle.sample_stationary_transitions(n=int(eval_n), seed=seed + 1_000_003)
                policy_hat = fit_behavior_policy_example1b(
                    oracle=oracle,
                    states=nuisance["states"],
                    actions=nuisance["actions"],
                    seed=seed,
                )
                reward_grid = np.log(np.clip(policy_hat.predict_proba(oracle.main_grid.states), EPS, None))
                reward_metrics = reward_grid_diagnostics(oracle=oracle, reward_grid=reward_grid, example_id="1b")
                continuation_grid, _, pi_star_grid = fit_soft_value_from_reward(
                    oracle,
                    reward_grid=reward_grid,
                    tau=oracle.config.tau_star,
                    allowed_mask=oracle.allowed_mask,
                )
                q_hat_grid, v_hat_grid = oracle.evaluate_policy(
                    reward_grid,
                    pi_star_grid,
                    gamma=oracle.config.gamma_behavior,
                    init_q=oracle.q_1b,
                    tol=oracle.config.nuisance_bellman_tol,
                    max_iterations=oracle.config.nuisance_max_iterations,
                )
                _, v_true_policy = oracle.evaluate_policy(oracle.r0, pi_star_grid, oracle.config.gamma_behavior)
                states_eval = eval_data["states"]
                next_states_eval = eval_data["next_states"]
                actions_eval = eval_data["actions"]
                idx = np.arange(actions_eval.shape[0], dtype=int)
                probs_eval = policy_hat.predict_proba(states_eval)
                pi_star_eval = oracle.policy_probs(states_eval, pi_star_grid)
                q_sa_hat = oracle.action_values(states_eval, q_hat_grid)[idx, actions_eval]
                v_hat = oracle.state_values(states_eval, v_hat_grid)
                v_next_hat = oracle.state_values(next_states_eval, v_hat_grid)
                q_sa_oracle = oracle.action_values(states_eval, oracle.q_1b)[idx, actions_eval]
                v_oracle = oracle.state_values(states_eval, oracle.v_1b)
                v_next_oracle = oracle.state_values(next_states_eval, oracle.v_1b)
                continuation_eval = oracle.action_values(states_eval, continuation_grid)
                continuation_next = oracle.action_values(next_states_eval, continuation_grid)
                continuation_oracle = oracle.action_values(states_eval, oracle.v_star)
                continuation_next_oracle = oracle.action_values(next_states_eval, oracle.v_star)
                reward_next = oracle.action_values(next_states_eval, reward_grid)
                reward_next_oracle = oracle.action_values(next_states_eval, oracle.r0)
                soft_next = oracle.config.tau_star * logsumexp(
                    (reward_next + oracle.config.gamma_behavior * continuation_next) / max(oracle.config.tau_star, EPS)
                    + np.where(oracle.allowed_mask[None, :], 0.0, -1e12),
                    axis=1,
                )
                soft_next_oracle = oracle.config.tau_star * logsumexp(
                    (reward_next_oracle + oracle.config.gamma_behavior * continuation_next_oracle) / max(oracle.config.tau_star, EPS)
                    + np.where(oracle.allowed_mask[None, :], 0.0, -1e12),
                    axis=1,
                )
                r_obs = np.log(np.clip(probs_eval[idx, actions_eval], EPS, None))
                rho_eval = oracle.state_values(states_eval, oracle.rho_star)
                tilde_rho_eval = oracle.state_values(states_eval, oracle.tilde_rho_star)
                pi_ratio = pi_star_eval[idx, actions_eval] / np.clip(probs_eval[idx, actions_eval], EPS, None)
                pi_star_true_eval = oracle.policy_probs(states_eval, oracle.pi_star)
                pi_ratio_true = pi_star_true_eval[idx, actions_eval] / np.clip(probs_eval[idx, actions_eval], EPS, None)
                plugin_est = float(np.mean(v_hat))
                d_eval = rho_eval * pi_ratio
                d_true_target_eval = rho_eval * pi_ratio_true
                tilde_d_hat = (
                    tilde_rho_eval * pi_ratio
                    + d_eval * (q_sa_hat - v_hat)
                )
                tilde_d_oracle_q = (
                    tilde_rho_eval * pi_ratio
                    + d_eval * (q_sa_oracle - v_oracle)
                )
                tilde_d_oracle_true_target = (
                    tilde_rho_eval * pi_ratio_true
                    + d_true_target_eval * (q_sa_oracle - v_oracle)
                )
                if_est = float(
                    np.mean(
                        signed_ratio_if_contribution(
                            v_hat,
                            d_eval,
                            rho_eval,
                            tilde_d_hat,
                            tilde_rho_eval,
                            r_obs + oracle.config.gamma_behavior * v_next_hat - q_sa_hat,
                            soft_next - continuation_eval[idx, actions_eval],
                            gamma=oracle.config.gamma_behavior,
                            temperature=oracle.config.tau_star,
                        )
                    )
                )
                if_oracle_q_est = float(
                    np.mean(
                        signed_ratio_if_contribution(
                            v_oracle,
                            d_eval,
                            rho_eval,
                            tilde_d_oracle_q,
                            tilde_rho_eval,
                            r_obs + oracle.config.gamma_behavior * v_next_oracle - q_sa_oracle,
                            soft_next - continuation_oracle[idx, actions_eval],
                            gamma=oracle.config.gamma_behavior,
                            temperature=oracle.config.tau_star,
                        )
                    )
                )
                if_oracle_q_true_pi_star_est = float(
                    np.mean(
                        signed_ratio_if_contribution(
                            v_oracle,
                            d_true_target_eval,
                            rho_eval,
                            tilde_d_oracle_true_target,
                            tilde_rho_eval,
                            r_obs + oracle.config.gamma_behavior * v_next_oracle - q_sa_oracle,
                            soft_next_oracle - continuation_oracle[idx, actions_eval],
                            gamma=oracle.config.gamma_behavior,
                            temperature=oracle.config.tau_star,
                        )
                    )
                )
                rows.append(
                    {
                        "policy_estimator": str(estimator),
                        "n": int(n),
                        "seed": seed,
                        **reward_metrics,
                        "psi_target_shift": float(np.dot(oracle.stationary_behavior, v_true_policy) - oracle.psi_1b),
                        "plugin_bias_large_n": plugin_est - oracle.psi_1b,
                        "if_bias_large_n": if_est - oracle.psi_1b,
                        "if_bias_oracle_q_large_n": if_oracle_q_est - oracle.psi_1b,
                        "if_bias_oracle_q_true_pi_star_large_n": if_oracle_q_true_pi_star_est - oracle.psi_1b,
                    }
                )
    summary_rows = summarize_audit_rows(
        rows,
        group_keys=("policy_estimator", "n"),
        metric_keys=(
            "reward_rmse_grid",
            "reward_rmse_stationary",
            "reward_rmse_rho_weighted",
            "psi_target_shift",
            "plugin_bias_large_n",
            "if_bias_large_n",
            "if_bias_oracle_q_large_n",
            "if_bias_oracle_q_true_pi_star_large_n",
        ),
    )
    return {"rows": rows, "summary": summary_rows}


def run_example_policy_candidate_coverage(
    base_config: Optional[JRSSBConfig],
    example_id: str,
    estimators: Sequence[str],
    sample_sizes: Sequence[int],
    repetitions: int,
    ratio_mode: str = "oracle",
    jobs: int = 1,
) -> List[Dict[str, float | str | int]]:
    base_config = JRSSBConfig() if base_config is None else base_config
    coverage_rows: List[Dict[str, float | str | int]] = []
    for estimator in estimators:
        if example_id == "1a":
            oracle = JRSSBOracle(replace(base_config, example1a_policy_estimator=str(estimator)))
        elif example_id == "1b":
            oracle = JRSSBOracle(replace(base_config, example1b_policy_estimator=str(estimator)))
        else:
            raise ValueError(f"Unsupported example id for candidate coverage: {example_id}")
        results = run_monte_carlo(
            oracle=oracle,
            sample_sizes=sample_sizes,
            repetitions=repetitions,
            example_ids=(example_id,),
            ratio_mode=ratio_mode,
            jobs=jobs,
        )
        for row in summarize_results(results):
            coverage_rows.append(
                {
                    **row,
                    "policy_estimator": str(estimator),
                    "coverage_distance_to_nominal_95": _coverage_distance_to_nominal(row),
                }
            )
    return coverage_rows


def _aggregate_estimator_score(
    rows: Sequence[Dict[str, float | str | int]],
    estimator: str,
    sample_sizes: Sequence[int],
    metric_names: Sequence[str],
    absolute_metrics: Sequence[bool],
) -> tuple[float, ...]:
    selected = [row for row in rows if str(row.get("policy_estimator")) == str(estimator) and int(row["n"]) in sample_sizes]
    score: List[float] = []
    for metric_name, use_abs in zip(metric_names, absolute_metrics):
        total = 0.0
        for n in sample_sizes:
            row = next(row for row in selected if int(row["n"]) == int(n))
            value = float(row[metric_name])
            total += abs(value) if use_abs else value
        score.append(total)
    return tuple(score)


def _candidate_beats_baseline(
    candidate_rows: Sequence[Dict[str, float | str | int]],
    baseline_rows: Sequence[Dict[str, float | str | int]],
    sample_sizes: Sequence[int],
    metric_names: Sequence[str],
    absolute_metrics: Sequence[bool],
) -> bool:
    for n in sample_sizes:
        candidate_row = next(row for row in candidate_rows if int(row["n"]) == int(n))
        baseline_row = next(row for row in baseline_rows if int(row["n"]) == int(n))
        for metric_name, use_abs in zip(metric_names, absolute_metrics):
            candidate_value = float(candidate_row[metric_name])
            baseline_value = float(baseline_row[metric_name])
            if use_abs:
                candidate_value = abs(candidate_value)
                baseline_value = abs(baseline_value)
            if not candidate_value < baseline_value:
                return False
    return True


def _select_example1a_shortlist(summary_rows: Sequence[Dict[str, float | str | int]]) -> tuple[str, ...]:
    baseline = "maxent"
    non_baseline = sorted({str(row["policy_estimator"]) for row in summary_rows if str(row["policy_estimator"]) != baseline})
    if not non_baseline:
        return (baseline,)
    scored = sorted(
        non_baseline,
        key=lambda estimator: _aggregate_estimator_score(
            summary_rows,
            estimator,
            sample_sizes=(5_000, 10_000),
            metric_names=("reward_rmse_rho_weighted",),
            absolute_metrics=(False,),
        ),
    )
    return tuple(dict.fromkeys((baseline, scored[0])))


def _select_example1a_candidate(
    large_summary: Sequence[Dict[str, float | str | int]],
    pilot_summary: Sequence[Dict[str, float | str | int]],
    sample_sizes: Sequence[int] = (5_000, 10_000),
) -> Dict[str, object]:
    candidates = sorted({str(row["policy_estimator"]) for row in large_summary})
    baseline = "maxent"
    ranking_rows: List[Dict[str, object]] = []
    for estimator in candidates:
        large_rows = [row for row in large_summary if str(row["policy_estimator"]) == estimator]
        pilot_rows = [row for row in pilot_summary if str(row["policy_estimator"]) == estimator]
        score = (
            *_aggregate_estimator_score(
                large_rows,
                estimator,
                sample_sizes=sample_sizes,
                metric_names=("if_bias_large_n", "reward_rmse_rho_weighted"),
                absolute_metrics=(True, False),
            ),
            sum(
                abs(float(next(row for row in pilot_rows if int(row["n"]) == int(n))["coverage_95"]) - 0.95)
                for n in sample_sizes
            ),
        )
        ranking_rows.append({"policy_estimator": estimator, "score_1": score[0], "score_2": score[1], "score_3": score[2]})
    ranking_rows.sort(key=lambda row: (row["score_1"], row["score_2"], row["score_3"]))
    winner = str(ranking_rows[0]["policy_estimator"]) if ranking_rows else baseline
    baseline_large = [row for row in large_summary if str(row["policy_estimator"]) == baseline]
    baseline_pilot = [row for row in pilot_summary if str(row["policy_estimator"]) == baseline]
    winner_large = [row for row in large_summary if str(row["policy_estimator"]) == winner]
    winner_pilot = [row for row in pilot_summary if str(row["policy_estimator"]) == winner]
    promoted = winner != baseline and _candidate_beats_baseline(
        candidate_rows=winner_large,
        baseline_rows=baseline_large,
        sample_sizes=sample_sizes,
        metric_names=("if_bias_large_n", "reward_rmse_rho_weighted"),
        absolute_metrics=(True, False),
    ) and _candidate_beats_baseline(
        candidate_rows=winner_pilot,
        baseline_rows=baseline_pilot,
        sample_sizes=sample_sizes,
        metric_names=("coverage_distance_to_nominal_95",),
        absolute_metrics=(False,),
    )
    selected = winner if promoted else baseline
    return {
        "baseline_estimator": baseline,
        "winner_estimator": winner,
        "selected_estimator": selected,
        "promoted": bool(promoted),
        "rankings": ranking_rows,
    }


def _select_example1b_candidate(
    large_summary: Sequence[Dict[str, float | str | int]],
    sample_sizes: Sequence[int] = (5_000, 10_000),
) -> Dict[str, object]:
    baseline = "structural-linear"
    candidates = sorted({str(row["policy_estimator"]) for row in large_summary})
    ranking_rows: List[Dict[str, object]] = []
    for estimator in candidates:
        score = _aggregate_estimator_score(
            large_summary,
            estimator,
            sample_sizes=sample_sizes,
            metric_names=("psi_target_shift", "if_bias_oracle_q_large_n", "reward_rmse_rho_weighted"),
            absolute_metrics=(True, True, False),
        )
        ranking_rows.append({"policy_estimator": estimator, "score_1": score[0], "score_2": score[1], "score_3": score[2]})
    ranking_rows.sort(key=lambda row: (row["score_1"], row["score_2"], row["score_3"]))
    winner = str(ranking_rows[0]["policy_estimator"]) if ranking_rows else baseline
    baseline_rows = [row for row in large_summary if str(row["policy_estimator"]) == baseline]
    winner_rows = [row for row in large_summary if str(row["policy_estimator"]) == winner]
    promoted = winner != baseline and _candidate_beats_baseline(
        candidate_rows=winner_rows,
        baseline_rows=baseline_rows,
        sample_sizes=sample_sizes,
        metric_names=("psi_target_shift", "if_bias_oracle_q_large_n", "reward_rmse_rho_weighted"),
        absolute_metrics=(True, True, False),
    )
    selected = winner if promoted else baseline
    return {
        "baseline_estimator": baseline,
        "winner_estimator": winner,
        "selected_estimator": selected,
        "promoted": bool(promoted),
        "rankings": ranking_rows,
    }


def _coverage_distance_to_nominal(row: Dict[str, float | str | int]) -> float:
    return abs(float(row["coverage_95"]) - 0.95)


def _acceptance_checks_for_row(row: Dict[str, float | str | int]) -> Dict[str, float | bool]:
    avg_estimated_se = float(row["avg_estimated_se"])
    if_sd = float(row["if_sd"])
    if_bias = float(row["if_bias"])
    sd_over_estse = float("nan") if not np.isfinite(avg_estimated_se) or avg_estimated_se <= 0.0 else if_sd / avg_estimated_se
    bias_over_estse = float("nan") if not np.isfinite(avg_estimated_se) or avg_estimated_se <= 0.0 else abs(if_bias) / avg_estimated_se
    return {
        "if_beats_plugin_rmse": bool(float(row["if_rmse"]) < float(row["plugin_rmse"])),
        "coverage_good": bool(0.90 <= float(row["coverage_95"]) <= 0.98),
        "sd_over_estse": sd_over_estse,
        "sd_over_estse_good": bool(np.isfinite(sd_over_estse) and 0.90 <= sd_over_estse <= 1.10),
        "bias_over_estse": bias_over_estse,
        "bias_over_estse_good": bool(np.isfinite(bias_over_estse) and bias_over_estse < 0.5),
    }


def _merge_summary_rows(
    stage4_rows: Sequence[Dict[str, float | str | int]],
    confirmation_rows: Sequence[Dict[str, float | str | int]],
) -> List[Dict[str, float | str | int]]:
    replacement = {
        (str(row["example_id"]), int(row["n"])): dict(row)
        for row in confirmation_rows
    }
    merged: List[Dict[str, float | str | int]] = []
    for row in stage4_rows:
        key = (str(row["example_id"]), int(row["n"]))
        merged.append(replacement.get(key, dict(row)))
    return merged


def run_bias_acceptance(
    base_config: Optional[JRSSBConfig] = None,
    ratio_mode: str = "oracle",
    jobs: int = 1,
    audit_repetitions: int = 10,
    large_n: int = 50_000,
    large_repetitions: int = 5,
    pilot_repetitions: int = 20,
    full_repetitions: int = 50,
    confirmation_repetitions: int = 100,
) -> Dict[str, object]:
    base_config = JRSSBConfig() if base_config is None else base_config
    baseline_config = replace(
        base_config,
        example1a_policy_estimator="maxent",
        example1b_policy_estimator="structural-linear",
    )
    candidate_sizes = (5_000, 10_000)
    full_sizes = (1_000, 2_500, 5_000, 10_000)

    invariants_oracle = JRSSBOracle(baseline_config)
    invariants = {
        "soft_policy_self_check": oracle_soft_policy_self_check(invariants_oracle),
        "oracle_diagnostics": invariants_oracle.run_oracle_diagnostics(
            large_n=min(large_n, 20_000),
            seed=11_003,
            sampler="continuous",
        ),
    }

    stage1_example1a = run_example1a_policy_audit(
        base_config=baseline_config,
        estimators=EXAMPLE1A_POLICY_ESTIMATORS,
        sample_sizes=candidate_sizes,
        repetitions=audit_repetitions,
    )
    stage1_example1b = run_example1b_policy_audit(
        base_config=baseline_config,
        estimators=EXAMPLE1B_POLICY_ESTIMATORS,
        sample_sizes=candidate_sizes,
        repetitions=audit_repetitions,
    )

    shortlist_1a = _select_example1a_shortlist(stage1_example1a["summary"])
    shortlist_1b = tuple(dict.fromkeys(("structural-linear", "structural")))

    stage2_example1a = run_example1a_large_sample_decomposition(
        base_config=baseline_config,
        estimators=shortlist_1a,
        sample_sizes=candidate_sizes,
        repetitions=large_repetitions,
        eval_n=large_n,
    )
    stage2_example1b = run_example1b_large_sample_decomposition(
        base_config=baseline_config,
        estimators=shortlist_1b,
        sample_sizes=candidate_sizes,
        repetitions=large_repetitions,
        eval_n=large_n,
    )

    stage3_pilot_1a = run_example_policy_candidate_coverage(
        base_config=baseline_config,
        example_id="1a",
        estimators=shortlist_1a,
        sample_sizes=candidate_sizes,
        repetitions=pilot_repetitions,
        ratio_mode=ratio_mode,
        jobs=jobs,
    )
    stage3_pilot_1b = run_example_policy_candidate_coverage(
        base_config=baseline_config,
        example_id="1b",
        estimators=shortlist_1b,
        sample_sizes=candidate_sizes,
        repetitions=pilot_repetitions,
        ratio_mode=ratio_mode,
        jobs=jobs,
    )
    stage3_pilot_2 = summarize_results(
        run_monte_carlo(
            oracle=JRSSBOracle(baseline_config),
            sample_sizes=candidate_sizes,
            repetitions=pilot_repetitions,
            example_ids=("2",),
            ratio_mode=ratio_mode,
            jobs=jobs,
        )
    )

    selection_1a = _select_example1a_candidate(
        large_summary=stage2_example1a["summary"],
        pilot_summary=stage3_pilot_1a,
        sample_sizes=candidate_sizes,
    )
    selection_1b = _select_example1b_candidate(
        large_summary=stage2_example1b["summary"],
        sample_sizes=candidate_sizes,
    )

    selected_config = replace(
        baseline_config,
        example1a_policy_estimator=str(selection_1a["selected_estimator"]),
        example1b_policy_estimator=str(selection_1b["selected_estimator"]),
    )
    selected_oracle = JRSSBOracle(selected_config)
    stage4_results = run_monte_carlo(
        oracle=selected_oracle,
        sample_sizes=full_sizes,
        repetitions=full_repetitions,
        example_ids=("1a", "1b", "2"),
        ratio_mode=ratio_mode,
        jobs=jobs,
    )
    stage4_summary = summarize_results(stage4_results)

    baseline_secondary_results = run_monte_carlo(
        oracle=JRSSBOracle(baseline_config),
        sample_sizes=(1_000, 2_500),
        repetitions=full_repetitions,
        example_ids=("1a", "1b", "2"),
        ratio_mode=ratio_mode,
        jobs=jobs,
    )
    baseline_secondary_summary = summarize_results(baseline_secondary_results)

    stage5_targets = [
        row
        for row in stage4_summary
        if int(row["n"]) in candidate_sizes and (
            0.87 <= float(row["coverage_95"]) < 0.90
            or 0.98 < float(row["coverage_95"]) <= 1.00
        )
    ]
    stage5_results: List[SingleRunResult] = []
    for row in stage5_targets:
        stage5_results.extend(
            run_monte_carlo(
                oracle=selected_oracle,
                sample_sizes=(int(row["n"]),),
                repetitions=confirmation_repetitions,
                example_ids=(str(row["example_id"]),),
                ratio_mode=ratio_mode,
                jobs=jobs,
            )
        )
    stage5_summary = summarize_results(stage5_results)
    final_summary = _merge_summary_rows(stage4_summary, stage5_summary)

    final_rows_with_checks: List[Dict[str, object]] = []
    for row in final_summary:
        final_rows_with_checks.append({**row, **_acceptance_checks_for_row(row)})

    gate_rows = [row for row in final_rows_with_checks if int(row["n"]) in candidate_sizes]
    primary_pass = all(
        bool(row["if_beats_plugin_rmse"])
        and bool(row["coverage_good"])
        and bool(row["sd_over_estse_good"])
        and bool(row["bias_over_estse_good"])
        for row in gate_rows
    )

    baseline_secondary_lookup = {
        (str(row["example_id"]), int(row["n"])): dict(row)
        for row in baseline_secondary_summary
    }
    secondary_rows: List[Dict[str, object]] = []
    for row in final_rows_with_checks:
        if int(row["n"]) not in (1_000, 2_500):
            continue
        baseline_row = baseline_secondary_lookup.get((str(row["example_id"]), int(row["n"])))
        improved = False if baseline_row is None else float(row["coverage_95"]) > float(baseline_row["coverage_95"])
        secondary_rows.append(
            {
                "example_id": row["example_id"],
                "n": row["n"],
                "coverage_selected": row["coverage_95"],
                "coverage_baseline": float(baseline_row["coverage_95"]) if baseline_row is not None else float("nan"),
                "improved_vs_baseline": bool(improved),
                "coverage_at_least_085": bool(float(row["coverage_95"]) >= 0.85),
            }
        )
    secondary_pass = all(
        bool(row["improved_vs_baseline"]) and bool(row["coverage_at_least_085"])
        for row in secondary_rows
    )

    accepted = bool(
        invariants["soft_policy_self_check"]["passes"]
        and primary_pass
        and secondary_pass
    )
    return {
        "candidate_diagnostics": {
            "example1a_audit": stage1_example1a,
            "example1b_audit": stage1_example1b,
            "example1a_large_sample": stage2_example1a,
            "example1b_large_sample": stage2_example1b,
        },
        "pilot_coverage": {
            "example1a": stage3_pilot_1a,
            "example1b": stage3_pilot_1b,
            "example2": stage3_pilot_2,
        },
        "selections": {
            "example1a": selection_1a,
            "example1b": selection_1b,
        },
        "stage4_full_coverage": stage4_summary,
        "stage5_confirmation_coverage": stage5_summary,
        "final_full_coverage": final_rows_with_checks,
        "baseline_secondary_coverage": baseline_secondary_summary,
        "secondary_finite_sample_checks": secondary_rows,
        "acceptance": {
            "accepted": accepted,
            "primary_pass": bool(primary_pass),
            "secondary_pass": bool(secondary_pass),
            "selected_example1a_policy_estimator": selection_1a["selected_estimator"],
            "selected_example1b_policy_estimator": selection_1b["selected_estimator"],
            "soft_policy_self_check_passes": bool(invariants["soft_policy_self_check"]["passes"]),
        },
        "invariants": invariants,
    }


def run_example2_nuisance_audit(
    base_config: Optional[JRSSBConfig] = None,
    estimators: Sequence[str] = EXAMPLE2_POLICY_ESTIMATORS,
    sample_sizes: Sequence[int] = (5_000, 10_000),
    repetitions: int = 10,
) -> Dict[str, object]:
    base_config = JRSSBConfig() if base_config is None else base_config
    rows: List[Dict[str, float | str | int]] = []
    for estimator in estimators:
        oracle = JRSSBOracle(replace(base_config, example2_policy_estimator=str(estimator)))
        for n in sample_sizes:
            for rep in range(repetitions):
                seed = 90_000 * (rep + 1) + 131 * int(n) + 2_003
                data = oracle.sample_stationary_transitions(n=int(n), seed=seed)
                policy_hat = fit_behavior_policy(
                    oracle=oracle,
                    states=data["states"],
                    actions=data["actions"],
                    seed=seed,
                )
                reward_grid = np.log(np.clip(policy_hat.predict_proba(oracle.main_grid.states), EPS, None))
                q_nu_init = oracle.q_nu + (reward_grid - oracle.r0)
                q_nu_grid, v_nu_grid = oracle.evaluate_policy(
                    reward_grid,
                    oracle.nu,
                    oracle.config.gamma_behavior,
                    init_q=q_nu_init,
                    tol=oracle.config.nuisance_bellman_tol,
                    max_iterations=oracle.config.nuisance_max_iterations,
                )
                reward_norm_grid = q_nu_grid - v_nu_grid[:, None]
                q_eval_init = oracle.q_2 + (reward_norm_grid - oracle.reward_norm)
                q_eval_grid, _ = oracle.evaluate_policy(
                    reward_norm_grid,
                    oracle.pi_fix,
                    oracle.config.gamma_example2,
                    init_q=q_eval_init,
                    tol=oracle.config.nuisance_bellman_tol,
                    max_iterations=oracle.config.nuisance_max_iterations,
                )
                rows.append(
                    {
                        "policy_estimator": str(estimator),
                        "n": int(n),
                        "seed": seed,
                        "reward_rmse": float(np.sqrt(np.mean((reward_grid - oracle.r0) ** 2))),
                        "reward_norm_rmse": float(np.sqrt(np.mean((reward_norm_grid - oracle.reward_norm) ** 2))),
                        "q_nu_rmse": float(np.sqrt(np.mean((q_nu_grid - oracle.q_nu) ** 2))),
                        "q_eval_rmse": float(np.sqrt(np.mean((q_eval_grid - oracle.q_2) ** 2))),
                    }
                )
    summary_rows = summarize_audit_rows(
        rows,
        group_keys=("policy_estimator", "n"),
        metric_keys=("reward_rmse", "reward_norm_rmse", "q_nu_rmse", "q_eval_rmse"),
    )
    return {"rows": rows, "summary": summary_rows}


def run_example2_semi_oracle_audit(
    oracle: JRSSBOracle,
    n: int = 2_500,
    seed: int = 101,
    ratio_mode: str = "oracle",
) -> List[Dict[str, float | str | int]]:
    data = oracle.sample_stationary_transitions(n=n, seed=seed)
    folds = fold_splits(n, seed, n_folds=oracle.config.crossfit_folds)
    all_idx = np.arange(n)
    cells = [
        ("all_estimated", False, False),
        ("oracle_qnu", True, False),
        ("oracle_qeval", False, True),
        ("both_oracle_q", True, True),
    ]
    rows: List[Dict[str, float | str | int]] = []
    for cell_name, use_oracle_q_nu, use_oracle_q_eval in cells:
        contributions_if = np.zeros(n, dtype=float)
        contributions_plugin = np.zeros(n, dtype=float)
        for fold_number, eval_idx in enumerate(folds):
            train_mask = np.ones(n, dtype=bool)
            train_mask[eval_idx] = False
            train_idx = all_idx[train_mask]
            fold_seed = seed * 10 + fold_number + 1
            policy_hat = fit_behavior_policy(
                oracle=oracle,
                states=data["states"][train_idx],
                actions=data["actions"][train_idx],
                seed=fold_seed,
            )
            nu_adapter = make_policy_adapter(oracle, deterministic_fn=oracle._reference_policy_probs)
            pi_fix_adapter = make_policy_adapter(oracle, deterministic_fn=oracle._fixed_policy_probs)
            train_probs = policy_hat.predict_proba(data["states"][train_idx])
            r_obs_train = np.log(np.clip(train_probs[np.arange(train_idx.shape[0]), data["actions"][train_idx]], EPS, None))
            q_nu_hat = fit_fqe_neural(
                states=data["states"][train_idx],
                actions=data["actions"][train_idx],
                rewards=r_obs_train,
                next_states=data["next_states"][train_idx],
                dones=np.zeros(train_idx.shape[0], dtype=float),
                policy=nu_adapter,
                n_actions=oracle.config.n_actions,
                gamma=oracle.config.gamma_behavior,
                hidden_sizes=oracle.config.fqe_hidden_sizes,
                learning_rate=oracle.config.fqe_learning_rate,
                n_fqe_iters=oracle.config.fqe_iters * oracle.config.example2_stage1_iters_multiplier,
                epochs_per_iter=oracle.config.fqe_epochs_per_iter * oracle.config.example2_stage1_epochs_multiplier,
                seed=fold_seed,
            )
            q_nu_all_train = q_nu_hat.predict_all_actions(data["states"][train_idx])
            reward_norm_train = (q_nu_all_train - q_nu_all_train[:, [0]])[
                np.arange(train_idx.shape[0]),
                data["actions"][train_idx],
            ]
            q_eval_hat = fit_fqe_neural(
                states=data["states"][train_idx],
                actions=data["actions"][train_idx],
                rewards=reward_norm_train,
                next_states=data["next_states"][train_idx],
                dones=np.zeros(train_idx.shape[0], dtype=float),
                policy=pi_fix_adapter,
                n_actions=oracle.config.n_actions,
                gamma=oracle.config.gamma_example2,
                hidden_sizes=oracle.config.fqe_hidden_sizes,
                learning_rate=oracle.config.fqe_learning_rate,
                n_fqe_iters=oracle.config.fqe_iters * oracle.config.example2_stage2_iters_multiplier,
                epochs_per_iter=oracle.config.fqe_epochs_per_iter * oracle.config.example2_stage2_epochs_multiplier,
                seed=fold_seed + 17,
            )
            states_eval = data["states"][eval_idx]
            next_states_eval = data["next_states"][eval_idx]
            actions_eval = data["actions"][eval_idx]
            probs_eval = policy_hat.predict_proba(states_eval)
            pi_fix_eval = oracle._fixed_policy_probs(states_eval)
            nu_eval = oracle._reference_policy_probs(states_eval)
            if use_oracle_q_nu:
                q_nu_all_eval = oracle.action_values(states_eval, oracle.q_nu)
                v_nu_next = oracle.state_values(next_states_eval, oracle.v_nu)
            else:
                q_nu_all_eval = q_nu_hat.predict_all_actions(states_eval)
                q_nu_all_next = q_nu_hat.predict_all_actions(next_states_eval)
                v_nu_next = q_nu_all_next[:, 0]
            if use_oracle_q_eval:
                q_eval_all = oracle.action_values(states_eval, oracle.q_2)
                v_eval = oracle.state_values(states_eval, oracle.v_2)
                v_eval_next = oracle.state_values(next_states_eval, oracle.v_2)
            else:
                q_eval_all = q_eval_hat.predict_all_actions(states_eval)
                q_eval_all_next = q_eval_hat.predict_all_actions(next_states_eval)
                v_eval = np.sum(pi_fix_eval * q_eval_all, axis=1)
                v_eval_next = np.sum(oracle._fixed_policy_probs(next_states_eval) * q_eval_all_next, axis=1)
            q_nu_sa = q_nu_all_eval[np.arange(eval_idx.shape[0]), actions_eval]
            q_eval_sa = q_eval_all[np.arange(eval_idx.shape[0]), actions_eval]
            reward_norm_eval = q_nu_all_eval - q_nu_all_eval[:, [0]]
            reward_norm_sa = reward_norm_eval[np.arange(eval_idx.shape[0]), actions_eval]
            r_obs_eval = np.log(np.clip(probs_eval[np.arange(eval_idx.shape[0]), actions_eval], EPS, None))
            rho_eval = oracle.state_values(states_eval, oracle.rho_fix_gamma_prime)
            pi_ratio = pi_fix_eval[np.arange(eval_idx.shape[0]), actions_eval] / np.clip(
                probs_eval[np.arange(eval_idx.shape[0]), actions_eval],
                EPS,
                None,
            )
            nu_ratio = nu_eval[np.arange(eval_idx.shape[0]), actions_eval] / np.clip(
                probs_eval[np.arange(eval_idx.shape[0]), actions_eval],
                EPS,
                None,
            )
            contributions_plugin[eval_idx] = v_eval
            contributions_if[eval_idx] = (
                v_eval
                + rho_eval * pi_ratio * (reward_norm_sa + oracle.config.gamma_example2 * v_eval_next - q_eval_sa)
                + rho_eval * (pi_ratio - nu_ratio) * (r_obs_eval + oracle.config.gamma_behavior * v_nu_next - q_nu_sa)
                + rho_eval * (pi_ratio - nu_ratio)
            )
        plugin_estimate = float(np.mean(contributions_plugin))
        if_estimate = float(np.mean(contributions_if))
        rows.append(
            {
                "cell": cell_name,
                "n": n,
                "seed": seed,
                "ratio_mode": ratio_mode,
                "plugin_estimate": plugin_estimate,
                "if_estimate": if_estimate,
                "truth": oracle.psi_2,
                "plugin_error": plugin_estimate - oracle.psi_2,
                "if_error": if_estimate - oracle.psi_2,
            }
        )
    return rows


def _make_method_oracle(base_config: JRSSBConfig, method: str) -> JRSSBOracle:
    return JRSSBOracle(replace(base_config, example2_nuisance_method=method))


def _example2_acceptance_from_summary(summary_rows: Sequence[Dict[str, float | str | int]]) -> Dict[str, object]:
    rows = [row for row in summary_rows if row["example_id"] == "2" and row["n"] in {5_000, 10_000}]
    rows = sorted(rows, key=lambda row: int(row["n"]))
    if not rows:
        return {"passes_rmse": False, "passes_coverage": False, "passes_overlap": False, "accepted": False}
    passes_rmse = all(float(row["if_rmse"]) < float(row["plugin_rmse"]) for row in rows)
    passes_coverage = all(0.90 <= float(row["coverage_95"]) <= 0.98 for row in rows)
    passes_overlap = all(
        float(row["avg_pi_ratio_q99"]) <= 100.0
        and float(row["avg_nu_ratio_q99"]) <= 100.0
        and float(row["avg_pi0_action0_q01"]) >= max(0.01, 0.5 * 0.02)
        for row in rows
    )
    return {
        "passes_rmse": passes_rmse,
        "passes_coverage": passes_coverage,
        "passes_overlap": passes_overlap,
        "accepted": bool(passes_rmse and passes_coverage and passes_overlap),
    }


def select_example2_method(summary_rows: Sequence[Dict[str, float | str | int]]) -> Dict[str, object]:
    methods = sorted({str(row["nuisance_method"]) for row in summary_rows if row["example_id"] == "2"})
    rankings: List[Dict[str, object]] = []
    for method in methods:
        rows = [
            row
            for row in summary_rows
            if row["example_id"] == "2" and row["nuisance_method"] == method and int(row["n"]) in {5_000, 10_000}
        ]
        if not rows:
            continue
        rmse_score = float(sum(float(row["if_rmse"]) for row in rows))
        coverage_score = float(sum(abs(float(row["coverage_95"]) - 0.95) for row in rows))
        ci_score = float(sum(float(row["avg_ci_length"]) for row in rows))
        stability_score = float(sum(float(row["avg_pi_ratio_q99"]) + float(row["avg_nu_ratio_q99"]) for row in rows))
        acceptance = _example2_acceptance_from_summary(rows)
        rankings.append(
            {
                "method": method,
                "rmse_score": rmse_score,
                "coverage_score": coverage_score,
                "ci_score": ci_score,
                "stability_score": stability_score,
                **acceptance,
            }
        )
    rankings.sort(key=lambda row: (row["rmse_score"], row["coverage_score"], row["ci_score"], row["stability_score"]))
    if not rankings:
        return {"selected_method": None, "decision": "no_results", "rankings": []}
    selected = rankings[0]
    if bool(selected["accepted"]):
        decision = "use_selected_method_main"
    elif bool(selected["passes_rmse"]) and not bool(selected["passes_coverage"]):
        decision = "use_selected_method_main_with_ci_sensitivity"
    else:
        decision = "fallback_to_stability_first_method"
    return {"selected_method": selected["method"], "decision": decision, "rankings": rankings}


def run_example2_method_selection(
    base_config: Optional[JRSSBConfig] = None,
    methods: Sequence[str] = ("neural", "neural-main-oracle-bellman", "neural-coarse-bellman"),
    sample_sizes: Sequence[int] = (2_500, 5_000, 10_000),
    repetitions: int = 100,
    ratio_mode: str = "oracle",
    jobs: int = 1,
) -> Dict[str, object]:
    base_config = JRSSBConfig() if base_config is None else base_config
    all_results: List[SingleRunResult] = []
    for method in methods:
        method_oracle = _make_method_oracle(base_config, method)
        all_results.extend(
            run_monte_carlo(
                oracle=method_oracle,
                sample_sizes=sample_sizes,
                repetitions=repetitions,
                example_ids=("2",),
                ratio_mode=ratio_mode,
                jobs=jobs,
            )
        )
    summary_rows = summarize_results(all_results)
    selection = select_example2_method(summary_rows)
    return {"results": all_results, "summary": summary_rows, "selection": selection}


def run_example2_paper_comparison(
    base_config: Optional[JRSSBConfig] = None,
    sample_sizes: Sequence[int] = (5_000, 10_000),
    repetitions: int = 20,
    ratio_mode: str = "oracle",
    jobs: int = 1,
) -> Dict[str, object]:
    base_config = JRSSBConfig() if base_config is None else base_config
    methods = (
        "neural-main-oracle-bellman",
        "oracle-q",
        "oracle-all",
    )
    labels = {
        "neural-main-oracle-bellman": "practical",
        "oracle-q": "quasi-oracle",
        "oracle-all": "full-oracle",
    }
    study = run_example2_method_selection(
        base_config=base_config,
        methods=methods,
        sample_sizes=sample_sizes,
        repetitions=repetitions,
        ratio_mode=ratio_mode,
        jobs=jobs,
    )
    comparison_rows: List[Dict[str, object]] = []
    for row in study["summary"]:
        method = str(row["nuisance_method"])
        comparison_rows.append(
            {
                **row,
                "paper_label": labels.get(method, method),
                "if_beats_plugin_rmse": bool(float(row["if_rmse"]) < float(row["plugin_rmse"])),
                "coverage_good": bool(0.90 <= float(row["coverage_95"]) <= 0.98),
                "overlap_good": bool(
                    float(row["avg_pi_ratio_q99"]) <= 100.0
                    and float(row["avg_nu_ratio_q99"]) <= 100.0
                    and float(row["avg_pi0_action0_q01"]) >= 0.01
                ),
            }
        )
    return {
        "results": study["results"],
        "summary": study["summary"],
        "comparison": comparison_rows,
        "selection": study["selection"],
        "paper_methods": labels,
    }


def run_example2_smoke_check(
    oracle: JRSSBOracle,
    sample_sizes: Sequence[int] = (5_000, 10_000),
    seeds: Sequence[int] = (201, 202, 203),
    ratio_mode: str = "oracle",
) -> Dict[str, object]:
    results = [
        run_single_replication(oracle=oracle, n=n, seed=seed, example_id="2", ratio_mode=ratio_mode)
        for n in sample_sizes
        for seed in seeds
    ]
    summary_rows = summarize_results(results)
    return {"results": results, "summary": summary_rows, "acceptance": _example2_acceptance_from_summary(summary_rows)}


def run_example2_shakedown(
    oracle: JRSSBOracle,
    sample_sizes: Sequence[int] = (5_000, 10_000),
    repetitions: int = 20,
    ratio_mode: str = "oracle",
    jobs: int = 1,
) -> Dict[str, object]:
    results = run_monte_carlo(
        oracle=oracle,
        sample_sizes=sample_sizes,
        repetitions=repetitions,
        example_ids=("2",),
        ratio_mode=ratio_mode,
        jobs=jobs,
    )
    summary_rows = summarize_results(results)
    finite_checks = all(
        np.isfinite(
            [
                row.plugin_estimate,
                row.if_estimate,
                row.estimated_se,
                row.reward_rmse,
                row.bellman_residual_rmse,
            ]
        ).all()
        for row in results
    )
    sensible_se = all(float(row["avg_estimated_se"]) > 0.0 and np.isfinite(float(row["avg_estimated_se"])) for row in summary_rows)
    if_beats_plugin = all(float(row["if_rmse"]) < float(row["plugin_rmse"]) for row in summary_rows)
    return {
        "results": results,
        "summary": summary_rows,
        "checks": {
            "no_numerical_blowups": bool(finite_checks),
            "sensible_standard_errors": bool(sensible_se),
            "if_improves_on_plugin": bool(if_beats_plugin),
            **_example2_acceptance_from_summary(summary_rows),
        },
    }


def run_validation_suite(
    oracle: JRSSBOracle,
    pilot_seed: int = 404,
    pilot_n: int = 10_000,
    oracle_sample_n: int = 5_000,
) -> Dict[str, float | bool]:
    diagnostics = dict(
        oracle.run_oracle_diagnostics(large_n=oracle_sample_n, seed=pilot_seed + 9_001, sampler="oracle-grid")
    )
    diagnostics.update(
        {
            f"soft_policy_self_check_{key}": value
            for key, value in oracle_soft_policy_self_check(oracle).items()
        }
    )
    diagnostics["nuisance_sample_mode"] = oracle.config.nuisance_sample_mode
    if oracle.config.nuisance_sample_mode == "crossfit":
        split = fold_splits(pilot_n, pilot_seed, n_folds=oracle.config.crossfit_folds)
        split_union = np.concatenate(split) if split else np.array([], dtype=int)
        diagnostics["crossfit_disjoint"] = bool(
            split_union.size == pilot_n and np.unique(split_union).size == pilot_n
        )
    else:
        diagnostics["crossfit_disjoint"] = True

    pilot_1b = run_single_replication(oracle=oracle, n=pilot_n, seed=pilot_seed, example_id="1b", ratio_mode="oracle")
    pilot_2 = run_single_replication(oracle=oracle, n=pilot_n, seed=pilot_seed + 1, example_id="2", ratio_mode="oracle")
    diagnostics["pilot_1b_if_better_than_plugin"] = bool(abs(pilot_1b.if_error) <= abs(pilot_1b.plugin_error))
    diagnostics["pilot_2_if_better_than_plugin"] = bool(abs(pilot_2.if_error) <= abs(pilot_2.plugin_error))
    diagnostics["pilot_1b_plugin_error"] = float(pilot_1b.plugin_error)
    diagnostics["pilot_1b_if_error"] = float(pilot_1b.if_error)
    diagnostics["pilot_2_plugin_error"] = float(pilot_2.plugin_error)
    diagnostics["pilot_2_if_error"] = float(pilot_2.if_error)
    semi_oracle = run_example2_semi_oracle_audit(oracle=oracle, n=min(2_500, pilot_n), seed=pilot_seed + 17)
    semi_oracle_both = next(row for row in semi_oracle if row["cell"] == "both_oracle_q")
    diagnostics["semi_oracle_both_oracle_q_if_error"] = float(semi_oracle_both["if_error"])
    diagnostics["semi_oracle_both_oracle_q_near_truth"] = bool(abs(float(semi_oracle_both["if_error"])) <= 0.10)
    smoke = run_example2_smoke_check(oracle=oracle)
    smoke_summary = {int(row["n"]): row for row in smoke["summary"]}
    for n in (5_000, 10_000):
        if n in smoke_summary:
            diagnostics[f"example2_smoke_if_rmse_better_{n}"] = bool(
                float(smoke_summary[n]["if_rmse"]) < float(smoke_summary[n]["plugin_rmse"])
            )
            diagnostics[f"example2_smoke_coverage_{n}"] = float(smoke_summary[n]["coverage_95"])
    return diagnostics
