"""Normalized scalar-score isotonic Bellman calibration.

This module implements the exact generalized-PAVA update used by the
occupancy-calibration paper.  It deliberately targets the standard normalized
discounted occupancy ratio.  Coverage-stopped estimands belong to a different
experiment and are rejected by this API.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import numpy as np


Array = np.ndarray
Direction = Literal["increasing", "decreasing"]
Initialization = Literal["unit", "score"]
SupportPolicy = Literal["error", "constant_extrapolation"]
_ORDER_TOL = 1e-12


@dataclass(frozen=True)
class IsotonicCalibrationConfig:
    """Configuration for normalized KL-projected FORE calibration."""

    num_iterations: int = 500
    tolerance: float = 1e-8
    direction: Direction = "increasing"
    positivity_floor: float = 0.0
    normalize: bool = True
    fixed_point_damping: float = 1.0
    initialization: Initialization = "unit"
    support_tol: float = 1e-12
    support_policy: SupportPolicy = "constant_extrapolation"
    minimum_boundary_block_observations: int = 1


@dataclass(frozen=True)
class IsotonicCalibrationResult:
    """Fitted normalized monotone score map and convergence diagnostics."""

    method: str
    status: str
    source_weights: Array
    grid: Array
    fitted_grid_values: Array
    diagnostics: dict[str, float | int | str | bool]
    history: list[dict[str, float | int | str | bool]]
    config: IsotonicCalibrationConfig
    runtime_sec: float

    def predict(self, score: Array) -> Array:
        """Evaluate the fitted behavior-knot step map at new scores."""

        transformed = _transform_score(_finite_vector(score, "score"), self.config.direction)
        return _predict_step(transformed, self.grid, self.fitted_grid_values)


@dataclass(frozen=True)
class _PreparedInputs:
    source_score_raw: Array
    source_score: Array
    next_score: Array
    initial_score: Array
    source_weight: Array
    next_source_index: Array
    next_weight: Array
    initial_weight: Array
    grid: Array
    source_grid_index: Array
    next_grid_index: Array
    initial_grid_index: Array
    source_exposure: Array
    source_observation_count: Array
    initial_mass: Array
    gamma: float
    extrapolated_next_fraction: float
    extrapolated_initial_mass: float


@dataclass(frozen=True)
class _GridFit:
    grid: Array
    exposure: Array
    target: Array
    observation_count: Array


def fit_isotonic_fori_pava(
    *,
    source_score: Array,
    next_score: Array,
    initial_score: Array,
    gamma: float,
    source_weight: Array | None = None,
    next_source_index: Array | None = None,
    next_weight: Array | None = None,
    initial_weight: Array | None = None,
    initial_omega: Array | None = None,
    config: IsotonicCalibrationConfig | None = None,
) -> IsotonicCalibrationResult:
    """Fit exact isotonic Bellman calibration by generalized PAVA.

    ``source_score``, ``next_score``, and ``initial_score`` may be pooled
    out-of-fold scores.  The routine fits one map to those pooled scores.  A
    cross-calibrated predictor should subsequently apply this one map to every
    fold-specific base predictor and aggregate the resulting predictions
    pointwise; see :mod:`occupancy_ratio_benchmark.calibration_crossfit`.
    """

    started = time.perf_counter()
    cfg = _validate_config(config)
    arrays = _prepare_inputs(
        source_score=source_score,
        next_score=next_score,
        initial_score=initial_score,
        gamma=gamma,
        source_weight=source_weight,
        next_source_index=next_source_index,
        next_weight=next_weight,
        initial_weight=initial_weight,
        direction=cfg.direction,
        support_tol=cfg.support_tol,
        support_policy=cfg.support_policy,
    )
    omega = _initial_weights(
        arrays.source_score_raw,
        arrays.source_weight,
        initial_omega=initial_omega,
        config=cfg,
    )
    history: list[dict[str, float | int | str | bool]] = []
    final_fit: _GridFit | None = None
    final_values: Array | None = None
    loop_converged = False

    for iteration in range(1, cfg.num_iterations + 1):
        grid_fit = _build_grid_fit(arrays, omega, support_tol=cfg.support_tol)
        values = _solve_generalized_pava(
            grid_fit.exposure,
            grid_fit.target,
            grid=grid_fit.grid,
            support_tol=cfg.support_tol,
        )
        values, floor_fraction = _floor_and_normalize(
            values,
            grid_fit.exposure,
            positivity_floor=cfg.positivity_floor,
            normalize=True,
            zero_tol=cfg.support_tol,
        )
        candidate = _predict_step(arrays.source_score, grid_fit.grid, values)
        updated = (1.0 - cfg.fixed_point_damping) * omega + cfg.fixed_point_damping * candidate
        updated, damped_floor_fraction = _floor_and_normalize(
            updated,
            arrays.source_weight,
            positivity_floor=cfg.positivity_floor,
            normalize=True,
            zero_tol=cfg.support_tol,
        )
        relative_change = _relative_change(omega, updated, arrays.source_weight)
        history.append(
            _diagnostics(
                arrays,
                grid_fit,
                values,
                updated,
                iteration=iteration,
                relative_change=relative_change,
                floor_fraction=max(floor_fraction, damped_floor_fraction),
                support_tol=cfg.support_tol,
                required_minimum_boundary_block_observations=(
                    cfg.minimum_boundary_block_observations
                ),
            )
        )
        omega = updated
        final_fit = grid_fit
        final_values = values
        if relative_change <= cfg.tolerance:
            loop_converged = True
            break

    if final_fit is None or final_values is None:
        raise RuntimeError("isotonic calibration loop did not run")

    # With damping, expose an actual monotone projection as the deployed map.
    if cfg.fixed_point_damping != 1.0:
        previous_omega = omega.copy()
        final_fit = _build_grid_fit(arrays, omega, support_tol=cfg.support_tol)
        final_values = _solve_generalized_pava(
            final_fit.exposure,
            final_fit.target,
            grid=final_fit.grid,
            support_tol=cfg.support_tol,
        )
        final_values, floor_fraction = _floor_and_normalize(
            final_values,
            final_fit.exposure,
            positivity_floor=cfg.positivity_floor,
            normalize=True,
            zero_tol=cfg.support_tol,
        )
        omega = _predict_step(arrays.source_score, final_fit.grid, final_values)
        history.append(
            _diagnostics(
                arrays,
                final_fit,
                final_values,
                omega,
                iteration=len(history) + 1,
                relative_change=_relative_change(previous_omega, omega, arrays.source_weight),
                floor_fraction=floor_fraction,
                support_tol=cfg.support_tol,
                required_minimum_boundary_block_observations=(
                    cfg.minimum_boundary_block_observations
                ),
                final_projection=True,
            )
        )

    # Boundary support is deployment regularization, not part of the fitted
    # fixed-point recursion.  Fit the paper's ordinary PAVA fixed point first,
    # then coarsen only undersupported endpoint blocks once.  Each adjacent
    # merge uses the pooled target/exposure ratio, so monotonicity, total mass,
    # and block balance on the coarsened endpoint partition are preserved.
    if cfg.minimum_boundary_block_observations > 1:
        previous_omega = omega.copy()
        ordinary_diagnostics = dict(history[-1])
        ordinary_history_count = len(history)
        final_values = _pool_small_boundary_pava_blocks(
            final_values,
            final_fit.exposure,
            final_fit.target,
            observation_count=final_fit.observation_count,
            minimum_boundary_block_observations=(
                cfg.minimum_boundary_block_observations
            ),
            support_tol=cfg.support_tol,
        )
        final_values, floor_fraction = _floor_and_normalize(
            final_values,
            final_fit.exposure,
            positivity_floor=cfg.positivity_floor,
            normalize=True,
            zero_tol=cfg.support_tol,
        )
        omega = _predict_step(arrays.source_score, final_fit.grid, final_values)
        postprocess_diagnostics = _diagnostics(
            arrays,
            final_fit,
            final_values,
            omega,
            iteration=int(ordinary_diagnostics["iteration"]),
            relative_change=float(ordinary_diagnostics["relative_change"]),
            floor_fraction=floor_fraction,
            support_tol=cfg.support_tol,
            required_minimum_boundary_block_observations=(
                cfg.minimum_boundary_block_observations
            ),
            final_projection=True,
        )
        postprocess_diagnostics["post_pava_boundary_pooling"] = True
        postprocess_diagnostics["post_pava_boundary_pooling_relative_change"] = (
            _relative_change(previous_omega, omega, arrays.source_weight)
        )
        postprocess_diagnostics["ordinary_pava_diagnostic_records"] = (
            ordinary_history_count
        )
        history.append(postprocess_diagnostics)

    final_diag = dict(history[-1])
    final_diag["converged"] = loop_converged
    final_diag["iterations"] = int(final_diag["iteration"])
    final_diag["diagnostic_records"] = len(history)
    return IsotonicCalibrationResult(
        method="pava",
        status="ok" if loop_converged else "max_iterations",
        source_weights=np.asarray(omega, dtype=np.float64),
        grid=np.asarray(final_fit.grid, dtype=np.float64),
        fitted_grid_values=np.asarray(final_values, dtype=np.float64),
        diagnostics=final_diag,
        history=history,
        config=cfg,
        runtime_sec=time.perf_counter() - started,
    )


def fit_isotonic_fori_pava_weights(**kwargs: object) -> Array:
    """Return only fitted source-row weights for convenience."""

    return fit_isotonic_fori_pava(**kwargs).source_weights


def _validate_config(config: IsotonicCalibrationConfig | None) -> IsotonicCalibrationConfig:
    cfg = IsotonicCalibrationConfig() if config is None else config
    if not isinstance(cfg, IsotonicCalibrationConfig):
        raise TypeError("config must be an IsotonicCalibrationConfig")
    if cfg.num_iterations <= 0:
        raise ValueError("num_iterations must be positive")
    if not np.isfinite(cfg.tolerance) or cfg.tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    if cfg.direction not in {"increasing", "decreasing"}:
        raise ValueError("direction must be 'increasing' or 'decreasing'")
    if cfg.initialization not in {"unit", "score"}:
        raise ValueError("initialization must be 'unit' or 'score'")
    if cfg.support_policy not in {"error", "constant_extrapolation"}:
        raise ValueError("support_policy must be 'error' or 'constant_extrapolation'")
    if not cfg.normalize:
        raise ValueError("standard normalized occupancy calibration requires normalize=True")
    if not np.isfinite(cfg.positivity_floor) or cfg.positivity_floor < 0.0:
        raise ValueError("positivity_floor must be finite and nonnegative")
    if not (0.0 < cfg.fixed_point_damping <= 1.0):
        raise ValueError("fixed_point_damping must be in (0, 1]")
    if not np.isfinite(cfg.support_tol) or cfg.support_tol < 0.0:
        raise ValueError("support_tol must be finite and nonnegative")
    if (
        isinstance(cfg.minimum_boundary_block_observations, bool)
        or int(cfg.minimum_boundary_block_observations)
        != cfg.minimum_boundary_block_observations
        or int(cfg.minimum_boundary_block_observations) <= 0
    ):
        raise ValueError(
            "minimum_boundary_block_observations must be a positive integer"
        )
    return cfg


def _prepare_inputs(
    *,
    source_score: Array,
    next_score: Array,
    initial_score: Array,
    gamma: float,
    source_weight: Array | None,
    next_source_index: Array | None,
    next_weight: Array | None,
    initial_weight: Array | None,
    direction: Direction,
    support_tol: float,
    support_policy: SupportPolicy,
) -> _PreparedInputs:
    source_raw = _finite_vector(source_score, "source_score")
    next_raw = _finite_vector(next_score, "next_score")
    initial_raw = _finite_vector(initial_score, "initial_score")
    if source_raw.size == 0 or initial_raw.size == 0:
        raise ValueError("source_score and initial_score must be nonempty")
    gamma_f = float(gamma)
    if not (0.0 <= gamma_f < 1.0):
        raise ValueError("gamma must be in [0, 1)")
    source_prob = _probability_weights(source_weight, source_raw.size, "source_weight")
    initial_prob = _probability_weights(initial_weight, initial_raw.size, "initial_weight")

    if next_source_index is None:
        if next_raw.size != source_raw.size:
            raise ValueError("next_source_index is required when next_score length differs from source_score")
        predecessor = np.arange(source_raw.size, dtype=np.int64)
    else:
        predecessor = np.asarray(next_source_index, dtype=np.int64).reshape(-1)
        if predecessor.size != next_raw.size:
            raise ValueError("next_source_index must have the same length as next_score")
        if np.any(predecessor < 0) or np.any(predecessor >= source_raw.size):
            raise ValueError("next_source_index contains an invalid source index")
    if next_weight is None:
        successor_joint = source_prob[predecessor].copy()
    else:
        successor_joint = _nonnegative_vector(next_weight, next_raw.size, "next_weight")
    row_mass = np.bincount(predecessor, weights=successor_joint, minlength=source_raw.size)
    if np.max(np.abs(row_mass - source_prob)) > max(10.0 * support_tol, 1e-10):
        raise ValueError("next_weight row sums must equal source_weight")

    source_z = _transform_score(source_raw, direction)
    next_z = _transform_score(next_raw, direction)
    initial_z = _transform_score(initial_raw, direction)
    lower = float(np.min(source_z))
    upper = float(np.max(source_z))
    next_outside = (next_z < lower) | (next_z > upper)
    initial_outside = (initial_z < lower) | (initial_z > upper)
    if support_policy == "error" and (np.any(next_outside) or np.any(initial_outside)):
        raise ValueError("target scores fall outside observed source-score support")
    if support_policy == "constant_extrapolation":
        next_z = np.clip(next_z, lower, upper)
        initial_z = np.clip(initial_z, lower, upper)

    # The paper's finite optimization class has knots only at observed
    # behavior scores. Successor and initial scores are evaluated through the
    # same paper-defined cells (t_{j-1}, t_j], with constant endpoint
    # extrapolation.
    # Avoiding target-only knots both matches that definition and reduces each
    # PAVA pass from up to 3n grid points to at most n.
    grid = np.unique(source_z)
    source_index = np.searchsorted(grid, source_z)
    next_index = np.clip(
        np.searchsorted(grid, next_z, side="left"), 0, grid.size - 1
    )
    initial_index = np.clip(
        np.searchsorted(grid, initial_z, side="left"), 0, grid.size - 1
    )
    source_exposure = np.bincount(source_index, weights=source_prob, minlength=grid.size).astype(np.float64)
    source_observation_count = np.bincount(source_index, minlength=grid.size).astype(np.int64)
    initial_mass = np.bincount(initial_index, weights=initial_prob, minlength=grid.size).astype(np.float64)
    return _PreparedInputs(
        source_score_raw=source_raw,
        source_score=source_z,
        next_score=next_z,
        initial_score=initial_z,
        source_weight=source_prob,
        next_source_index=predecessor,
        next_weight=successor_joint,
        initial_weight=initial_prob,
        grid=grid,
        source_grid_index=source_index,
        next_grid_index=next_index,
        initial_grid_index=initial_index,
        source_exposure=source_exposure,
        source_observation_count=source_observation_count,
        initial_mass=initial_mass,
        gamma=gamma_f,
        extrapolated_next_fraction=float(np.mean(next_outside)) if next_outside.size else 0.0,
        extrapolated_initial_mass=float(np.sum(initial_prob[initial_outside])),
    )


def _initial_weights(
    source_score: Array,
    source_weight: Array,
    *,
    initial_omega: Array | None,
    config: IsotonicCalibrationConfig,
) -> Array:
    if initial_omega is not None:
        omega = _finite_vector(initial_omega, "initial_omega")
        if omega.shape != source_score.shape:
            raise ValueError("initial_omega must have the same shape as source_score")
        if np.any(omega < 0.0):
            raise ValueError("initial_omega must be nonnegative")
    elif config.initialization == "score":
        omega = np.maximum(source_score, 0.0)
        if not np.any(omega > 0.0):
            omega = np.ones_like(source_score)
    else:
        omega = np.ones_like(source_score)
    return _floor_and_normalize(
        omega,
        source_weight,
        positivity_floor=config.positivity_floor,
        normalize=True,
        zero_tol=config.support_tol,
    )[0]


def _build_grid_fit(arrays: _PreparedInputs, omega: Array, *, support_tol: float) -> _GridFit:
    mass = float(np.sum(arrays.source_weight * omega))
    if mass <= support_tol:
        raise ValueError("current source weights have zero mass")
    successor_weight = arrays.next_weight * omega[arrays.next_source_index] / mass
    successor_mass = np.bincount(
        arrays.next_grid_index,
        weights=successor_weight,
        minlength=arrays.grid.size,
    ).astype(np.float64)
    target = (1.0 - arrays.gamma) * arrays.initial_mass + arrays.gamma * successor_mass
    _raise_if_unbounded(arrays.source_exposure, target, arrays.grid, support_tol=support_tol)
    return _GridFit(
        arrays.grid,
        arrays.source_exposure,
        target,
        arrays.source_observation_count,
    )


def _solve_generalized_pava(
    exposure: Array,
    target: Array,
    *,
    grid: Array,
    support_tol: float,
) -> Array:
    a = _finite_vector(exposure, "exposure")
    b = _finite_vector(target, "target")
    if a.shape != b.shape:
        raise ValueError("exposure and target must have the same shape")
    if np.any(a < -support_tol) or np.any(b < -support_tol):
        raise ValueError("exposure and target must be nonnegative")
    a = np.maximum(a, 0.0)
    b = np.maximum(b, 0.0)
    _raise_if_unbounded(a, b, grid, support_tol=support_tol)
    if np.all(a > support_tol):
        # For positive exposure, generalized KL-PAVA has block value
        # sum(target)/sum(exposure), exactly the weighted-L2 isotonic solution
        # for response target/exposure. Use sklearn's compiled public solver
        # when available; the dependency remains optional.
        try:
            from sklearn.isotonic import isotonic_regression  # noqa: PLC0415
        except ImportError:
            pass
        else:
            values = np.asarray(
                isotonic_regression(
                    b / a,
                    sample_weight=a,
                    increasing=True,
                ),
                dtype=np.float64,
            )
            return values
    blocks: list[list[float | int]] = []
    for index, (a_value, b_value) in enumerate(zip(a, b, strict=True)):
        a_mass = float(a_value)
        b_mass = float(b_value)
        value = b_mass / a_mass if a_mass > support_tol else (np.inf if b_mass > support_tol else 0.0)
        blocks.append([index, index + 1, a_mass, b_mass, value])
        while len(blocks) >= 2 and float(blocks[-2][4]) > float(blocks[-1][4]) + _ORDER_TOL:
            right = blocks.pop()
            left = blocks.pop()
            a_mass = float(left[2]) + float(right[2])
            b_mass = float(left[3]) + float(right[3])
            value = b_mass / a_mass if a_mass > support_tol else (np.inf if b_mass > support_tol else 0.0)
            blocks.append([int(left[0]), int(right[1]), a_mass, b_mass, value])
    values = np.empty_like(a)
    for start, stop, _, _, value in blocks:
        values[int(start) : int(stop)] = float(value)
    return values


def _pool_small_boundary_pava_blocks(
    values: Array,
    exposure: Array,
    target: Array,
    *,
    observation_count: Array | None,
    minimum_boundary_block_observations: int,
    support_tol: float,
) -> Array:
    """Pool only undersupported endpoint blocks into their neighbors."""

    minimum = int(minimum_boundary_block_observations)
    fitted = np.asarray(values, dtype=np.float64).reshape(-1)
    if minimum <= 1:
        return fitted
    if observation_count is None:
        raise ValueError(
            "observation_count is required when minimum_boundary_block_observations exceeds one"
        )
    counts = np.asarray(observation_count, dtype=np.int64).reshape(-1)
    a = np.asarray(exposure, dtype=np.float64).reshape(-1)
    b = np.asarray(target, dtype=np.float64).reshape(-1)
    if not (fitted.shape == counts.shape == a.shape == b.shape):
        raise ValueError("PAVA values, masses, and observation counts must have matching shapes")
    if np.any(counts < 0):
        raise ValueError("observation_count must be nonnegative")
    if int(np.sum(counts)) < minimum:
        raise ValueError(
            "minimum_boundary_block_observations cannot exceed the source sample size"
        )

    boundaries = np.concatenate(
        ([0], np.flatnonzero(np.abs(np.diff(fitted)) > _ORDER_TOL) + 1, [fitted.size])
    )
    blocks: list[list[float | int]] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        start_i = int(start)
        stop_i = int(stop)
        a_mass = float(np.sum(a[start_i:stop_i]))
        b_mass = float(np.sum(b[start_i:stop_i]))
        blocks.append(
            [
                start_i,
                stop_i,
                a_mass,
                b_mass,
                int(np.sum(counts[start_i:stop_i])),
                _block_value(a_mass, b_mass, support_tol=support_tol),
            ]
        )

    while len(blocks) > 1 and int(blocks[0][4]) < minimum:
        blocks[:2] = [
            _merge_pava_blocks(blocks[0], blocks[1], support_tol=support_tol)
        ]
    while len(blocks) > 1 and int(blocks[-1][4]) < minimum:
        blocks[-2:] = [
            _merge_pava_blocks(blocks[-2], blocks[-1], support_tol=support_tol)
        ]

    pooled = np.empty_like(fitted)
    for start, stop, _, _, _, value in blocks:
        pooled[int(start) : int(stop)] = float(value)
    return pooled


def _block_value(a_mass: float, b_mass: float, *, support_tol: float) -> float:
    if a_mass <= support_tol:
        return np.inf if b_mass > support_tol else 0.0
    return b_mass / a_mass


def _merge_pava_blocks(
    left: list[float | int],
    right: list[float | int],
    *,
    support_tol: float,
) -> list[float | int]:
    a_mass = float(left[2]) + float(right[2])
    b_mass = float(left[3]) + float(right[3])
    return [
        int(left[0]),
        int(right[1]),
        a_mass,
        b_mass,
        int(left[4]) + int(right[4]),
        _block_value(a_mass, b_mass, support_tol=support_tol),
    ]


def _raise_if_unbounded(exposure: Array, target: Array, grid: Array, *, support_tol: float) -> None:
    positive_exposure = np.flatnonzero(np.asarray(exposure) > support_tol)
    if positive_exposure.size == 0:
        raise ValueError("isotonic calibration objective has no source-score exposure")
    last = int(positive_exposure[-1])
    upper_mass = float(np.sum(np.asarray(target)[last + 1 :]))
    if upper_mass > support_tol:
        first_score = float(np.asarray(grid)[last + 1])
        raise ValueError(
            "isotonic calibration objective is unbounded above the largest exposed "
            f"source score (first unsupported score {first_score:g})"
        )


def _floor_and_normalize(
    values: Array,
    weights: Array,
    *,
    positivity_floor: float,
    normalize: bool,
    zero_tol: float,
) -> tuple[Array, float]:
    out = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    prob = np.asarray(weights, dtype=np.float64).reshape(-1)
    if out.shape != prob.shape:
        raise ValueError("values and weights must have the same shape")
    floored = out < positivity_floor
    if positivity_floor > 0.0:
        out = np.maximum(out, positivity_floor)
    floor_fraction = float(np.sum(prob * floored))
    if normalize:
        mass = float(np.sum(prob * out))
        if not np.isfinite(mass) or mass <= zero_tol:
            raise ValueError("cannot normalize nonfinite or zero-mass calibrated weights")
        out /= mass
    return out, floor_fraction


def _diagnostics(
    arrays: _PreparedInputs,
    grid_fit: _GridFit,
    values: Array,
    source_weights: Array,
    *,
    iteration: int,
    relative_change: float,
    floor_fraction: float,
    support_tol: float,
    required_minimum_boundary_block_observations: int,
    final_projection: bool = False,
) -> dict[str, float | int | str | bool]:
    differences = np.diff(values)
    source_mass = float(np.sum(arrays.source_weight * source_weights))
    second = float(np.sum(arrays.source_weight * source_weights**2))
    residuals: list[float] = []
    start = 0
    for stop in range(1, values.size + 1):
        if stop == values.size or abs(float(values[stop]) - float(values[start])) > _ORDER_TOL:
            residuals.append(
                float(np.sum(grid_fit.exposure[start:stop]) * values[start] - np.sum(grid_fit.target[start:stop]))
            )
            start = stop
    positive = source_weights[source_weights > 0.0]
    median = float(np.median(positive)) if positive.size else 0.0
    fitted_block_counts: list[int] = []
    start = 0
    for stop in range(1, values.size + 1):
        if stop == values.size or abs(float(values[stop]) - float(values[start])) > _ORDER_TOL:
            fitted_block_counts.append(
                int(np.sum(arrays.source_observation_count[start:stop]))
            )
            start = stop
    return {
        "method": "pava",
        "estimand": "normalized_discounted",
        "iteration": iteration,
        "relative_change": float(relative_change),
        "source_weighted_mass": source_mass,
        "normalization_error": source_mass - 1.0,
        "grid_size": int(values.size),
        "num_blocks": int(1 + np.sum(np.abs(differences) > _ORDER_TOL)),
        "left_boundary_block_observations": int(
            fitted_block_counts[0] if fitted_block_counts else 0
        ),
        "right_boundary_block_observations": int(
            fitted_block_counts[-1] if fitted_block_counts else 0
        ),
        "minimum_fitted_block_observations": int(
            min(fitted_block_counts) if fitted_block_counts else 0
        ),
        "required_minimum_boundary_block_observations": int(
            required_minimum_boundary_block_observations
        ),
        "monotone_violations": int(np.sum(differences < -1e-10)),
        "monotone_min_diff": float(np.min(differences)) if differences.size else 0.0,
        "block_balance_max_abs": float(np.max(np.abs(residuals))) if residuals else 0.0,
        "effective_sample_size_fraction": source_mass**2 / max(second, 1e-12),
        "weight_min": float(np.min(source_weights)),
        "weight_q50": median,
        "weight_q90": float(np.quantile(source_weights, 0.90)),
        "weight_q99": float(np.quantile(source_weights, 0.99)),
        "weight_max": float(np.max(source_weights)),
        "max_to_median_ratio": float(np.max(source_weights) / max(median, 1e-12)),
        "floor_fraction": float(floor_fraction),
        "target_mass": float(np.sum(grid_fit.target)),
        "source_exposure_mass": float(np.sum(grid_fit.exposure)),
        "constant_extrapolation_next_fraction": arrays.extrapolated_next_fraction,
        "constant_extrapolation_initial_mass": arrays.extrapolated_initial_mass,
        "final_projection": bool(final_projection),
        "support_tolerance": float(support_tol),
    }


def _predict_step(score: Array, grid: Array, values: Array) -> Array:
    z = np.asarray(score, dtype=np.float64).reshape(-1)
    knots = np.asarray(grid, dtype=np.float64).reshape(-1)
    fitted = np.asarray(values, dtype=np.float64).reshape(-1)
    if knots.size == 0 or knots.shape != fitted.shape:
        raise ValueError("grid and values must be nonempty with matching shapes")
    indices = np.searchsorted(knots, z, side="left")
    return fitted[np.clip(indices, 0, knots.size - 1)]


def _relative_change(old: Array, new: Array, weights: Array) -> float:
    delta = np.asarray(new) - np.asarray(old)
    numerator = float(np.sqrt(np.sum(weights * delta**2)))
    denominator = float(np.sqrt(np.sum(weights * np.asarray(old) ** 2)))
    return numerator / max(denominator, 1e-12)


def _transform_score(score: Array, direction: Direction) -> Array:
    return score.copy() if direction == "increasing" else -score


def _finite_vector(value: Array, name: str) -> Array:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _nonnegative_vector(value: Array, expected_size: int, name: str) -> Array:
    array = _finite_vector(value, name)
    if array.size != expected_size:
        raise ValueError(f"{name} must have length {expected_size}")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _probability_weights(value: Array | None, size: int, name: str) -> Array:
    if size <= 0:
        raise ValueError(f"{name} cannot be constructed for an empty sample")
    if value is None:
        return np.full(size, 1.0 / size, dtype=np.float64)
    weights = _nonnegative_vector(value, size, name)
    mass = float(np.sum(weights))
    if mass <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return weights / mass


__all__ = [
    "IsotonicCalibrationConfig",
    "IsotonicCalibrationResult",
    "fit_isotonic_fori_pava",
    "fit_isotonic_fori_pava_weights",
]
