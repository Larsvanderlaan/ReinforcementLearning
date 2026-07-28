"""Evaluation metrics for normalized occupancy-ratio calibration experiments.

The helpers in this module score already-produced candidate weights.  They do
not prescribe how an estimator is fitted, calibrated, or aggregated across
folds.  Oracle ratios enter only the explicitly controlled ratio metrics; the
Bellman calibration score is truth-free.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class CrossMomentScaleEstimate:
    """Cross-moment estimate at one candidate-specific histogram resolution.

    Attributes
    ----------
    requested_bins:
        Requested number of quantile bins before ties or near-constant values
        are collapsed.
    effective_bins:
        Number of nonempty bins induced by the basis sample.
    bin_upper_edges:
        Upper observed boundary of every bin except the last.
    gram_diagonal:
        Diagonal of the empirical basis Gram matrix.  Candidate-specific
        quantile indicators are disjoint, so the off-diagonal entries are zero.
    ridge:
        Added diagonal ridge, equal to ``ridge_scale * trace(G) / K``.
    moment_a, moment_b:
        Bellman moment vectors on the two episode-group audit halves.
    signed_squared_error:
        The cross-product ``b_A.T @ (G + ridge I)^-1 @ b_B``.  It is preserved
        with its sign because finite-sample cross-moment estimates can be
        negative.
    positive_part_root_error:
        Square root of the positive part of ``signed_squared_error``.  This is
        intended for plots only; inference should retain the signed estimate.
    """

    requested_bins: int
    effective_bins: int
    bin_upper_edges: tuple[float, ...]
    gram_diagonal: tuple[float, ...]
    ridge: float
    moment_a: tuple[float, ...]
    moment_b: tuple[float, ...]
    signed_squared_error: float
    positive_part_root_error: float


@dataclass(frozen=True)
class BellmanCrossMomentMetrics:
    """Bellman calibration metrics and audit-split diagnostics.

    ``basis_audit_groups_disjoint`` is ``None`` when basis group identifiers
    were not supplied.  A value of ``False`` does not prevent computation, but
    it means the basis and Gram matrix are data-dependent on the audit sample;
    in that case the usual conditional debiasing interpretation of the A/B
    cross-product is not justified without an additional argument.
    """

    base: CrossMomentScaleEstimate
    half_bins: CrossMomentScaleEstimate
    double_bins: CrossMomentScaleEstimate
    mass: CrossMomentScaleEstimate
    automatic_bin_count: int
    basis_near_constant: bool
    n_basis: int
    n_audit_transitions: int
    n_audit_initial: int
    n_audit_groups: int
    n_transition_groups: int
    n_initial_groups: int
    n_union_groups: int
    n_groups_a: int
    n_groups_b: int
    n_transition_groups_a: int
    n_transition_groups_b: int
    n_initial_groups_a: int
    n_initial_groups_b: int
    split_seed: int
    basis_audit_groups_disjoint: bool | None


@dataclass(frozen=True)
class ControlledRatioMetrics:
    """Untruncated controlled-setting occupancy-ratio errors."""

    n: int
    mse: float
    relative_mse: float
    rmse: float
    l1: float
    generalized_kl: float


@dataclass(frozen=True)
class OracleFloorKLPoint:
    """One oracle-denominator floor sensitivity point."""

    multiplier: float
    floor: float
    generalized_kl: float


@dataclass(frozen=True)
class OracleFloorKLSensitivity:
    """Generalized-KL sensitivity to oracle-only finite-support flooring."""

    smallest_positive_oracle: float
    exact_generalized_kl: float
    points: tuple[OracleFloorKLPoint, ...]


@dataclass(frozen=True)
class OccupancyFunctionalPanelMetadata:
    """Reconstruction metadata for a deterministic bounded RFF reward panel."""

    reward_count: int
    state_dimension: int
    action_dimension: int
    input_dimension: int
    bandwidths: tuple[float, ...]
    rewards_per_bandwidth: tuple[int, ...]
    seed: int
    standardization_scale_floor: float
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    reward_lower_bound: float
    reward_upper_bound: float
    evaluation_batch_size: int
    panel_sha256: str


@dataclass(frozen=True)
class MultiRewardOccupancyFunctionalMetrics:
    """Truth-free errors over a deterministic panel of bounded rewards.

    The two target pools are used only for evaluation.  Their signed error
    cross-product estimates the squared functional error without the additive
    target-sample variance of a same-pool squared error.  The pooled summaries
    are descriptive plug-in errors and are not debiased.
    """

    signed_cross_pool_mse: float
    positive_part_root_mse: float
    pooled_rmse: float
    pooled_mae: float
    pooled_max_absolute_error: float
    source_functionals: tuple[float, ...]
    target_functionals_a: tuple[float, ...]
    target_functionals_b: tuple[float, ...]
    error_a: tuple[float, ...]
    error_b: tuple[float, ...]
    pooled_error: tuple[float, ...]
    n_source: int
    n_target_a: int
    n_target_b: int
    panel: OccupancyFunctionalPanelMetadata


def estimate_bellman_cross_moment_error(
    *,
    basis_candidate_weights: Array,
    audit_current_weights: Array,
    audit_next_weights: Array,
    audit_initial_weights: Array,
    audit_transition_group_ids: Array,
    audit_initial_group_ids: Array,
    gamma: float,
    basis_group_ids: Array | None = None,
    bin_count: int | None = None,
    split_seed: int = 0,
    ridge_scale: float = 1e-8,
    near_constant_rtol: float = 1e-8,
    near_constant_atol: float = 1e-12,
) -> BellmanCrossMomentMetrics:
    """Estimate a cross-moment Bellman calibration error.

    For candidate occupancy weights ``omega`` and a bin indicator ``g``, the
    empirical moment on each audit half is

    ``(1-gamma) E_0[g(omega(X0))]``
    ``+ gamma E[omega(X) g(omega(X+))]``
    ``- E[omega(X) g(omega(X))]``.

    Candidate-specific quantile bins and their Gram matrix are built only from
    ``basis_candidate_weights``.  Audit episodes are split deterministically
    into two nonoverlapping group halves, and the reported signed estimate is
    ``b_A.T @ (G + lambda I)^-1 @ b_B``.  Conditional on a fixed basis/Gram and
    independent episode groups, the cross-product removes the additive
    within-audit moment-noise bias of a plug-in squared norm.

    The function is agnostic to estimator construction.  If the basis was
    built from a separate group sample, pass ``basis_group_ids`` to make that
    disjointness auditable.  Reusing audit groups for basis construction is
    allowed computationally and is reported, but requires a separate
    dependence argument before calling the score debiased.

    Parameters
    ----------
    basis_candidate_weights:
        One-dimensional candidate weights used only to construct quantile bins
        and the empirical Gram matrix.
    audit_current_weights, audit_next_weights, audit_initial_weights:
        Candidate predictions at audit ``X``, ``X+``, and ``X0`` points.
    audit_transition_group_ids, audit_initial_group_ids:
        Episode or independent-sample identifiers.  The sets may differ.  The
        split is constructed on their union, so any identifier shared by the
        transition and initial samples remains in the same audit half.
    gamma:
        Discount in ``[0, 1)``.
    basis_group_ids:
        Optional group identifiers for ``basis_candidate_weights``.  When
        supplied, overlap with audit groups is recorded in the result.
    bin_count:
        Main requested bin count.  By default this is
        ``min(20, max(5, floor(n_audit**(1/3))))``.
    split_seed:
        Seed mixed into a stable hash of episode identifiers.
    ridge_scale:
        Nonnegative multiplier in ``lambda = ridge_scale * trace(G) / K``.
    near_constant_rtol, near_constant_atol:
        Tolerances for collapsing a numerically constant candidate to one bin.

    Returns
    -------
    BellmanCrossMomentMetrics
        Main, half-resolution, double-resolution, and one-bin mass estimates.

    Raises
    ------
    ValueError
        If weights, discounts, tuning constants, or group arrays are invalid.
    """

    basis = _as_nonnegative_finite_vector(basis_candidate_weights, name="basis_candidate_weights")
    current = _as_nonnegative_finite_vector(audit_current_weights, name="audit_current_weights")
    next_weight = _as_nonnegative_finite_vector(audit_next_weights, name="audit_next_weights")
    initial = _as_nonnegative_finite_vector(audit_initial_weights, name="audit_initial_weights")
    if next_weight.size != current.size:
        raise ValueError("audit_next_weights must have the same length as audit_current_weights")
    discount = _validate_gamma(gamma)
    seed = _validate_integer(split_seed, name="split_seed")
    ridge_multiplier = _validate_nonnegative_finite(ridge_scale, name="ridge_scale")
    relative_tolerance = _validate_nonnegative_finite(near_constant_rtol, name="near_constant_rtol")
    absolute_tolerance = _validate_nonnegative_finite(near_constant_atol, name="near_constant_atol")

    transition_keys = _group_keys(
        audit_transition_group_ids,
        expected_size=current.size,
        name="audit_transition_group_ids",
    )
    initial_keys = _group_keys(
        audit_initial_group_ids,
        expected_size=initial.size,
        name="audit_initial_group_ids",
    )
    transition_groups = set(transition_keys)
    initial_groups = set(initial_keys)
    if len(transition_groups) < 2:
        raise ValueError("at least two audit transition groups are required")
    if len(initial_groups) < 2:
        raise ValueError("at least two audit initial groups are required")
    audit_groups = transition_groups | initial_groups

    groups_a, groups_b = _deterministic_group_halves(
        transition_groups,
        initial_groups,
        seed=seed,
    )
    transition_is_a = np.fromiter(
        (key in groups_a for key in transition_keys),
        dtype=np.bool_,
        count=current.size,
    )
    initial_is_a = np.fromiter(
        (key in groups_a for key in initial_keys),
        dtype=np.bool_,
        count=initial.size,
    )
    if not np.any(transition_is_a) or np.all(transition_is_a):
        raise RuntimeError("deterministic group split produced an empty transition half")
    if not np.any(initial_is_a) or np.all(initial_is_a):
        raise RuntimeError("deterministic group split produced an empty initial half")

    automatic_bins = min(20, max(5, int(np.floor(current.size ** (1.0 / 3.0)))))
    requested_bins = automatic_bins if bin_count is None else _validate_positive_integer(bin_count, name="bin_count")
    half_requested = max(1, requested_bins // 2)
    double_requested = 2 * requested_bins
    near_constant = _is_near_constant(basis, rtol=relative_tolerance, atol=absolute_tolerance)

    estimate_kwargs = {
        "basis": basis,
        "current": current,
        "next_weight": next_weight,
        "initial": initial,
        "transition_is_a": transition_is_a,
        "initial_is_a": initial_is_a,
        "gamma": discount,
        "ridge_scale": ridge_multiplier,
        "near_constant": near_constant,
    }
    base = _estimate_at_resolution(requested_bins=requested_bins, **estimate_kwargs)
    half = _estimate_at_resolution(requested_bins=half_requested, **estimate_kwargs)
    double = _estimate_at_resolution(requested_bins=double_requested, **estimate_kwargs)
    mass = _estimate_at_resolution(requested_bins=1, **estimate_kwargs)

    basis_disjoint: bool | None = None
    if basis_group_ids is not None:
        basis_keys = _group_keys(
            basis_group_ids,
            expected_size=basis.size,
            name="basis_group_ids",
        )
        basis_disjoint = set(basis_keys).isdisjoint(audit_groups)

    return BellmanCrossMomentMetrics(
        base=base,
        half_bins=half,
        double_bins=double,
        mass=mass,
        automatic_bin_count=automatic_bins,
        basis_near_constant=near_constant,
        n_basis=int(basis.size),
        n_audit_transitions=int(current.size),
        n_audit_initial=int(initial.size),
        n_audit_groups=len(audit_groups),
        n_transition_groups=len(transition_groups),
        n_initial_groups=len(initial_groups),
        n_union_groups=len(audit_groups),
        n_groups_a=len(groups_a),
        n_groups_b=len(groups_b),
        n_transition_groups_a=len(transition_groups & groups_a),
        n_transition_groups_b=len(transition_groups & groups_b),
        n_initial_groups_a=len(initial_groups & groups_a),
        n_initial_groups_b=len(initial_groups & groups_b),
        split_seed=seed,
        basis_audit_groups_disjoint=basis_disjoint,
    )


def estimate_multi_reward_occupancy_functional_error(
    *,
    source_states: Array,
    source_actions: Array,
    candidate_weights: Array,
    target_states_a: Array,
    target_actions_a: Array,
    target_states_b: Array,
    target_actions_b: Array,
    reward_count: int = 64,
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0),
    seed: int = 0,
    standardization_scale_floor: float = 1e-6,
    evaluation_batch_size: int = 16_384,
) -> MultiRewardOccupancyFunctionalMetrics:
    """Compare weighted source and target occupancies over bounded rewards.

    A deterministic panel of random Fourier feature (RFF) rewards is built on
    the concatenated state-action input.  Source-derived coordinate means and
    standard deviations standardize all three samples.  Each reward is

    ``r_j(x) = cos(omega_j.T @ standardized(x) + phase_j)``,

    so every reward lies in ``[-1, 1]``.  Bandwidths are assigned cyclically to
    the panel, and ``omega_j`` is Gaussian with coordinate standard deviation
    equal to the inverse assigned bandwidth.

    For each reward, the candidate occupancy functional is
    ``mean_source(candidate_weight * reward)``.  Candidate weights are not
    renormalized: doing so would conceal occupancy-mass error.  Let ``e_A`` and
    ``e_B`` be its errors against independent target occupancy pools.  The
    primary signed statistic is ``mean(e_A * e_B)``.  Conditional on the
    source rows, candidate, and fixed reward panel, independent target pools
    remove the additive target Monte Carlo variance from this cross-product.
    Its positive-part square root is supplied only as a descriptive plot
    quantity.  RMSE, MAE, and maximum error use the size-weighted pooled target
    mean and retain ordinary plug-in target-sample noise.

    Parameters
    ----------
    source_states, source_actions:
        Source-distribution state and action rows.  One-dimensional inputs are
        interpreted as a single coordinate; two-dimensional inputs are
        interpreted as feature matrices.
    candidate_weights:
        One-dimensional nonnegative finite candidate occupancy weights, with
        one value per source row.
    target_states_a, target_actions_a, target_states_b, target_actions_b:
        Two nonempty, independent target-occupancy evaluation pools.  State and
        action dimensions must match the source arrays.
    reward_count:
        Positive number of bounded RFF rewards.
    bandwidths:
        Nonempty sequence of positive finite RFF bandwidths.
    seed:
        Integer seed for the deterministic reward panel.
    standardization_scale_floor:
        Positive lower bound for source coordinate standard deviations.
    evaluation_batch_size:
        Positive row batch size used to avoid materializing a full
        ``rows x reward_count`` matrix.

    Returns
    -------
    MultiRewardOccupancyFunctionalMetrics
        Cross-pool and pooled functional errors, per-reward audit values, sample
        sizes, and complete panel reconstruction metadata.

    Raises
    ------
    ValueError
        If an input is empty, nonfinite, shape-incompatible, or if a tuning
        value is invalid.
    """

    source_state = _as_feature_matrix(source_states, name="source_states")
    source_action = _as_feature_matrix(source_actions, name="source_actions")
    target_state_a = _as_feature_matrix(target_states_a, name="target_states_a")
    target_action_a = _as_feature_matrix(target_actions_a, name="target_actions_a")
    target_state_b = _as_feature_matrix(target_states_b, name="target_states_b")
    target_action_b = _as_feature_matrix(target_actions_b, name="target_actions_b")
    _validate_state_action_rows(
        source_state,
        source_action,
        state_name="source_states",
        action_name="source_actions",
    )
    _validate_state_action_rows(
        target_state_a,
        target_action_a,
        state_name="target_states_a",
        action_name="target_actions_a",
    )
    _validate_state_action_rows(
        target_state_b,
        target_action_b,
        state_name="target_states_b",
        action_name="target_actions_b",
    )
    if target_state_a.shape[1] != source_state.shape[1] or target_state_b.shape[1] != source_state.shape[1]:
        raise ValueError("target state dimensions must match source_states")
    if target_action_a.shape[1] != source_action.shape[1] or target_action_b.shape[1] != source_action.shape[1]:
        raise ValueError("target action dimensions must match source_actions")

    weight = _as_nonnegative_finite_vector(candidate_weights, name="candidate_weights")
    if weight.size != source_state.shape[0]:
        raise ValueError(f"candidate_weights must have length {source_state.shape[0]}")
    n_rewards = _validate_positive_integer(reward_count, name="reward_count")
    panel_seed = _validate_integer(seed, name="seed")
    scale_floor = _validate_positive_finite(
        standardization_scale_floor,
        name="standardization_scale_floor",
    )
    batch_size = _validate_positive_integer(
        evaluation_batch_size,
        name="evaluation_batch_size",
    )
    panel_bandwidths = _positive_finite_tuple(bandwidths, name="bandwidths")

    source_input = np.concatenate((source_state, source_action), axis=1)
    target_input_a = np.concatenate((target_state_a, target_action_a), axis=1)
    target_input_b = np.concatenate((target_state_b, target_action_b), axis=1)
    center = np.mean(source_input, axis=0, dtype=np.float64)
    scale = np.std(source_input, axis=0, dtype=np.float64)
    scale = np.maximum(scale, scale_floor)

    bandwidth_index = np.arange(n_rewards, dtype=np.int64) % len(panel_bandwidths)
    assigned_bandwidth = np.asarray(panel_bandwidths, dtype=np.float64)[bandwidth_index]
    rng = np.random.default_rng(panel_seed)
    frequency = rng.standard_normal((n_rewards, source_input.shape[1]))
    frequency /= assigned_bandwidth[:, None]
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_rewards)

    source_mean = _bounded_rff_mean(
        source_input,
        center=center,
        scale=scale,
        frequency=frequency,
        phase=phase,
        row_weight=weight,
        batch_size=batch_size,
    )
    target_mean_a = _bounded_rff_mean(
        target_input_a,
        center=center,
        scale=scale,
        frequency=frequency,
        phase=phase,
        row_weight=None,
        batch_size=batch_size,
    )
    target_mean_b = _bounded_rff_mean(
        target_input_b,
        center=center,
        scale=scale,
        frequency=frequency,
        phase=phase,
        row_weight=None,
        batch_size=batch_size,
    )
    error_a = source_mean - target_mean_a
    error_b = source_mean - target_mean_b
    target_pooled = (
        target_input_a.shape[0] * target_mean_a
        + target_input_b.shape[0] * target_mean_b
    ) / (target_input_a.shape[0] + target_input_b.shape[0])
    pooled_error = source_mean - target_pooled

    signed_mse = float(np.mean(error_a * error_b))
    pooled_squared_error = float(np.mean(np.square(pooled_error)))
    reward_counts = np.bincount(
        bandwidth_index,
        minlength=len(panel_bandwidths),
    )
    panel_digest = _rff_panel_digest(
        center=center,
        scale=scale,
        frequency=frequency,
        phase=phase,
        bandwidths=panel_bandwidths,
        seed=panel_seed,
    )
    metadata = OccupancyFunctionalPanelMetadata(
        reward_count=n_rewards,
        state_dimension=int(source_state.shape[1]),
        action_dimension=int(source_action.shape[1]),
        input_dimension=int(source_input.shape[1]),
        bandwidths=panel_bandwidths,
        rewards_per_bandwidth=tuple(int(value) for value in reward_counts),
        seed=panel_seed,
        standardization_scale_floor=scale_floor,
        feature_center=tuple(float(value) for value in center),
        feature_scale=tuple(float(value) for value in scale),
        reward_lower_bound=-1.0,
        reward_upper_bound=1.0,
        evaluation_batch_size=batch_size,
        panel_sha256=panel_digest,
    )
    return MultiRewardOccupancyFunctionalMetrics(
        signed_cross_pool_mse=signed_mse,
        positive_part_root_mse=float(np.sqrt(max(signed_mse, 0.0))),
        pooled_rmse=float(np.sqrt(pooled_squared_error)),
        pooled_mae=float(np.mean(np.abs(pooled_error))),
        pooled_max_absolute_error=float(np.max(np.abs(pooled_error))),
        source_functionals=tuple(float(value) for value in source_mean),
        target_functionals_a=tuple(float(value) for value in target_mean_a),
        target_functionals_b=tuple(float(value) for value in target_mean_b),
        error_a=tuple(float(value) for value in error_a),
        error_b=tuple(float(value) for value in error_b),
        pooled_error=tuple(float(value) for value in pooled_error),
        n_source=int(source_input.shape[0]),
        n_target_a=int(target_input_a.shape[0]),
        n_target_b=int(target_input_b.shape[0]),
        panel=metadata,
    )


def generalized_kl_divergence(
    estimated_ratio: Array,
    oracle_ratio: Array,
    *,
    sample_weight: Array | None = None,
) -> float:
    """Return ``D_nu(estimated_ratio || oracle_ratio)`` without flooring.

    The pointwise divergence is

    ``estimated * log(estimated / oracle) - estimated + oracle``.

    Extended-real conventions are exact: ``0 log(0 / q) = 0`` for ``q >= 0``;
    a positive estimate at a zero oracle value contributes ``+inf``; and two
    zero values contribute zero.  Negative ratios, nonfinite ratios, and invalid
    sample weights raise ``ValueError``.

    Parameters
    ----------
    estimated_ratio, oracle_ratio:
        One-dimensional nonnegative finite arrays of equal length.
    sample_weight:
        Optional nonnegative finite reference-distribution weights.  They are
        normalized internally; zero-weight rows do not affect extended values.

    Returns
    -------
    float
        The generalized KL divergence, possibly ``+inf``.
    """

    estimated, oracle, normalized_weight = _ratio_inputs(estimated_ratio, oracle_ratio, sample_weight=sample_weight)
    active = normalized_weight > 0.0
    estimated = estimated[active]
    oracle = oracle[active]
    weight = normalized_weight[active]

    contribution = np.empty_like(estimated)
    estimated_zero = estimated == 0.0
    oracle_zero = oracle == 0.0
    contribution[estimated_zero] = oracle[estimated_zero]
    infinite = (~estimated_zero) & oracle_zero
    contribution[infinite] = np.inf
    regular = (~estimated_zero) & (~oracle_zero)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        contribution[regular] = (
            estimated[regular] * (np.log(estimated[regular]) - np.log(oracle[regular]))
            - estimated[regular]
            + oracle[regular]
        )
    if np.any(np.isposinf(contribution)):
        return float("inf")
    return float(np.dot(weight, contribution))


def controlled_ratio_errors(
    estimated_ratio: Array,
    oracle_ratio: Array,
    *,
    sample_weight: Array | None = None,
) -> ControlledRatioMetrics:
    """Compute primary untruncated ratio errors in a controlled setting.

    Relative MSE is ``E[(estimated-oracle)^2] / E[oracle^2]``.  It is zero
    when both numerator and denominator are zero, and ``+inf`` when only the
    denominator is zero.  MSE, RMSE, and L1 are never clipped or floored.

    Parameters
    ----------
    estimated_ratio, oracle_ratio:
        One-dimensional nonnegative finite arrays of equal length.
    sample_weight:
        Optional nonnegative finite reference-distribution weights.

    Returns
    -------
    ControlledRatioMetrics
        MSE, relative MSE, RMSE, L1, and exact generalized KL.
    """

    estimated, oracle, normalized_weight = _ratio_inputs(estimated_ratio, oracle_ratio, sample_weight=sample_weight)
    active = normalized_weight > 0.0
    estimated_active = estimated[active]
    oracle_active = oracle[active]
    weight_active = normalized_weight[active]
    with np.errstate(over="ignore", invalid="ignore"):
        squared_error = np.square(estimated_active - oracle_active)
    mse = _weighted_nonnegative_extended_mean(squared_error, weight_active)
    relative_mse = _stable_relative_mse(estimated_active, oracle_active, weight_active)
    l1 = float(np.dot(weight_active, np.abs(estimated_active - oracle_active)))
    return ControlledRatioMetrics(
        n=int(estimated.size),
        mse=mse,
        relative_mse=relative_mse,
        rmse=float(np.sqrt(mse)),
        l1=l1,
        generalized_kl=generalized_kl_divergence(estimated, oracle, sample_weight=normalized_weight),
    )


def oracle_floor_kl_sensitivity(
    estimated_ratio: Array,
    oracle_ratio: Array,
    *,
    oracle_is_exact_finite_support: bool,
    sample_weight: Array | None = None,
) -> OracleFloorKLSensitivity:
    """Compute the prespecified oracle-only generalized-KL floor sensitivity.

    This diagnostic is deliberately guarded: callers must affirm that the
    oracle is exact and defined on finite support.  Let ``epsilon`` be the
    smallest positive oracle ratio on a positive-reference-mass support point.
    The function reports generalized KL after replacing only the oracle ratio
    by ``max(oracle, c * epsilon)`` for ``c`` in ``{0.1, 1, 10}``.  Candidate
    predictions are never floored, and MSE is not part of this sensitivity.

    Parameters
    ----------
    estimated_ratio, oracle_ratio:
        Controlled-setting candidate and exact oracle ratios.
    oracle_is_exact_finite_support:
        Must be explicitly ``True``.  Continuous or Monte Carlo truth is not a
        valid input to this diagnostic.
    sample_weight:
        Optional nonnegative finite reference-distribution weights.

    Returns
    -------
    OracleFloorKLSensitivity
        Exact KL and the three prespecified oracle-floor sensitivity values.
    """

    if oracle_is_exact_finite_support is not True:
        raise ValueError("oracle floor sensitivity is restricted to exact finite-support truth")
    estimated, oracle, normalized_weight = _ratio_inputs(estimated_ratio, oracle_ratio, sample_weight=sample_weight)
    active_positive = (normalized_weight > 0.0) & (oracle > 0.0)
    if not np.any(active_positive):
        raise ValueError("oracle_ratio must be positive on at least one positive-mass support point")
    epsilon = float(np.min(oracle[active_positive]))
    exact = generalized_kl_divergence(estimated, oracle, sample_weight=normalized_weight)
    points = []
    for multiplier in (0.1, 1.0, 10.0):
        floor = multiplier * epsilon
        floored_oracle = np.maximum(oracle, floor)
        points.append(
            OracleFloorKLPoint(
                multiplier=multiplier,
                floor=float(floor),
                generalized_kl=generalized_kl_divergence(
                    estimated,
                    floored_oracle,
                    sample_weight=normalized_weight,
                ),
            )
        )
    return OracleFloorKLSensitivity(
        smallest_positive_oracle=epsilon,
        exact_generalized_kl=exact,
        points=tuple(points),
    )


def _estimate_at_resolution(
    *,
    requested_bins: int,
    basis: Array,
    current: Array,
    next_weight: Array,
    initial: Array,
    transition_is_a: Array,
    initial_is_a: Array,
    gamma: float,
    ridge_scale: float,
    near_constant: bool,
) -> CrossMomentScaleEstimate:
    upper_edges = _quantile_bin_upper_edges(
        basis,
        requested_bins=requested_bins,
        force_one_bin=near_constant,
    )
    effective_bins = int(upper_edges.size + 1)
    basis_label = np.searchsorted(upper_edges, basis, side="left")
    gram_diagonal = np.bincount(basis_label, minlength=effective_bins).astype(np.float64)
    gram_diagonal /= basis.size
    ridge = float(ridge_scale * np.sum(gram_diagonal) / effective_bins)

    current_label = np.searchsorted(upper_edges, current, side="left")
    next_label = np.searchsorted(upper_edges, next_weight, side="left")
    initial_label = np.searchsorted(upper_edges, initial, side="left")
    moment_a = _bellman_moment_vector(
        current=current,
        current_label=current_label,
        next_label=next_label,
        initial_label=initial_label,
        transition_mask=transition_is_a,
        initial_mask=initial_is_a,
        gamma=gamma,
        bins=effective_bins,
    )
    moment_b = _bellman_moment_vector(
        current=current,
        current_label=current_label,
        next_label=next_label,
        initial_label=initial_label,
        transition_mask=~transition_is_a,
        initial_mask=~initial_is_a,
        gamma=gamma,
        bins=effective_bins,
    )
    denominator = gram_diagonal + ridge
    if np.any(denominator <= 0.0):
        raise RuntimeError("basis Gram matrix has an unregularized empty bin")
    signed = float(np.sum(moment_a * moment_b / denominator))
    return CrossMomentScaleEstimate(
        requested_bins=requested_bins,
        effective_bins=effective_bins,
        bin_upper_edges=tuple(float(value) for value in upper_edges),
        gram_diagonal=tuple(float(value) for value in gram_diagonal),
        ridge=ridge,
        moment_a=tuple(float(value) for value in moment_a),
        moment_b=tuple(float(value) for value in moment_b),
        signed_squared_error=signed,
        positive_part_root_error=float(np.sqrt(max(signed, 0.0))),
    )


def _bellman_moment_vector(
    *,
    current: Array,
    current_label: Array,
    next_label: Array,
    initial_label: Array,
    transition_mask: Array,
    initial_mask: Array,
    gamma: float,
    bins: int,
) -> Array:
    transition_count = int(np.count_nonzero(transition_mask))
    initial_count = int(np.count_nonzero(initial_mask))
    if transition_count == 0 or initial_count == 0:
        raise RuntimeError("each audit half must contain transition and initial rows")
    initial_term = np.bincount(initial_label[initial_mask], minlength=bins).astype(np.float64)
    initial_term /= initial_count
    next_term = np.bincount(
        next_label[transition_mask],
        weights=current[transition_mask],
        minlength=bins,
    )
    next_term /= transition_count
    current_term = np.bincount(
        current_label[transition_mask],
        weights=current[transition_mask],
        minlength=bins,
    )
    current_term /= transition_count
    return (1.0 - gamma) * initial_term + gamma * next_term - current_term


def _quantile_bin_upper_edges(
    values: Array,
    *,
    requested_bins: int,
    force_one_bin: bool,
) -> Array:
    if requested_bins == 1 or force_one_bin:
        return np.empty(0, dtype=np.float64)
    unique_value, counts = np.unique(values, return_counts=True)
    if unique_value.size == 1:
        return np.empty(0, dtype=np.float64)
    cumulative = np.cumsum(counts)
    target_ranks = np.arange(1, requested_bins, dtype=np.float64) * values.size / requested_bins
    boundary_index = np.searchsorted(cumulative, target_ranks, side="left")
    boundary_index = np.unique(boundary_index)
    boundary_index = boundary_index[boundary_index < unique_value.size - 1]
    return np.asarray(unique_value[boundary_index], dtype=np.float64)


def _is_near_constant(values: Array, *, rtol: float, atol: float) -> bool:
    scale = max(1.0, abs(float(np.mean(values))))
    return bool(float(np.max(values) - np.min(values)) <= atol + rtol * scale)


def _deterministic_group_halves(
    transition_group_keys: set[bytes],
    initial_group_keys: set[bytes],
    *,
    seed: int,
) -> tuple[set[bytes], set[bytes]]:
    seed_bytes = str(seed).encode("ascii")

    def ordering(key: bytes) -> tuple[bytes, bytes]:
        digest = hashlib.blake2b(
            seed_bytes + b"\0" + key,
            digest_size=16,
            person=b"or-cal-audit",
        ).digest()
        return digest, key

    ordered = sorted(transition_group_keys | initial_group_keys, key=ordering)
    ordered_transition = [key for key in ordered if key in transition_group_keys]
    groups_a = {ordered_transition[0]}
    groups_b = {ordered_transition[1]}

    # Anchor each sample type in both halves before balancing the remaining
    # union groups.  This avoids empty empirical Bellman terms when the
    # transition and initial group sets are disjoint or only partly overlap.
    if not (initial_group_keys & groups_a):
        groups_a.add(next(key for key in ordered if key in initial_group_keys and key not in groups_b))
    if not (initial_group_keys & groups_b):
        groups_b.add(next(key for key in ordered if key in initial_group_keys and key not in groups_a))

    for key in ordered:
        if key in groups_a or key in groups_b:
            continue
        if len(groups_a) <= len(groups_b):
            groups_a.add(key)
        else:
            groups_b.add(key)
    return groups_a, groups_b


def _group_keys(values: Array, *, expected_size: int, name: str) -> list[bytes]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size != expected_size:
        raise ValueError(f"{name} must have length {expected_size}")
    return [_stable_group_key(value, name=name) for value in array]


def _stable_group_key(value: object, *, name: str) -> bytes:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return b"bool:" + (b"1" if value else b"0")
    if isinstance(value, int):
        return f"int:{value}".encode("ascii")
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{name} must contain finite identifiers")
        return f"float:{value.hex()}".encode("ascii")
    if isinstance(value, str):
        return b"str:" + value.encode("utf-8")
    if isinstance(value, bytes):
        return b"bytes:" + value
    raise ValueError(f"{name} identifiers must be bool, integer, finite float, string, or bytes")


def _ratio_inputs(
    estimated_ratio: Array,
    oracle_ratio: Array,
    *,
    sample_weight: Array | None,
) -> tuple[Array, Array, Array]:
    estimated = _as_nonnegative_finite_vector(estimated_ratio, name="estimated_ratio")
    oracle = _as_nonnegative_finite_vector(oracle_ratio, name="oracle_ratio")
    if estimated.size != oracle.size:
        raise ValueError("estimated_ratio and oracle_ratio must have equal length")
    normalized_weight = _normalized_sample_weight(sample_weight, expected_size=estimated.size)
    return estimated, oracle, normalized_weight


def _normalized_sample_weight(sample_weight: Array | None, *, expected_size: int) -> Array:
    if sample_weight is None:
        return np.full(expected_size, 1.0 / expected_size, dtype=np.float64)
    weight = _as_nonnegative_finite_vector(sample_weight, name="sample_weight")
    if weight.size != expected_size:
        raise ValueError(f"sample_weight must have length {expected_size}")
    total = float(np.sum(weight))
    if total <= 0.0:
        raise ValueError("sample_weight must have positive total mass")
    return weight / total


def _weighted_nonnegative_extended_mean(values: Array, weight: Array) -> float:
    if np.any(np.isposinf(values)):
        return float("inf")
    return float(np.dot(weight, values))


def _stable_relative_mse(estimated: Array, oracle: Array, weight: Array) -> float:
    absolute_error = np.abs(estimated - oracle)
    scale = max(float(np.max(absolute_error)), float(np.max(oracle)))
    if scale == 0.0:
        return 0.0
    scaled_numerator = float(np.dot(weight, np.square(absolute_error / scale)))
    scaled_denominator = float(np.dot(weight, np.square(oracle / scale)))
    if scaled_denominator == 0.0:
        return float("inf")
    return scaled_numerator / scaled_denominator


def _as_nonnegative_finite_vector(values: Array, *, name: str) -> Array:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _as_feature_matrix(values: Array, *, name: str) -> Array:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_state_action_rows(
    states: Array,
    actions: Array,
    *,
    state_name: str,
    action_name: str,
) -> None:
    if states.shape[0] != actions.shape[0]:
        raise ValueError(f"{state_name} and {action_name} must have equal row counts")


def _positive_finite_tuple(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty sequence")
    try:
        result = tuple(float(value) for value in values)
    except TypeError as error:
        raise ValueError(f"{name} must be a nonempty sequence") from error
    if not result:
        raise ValueError(f"{name} must be a nonempty sequence")
    if any(not np.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain only positive finite values")
    return result


def _bounded_rff_mean(
    inputs: Array,
    *,
    center: Array,
    scale: Array,
    frequency: Array,
    phase: Array,
    row_weight: Array | None,
    batch_size: int,
) -> Array:
    total = np.zeros(frequency.shape[0], dtype=np.float64)
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        standardized = (inputs[start:stop] - center) / scale
        with np.errstate(over="ignore", invalid="ignore"):
            projection = standardized @ frequency.T
        if not np.all(np.isfinite(projection)):
            raise ValueError("standardized RFF projections must be finite")
        reward = np.cos(projection + phase)
        if row_weight is None:
            total += np.sum(reward, axis=0)
        else:
            total += row_weight[start:stop] @ reward
    return total / inputs.shape[0]


def _rff_panel_digest(
    *,
    center: Array,
    scale: Array,
    frequency: Array,
    phase: Array,
    bandwidths: tuple[float, ...],
    seed: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"occupancy-functional-rff-v1\0")
    digest.update(str(seed).encode("ascii"))
    for array in (center, scale, frequency, phase, np.asarray(bandwidths)):
        contiguous = np.ascontiguousarray(array, dtype="<f8")
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _validate_gamma(gamma: float) -> float:
    value = float(gamma)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("gamma must be finite and in [0, 1)")
    return value


def _validate_nonnegative_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _validate_positive_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _validate_positive_integer(value: int, *, name: str) -> int:
    result = _validate_integer(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


__all__: Sequence[str] = (
    "BellmanCrossMomentMetrics",
    "ControlledRatioMetrics",
    "CrossMomentScaleEstimate",
    "MultiRewardOccupancyFunctionalMetrics",
    "OccupancyFunctionalPanelMetadata",
    "OracleFloorKLPoint",
    "OracleFloorKLSensitivity",
    "controlled_ratio_errors",
    "estimate_bellman_cross_moment_error",
    "estimate_multi_reward_occupancy_functional_error",
    "generalized_kl_divergence",
    "oracle_floor_kl_sensitivity",
)
