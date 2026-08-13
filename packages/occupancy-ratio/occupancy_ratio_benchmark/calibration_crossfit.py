"""Grouped cross-calibration for normalized discounted occupancy ratios.

The construction in this module has two distinct stages.  First, each base
learner scores only its held-out groups and the resulting current, successor,
and initial scores are pooled.  A *single* calibration map is fit to that
pooled OOF sample.  Second, at a new point, every fold learner is evaluated,
the common calibration map is applied to every fold score, and the calibrated
predictions are aggregated pointwise by their median.

No rewards, oracle ratios, or target-policy values enter this construction.
The only supported estimand is the standard normalized discounted occupancy
ratio; coverage stopping is deliberately absent from the API.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Callable, Literal, Protocol, Sequence

import numpy as np


Array = np.ndarray
Direction = Literal["increasing", "decreasing"]


class RatioScorePredictor(Protocol):
    """Protocol implemented by fitted occupancy-ratio base learners.

    Cross-calibration always requests ``clip=False``. The fold fitter must
    disable estimator-side upper caps and normalization so this is the one
    deployable nonnegative ratio estimate used by every candidate.
    """

    def predict_state_action_ratio(
        self,
        states: Array,
        actions: Array,
        *,
        clip: bool = True,
    ) -> Array:
        """Predict the state-action occupancy ratio or its unclipped score."""


class CalibrationMap(Protocol):
    """A scalar calibration map fit to the pooled OOF Bellman sample."""

    def predict(self, score: Array) -> Array:
        """Map one-dimensional base scores to calibrated ratio weights."""


class MaterialNegativePredictionError(ValueError):
    """Structured failure for a materially negative fold ratio prediction."""

    def __init__(
        self,
        *,
        fold_index: int,
        role: str,
        negative_count: int,
        minimum: float,
        tolerance: float,
    ) -> None:
        self.fold_index = int(fold_index)
        self.role = str(role)
        self.negative_count = int(negative_count)
        self.minimum = float(minimum)
        self.tolerance = float(tolerance)
        super().__init__(
            f"Fold {self.fold_index} {self.role} has {self.negative_count} "
            f"predictions below -{self.tolerance:g}; minimum={self.minimum:g}."
        )


@dataclass(frozen=True)
class CrossCalibrationConfig:
    """Configuration for grouped normalized-ratio cross-calibration.

    Parameters
    ----------
    num_folds:
        Number of grouped base-learner fits. At least two source groups must be
        available for every requested fold.
    seed:
        Seed for deterministic, size-balanced group assignment.
    scalar_mean_floor:
        Smallest admissible pooled OOF mean for scalar normalization.
    negative_tolerance:
        Absolute tolerance for projecting tiny numerical negatives to zero.
        Predictions below minus this threshold fail the fold.
    pava_num_iterations:
        Maximum number of normalized isotonic fixed-point iterations.
    pava_tolerance:
        Fixed-point stopping tolerance.
    pava_direction:
        Monotonicity direction of the learned scalar map.
    pava_fixed_point_damping:
        Damping used by the normalized isotonic fixed-point solver.
    pava_support_policy:
        Score-support policy, when supported by the installed solver. Constant
        endpoint extrapolation changes only the learned score map; it does not
        stop or truncate the occupancy recursion.
    pava_minimum_boundary_block_observations:
        Minimum number of pooled OOF behavior observations represented by
        each endpoint PAVA block. Interior blocks are unchanged.
    """

    num_folds: int = 10
    seed: int = 0
    scalar_mean_floor: float = 1e-12
    negative_tolerance: float = 1e-10
    pava_num_iterations: int = 3_000
    pava_tolerance: float = 1e-8
    pava_direction: Direction = "increasing"
    pava_fixed_point_damping: float = 1.0
    pava_support_policy: str = "constant_extrapolation"
    pava_minimum_boundary_block_observations: int = 1

    def __post_init__(self) -> None:
        if int(self.num_folds) < 2:
            raise ValueError("num_folds must be at least two.")
        if not np.isfinite(float(self.scalar_mean_floor)) or float(self.scalar_mean_floor) <= 0.0:
            raise ValueError("scalar_mean_floor must be positive and finite.")
        if not np.isfinite(float(self.negative_tolerance)) or float(self.negative_tolerance) < 0.0:
            raise ValueError("negative_tolerance must be nonnegative and finite.")
        if int(self.pava_num_iterations) <= 0:
            raise ValueError("pava_num_iterations must be positive.")
        if not np.isfinite(float(self.pava_tolerance)) or float(self.pava_tolerance) < 0.0:
            raise ValueError("pava_tolerance must be nonnegative and finite.")
        if self.pava_direction not in {"increasing", "decreasing"}:
            raise ValueError("pava_direction must be 'increasing' or 'decreasing'.")
        damping = float(self.pava_fixed_point_damping)
        if not np.isfinite(damping) or not (0.0 < damping <= 1.0):
            raise ValueError("pava_fixed_point_damping must be in (0, 1].")
        if str(self.pava_support_policy) not in {"error", "constant_extrapolation"}:
            raise ValueError("pava_support_policy must be 'error' or 'constant_extrapolation'.")
        if (
            isinstance(self.pava_minimum_boundary_block_observations, bool)
            or int(self.pava_minimum_boundary_block_observations)
            != self.pava_minimum_boundary_block_observations
            or int(self.pava_minimum_boundary_block_observations) <= 0
        ):
            raise ValueError(
                "pava_minimum_boundary_block_observations must be a positive integer."
            )


@dataclass(frozen=True)
class CrossCalibrationSample:
    """Observed Bellman rows and episode groups used for cross-calibration.

    Every source and initial group is assigned to one OOF fold. Matching source
    and initial group identifiers receive the same fold, preventing an episode
    used as a base-learner training source from also supplying its OOF initial
    score.
    """

    states: Array
    actions: Array
    next_states: Array
    next_target_actions: Array
    initial_states: Array
    initial_actions: Array
    source_groups: Array
    initial_groups: Array
    gamma: float
    source_weights: Array | None = None
    initial_weights: Array | None = None

    def __post_init__(self) -> None:
        n_source = _row_count(self.states, "states")
        for name in ("actions", "next_states", "next_target_actions", "source_groups"):
            _require_rows(getattr(self, name), name, n_source)
        n_initial = _row_count(self.initial_states, "initial_states")
        for name in ("initial_actions", "initial_groups"):
            _require_rows(getattr(self, name), name, n_initial)
        if n_source == 0:
            raise ValueError("At least one source row is required.")
        if n_initial == 0:
            raise ValueError("At least one initial row is required.")
        gamma = float(self.gamma)
        if not np.isfinite(gamma) or not (0.0 <= gamma < 1.0):
            raise ValueError("gamma must be finite and in [0, 1).")
        if self.source_weights is not None:
            _probability_weights(self.source_weights, "source_weights", n_source)
        if self.initial_weights is not None:
            _probability_weights(self.initial_weights, "initial_weights", n_initial)

    @property
    def n_source(self) -> int:
        """Number of source Bellman rows."""

        return int(np.asarray(self.states).shape[0])

    @property
    def n_initial(self) -> int:
        """Number of target-initial rows."""

        return int(np.asarray(self.initial_states).shape[0])


@dataclass(frozen=True)
class GroupedFoldAssignment:
    """Deterministic grouped fold identifiers for source and initial rows."""

    source_fold_ids: Array
    initial_fold_ids: Array
    num_folds: int
    seed: int


@dataclass(frozen=True)
class CrossFitFoldRequest:
    """Indices supplied to one base-learner fit callback."""

    fold_index: int
    fit_seed: int
    train_source_indices: Array
    oof_source_indices: Array
    train_initial_indices: Array
    oof_initial_indices: Array


@dataclass(frozen=True)
class CrossFitFoldArtifact:
    """Checkpointable output of one atomic grouped fold fit.

    The artifact contains the fitted predictor and every OOF score needed by
    the pooled calibration stage. It can be pickled when the underlying base
    predictor is pickle-compatible. Passing completed artifacts back to
    :func:`fit_cross_calibrated_ensemble` resumes without refitting those folds.
    """

    fold_index: int
    fit_seed: int
    train_source_indices: Array
    oof_source_indices: Array
    train_initial_indices: Array
    oof_initial_indices: Array
    source_q: Array
    next_q: Array
    initial_q: Array
    tiny_negative_count: int
    tiny_negative_mass: float
    predictor: RatioScorePredictor


@dataclass(frozen=True)
class PooledOOFPredictions:
    """OOF base predictions aligned to the original source and initial rows."""

    source_q: Array
    next_q: Array
    initial_q: Array
    source_fold_ids: Array
    initial_fold_ids: Array
    source_weights: Array
    initial_weights: Array
    scalar_scale: float
    scalar_source_weight: Array
    source_score: Array | None = None
    next_score: Array | None = None
    initial_score: Array | None = None
    scalar_log_shift: float | None = None
    score_space: str = "ratio"


@dataclass(frozen=True)
class PooledCalibrationInput:
    """The single pooled OOF sample passed to a calibration-map fitter."""

    source_score: Array
    next_score: Array
    initial_score: Array
    gamma: float
    source_weights: Array
    initial_weights: Array
    scalar_source_weight: Array


@dataclass(frozen=True)
class CrossCalibratedPredictions:
    """Raw, scalar-normalized, and PAVA predictions at query points."""

    raw: Array
    scalar: Array
    pava: Array
    raw_by_fold: Array
    scalar_by_fold: Array
    pava_by_fold: Array
    tiny_negative_count_by_fold: Array
    tiny_negative_mass_by_fold: Array


@dataclass(frozen=True)
class CrossCalibratedEnsemble:
    """Fitted fold ensemble and one common pooled-OOF calibration map."""

    assignment: GroupedFoldAssignment
    folds: tuple[CrossFitFoldArtifact, ...]
    pooled_oof: PooledOOFPredictions
    calibrator: CalibrationMap
    config: CrossCalibrationConfig
    diagnostics: dict[str, Any]

    def predict(self, states: Array, actions: Array) -> CrossCalibratedPredictions:
        """Predict all three cross-calibrated variants at new query points.

        The common scalar or isotonic transformation is applied separately to
        each fold learner. The reported prediction is then the pointwise median
        across folds; calibration is never refit at prediction time.
        """

        n = _row_count(states, "states")
        _require_rows(actions, "actions", n)
        raw_rows: list[Array] = []
        scalar_rows: list[Array] = []
        pava_rows: list[Array] = []
        tiny_negative_counts: list[int] = []
        tiny_negative_masses: list[float] = []
        for artifact in self.folds:
            q, negative_count, negative_mass = _predict_q(
                artifact.predictor,
                states,
                actions,
                fold_index=int(artifact.fold_index),
                role="query_q",
                negative_tolerance=float(self.config.negative_tolerance),
            )
            calibrated = _predict_calibration(self.calibrator, q, n)
            raw_rows.append(q)
            scalar_rows.append(float(self.pooled_oof.scalar_scale) * q)
            pava_rows.append(calibrated)
            tiny_negative_counts.append(negative_count)
            tiny_negative_masses.append(negative_mass)
        raw_by_fold = np.stack(raw_rows, axis=0)
        scalar_by_fold = np.stack(scalar_rows, axis=0)
        pava_by_fold = np.stack(pava_rows, axis=0)
        return CrossCalibratedPredictions(
            raw=np.median(raw_by_fold, axis=0),
            scalar=np.median(scalar_by_fold, axis=0),
            pava=np.median(pava_by_fold, axis=0),
            raw_by_fold=raw_by_fold,
            scalar_by_fold=scalar_by_fold,
            pava_by_fold=pava_by_fold,
            tiny_negative_count_by_fold=np.asarray(tiny_negative_counts, dtype=np.int64),
            tiny_negative_mass_by_fold=np.asarray(tiny_negative_masses, dtype=np.float64),
        )


@dataclass(frozen=True)
class CrossCalibratedMatrixResult:
    """Cross-calibration fit from checkpointed all-row fold predictions.

    ``source``, ``next``, and ``initial`` each contain pointwise fold medians
    and their underlying fold matrices. No fitted base model is retained or
    required after its three prediction arrays have been checkpointed.
    """

    assignment: GroupedFoldAssignment
    source: CrossCalibratedPredictions
    next: CrossCalibratedPredictions
    initial: CrossCalibratedPredictions
    pooled_oof: PooledOOFPredictions
    calibrator: CalibrationMap
    config: CrossCalibrationConfig
    diagnostics: dict[str, Any]


CalibrationFitter = Callable[[PooledCalibrationInput, CrossCalibrationConfig], CalibrationMap]
FoldFitter = Callable[[CrossFitFoldRequest], RatioScorePredictor]
FoldCompleteCallback = Callable[[CrossFitFoldArtifact], None]


def make_grouped_fold_assignment(
    source_groups: Array,
    initial_groups: Array,
    *,
    num_folds: int,
    seed: int,
) -> GroupedFoldAssignment:
    """Assign whole source and initial groups to deterministic balanced folds.

    Parameters
    ----------
    source_groups:
        One episode/group identifier per source row.
    initial_groups:
        One episode/group identifier per target-initial row. Identifiers also
        present in ``source_groups`` inherit the corresponding source fold.
    num_folds:
        Exact requested number of folds.
    seed:
        Random seed used only to break size ties deterministically.

    Returns
    -------
    GroupedFoldAssignment
        Row-aligned source and initial fold identifiers.
    """

    requested = int(num_folds)
    if requested < 2:
        raise ValueError("num_folds must be at least two.")
    source = _group_vector(source_groups, "source_groups")
    initial = _group_vector(initial_groups, "initial_groups")
    source_buckets = _group_buckets(source, "source_groups")
    if len(source_buckets) < requested:
        raise ValueError(
            f"num_folds={requested} requires at least {requested} source groups; received {len(source_buckets)}."
        )
    source_fold_ids, source_group_to_fold = _balanced_group_folds(
        source_buckets,
        requested,
        seed=int(seed),
    )

    initial_buckets = _group_buckets(initial, "initial_groups")
    initial_fold_ids = np.empty(initial.shape[0], dtype=np.int64)
    fold_counts = np.zeros(requested, dtype=np.int64)
    unmatched: dict[Any, list[int]] = {}
    for key, indices in initial_buckets.items():
        if key in source_group_to_fold:
            fold = source_group_to_fold[key]
            initial_fold_ids[np.asarray(indices, dtype=np.int64)] = fold
            fold_counts[fold] += len(indices)
        else:
            unmatched[key] = indices
    if unmatched:
        _, unmatched_group_to_fold = _balanced_group_folds(
            unmatched,
            requested,
            seed=int(seed) + 17_171,
            starting_counts=fold_counts,
        )
        for key, indices in unmatched.items():
            initial_fold_ids[np.asarray(indices, dtype=np.int64)] = unmatched_group_to_fold[key]

    for fold in range(requested):
        if not np.any(source_fold_ids == fold):
            raise RuntimeError(f"Grouped assignment produced empty source fold {fold}.")
        if not np.any(initial_fold_ids != fold):
            raise ValueError(f"Fold {fold} would have no initial rows available for base-learner training.")
    return GroupedFoldAssignment(
        source_fold_ids=source_fold_ids,
        initial_fold_ids=initial_fold_ids,
        num_folds=requested,
        seed=int(seed),
    )


def fit_cross_calibrated_ensemble(
    sample: CrossCalibrationSample,
    fit_fold: FoldFitter,
    *,
    config: CrossCalibrationConfig | None = None,
    calibrator_fitter: CalibrationFitter | None = None,
    completed_folds: Sequence[CrossFitFoldArtifact] = (),
    on_fold_complete: FoldCompleteCallback | None = None,
) -> CrossCalibratedEnsemble:
    """Fit grouped base learners and one pooled-OOF calibration map.

    Parameters
    ----------
    sample:
        Bellman rows, initial rows, weights, and episode groups.
    fit_fold:
        Callback that fits a base predictor using exactly the training indices
        in a :class:`CrossFitFoldRequest`. Estimator-side upper caps and
        normalization must be disabled. Its finite nonnegative ``clip=False``
        output is the shared ``q`` used by native, scalar, and PAVA candidates.
    config:
        Cross-calibration and normalized PAVA settings.
    calibrator_fitter:
        Optional pooled calibration callback. When omitted, the package's exact
        normalized isotonic FORE/PAVA solver is imported lazily.
    completed_folds:
        Previously checkpointed fold artifacts. Their indices and OOF payloads
        are validated before reuse.
    on_fold_complete:
        Callback invoked immediately after each newly fitted and validated
        atomic fold artifact, suitable for an atomic checkpoint writer.

    Returns
    -------
    CrossCalibratedEnsemble
        Raw cross-fit median, scalar-normalized median, and pooled-PAVA median
        predictor sharing the same fold learners.
    """

    cfg = CrossCalibrationConfig() if config is None else config
    assignment = make_grouped_fold_assignment(
        sample.source_groups,
        sample.initial_groups,
        num_folds=int(cfg.num_folds),
        seed=int(cfg.seed),
    )
    completed = _completed_fold_map(completed_folds, assignment, sample)
    artifacts: list[CrossFitFoldArtifact] = []
    for fold in range(assignment.num_folds):
        request = _fold_request(assignment, fold)
        is_new = fold not in completed
        if fold in completed:
            artifact = completed[fold]
        else:
            predictor = fit_fold(request)
            if not hasattr(predictor, "predict_state_action_ratio"):
                raise TypeError("fit_fold must return an object with predict_state_action_ratio(...).")
            artifact = _score_fold(sample, request, predictor, cfg)
        _validate_fold_artifact(artifact, request, sample)
        if is_new and on_fold_complete is not None:
            on_fold_complete(artifact)
        artifacts.append(artifact)

    pooled = _pool_oof(sample, assignment, artifacts, cfg)
    fitter = fit_normalized_pava_calibrator if calibrator_fitter is None else calibrator_fitter
    calibration_input = PooledCalibrationInput(
        source_score=pooled.source_q,
        next_score=pooled.next_q,
        initial_score=pooled.initial_q,
        gamma=float(sample.gamma),
        source_weights=pooled.source_weights,
        initial_weights=pooled.initial_weights,
        scalar_source_weight=pooled.scalar_source_weight,
    )
    calibrator = fitter(calibration_input, cfg)
    if not hasattr(calibrator, "predict"):
        raise TypeError("calibrator_fitter must return an object with predict(score).")
    pava_oof = _predict_calibration(calibrator, pooled.source_q, sample.n_source)
    pava_mean = float(np.dot(pooled.source_weights, pava_oof))
    diagnostics = {
        "algorithm": "grouped_cross_calibration",
        "occupancy_estimand": "normalized_discounted",
        "num_folds": int(assignment.num_folds),
        "pooled_calibrator_count": 1,
        "aggregation": "pointwise_median",
        "pooled_oof_raw_mean": float(np.dot(pooled.source_weights, pooled.source_q)),
        "pooled_oof_scalar_mean": float(np.dot(pooled.source_weights, pooled.scalar_source_weight)),
        "pooled_oof_pava_mean": pava_mean,
        "scalar_scale": float(pooled.scalar_scale),
        "uses_rewards": False,
        "uses_oracle_ratio": False,
        "uses_target_value": False,
        "uses_coverage_stopping": False,
        "tiny_negative_projection_count": int(sum(artifact.tiny_negative_count for artifact in artifacts)),
        "tiny_negative_projection_mass": float(sum(artifact.tiny_negative_mass for artifact in artifacts)),
        "material_negative_predictions": 0,
        "base_upper_cap_enabled": False,
        "base_normalization_enabled": False,
    }
    return CrossCalibratedEnsemble(
        assignment=assignment,
        folds=tuple(artifacts),
        pooled_oof=pooled,
        calibrator=calibrator,
        config=cfg,
        diagnostics=diagnostics,
    )


def fit_cross_calibrated_matrices(
    *,
    source_q_by_fold: Array,
    next_q_by_fold: Array,
    initial_q_by_fold: Array,
    source_log_score_by_fold: Array | None = None,
    next_log_score_by_fold: Array | None = None,
    initial_log_score_by_fold: Array | None = None,
    assignment: GroupedFoldAssignment,
    gamma: float,
    source_weights: Array | None = None,
    initial_weights: Array | None = None,
    config: CrossCalibrationConfig | None = None,
    calibrator_fitter: CalibrationFitter | None = None,
) -> CrossCalibratedMatrixResult:
    """Cross-calibrate checkpointed all-row fold-prediction matrices.

    Parameters
    ----------
    source_q_by_fold:
        Matrix with shape ``(K, n)``. Row ``k`` contains fold model ``k``'s
        shared nonnegative ratio prediction on every source row.
    next_q_by_fold:
        Matrix with shape ``(K, n)`` on target-successor rows.
    initial_q_by_fold:
        Matrix with shape ``(K, m)`` on target-initial rows.
    source_log_score_by_fold, next_log_score_by_fold, initial_log_score_by_fold:
        Optional finite KL log-score matrices. When supplied together, the one
        pooled PAVA map is fit and evaluated on these scores directly, while
        native ratios remain available for ratio-error diagnostics. Scalar
        normalization is computed by weighted log-mean-exp.
    assignment:
        Grouped OOF assignment used to train the fold models. Its diagonal
        entries select the held-out predictions used to fit the calibrator.
    gamma:
        Discount factor in ``[0, 1)``.
    source_weights:
        Optional source-distribution weights.
    initial_weights:
        Optional target-initial weights.
    config:
        Cross-calibration and normalized PAVA settings. ``num_folds`` must
        match the first matrix dimension.
    calibrator_fitter:
        Optional pooled calibration callback; the exact normalized PAVA solver
        is used by default.

    Returns
    -------
    CrossCalibratedMatrixResult
        One pooled calibrator and raw/scalar/PAVA pointwise medians for current,
        successor, and initial rows.

    Notes
    -----
    The calibration fit sees only diagonal grouped OOF entries. After that one
    fit, its common map is applied to every element of every fold matrix before
    taking pointwise medians. This is the model-free resume path for base
    learners whose fitted objects are not reliably pickleable.
    """

    cfg = CrossCalibrationConfig() if config is None else config
    gamma_f = float(gamma)
    if not np.isfinite(gamma_f) or not (0.0 <= gamma_f < 1.0):
        raise ValueError("gamma must be finite and in [0, 1).")
    num_folds = int(assignment.num_folds)
    if int(cfg.num_folds) != num_folds:
        raise ValueError("config.num_folds must match assignment.num_folds.")
    source_fold_ids = _validated_fold_ids(
        assignment.source_fold_ids,
        "assignment.source_fold_ids",
        num_folds,
    )
    if not np.array_equal(np.unique(source_fold_ids), np.arange(num_folds, dtype=np.int64)):
        raise ValueError("assignment.source_fold_ids must use every fold at least once.")
    initial_fold_ids = _validated_fold_ids(
        assignment.initial_fold_ids,
        "assignment.initial_fold_ids",
        num_folds,
    )
    n_source = int(source_fold_ids.shape[0])
    n_initial = int(initial_fold_ids.shape[0])
    source_matrix, source_negative_count, source_negative_mass = _q_matrix(
        source_q_by_fold,
        "source_q_by_fold",
        num_folds=num_folds,
        num_rows=n_source,
        negative_tolerance=float(cfg.negative_tolerance),
    )
    next_matrix, next_negative_count, next_negative_mass = _q_matrix(
        next_q_by_fold,
        "next_q_by_fold",
        num_folds=num_folds,
        num_rows=n_source,
        negative_tolerance=float(cfg.negative_tolerance),
    )
    initial_matrix, initial_negative_count, initial_negative_mass = _q_matrix(
        initial_q_by_fold,
        "initial_q_by_fold",
        num_folds=num_folds,
        num_rows=n_initial,
        negative_tolerance=float(cfg.negative_tolerance),
    )

    log_inputs = (
        source_log_score_by_fold,
        next_log_score_by_fold,
        initial_log_score_by_fold,
    )
    if any(value is not None for value in log_inputs) and not all(
        value is not None for value in log_inputs
    ):
        raise ValueError("all three log-score matrices must be supplied together")
    uses_log_scores = all(value is not None for value in log_inputs)
    if uses_log_scores:
        source_score_matrix = _finite_matrix(
            source_log_score_by_fold,
            "source_log_score_by_fold",
            num_folds=num_folds,
            num_rows=n_source,
        )
        next_score_matrix = _finite_matrix(
            next_log_score_by_fold,
            "next_log_score_by_fold",
            num_folds=num_folds,
            num_rows=n_source,
        )
        initial_score_matrix = _finite_matrix(
            initial_log_score_by_fold,
            "initial_log_score_by_fold",
            num_folds=num_folds,
            num_rows=n_initial,
        )
    else:
        source_score_matrix = source_matrix
        next_score_matrix = next_matrix
        initial_score_matrix = initial_matrix

    source_oof = source_matrix[source_fold_ids, np.arange(n_source)]
    next_oof = next_matrix[source_fold_ids, np.arange(n_source)]
    initial_oof = initial_matrix[initial_fold_ids, np.arange(n_initial)]
    source_score_oof = source_score_matrix[source_fold_ids, np.arange(n_source)]
    next_score_oof = next_score_matrix[source_fold_ids, np.arange(n_source)]
    initial_score_oof = initial_score_matrix[initial_fold_ids, np.arange(n_initial)]
    source_probability = _probability_weights(source_weights, "source_weights", n_source)
    initial_probability = _probability_weights(initial_weights, "initial_weights", n_initial)
    scalar_log_shift: float | None = None
    if uses_log_scores:
        log_raw_mean = _weighted_log_mean_exp(source_score_oof, source_probability)
        scalar_log_shift = -log_raw_mean
        raw_mean = _finite_exp(log_raw_mean)
        scalar_scale = _finite_exp(scalar_log_shift)
        scalar_source_weight = _finite_exp_array(
            source_score_oof + scalar_log_shift
        )
    else:
        raw_mean = float(np.dot(source_probability, source_oof))
        if not np.isfinite(raw_mean) or raw_mean <= float(cfg.scalar_mean_floor):
            raise ValueError(
                "The pooled OOF q predictions have nonpositive or numerically zero source-weighted mean."
            )
        scalar_scale = 1.0 / raw_mean
        scalar_source_weight = scalar_scale * source_oof
    pooled = PooledOOFPredictions(
        source_q=source_oof,
        next_q=next_oof,
        initial_q=initial_oof,
        source_fold_ids=source_fold_ids.copy(),
        initial_fold_ids=initial_fold_ids.copy(),
        source_weights=source_probability,
        initial_weights=initial_probability,
        scalar_scale=scalar_scale,
        scalar_source_weight=scalar_source_weight,
        source_score=source_score_oof,
        next_score=next_score_oof,
        initial_score=initial_score_oof,
        scalar_log_shift=scalar_log_shift,
        score_space="log_ratio" if uses_log_scores else "ratio",
    )
    calibration_input = PooledCalibrationInput(
        source_score=source_score_oof,
        next_score=next_score_oof,
        initial_score=initial_score_oof,
        gamma=gamma_f,
        source_weights=source_probability,
        initial_weights=initial_probability,
        scalar_source_weight=pooled.scalar_source_weight,
    )
    fitter = fit_normalized_pava_calibrator if calibrator_fitter is None else calibrator_fitter
    calibrator = fitter(calibration_input, cfg)
    if not hasattr(calibrator, "predict"):
        raise TypeError("calibrator_fitter must return an object with predict(score).")

    source = _matrix_predictions(
        source_matrix,
        score_by_fold=source_score_matrix,
        scalar_scale=scalar_scale,
        scalar_log_shift=scalar_log_shift,
        calibrator=calibrator,
        tiny_negative_count=source_negative_count,
        tiny_negative_mass=source_negative_mass,
    )
    next_result = _matrix_predictions(
        next_matrix,
        score_by_fold=next_score_matrix,
        scalar_scale=scalar_scale,
        scalar_log_shift=scalar_log_shift,
        calibrator=calibrator,
        tiny_negative_count=next_negative_count,
        tiny_negative_mass=next_negative_mass,
    )
    initial = _matrix_predictions(
        initial_matrix,
        score_by_fold=initial_score_matrix,
        scalar_scale=scalar_scale,
        scalar_log_shift=scalar_log_shift,
        calibrator=calibrator,
        tiny_negative_count=initial_negative_count,
        tiny_negative_mass=initial_negative_mass,
    )
    pava_oof = _predict_calibration(calibrator, source_score_oof, n_source)
    diagnostics = {
        "algorithm": "grouped_cross_calibration_matrix",
        "occupancy_estimand": "normalized_discounted",
        "num_folds": num_folds,
        "pooled_calibrator_count": 1,
        "aggregation": "pointwise_median",
        "checkpoint_payload": "prediction_matrices",
        "pooled_oof_raw_mean": raw_mean,
        "pooled_oof_scalar_mean": float(np.dot(source_probability, pooled.scalar_source_weight)),
        "pooled_oof_pava_mean": float(np.dot(source_probability, pava_oof)),
        "scalar_scale": scalar_scale,
        "scalar_log_shift": scalar_log_shift,
        "calibration_score_space": "log_ratio" if uses_log_scores else "ratio",
        "tiny_negative_projection_count": int(
            np.sum(source_negative_count) + np.sum(next_negative_count) + np.sum(initial_negative_count)
        ),
        "tiny_negative_projection_mass": float(
            np.sum(source_negative_mass) + np.sum(next_negative_mass) + np.sum(initial_negative_mass)
        ),
        "material_negative_predictions": 0,
        "uses_rewards": False,
        "uses_oracle_ratio": False,
        "uses_target_value": False,
        "uses_coverage_stopping": False,
    }
    return CrossCalibratedMatrixResult(
        assignment=assignment,
        source=source,
        next=next_result,
        initial=initial,
        pooled_oof=pooled,
        calibrator=calibrator,
        config=cfg,
        diagnostics=diagnostics,
    )


def fit_normalized_pava_calibrator(
    pooled: PooledCalibrationInput,
    config: CrossCalibrationConfig,
) -> CalibrationMap:
    """Fit the exact normalized isotonic FORE/PAVA map to pooled OOF scores.

    The import is lazy so benchmark-module import does not require a particular
    calibration backend. If the installed solver exposes an ``estimand``
    option, this adapter fixes it to ``"normalized_discounted"`` and never
    forwards coverage-retention inputs.
    """

    try:
        from occupancy_ratio.isotonic_calibration import (  # noqa: PLC0415
            IsotonicCalibrationConfig,
            fit_isotonic_fori_pava,
        )
    except ImportError as exc:
        raise ImportError("Normalized PAVA cross-calibration requires occupancy_ratio.isotonic_calibration.") from exc

    parameters = inspect.signature(IsotonicCalibrationConfig).parameters
    config_kwargs: dict[str, Any] = {
        "num_iterations": int(config.pava_num_iterations),
        "tolerance": float(config.pava_tolerance),
        "direction": str(config.pava_direction),
        "positivity_floor": 0.0,
        "normalize": True,
        "fixed_point_damping": float(config.pava_fixed_point_damping),
        "initialization": "unit",
    }
    if "estimand" in parameters:
        config_kwargs["estimand"] = "normalized_discounted"
    if "support_policy" in parameters:
        config_kwargs["support_policy"] = str(config.pava_support_policy)
    if "minimum_boundary_block_observations" in parameters:
        config_kwargs["minimum_boundary_block_observations"] = int(
            config.pava_minimum_boundary_block_observations
        )
    calibration_config = IsotonicCalibrationConfig(**config_kwargs)
    return fit_isotonic_fori_pava(
        source_score=pooled.source_score,
        next_score=pooled.next_score,
        initial_score=pooled.initial_score,
        gamma=float(pooled.gamma),
        source_weight=pooled.source_weights,
        initial_weight=pooled.initial_weights,
        initial_omega=pooled.scalar_source_weight,
        config=calibration_config,
    )


def apply_fitted_cross_calibration(
    result: CrossCalibratedMatrixResult,
    *,
    source_q_by_fold: Array,
    next_q_by_fold: Array,
    initial_q_by_fold: Array,
    source_log_score_by_fold: Array | None = None,
    next_log_score_by_fold: Array | None = None,
    initial_log_score_by_fold: Array | None = None,
) -> tuple[CrossCalibratedPredictions, CrossCalibratedPredictions, CrossCalibratedPredictions]:
    """Apply a fitted pooled map foldwise to an external query dataset."""

    num_folds = int(result.assignment.num_folds)
    matrices: list[Array] = []
    counts: list[Array] = []
    masses: list[Array] = []
    for value, name in (
        (source_q_by_fold, "source_q_by_fold"),
        (next_q_by_fold, "next_q_by_fold"),
        (initial_q_by_fold, "initial_q_by_fold"),
    ):
        raw = np.asarray(value)
        if raw.ndim != 2 or raw.shape[0] != num_folds or raw.shape[1] == 0:
            raise ValueError(f"{name} must have shape ({num_folds}, n) with n>0")
        matrix, count, mass = _q_matrix(
            raw,
            name,
            num_folds=num_folds,
            num_rows=int(raw.shape[1]),
            negative_tolerance=float(result.config.negative_tolerance),
        )
        matrices.append(matrix)
        counts.append(count)
        masses.append(mass)

    log_values = (
        source_log_score_by_fold,
        next_log_score_by_fold,
        initial_log_score_by_fold,
    )
    if any(value is not None for value in log_values) and not all(
        value is not None for value in log_values
    ):
        raise ValueError("all three external log-score matrices must be supplied together")
    if all(value is not None for value in log_values):
        scores = [
            _finite_matrix(
                value,
                name,
                num_folds=num_folds,
                num_rows=matrix.shape[1],
            )
            for value, name, matrix in zip(
                log_values,
                (
                    "source_log_score_by_fold",
                    "next_log_score_by_fold",
                    "initial_log_score_by_fold",
                ),
                matrices,
                strict=True,
            )
        ]
    else:
        scores = list(matrices)

    return tuple(
        _matrix_predictions(
            matrix,
            score_by_fold=score,
            scalar_scale=float(result.pooled_oof.scalar_scale),
            scalar_log_shift=result.pooled_oof.scalar_log_shift,
            calibrator=result.calibrator,
            tiny_negative_count=count,
            tiny_negative_mass=mass,
        )
        for matrix, score, count, mass in zip(
            matrices, scores, counts, masses, strict=True
        )
    )  # type: ignore[return-value]


def _score_fold(
    sample: CrossCalibrationSample,
    request: CrossFitFoldRequest,
    predictor: RatioScorePredictor,
    config: CrossCalibrationConfig,
) -> CrossFitFoldArtifact:
    source_indices = request.oof_source_indices
    initial_indices = request.oof_initial_indices
    source_q, source_negative_count, source_negative_mass = _predict_q(
        predictor,
        np.asarray(sample.states)[source_indices],
        np.asarray(sample.actions)[source_indices],
        fold_index=int(request.fold_index),
        role="source_q",
        negative_tolerance=float(config.negative_tolerance),
    )
    next_q, next_negative_count, next_negative_mass = _predict_q(
        predictor,
        np.asarray(sample.next_states)[source_indices],
        np.asarray(sample.next_target_actions)[source_indices],
        fold_index=int(request.fold_index),
        role="next_q",
        negative_tolerance=float(config.negative_tolerance),
    )
    if initial_indices.size:
        initial_q, initial_negative_count, initial_negative_mass = _predict_q(
            predictor,
            np.asarray(sample.initial_states)[initial_indices],
            np.asarray(sample.initial_actions)[initial_indices],
            fold_index=int(request.fold_index),
            role="initial_q",
            negative_tolerance=float(config.negative_tolerance),
        )
    else:
        initial_q = np.empty(0, dtype=np.float64)
        initial_negative_count = 0
        initial_negative_mass = 0.0
    return CrossFitFoldArtifact(
        fold_index=int(request.fold_index),
        fit_seed=int(request.fit_seed),
        train_source_indices=request.train_source_indices.copy(),
        oof_source_indices=source_indices.copy(),
        train_initial_indices=request.train_initial_indices.copy(),
        oof_initial_indices=initial_indices.copy(),
        source_q=source_q,
        next_q=next_q,
        initial_q=initial_q,
        tiny_negative_count=int(source_negative_count + next_negative_count + initial_negative_count),
        tiny_negative_mass=float(source_negative_mass + next_negative_mass + initial_negative_mass),
        predictor=predictor,
    )


def _pool_oof(
    sample: CrossCalibrationSample,
    assignment: GroupedFoldAssignment,
    artifacts: Sequence[CrossFitFoldArtifact],
    config: CrossCalibrationConfig,
) -> PooledOOFPredictions:
    source_q = np.empty(sample.n_source, dtype=np.float64)
    next_q = np.empty(sample.n_source, dtype=np.float64)
    initial_q = np.empty(sample.n_initial, dtype=np.float64)
    source_seen = np.zeros(sample.n_source, dtype=np.int64)
    initial_seen = np.zeros(sample.n_initial, dtype=np.int64)
    for artifact in artifacts:
        source_indices = np.asarray(artifact.oof_source_indices, dtype=np.int64)
        initial_indices = np.asarray(artifact.oof_initial_indices, dtype=np.int64)
        source_q[source_indices] = artifact.source_q
        next_q[source_indices] = artifact.next_q
        initial_q[initial_indices] = artifact.initial_q
        source_seen[source_indices] += 1
        initial_seen[initial_indices] += 1
    if not np.all(source_seen == 1):
        raise ValueError("Every source row must occur in exactly one OOF artifact.")
    if not np.all(initial_seen == 1):
        raise ValueError("Every initial row must occur in exactly one OOF artifact.")
    source_weights = _probability_weights(sample.source_weights, "source_weights", sample.n_source)
    initial_weights = _probability_weights(sample.initial_weights, "initial_weights", sample.n_initial)
    raw_mean = float(np.dot(source_weights, source_q))
    if not np.isfinite(raw_mean) or raw_mean <= float(config.scalar_mean_floor):
        raise ValueError("The pooled OOF raw weights have nonpositive or numerically zero source-weighted mean.")
    scalar_scale = 1.0 / raw_mean
    return PooledOOFPredictions(
        source_q=source_q,
        next_q=next_q,
        initial_q=initial_q,
        source_fold_ids=assignment.source_fold_ids.copy(),
        initial_fold_ids=assignment.initial_fold_ids.copy(),
        source_weights=source_weights,
        initial_weights=initial_weights,
        scalar_scale=scalar_scale,
        scalar_source_weight=scalar_scale * source_q,
    )


def _completed_fold_map(
    completed_folds: Sequence[CrossFitFoldArtifact],
    assignment: GroupedFoldAssignment,
    sample: CrossCalibrationSample,
) -> dict[int, CrossFitFoldArtifact]:
    completed: dict[int, CrossFitFoldArtifact] = {}
    for artifact in completed_folds:
        fold = int(artifact.fold_index)
        if fold in completed:
            raise ValueError(f"Duplicate completed artifact for fold {fold}.")
        if fold < 0 or fold >= int(assignment.num_folds):
            raise ValueError(f"Completed artifact has invalid fold_index={fold}.")
        _validate_fold_artifact(artifact, _fold_request(assignment, fold), sample)
        completed[fold] = artifact
    return completed


def _fold_request(assignment: GroupedFoldAssignment, fold: int) -> CrossFitFoldRequest:
    source_fold_ids = np.asarray(assignment.source_fold_ids, dtype=np.int64)
    initial_fold_ids = np.asarray(assignment.initial_fold_ids, dtype=np.int64)
    return CrossFitFoldRequest(
        fold_index=int(fold),
        fit_seed=int(assignment.seed) + int(fold) * 10_003,
        train_source_indices=np.flatnonzero(source_fold_ids != fold),
        oof_source_indices=np.flatnonzero(source_fold_ids == fold),
        train_initial_indices=np.flatnonzero(initial_fold_ids != fold),
        oof_initial_indices=np.flatnonzero(initial_fold_ids == fold),
    )


def _validate_fold_artifact(
    artifact: CrossFitFoldArtifact,
    request: CrossFitFoldRequest,
    sample: CrossCalibrationSample,
) -> None:
    if int(artifact.fold_index) != int(request.fold_index):
        raise ValueError("Fold artifact index does not match its deterministic assignment.")
    if int(artifact.fit_seed) != int(request.fit_seed):
        raise ValueError("Fold artifact seed does not match its deterministic assignment.")
    for name in (
        "train_source_indices",
        "oof_source_indices",
        "train_initial_indices",
        "oof_initial_indices",
    ):
        observed = np.asarray(getattr(artifact, name), dtype=np.int64).reshape(-1)
        expected = np.asarray(getattr(request, name), dtype=np.int64).reshape(-1)
        if not np.array_equal(observed, expected):
            raise ValueError(f"Fold artifact {name} does not match its deterministic assignment.")
    source_q = _finite_vector(artifact.source_q, "artifact.source_q", request.oof_source_indices.size)
    _require_nonnegative(source_q, "artifact.source_q")
    next_q = _finite_vector(artifact.next_q, "artifact.next_q", request.oof_source_indices.size)
    _require_nonnegative(next_q, "artifact.next_q")
    initial_q = _finite_vector(artifact.initial_q, "artifact.initial_q", request.oof_initial_indices.size)
    _require_nonnegative(initial_q, "artifact.initial_q")
    if int(artifact.tiny_negative_count) < 0:
        raise ValueError("artifact.tiny_negative_count must be nonnegative.")
    if not np.isfinite(float(artifact.tiny_negative_mass)) or float(artifact.tiny_negative_mass) < 0.0:
        raise ValueError("artifact.tiny_negative_mass must be nonnegative and finite.")
    if not hasattr(artifact.predictor, "predict_state_action_ratio"):
        raise TypeError("artifact.predictor must implement predict_state_action_ratio(...).")
    source_train_groups = {
        _group_key(value, "source_groups") for value in np.asarray(sample.source_groups)[request.train_source_indices]
    }
    source_oof_groups = {
        _group_key(value, "source_groups") for value in np.asarray(sample.source_groups)[request.oof_source_indices]
    }
    if source_train_groups.intersection(source_oof_groups):
        raise ValueError("A source group appears in both training and OOF rows.")
    initial_train_groups = {
        _group_key(value, "initial_groups")
        for value in np.asarray(sample.initial_groups)[request.train_initial_indices]
    }
    initial_oof_groups = {
        _group_key(value, "initial_groups") for value in np.asarray(sample.initial_groups)[request.oof_initial_indices]
    }
    if initial_train_groups.intersection(initial_oof_groups):
        raise ValueError("An initial group appears in both training and OOF rows.")
    if source_train_groups.intersection(initial_oof_groups):
        raise ValueError("An OOF initial group appears among source-training episodes.")
    if initial_train_groups.intersection(source_oof_groups):
        raise ValueError("An OOF source group appears among initial-training episodes.")


def _balanced_group_folds(
    buckets: dict[Any, list[int]],
    num_folds: int,
    *,
    seed: int,
    starting_counts: Array | None = None,
) -> tuple[Array, dict[Any, int]]:
    rng = np.random.default_rng(int(seed))
    items = list(buckets.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    counts = (
        np.zeros(int(num_folds), dtype=np.int64)
        if starting_counts is None
        else np.asarray(starting_counts, dtype=np.int64).copy()
    )
    row_count = sum(len(indices) for _, indices in items)
    fold_ids = np.empty(row_count, dtype=np.int64)
    group_to_fold: dict[Any, int] = {}
    row_order: list[int] = []
    row_folds: list[int] = []
    for key, indices in items:
        fold = int(np.argmin(counts))
        group_to_fold[key] = fold
        counts[fold] += len(indices)
        row_order.extend(int(index) for index in indices)
        row_folds.extend([fold] * len(indices))
    if row_order:
        max_row = max(row_order)
        if max_row >= row_count:
            fold_ids = np.empty(max_row + 1, dtype=np.int64)
        fold_ids[np.asarray(row_order, dtype=np.int64)] = np.asarray(row_folds, dtype=np.int64)
    return fold_ids, group_to_fold


def _group_buckets(groups: Array, name: str) -> dict[Any, list[int]]:
    buckets: dict[Any, list[int]] = {}
    for row, value in enumerate(np.asarray(groups, dtype=object).reshape(-1)):
        buckets.setdefault(_group_key(value, name), []).append(row)
    if not buckets:
        raise ValueError(f"{name} must not be empty.")
    return buckets


def _group_vector(groups: Array, name: str) -> Array:
    arr = np.asarray(groups, dtype=object)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def _group_key(value: Any, name: str) -> Any:
    key = value.item() if isinstance(value, np.generic) else value
    if isinstance(key, float) and not np.isfinite(key):
        raise ValueError(f"{name} contains a nonfinite group identifier.")
    try:
        hash(key)
    except TypeError as exc:
        raise ValueError(f"{name} identifiers must be hashable scalars.") from exc
    return key


def _predict_q(
    predictor: RatioScorePredictor,
    states: Array,
    actions: Array,
    *,
    fold_index: int,
    role: str,
    negative_tolerance: float,
) -> tuple[Array, int, float]:
    n = _row_count(states, "states")
    _require_rows(actions, "actions", n)
    values = _finite_vector(
        predictor.predict_state_action_ratio(states, actions, clip=False),
        role,
        n,
    )
    material = values < -float(negative_tolerance)
    if np.any(material):
        raise MaterialNegativePredictionError(
            fold_index=int(fold_index),
            role=role,
            negative_count=int(np.sum(material)),
            minimum=float(np.min(values)),
            tolerance=float(negative_tolerance),
        )
    tiny = (values < 0.0) & ~material
    count = int(np.sum(tiny))
    mass = float(-np.sum(values[tiny]))
    if count:
        values = values.copy()
        values[tiny] = 0.0
    return values, count, mass


def _predict_calibration(calibrator: CalibrationMap, score: Array, n: int) -> Array:
    values = _finite_vector(calibrator.predict(score), "calibrated prediction", n)
    _require_nonnegative(values, "calibrated prediction")
    return values


def _matrix_predictions(
    q_by_fold: Array,
    *,
    score_by_fold: Array | None = None,
    scalar_scale: float,
    scalar_log_shift: float | None = None,
    calibrator: CalibrationMap,
    tiny_negative_count: Array,
    tiny_negative_mass: Array,
) -> CrossCalibratedPredictions:
    q = np.asarray(q_by_fold, dtype=np.float64)
    score = q if score_by_fold is None else np.asarray(score_by_fold, dtype=np.float64)
    if score.shape != q.shape or not np.all(np.isfinite(score)):
        raise ValueError("score_by_fold must be finite and match q_by_fold")
    scalar = (
        float(scalar_scale) * q
        if scalar_log_shift is None
        else _finite_exp_array(score + float(scalar_log_shift))
    )
    pava = _predict_calibration(
        calibrator, score.reshape(-1), score.size
    ).reshape(score.shape)
    return CrossCalibratedPredictions(
        raw=np.median(q, axis=0),
        scalar=np.median(scalar, axis=0),
        pava=np.median(pava, axis=0),
        raw_by_fold=q,
        scalar_by_fold=scalar,
        pava_by_fold=pava,
        tiny_negative_count_by_fold=np.asarray(tiny_negative_count, dtype=np.int64),
        tiny_negative_mass_by_fold=np.asarray(tiny_negative_mass, dtype=np.float64),
    )


def _finite_matrix(
    value: Array | None,
    name: str,
    *,
    num_folds: int,
    num_rows: int,
) -> Array:
    matrix = np.asarray(value, dtype=np.float64)
    expected_shape = (int(num_folds), int(num_rows))
    if matrix.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}; received {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _weighted_log_mean_exp(log_value: Array, probability: Array) -> float:
    """Return log(sum_i probability_i exp(log_value_i)) stably."""

    score = np.asarray(log_value, dtype=np.float64).reshape(-1)
    weight = np.asarray(probability, dtype=np.float64).reshape(-1)
    if score.shape != weight.shape or score.size == 0:
        raise ValueError("log values and probabilities must be nonempty and aligned")
    positive = weight > 0.0
    if not np.any(positive):
        raise ValueError("probabilities must have positive mass")
    maximum = float(np.max(score[positive]))
    scaled = float(np.dot(weight[positive], np.exp(score[positive] - maximum)))
    if not np.isfinite(scaled) or scaled <= 0.0:
        raise ValueError("weighted log-mean-exp normalization failed")
    return maximum + float(np.log(scaled))


def _finite_exp(value: float) -> float:
    return float(_finite_exp_array(np.asarray([value], dtype=np.float64))[0])


def _finite_exp_array(value: Array) -> Array:
    score = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(score)):
        raise ValueError("log-domain values must be finite")
    lower = float(np.log(np.nextafter(0.0, 1.0)))
    upper = float(np.nextafter(np.log(np.finfo(np.float64).max), -np.inf))
    result = np.exp(np.clip(score, lower, upper))
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise AssertionError("finite clipped log-domain values must exponentiate safely")
    return result


def _q_matrix(
    value: Array,
    name: str,
    *,
    num_folds: int,
    num_rows: int,
    negative_tolerance: float,
) -> tuple[Array, Array, Array]:
    matrix = np.asarray(value, dtype=np.float64)
    expected_shape = (int(num_folds), int(num_rows))
    if matrix.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}; received {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    projected = matrix.copy()
    counts = np.zeros(int(num_folds), dtype=np.int64)
    masses = np.zeros(int(num_folds), dtype=np.float64)
    for fold in range(int(num_folds)):
        material = projected[fold] < -float(negative_tolerance)
        if np.any(material):
            raise MaterialNegativePredictionError(
                fold_index=fold,
                role=name,
                negative_count=int(np.sum(material)),
                minimum=float(np.min(projected[fold])),
                tolerance=float(negative_tolerance),
            )
        tiny = (projected[fold] < 0.0) & ~material
        counts[fold] = int(np.sum(tiny))
        masses[fold] = float(-np.sum(projected[fold, tiny]))
        projected[fold, tiny] = 0.0
    return projected, counts, masses


def _validated_fold_ids(value: Array, name: str, num_folds: int) -> Array:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    try:
        fold_ids = np.asarray(raw, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer fold identifiers.") from exc
    if not np.all(raw == fold_ids):
        raise ValueError(f"{name} must contain integer fold identifiers.")
    if np.any(fold_ids < 0) or np.any(fold_ids >= int(num_folds)):
        raise ValueError(f"{name} contains a fold outside [0, {int(num_folds)}).")
    return fold_ids


def _finite_vector(value: Array, name: str, n: int) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.shape[0] != int(n):
        raise ValueError(f"{name} must have {n} rows; received {arr.shape[0]}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _require_nonnegative(value: Array, name: str) -> None:
    if np.any(np.asarray(value, dtype=np.float64) < 0.0):
        raise ValueError(f"{name} must be nonnegative.")


def _probability_weights(value: Array | None, name: str, n: int) -> Array:
    if value is None:
        return np.full(int(n), 1.0 / int(n), dtype=np.float64)
    weights = _finite_vector(value, name, n)
    _require_nonnegative(weights, name)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass.")
    return weights / total


def _row_count(value: Array, name: str) -> int:
    arr = np.asarray(value)
    if arr.ndim == 0:
        raise ValueError(f"{name} must have a row dimension.")
    return int(arr.shape[0])


def _require_rows(value: Array, name: str, n: int) -> None:
    rows = _row_count(value, name)
    if rows != int(n):
        raise ValueError(f"{name} must have {n} rows; received {rows}.")
