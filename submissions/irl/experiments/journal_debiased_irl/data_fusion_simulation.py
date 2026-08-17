"""Abundant-outcome data fusion in the stationary JASA MDP."""

from __future__ import annotations

import concurrent.futures
import json
import math
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

try:
    from .fore_ratio import (
        FOREFitOptions,
        fit_selected_fore_ratio,
        paper_early_stopping_candidates,
        ratio_diagnostics,
    )
    from .jrssb_simulation import (
        EPS,
        JRSSBConfig,
        JRSSBOracle,
        GridProbabilityPolicy,
        ProbabilityPolicyAdapter,
        combine_repeated_split_se,
        crossfit_critical_value,
        crossfit_se_diagnostics,
        fold_splits,
        normal_cdf,
        selected_crossfit_se,
    )
except ImportError:  # pragma: no cover - direct script execution
    from fore_ratio import (
        FOREFitOptions,
        fit_selected_fore_ratio,
        paper_early_stopping_candidates,
        ratio_diagnostics,
    )
    from jrssb_simulation import (
        EPS,
        JRSSBConfig,
        JRSSBOracle,
        GridProbabilityPolicy,
        ProbabilityPolicyAdapter,
        combine_repeated_split_se,
        crossfit_critical_value,
        crossfit_se_diagnostics,
        fold_splits,
        normal_cdf,
        selected_crossfit_se,
    )

from data_fusion import OutcomeSource, StateRewardRegressor
Array = np.ndarray


@dataclass(frozen=True)
class DataFusionTruth:
    """Fine-grid truth used only for labeled simulation evaluation."""

    g_grid: Array
    reward_grid: Array
    q_grid: Array
    v_grid: Array
    psi: float


@dataclass(frozen=True)
class FrozenOutcomeRegression:
    """Near-population outcome regression shared across behavior repetitions."""

    model: StateRewardRegressor
    sample_size: int
    seed: int
    outcome_noise_sd: float
    g_rmse: float
    estimand_shift: float
    selected_outcome_config: dict[str, float | int] = field(default_factory=dict)
    outcome_validation_mse: float = float("nan")
    outcome_candidate_scores: tuple[dict[str, float | int], ...] = ()
    normalization_policy_mode: str = "known-uniform"
    source_state_mode: str = "uniform-domain"
    outcome_model_family: str = "outcome-summary-lbfgs-tanh-ensemble-v4"
    target_gamma: float = 0.80

    def predict(self, states: Array) -> Array:
        """Predict the statewise normalization function."""
        return self.model.predict(states)

    def save(self, path: Path) -> None:
        """Serialize the fixed auxiliary regression."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "FrozenOutcomeRegression":
        """Load a fixed auxiliary regression."""
        with path.open("rb") as handle:
            result = pickle.load(handle)
        if not isinstance(result, cls):
            raise TypeError(f"Expected FrozenOutcomeRegression, got {type(result).__name__}.")
        # Keep caches created before source-only outcome model selection readable.
        for name, value in (
            ("selected_outcome_config", {}),
            ("outcome_validation_mse", float("nan")),
            ("outcome_candidate_scores", ()),
            ("normalization_policy_mode", "behavior"),
            ("source_state_mode", "behavior-stationary"),
            ("outcome_model_family", "lightgbm-v0"),
            ("target_gamma", float("nan")),
        ):
            if not hasattr(result, name):
                object.__setattr__(result, name, value)
        return result


@dataclass(frozen=True)
class CalibratedPredictionEnsemble:
    """Average smooth regressors and apply a source-only intercept calibration."""

    models: tuple[Any, ...]
    intercept_offset: float

    def predict(self, states: Array) -> Array:
        """Return the calibrated ensemble mean prediction."""
        predictions = np.column_stack(
            [np.asarray(model.predict(states), dtype=float) for model in self.models]
        )
        return np.mean(predictions, axis=1) + self.intercept_offset


@dataclass(frozen=True)
class SieveLogitPolicy:
    """Smooth source-only multinomial policy estimate."""

    pipeline: Any
    n_actions: int
    probability_floor: float = 1e-3

    def predict_proba(self, states: Array) -> Array:
        """Predict clipped probabilities in the canonical action order."""
        states = np.asarray(states, dtype=float)
        raw = np.asarray(self.pipeline.predict_proba(states), dtype=float)
        probabilities = np.zeros((states.shape[0], self.n_actions), dtype=float)
        classes = np.asarray(self.pipeline.named_steps["logit"].classes_, dtype=int)
        probabilities[:, classes] = raw
        probabilities = np.clip(probabilities, self.probability_floor, None)
        return probabilities / np.sum(probabilities, axis=1, keepdims=True)

    def clipping_fraction(self, states: Array) -> float:
        """Return the fraction of raw action probabilities below the floor."""
        states = np.asarray(states, dtype=float)
        raw = np.asarray(self.pipeline.predict_proba(states), dtype=float)
        return float(np.mean(raw < self.probability_floor))


@dataclass(frozen=True)
class SieveLogitSelection:
    """Truth-blind held-out selection record for the behavior policy."""

    policy: SieveLogitPolicy
    selected_degree: int
    selected_c: float
    validation_nll: float
    candidate_scores: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class GaussianTransitionSieve:
    """Action-specific smooth Gaussian transition model fit from observed rows."""

    models: tuple[Any, ...]
    residual_sds: Array
    n_actions: int

    def predict_mean(self, states: Array, actions: Array) -> Array:
        """Predict the two-dimensional conditional transition mean."""
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=int).reshape(-1)
        if states.ndim != 2 or states.shape[0] != actions.size:
            raise ValueError("states and actions must have compatible row counts.")
        prediction = np.empty_like(states, dtype=float)
        for action, model in enumerate(self.models):
            mask = actions == action
            if np.any(mask):
                prediction[mask] = np.asarray(model.predict(states[mask]), dtype=float)
        if not np.all(np.isfinite(prediction)):
            raise FloatingPointError("Transition sieve produced nonfinite means.")
        return prediction


@dataclass(frozen=True)
class GaussianTransitionSelection:
    """Held-out source-only selection record for the transition model."""

    model: GaussianTransitionSieve
    selected_alpha: float
    validation_mse: float
    candidate_scores: tuple[dict[str, float], ...]


@dataclass
class DataFusionRunResult:
    """One cross-fitted data-fusion simulation result."""

    n: int
    seed: int
    outcome_sample_size: int
    truth: float
    plugin_estimate: float
    if_estimate: float
    plugin_error: float
    if_error: float
    estimated_se: float
    ci_lower: float
    ci_upper: float
    covered: float
    ci_length: float
    reward_rmse: float
    g_rmse: float
    g_estimand_shift: float
    ratio_q99: float
    ratio_max: float
    ratio_ess: float
    fore_selected_iterations: float
    fore_apbv_score: float
    fore_fit_seconds: float
    fore_normalized_mass: float
    fore_logit_cap_fraction: float
    behavior_policy_selected_c: float
    behavior_policy_validation_nll: float
    transition_selected_alpha: float
    transition_validation_mse: float
    transition_residual_sd_x: float
    transition_residual_sd_z: float
    ratio_failure: float
    fore_selected_iterations_by_fold: str
    fore_apbv_scores_by_fold: str
    failure_message: str = ""
    iid_estimated_se: float = float("nan")
    fold_cluster_estimated_se: float = float("nan")
    crossfit_fold_count: float = float("nan")
    ci_critical_value: float = float("nan")
    behavior_policy_selected_degree: float = float("nan")
    data_fusion_policy_mode: str = "known-logging"
    data_fusion_transition_mode: str = "sieve"
    data_fusion_g_mode: str = "frozen"
    data_fusion_ratio_mode: str = "neural-fore"
    normalization_policy_mode: str = "known-uniform"
    data_fusion_target_gamma: float = 0.80
    data_fusion_repeated_splits: float = 1.0
    split_estimate_sd: float = 0.0
    behavior_probability_floor: float = 0.02
    behavior_probability_clipping_fraction: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable row."""
        return asdict(self)


def data_fusion_normalization_policy(
    oracle: JRSSBOracle,
    states: Array,
) -> Array:
    """Return the known randomized policy used by the outcome-only source."""
    states = np.asarray(states, dtype=float)
    return np.full(
        (states.shape[0], oracle.config.n_actions),
        1.0 / oracle.config.n_actions,
        dtype=float,
    )


def data_fusion_true_g(oracle: JRSSBOracle, states: Array) -> Array:
    """Return E[Y|S] under the known randomized normalization policy."""
    normalization = data_fusion_normalization_policy(oracle, states)
    reward = oracle.action_values(states, oracle.reward_dagger)
    return np.sum(normalization * reward, axis=1)


def build_data_fusion_truth(
    oracle: JRSSBOracle,
    g_grid: Optional[Array] = None,
) -> DataFusionTruth:
    """Construct the reward identified by a fixed randomized policy and ``g``."""
    if g_grid is None:
        g_grid = data_fusion_true_g(oracle, oracle.main_grid.states)
    g_grid = np.asarray(g_grid, dtype=float).reshape(-1)
    if g_grid.shape[0] != oracle.main_grid.n_states or not np.all(np.isfinite(g_grid)):
        raise ValueError("g_grid must be finite with one value per main-grid state.")
    normalization_grid = data_fusion_normalization_policy(
        oracle, oracle.main_grid.states
    )
    q_mu, v_mu = oracle.evaluate_policy(
        oracle.config.tau_behavior * oracle.r0 - g_grid[:, None],
        normalization_grid,
        oracle.config.gamma_behavior,
    )
    reward_grid = q_mu - v_mu[:, None] + g_grid[:, None]
    q_grid, v_grid = oracle.evaluate_policy(
        reward_grid,
        oracle.pi_fix,
        oracle.config.data_fusion_target_gamma,
    )
    psi = float(np.dot(oracle.stationary_behavior, v_grid))
    return DataFusionTruth(
        g_grid=g_grid,
        reward_grid=reward_grid,
        q_grid=q_grid,
        v_grid=v_grid,
        psi=psi,
    )


def generate_outcome_source(
    oracle: JRSSBOracle,
    *,
    sample_size: int,
    seed: int,
    outcome_noise_sd: float = 0.5,
) -> OutcomeSource:
    """Generate independent ``(S,Y)`` data and deliberately discard actions."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if outcome_noise_sd < 0.0:
        raise ValueError("outcome_noise_sd must be nonnegative.")
    rng = np.random.default_rng(seed)
    # The auxiliary study has broad covariate coverage.  Its state marginal
    # need not equal the behavior-transition marginal because the shared
    # object is the conditional outcome regression g(s).
    states = rng.uniform(
        low=oracle.config.state_low,
        high=oracle.config.state_high,
        size=(sample_size, 2),
    )
    # The auxiliary source reports a noisy state-level outcome summary.  Its
    # conditional mean is the reward averaged under the known randomized
    # normalization policy; individual treatment assignments are unavailable.
    outcomes = data_fusion_true_g(oracle, states)
    if outcome_noise_sd > 0.0:
        outcomes = outcomes + rng.normal(scale=outcome_noise_sd, size=sample_size)
    # Each auxiliary draw is independent, so each row is its own complete group.
    return OutcomeSource(
        states=states,
        outcomes=outcomes,
        trajectory_ids=np.arange(sample_size, dtype=int),
        n_trajectories=sample_size,
    )


def fit_frozen_outcome_regression(
    oracle: JRSSBOracle,
    *,
    sample_size: int = 1_000_000,
    seed: int = 91_001,
    outcome_noise_sd: float = 0.5,
) -> FrozenOutcomeRegression:
    """Fit the fixed near-population ``g`` regression without truth-based tuning."""
    source = generate_outcome_source(
        oracle,
        sample_size=sample_size,
        seed=seed,
        outcome_noise_sd=outcome_noise_sd,
    )
    # Capacity is selected solely by held-out outcome prediction error.  A
    # tanh network on quadratic features is a deliberately smooth low-
    # dimensional model; unlike boosted trees, it does not create unstable
    # tail steps that the long-horizon value map can amplify.
    candidate_library: tuple[dict[str, float | int], ...] = tuple(
        {"hidden_units": hidden_units, "alpha": alpha}
        for hidden_units in (4, 8)
        for alpha in (1e-2, 1e-3)
    )
    rng = np.random.default_rng(seed + 17)
    validation_size = max(1, int(round(0.20 * sample_size)))
    validation_idx = np.sort(rng.choice(sample_size, size=validation_size, replace=False))
    training_mask = np.ones(sample_size, dtype=bool)
    training_mask[validation_idx] = False
    training_idx = np.flatnonzero(training_mask)
    training_source = OutcomeSource(
        states=source.states[training_idx],
        outcomes=source.outcomes[training_idx],
        trajectory_ids=np.arange(training_idx.size, dtype=int),
        n_trajectories=training_idx.size,
    )
    scored_candidates: list[dict[str, float | int]] = []
    validation_losses: list[Array] = []

    def fit_candidate(
        candidate: dict[str, float | int],
        *,
        fit_states: Array,
        fit_outcomes: Array,
        candidate_seed: int,
    ) -> StateRewardRegressor:
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        pipeline = Pipeline(
            steps=[
                ("quadratic", PolynomialFeatures(degree=2, include_bias=False)),
                ("scale", StandardScaler()),
                (
                    "network",
                    MLPRegressor(
                        hidden_layer_sizes=(int(candidate["hidden_units"]),),
                        activation="tanh",
                        solver="lbfgs",
                        alpha=float(candidate["alpha"]),
                        max_iter=500,
                        tol=1e-9,
                        random_state=candidate_seed,
                    ),
                ),
            ]
        )
        pipeline.fit(fit_states, fit_outcomes)
        return StateRewardRegressor(model=pipeline)

    for candidate_index, candidate in enumerate(candidate_library):
        candidate_model = fit_candidate(
            candidate,
            fit_states=training_source.states,
            fit_outcomes=training_source.outcomes,
            candidate_seed=seed + 101 + candidate_index,
        )
        validation_prediction = candidate_model.predict(source.states[validation_idx])
        losses = np.square(validation_prediction - source.outcomes[validation_idx])
        validation_mse = float(np.mean(losses))
        validation_losses.append(losses)
        scored_candidates.append({**candidate, "validation_mse": validation_mse})
    best_index = min(
        range(len(scored_candidates)),
        key=lambda index: (
            float(scored_candidates[index]["validation_mse"]),
            int(scored_candidates[index]["hidden_units"]),
            -float(scored_candidates[index]["alpha"]),
        ),
    )
    best_standard_error = float(
        np.std(validation_losses[best_index], ddof=1)
        / math.sqrt(validation_losses[best_index].size)
    )
    one_se_threshold = float(
        scored_candidates[best_index]["validation_mse"] + best_standard_error
    )
    eligible = [
        index
        for index, row in enumerate(scored_candidates)
        if float(row["validation_mse"]) <= one_se_threshold + 1e-12
    ]
    selected_index = min(
        eligible,
        key=lambda index: (
            int(scored_candidates[index]["hidden_units"]),
            -float(scored_candidates[index]["alpha"]),
            float(scored_candidates[index]["validation_mse"]),
        ),
    )
    for index, row in enumerate(scored_candidates):
        row["one_se_eligible"] = int(index in eligible)
        row["one_se_threshold"] = one_se_threshold
    selected = scored_candidates[selected_index]
    selected_config = {
        key: selected[key] for key in ("hidden_units", "alpha")
    }
    ensemble_size = 3
    ensemble_models: list[Any] = []
    ensemble_prediction = np.zeros(sample_size, dtype=float)
    for ensemble_index in range(ensemble_size):
        member = fit_candidate(
            selected_config,
            fit_states=source.states,
            fit_outcomes=source.outcomes,
            candidate_seed=seed + 10_001 + ensemble_index,
        )
        ensemble_models.append(member.model)
        ensemble_prediction += member.predict(source.states) / ensemble_size
    intercept_offset = float(np.mean(source.outcomes - ensemble_prediction))
    model = StateRewardRegressor(
        model=CalibratedPredictionEnsemble(
            models=tuple(ensemble_models),
            intercept_offset=intercept_offset,
        )
    )
    selected_config = {
        **selected_config,
        "ensemble_size": ensemble_size,
        "intercept_offset": intercept_offset,
    }
    grid_states = oracle.main_grid.states
    g_true = data_fusion_true_g(oracle, grid_states)
    g_hat = model.predict(grid_states)
    if not np.all(np.isfinite(g_hat)):
        raise FloatingPointError("The fixed outcome regression produced nonfinite predictions.")
    truth = build_data_fusion_truth(oracle, g_true)
    fitted_target = build_data_fusion_truth(oracle, g_hat)
    return FrozenOutcomeRegression(
        model=model,
        sample_size=sample_size,
        seed=seed,
        outcome_noise_sd=outcome_noise_sd,
        g_rmse=float(np.sqrt(np.mean(np.square(g_hat - g_true)))),
        estimand_shift=float(fitted_target.psi - truth.psi),
        selected_outcome_config=selected_config,
        outcome_validation_mse=float(selected["validation_mse"]),
        outcome_candidate_scores=tuple(scored_candidates),
        normalization_policy_mode="known-uniform",
        source_state_mode="uniform-domain",
        outcome_model_family="outcome-summary-lbfgs-tanh-ensemble-v4",
        target_gamma=oracle.config.data_fusion_target_gamma,
    )


def fit_sieve_logit_behavior_policy(
    *,
    states: Array,
    actions: Array,
    n_actions: int,
    seed: int,
    degree: int | None = None,
    candidate_degrees: Sequence[int] = (2, 3, 4),
    candidate_c: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    validation_fraction: float = 0.20,
    probability_floor: float = 1e-3,
) -> SieveLogitSelection:
    """Fit a polynomial-logit policy with truth-blind one-SE selection.

    The one-standard-error rule protects against choosing a visibly more
    complex sieve for immaterial held-out log-loss gains.  Ties favor lower
    polynomial degree and stronger regularization (smaller ``C``).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    if states.ndim != 2 or states.shape[0] != actions.size:
        raise ValueError("states and actions must have compatible row counts.")
    if states.shape[0] < 10 or np.unique(actions).size != n_actions:
        raise ValueError("Sieve logit requires at least ten rows and every action class.")
    degree_values = (
        (int(degree),)
        if degree is not None
        else tuple(int(value) for value in candidate_degrees)
    )
    if (
        not degree_values
        or any(value <= 0 for value in degree_values)
        or not 0.0 < validation_fraction < 1.0
    ):
        raise ValueError(
            "candidate degrees and validation_fraction must be positive and valid."
        )
    candidate_values = tuple(float(value) for value in candidate_c)
    if not candidate_values or any(value <= 0.0 for value in candidate_values):
        raise ValueError("candidate_c must contain positive values.")
    fit_idx, validation_idx = train_test_split(
        np.arange(actions.size),
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=actions,
    )

    def build_pipeline(degree_value: int, c_value: float) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "polynomial",
                    PolynomialFeatures(degree=degree_value, include_bias=False),
                ),
                ("scale", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        C=c_value,
                        solver="lbfgs",
                        max_iter=2_000,
                        random_state=seed,
                    ),
                ),
            ]
        )

    score_rows: list[dict[str, float]] = []
    validation_losses: list[Array] = []
    for degree_value in degree_values:
        for c_value in candidate_values:
            pipeline = build_pipeline(degree_value, c_value)
            pipeline.fit(states[fit_idx], actions[fit_idx])
            validation_probabilities = np.asarray(
                pipeline.predict_proba(states[validation_idx]), dtype=float
            )
            class_order = np.asarray(
                pipeline.named_steps["logit"].classes_, dtype=int
            )
            class_columns = {
                int(action): column for column, action in enumerate(class_order)
            }
            observed_columns = np.asarray(
                [class_columns[int(action)] for action in actions[validation_idx]],
                dtype=int,
            )
            losses = -np.log(
                np.clip(
                    validation_probabilities[
                        np.arange(validation_idx.size), observed_columns
                    ],
                    probability_floor,
                    None,
                )
            )
            validation_losses.append(losses)
            score_rows.append(
                {
                    "degree": float(degree_value),
                    "c": c_value,
                    "validation_nll": float(np.mean(losses)),
                }
            )
    best_index = min(
        range(len(score_rows)),
        key=lambda index: (
            score_rows[index]["validation_nll"],
            score_rows[index]["degree"],
            score_rows[index]["c"],
        ),
    )
    # Candidate losses are paired on the same validation observations.  The
    # usual unpaired shortcut, ``best mean + SE(best losses)``, is far too
    # permissive here because most of that variation is shared across models;
    # in the pilot it marked the entire library equivalent.  Apply the one-SE
    # rule to the paired excess loss instead.
    best_losses = validation_losses[best_index]
    paired_excess_means: list[float] = []
    paired_excess_ses: list[float] = []
    eligible: list[int] = []
    for index, losses in enumerate(validation_losses):
        excess = np.asarray(losses, dtype=float) - best_losses
        excess_mean = float(np.mean(excess))
        excess_se = float(np.std(excess, ddof=1) / math.sqrt(excess.size))
        paired_excess_means.append(excess_mean)
        paired_excess_ses.append(excess_se)
        if excess_mean <= excess_se + 1e-12:
            eligible.append(index)
    selected_index = min(
        eligible,
        key=lambda index: (
            score_rows[index]["degree"],
            score_rows[index]["c"],
            score_rows[index]["validation_nll"],
        ),
    )
    for index, row in enumerate(score_rows):
        row["one_se_eligible"] = float(index in eligible)
        row["paired_excess_nll"] = paired_excess_means[index]
        row["paired_excess_se"] = paired_excess_ses[index]
    selected = score_rows[selected_index]
    selected_degree = int(selected["degree"])
    refit = build_pipeline(selected_degree, float(selected["c"]))
    refit.fit(states, actions)
    policy = SieveLogitPolicy(
        pipeline=refit,
        n_actions=n_actions,
        probability_floor=probability_floor,
    )
    return SieveLogitSelection(
        policy=policy,
        selected_degree=selected_degree,
        selected_c=float(selected["c"]),
        validation_nll=float(selected["validation_nll"]),
        candidate_scores=tuple(score_rows),
    )


def fit_gaussian_transition_sieve(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    n_actions: int,
    seed: int,
    degree: int = 3,
    candidate_alpha: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    validation_fraction: float = 0.20,
    residual_sd_floor: float = 0.05,
) -> GaussianTransitionSelection:
    """Fit a Gaussian transition sieve selected by held-out next-state MSE."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    next_states = np.asarray(next_states, dtype=float)
    if (
        states.ndim != 2
        or next_states.shape != states.shape
        or states.shape[0] != actions.size
    ):
        raise ValueError("Transition rows must have compatible state/action shapes.")
    if np.unique(actions).size != n_actions:
        raise ValueError("Transition sieve requires every action in its training source.")
    alpha_values = tuple(float(value) for value in candidate_alpha)
    if not alpha_values or any(value <= 0.0 for value in alpha_values):
        raise ValueError("candidate_alpha must contain positive values.")
    fit_idx, validation_idx = train_test_split(
        np.arange(actions.size),
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=actions,
    )

    def build_pipeline(alpha: float) -> Pipeline:
        return Pipeline(
            steps=[
                ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )

    def fit_action_models(indices: Array, alpha: float) -> tuple[Any, ...]:
        models: list[Any] = []
        for action in range(n_actions):
            action_idx = indices[actions[indices] == action]
            if action_idx.size < 5:
                raise ValueError("Each action needs at least five transition rows.")
            pipeline = build_pipeline(alpha)
            pipeline.fit(states[action_idx], next_states[action_idx])
            models.append(pipeline)
        return tuple(models)

    def predict_with(models: tuple[Any, ...], indices: Array) -> Array:
        prediction = np.empty_like(next_states[indices], dtype=float)
        for action, model in enumerate(models):
            local = np.flatnonzero(actions[indices] == action)
            if local.size:
                prediction[local] = model.predict(states[indices[local]])
        return prediction

    score_rows: list[dict[str, float]] = []
    validation_models: list[tuple[Any, ...]] = []
    for alpha in alpha_values:
        models = fit_action_models(fit_idx, alpha)
        prediction = predict_with(models, validation_idx)
        score_rows.append(
            {
                "alpha": alpha,
                "validation_mse": float(
                    np.mean(np.square(prediction - next_states[validation_idx]))
                ),
            }
        )
        validation_models.append(models)
    selected_index = min(
        range(len(score_rows)),
        key=lambda index: (
            score_rows[index]["validation_mse"],
            -score_rows[index]["alpha"],
        ),
    )
    selected_alpha = float(score_rows[selected_index]["alpha"])
    selected_validation_models = validation_models[selected_index]
    selected_validation_prediction = predict_with(
        selected_validation_models, validation_idx
    )
    residual_sds = np.empty((n_actions, states.shape[1]), dtype=float)
    for action in range(n_actions):
        local = actions[validation_idx] == action
        residual = (
            next_states[validation_idx][local]
            - selected_validation_prediction[local]
        )
        residual_sds[action] = np.maximum(
            np.std(residual, axis=0, ddof=1), residual_sd_floor
        )
    refit_models = fit_action_models(np.arange(actions.size), selected_alpha)
    model = GaussianTransitionSieve(
        models=refit_models,
        residual_sds=residual_sds,
        n_actions=n_actions,
    )
    return GaussianTransitionSelection(
        model=model,
        selected_alpha=selected_alpha,
        validation_mse=float(score_rows[selected_index]["validation_mse"]),
        candidate_scores=tuple(score_rows),
    )


def build_transition_factors_from_sieve(
    *,
    oracle: JRSSBOracle,
    transition: GaussianTransitionSieve,
) -> dict[str, Array]:
    """Discretize the fitted Gaussian transition on the fixed evaluation grid."""
    grid = oracle.main_grid
    n_states = grid.n_states
    px = np.empty((oracle.config.n_actions, n_states, grid.n_points), dtype=float)
    pz = np.empty_like(px)
    for action in range(oracle.config.n_actions):
        actions = np.full(n_states, action, dtype=int)
        means = np.clip(
            transition.predict_mean(grid.states, actions),
            oracle.config.state_low,
            oracle.config.state_high,
        )
        x_hi = normal_cdf(
            (grid.x_bounds[1:][None, :] - means[:, [0]])
            / transition.residual_sds[action, 0]
        )
        x_lo = normal_cdf(
            (grid.x_bounds[:-1][None, :] - means[:, [0]])
            / transition.residual_sds[action, 0]
        )
        z_hi = normal_cdf(
            (grid.z_bounds[1:][None, :] - means[:, [1]])
            / transition.residual_sds[action, 1]
        )
        z_lo = normal_cdf(
            (grid.z_bounds[:-1][None, :] - means[:, [1]])
            / transition.residual_sds[action, 1]
        )
        px[action] = np.clip(x_hi - x_lo, 0.0, None)
        pz[action] = np.clip(z_hi - z_lo, 0.0, None)
        px[action] /= np.clip(np.sum(px[action], axis=1, keepdims=True), EPS, None)
        pz[action] /= np.clip(np.sum(pz[action], axis=1, keepdims=True), EPS, None)
    return {"px": px, "pz": pz}


def evaluate_policy_with_transition_factors(
    *,
    oracle: JRSSBOracle,
    reward_grid: Array,
    policy_grid: Array,
    gamma: float,
    transition_factors: dict[str, Array],
) -> tuple[Array, Array]:
    """Solve a policy Bellman equation using only fitted transition factors."""
    reward_grid = np.asarray(reward_grid, dtype=float)
    policy_grid = np.asarray(policy_grid, dtype=float)
    policy_grid = np.clip(policy_grid, EPS, None)
    policy_grid /= np.sum(policy_grid, axis=1, keepdims=True)
    q_grid = reward_grid.copy()
    n_points = oracle.main_grid.n_points
    for _ in range(oracle.config.max_iterations):
        value = np.sum(policy_grid * q_grid, axis=1)
        value_matrix = value.reshape(n_points, n_points)
        tmp = np.einsum(
            "asx,xz->asz", transition_factors["px"], value_matrix, optimize=True
        )
        continuation = np.einsum(
            "asz,asz->sa", tmp, transition_factors["pz"], optimize=True
        )
        q_next = reward_grid + gamma * continuation
        if np.max(np.abs(q_next - q_grid)) < oracle.config.nuisance_bellman_tol:
            q_grid = q_next
            break
        q_grid = q_next
    value = np.sum(policy_grid * q_grid, axis=1)
    return q_grid, value


def data_fusion_readiness(
    outcome_regression: FrozenOutcomeRegression,
    pilot_median_se: float,
) -> dict[str, float | bool]:
    """Apply the predeclared 10%-of-pilot-SE auxiliary-nuisance gate."""
    if not np.isfinite(pilot_median_se) or pilot_median_se <= 0.0:
        raise ValueError("pilot_median_se must be finite and positive.")
    threshold = 0.10 * float(pilot_median_se)
    shift = abs(float(outcome_regression.estimand_shift))
    return {
        "pilot_median_se": float(pilot_median_se),
        "allowed_absolute_shift": threshold,
        "observed_absolute_shift": shift,
        "passed": bool(shift <= threshold),
    }


def data_fusion_if_contribution(
    *,
    value: Array,
    state_action_ratio: Array,
    normalization_state_action_ratio: Array,
    target_bellman_residual: Array,
    normalization_bellman_residual: Array,
    behavior_log_scale: float,
) -> Array:
    """Evaluate the normalized-reward EIF at a known softmax scale."""
    contrast = state_action_ratio - normalization_state_action_ratio
    return (
        value
        + state_action_ratio * target_bellman_residual
        + contrast * normalization_bellman_residual
        + behavior_log_scale * contrast
    )


def run_data_fusion_replication(
    *,
    oracle: JRSSBOracle,
    outcome_regression: FrozenOutcomeRegression,
    n: int,
    seed: int,
) -> DataFusionRunResult:
    """Run one configurable K-fold cross-fitted data-fusion replication."""
    if n < 10:
        raise ValueError("n must be at least 10.")
    data = oracle.sample_stationary_transitions(n=n, seed=seed)
    if oracle.config.data_fusion_policy_mode == "known-logging":
        data["logging_probabilities"] = oracle.policy_probs(data["states"], oracle.pi0)
    truth = build_data_fusion_truth(oracle)
    repeated_splits = max(1, int(oracle.config.data_fusion_repeated_splits))
    plugin_split_estimates: list[float] = []
    if_split_estimates: list[float] = []
    selected_split_ses: list[float] = []
    iid_split_ses: list[float] = []
    fold_cluster_split_ses: list[float] = []
    reward_squared_errors: list[Array] = []
    ratio_values: list[Array] = []
    ratio_diagnostic_rows: list[dict[str, Any]] = []
    all_indices = np.arange(n)
    for split_number in range(repeated_splits):
        split_seed = seed + 100_003 * split_number
        plugin = np.zeros(n, dtype=float)
        influence = np.zeros(n, dtype=float)
        fold_indices = fold_splits(
            n,
            seed=split_seed + 101,
            n_folds=oracle.config.crossfit_folds,
        )
        for fold_number, eval_idx in enumerate(fold_indices):
            train_mask = np.ones(n, dtype=bool)
            train_mask[eval_idx] = False
            train_idx = all_indices[train_mask]
            fold_seed = split_seed * 10 + fold_number + 1
            fold = _evaluate_data_fusion_fold(
                oracle=oracle,
                outcome_regression=outcome_regression,
                data=data,
                train_idx=train_idx,
                eval_idx=eval_idx,
                seed=fold_seed,
            )
            plugin[eval_idx] = fold["plugin"]
            influence[eval_idx] = fold["if"]
            reward_squared_errors.append(fold["reward_squared_error"])
            ratio_values.append(fold["ratio"])
            ratio_diagnostic_rows.append(fold["ratio_diagnostics"])
        se_diagnostics = crossfit_se_diagnostics(influence, fold_indices)
        plugin_split_estimates.append(float(np.mean(plugin)))
        if_split_estimates.append(float(np.mean(influence)))
        selected_split_ses.append(
            selected_crossfit_se(se_diagnostics, oracle.config.crossfit_se_method)
        )
        iid_split_ses.append(float(se_diagnostics["iid_se"]))
        fold_cluster_split_ses.append(float(se_diagnostics["fold_cluster_se"]))
    plugin_estimate = float(np.mean(plugin_split_estimates))
    if_estimate = float(np.mean(if_split_estimates))
    estimated_se = combine_repeated_split_se(
        if_split_estimates,
        selected_split_ses,
    )
    critical_value = crossfit_critical_value(
        1.96,
        int(se_diagnostics["fold_count"]),
        oracle.config.crossfit_ci_method,
    )
    halfwidth = critical_value * estimated_se
    ratios = np.concatenate(ratio_values)
    squared_sum = float(np.sum(np.square(ratios)))
    ess = (
        float(np.square(np.sum(ratios)) / squared_sum) / repeated_splits
        if squared_sum > 0.0
        else 0.0
    )

    def mean_diagnostic(name: str) -> float:
        return float(np.mean([float(row[name]) for row in ratio_diagnostic_rows]))

    return DataFusionRunResult(
        n=n,
        seed=seed,
        outcome_sample_size=outcome_regression.sample_size,
        truth=truth.psi,
        plugin_estimate=plugin_estimate,
        if_estimate=if_estimate,
        plugin_error=plugin_estimate - truth.psi,
        if_error=if_estimate - truth.psi,
        estimated_se=estimated_se,
        ci_lower=if_estimate - halfwidth,
        ci_upper=if_estimate + halfwidth,
        covered=float(if_estimate - halfwidth <= truth.psi <= if_estimate + halfwidth),
        ci_length=2.0 * halfwidth,
        reward_rmse=float(np.sqrt(np.mean(np.concatenate(reward_squared_errors)))),
        g_rmse=outcome_regression.g_rmse,
        g_estimand_shift=outcome_regression.estimand_shift,
        ratio_q99=float(np.quantile(ratios, 0.99)),
        ratio_max=float(np.max(ratios)),
        ratio_ess=ess,
        fore_selected_iterations=mean_diagnostic("selected_iterations"),
        fore_apbv_score=mean_diagnostic("apbv_score"),
        fore_fit_seconds=mean_diagnostic("fit_seconds"),
        fore_normalized_mass=mean_diagnostic("normalized_mass"),
        fore_logit_cap_fraction=mean_diagnostic("logit_cap_fraction"),
        behavior_policy_selected_c=mean_diagnostic("behavior_policy_selected_c"),
        behavior_policy_validation_nll=mean_diagnostic(
            "behavior_policy_validation_nll"
        ),
        transition_selected_alpha=mean_diagnostic("transition_selected_alpha"),
        transition_validation_mse=mean_diagnostic("transition_validation_mse"),
        transition_residual_sd_x=mean_diagnostic("transition_residual_sd_x"),
        transition_residual_sd_z=mean_diagnostic("transition_residual_sd_z"),
        ratio_failure=float(any(bool(row["nonfinite"]) for row in ratio_diagnostic_rows)),
        fore_selected_iterations_by_fold=json.dumps(
            [
                int(row["selected_iterations"])
                if np.isfinite(float(row["selected_iterations"]))
                else None
                for row in ratio_diagnostic_rows
            ]
        ),
        fore_apbv_scores_by_fold=json.dumps(
            [
                float(row["apbv_score"])
                if np.isfinite(float(row["apbv_score"]))
                else None
                for row in ratio_diagnostic_rows
            ]
        ),
        iid_estimated_se=float(np.sqrt(np.mean(np.square(iid_split_ses)))),
        fold_cluster_estimated_se=float(
            np.sqrt(np.mean(np.square(fold_cluster_split_ses)))
        ),
        crossfit_fold_count=float(oracle.config.crossfit_folds),
        ci_critical_value=critical_value,
        behavior_policy_selected_degree=mean_diagnostic(
            "behavior_policy_selected_degree"
        ),
        data_fusion_policy_mode=oracle.config.data_fusion_policy_mode,
        data_fusion_transition_mode=oracle.config.data_fusion_transition_mode,
        data_fusion_g_mode=oracle.config.data_fusion_g_mode,
        data_fusion_ratio_mode=oracle.config.data_fusion_ratio_mode,
        normalization_policy_mode=outcome_regression.normalization_policy_mode,
        data_fusion_target_gamma=oracle.config.data_fusion_target_gamma,
        data_fusion_repeated_splits=float(repeated_splits),
        split_estimate_sd=(
            float(np.std(if_split_estimates, ddof=1))
            if repeated_splits > 1
            else 0.0
        ),
        behavior_probability_floor=oracle.config.data_fusion_probability_floor,
        behavior_probability_clipping_fraction=mean_diagnostic(
            "behavior_probability_clipping_fraction"
        ),
    )


def run_data_fusion_monte_carlo(
    *,
    oracle: JRSSBOracle,
    outcome_regression: FrozenOutcomeRegression,
    sample_sizes: Sequence[int] = (2500, 5000, 10000),
    repetitions: int = 300,
    seed: int = 404,
    jobs: int = 1,
    replication_indices: Optional[Sequence[int]] = None,
    on_result: Optional[Callable[[DataFusionRunResult], None]] = None,
) -> list[DataFusionRunResult]:
    """Run deterministic cells, optionally checkpointing parent-side results."""
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
    tasks = [
        (int(n), seed + 100_003 * repetition + 97 * int(n))
        for n in sample_sizes
        for repetition in requested_replications
    ]

    def collect(
        iterator: Iterable[DataFusionRunResult],
    ) -> list[DataFusionRunResult]:
        collected: list[DataFusionRunResult] = []
        for result in iterator:
            collected.append(result)
            if on_result is not None:
                on_result(result)
        return collected

    if jobs == 1:
        return collect(
            run_data_fusion_replication_safe(
                oracle=oracle,
                outcome_regression=outcome_regression,
                n=n,
                seed=replication_seed,
            )
            for n, replication_seed in tasks
        )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs,
        initializer=_initialize_data_fusion_worker,
        initargs=(oracle.config, outcome_regression),
    ) as executor:
        return collect(executor.map(_run_data_fusion_worker, tasks))


_DATA_FUSION_WORKER_ORACLE: Optional[JRSSBOracle] = None
_DATA_FUSION_WORKER_OUTCOME: Optional[FrozenOutcomeRegression] = None


def _initialize_data_fusion_worker(
    config: JRSSBConfig,
    outcome_regression: FrozenOutcomeRegression,
) -> None:
    """Initialize one process with the immutable oracle and frozen auxiliary fit."""
    global _DATA_FUSION_WORKER_ORACLE, _DATA_FUSION_WORKER_OUTCOME
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
    _DATA_FUSION_WORKER_ORACLE = JRSSBOracle(config)
    _DATA_FUSION_WORKER_OUTCOME = outcome_regression


def _run_data_fusion_worker(task: tuple[int, int]) -> DataFusionRunResult:
    """Execute a deterministic replication inside an initialized worker."""
    if _DATA_FUSION_WORKER_ORACLE is None or _DATA_FUSION_WORKER_OUTCOME is None:
        raise RuntimeError("The data-fusion worker was not initialized.")
    n, seed = task
    return run_data_fusion_replication_safe(
        oracle=_DATA_FUSION_WORKER_ORACLE,
        outcome_regression=_DATA_FUSION_WORKER_OUTCOME,
        n=n,
        seed=seed,
    )


def _recordable_data_fusion_failure(exc: Exception) -> bool:
    if isinstance(exc, (FloatingPointError, OverflowError, np.linalg.LinAlgError)):
        return True
    message = str(exc).lower()
    return isinstance(exc, (RuntimeError, ValueError)) and any(
        marker in message
        for marker in (
            "nonfinite",
            "non-finite",
            "nan",
            "infinite",
            "numerical",
            "singular",
        )
    )


def failed_data_fusion_result(
    *,
    oracle: JRSSBOracle,
    outcome_regression: FrozenOutcomeRegression,
    n: int,
    seed: int,
    exc: Exception,
) -> DataFusionRunResult:
    """Create a labeled numerical-failure row without stopping the cell."""
    nan = float("nan")
    return DataFusionRunResult(
        n=n,
        seed=seed,
        outcome_sample_size=outcome_regression.sample_size,
        truth=build_data_fusion_truth(oracle).psi,
        plugin_estimate=nan,
        if_estimate=nan,
        plugin_error=nan,
        if_error=nan,
        estimated_se=nan,
        ci_lower=nan,
        ci_upper=nan,
        covered=nan,
        ci_length=nan,
        reward_rmse=nan,
        g_rmse=outcome_regression.g_rmse,
        g_estimand_shift=outcome_regression.estimand_shift,
        ratio_q99=nan,
        ratio_max=nan,
        ratio_ess=nan,
        fore_selected_iterations=nan,
        fore_apbv_score=nan,
        fore_fit_seconds=nan,
        fore_normalized_mass=nan,
        fore_logit_cap_fraction=nan,
        behavior_policy_selected_c=nan,
        behavior_policy_validation_nll=nan,
        transition_selected_alpha=nan,
        transition_validation_mse=nan,
        transition_residual_sd_x=nan,
        transition_residual_sd_z=nan,
        ratio_failure=1.0,
        fore_selected_iterations_by_fold="",
        fore_apbv_scores_by_fold="",
        failure_message=f"{type(exc).__name__}: {exc}",
        iid_estimated_se=nan,
        fold_cluster_estimated_se=nan,
        crossfit_fold_count=float(oracle.config.crossfit_folds),
        ci_critical_value=nan,
        behavior_policy_selected_degree=nan,
        data_fusion_policy_mode=oracle.config.data_fusion_policy_mode,
        data_fusion_transition_mode=oracle.config.data_fusion_transition_mode,
        data_fusion_g_mode=oracle.config.data_fusion_g_mode,
        data_fusion_ratio_mode=oracle.config.data_fusion_ratio_mode,
    )


def run_data_fusion_replication_safe(
    *,
    oracle: JRSSBOracle,
    outcome_regression: FrozenOutcomeRegression,
    n: int,
    seed: int,
) -> DataFusionRunResult:
    """Record numerical failures while preserving programming errors."""
    try:
        return run_data_fusion_replication(
            oracle=oracle,
            outcome_regression=outcome_regression,
            n=n,
            seed=seed,
        )
    except Exception as exc:
        if not _recordable_data_fusion_failure(exc):
            raise
        return failed_data_fusion_result(
            oracle=oracle,
            outcome_regression=outcome_regression,
            n=n,
            seed=seed,
            exc=exc,
        )


def summarize_data_fusion_results(
    results: Sequence[DataFusionRunResult],
) -> list[dict[str, Any]]:
    """Summarize point and interval performance by behavior-sample size."""
    rows: list[dict[str, Any]] = []
    for n in sorted({result.n for result in results}):
        all_cell = [result for result in results if result.n == n]
        cell = [
            result
            for result in all_cell
            if result.ratio_failure == 0.0
            and np.isfinite(result.plugin_estimate)
            and np.isfinite(result.if_estimate)
        ]
        plugin_error = np.asarray([result.plugin_error for result in cell])
        if_error = np.asarray([result.if_error for result in cell])
        plugin_estimate = np.asarray([result.plugin_estimate for result in cell])
        if_estimate = np.asarray([result.if_estimate for result in cell])
        rows.append(
            {
                "n": n,
                "repetitions": len(all_cell),
                "successful_repetitions": len(cell),
                "truth": all_cell[0].truth,
                "outcome_sample_size": all_cell[0].outcome_sample_size,
                "plugin_bias": _safe_mean(plugin_error),
                "plugin_sd": float(np.std(plugin_estimate, ddof=1)) if len(cell) > 1 else np.nan,
                "plugin_rmse": float(np.sqrt(_safe_mean(np.square(plugin_error)))),
                "if_bias": _safe_mean(if_error),
                "if_sd": float(np.std(if_estimate, ddof=1)) if len(cell) > 1 else np.nan,
                "if_rmse": float(np.sqrt(_safe_mean(np.square(if_error)))),
                "avg_estimated_se": _safe_mean([result.estimated_se for result in cell]),
                "avg_iid_estimated_se": _safe_mean(
                    [result.iid_estimated_se for result in cell]
                ),
                "avg_fold_cluster_estimated_se": _safe_mean(
                    [result.fold_cluster_estimated_se for result in cell]
                ),
                "avg_ci_critical_value": _safe_mean(
                    [result.ci_critical_value for result in cell]
                ),
                "repeated_splits": all_cell[0].data_fusion_repeated_splits,
                "avg_split_estimate_sd": _safe_mean(
                    [result.split_estimate_sd for result in cell]
                ),
                "coverage_95": _safe_mean([result.covered for result in cell]),
                "avg_ci_length": _safe_mean([result.ci_length for result in cell]),
                "avg_reward_rmse": _safe_mean([result.reward_rmse for result in cell]),
                "g_rmse": all_cell[0].g_rmse,
                "g_estimand_shift": all_cell[0].g_estimand_shift,
                "avg_ratio_q99": _safe_mean([result.ratio_q99 for result in cell]),
                "avg_ratio_ess": _safe_mean([result.ratio_ess for result in cell]),
                "avg_fore_iterations": _safe_mean([result.fore_selected_iterations for result in cell]),
                "avg_fore_apbv_score": _safe_mean([result.fore_apbv_score for result in cell]),
                "avg_fore_fit_seconds": _safe_mean([result.fore_fit_seconds for result in cell]),
                "avg_behavior_policy_selected_c": _safe_mean(
                    [result.behavior_policy_selected_c for result in cell]
                ),
                "avg_behavior_policy_selected_degree": _safe_mean(
                    [result.behavior_policy_selected_degree for result in cell]
                ),
                "avg_behavior_policy_validation_nll": _safe_mean(
                    [result.behavior_policy_validation_nll for result in cell]
                ),
                "behavior_probability_floor": all_cell[0].behavior_probability_floor,
                "avg_behavior_probability_clipping_fraction": _safe_mean(
                    [result.behavior_probability_clipping_fraction for result in cell]
                ),
                "avg_transition_selected_alpha": _safe_mean(
                    [result.transition_selected_alpha for result in cell]
                ),
                "avg_transition_validation_mse": _safe_mean(
                    [result.transition_validation_mse for result in cell]
                ),
                "avg_transition_residual_sd_x": _safe_mean(
                    [result.transition_residual_sd_x for result in cell]
                ),
                "avg_transition_residual_sd_z": _safe_mean(
                    [result.transition_residual_sd_z for result in cell]
                ),
                "ratio_failure_rate": float(
                    np.mean([result.ratio_failure for result in all_cell])
                ),
            }
        )
    return rows


def _safe_mean(values: Sequence[float] | Array) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmean(array))


def _evaluate_data_fusion_fold(
    *,
    oracle: JRSSBOracle,
    outcome_regression: FrozenOutcomeRegression,
    data: dict[str, Array],
    train_idx: Array,
    eval_idx: Array,
    seed: int,
) -> dict[str, Any]:
    train_states = data["states"][train_idx]
    train_actions = data["actions"][train_idx]
    train_next_states = data["next_states"][train_idx]
    eval_states = data["states"][eval_idx]
    eval_actions = data["actions"][eval_idx]
    eval_next_states = data["next_states"][eval_idx]

    policy_selection: SieveLogitSelection | None
    if oracle.config.data_fusion_policy_mode in {"known-logging", "oracle"}:
        policy_selection = None
        policy_hat = GridProbabilityPolicy(oracle=oracle, policy_grid=oracle.pi0)
    else:
        policy_selection = fit_sieve_logit_behavior_policy(
            states=train_states,
            actions=train_actions,
            n_actions=oracle.config.n_actions,
            seed=seed,
            probability_floor=oracle.config.data_fusion_probability_floor,
        )
        policy_hat = policy_selection.policy
    target_adapter = ProbabilityPolicyAdapter(
        oracle._fixed_policy_probs, oracle.config.n_actions
    )
    transition_selection: GaussianTransitionSelection | None
    if oracle.config.data_fusion_transition_mode == "oracle":
        transition_selection = None
        transition_factors = oracle._transition_factors_main
    else:
        transition_selection = fit_gaussian_transition_sieve(
            states=train_states,
            actions=train_actions,
            next_states=train_next_states,
            n_actions=oracle.config.n_actions,
            seed=seed + 71,
        )
        transition_factors = build_transition_factors_from_sieve(
            oracle=oracle,
            transition=transition_selection.model,
        )
    grid_states = oracle.main_grid.states
    behavior_grid = policy_hat.predict_proba(grid_states)
    g_grid = (
        data_fusion_true_g(oracle, grid_states)
        if oracle.config.data_fusion_g_mode == "oracle"
        else outcome_regression.predict(grid_states)
    )
    u_grid = oracle.config.tau_behavior * np.log(np.clip(behavior_grid, EPS, None))
    normalization_grid = data_fusion_normalization_policy(oracle, grid_states)
    q_mu_grid, v_mu_grid = evaluate_policy_with_transition_factors(
        oracle=oracle,
        reward_grid=u_grid - g_grid[:, None],
        policy_grid=normalization_grid,
        gamma=oracle.config.gamma_behavior,
        transition_factors=transition_factors,
    )
    recovered_reward_grid = q_mu_grid - v_mu_grid[:, None] + g_grid[:, None]
    target_grid = oracle._fixed_policy_probs(grid_states)
    q_target_grid, _ = evaluate_policy_with_transition_factors(
        oracle=oracle,
        reward_grid=recovered_reward_grid,
        policy_grid=target_grid,
        gamma=oracle.config.data_fusion_target_gamma,
        transition_factors=transition_factors,
    )

    candidates = paper_early_stopping_candidates(
        hidden_dims=tuple(oracle.config.fore_hidden_sizes),
        learning_rate=oracle.config.fore_learning_rate,
        weight_decay=oracle.config.fore_weight_decay,
        iteration_budgets=oracle.config.fore_iteration_budgets,
    )
    options = FOREFitOptions(
        batch_size=oracle.config.fore_batch_size,
        optimizer_steps=oracle.config.fore_optimizer_steps,
        target_action_draws=oracle.config.fore_target_action_draws,
        logit_clip=oracle.config.fore_logit_clip,
        grad_clip_norm=oracle.config.fore_grad_clip_norm,
        device=oracle.config.fore_device,
    )
    selected_fore = None
    if oracle.config.data_fusion_ratio_mode == "neural-fore":
        selected_fore = fit_selected_fore_ratio(
            states=train_states,
            actions=train_actions,
            next_states=train_next_states,
            target_policy=target_adapter,
            gamma=oracle.config.data_fusion_target_gamma,
            n_actions=oracle.config.n_actions,
            candidates=candidates,
            seed=seed + 307,
            options=options,
        )

    behavior_eval = policy_hat.predict_proba(eval_states)
    normalization_eval = data_fusion_normalization_policy(oracle, eval_states)
    normalization_next = data_fusion_normalization_policy(
        oracle, eval_next_states
    )
    q_mu_eval = oracle.action_values(eval_states, q_mu_grid)
    q_mu_next = oracle.action_values(eval_next_states, q_mu_grid)
    v_mu_eval = np.sum(normalization_eval * q_mu_eval, axis=1)
    v_mu_next = np.sum(normalization_next * q_mu_next, axis=1)
    g_eval = (
        data_fusion_true_g(oracle, eval_states)
        if oracle.config.data_fusion_g_mode == "oracle"
        else outcome_regression.predict(eval_states)
    )
    recovered_reward_eval = q_mu_eval - v_mu_eval[:, None] + g_eval[:, None]
    q_target_eval = oracle.action_values(eval_states, q_target_grid)
    q_target_next = oracle.action_values(eval_next_states, q_target_grid)
    target_eval = oracle._fixed_policy_probs(eval_states)
    target_next = oracle._fixed_policy_probs(eval_next_states)
    v_target = np.sum(target_eval * q_target_eval, axis=1)
    v_target_next = np.sum(target_next * q_target_next, axis=1)
    q_mu_sa = q_mu_eval[np.arange(eval_idx.size), eval_actions]
    q_target_sa = q_target_eval[np.arange(eval_idx.size), eval_actions]
    reward_sa = recovered_reward_eval[np.arange(eval_idx.size), eval_actions]
    u_eval = oracle.config.tau_behavior * np.log(
        np.clip(behavior_eval[np.arange(eval_idx.size), eval_actions], EPS, None)
    )
    if selected_fore is None:
        rho_eval = oracle.state_values(eval_states, oracle.rho_fix_data_fusion)
        target_action_probability = target_eval[
            np.arange(eval_idx.size), eval_actions
        ]
        behavior_action_probability = behavior_eval[
            np.arange(eval_idx.size), eval_actions
        ]
        d_eval = rho_eval * target_action_probability / np.clip(
            behavior_action_probability,
            EPS,
            None,
        )
    else:
        d_eval = selected_fore.ratio.predict_unnormalized(
            eval_states, eval_actions
        )
        rho_eval = selected_fore.ratio.predict_state_ratio(
            eval_states, behavior_eval
        )
    behavior_action_probability = behavior_eval[
        np.arange(eval_idx.size), eval_actions
    ]
    normalization_action_probability = normalization_eval[
        np.arange(eval_idx.size), eval_actions
    ]
    normalization_d_eval = (
        rho_eval
        * normalization_action_probability
        / np.clip(behavior_action_probability, EPS, None)
    )
    target_bellman = (
        reward_sa
        + oracle.config.data_fusion_target_gamma * v_target_next
        - q_target_sa
    )
    normalization_bellman = (
        u_eval - g_eval + oracle.config.gamma_behavior * v_mu_next - q_mu_sa
    )
    contribution = data_fusion_if_contribution(
        value=v_target,
        state_action_ratio=d_eval,
        normalization_state_action_ratio=normalization_d_eval,
        target_bellman_residual=target_bellman,
        normalization_bellman_residual=normalization_bellman,
        behavior_log_scale=oracle.config.tau_behavior,
    )
    reward_true = oracle.action_values(eval_states, oracle.reward_dagger)
    if selected_fore is None:
        diagnostics = {
            "selected_iterations": np.nan,
            "apbv_score": np.nan,
            "fit_seconds": 0.0,
            "normalized_mass": float(
                (1.0 - oracle.config.data_fusion_target_gamma) * np.mean(d_eval)
            ),
            "logit_cap_fraction": 0.0,
            "nonfinite": float(not np.all(np.isfinite(d_eval))),
        }
    else:
        diagnostics = ratio_diagnostics(
            selected_fore.ratio, train_states, train_actions
        )
        diagnostics["apbv_score"] = float(
            selected_fore.selection.worst_case_scores[
                selected_fore.selection.selected_index
            ]
        )
        diagnostics["fit_seconds"] = selected_fore.total_fit_seconds
    diagnostics["behavior_policy_selected_c"] = (
        policy_selection.selected_c if policy_selection is not None else np.nan
    )
    diagnostics["behavior_policy_selected_degree"] = (
        float(policy_selection.selected_degree)
        if policy_selection is not None
        else np.nan
    )
    diagnostics["behavior_policy_validation_nll"] = (
        policy_selection.validation_nll if policy_selection is not None else np.nan
    )
    diagnostics["behavior_probability_clipping_fraction"] = (
        policy_selection.policy.clipping_fraction(eval_states)
        if policy_selection is not None
        else 0.0
    )
    diagnostics["transition_selected_alpha"] = (
        transition_selection.selected_alpha
        if transition_selection is not None
        else np.nan
    )
    diagnostics["transition_validation_mse"] = (
        transition_selection.validation_mse
        if transition_selection is not None
        else np.nan
    )
    diagnostics["transition_residual_sd_x"] = (
        float(np.mean(transition_selection.model.residual_sds[:, 0]))
        if transition_selection is not None
        else oracle.config.noise_scale
    )
    diagnostics["transition_residual_sd_z"] = (
        float(np.mean(transition_selection.model.residual_sds[:, 1]))
        if transition_selection is not None
        else oracle.config.noise_scale
    )
    return {
        "plugin": v_target,
        "if": contribution,
        "reward_squared_error": np.square(recovered_reward_eval - reward_true).reshape(-1),
        "ratio": d_eval,
        "ratio_diagnostics": diagnostics,
    }


__all__ = [
    "DataFusionRunResult",
    "DataFusionTruth",
    "FrozenOutcomeRegression",
    "CalibratedPredictionEnsemble",
    "GaussianTransitionSelection",
    "GaussianTransitionSieve",
    "SieveLogitPolicy",
    "SieveLogitSelection",
    "build_data_fusion_truth",
    "data_fusion_normalization_policy",
    "data_fusion_readiness",
    "data_fusion_true_g",
    "data_fusion_if_contribution",
    "build_transition_factors_from_sieve",
    "evaluate_policy_with_transition_factors",
    "fit_frozen_outcome_regression",
    "fit_gaussian_transition_sieve",
    "fit_sieve_logit_behavior_policy",
    "generate_outcome_source",
    "run_data_fusion_monte_carlo",
    "run_data_fusion_replication",
    "summarize_data_fusion_results",
]
