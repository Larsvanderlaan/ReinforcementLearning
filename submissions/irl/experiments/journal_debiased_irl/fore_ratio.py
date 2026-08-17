"""Neural FORE ratios and truth-blind pairwise Bellman validation.

This module is intentionally experiment-local.  It adapts the public neural
KL-FORE implementation to the JASA simulation without changing the shared
``occupancy_ratio`` package while that package has other work in progress.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional, Protocol, Sequence

import numpy as np


Array = np.ndarray
NUMERICAL_EPS = 1e-12


class ProbabilityPolicy(Protocol):
    """Minimal target-policy interface used by the experiment adapter."""

    def predict_proba(self, states: Array) -> Array:
        """Return one probability row per state."""


@dataclass(frozen=True)
class FORECandidateConfig:
    """One neural FORE path candidate."""

    hidden_dims: tuple[int, ...] = (64, 64)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    num_iterations: int = 30

    def __post_init__(self) -> None:
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative.")
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive.")

    @property
    def candidate_id(self) -> str:
        widths = "x".join(str(width) for width in self.hidden_dims)
        return (
            f"h{widths}_lr{self.learning_rate:g}_wd{self.weight_decay:g}"
            f"_k{self.num_iterations}"
        )


@dataclass(frozen=True)
class FOREFitOptions:
    """Fixed numerical settings shared by all paper candidates."""

    batch_size: int = 512
    optimizer_steps: int = 5
    target_action_draws: int = 4
    logit_clip: float = 10.0
    grad_clip_norm: float = 10.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.optimizer_steps <= 0:
            raise ValueError("batch_size and optimizer_steps must be positive.")
        if self.target_action_draws <= 0:
            raise ValueError("target_action_draws must be positive.")
        if self.logit_clip <= 0.0 or self.grad_clip_norm <= 0.0:
            raise ValueError("logit_clip and grad_clip_norm must be positive.")


@dataclass
class FittedFORERatio:
    """A positive normalized FORE ratio and its experiment metadata."""

    model: Any
    gamma: float
    n_actions: int
    candidate: FORECandidateConfig
    fit_seconds: float

    def predict_normalized(self, states: Array, actions: Array) -> Array:
        """Predict the normalized discounted state-action ratio."""
        action_features = one_hot_actions(actions, self.n_actions)
        values = np.asarray(
            self.model.predict_state_action_ratio(states, action_features),
            dtype=float,
        ).reshape(-1)
        _validate_finite_nonnegative(values, "FORE predictions")
        return values

    def predict_unnormalized(self, states: Array, actions: Array) -> Array:
        """Predict the unnormalized JASA discounted state-action ratio."""
        return self.predict_normalized(states, actions) / (1.0 - self.gamma)

    def predict_all_normalized(self, states: Array) -> Array:
        """Predict one normalized ratio for every state-action pair."""
        states = _as_states(states, "states")
        repeated_states = np.repeat(states, self.n_actions, axis=0)
        tiled_actions = np.tile(np.arange(self.n_actions), states.shape[0])
        return self.predict_normalized(repeated_states, tiled_actions).reshape(
            states.shape[0], self.n_actions
        )

    def predict_state_ratio(self, states: Array, behavior_probs: Array) -> Array:
        """Recover the unnormalized state ratio under the behavior law."""
        probs = _validate_probabilities(behavior_probs, self.n_actions)
        all_ratios = self.predict_all_normalized(states) / (1.0 - self.gamma)
        return np.sum(probs * all_ratios, axis=1)


@dataclass(frozen=True)
class APBVSelectionResult:
    """Result of adversarial pairwise Bellman validation."""

    candidate_ids: tuple[str, ...]
    score_matrix: Array
    worst_case_scores: Array
    selected_index: int

    @property
    def selected_candidate_id(self) -> str:
        return self.candidate_ids[self.selected_index]

    def rows(self, component: str = "ordinary") -> list[dict[str, Any]]:
        """Return serializable candidate-level selection telemetry."""
        return [
            {
                "component": component,
                "candidate_id": candidate_id,
                "candidate_index": index,
                "worst_case_apbv": float(self.worst_case_scores[index]),
                "selected": bool(index == self.selected_index),
            }
            for index, candidate_id in enumerate(self.candidate_ids)
        ]


@dataclass
class SelectedFORERatio:
    """Final refitted ratio and its held-out selection record."""

    ratio: FittedFORERatio
    selection_ratio: FittedFORERatio
    selection: APBVSelectionResult
    split_sizes: tuple[int, int, int]
    candidate_fit_seconds: float = 0.0
    total_fit_seconds: float = 0.0


@dataclass
class SignedFORERatio:
    """Signed advantage-weighted ratio represented by a Jordan decomposition."""

    gamma: float
    n_actions: int
    positive: Optional[FittedFORERatio]
    negative: Optional[FittedFORERatio]
    positive_mass: float
    negative_mass: float
    positive_selection: Optional[APBVSelectionResult]
    negative_selection: Optional[APBVSelectionResult]
    positive_candidate_fit_seconds: float = 0.0
    negative_candidate_fit_seconds: float = 0.0
    positive_total_fit_seconds: float = 0.0
    negative_total_fit_seconds: float = 0.0

    def predict_future_unnormalized(self, states: Array, actions: Array) -> Array:
        """Predict the post-transition part of the signed ratio.

        The Jordan FORE fits start from successor states, so they estimate the
        strictly future resolvent term.  The full signed representer also has
        the current term ``d(s, a) * (q(s, a) - V(s))``; callers must add it
        before evaluating the adaptive-policy influence function.
        """
        states = _as_states(states, "states")
        actions = _as_integer_actions(actions, states.shape[0], self.n_actions)
        positive = np.zeros(states.shape[0], dtype=float)
        negative = np.zeros(states.shape[0], dtype=float)
        if self.positive is not None:
            positive = self.positive.predict_normalized(states, actions)
        if self.negative is not None:
            negative = self.negative.predict_normalized(states, actions)
        scale = self.gamma / (1.0 - self.gamma)
        return scale * (
            self.positive_mass * positive - self.negative_mass * negative
        )

    def predict_future_state_ratio(
        self, states: Array, behavior_probs: Array
    ) -> Array:
        """Average the post-transition signed ratio under the behavior law."""
        states = _as_states(states, "states")
        probs = _validate_probabilities(behavior_probs, self.n_actions)
        repeated_states = np.repeat(states, self.n_actions, axis=0)
        tiled_actions = np.tile(np.arange(self.n_actions), states.shape[0])
        all_ratios = self.predict_future_unnormalized(
            repeated_states, tiled_actions
        ).reshape(states.shape[0], self.n_actions)
        return np.sum(probs * all_ratios, axis=1)


def paper_early_stopping_candidates(
    *,
    hidden_dims: tuple[int, ...] = (64, 64),
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    iteration_budgets: Sequence[int] = (10, 30, 100),
) -> tuple[FORECandidateConfig, ...]:
    """Return the frozen-config early-stopping library."""
    return tuple(
        FORECandidateConfig(
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_iterations=int(budget),
        )
        for budget in iteration_budgets
    )


def paper_pilot_candidates(
    iteration_budgets: Sequence[int] = (10, 30, 100),
) -> tuple[FORECandidateConfig, ...]:
    """Return the predeclared truth-blind pilot library."""
    return tuple(
        FORECandidateConfig(
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_iterations=int(iterations),
        )
        for hidden_dims in ((64, 64), (128, 128))
        for learning_rate in (3e-4, 1e-3)
        for weight_decay in (1e-3, 1e-2)
        for iterations in iteration_budgets
    )


def one_hot_actions(actions: Array, n_actions: int) -> Array:
    """Encode integer actions without imposing an ordinal geometry."""
    actions = np.asarray(actions)
    if actions.ndim == 2 and actions.shape[1] == n_actions:
        encoded = np.asarray(actions, dtype=float)
        if not np.all(np.isfinite(encoded)):
            raise ValueError("one-hot actions must be finite.")
        return encoded
    integer = _as_integer_actions(actions, np.asarray(actions).size, n_actions)
    return np.eye(n_actions, dtype=float)[integer]


def deterministic_three_way_split(
    n_rows: int,
    seed: int,
    fractions: tuple[float, float, float] = (0.60, 0.20, 0.20),
) -> tuple[Array, Array, Array]:
    """Split rows into fit, ordinary-validation, and signed-validation sets."""
    if n_rows < 5:
        raise ValueError("At least five rows are required for a 60/20/20 split.")
    fractions_array = np.asarray(fractions, dtype=float)
    if fractions_array.shape != (3,) or np.any(fractions_array <= 0.0):
        raise ValueError("fractions must contain three positive values.")
    fractions_array /= fractions_array.sum()
    first = max(1, int(math.floor(fractions_array[0] * n_rows)))
    second = max(first + 1, int(math.floor(np.sum(fractions_array[:2]) * n_rows)))
    second = min(second, n_rows - 1)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_rows)
    return permutation[:first], permutation[first:second], permutation[second:]


def adversarial_pairwise_bellman_validation(
    candidates: Sequence[FittedFORERatio],
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    initial_states: Array,
    target_policy: ProbabilityPolicy,
    gamma: float,
    initial_weights: Optional[Array] = None,
) -> APBVSelectionResult:
    """Select a positive ratio by held-out adversarial pairwise validation.

    The score is the A-PBV ratio score from the supplied model-selection
    manuscript.  It uses only held-out transitions, the target policy, and the
    empirical initial source.  No oracle ratio, reward, value, or estimand is an
    input to this function.
    """
    if not candidates:
        raise ValueError("At least one candidate is required.")
    states = _as_states(states, "states")
    next_states = _as_states(next_states, "next_states")
    if next_states.shape != states.shape:
        raise ValueError("states and next_states must have identical shapes.")
    actions = _as_integer_actions(actions, states.shape[0], candidates[0].n_actions)
    initial_states = _as_states(initial_states, "initial_states")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1).")
    n_actions = candidates[0].n_actions
    if any(candidate.n_actions != n_actions for candidate in candidates):
        raise ValueError("All candidates must use the same action space.")

    current = np.vstack(
        [candidate.predict_normalized(states, actions) for candidate in candidates]
    )
    next_all = np.stack(
        [candidate.predict_all_normalized(next_states) for candidate in candidates]
    )
    initial_all = np.stack(
        [candidate.predict_all_normalized(initial_states) for candidate in candidates]
    )
    target_next = _validate_probabilities(target_policy.predict_proba(next_states), n_actions)
    target_initial = _validate_probabilities(
        target_policy.predict_proba(initial_states), n_actions
    )
    init_weight = _normalized_weights(initial_weights, initial_states.shape[0])

    log_current = np.log(np.clip(current, NUMERICAL_EPS, None))
    log_next = np.log(np.clip(next_all, NUMERICAL_EPS, None))
    log_initial = np.log(np.clip(initial_all, NUMERICAL_EPS, None))
    n_candidates = len(candidates)
    score_matrix = np.empty((n_candidates, n_candidates), dtype=float)
    for index_i in range(n_candidates):
        for index_j in range(n_candidates):
            ell_current = log_current[index_i] - log_current[index_j]
            ell_next = np.sum(
                target_next * (log_next[index_i] - log_next[index_j]), axis=1
            )
            ell_initial = np.sum(
                target_initial * (log_initial[index_i] - log_initial[index_j]),
                axis=1,
            )
            hellinger = np.mean(
                np.square(np.sqrt(current[index_i]) - np.sqrt(current[index_j]))
            )
            score_matrix[index_i, index_j] = (
                np.mean(current[index_i] * (ell_current - gamma * ell_next))
                - (1.0 - gamma) * float(np.sum(init_weight * ell_initial))
                - 0.25 * (1.0 - gamma) * hellinger
            )
    if not np.all(np.isfinite(score_matrix)):
        raise FloatingPointError("A-PBV produced a nonfinite score.")
    worst_case = np.max(score_matrix, axis=1)
    selected_index = int(np.flatnonzero(worst_case == np.min(worst_case))[0])
    return APBVSelectionResult(
        candidate_ids=tuple(candidate.candidate.candidate_id for candidate in candidates),
        score_matrix=score_matrix,
        worst_case_scores=worst_case,
        selected_index=selected_index,
    )


def fit_selected_fore_ratio(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    target_policy: ProbabilityPolicy,
    gamma: float,
    n_actions: int,
    candidates: Sequence[FORECandidateConfig],
    seed: int,
    options: FOREFitOptions = FOREFitOptions(),
    split: Optional[tuple[Array, Array, Array]] = None,
) -> SelectedFORERatio:
    """Select on an internal holdout and refit the ordinary FORE ratio."""
    states, actions, next_states = _validate_transition_arrays(
        states, actions, next_states, n_actions
    )
    if not candidates:
        raise ValueError("candidates must be nonempty.")
    split = deterministic_three_way_split(states.shape[0], seed) if split is None else split
    fit_idx, validation_idx, signed_validation_idx = split
    _validate_disjoint_split(split, states.shape[0])
    fitted = _fit_candidate_library(
        states=states[fit_idx],
        actions=actions[fit_idx],
        next_states=next_states[fit_idx],
        initial_states=states[fit_idx],
        initial_weights=None,
        target_policy=target_policy,
        gamma=gamma,
        n_actions=n_actions,
        candidates=candidates,
        seed=seed + 101,
        options=options,
    )
    selection = adversarial_pairwise_bellman_validation(
        fitted,
        states=states[validation_idx],
        actions=actions[validation_idx],
        next_states=next_states[validation_idx],
        initial_states=states[validation_idx],
        target_policy=target_policy,
        gamma=gamma,
    )
    selected_config = candidates[selection.selected_index]
    final_ratio = _fit_one_fore(
        states=states,
        actions=actions,
        next_states=next_states,
        initial_states=states,
        initial_weights=None,
        target_policy=target_policy,
        gamma=gamma,
        n_actions=n_actions,
        candidate=selected_config,
        seed=seed + 50_003,
        options=options,
    )
    return SelectedFORERatio(
        ratio=final_ratio,
        selection_ratio=fitted[selection.selected_index],
        selection=selection,
        split_sizes=(fit_idx.size, validation_idx.size, signed_validation_idx.size),
        candidate_fit_seconds=float(sum(candidate.fit_seconds for candidate in fitted)),
        total_fit_seconds=float(
            sum(candidate.fit_seconds for candidate in fitted) + final_ratio.fit_seconds
        ),
    )


def fit_selected_signed_fore_ratio(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    source_weights: Array,
    target_policy: ProbabilityPolicy,
    gamma: float,
    n_actions: int,
    candidates: Sequence[FORECandidateConfig],
    seed: int,
    split: tuple[Array, Array, Array],
    options: FOREFitOptions = FOREFitOptions(),
    mass_tolerance: float = 1e-10,
    refit_source_weights: Optional[Array] = None,
) -> SignedFORERatio:
    """Fit a signed advantage ratio as positive and negative FORE components."""
    states, actions, next_states = _validate_transition_arrays(
        states, actions, next_states, n_actions
    )
    weights = np.asarray(source_weights, dtype=float).reshape(-1)
    if weights.shape[0] != states.shape[0] or not np.all(np.isfinite(weights)):
        raise ValueError("source_weights must be finite with one value per transition.")
    refit_weights = (
        weights
        if refit_source_weights is None
        else np.asarray(refit_source_weights, dtype=float).reshape(-1)
    )
    if refit_weights.shape != weights.shape or not np.all(np.isfinite(refit_weights)):
        raise ValueError(
            "refit_source_weights must be finite with one value per transition."
        )
    _validate_disjoint_split(split, states.shape[0])
    fit_idx, _, validation_idx = split
    (
        positive,
        positive_selection,
        positive_mass,
        positive_candidate_seconds,
        positive_total_seconds,
    ) = _fit_signed_component(
        component_weights=np.maximum(weights, 0.0),
        final_component_weights=np.maximum(refit_weights, 0.0),
        component_name="positive",
        states=states,
        actions=actions,
        next_states=next_states,
        fit_idx=fit_idx,
        validation_idx=validation_idx,
        target_policy=target_policy,
        gamma=gamma,
        n_actions=n_actions,
        candidates=candidates,
        seed=seed + 1_009,
        options=options,
        mass_tolerance=mass_tolerance,
    )
    (
        negative,
        negative_selection,
        negative_mass,
        negative_candidate_seconds,
        negative_total_seconds,
    ) = _fit_signed_component(
        component_weights=np.maximum(-weights, 0.0),
        final_component_weights=np.maximum(-refit_weights, 0.0),
        component_name="negative",
        states=states,
        actions=actions,
        next_states=next_states,
        fit_idx=fit_idx,
        validation_idx=validation_idx,
        target_policy=target_policy,
        gamma=gamma,
        n_actions=n_actions,
        candidates=candidates,
        seed=seed + 2_009,
        options=options,
        mass_tolerance=mass_tolerance,
    )
    return SignedFORERatio(
        gamma=gamma,
        n_actions=n_actions,
        positive=positive,
        negative=negative,
        positive_mass=positive_mass,
        negative_mass=negative_mass,
        positive_selection=positive_selection,
        negative_selection=negative_selection,
        positive_candidate_fit_seconds=positive_candidate_seconds,
        negative_candidate_fit_seconds=negative_candidate_seconds,
        positive_total_fit_seconds=positive_total_seconds,
        negative_total_fit_seconds=negative_total_seconds,
    )


def ratio_diagnostics(
    ratio: FittedFORERatio,
    states: Array,
    actions: Array,
) -> dict[str, Any]:
    """Return deployable ratio-tail and optimization telemetry."""
    normalized = ratio.predict_normalized(states, actions)
    unnormalized = normalized / (1.0 - ratio.gamma)
    denominator = float(np.sum(np.square(unnormalized)))
    ess = (
        float(np.square(np.sum(unnormalized)) / denominator)
        if denominator > 0.0
        else 0.0
    )
    log_ratio = np.asarray(
        ratio.model.predict_state_action_log_ratio(
            states, one_hot_actions(actions, ratio.n_actions), clip=False
        ),
        dtype=float,
    )
    clip = float(getattr(ratio.model, "logit_clip", np.inf) or np.inf)
    return {
        "candidate_id": ratio.candidate.candidate_id,
        "selected_iterations": int(ratio.candidate.num_iterations),
        "fit_seconds": float(ratio.fit_seconds),
        "normalized_mass": float(np.mean(normalized)),
        "ess": ess,
        "ess_fraction": ess / max(1, normalized.size),
        "ratio_q99": float(np.quantile(unnormalized, 0.99)),
        "ratio_max": float(np.max(unnormalized)),
        "logit_cap_fraction": float(np.mean(np.abs(log_ratio) >= clip)),
        "nonfinite": bool(not np.all(np.isfinite(unnormalized))),
    }


def selection_manifest(
    candidates: Sequence[FORECandidateConfig],
    options: FOREFitOptions,
) -> dict[str, Any]:
    """Serialize the predeclared library without estimator truth."""
    return {
        "selector": "adversarial-pairwise-bellman-validation",
        "exact_validation_action_summation": True,
        "validation_empirical_terms": "same-held-out-rows",
        "candidates": [asdict(candidate) | {"candidate_id": candidate.candidate_id} for candidate in candidates],
        "fit_options": asdict(options),
        "uses_oracle_truth": False,
    }


def aggregate_pilot_selections(
    events: Sequence[tuple[Sequence[FORECandidateConfig], APBVSelectionResult]],
) -> dict[str, Any]:
    """Freeze hyperparameters from truth-blind A-PBV pilot events.

    Selection frequency is primary.  Ties use mean within-event rank, then the
    smaller network and stronger weight decay.  Iteration count is deliberately
    excluded because it remains a fold-specific early-stopping choice.
    """
    if not events:
        raise ValueError("At least one pilot event is required.")
    aggregate: dict[tuple[tuple[int, ...], float, float], dict[str, Any]] = {}
    event_rows: list[dict[str, Any]] = []
    for event_index, (candidate_configs, selection) in enumerate(events):
        configs = tuple(candidate_configs)
        if len(configs) != len(selection.candidate_ids):
            raise ValueError("Pilot candidates and selection scores have different lengths.")
        bases = [
            (tuple(config.hidden_dims), float(config.learning_rate), float(config.weight_decay))
            for config in configs
        ]
        unique_bases = list(dict.fromkeys(bases))
        base_scores = {
            base: min(
                float(selection.worst_case_scores[index])
                for index, candidate_base in enumerate(bases)
                if candidate_base == base
            )
            for base in unique_bases
        }
        ordered = sorted(
            unique_bases,
            key=lambda base: (
                base_scores[base],
                sum(base[0]),
                -base[2],
                base[1],
            ),
        )
        ranks = {base: rank + 1 for rank, base in enumerate(ordered)}
        selected_base = bases[selection.selected_index]
        for base in unique_bases:
            record = aggregate.setdefault(
                base,
                {
                    "hidden_dims": base[0],
                    "learning_rate": base[1],
                    "weight_decay": base[2],
                    "selection_count": 0,
                    "ranks": [],
                },
            )
            record["selection_count"] += int(base == selected_base)
            record["ranks"].append(ranks[base])
            event_rows.append(
                {
                    "event_index": event_index,
                    "hidden_dims": "x".join(str(width) for width in base[0]),
                    "learning_rate": base[1],
                    "weight_decay": base[2],
                    "best_apbv_over_budgets": base_scores[base],
                    "within_event_rank": ranks[base],
                    "selected_hyperparameters": bool(base == selected_base),
                }
            )
    summary_rows = []
    for record in aggregate.values():
        summary_rows.append(
            {
                "hidden_dims": tuple(record["hidden_dims"]),
                "learning_rate": float(record["learning_rate"]),
                "weight_decay": float(record["weight_decay"]),
                "selection_count": int(record["selection_count"]),
                "mean_within_event_rank": float(np.mean(record["ranks"])),
            }
        )
    winner = min(
        summary_rows,
        key=lambda row: (
            -row["selection_count"],
            row["mean_within_event_rank"],
            sum(row["hidden_dims"]),
            -row["weight_decay"],
            row["learning_rate"],
        ),
    )
    return {
        "uses_oracle_truth": False,
        "number_of_events": len(events),
        "selected": winner,
        "summary_rows": summary_rows,
        "event_rows": event_rows,
    }


def _fit_signed_component(
    *,
    component_weights: Array,
    final_component_weights: Array,
    component_name: str,
    states: Array,
    actions: Array,
    next_states: Array,
    fit_idx: Array,
    validation_idx: Array,
    target_policy: ProbabilityPolicy,
    gamma: float,
    n_actions: int,
    candidates: Sequence[FORECandidateConfig],
    seed: int,
    options: FOREFitOptions,
    mass_tolerance: float,
) -> tuple[
    Optional[FittedFORERatio],
    Optional[APBVSelectionResult],
    float,
    float,
    float,
]:
    full_mass = float(np.mean(final_component_weights))
    fit_mass = float(np.mean(component_weights[fit_idx]))
    if full_mass <= mass_tolerance or fit_mass <= mass_tolerance:
        return None, None, full_mass, 0.0, 0.0
    fitted = _fit_candidate_library(
        states=states[fit_idx],
        actions=actions[fit_idx],
        next_states=next_states[fit_idx],
        initial_states=next_states[fit_idx],
        initial_weights=component_weights[fit_idx],
        target_policy=target_policy,
        gamma=gamma,
        n_actions=n_actions,
        candidates=candidates,
        seed=seed,
        options=options,
    )
    validation_mass = float(np.sum(component_weights[validation_idx]))
    if validation_mass <= mass_tolerance:
        score_matrix = np.zeros((len(fitted), len(fitted)), dtype=float)
        selection = APBVSelectionResult(
            candidate_ids=tuple(candidate.candidate.candidate_id for candidate in fitted),
            score_matrix=score_matrix,
            worst_case_scores=np.zeros(len(fitted), dtype=float),
            selected_index=0,
        )
    else:
        selection = adversarial_pairwise_bellman_validation(
            fitted,
            states=states[validation_idx],
            actions=actions[validation_idx],
            next_states=next_states[validation_idx],
            initial_states=next_states[validation_idx],
            initial_weights=component_weights[validation_idx],
            target_policy=target_policy,
            gamma=gamma,
        )
    final_ratio = _fit_one_fore(
        states=states,
        actions=actions,
        next_states=next_states,
        initial_states=next_states,
        initial_weights=final_component_weights,
        target_policy=target_policy,
        gamma=gamma,
        n_actions=n_actions,
        candidate=candidates[selection.selected_index],
        seed=seed + 40_001,
        options=options,
    )
    candidate_seconds = float(sum(candidate.fit_seconds for candidate in fitted))
    del component_name
    return (
        final_ratio,
        selection,
        full_mass,
        candidate_seconds,
        candidate_seconds + final_ratio.fit_seconds,
    )


def _fit_candidate_library(**kwargs: Any) -> list[FittedFORERatio]:
    candidates = tuple(kwargs.pop("candidates"))
    base_seed = int(kwargs.pop("seed"))
    return [
        _fit_one_fore(candidate=candidate, seed=base_seed, **kwargs)
        for candidate in candidates
    ]


def _fit_one_fore(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    initial_states: Array,
    initial_weights: Optional[Array],
    target_policy: ProbabilityPolicy,
    gamma: float,
    n_actions: int,
    candidate: FORECandidateConfig,
    seed: int,
    options: FOREFitOptions,
) -> FittedFORERatio:
    try:
        from occupancy_ratio import KLFORIConfig, fit_kl_fori_neural
    except ImportError as exc:  # pragma: no cover - exercised in dependency preflight
        raise RuntimeError(
            "Neural FORE requires the local occupancy-ratio package and Torch."
        ) from exc
    target_next_actions = _sample_action_features(
        target_policy,
        next_states,
        n_actions=n_actions,
        n_draws=options.target_action_draws,
        seed=seed + 17,
    )
    initial_actions = _sample_action_features(
        target_policy,
        initial_states,
        n_actions=n_actions,
        n_draws=options.target_action_draws,
        seed=seed + 29,
    )
    config = KLFORIConfig(
        backend="neural",
        num_iterations=candidate.num_iterations,
        optimizer_steps=options.optimizer_steps,
        learning_rate=candidate.learning_rate,
        validation_fraction=0.0,
        early_stopping=False,
        logit_clip=options.logit_clip,
        seed=seed,
        neural_hidden_dims=candidate.hidden_dims,
        neural_log_partition_mode="variational",
        neural_batch_size=options.batch_size,
        neural_variational_gauge_fix="auto",
        neural_weight_decay=candidate.weight_decay,
        neural_grad_clip_norm=options.grad_clip_norm,
        device=options.device,
        show_progress=False,
    )
    started = time.perf_counter()
    model = fit_kl_fori_neural(
        states=states,
        actions=one_hot_actions(actions, n_actions),
        next_states=next_states,
        target_next_actions=target_next_actions,
        gamma=gamma,
        initial_states=initial_states,
        initial_actions=initial_actions,
        initial_weights=initial_weights,
        config=config,
    )
    return FittedFORERatio(
        model=model,
        gamma=float(gamma),
        n_actions=int(n_actions),
        candidate=candidate,
        fit_seconds=time.perf_counter() - started,
    )


def _sample_action_features(
    policy: ProbabilityPolicy,
    states: Array,
    *,
    n_actions: int,
    n_draws: int,
    seed: int,
) -> Array:
    states = _as_states(states, "states")
    probabilities = _validate_probabilities(policy.predict_proba(states), n_actions)
    rng = np.random.default_rng(seed)
    uniforms = rng.random((states.shape[0], n_draws))
    cumulative = np.cumsum(probabilities, axis=1)
    sampled = np.sum(uniforms[:, :, None] > cumulative[:, None, :], axis=2)
    sampled = np.clip(sampled, 0, n_actions - 1)
    return np.eye(n_actions, dtype=float)[sampled]


def _validate_transition_arrays(
    states: Array,
    actions: Array,
    next_states: Array,
    n_actions: int,
) -> tuple[Array, Array, Array]:
    states = _as_states(states, "states")
    next_states = _as_states(next_states, "next_states")
    if states.shape != next_states.shape:
        raise ValueError("states and next_states must have identical shapes.")
    actions = _as_integer_actions(actions, states.shape[0], n_actions)
    return states, actions, next_states


def _as_states(values: Array, name: str) -> Array:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _as_integer_actions(values: Array, n_rows: int, n_actions: int) -> Array:
    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    values = values.reshape(-1)
    if values.shape[0] != n_rows:
        raise ValueError("actions must have one value per state.")
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError("actions must contain finite integers.")
    actions = values.astype(int)
    if np.any(actions < 0) or np.any(actions >= n_actions):
        raise ValueError("actions lie outside the configured action space.")
    return actions


def _validate_probabilities(values: Array, n_actions: int) -> Array:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != n_actions:
        raise ValueError("policy probabilities have the wrong shape.")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("policy probabilities must be finite and nonnegative.")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("policy probability rows must have positive mass.")
    return probabilities / row_sums


def _normalized_weights(values: Optional[Array], n_rows: int) -> Array:
    if values is None:
        return np.full(n_rows, 1.0 / n_rows, dtype=float)
    weights = np.asarray(values, dtype=float).reshape(-1)
    if weights.shape[0] != n_rows:
        raise ValueError("initial_weights must match initial_states.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("initial_weights must be finite and nonnegative.")
    mass = float(np.sum(weights))
    if mass <= 0.0:
        raise ValueError("initial_weights must have positive mass.")
    return weights / mass


def _validate_finite_nonnegative(values: Array, name: str) -> None:
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise FloatingPointError(f"{name} must be finite and nonnegative.")


def _validate_disjoint_split(split: tuple[Array, Array, Array], n_rows: int) -> None:
    if len(split) != 3:
        raise ValueError("split must contain exactly three index arrays.")
    arrays = [np.asarray(indices, dtype=int).reshape(-1) for indices in split]
    if any(indices.size == 0 for indices in arrays):
        raise ValueError("all split parts must be nonempty.")
    combined = np.concatenate(arrays)
    if np.any(combined < 0) or np.any(combined >= n_rows):
        raise ValueError("split indices are out of range.")
    if np.unique(combined).size != combined.size:
        raise ValueError("split parts must be disjoint.")
    if np.unique(combined).size != n_rows:
        raise ValueError("split parts must cover every row exactly once.")


__all__ = [
    "APBVSelectionResult",
    "FORECandidateConfig",
    "FOREFitOptions",
    "FittedFORERatio",
    "SelectedFORERatio",
    "SignedFORERatio",
    "adversarial_pairwise_bellman_validation",
    "aggregate_pilot_selections",
    "deterministic_three_way_split",
    "fit_selected_fore_ratio",
    "fit_selected_signed_fore_ratio",
    "one_hot_actions",
    "paper_early_stopping_candidates",
    "paper_pilot_candidates",
    "ratio_diagnostics",
    "selection_manifest",
]
