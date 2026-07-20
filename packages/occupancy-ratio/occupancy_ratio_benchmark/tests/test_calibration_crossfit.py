from __future__ import annotations

from dataclasses import replace
import pickle
import sys
from types import ModuleType

import numpy as np
import pytest

from occupancy_ratio_benchmark.calibration_crossfit import (
    CrossCalibrationConfig,
    CrossCalibrationSample,
    MaterialNegativePredictionError,
    PooledCalibrationInput,
    fit_cross_calibrated_ensemble,
    fit_cross_calibrated_matrices,
    fit_normalized_pava_calibrator,
    make_grouped_fold_assignment,
)


class _FoldPredictor:
    def __init__(self, fold: int) -> None:
        self.fold = int(fold)

    def predict_state_action_ratio(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        *,
        clip: bool = True,
    ) -> np.ndarray:
        del actions
        assert clip is False
        score = np.asarray(states, dtype=np.float64).reshape(len(states), -1)[:, 0] + self.fold + 1.0
        return score


class _AffineCalibration:
    def __init__(self, slope: float = 0.25, intercept: float = 1.0) -> None:
        self.slope = float(slope)
        self.intercept = float(intercept)

    def predict(self, score: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(score, dtype=np.float64)


class _TinyNegativePredictor:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict_state_action_ratio(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        *,
        clip: bool = True,
    ) -> np.ndarray:
        del actions
        assert clip is False
        values = np.ones(len(states), dtype=np.float64)
        first_state = np.asarray(states, dtype=np.float64).reshape(len(states), -1)[:, 0]
        values[np.isclose(first_state, 0.0)] = self.value
        return values


class _RecordingCalibrationFitter:
    def __init__(self) -> None:
        self.calls: list[PooledCalibrationInput] = []

    def __call__(
        self,
        pooled: PooledCalibrationInput,
        config: CrossCalibrationConfig,
    ) -> _AffineCalibration:
        del config
        self.calls.append(pooled)
        return _AffineCalibration()


def _sample() -> CrossCalibrationSample:
    n = 18
    states = np.arange(n, dtype=np.float64).reshape(-1, 1)
    groups = np.repeat(np.arange(9), 2)
    initial_states = (0.5 + np.arange(9, dtype=np.float64)).reshape(-1, 1)
    return CrossCalibrationSample(
        states=states,
        actions=np.zeros((n, 1), dtype=np.float64),
        next_states=states + 0.25,
        next_target_actions=np.ones((n, 1), dtype=np.float64),
        initial_states=initial_states,
        initial_actions=np.ones((9, 1), dtype=np.float64),
        source_groups=groups,
        initial_groups=np.arange(9),
        gamma=0.95,
        source_weights=np.linspace(1.0, 2.0, n),
        initial_weights=np.linspace(2.0, 1.0, 9),
    )


def test_group_assignment_keeps_episodes_intact_and_aligns_initial_groups() -> None:
    sample = _sample()
    assignment = make_grouped_fold_assignment(
        sample.source_groups,
        sample.initial_groups,
        num_folds=3,
        seed=17,
    )
    for group in np.unique(sample.source_groups):
        source_folds = np.unique(assignment.source_fold_ids[np.asarray(sample.source_groups) == group])
        assert source_folds.size == 1
        initial_fold = assignment.initial_fold_ids[np.asarray(sample.initial_groups) == group]
        assert initial_fold.size == 1
        assert int(initial_fold[0]) == int(source_folds[0])
    assert set(assignment.source_fold_ids) == {0, 1, 2}


def test_confirmatory_pava_defaults_are_frozen() -> None:
    config = CrossCalibrationConfig()
    assert config.num_folds == 10
    assert config.pava_num_iterations == 3_000
    assert config.pava_tolerance == 1e-8
    assert config.pava_direction == "increasing"
    assert config.pava_fixed_point_damping == 1.0
    assert config.pava_support_policy == "constant_extrapolation"


def test_one_pooled_calibrator_then_pointwise_fold_median() -> None:
    sample = _sample()
    requests = []
    fitter = _RecordingCalibrationFitter()

    def fit_fold(request):
        requests.append(request)
        source_groups = np.asarray(sample.source_groups)
        initial_groups = np.asarray(sample.initial_groups)
        train_source_groups = set(source_groups[request.train_source_indices])
        oof_source_groups = set(source_groups[request.oof_source_indices])
        train_initial_groups = set(initial_groups[request.train_initial_indices])
        oof_initial_groups = set(initial_groups[request.oof_initial_indices])
        assert train_source_groups.isdisjoint(oof_source_groups)
        assert train_initial_groups.isdisjoint(oof_initial_groups)
        assert train_source_groups.isdisjoint(oof_initial_groups)
        assert train_initial_groups.isdisjoint(oof_source_groups)
        return _FoldPredictor(request.fold_index)

    ensemble = fit_cross_calibrated_ensemble(
        sample,
        fit_fold,
        config=CrossCalibrationConfig(num_folds=3, seed=17),
        calibrator_fitter=fitter,
    )

    assert len(requests) == 3
    assert len(fitter.calls) == 1
    pooled = fitter.calls[0]
    expected_oof_score = np.asarray(sample.states).reshape(-1) + ensemble.assignment.source_fold_ids + 1.0
    expected_next_score = np.asarray(sample.next_states).reshape(-1) + ensemble.assignment.source_fold_ids + 1.0
    expected_initial_score = np.asarray(sample.initial_states).reshape(-1) + ensemble.assignment.initial_fold_ids + 1.0
    assert np.allclose(pooled.source_score, expected_oof_score)
    assert np.allclose(pooled.next_score, expected_next_score)
    assert np.allclose(pooled.initial_score, expected_initial_score)
    assert np.isclose(np.dot(pooled.source_weights, pooled.scalar_source_weight), 1.0)

    query_states = np.array([[2.0], [10.0]])
    query_actions = np.zeros((2, 1))
    predictions = ensemble.predict(query_states, query_actions)
    fold_score = np.stack(
        [query_states[:, 0] + fold + 1.0 for fold in range(3)],
        axis=0,
    )
    expected_raw_by_fold = fold_score
    expected_scalar_by_fold = ensemble.pooled_oof.scalar_scale * expected_raw_by_fold
    expected_pava_by_fold = 1.0 + 0.25 * fold_score
    assert np.allclose(predictions.raw_by_fold, expected_raw_by_fold)
    assert np.allclose(predictions.scalar_by_fold, expected_scalar_by_fold)
    assert np.allclose(predictions.pava_by_fold, expected_pava_by_fold)
    assert np.allclose(predictions.raw, np.median(expected_raw_by_fold, axis=0))
    assert np.allclose(predictions.scalar, np.median(expected_scalar_by_fold, axis=0))
    assert np.allclose(predictions.pava, np.median(expected_pava_by_fold, axis=0))
    assert ensemble.diagnostics["pooled_calibrator_count"] == 1
    assert ensemble.diagnostics["aggregation"] == "pointwise_median"
    assert ensemble.diagnostics["occupancy_estimand"] == "normalized_discounted"
    assert ensemble.diagnostics["uses_coverage_stopping"] is False


def test_matrix_path_selects_oof_diagonal_and_uses_one_common_map() -> None:
    sample = _sample()
    config = CrossCalibrationConfig(num_folds=3, seed=17)
    assignment = make_grouped_fold_assignment(
        sample.source_groups,
        sample.initial_groups,
        num_folds=3,
        seed=17,
    )
    source_base = np.asarray(sample.states).reshape(-1)
    next_base = np.asarray(sample.next_states).reshape(-1)
    initial_base = np.asarray(sample.initial_states).reshape(-1)
    source_matrix = np.stack([source_base + fold + 1.0 for fold in range(3)])
    next_matrix = np.stack([next_base + fold + 1.0 for fold in range(3)])
    initial_matrix = np.stack([initial_base + fold + 1.0 for fold in range(3)])
    fitter = _RecordingCalibrationFitter()
    result = fit_cross_calibrated_matrices(
        source_q_by_fold=source_matrix,
        next_q_by_fold=next_matrix,
        initial_q_by_fold=initial_matrix,
        assignment=assignment,
        gamma=sample.gamma,
        source_weights=sample.source_weights,
        initial_weights=sample.initial_weights,
        config=config,
        calibrator_fitter=fitter,
    )

    assert len(fitter.calls) == 1
    pooled = fitter.calls[0]
    source_oof = source_matrix[assignment.source_fold_ids, np.arange(sample.n_source)]
    next_oof = next_matrix[assignment.source_fold_ids, np.arange(sample.n_source)]
    initial_oof = initial_matrix[assignment.initial_fold_ids, np.arange(sample.n_initial)]
    assert np.allclose(pooled.source_score, source_oof)
    assert np.allclose(pooled.next_score, next_oof)
    assert np.allclose(pooled.initial_score, initial_oof)
    assert np.allclose(result.source.raw, np.median(source_matrix, axis=0))
    assert np.allclose(result.next.raw, np.median(next_matrix, axis=0))
    assert np.allclose(result.initial.raw, np.median(initial_matrix, axis=0))
    assert np.allclose(result.source.scalar_by_fold, result.pooled_oof.scalar_scale * source_matrix)
    assert np.allclose(result.source.pava_by_fold, 1.0 + 0.25 * source_matrix)
    assert np.allclose(result.next.pava_by_fold, 1.0 + 0.25 * next_matrix)
    assert np.allclose(result.initial.pava_by_fold, 1.0 + 0.25 * initial_matrix)
    assert result.diagnostics["checkpoint_payload"] == "prediction_matrices"
    assert result.diagnostics["pooled_calibrator_count"] == 1


def test_matrix_path_rejects_material_negative_before_calibration() -> None:
    sample = _sample()
    assignment = make_grouped_fold_assignment(
        sample.source_groups,
        sample.initial_groups,
        num_folds=3,
        seed=17,
    )
    source_matrix = np.ones((3, sample.n_source))
    source_matrix[1, 4] = -1e-3
    fitter = _RecordingCalibrationFitter()
    with pytest.raises(MaterialNegativePredictionError) as caught:
        fit_cross_calibrated_matrices(
            source_q_by_fold=source_matrix,
            next_q_by_fold=np.ones((3, sample.n_source)),
            initial_q_by_fold=np.ones((3, sample.n_initial)),
            assignment=assignment,
            gamma=sample.gamma,
            config=CrossCalibrationConfig(num_folds=3, seed=17),
            calibrator_fitter=fitter,
        )
    assert caught.value.fold_index == 1
    assert caught.value.role == "source_q_by_fold"
    assert fitter.calls == []


def test_completed_fold_artifacts_resume_without_refitting() -> None:
    sample = _sample()
    completed = []
    first = fit_cross_calibrated_ensemble(
        sample,
        lambda request: _FoldPredictor(request.fold_index),
        config=CrossCalibrationConfig(num_folds=3, seed=9),
        calibrator_fitter=_RecordingCalibrationFitter(),
        on_fold_complete=completed.append,
    )
    assert len(completed) == 3

    def fail_if_called(request):
        raise AssertionError(f"fold {request.fold_index} should have resumed")

    resumed = fit_cross_calibrated_ensemble(
        sample,
        fail_if_called,
        config=CrossCalibrationConfig(num_folds=3, seed=9),
        calibrator_fitter=_RecordingCalibrationFitter(),
        completed_folds=completed,
    )
    query_states = np.array([[1.5], [4.5]])
    query_actions = np.zeros_like(query_states)
    assert np.allclose(
        first.predict(query_states, query_actions).pava, resumed.predict(query_states, query_actions).pava
    )
    pickle.loads(pickle.dumps(resumed))


def test_completed_artifact_assignment_mismatch_is_rejected() -> None:
    sample = _sample()
    artifacts = []
    fit_cross_calibrated_ensemble(
        sample,
        lambda request: _FoldPredictor(request.fold_index),
        config=CrossCalibrationConfig(num_folds=3, seed=3),
        calibrator_fitter=_RecordingCalibrationFitter(),
        on_fold_complete=artifacts.append,
    )
    bad = replace(artifacts[0], fit_seed=artifacts[0].fit_seed + 1)
    with pytest.raises(ValueError, match="seed"):
        fit_cross_calibrated_ensemble(
            sample,
            lambda request: _FoldPredictor(request.fold_index),
            config=CrossCalibrationConfig(num_folds=3, seed=3),
            calibrator_fitter=_RecordingCalibrationFitter(),
            completed_folds=[bad],
        )


def test_tiny_negatives_are_projected_and_reported() -> None:
    sample = _sample()
    ensemble = fit_cross_calibrated_ensemble(
        sample,
        lambda request: _TinyNegativePredictor(-5e-11),
        config=CrossCalibrationConfig(num_folds=3, seed=3, negative_tolerance=1e-10),
        calibrator_fitter=_RecordingCalibrationFitter(),
    )
    assert ensemble.pooled_oof.source_q[0] == 0.0
    assert ensemble.diagnostics["tiny_negative_projection_count"] == 1
    assert np.isclose(ensemble.diagnostics["tiny_negative_projection_mass"], 5e-11)
    query = ensemble.predict(np.array([[0.0]]), np.array([[0.0]]))
    assert query.raw[0] == 0.0
    assert np.array_equal(query.tiny_negative_count_by_fold, np.ones(3, dtype=np.int64))


def test_material_negative_is_a_structured_fold_failure() -> None:
    sample = _sample()
    with pytest.raises(MaterialNegativePredictionError) as caught:
        fit_cross_calibrated_ensemble(
            sample,
            lambda request: _TinyNegativePredictor(-1e-3),
            config=CrossCalibrationConfig(num_folds=3, seed=3, negative_tolerance=1e-10),
            calibrator_fitter=_RecordingCalibrationFitter(),
        )
    assert caught.value.fold_index in {0, 1, 2}
    assert caught.value.role == "source_q"
    assert caught.value.negative_count == 1
    assert caught.value.minimum == -1e-3


def test_lazy_pava_adapter_forces_normalized_estimand(monkeypatch) -> None:
    captured = {}
    fake_module = ModuleType("occupancy_ratio.isotonic_calibration")

    class FakeConfig:
        def __init__(
            self,
            *,
            num_iterations,
            tolerance,
            direction,
            positivity_floor,
            normalize,
            estimand,
            fixed_point_damping,
            initialization,
            support_policy,
        ) -> None:
            captured["config"] = {
                "num_iterations": num_iterations,
                "tolerance": tolerance,
                "direction": direction,
                "positivity_floor": positivity_floor,
                "normalize": normalize,
                "estimand": estimand,
                "fixed_point_damping": fixed_point_damping,
                "initialization": initialization,
                "support_policy": support_policy,
            }

    def fake_fit(**kwargs):
        captured["fit"] = kwargs
        return _AffineCalibration()

    fake_module.IsotonicCalibrationConfig = FakeConfig
    fake_module.fit_isotonic_fori_pava = fake_fit
    monkeypatch.setitem(sys.modules, "occupancy_ratio.isotonic_calibration", fake_module)
    pooled = PooledCalibrationInput(
        source_score=np.array([1.0, 2.0]),
        next_score=np.array([1.5, 2.5]),
        initial_score=np.array([1.25, 2.25]),
        gamma=0.9,
        source_weights=np.array([0.5, 0.5]),
        initial_weights=np.array([0.5, 0.5]),
        scalar_source_weight=np.array([2.0 / 3.0, 4.0 / 3.0]),
    )
    result = fit_normalized_pava_calibrator(
        pooled,
        CrossCalibrationConfig(num_folds=2),
    )
    assert isinstance(result, _AffineCalibration)
    assert captured["config"]["normalize"] is True
    assert captured["config"]["estimand"] == "normalized_discounted"
    assert captured["fit"]["initial_omega"] is pooled.scalar_source_weight
    assert "next_retention" not in captured["fit"]
    assert "initial_retention" not in captured["fit"]


def test_too_few_source_groups_is_rejected() -> None:
    with pytest.raises(ValueError, match="source groups"):
        make_grouped_fold_assignment(
            np.array([0, 0, 1, 1]),
            np.array([0, 1]),
            num_folds=3,
            seed=0,
        )
