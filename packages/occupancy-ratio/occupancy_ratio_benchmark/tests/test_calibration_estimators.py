from __future__ import annotations

import numpy as np
import pytest

import occupancy_ratio_benchmark.calibration_estimators as adapters
from occupancy_ratio_benchmark.discrete import make_discrete_dataset


class _PositiveModel:
    diagnostics = {"fit": "stub"}

    def predict_state_action_ratio(self, states, actions, *, clip=True):
        assert clip is False
        return 1.0 + 0.01 * np.sum(states, axis=1) + 0.02 * np.sum(actions, axis=1)

    def predict_state_action_log_ratio(self, states, actions, *, clip=True):
        return np.log(self.predict_state_action_ratio(states, actions, clip=clip))


@pytest.mark.parametrize(
    ("estimator", "fit_name"),
    [
        ("neural_fori", "fit_kl_fori_neural"),
        ("google_dualdice", "fit_google_dualdice_occupancy_ratio"),
        ("scope_mwl", "fit_minimax_weight"),
        ("bestdice", "fit_minimax_weight"),
    ],
)
def test_fold_adapter_fits_once_and_predicts_all_rows(monkeypatch, estimator, fit_name) -> None:
    dataset = make_discrete_dataset(
        setting="random_tabular_mdp",
        gamma=0.9,
        sample_size=40,
        seed=3,
        n_states=8,
        n_actions=3,
        policy_shift=0.35,
    )
    calls = []

    def fit_stub(**kwargs):
        calls.append(kwargs)
        return _PositiveModel()

    monkeypatch.setattr(adapters, fit_name, fit_stub)
    result = adapters.fit_fold_predictions(
        estimator_id=estimator,
        dataset=dataset,
        train_source_indices=np.arange(30),
        train_initial_indices=np.arange(100),
        fold_index=1,
        fit_seed=10_004,
        registry_entry={
            "schedule": {
                "updates": 2,
                "outer_iterations": 1,
                "variational_steps": 1,
                "prediction_clamp": adapters.NEURAL_FORI_PREDICTION_CLAMP,
            }
        },
    )

    assert len(calls) == 1
    assert result.source_q.shape == (dataset.n,)
    assert result.next_q.shape == (dataset.n,)
    assert result.initial_q.shape == (dataset.initial_states.shape[0],)
    assert np.all(result.source_q > 0.0)
    assert result.diagnostics["base_upper_cap_enabled"] is (estimator == "neural_fori")
    assert result.diagnostics["base_query_normalization_enabled"] is False
    if estimator == "neural_fori":
        assert calls[0]["config"].logit_clip is None
    elif estimator == "google_dualdice":
        assert calls[0]["config"].prediction_max is None
        assert calls[0]["config"].normalize_predictions is False
    elif estimator == "scope_mwl":
        scope = calls[0]["config"].scope_rl
        assert scope.standardize_inputs is True
        assert scope.bandwidth_selection == "median"
        assert calls[0]["step_per_trajectory"] == 1
    else:
        dice = calls[0]["config"].google_dice_rl
        assert dice.prediction_max is None
        assert dice.normalize_predictions is False


def test_signed_critic_is_projected_once_to_ratio_cone(monkeypatch) -> None:
    dataset = make_discrete_dataset(
        setting="random_tabular_mdp",
        gamma=0.9,
        sample_size=20,
        seed=4,
        n_states=6,
        n_actions=2,
    )

    class NegativeModel(_PositiveModel):
        def predict_state_action_ratio(self, states, actions, *, clip=True):
            values = super().predict_state_action_ratio(states, actions, clip=clip)
            values[0] = -0.5
            return values

    monkeypatch.setattr(
        adapters,
        "fit_google_dualdice_occupancy_ratio",
        lambda **kwargs: NegativeModel(),
    )
    result = adapters.fit_fold_predictions(
        estimator_id="google_dualdice",
        dataset=dataset,
        train_source_indices=np.arange(15),
        train_initial_indices=np.arange(100),
        fold_index=0,
        fit_seed=0,
        registry_entry={"schedule": {"updates": 1}},
    )
    assert result.source_q[0] == 0.0
    assert result.diagnostics["material_negative_projection_count"] >= 1
    assert result.diagnostics["negative_projection_mass"] > 0.0


def test_neural_fori_clamps_extrapolated_log_scores_to_current_support(monkeypatch) -> None:
    dataset = make_discrete_dataset(
        setting="random_tabular_mdp",
        gamma=0.9,
        sample_size=20,
        seed=4,
        n_states=6,
        n_actions=2,
    )

    class OverflowModel(_PositiveModel):
        def predict_state_action_log_ratio(self, states, actions, *, clip=True):
            values = np.zeros(np.asarray(states).shape[0], dtype=np.float64)
            if values.size == dataset.n:
                values[0] = np.inf
                values[1] = -np.inf
            return values

    monkeypatch.setattr(adapters, "fit_kl_fori_neural", lambda **kwargs: OverflowModel())
    result = adapters.fit_fold_predictions(
        estimator_id="neural_fori",
        dataset=dataset,
        train_source_indices=np.arange(15),
        train_initial_indices=np.arange(80),
        calibration_source_indices=np.arange(15, 20),
        fold_index=0,
        fit_seed=0,
        registry_entry={
            "schedule": {
                "outer_iterations": 1,
                "variational_steps": 1,
                "prediction_clamp": adapters.NEURAL_FORI_PREDICTION_CLAMP,
            }
        },
    )
    assert np.all(np.isfinite(result.source_q))
    assert np.all(np.isfinite(result.next_q))
    np.testing.assert_array_equal(result.source_q[15:20], np.ones(5))
    assert result.diagnostics["base_prediction_upper_clamp_count"] >= 1
    assert result.diagnostics["base_prediction_lower_clamp_count"] >= 1
    assert result.diagnostics["base_prediction_clamp_reference"] == (
        "held_out_current_calibration"
    )


def test_neural_fori_audit_scores_reuse_training_held_out_bounds(monkeypatch) -> None:
    dataset = make_discrete_dataset(
        setting="random_tabular_mdp",
        gamma=0.9,
        sample_size=30,
        seed=7,
        n_states=6,
        n_actions=2,
    )
    audit = make_discrete_dataset(
        setting="random_tabular_mdp",
        gamma=0.9,
        sample_size=17,
        seed=7,
        sample_seed=88,
        n_states=6,
        n_actions=2,
    )
    audit.states = np.full_like(audit.states, 100.0)
    audit.next_states = np.full_like(audit.next_states, -100.0)
    audit.initial_states = np.full_like(audit.initial_states, 50.0)

    class StateScoreModel(_PositiveModel):
        def predict_state_action_log_ratio(self, states, actions, *, clip=True):
            del actions
            assert clip is False
            return np.asarray(states, dtype=np.float64)[:, 0]

    monkeypatch.setattr(
        adapters,
        "fit_kl_fori_neural",
        lambda **kwargs: StateScoreModel(),
    )
    held_out = np.arange(20, 30)
    result = adapters.fit_fold_predictions(
        estimator_id="neural_fori",
        dataset=dataset,
        train_source_indices=np.arange(20),
        train_initial_indices=np.arange(80),
        calibration_source_indices=held_out,
        fold_index=0,
        fit_seed=0,
        registry_entry={
            "schedule": {
                "outer_iterations": 1,
                "variational_steps": 1,
                "prediction_clamp": adapters.NEURAL_FORI_PREDICTION_CLAMP,
            }
        },
        audit_dataset=audit,
    )

    expected_lower = float(np.min(dataset.states[held_out, 0]))
    expected_upper = float(np.max(dataset.states[held_out, 0]))
    np.testing.assert_allclose(result.audit_source_log_score, expected_upper)
    np.testing.assert_allclose(result.audit_next_log_score, expected_lower)
    np.testing.assert_allclose(result.audit_initial_log_score, expected_upper)
    assert result.diagnostics["independent_audit_scored"] is True
    assert result.diagnostics["independent_audit_used_for_fit"] is False
    assert result.diagnostics["independent_audit_used_for_clamp"] is False


def test_log_bounds_keep_full_representable_range() -> None:
    class ScoreModel:
        def predict_state_action_log_ratio(self, states, actions, *, clip=True):
            assert clip is False
            return np.asarray(states, dtype=np.float64)[:, 0]

    bounds, diagnostics = adapters._finite_log_bounds(
        ScoreModel(),
        np.asarray([[-100.0], [0.0], [1.0], [2.0], [100.0]]),
        np.zeros((5, 1)),
        chunk_size=2,
    )
    assert bounds == (-100.0, 100.0)
    assert diagnostics["calibration_rows"] == 5
    assert diagnostics["representable_distinct_scores"] == 5


def test_log_bounds_fall_back_to_finite_scores_below_exp_range() -> None:
    class ScoreModel:
        def predict_state_action_log_ratio(self, states, actions, *, clip=True):
            assert clip is False
            return np.asarray(states, dtype=np.float64)[:, 0]

    bounds, diagnostics = adapters._finite_log_bounds(
        ScoreModel(),
        np.asarray([[-2_000.0], [-1_500.0], [-1_000.0]]),
        np.zeros((3, 1)),
        chunk_size=2,
    )
    assert bounds == (-2_000.0, -1_000.0)
    assert diagnostics["representable_positive_rows"] == 0
    assert diagnostics["log_score_fallback_used"] is True


def test_worker_thread_limits_are_forced(monkeypatch) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "8")
    diagnostics = adapters.configure_calibration_worker_threads("neural_fori")
    assert set(diagnostics["environment"].values()) == {"1"}
    assert diagnostics["torch_threads_applied"] is True
