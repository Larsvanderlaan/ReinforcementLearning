"""Tests for the structural-support external-test stopped-FORE benchmark."""

from __future__ import annotations

import json

import numpy as np
import pytest

from occupancy_ratio_benchmark._stopped_fore_data import (
    StoppedFOREExternalConfig,
    make_structural_stopped_dataset,
)
from occupancy_ratio_benchmark._stopped_fore_execution import run_external_test_cell
from occupancy_ratio_benchmark.stopped_fore_external import (
    merge_benchmark,
    run_benchmark,
)


def test_structural_truth_has_genuine_initial_and_successor_stopping() -> None:
    initial = make_structural_stopped_dataset(
        n=200_000,
        gamma=0.9,
        contexts=8,
        behavior_probability=0.25,
        support_fraction=0.5,
        failure_mode="initial",
        seed=1,
    )
    successor = make_structural_stopped_dataset(
        n=200_000,
        gamma=0.9,
        contexts=8,
        behavior_probability=0.25,
        support_fraction=0.5,
        failure_mode="successor",
        seed=2,
    )

    assert initial.truth.retained_mass == pytest.approx(0.5)
    assert successor.truth.retained_mass == pytest.approx(0.55)
    assert initial.truth.max_positive_ratio == pytest.approx(4.0)
    assert successor.truth.max_positive_ratio == pytest.approx(4.0)
    assert np.any(initial.truth.initial_behavior_probability == 0.0)
    assert np.any(successor.truth.hub_behavior_probability == 0.0)
    assert np.mean(initial.stopped_ratio) == pytest.approx(0.5, abs=0.01)
    assert np.mean(successor.stopped_ratio) == pytest.approx(0.55, abs=0.01)


def test_upper_envelope_must_be_inactive_on_positive_truth() -> None:
    with pytest.raises(ValueError, match="upper clipping is inactive"):
        StoppedFOREExternalConfig(
            behavior_probability=0.25,
            tau_upper=4.0,
        )


def test_train_and_external_test_share_truth_but_not_rows() -> None:
    kwargs = {
        "n": 1_000,
        "gamma": 0.95,
        "contexts": 8,
        "behavior_probability": 0.25,
        "support_fraction": 0.5,
        "failure_mode": "successor",
    }
    train = make_structural_stopped_dataset(**kwargs, seed=10)
    test = make_structural_stopped_dataset(**kwargs, seed=20)

    assert train.truth.retained_mass == test.truth.retained_mass
    assert np.array_equal(
        train.truth.hub_behavior_probability,
        test.truth.hub_behavior_probability,
    )
    assert not np.array_equal(train.context_id, test.context_id)
    assert not np.array_equal(train.action_id, test.action_id)


def test_fit_contract_excludes_rewards_and_oracle_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    import occupancy_ratio_benchmark._stopped_fore_execution as execution

    seen: dict[str, object] = {}

    class FakeModel:
        tau_upper = 20.0
        diagnostics = {"converged": True}

        def predict_state_action_ratio(self, states, actions):
            del actions
            return np.ones(np.asarray(states).shape[0])

        def predict_gate_indicator(self, states, actions):
            del actions
            return np.ones(np.asarray(states).shape[0])

    def fake_fit(**kwargs):
        seen.update(kwargs)
        return FakeModel()

    monkeypatch.setattr(execution, "fit_clipped_kl_fori", fake_fit)
    config = StoppedFOREExternalConfig(
        n_train=100,
        n_test=200,
        repetitions=1,
        contexts=4,
        support_fractions=(0.5,),
        failure_modes=("initial",),
        backends=("linear",),
        methods=("stopped_fori_learned_gate",),
        stopped_num_iterations=1,
        stopped_gate_steps=1,
        stopped_ratio_steps=1,
    )
    train = make_structural_stopped_dataset(
        n=config.n_train,
        gamma=config.gamma,
        contexts=config.contexts,
        behavior_probability=config.behavior_probability,
        support_fraction=0.5,
        failure_mode="initial",
        seed=1,
    )
    test = make_structural_stopped_dataset(
        n=config.n_test,
        gamma=config.gamma,
        contexts=config.contexts,
        behavior_probability=config.behavior_probability,
        support_fraction=0.5,
        failure_mode="initial",
        seed=2,
    )

    rows, _ = run_external_test_cell(
        train,
        test,
        config,
        backend="linear",
        repetition=0,
        train_seed=1,
        test_seed=2,
    )

    assert "rewards" not in seen
    assert "stopped_ratio" not in seen
    assert rows[0]["oracle_used_for_fitting_or_selection"] is False
    assert rows[0]["calibration_method"] == "none"
    assert rows[0]["crossfit_folds"] == 0


def test_end_to_end_smoke_writes_external_artifacts(tmp_path) -> None:
    config = StoppedFOREExternalConfig(
        n_train=120,
        n_test=240,
        repetitions=1,
        contexts=4,
        support_fractions=(0.5,),
        failure_modes=("initial",),
        backends=("linear",),
        stopped_num_iterations=1,
        stopped_gate_steps=2,
        stopped_ratio_steps=2,
        standard_num_iterations=1,
        standard_optimizer_steps=2,
    )
    rows = run_benchmark(config, output_dir=tmp_path, fail_fast=True)
    artifacts = merge_benchmark(config, output_dir=tmp_path)

    assert {row["method"] for row in rows} == set(config.methods)
    assert all(row["fit_scope"] == "single_full_training_sample" for row in rows)
    assert all(row["evaluation_scope"] == "independent_external_test" for row in rows)
    assert all(row["calibration_method"] == "none" for row in rows)
    assert len(list((tmp_path / "arrays").glob("*.npz"))) == 1
    assert artifacts["results"].exists()
    manifest = json.loads(artifacts["manifest"].read_text())
    assert manifest["crossfit_folds"] == 0
    assert manifest["calibration_method"] == "none"
    assert manifest["completed_cells"] == 1


def test_repository_metadata_runs_git_from_a_directory(monkeypatch) -> None:
    import occupancy_ratio_benchmark.stopped_fore_external as benchmark

    working_directories = []

    def fake_check_output(command, *, cwd, text):
        assert text is True
        assert cwd.is_dir()
        working_directories.append(cwd)
        return "abc123\n" if command[1:3] == ["rev-parse", "HEAD"] else ""

    monkeypatch.setattr(benchmark.subprocess, "check_output", fake_check_output)

    metadata = benchmark._repository_metadata()

    assert metadata == {"commit": "abc123", "dirty": False, "status": []}
    assert len(working_directories) == 2
