from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from occupancy_ratio import (
    ClippedKLFORIConvergenceError,
    ClippedKLFORIConfig,
    ClippedKLFORIModel,
    fit_clipped_kl_fori,
    fit_clipped_kl_fori_neural,
    fit_kl_fori,
    KLFORIConfig,
)
from occupancy_ratio import _clipped_kl_fori_impl as impl
from occupancy_ratio import _clipped_kl_fori_objectives as objective_spec
from occupancy_ratio_benchmark._clipped_coverage_artifacts import (
    configuration_hash,
    environment_metadata,
    read_cell,
    repository_metadata,
    write_cell,
)
from occupancy_ratio_benchmark._clipped_coverage_oracle import shared_hub_box_oracle
from occupancy_ratio_benchmark._clipped_coverage_freeze import (
    LINEAR_PILOT_SHA256,
    apply_freeze,
    compare_linear_reproduction,
    load_freeze_manifest,
)
from occupancy_ratio_benchmark._clipped_coverage_pilots import (
    neural_finalist_configs,
    neural_screen_configs,
    promoted_neural_ids,
    select_standard_pilot,
    standard_pilot_configs,
)
from occupancy_ratio_benchmark import clipped_coverage as coverage_module
from occupancy_ratio_benchmark import clipped_coverage_freeze as freeze_cli
from occupancy_ratio_benchmark.clipped_coverage_merge import merge_shard_directories
from occupancy_ratio_benchmark.clipped_coverage import (
    CoverageRunConfig,
    SharedHubTruth,
    analytic_shared_hub_rows,
    make_shared_hub_dataset,
    merge_benchmark_shards,
    optimizer_candidate_id,
    pilot_optimizer_configs,
    run_coverage_benchmark,
    select_optimizer_pilot,
    solve_context_coverage_scale,
    validate_benchmark_completeness,
    write_benchmark_artifacts,
)


def _finite_difference(function, theta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = np.empty_like(theta)
    for j in range(theta.size):
        step = np.zeros_like(theta)
        step[j] = eps
        out[j] = (function(theta + step) - function(theta - step)) / (2.0 * eps)
    return out


def _exact_population_shared_hub() -> tuple[dict[str, object], float]:
    dataset = make_shared_hub_dataset(
        n=1_000, gamma=0.5, tau_upper=10.0, oracle_mass=0.5, seed=88
    )
    rows: list[int] = []
    for category, count in ((0, 10), (1, 190), (2, 200)):
        template = int(np.flatnonzero(dataset.category == category)[0])
        rows.extend([template] * count)
    index = np.asarray(rows, dtype=np.int64)
    kwargs: dict[str, object] = {
        "states": dataset.states[index],
        "actions": dataset.actions[index],
        "next_states": dataset.next_states[index],
        "target_next_actions": dataset.target_next_actions[index],
        "initial_states": np.repeat(dataset.initial_states[:1], 400, axis=0),
        "initial_actions": np.repeat(dataset.initial_actions[:1], 400, axis=0),
        "gamma": 0.5,
    }
    return kwargs, dataset.truth.projected_mass(1e-4)


def test_clipped_config_validates_independent_envelopes() -> None:
    config = ClippedKLFORIConfig(tau_lower=1e-4, tau_upper=10.0)
    assert config.tau_lower != pytest.approx(1.0 / config.tau_upper)
    assert config.resolved_gate_learning_rate == pytest.approx(0.05)
    assert ClippedKLFORIConfig(
        backend="neural"
    ).resolved_ratio_learning_rate == pytest.approx(1e-3)
    with pytest.raises(ValueError, match="tau_lower"):
        ClippedKLFORIConfig(tau_lower=0.0)
    with pytest.raises(ValueError, match="backend"):
        ClippedKLFORIConfig(backend="boosting")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tau_upper": np.inf}, "finite"),
        ({"num_iterations": -1}, "nonnegative"),
        ({"min_iterations": -1}, "nonnegative"),
        ({"num_iterations": 1, "min_iterations": 2}, "cannot exceed"),
        ({"outer_tolerance": 0.0}, "positive"),
        ({"outer_patience": 0}, "positive"),
        ({"gate_optimizer_steps": 0}, "positive"),
        ({"ratio_optimizer_steps": 0}, "positive"),
        ({"gate_learning_rate": 0.0}, "positive"),
        ({"ratio_learning_rate": -1.0}, "positive"),
        ({"gate_l2_penalty": -1.0}, "nonnegative"),
        ({"validation_fraction": 1.0}, "validation_fraction"),
        ({"neural_hidden_dims": (0,)}, "positive"),
        ({"neural_weight_decay": -1.0}, "nonnegative"),
        ({"neural_grad_clip_norm": 0.0}, "positive"),
        ({"normalize_eps": 0.0}, "positive"),
        ({"inner_relative_tolerance": 0.0}, "positive"),
        ({"inner_gradient_tolerance": 0.0}, "positive"),
        ({"inner_patience": 0}, "positive"),
        ({"initialization_perturbation_scale": -1.0}, "nonnegative"),
    ],
)
def test_clipped_config_validation_matrix(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ClippedKLFORIConfig(**kwargs)


def test_bounded_ratio_initializes_exactly_at_one() -> None:
    raw = impl._uniform_raw_score(1e-4, 10.0)
    log_ratio, derivative = impl._bounded_log_ratio(
        np.array([raw]), tau_lower=1e-4, tau_upper=10.0
    )
    assert log_ratio[0] == pytest.approx(0.0, abs=1e-12)
    assert derivative[0] > 0.0


def test_package_import_and_linear_fit_leave_torch_lazy() -> None:
    script = """
import sys
import numpy as np
import occupancy_ratio
assert "torch" not in sys.modules
config = occupancy_ratio.ClippedKLFORIConfig(
    num_iterations=0,
    min_iterations=0,
    gate_optimizer_steps=1,
    ratio_optimizer_steps=1,
    validation_fraction=0.0,
)
occupancy_ratio.fit_clipped_kl_fori(
    states=np.array([[0.0], [1.0]]),
    actions=np.array([[0.0], [1.0]]),
    next_states=np.array([[1.0], [1.0]]),
    target_next_actions=np.array([[1.0], [1.0]]),
    initial_states=np.array([[0.0], [0.0]]),
    initial_actions=np.array([[0.0], [0.0]]),
    gamma=0.5,
    config=config,
)
assert "torch" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_independent_objective_specification_matches_hand_computation() -> None:
    scores_ref = np.array([-0.3, 0.4])
    scores_init = np.array([0.2, -0.1])
    scores_plus = np.array([0.5, -0.7])
    init_probs = np.array([0.25, 0.75])
    source_weights = np.array([0.4, 1.6])
    continuation = np.array([1.0, 0.5])
    expected = 7.0 * np.mean(np.logaddexp(0.0, -scores_ref))
    expected += 0.2 * np.sum(init_probs * np.logaddexp(0.0, scores_init))
    expected += 0.8 * np.mean(
        source_weights * continuation * np.logaddexp(0.0, scores_plus)
    )
    assert objective_spec.gate_loss_from_scores(
        scores_ref=scores_ref,
        scores_init=scores_init,
        scores_plus=scores_plus,
        init_probs=init_probs,
        source_weights=source_weights,
        continuation=continuation,
        gamma=0.8,
        tau_upper=7.0,
    ) == pytest.approx(expected)


def test_gate_weight_scaling_is_invariant_to_row_replication() -> None:
    scores_ref = np.array([-0.3, 0.4])
    scores_init = np.array([0.2, -0.1])
    scores_plus = np.array([0.5, -0.7])
    init_probs = np.array([0.25, 0.75])
    source_weights = np.array([0.4, 1.6])
    continuation = np.array([1.0, 0.5])
    kwargs = dict(
        scores_ref=scores_ref,
        scores_init=scores_init,
        scores_plus=scores_plus,
        init_probs=init_probs,
        source_weights=source_weights,
        continuation=continuation,
        gamma=0.8,
        tau_upper=7.0,
    )
    baseline = objective_spec.gate_loss_from_scores(**kwargs)
    repeats = 3
    replicated = objective_spec.gate_loss_from_scores(
        scores_ref=np.repeat(scores_ref, repeats),
        scores_init=np.repeat(scores_init, repeats),
        scores_plus=np.repeat(scores_plus, repeats),
        init_probs=np.repeat(init_probs / repeats, repeats),
        source_weights=np.repeat(source_weights, repeats),
        continuation=np.repeat(continuation, repeats),
        gamma=0.8,
        tau_upper=7.0,
    )
    assert replicated == pytest.approx(baseline)


def test_gate_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(4)
    phi_ref = rng.normal(size=(7, 4))
    phi_init = rng.normal(size=(5, 4))
    phi_plus = rng.normal(size=(6, 4))
    init_probs = np.arange(1, 6, dtype=float)
    init_probs /= init_probs.sum()
    source_weights = rng.uniform(0.2, 1.8, size=6)
    continuation = rng.uniform(0.0, 1.0, size=6)
    theta = rng.normal(scale=0.2, size=4)

    def objective(value: np.ndarray) -> float:
        return impl._linear_gate_objective_and_grad(
            value,
            Phi_ref=phi_ref,
            Phi_init=phi_init,
            Phi_plus=phi_plus,
            init_probs=init_probs,
            source_weights=source_weights,
            continuation=continuation,
            gamma=0.8,
            tau_upper=7.0,
            l2_penalty=0.03,
        )[0]

    _, analytic = impl._linear_gate_objective_and_grad(
        theta,
        Phi_ref=phi_ref,
        Phi_init=phi_init,
        Phi_plus=phi_plus,
        init_probs=init_probs,
        source_weights=source_weights,
        continuation=continuation,
        gamma=0.8,
        tau_upper=7.0,
        l2_penalty=0.03,
    )
    np.testing.assert_allclose(
        analytic, _finite_difference(objective, theta), rtol=1e-5, atol=1e-5
    )


def test_projection_gradient_and_unnormalized_successor_term() -> None:
    rng = np.random.default_rng(8)
    phi_ref = rng.normal(size=(8, 3))
    phi_init = rng.normal(size=(4, 3))
    phi_plus = rng.normal(size=(6, 3))
    init_probs = np.full(4, 0.25)
    source_weights = rng.uniform(0.5, 1.5, size=6)
    continuation = rng.uniform(0.3, 1.0, size=6)
    gate_ref = rng.integers(0, 2, size=8).astype(float)
    gate_init = rng.integers(0, 2, size=4).astype(float)
    gate_plus = rng.integers(0, 2, size=6).astype(float)
    theta = rng.normal(scale=0.2, size=3)

    def objective(value: np.ndarray, scale: float = 1.0) -> float:
        return impl._linear_ratio_objective_and_grad(
            value,
            Phi_ref=phi_ref,
            Phi_init=phi_init,
            Phi_plus=phi_plus,
            init_probs=init_probs,
            source_weights=scale * source_weights,
            continuation=continuation,
            gate_ref=gate_ref,
            gate_init=gate_init,
            gate_plus=gate_plus,
            gamma=0.7,
            tau_lower=1e-4,
            tau_upper=10.0,
            l2_penalty=0.02,
        )[0]

    _, analytic = impl._linear_ratio_objective_and_grad(
        theta,
        Phi_ref=phi_ref,
        Phi_init=phi_init,
        Phi_plus=phi_plus,
        init_probs=init_probs,
        source_weights=source_weights,
        continuation=continuation,
        gate_ref=gate_ref,
        gate_init=gate_init,
        gate_plus=gate_plus,
        gamma=0.7,
        tau_lower=1e-4,
        tau_upper=10.0,
        l2_penalty=0.02,
    )
    np.testing.assert_allclose(
        analytic, _finite_difference(objective, theta), rtol=2e-5, atol=2e-5
    )
    assert objective(theta, scale=2.0) != pytest.approx(objective(theta, scale=1.0))


@pytest.mark.parametrize("mass", [0.0, 0.5, 1.0])
def test_shared_hub_analytic_truth(mass: float) -> None:
    dataset = make_shared_hub_dataset(
        n=10_000,
        gamma=0.95,
        tau_upper=10.0,
        oracle_mass=mass,
        contexts=1,
        seed=5,
    )
    truth = dataset.truth
    assert truth.retained_mass == pytest.approx(mass)
    assert truth.posthoc_mass == pytest.approx(0.95 + 0.05 * mass)
    assert truth.values("constant")["clipped"] == pytest.approx(mass)
    assert truth.values("hub")["clipped"] == pytest.approx(0.95 * mass)
    assert truth.values("gate")["clipped"] == pytest.approx(0.05 * mass)
    expected_projected = mass + (1.0 - truth.q_by_context[0]) * 1e-4
    assert truth.projected_mass(1e-4) == pytest.approx(expected_projected)
    if mass == 0.0:
        assert not np.any(dataset.category == 0)
        assert np.all(np.isfinite(dataset.full_ratio))


def test_shared_hub_clipping_tie_has_unit_mass() -> None:
    truth = SharedHubTruth(
        gamma=0.95,
        tau_upper=10.0,
        q_by_context=np.array([0.1]),
        alpha_by_context=np.array([1.0]),
        context_probability=np.array([1.0]),
    )
    assert truth.retained_mass == pytest.approx(1.0)
    assert truth.posthoc_mass == pytest.approx(1.0)


def test_context_bisection_hits_requested_mass() -> None:
    theta = 2.0 * np.pi * (np.arange(64) + 0.5) / 64.0
    base = 0.1 + 0.4 / (1.0 + np.exp(-3.0 * np.sin(3.0 * theta)))
    probability = np.full(64, 1.0 / 64.0)
    for target in (0.05, 0.5, 0.98, 1.0):
        rho = solve_context_coverage_scale(
            oracle_mass=target,
            base_overlap=base,
            tau_upper=10.0,
            context_probability=probability,
        )
        attained = probability @ np.minimum(1.0, 10.0 * rho * base)
        assert attained == pytest.approx(target, abs=1e-10)


def test_trajectory_sampling_and_irrelevant_features_preserve_truth() -> None:
    dataset = make_shared_hub_dataset(
        n=20_000,
        gamma=0.8,
        tau_upper=10.0,
        oracle_mass=0.5,
        contexts=1,
        seed=118,
        sampling_mode="trajectory",
        irrelevant_features=5,
    )
    assert dataset.states.shape[1] == 8
    assert dataset.next_states.shape[1] == 8
    assert dataset.initial_states.shape[1] == 8
    assert dataset.truth.retained_mass == pytest.approx(0.5)
    empirical = np.bincount(dataset.category, minlength=3) / dataset.n
    expected = np.array([0.2 * 0.05, 0.2 * 0.95, 0.8])
    np.testing.assert_allclose(empirical, expected, atol=0.01)


def test_independent_box_oracle_matches_shared_hub_formulas() -> None:
    rng = np.random.default_rng(117)
    for _ in range(30):
        q = rng.uniform(0.0, 0.4, size=7)
        probability = rng.uniform(size=7)
        probability /= probability.sum()
        tau_lower = float(10.0 ** rng.uniform(-6.0, -3.0))
        tau_upper = float(rng.uniform(2.0, 20.0))
        oracle = shared_hub_box_oracle(
            q=q,
            gamma=0.95,
            tau_lower=tau_lower,
            tau_upper=tau_upper,
            context_probability=probability,
        )
        alpha = np.minimum(1.0, tau_upper * q)
        assert oracle.recursive_mass == pytest.approx(probability @ alpha, abs=1e-12)
        assert oracle.projected_mass == pytest.approx(
            probability @ np.minimum(tau_upper, alpha + (1.0 - q) * tau_lower),
            abs=1e-12,
        )
        assert oracle.value("hub") == pytest.approx(0.95 * oracle.recursive_mass)
    with pytest.raises(ValueError, match="reward"):
        oracle.value("unknown")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"q": np.array([])},
        {"q": np.array([-0.1])},
        {"q": np.array([0.1]), "gamma": 1.0},
        {"q": np.array([0.1]), "tau_lower": 0.0},
        {"q": np.array([0.1, 0.2]), "context_probability": np.array([1.0])},
        {"q": np.array([0.1]), "context_probability": np.array([0.0])},
    ],
)
def test_box_oracle_validation(kwargs: dict[str, object]) -> None:
    base = {"q": np.array([0.1]), "gamma": 0.9, "tau_lower": 1e-4, "tau_upper": 10.0}
    with pytest.raises(ValueError):
        shared_hub_box_oracle(**(base | kwargs))


def test_linear_fit_agrees_with_exact_population_oracle() -> None:
    kwargs, projected_mass = _exact_population_shared_hub()
    model = fit_clipped_kl_fori(
        **kwargs,
        config=ClippedKLFORIConfig(
            num_iterations=50,
            min_iterations=5,
            gate_optimizer_steps=50,
            ratio_optimizer_steps=50,
            validation_fraction=0.0,
            seed=2,
        ),
    )
    assert model.diagnostics["empirical_mass"] == pytest.approx(
        projected_mass, abs=5e-3
    )


def test_neural_fit_agrees_with_exact_population_oracle() -> None:
    pytest.importorskip("torch")
    kwargs, projected_mass = _exact_population_shared_hub()
    model = fit_clipped_kl_fori_neural(
        **kwargs,
        config=ClippedKLFORIConfig(
            backend="neural",
            neural_hidden_dims=(8,),
            num_iterations=30,
            min_iterations=5,
            gate_optimizer_steps=30,
            ratio_optimizer_steps=30,
            validation_fraction=0.0,
            seed=2,
        ),
    )
    assert model.diagnostics["empirical_mass"] == pytest.approx(
        projected_mass, abs=2e-2
    )


def test_linear_fit_helpers_and_serialization(tmp_path: Path) -> None:
    dataset = make_shared_hub_dataset(
        n=400,
        gamma=0.8,
        tau_upper=10.0,
        oracle_mass=0.5,
        seed=9,
    )
    model = fit_clipped_kl_fori(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_next_actions=dataset.target_next_actions,
        initial_states=dataset.initial_states,
        initial_actions=dataset.initial_actions,
        gamma=0.8,
        config=ClippedKLFORIConfig(
            num_iterations=3,
            min_iterations=1,
            outer_tolerance=None,
            gate_optimizer_steps=10,
            ratio_optimizer_steps=15,
            validation_fraction=0.0,
            seed=2,
        ),
    )
    ratio = model.predict_state_action_ratio(dataset.states, dataset.actions)
    assert np.all(ratio >= model.tau_lower)
    assert np.all(ratio <= model.tau_upper)
    assert model.estimate_retained_mass(
        dataset.states, dataset.actions
    ) == pytest.approx(np.mean(ratio))
    expected = np.mean(ratio * dataset.rewards["hub"])
    assert model.estimate_stopped_value(
        dataset.states, dataset.actions, dataset.rewards["hub"]
    ) == pytest.approx(expected)
    assert "X_ref" not in model.fit_payload
    path = tmp_path / "clipped.npz"
    model.save(path)
    loaded = ClippedKLFORIModel.load(path)
    np.testing.assert_allclose(
        loaded.predict_state_action_ratio(dataset.states, dataset.actions), ratio
    )
    assert model.diagnostics["selector_uses_oracle_truth"] == 0.0
    assert loaded.fit_payload["algorithm"] == "clipped_kl_fori"
    assert model.history[-1]["gate_termination_reason"] in {
        "gradient_tolerance",
        "objective_stability",
        "max_steps",
    }
    assert model.history[-1]["ratio_termination_reason"] in {
        "gradient_tolerance",
        "objective_stability",
        "max_steps",
    }
    assert model.history[-1]["outer_relative_change"] == pytest.approx(
        max(
            model.history[-1]["outer_ratio_relative_change"],
            model.history[-1]["outer_gate_change_fraction"],
        )
    )


def test_pre_hardening_standard_and_clipped_behavior_fixture() -> None:
    dataset = make_shared_hub_dataset(
        n=48, gamma=0.7, tau_upper=10.0, oracle_mass=0.5, seed=314
    )
    kwargs = dict(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_next_actions=dataset.target_next_actions,
        initial_states=dataset.initial_states,
        initial_actions=dataset.initial_actions,
        gamma=0.7,
    )
    standard = fit_kl_fori(
        **kwargs,
        config=KLFORIConfig(
            backend="linear",
            num_iterations=2,
            optimizer_steps=5,
            validation_fraction=0.0,
            early_stopping=False,
            seed=2718,
        ),
    )
    clipped = fit_clipped_kl_fori(
        **kwargs,
        config=ClippedKLFORIConfig(
            num_iterations=2,
            min_iterations=1,
            outer_tolerance=None,
            gate_optimizer_steps=4,
            ratio_optimizer_steps=5,
            validation_fraction=0.0,
            seed=2718,
        ),
    )
    expected_standard = {
        0: 15.488199519085562,
        1: 0.18727195900571256,
        2: 0.8647010563670257,
    }
    expected_clipped = {
        0: 8.199508606051435,
        1: 0.018172420020128123,
        2: 0.8227661899005718,
    }
    standard_ratio = standard.predict_state_action_ratio(
        dataset.states, dataset.actions
    )
    clipped_ratio = clipped.predict_state_action_ratio(dataset.states, dataset.actions)
    for category in (0, 1, 2):
        np.testing.assert_allclose(
            standard_ratio[dataset.category == category],
            expected_standard[category],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            clipped_ratio[dataset.category == category],
            expected_clipped[category],
            atol=1e-12,
        )


def test_serialization_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, metadata_json=np.asarray('{"schema":"unknown"}'))
    with pytest.raises(ValueError, match="unsupported"):
        ClippedKLFORIModel.load(path)


def test_serialization_rejects_malformed_and_nonfinite_artifacts(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-metadata.npz"
    np.savez(
        missing,
        metadata_json=np.asarray('{"schema":"clipped-kl-fori-model-v1"}'),
        ratio_coef=np.zeros(3),
        gate_coef=np.zeros(3),
        mean=np.zeros(2),
        scale=np.ones(2),
    )
    with pytest.raises(ValueError, match="missing metadata"):
        ClippedKLFORIModel.load(missing)

    model = ClippedKLFORIModel(
        ratio_coef=np.zeros(3),
        gate_coef=np.zeros(3),
        mean=np.zeros(2),
        scale=np.ones(2),
        gamma=0.5,
        state_dim=1,
        action_dim=1,
        tau_lower=1e-4,
        tau_upper=10.0,
        feature_include_quadratic=False,
        backend="linear",
        neural_hidden_dims=(4,),
        ratio_neural_state_dict={},
        gate_neural_state_dict={},
        history=[],
        diagnostics={},
    )
    valid = tmp_path / "valid.npz"
    model.save(valid)
    with np.load(valid, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}

    nonfinite = tmp_path / "nonfinite.npz"
    arrays["ratio_coef"][0] = np.nan
    np.savez(nonfinite, **arrays)
    with pytest.raises(ValueError, match="nonfinite"):
        ClippedKLFORIModel.load(nonfinite)

    incompatible = tmp_path / "incompatible.npz"
    arrays["ratio_coef"][0] = 0.0
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["backend"] = "torch"
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez(incompatible, **arrays)
    with pytest.raises(ValueError, match="backend metadata"):
        ClippedKLFORIModel.load(incompatible)


def test_loaded_model_validation_rejects_corrupt_shapes(tmp_path: Path) -> None:
    dataset = make_shared_hub_dataset(
        n=80, gamma=0.5, tau_upper=10.0, oracle_mass=0.5, seed=444
    )
    model = fit_clipped_kl_fori(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_next_actions=dataset.target_next_actions,
        initial_states=dataset.initial_states,
        initial_actions=dataset.initial_actions,
        gamma=0.5,
        config=ClippedKLFORIConfig(
            num_iterations=0,
            min_iterations=0,
            gate_optimizer_steps=1,
            ratio_optimizer_steps=1,
            validation_fraction=0.0,
        ),
    )
    for corrupted, message in (
        (replace(model, gamma=1.0), "gamma"),
        (replace(model, tau_lower=0.0), "envelope"),
        (replace(model, state_dim=0), "dimensions"),
        (replace(model, mean=np.zeros(1)), "standardizer"),
        (replace(model, scale=np.zeros_like(model.scale)), "scale"),
        (replace(model, ratio_coef=np.zeros(1)), "coefficient"),
        (replace(model, ratio_neural_state_dict={"x": np.ones(1)}), "neural"),
    ):
        with pytest.raises(ValueError, match=message):
            impl._validate_loaded_model(corrupted)


def test_require_convergence_raises_with_diagnostics() -> None:
    dataset = make_shared_hub_dataset(
        n=80, gamma=0.8, tau_upper=10.0, oracle_mass=0.5, seed=901
    )
    with pytest.raises(ClippedKLFORIConvergenceError) as caught:
        fit_clipped_kl_fori(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_next_actions=dataset.target_next_actions,
            initial_states=dataset.initial_states,
            initial_actions=dataset.initial_actions,
            gamma=0.8,
            config=ClippedKLFORIConfig(
                num_iterations=1,
                min_iterations=1,
                gate_optimizer_steps=1,
                ratio_optimizer_steps=1,
                validation_fraction=0.0,
                require_convergence=True,
            ),
        )
    assert caught.value.diagnostics["converged"] is False
    assert (
        caught.value.model.diagnostics["outer_termination_reason"]
        == "configured_iterations"
    )


def test_grouped_validation_terminal_handling_and_seed_determinism() -> None:
    dataset = make_shared_hub_dataset(
        n=96,
        gamma=0.7,
        tau_upper=10.0,
        oracle_mass=0.75,
        seed=17,
    )
    groups = np.arange(dataset.n) // 4
    initial_groups = np.arange(dataset.initial_states.shape[0]) // 4
    terminals = np.zeros(dataset.n, dtype=bool)
    terminals[::11] = True
    config = ClippedKLFORIConfig(
        num_iterations=2,
        min_iterations=1,
        outer_tolerance=None,
        gate_optimizer_steps=3,
        ratio_optimizer_steps=4,
        validation_fraction=0.25,
        seed=21,
    )
    kwargs = dict(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_next_actions=dataset.target_next_actions,
        initial_states=dataset.initial_states,
        initial_actions=dataset.initial_actions,
        terminals=terminals,
        groups=groups,
        initial_groups=initial_groups,
        gamma=0.7,
        config=config,
    )
    first = fit_clipped_kl_fori(**kwargs)
    second = fit_clipped_kl_fori(**kwargs)
    np.testing.assert_array_equal(
        first.predict_state_action_ratio(dataset.states, dataset.actions),
        second.predict_state_action_ratio(dataset.states, dataset.actions),
    )
    assert first.diagnostics["reference_split_grouped"] is True
    assert first.diagnostics["initial_split_grouped"] is True
    assert first.diagnostics["refit_after_validation"] is True
    sample_weight = np.linspace(1.0, 2.0, dataset.n)
    ratio = first.predict_state_action_ratio(dataset.states, dataset.actions)
    assert first.estimate_retained_mass(
        dataset.states, dataset.actions, sample_weight=sample_weight
    ) == pytest.approx(np.average(ratio, weights=sample_weight))


def test_target_policy_initial_weights_timeouts_and_zero_iteration() -> None:
    dataset = make_shared_hub_dataset(
        n=32, gamma=0.8, tau_upper=10.0, oracle_mass=0.5, seed=171
    )

    def target_policy(states, rng):
        del rng
        category = np.where(np.asarray(states)[:, 0] > 0.5, 2, 0)
        return np.eye(3)[category]

    base = dict(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_policy=target_policy,
        initial_states=dataset.initial_states,
        initial_weights=np.linspace(1.0, 2.0, dataset.initial_states.shape[0]),
        gamma=0.8,
        config=ClippedKLFORIConfig(
            num_iterations=0,
            min_iterations=0,
            gate_optimizer_steps=1,
            ratio_optimizer_steps=1,
            validation_fraction=0.0,
            seed=12,
        ),
    )
    model = fit_clipped_kl_fori(**base)
    np.testing.assert_allclose(
        model.predict_state_action_ratio(dataset.states, dataset.actions), 1.0
    )
    assert model.diagnostics["initial_action_source"] == "target_policy"
    assert model.diagnostics["successor_action_source"] == "target_policy"
    assert model.diagnostics["iterations_completed"] == 0

    terminal = np.zeros(dataset.n, dtype=bool)
    timeout = np.zeros(dataset.n, dtype=bool)
    terminal[:4] = True
    timeout[4:8] = True
    explicit = base | {
        "target_policy": None,
        "target_next_actions": dataset.target_next_actions,
        "initial_actions": dataset.initial_actions,
        "terminals": terminal,
        "timeouts": timeout,
        "handle_timeouts": "terminal",
    }
    stopped = fit_clipped_kl_fori(**explicit)
    assert stopped.diagnostics["continuation_mean"] == pytest.approx(0.75)
    absorbing = fit_clipped_kl_fori(**(explicit | {"absorbing_state": True}))
    assert absorbing.diagnostics["continuation_mean"] == pytest.approx(1.0)


def test_clipped_input_validation_and_extreme_finite_stress() -> None:
    dataset = make_shared_hub_dataset(
        n=32, gamma=0.9, tau_upper=10.0, oracle_mass=0.5, seed=172
    )
    kwargs = dict(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_next_actions=dataset.target_next_actions,
        initial_states=dataset.initial_states,
        initial_actions=dataset.initial_actions,
        gamma=0.9,
        config=ClippedKLFORIConfig(
            num_iterations=1,
            min_iterations=1,
            outer_tolerance=None,
            gate_optimizer_steps=2,
            ratio_optimizer_steps=2,
            validation_fraction=0.0,
        ),
    )
    with pytest.raises(ValueError, match="same number of rows"):
        fit_clipped_kl_fori(**(kwargs | {"actions": dataset.actions[:-1]}))
    bad_states = dataset.states.copy()
    bad_states[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_clipped_kl_fori(**(kwargs | {"states": bad_states}))
    with pytest.raises(ValueError, match="nonnegative"):
        fit_clipped_kl_fori(
            **(kwargs | {"initial_weights": -np.ones(dataset.initial_states.shape[0])})
        )

    scale = 1e100
    stressed = fit_clipped_kl_fori(
        **(
            kwargs
            | {
                "states": dataset.states * scale,
                "next_states": dataset.next_states * scale,
                "initial_states": dataset.initial_states * scale,
                "gamma": 1.0 - 1e-10,
                "terminals": np.ones(dataset.n, dtype=bool),
                "config": replace(kwargs["config"], tau_lower=1e-12, tau_upper=1e6),
            }
        )
    )
    ratio = stressed.predict_state_action_ratio(dataset.states * scale, dataset.actions)
    assert np.all(np.isfinite(ratio))
    assert np.all((ratio >= 1e-12) & (ratio <= 1e6))


def test_ratio_optimizer_does_not_mutate_frozen_gate(monkeypatch) -> None:
    dataset = make_shared_hub_dataset(
        n=64, gamma=0.6, tau_upper=10.0, oracle_mass=0.5, seed=72
    )
    original = impl._fit_linear_ratio_adam

    def checked(**kwargs):
        before = {
            name: np.asarray(kwargs[name]).copy()
            for name in ("gate_ref", "gate_init", "gate_plus")
        }
        result = original(**kwargs)
        for name, value in before.items():
            np.testing.assert_array_equal(kwargs[name], value)
        return result

    monkeypatch.setattr(impl, "_fit_linear_ratio_adam", checked)
    fit_clipped_kl_fori(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_next_actions=dataset.target_next_actions,
        initial_states=dataset.initial_states,
        initial_actions=dataset.initial_actions,
        gamma=0.6,
        config=ClippedKLFORIConfig(
            num_iterations=1,
            min_iterations=1,
            outer_tolerance=None,
            gate_optimizer_steps=2,
            ratio_optimizer_steps=2,
            validation_fraction=0.0,
        ),
    )


def test_nonfinite_inner_objective_fails_immediately(monkeypatch) -> None:
    dataset = make_shared_hub_dataset(
        n=32, gamma=0.5, tau_upper=10.0, oracle_mass=0.5, seed=73
    )

    def nonfinite(*args, **kwargs):
        theta = np.asarray(args[0] if args else kwargs["theta"])
        return float("nan"), np.zeros_like(theta)

    monkeypatch.setattr(impl, "_linear_gate_objective_and_grad", nonfinite)
    with pytest.raises(FloatingPointError, match="objective"):
        fit_clipped_kl_fori(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_next_actions=dataset.target_next_actions,
            initial_states=dataset.initial_states,
            initial_actions=dataset.initial_actions,
            gamma=0.5,
            config=ClippedKLFORIConfig(
                num_iterations=1,
                min_iterations=1,
                gate_optimizer_steps=1,
                ratio_optimizer_steps=1,
                validation_fraction=0.0,
            ),
        )


def test_inner_and_outer_termination_reasons_are_executable() -> None:
    dataset = make_shared_hub_dataset(
        n=48, gamma=0.5, tau_upper=10.0, oracle_mass=0.5, seed=74
    )
    kwargs = dict(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_next_actions=dataset.target_next_actions,
        initial_states=dataset.initial_states,
        initial_actions=dataset.initial_actions,
        gamma=0.5,
    )
    gradient = fit_clipped_kl_fori(
        **kwargs,
        config=ClippedKLFORIConfig(
            num_iterations=5,
            min_iterations=1,
            outer_tolerance=1e9,
            outer_patience=1,
            gate_optimizer_steps=5,
            ratio_optimizer_steps=5,
            inner_gradient_tolerance=1e9,
            validation_fraction=0.0,
        ),
    )
    assert gradient.history[-1]["gate_termination_reason"] == "gradient_tolerance"
    assert gradient.history[-1]["ratio_termination_reason"] == "gradient_tolerance"
    assert gradient.diagnostics["outer_termination_reason"] == "deployable_tolerance"
    assert gradient.diagnostics["iterations_completed"] == 1

    stable = fit_clipped_kl_fori(
        **kwargs,
        config=ClippedKLFORIConfig(
            num_iterations=1,
            min_iterations=1,
            outer_tolerance=None,
            gate_optimizer_steps=5,
            ratio_optimizer_steps=5,
            inner_relative_tolerance=1e9,
            inner_gradient_tolerance=1e-30,
            inner_patience=1,
            validation_fraction=0.0,
        ),
    )
    assert stable.history[-1]["gate_termination_reason"] == "objective_stability"
    assert stable.history[-1]["ratio_termination_reason"] == "objective_stability"
    assert impl._json_safe(np.array([1.0, np.nan])) == [1.0, None]
    assert impl._json_safe(np.int64(3)) == 3
    assert isinstance(impl._json_safe(object()), str)


def test_outer_convergence_requires_configured_patience() -> None:
    dataset = make_shared_hub_dataset(
        n=48, gamma=0.5, tau_upper=10.0, oracle_mass=0.5, seed=741
    )
    kwargs = dict(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_next_actions=dataset.target_next_actions,
        initial_states=dataset.initial_states,
        initial_actions=dataset.initial_actions,
        gamma=0.5,
    )
    base = ClippedKLFORIConfig(
        num_iterations=2,
        min_iterations=1,
        outer_tolerance=1e9,
        outer_patience=3,
        gate_optimizer_steps=1,
        ratio_optimizer_steps=1,
        validation_fraction=0.0,
    )
    insufficient = fit_clipped_kl_fori(**kwargs, config=base)
    assert insufficient.diagnostics["converged"] is False
    assert insufficient.diagnostics["outer_termination_reason"] == (
        "configured_iterations"
    )
    sufficient = fit_clipped_kl_fori(**kwargs, config=replace(base, num_iterations=5))
    assert sufficient.diagnostics["converged"] is True
    assert sufficient.diagnostics["iterations_completed"] == 3


def test_neural_backend_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    dataset = make_shared_hub_dataset(
        n=80,
        gamma=0.5,
        tau_upper=10.0,
        oracle_mass=1.0,
        seed=11,
    )
    model = fit_clipped_kl_fori_neural(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_next_actions=dataset.target_next_actions,
        initial_states=dataset.initial_states,
        initial_actions=dataset.initial_actions,
        gamma=0.5,
        config=ClippedKLFORIConfig(
            backend="neural",
            neural_hidden_dims=(4,),
            num_iterations=1,
            min_iterations=1,
            outer_tolerance=None,
            gate_optimizer_steps=1,
            ratio_optimizer_steps=1,
            validation_fraction=0.0,
            seed=3,
        ),
    )
    ratio = model.predict_state_action_ratio(dataset.states, dataset.actions)
    assert np.all(np.isfinite(ratio))
    assert model.backend == "neural"
    assert model.diagnostics["neural_deterministic_requested"] is True
    path = tmp_path / "neural.npz"
    model.save(path)
    restored = ClippedKLFORIModel.load(path)
    np.testing.assert_allclose(
        restored.predict_state_action_ratio(dataset.states, dataset.actions),
        ratio,
        atol=1e-7,
        rtol=0.0,
    )


def test_neural_initialization_and_seed_reproducibility() -> None:
    pytest.importorskip("torch")
    dataset = make_shared_hub_dataset(
        n=40, gamma=0.5, tau_upper=10.0, oracle_mass=1.0, seed=110
    )
    config = ClippedKLFORIConfig(
        backend="neural",
        neural_hidden_dims=(4,),
        num_iterations=0,
        min_iterations=0,
        gate_optimizer_steps=1,
        ratio_optimizer_steps=1,
        validation_fraction=0.0,
        seed=44,
    )
    kwargs = dict(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_next_actions=dataset.target_next_actions,
        initial_states=dataset.initial_states,
        initial_actions=dataset.initial_actions,
        gamma=0.5,
        config=config,
    )
    first = fit_clipped_kl_fori_neural(**kwargs)
    second = fit_clipped_kl_fori_neural(**kwargs)
    first_ratio = first.predict_state_action_ratio(dataset.states, dataset.actions)
    second_ratio = second.predict_state_action_ratio(dataset.states, dataset.actions)
    np.testing.assert_allclose(first_ratio, 1.0, atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(first_ratio, second_ratio, atol=1e-7, rtol=0.0)


def test_benchmark_smoke_writes_reproducible_artifacts(tmp_path: Path) -> None:
    config = CoverageRunConfig(
        n=100,
        repetitions=1,
        mass_grid=(0.0, 1.0),
        contexts=1,
        backends=("linear",),
        clipped_num_iterations=1,
        clipped_gate_steps=2,
        clipped_ratio_steps=2,
        standard_num_iterations=1,
        standard_optimizer_steps=2,
        validation_fraction=0.0,
    )
    rows = run_coverage_benchmark(config)
    assert any(
        row["method"] == "clipped_fori" and row["status"] == "nonconverged"
        for row in rows
    )
    assert any(
        row["method"] == "standard_fori" and row["status"] == "out_of_regime"
        for row in rows
    )
    artifacts = write_benchmark_artifacts(
        tmp_path, rows=rows, config=config, make_plots=False
    )
    assert all(path.exists() for path in artifacts.values())
    manifest = (tmp_path / "manifest.json").read_text()
    assert '"oracle_used_for_selection": false' in manifest
    assert '"schema": "clipped-coverage-v2"' in manifest
    assert '"git_revision"' in manifest
    assert configuration_hash(config) in manifest
    assert len(analytic_shared_hub_rows(mass_grid=(0.0, 1.0))) == 6
    plotted = write_benchmark_artifacts(
        tmp_path / "plotted", rows=rows, config=config, make_plots=True
    )
    assert plotted["figure_pdf"].exists()
    assert plotted["figure_png"].exists()
    reversed_artifacts = write_benchmark_artifacts(
        tmp_path / "reversed",
        rows=[
            dict(row, runtime_sec=999.0 + index)
            for index, row in enumerate(reversed(rows))
        ],
        config=config,
        make_plots=False,
    )
    original_manifest = json.loads(artifacts["manifest"].read_text())
    reversed_manifest = json.loads(reversed_artifacts["manifest"].read_text())
    assert (
        original_manifest["deterministic_results_hash"]
        == reversed_manifest["deterministic_results_hash"]
    )


def test_benchmark_cells_resume_and_shard_reproducibly(
    tmp_path: Path, monkeypatch
) -> None:
    config = CoverageRunConfig(
        n=80,
        repetitions=1,
        mass_grid=(0.0, 0.5, 1.0),
        backends=("linear",),
        clipped_num_iterations=1,
        clipped_gate_steps=1,
        clipped_ratio_steps=1,
        standard_num_iterations=1,
        standard_optimizer_steps=1,
    )
    full = run_coverage_benchmark(config, cell_dir=tmp_path / "cells")
    assert len(list((tmp_path / "cells").glob("*.json"))) == 6
    assert all(len(json.loads(row["fold_cell_ids"])) == 2 for row in full)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("resume should not refit completed cells")

    monkeypatch.setattr(
        "occupancy_ratio_benchmark.clipped_coverage._run_backend_fold", fail_if_called
    )
    resumed = run_coverage_benchmark(config, cell_dir=tmp_path / "cells", resume=True)
    assert {row["cell_id"] for row in resumed} == {row["cell_id"] for row in full}
    shard_zero = run_coverage_benchmark(
        config, cell_dir=tmp_path / "cells", shard_index=0, num_shards=2
    )
    shard_one = run_coverage_benchmark(
        config, cell_dir=tmp_path / "cells", shard_index=1, num_shards=2
    )
    assert {row["cell_id"] for row in shard_zero + shard_one} == {
        row["cell_id"] for row in full
    }
    assert (
        merge_benchmark_shards([shard_zero, shard_one], configs=[config])
        == shard_zero + shard_one
    )
    with pytest.raises(ValueError, match="incomplete"):
        validate_benchmark_completeness(full[:-3], [config])
    with pytest.raises(ValueError, match="duplicate"):
        write_benchmark_artifacts(
            tmp_path / "duplicate",
            rows=[*full, dict(full[0])],
            config=config,
            make_plots=False,
        )


def test_shard_directory_merge_requires_complete_compatible_set(
    tmp_path: Path, monkeypatch
) -> None:
    config = CoverageRunConfig(
        n=50,
        repetitions=1,
        mass_grid=(0.0, 0.5),
        methods=("clipped_fori",),
        clipped_num_iterations=1,
        clipped_gate_steps=1,
        clipped_ratio_steps=1,
    )
    monkeypatch.setattr(coverage_module, "smoke_config", lambda seed: config)
    output = tmp_path / "sharded"
    for shard in (0, 1):
        assert coverage_module.main(
            [
                "--stage",
                "smoke",
                "--output-dir",
                str(output),
                "--num-shards",
                "2",
                "--shard-index",
                str(shard),
                "--no-plot",
            ]
        ) == 0
    with pytest.raises(ValueError, match="all declared shards"):
        merge_shard_directories(
            [output / "shard_000"], output_dir=tmp_path / "incomplete"
        )
    artifacts = merge_shard_directories(
        [output / "shard_000", output / "shard_001"],
        output_dir=tmp_path / "merged",
        make_plots=False,
    )
    assert artifacts["results"].exists()
    assert artifacts["figure_inputs"].exists()


def test_pilot_selector_uses_only_deployable_diagnostics() -> None:
    optimizer_config = {
        "clipped_gate_steps": 200,
        "clipped_ratio_steps": 300,
        "clipped_gate_learning_rate": 0.05,
        "clipped_ratio_learning_rate": 0.02,
        "clipped_inner_relative_tolerance": 1e-10,
        "clipped_inner_gradient_tolerance": 1e-6,
        "clipped_inner_patience": 5,
    }
    candidates = [
        (
            "slow",
            [
                {
                    "method": "clipped_fori",
                    "status": "ok",
                    "objective_stability": 5e-5,
                    "optimizer_restart_objective_gap": 3e-5,
                    "optimizer_restart_ratio_l1": 2e-5,
                    "optimizer_restart_gate_disagreement": 0.0,
                    "runtime_sec": 2.0,
                    "mass_abs_error_vs_recursive": 0.0,
                    **optimizer_config,
                }
            ],
        ),
        (
            "fast",
            [
                {
                    "method": "clipped_fori",
                    "status": "ok",
                    "objective_stability": 8e-5,
                    "optimizer_restart_objective_gap": 4e-5,
                    "optimizer_restart_ratio_l1": 3e-5,
                    "optimizer_restart_gate_disagreement": 0.0,
                    "runtime_sec": 1.0,
                    "mass_abs_error_vs_recursive": 10.0,
                    **optimizer_config,
                }
            ],
        ),
    ]
    result = select_optimizer_pilot(candidates)
    assert result["selected_candidate_id"] == "fast"
    assert result["selector_uses_oracle_truth"] is False
    assert result["candidates"][1]["optimizer_config"] == optimizer_config


def test_pilot_configs_are_paired_and_ids_cover_tuned_fields() -> None:
    configs = pilot_optimizer_configs(seed=91)
    assert {config.seed for config in configs} == {91}
    assert len({optimizer_candidate_id(config) for config in configs}) == len(configs)
    assert {config.clipped_ratio_learning_rate for config in configs} == {0.02, 0.05}
    assert {config.clipped_inner_relative_tolerance for config in configs} == {
        1e-8,
        1e-10,
    }


def test_pilot_selector_reports_failed_restart_metrics() -> None:
    row = {
        "method": "clipped_fori",
        "status": "optimizer_unstable",
        "objective_stability": 2e-4,
        "optimizer_restart_objective_gap": 0.3,
        "optimizer_restart_ratio_l1": 0.2,
        "optimizer_restart_gate_disagreement": 0.1,
        "runtime_sec": 1.0,
        "clipped_gate_steps": 200,
        "clipped_ratio_steps": 300,
        "clipped_gate_learning_rate": 0.05,
        "clipped_ratio_learning_rate": 0.02,
        "clipped_inner_relative_tolerance": 1e-10,
        "clipped_inner_gradient_tolerance": 1e-6,
        "clipped_inner_patience": 5,
    }
    audit = select_optimizer_pilot([("failed", [row])])["candidates"][0]
    assert audit["optimizer_restart_objective_gap_max"] == pytest.approx(0.3)
    assert audit["optimizer_restart_ratio_l1_max"] == pytest.approx(0.2)
    assert audit["optimizer_restart_gate_disagreement_max"] == pytest.approx(0.1)


def test_nonconvergence_precedes_optimizer_instability() -> None:
    config = CoverageRunConfig(
        n=60,
        repetitions=1,
        mass_grid=(0.5,),
        backends=("linear",),
        clipped_num_iterations=1,
        clipped_gate_steps=1,
        clipped_ratio_steps=1,
        standard_num_iterations=1,
        standard_optimizer_steps=1,
        optimizer_stability_restarts=3,
        optimizer_perturbation_scale=1.0,
    )
    rows = run_coverage_benchmark(config)
    clipped = next(row for row in rows if row["method"] == "clipped_fori")
    assert clipped["status"] == "nonconverged"
    assert clipped["optimizer_restart_objective_gap"] > 1e-4
    assert np.isfinite(clipped["optimizer_restart_ratio_l1"])
    assert np.isfinite(clipped["optimizer_restart_gate_disagreement"])


def test_benchmark_fit_exception_is_structured_error(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise FloatingPointError("deliberate nonfinite audit failure")

    monkeypatch.setattr(
        "occupancy_ratio_benchmark._clipped_coverage_execution.fit_clipped_kl_fori",
        fail,
    )
    config = CoverageRunConfig(
        n=40,
        repetitions=1,
        mass_grid=(0.5,),
        backends=("linear",),
        clipped_num_iterations=1,
        clipped_gate_steps=1,
        clipped_ratio_steps=1,
        standard_num_iterations=1,
        standard_optimizer_steps=1,
    )
    rows = run_coverage_benchmark(config)
    clipped = next(row for row in rows if row["method"] == "clipped_fori")
    assert clipped["status"] == "error"
    assert clipped["failure_type"] == "fit_exception"
    assert "FloatingPointError" in clipped["error"]


@pytest.mark.parametrize(
    "termination_encoding", ["terminal_absorbing", "timeout_absorbing"]
)
def test_benchmark_executes_trajectory_and_irrelevant_feature_stress(
    termination_encoding: str,
) -> None:
    config = CoverageRunConfig(
        n=64,
        repetitions=1,
        mass_grid=(0.5,),
        contexts=4,
        backends=("linear",),
        clipped_num_iterations=1,
        clipped_gate_steps=2,
        clipped_ratio_steps=2,
        standard_num_iterations=1,
        standard_optimizer_steps=2,
        sampling_mode="trajectory",
        irrelevant_features=3,
        termination_encoding=termination_encoding,
    )
    rows = run_coverage_benchmark(config)
    assert {row["method"] for row in rows} == {
        "clipped_fori",
        "standard_fori",
        "posthoc_winsorized",
    }
    assert all(row["sampling_mode"] == "trajectory" for row in rows)
    assert all(row["irrelevant_features"] == 3 for row in rows)
    assert all(row["termination_encoding"] == termination_encoding for row in rows)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0},
        {"gamma": 1.0},
        {"tau_lower": 0.0},
        {"contexts": 0},
        {"crossfit_folds": 3},
        {"mass_grid": (-0.1,)},
        {"backends": ("boosting",)},
        {"methods": ()},
        {"methods": ("posthoc_winsorized",)},
        {"methods": ("unknown",)},
        {"optimizer_stability_restarts": 2},
        {"optimizer_perturbation_scale": 0.0},
        {"clipped_inner_relative_tolerance": 0.0},
        {"clipped_inner_gradient_tolerance": 0.0},
        {"clipped_inner_patience": 0},
        {"sampling_mode": "bad"},
        {"irrelevant_features": -1},
        {"termination_encoding": "bad"},
    ],
)
def test_benchmark_config_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CoverageRunConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0},
        {"sampling_mode": "bad"},
        {"irrelevant_features": -1},
        {"oracle_mass": 2.0},
        {"gamma": 1.0},
        {"termination_encoding": "bad"},
    ],
)
def test_shared_hub_dataset_validation(kwargs: dict[str, object]) -> None:
    base = {"n": 10, "gamma": 0.9, "tau_upper": 10.0, "oracle_mass": 0.5}
    with pytest.raises(ValueError):
        make_shared_hub_dataset(**(base | kwargs))


def test_benchmark_cli_smoke_and_backend_freeze_validation(tmp_path: Path) -> None:
    assert (
        coverage_module.main(
            ["--stage", "smoke", "--output-dir", str(tmp_path / "cli"), "--no-plot"]
        )
        == 0
    )
    confirmatory = coverage_module.confirmatory_configs()
    sensitivities = coverage_module.sensitivity_configs()
    assert len(confirmatory) == 3
    assert len(sensitivities) == 4
    assert sum(
        config.repetitions * len(config.mass_grid) * len(config.methods)
        for config in confirmatory
    ) == 7_500
    assert sum(
        config.repetitions * len(config.mass_grid) * len(config.backends) * 2
        for config in confirmatory
    ) == 5_000
    clipped_linear = {
        "clipped_gate_steps": 200,
        "clipped_ratio_steps": 300,
        "clipped_gate_learning_rate": 0.05,
        "clipped_ratio_learning_rate": 0.02,
        "clipped_inner_relative_tolerance": 1e-10,
        "clipped_inner_gradient_tolerance": 1e-6,
        "clipped_inner_patience": 5,
    }
    standard_linear = {
        "standard_num_iterations": 100,
        "standard_optimizer_steps": 300,
        "standard_outer_tolerance": 1e-4,
        "standard_objective_tolerance": 1e-5,
        "standard_mass_tolerance": 1e-10,
        "standard_require_convergence": True,
    }
    valid = {
        "schema": "clipped-coverage-freeze-v1",
        "source_revision": "abc123",
        "oracle_used_for_selection": False,
        "optimizer_configs": {
            "clipped_linear": clipped_linear,
            "standard_linear": standard_linear,
        },
        "pilot_artifacts": {
            "clipped_linear": {"sha256": LINEAR_PILOT_SHA256},
            "standard_linear": {"sha256": "standard"},
        },
        "excluded_components": {
            "contextual_neural_appendix": {
                "audit_sha256": "neural-null-pilot"
            }
        },
        "source_compatibility": {
            "reproduction_critical_files_unchanged": True
        },
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(valid))
    frozen = load_freeze_manifest(freeze_path, expected_revision="abc123")
    linear = apply_freeze(confirmatory[0], frozen)
    assert linear.clipped_ratio_learning_rate == pytest.approx(0.02)
    assert linear.standard_num_iterations == 100
    with pytest.raises(ValueError, match="only linear"):
        apply_freeze(replace(confirmatory[0], backends=("neural",)), frozen)
    freeze_path.write_text(
        json.dumps(valid | {"oracle_used_for_selection": True})
    )
    with pytest.raises(ValueError, match="oracle-free"):
        load_freeze_manifest(freeze_path)


def test_benchmark_guardrail_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shard_index"):
        run_coverage_benchmark(CoverageRunConfig(), shard_index=2, num_shards=2)
    with pytest.raises(ValueError, match="attainable"):
        solve_context_coverage_scale(
            oracle_mass=1.0,
            base_overlap=np.array([0.01]),
            tau_upper=2.0,
            context_probability=np.array([1.0]),
        )


def test_linear_reproduction_compares_fold_predictions(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    reproduced = tmp_path / "reproduced"
    reference.mkdir()
    reproduced.mkdir()
    row = {
        "fit_seed": 11,
        "fold": 0,
        "requested_mass": 0.5,
        "clipped": {
            "status": "ok",
            "prediction": [0.2, 0.8],
            "gate_prediction": [0.0, 1.0],
        },
    }
    payload = {"rows": [row]}
    (reference / "a.json").write_text(json.dumps(payload))
    (reproduced / "b.json").write_text(json.dumps(payload))
    audit = compare_linear_reproduction(reference, reproduced)
    assert audit["matched"] is True
    assert audit["reference_prediction_hash"] == audit["reproduced_prediction_hash"]
    changed = json.loads(json.dumps(payload))
    changed["rows"][0]["clipped"]["prediction"][0] = 0.3
    (reproduced / "b.json").write_text(json.dumps(changed))
    audit = compare_linear_reproduction(reference, reproduced)
    assert audit["matched"] is False
    assert audit["maximum_prediction_abs_error"] == pytest.approx(0.1)


def test_freeze_payload_records_exact_grids_and_truth_blind_pilots(
    tmp_path: Path, monkeypatch
) -> None:
    clipped_linear = {
        "clipped_gate_steps": 200,
        "clipped_ratio_steps": 300,
        "clipped_gate_learning_rate": 0.05,
        "clipped_ratio_learning_rate": 0.02,
        "clipped_inner_relative_tolerance": 1e-10,
        "clipped_inner_gradient_tolerance": 1e-6,
        "clipped_inner_patience": 5,
    }
    standard = {
        "standard_num_iterations": 100,
        "standard_optimizer_steps": 300,
        "standard_outer_tolerance": 1e-4,
        "standard_objective_tolerance": 1e-5,
        "standard_mass_tolerance": 1e-10,
        "standard_require_convergence": True,
    }

    def write_selection(name: str, config: dict[str, object]) -> Path:
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "selected_candidate_id": "selected",
                    "selector_uses_oracle_truth": False,
                    "candidates": [
                        {
                            "candidate_id": "selected",
                            "eligible": True,
                            "optimizer_config": config,
                        }
                    ],
                }
            )
        )
        return path

    linear = write_selection("linear-original.json", clipped_linear)
    verification = write_selection("linear-verification.json", clipped_linear)
    standard_path = write_selection("standard.json", standard)
    neural_audit = tmp_path / "neural-exclusion.json"
    neural_audit.write_text(
        json.dumps(
            {
                "schema": "clipped-neural-exclusion-v1",
                "decision": "exclude_contextual_neural_appendix",
                "reason": "three completed candidates failed the zero-failure gate",
                "completed_candidates": 3,
                "eligible_candidates": 0,
                "oracle_used_for_selection": False,
            }
        )
    )

    def fake_artifact(path):
        value = Path(path)
        return {
            "path": str(value),
            "sha256": (
                LINEAR_PILOT_SHA256
                if value.name == "linear-original.json"
                else value.name
            ),
        }

    monkeypatch.setattr(freeze_cli, "selection_artifact", fake_artifact)
    payload = freeze_cli.build_freeze_payload(
        source_revision="revision",
        environment={"python": "test"},
        linear_selection_path=linear,
        linear_verification_path=verification,
        standard_selection_path=standard_path,
        neural_exclusion_audit_path=neural_audit,
        reproduction_audit={"matched": True},
        source_compatibility={"reproduction_critical_files_unchanged": True},
    )
    assert payload["oracle_used_for_selection"] is False
    assert len(payload["grids"]["confirmatory"]) == 3
    assert len(payload["grids"]["sensitivity"]) == 4
    assert payload["expected_counts"]["confirmatory_method_rows"] == 7_500
    assert payload["optimizer_configs"]["standard_linear"] == standard
    with pytest.raises(ValueError, match="predictions"):
        freeze_cli.build_freeze_payload(
            source_revision="revision",
            environment={},
            linear_selection_path=linear,
            linear_verification_path=verification,
            standard_selection_path=standard_path,
            neural_exclusion_audit_path=neural_audit,
            reproduction_audit={"matched": False},
            source_compatibility={"reproduction_critical_files_unchanged": True},
        )


def test_method_routing_skips_unrequested_standard_fit(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("standard FORE must not be fitted")

    monkeypatch.setattr(
        "occupancy_ratio_benchmark._clipped_coverage_execution.fit_kl_fori", fail
    )
    config = CoverageRunConfig(
        n=60,
        repetitions=1,
        mass_grid=(0.5,),
        methods=("clipped_fori",),
        clipped_num_iterations=1,
        clipped_gate_steps=1,
        clipped_ratio_steps=1,
    )
    rows = run_coverage_benchmark(config)
    assert {row["method"] for row in rows} == {"clipped_fori"}


def test_single_context_sampling_error_decomposition_is_exact() -> None:
    config = CoverageRunConfig(
        n=200,
        repetitions=1,
        mass_grid=(0.5,),
        methods=("clipped_fori",),
        clipped_num_iterations=1,
        clipped_gate_steps=1,
        clipped_ratio_steps=1,
    )
    row = run_coverage_benchmark(config)[0]
    assert row["observed_initial_rows"] >= 0
    assert row["observed_target_branch_rows"] >= 0
    assert row["expected_target_branch_rows"] == pytest.approx(
        config.n * (1.0 - config.gamma) * row["q_min"]
    )
    assert row["mass_error_vs_projected"] == pytest.approx(
        row["mass_fitting_error_vs_empirical_support"]
        + row["mass_sampling_error_empirical_vs_population"]
    )


def test_standard_selector_is_truth_blind() -> None:
    base = {
        "method": "standard_fori",
        "status": "ok",
        "standard_num_iterations": 30,
        "standard_optimizer_steps": 300,
        "standard_outer_tolerance": 1e-4,
        "standard_objective_tolerance": 1e-5,
        "standard_mass_tolerance": 1e-10,
        "standard_weight_step_l1": 2e-5,
        "standard_objective_relative_change": 2e-6,
        "standard_mass_abs_error": 2e-12,
    }
    candidates = [
        ("slow", [base | {"runtime_sec": 2.0, "ratio_l1_error": 0.0}]),
        ("fast", [base | {"runtime_sec": 1.0, "ratio_l1_error": 100.0}]),
    ]
    selected = select_standard_pilot(candidates)
    stripped = select_standard_pilot(
        [
            (candidate_id, [{k: v for k, v in row.items() if k != "ratio_l1_error"}])
            for candidate_id, rows in candidates
            for row in rows
        ]
    )
    assert selected["selected_candidate_id"] == "fast"
    assert stripped["selected_candidate_id"] == "fast"
    assert selected["selector_uses_oracle_truth"] is False


def test_paper_pilots_are_paired_and_neural_promotion_is_predeclared() -> None:
    standard = standard_pilot_configs(seed=123)
    assert [config.standard_num_iterations for config in standard] == [30, 100, 300]
    assert {config.seed for config in standard} == {123}
    screen = neural_screen_configs(seed=456)
    assert len(screen) == 4
    assert {config.seed for config in screen} == {456}
    selection = {
        "candidates": [
            {"candidate_id": optimizer_candidate_id(config), "eligible": True,
             "runtime_sec_mean": float(index)}
            for index, config in enumerate(screen)
        ]
    }
    promoted = promoted_neural_ids(selection)
    finalists = neural_finalist_configs(promoted, seed=789)
    assert len(finalists) == 2
    assert {config.seed for config in finalists} == {789}
    assert all(config.optimizer_stability_restarts == 3 for config in finalists)


@pytest.mark.parametrize(
    "stage",
    [
        "pilot",
        "linear-verify",
        "standard-pilot",
        "standard-extension",
        "neural-screen",
        "neural-final",
    ],
)
def test_cli_routes_every_pilot_stage_without_oracle_selection(
    stage: str, tmp_path: Path, monkeypatch
) -> None:
    config = CoverageRunConfig(
        n=20,
        repetitions=1,
        mass_grid=(0.5,),
        methods=("clipped_fori",),
        clipped_num_iterations=1,
        clipped_gate_steps=1,
        clipped_ratio_steps=1,
    )
    monkeypatch.setattr(coverage_module, "run_coverage_benchmark", lambda *a, **k: [])
    monkeypatch.setattr(coverage_module, "write_benchmark_artifacts", lambda *a, **k: {})
    monkeypatch.setattr(coverage_module, "pilot_optimizer_configs", lambda seed: [config])
    monkeypatch.setattr(coverage_module, "linear_verification_config", lambda: config)
    monkeypatch.setattr(coverage_module, "standard_pilot_configs", lambda seed: [config])
    monkeypatch.setattr(coverage_module, "standard_extension_config", lambda seed: config)
    monkeypatch.setattr(coverage_module, "neural_screen_configs", lambda seed: [config])
    monkeypatch.setattr(
        coverage_module, "neural_finalist_configs", lambda ids, seed: [config]
    )
    selection = {
        "selected_candidate_id": "selected",
        "selector_uses_oracle_truth": False,
        "candidates": [],
    }
    monkeypatch.setattr(
        coverage_module, "select_optimizer_pilot", lambda rows: dict(selection)
    )
    monkeypatch.setattr(
        coverage_module, "select_standard_pilot", lambda rows: dict(selection)
    )
    monkeypatch.setattr(
        coverage_module, "promoted_neural_ids", lambda selected: ["a", "b"]
    )
    args = [
        "--stage",
        stage,
        "--output-dir",
        str(tmp_path / stage),
        "--no-plot",
    ]
    if stage == "neural-final":
        screen = tmp_path / "screen.json"
        screen.write_text(json.dumps(selection))
        args.extend(["--screen-selection", str(screen)])
    assert coverage_module.main(args) == 0


def test_confirmatory_cli_requires_freeze_on_clean_revision(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        coverage_module,
        "repository_metadata",
        lambda path: {"git_dirty": False, "git_revision": "revision"},
    )
    with pytest.raises(SystemExit):
        coverage_module.main(
            [
                "--stage",
                "confirmatory",
                "--output-dir",
                str(tmp_path),
                "--no-plot",
            ]
        )
    with pytest.raises(ValueError, match="reward"):
        SharedHubTruth(
            gamma=0.9,
            tau_upper=10.0,
            q_by_context=np.array([0.1]),
            alpha_by_context=np.array([1.0]),
            context_probability=np.array([1.0]),
        ).values("bad")


def test_artifact_repository_and_cell_validation(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    tracked = repository / "tracked.txt"
    tracked.write_text("baseline\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-q",
            "-m",
            "baseline",
        ],
        cwd=repository,
        check=True,
    )
    clean = repository_metadata(tracked)
    assert clean["git_dirty"] is False
    tracked.write_text("changed\n")
    (repository / "untracked.txt").write_text("untracked\n")
    dirty = repository_metadata(repository)
    assert dirty["git_dirty"] is True
    assert dirty["git_diff_hash"]
    torch_metadata = environment_metadata(include_torch=True)
    assert torch_metadata["torch"]["requested"] is True

    cell = tmp_path / "cell.json"
    write_cell(cell, cell_id="abc", config_hash="cfg", rows=[{"status": "ok"}])
    assert read_cell(cell, cell_id="abc", config_hash="cfg")[0]["status"] == "ok"
    with pytest.raises(ValueError, match="cell id"):
        read_cell(cell, cell_id="wrong", config_hash="cfg")
    with pytest.raises(ValueError, match="configuration"):
        read_cell(cell, cell_id="abc", config_hash="wrong")
    cell.write_text("{}")
    with pytest.raises(ValueError, match="schema"):
        read_cell(cell, cell_id="abc", config_hash="cfg")
