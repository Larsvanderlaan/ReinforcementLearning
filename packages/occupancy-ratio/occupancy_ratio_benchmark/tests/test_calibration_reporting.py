from __future__ import annotations

import pytest

from occupancy_ratio_benchmark.calibration_reporting import (
    cluster_bootstrap_interval,
    pair_candidate_rows,
)


def _row(candidate, seed, calibration, value):
    return {
        "study_id": "study",
        "cell_id": "cell",
        "sample_size": 100,
        "gamma": 0.9,
        "seed": seed,
        "estimator_id": "base",
        "score_distortion": None,
        "candidate_id": candidate,
        "status": "ok",
        "benchmark_family": "random_tabular",
        "cross_moment_signed_squared_error": calibration,
        "policy_value_absolute_error": value,
        "policy_value_safety_margin": 0.2,
        "ratio_mse_untruncated": value,
        "ratio_generalized_kl_extended": value,
        "pava_converged": candidate.endswith("pava_pointwise_median"),
    }


def test_pairing_computes_frozen_deltas_and_margin() -> None:
    rows = []
    for seed in range(3):
        rows.extend(
            [
                _row("scalar_normalized_pointwise_median", seed, 1.0, 0.5),
                _row("pava_pointwise_median", seed, 0.5, 0.4),
            ]
        )
    pairs, duplicates = pair_candidate_rows(rows, control_id="scalar_normalized_pointwise_median")
    assert duplicates == 0
    assert len(pairs) == 3
    assert pairs[0]["is_learned_estimator"] is True
    assert pairs[0]["calibration_delta"] == pytest.approx(-0.5)
    assert pairs[0]["value_delta_minus_margin"] == pytest.approx(-0.3)
    assert pairs[0]["ratio_mse_delta"] == pytest.approx(-0.1)


def test_pairing_uses_explicit_full_data_control() -> None:
    rows = [
        _row("full_data_raw", 0, 2.0, 0.8),
        _row("scalar_normalized_pointwise_median", 0, 1.0, 0.5),
        _row("pava_pointwise_median", 0, 0.5, 0.4),
    ]

    pairs, duplicates = pair_candidate_rows(rows, control_id="full_data_raw")

    assert duplicates == 0
    assert len(pairs) == 1
    assert pairs[0]["control_id"] == "full_data_raw"
    assert pairs[0]["calibration_control"] == pytest.approx(2.0)
    assert pairs[0]["calibration_delta"] == pytest.approx(-1.5)
    assert pairs[0]["value_delta"] == pytest.approx(-0.4)


def test_pairing_rejects_ambiguous_candidate_ids() -> None:
    with pytest.raises(ValueError, match="distinct"):
        pair_candidate_rows([], control_id="pava_pointwise_median")


def test_cluster_bootstrap_is_deterministic_and_requires_two_clusters() -> None:
    rows = [
        {"cluster_id": "a", "delta": -1.0},
        {"cluster_id": "b", "delta": -2.0},
        {"cluster_id": "c", "delta": -3.0},
    ]
    first = cluster_bootstrap_interval(rows, value_key="delta", repetitions=1_000, seed=9)
    second = cluster_bootstrap_interval(rows, value_key="delta", repetitions=1_000, seed=9)
    assert first == second
    assert first["ci95_high"] < 0.0
    insufficient = cluster_bootstrap_interval(rows[:1], value_key="delta", repetitions=10, seed=1)
    assert insufficient["status"] == "insufficient_clusters"
