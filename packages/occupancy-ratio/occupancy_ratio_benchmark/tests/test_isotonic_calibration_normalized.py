from __future__ import annotations

import numpy as np
import pytest

from occupancy_ratio import IsotonicCalibrationConfig, fit_isotonic_fori_pava


def test_normalized_pava_constant_fixed_point() -> None:
    result = fit_isotonic_fori_pava(
        source_score=np.ones(80),
        next_score=np.ones(80),
        initial_score=np.ones(20),
        gamma=0.97,
        config=IsotonicCalibrationConfig(num_iterations=20, tolerance=1e-12),
    )

    assert result.status == "ok"
    assert np.allclose(result.source_weights, 1.0)
    assert result.diagnostics["estimand"] == "normalized_discounted"
    assert abs(float(result.diagnostics["normalization_error"])) < 1e-12


def test_normalized_pava_is_monotone_and_mean_one() -> None:
    score = np.repeat(np.arange(4, dtype=np.float64), 40)
    next_score = np.roll(score, 20)
    result = fit_isotonic_fori_pava(
        source_score=score,
        next_score=next_score,
        initial_score=np.full(30, 3.0),
        gamma=0.8,
        config=IsotonicCalibrationConfig(num_iterations=500, tolerance=1e-10),
    )

    assert np.all(np.diff(result.fitted_grid_values) >= -1e-12)
    assert np.isclose(np.mean(result.source_weights), 1.0, atol=1e-12)
    assert int(result.diagnostics["monotone_violations"]) == 0


def test_constant_extrapolation_reports_out_of_support_mass() -> None:
    result = fit_isotonic_fori_pava(
        source_score=np.linspace(0.0, 1.0, 50),
        next_score=np.linspace(-1.0, 2.0, 50),
        initial_score=np.array([-3.0, 0.5, 4.0]),
        gamma=0.5,
        config=IsotonicCalibrationConfig(
            num_iterations=100,
            tolerance=1e-8,
            support_policy="constant_extrapolation",
        ),
    )

    assert float(result.diagnostics["constant_extrapolation_next_fraction"]) > 0.0
    assert float(result.diagnostics["constant_extrapolation_initial_mass"]) > 0.0
    assert np.all(np.isfinite(result.predict(np.array([-100.0, 100.0]))))


def test_normalized_solver_rejects_stopped_style_no_normalization() -> None:
    with pytest.raises(ValueError, match="normalize=True"):
        fit_isotonic_fori_pava(
            source_score=np.ones(5),
            next_score=np.ones(5),
            initial_score=np.ones(2),
            gamma=0.9,
            config=IsotonicCalibrationConfig(normalize=False),
        )
