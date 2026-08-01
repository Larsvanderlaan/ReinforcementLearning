from __future__ import annotations

from occupancy_ratio_benchmark.calibration_run import _fold_clamp_diagnostics


def test_fold_clamp_diagnostics_aggregate_counts_bounds_and_fraction() -> None:
    dependencies = [
        {
            "metadata": {
                "base_upper_cap_enabled": True,
                "base_prediction_lower_clamp_count": 2,
                "base_prediction_upper_clamp_count": 5,
                "base_prediction_log_lower_bound": -4.0,
                "base_prediction_log_upper_bound": 6.0,
            }
        },
        {
            "metadata": {
                "base_upper_cap_enabled": True,
                "base_prediction_lower_clamp_count": 1,
                "base_prediction_upper_clamp_count": 4,
                "base_prediction_log_lower_bound": -3.0,
                "base_prediction_log_upper_bound": 8.0,
            }
        },
    ]
    result = _fold_clamp_diagnostics(
        dependencies,
        source_rows=10,
        initial_rows=5,
    )
    assert result["base_prediction_clamp_enabled_fold_count"] == 2
    assert result["base_prediction_lower_clamp_count_across_folds"] == 3
    assert result["base_prediction_upper_clamp_count_across_folds"] == 9
    assert result["base_prediction_clamp_fraction_across_folds"] == 12 / 50
    assert result["base_prediction_log_lower_bound_min_across_folds"] == -4.0
    assert result["base_prediction_log_lower_bound_max_across_folds"] == -3.0
    assert result["base_prediction_log_upper_bound_min_across_folds"] == 6.0
    assert result["base_prediction_log_upper_bound_max_across_folds"] == 8.0


def test_fold_clamp_diagnostics_are_zero_for_uncapped_estimators() -> None:
    result = _fold_clamp_diagnostics(
        [{"metadata": {"base_upper_cap_enabled": False}}],
        source_rows=10,
        initial_rows=5,
    )
    assert result["base_prediction_clamp_enabled_fold_count"] == 0
    assert result["base_prediction_clamp_fraction_across_folds"] == 0.0
    assert result["base_prediction_log_upper_bound_max_across_folds"] is None
