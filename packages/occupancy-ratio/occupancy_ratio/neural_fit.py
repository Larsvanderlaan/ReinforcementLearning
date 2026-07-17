"""Neural estimator fit and CV entrypoints."""

from occupancy_ratio._neural_impl import (
    fit_discounted_occupancy_ratio_neural as _fit_regression_fori_neural,
    tune_discounted_occupancy_ratio_neural_cv,
)

fit_regression_fori_neural = _fit_regression_fori_neural
fit_discounted_occupancy_ratio_neural = fit_regression_fori_neural

__all__ = [
    "fit_regression_fori_neural",
    "fit_discounted_occupancy_ratio_neural",
    "tune_discounted_occupancy_ratio_neural_cv",
]
