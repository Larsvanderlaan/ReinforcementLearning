"""Public recursively clipped KL-FORI estimator facade."""

from occupancy_ratio._clipped_kl_fori_impl import (
    ClippedKLFORIConvergenceError,
    ClippedKLFORIConfig,
    ClippedKLFORIModel,
    fit_clipped_kl_fori,
    fit_clipped_kl_fori_neural,
)

ClippedKLFORIConfig.__module__ = __name__
ClippedKLFORIModel.__module__ = __name__

__all__ = [
    "ClippedKLFORIConvergenceError",
    "ClippedKLFORIConfig",
    "ClippedKLFORIModel",
    "fit_clipped_kl_fori",
    "fit_clipped_kl_fori_neural",
]
