"""Public KL-FORI estimator facade."""

from occupancy_ratio._kl_fori_impl import (
    KLFORIConfig,
    KLFORIModel,
    fit_kl_fori,
    fit_kl_fori_boosting,
    fit_kl_fori_neural,
)

KLFORIConfig.__module__ = __name__
KLFORIModel.__module__ = __name__

__all__ = [
    "KLFORIConfig",
    "KLFORIModel",
    "fit_kl_fori",
    "fit_kl_fori_boosting",
    "fit_kl_fori_neural",
]
