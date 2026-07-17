"""Independent analytic box-constrained oracle for shared-hub clipping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SharedHubBoxOracle:
    """Population ratios and values for the tabular shared-hub construction."""

    gamma: float
    tau_lower: float
    tau_upper: float
    q: Array
    context_probability: Array
    target_ratio: Array
    bypass_ratio: Array
    hub_ratio: Array

    @property
    def recursive_mass(self) -> float:
        return float(
            self.context_probability @ np.minimum(1.0, self.tau_upper * self.q)
        )

    @property
    def projected_mass(self) -> float:
        return float(self.context_probability @ self.hub_ratio)

    @property
    def posthoc_mass(self) -> float:
        return float(self.gamma + (1.0 - self.gamma) * self.recursive_mass)

    def value(self, reward: str, *, projected: bool = False) -> float:
        """Return the recursive or lower-envelope projected stopped value."""
        mass = self.projected_mass if projected else self.recursive_mass
        if reward == "constant":
            return mass
        if reward == "hub":
            return float(self.gamma * mass)
        if reward == "gate":
            target_mass = self.q * self.target_ratio
            return float((1.0 - self.gamma) * (self.context_probability @ target_mass))
        raise ValueError("reward must be 'constant', 'hub', or 'gate'")


def shared_hub_box_oracle(
    *,
    q: Array,
    gamma: float,
    tau_lower: float,
    tau_upper: float,
    context_probability: Array | None = None,
) -> SharedHubBoxOracle:
    """Solve the population shared-hub fixed point under ratio box constraints.

    The bypass floor propagates recursively into the hub.  Thus the projected
    hub ratio is ``alpha + (1-q) * tau_lower``, capped at ``tau_upper``; this is
    not a post-hoc lower clipping of the unbounded recursive oracle.
    """
    q_value = np.asarray(q, dtype=np.float64).reshape(-1)
    if q_value.size == 0 or np.any((q_value < 0.0) | (q_value > 1.0)):
        raise ValueError("q must be a nonempty vector in [0, 1]")
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must lie in [0, 1)")
    if not (0.0 < tau_lower <= 1.0 <= tau_upper):
        raise ValueError("invalid ratio envelope")
    if context_probability is None:
        probability = np.full(q_value.size, 1.0 / q_value.size)
    else:
        probability = np.asarray(context_probability, dtype=np.float64).reshape(-1)
        if probability.shape != q_value.shape or np.any(probability < 0.0):
            raise ValueError("context_probability must match q and be nonnegative")
        total = float(np.sum(probability))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("context_probability must have positive finite mass")
        probability = probability / total
    alpha = np.minimum(1.0, tau_upper * q_value)
    target_ratio = np.where(
        q_value > 0.0,
        np.minimum(1.0 / np.maximum(q_value, np.finfo(float).tiny), tau_upper),
        tau_upper,
    )
    bypass_ratio = np.full_like(q_value, tau_lower)
    hub_ratio = np.minimum(tau_upper, alpha + (1.0 - q_value) * tau_lower)
    return SharedHubBoxOracle(
        gamma=float(gamma),
        tau_lower=float(tau_lower),
        tau_upper=float(tau_upper),
        q=q_value,
        context_probability=probability,
        target_ratio=target_ratio,
        bypass_ratio=bypass_ratio,
        hub_ratio=hub_ratio,
    )


__all__ = ["SharedHubBoxOracle", "shared_hub_box_oracle"]
