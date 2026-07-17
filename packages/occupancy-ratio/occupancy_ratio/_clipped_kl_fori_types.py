"""Typed internal records for recursively clipped KL-FORI."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class InnerOptimizerResult:
    """Result from one gate or ratio inner optimization."""

    parameters: Any
    objective: float
    gradient_norm: float
    steps_completed: int
    termination_reason: str


@dataclass
class FitResult(Mapping[str, Any]):
    """Backend result with a temporary mapping view for orchestration."""

    ratio_coef: Array
    gate_coef: Array
    ratio_neural_state_dict: dict[str, Array]
    gate_neural_state_dict: dict[str, Array]
    history: list[dict[str, Any]]
    weights_ref: Array
    gate_ref: Array
    iterations_completed: int
    neural_deterministic_enabled: bool
    neural_deterministic_error: str

    def __getitem__(self, key: str) -> Any:
        if key not in self.__dataclass_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)


@dataclass(frozen=True)
class IterationRecord:
    """Typed wrapper whose public representation remains JSON-compatible."""

    values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass(frozen=True)
class FitDiagnostics:
    """Typed wrapper for the public diagnostics mapping."""

    values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


__all__ = ["FitDiagnostics", "FitResult", "InnerOptimizerResult", "IterationRecord"]
