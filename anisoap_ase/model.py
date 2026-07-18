"""Energy-model helpers for AniSOAP-ASE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def evaluate_model(model: Any, features: np.ndarray) -> float:
    """Evaluate a callable or scikit-learn-style model on one feature vector."""
    vector = np.asarray(features, dtype=float).reshape(-1)
    if vector.size == 0:
        raise ValueError("Descriptor returned an empty feature vector.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Descriptor returned non-finite values.")

    if hasattr(model, "predict"):
        raw_energy = model.predict(vector.reshape(1, -1))
    elif callable(model):
        raw_energy = model(vector)
    else:
        raise TypeError("model must be callable or expose a predict method.")

    values = np.asarray(raw_energy, dtype=float).reshape(-1)
    if values.size != 1:
        raise ValueError(
            "Energy model must return exactly one scalar for a single ASE structure."
        )

    energy = float(values[0])
    if not np.isfinite(energy):
        raise ValueError("Energy model returned a non-finite value.")
    return energy


@dataclass(frozen=True)
class LinearModel:
    """Small, pickle-free linear energy model for examples and deployment."""

    coefficients: np.ndarray
    intercept: float = 0.0

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=float).reshape(-1)
        if coefficients.size == 0:
            raise ValueError("coefficients must contain at least one value.")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients must be finite.")
        if not np.isfinite(self.intercept):
            raise ValueError("intercept must be finite.")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", float(self.intercept))

    def __call__(self, features: np.ndarray) -> float:
        vector = np.asarray(features, dtype=float).reshape(-1)
        if vector.shape != self.coefficients.shape:
            raise ValueError(
                "Feature length does not match the linear model: "
                f"expected {self.coefficients.size}, received {vector.size}."
            )
        return float(vector @ self.coefficients + self.intercept)

    def save(self, path: str | Path) -> None:
        """Store model parameters in NumPy's non-executable NPZ format."""
        np.savez(
            Path(path),
            coefficients=self.coefficients,
            intercept=np.array(self.intercept),
        )

    @classmethod
    def load(cls, path: str | Path) -> LinearModel:
        """Load model parameters written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                coefficients=np.asarray(data["coefficients"], dtype=float),
                intercept=float(np.asarray(data["intercept"]).reshape(-1)[0]),
            )
