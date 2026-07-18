"""Descriptor adapters used by AniSOAP-ASE."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ase import Atoms

DEFAULT_HYPERS: dict[str, Any] = {
    "max_angular": 9,
    "max_radial": 6,
    "radial_basis_name": "gto",
    "subtract_center_contribution": True,
    "rotation_type": "quaternion",
    "rotation_key": "c_q",
    "cutoff_radius": 7.0,
    "radial_gaussian_width": 1.5,
    "basis_rcond": 1e-8,
    "basis_tol": 1e-3,
}

_DIAMETER_KEYS = ("c_diameter[1]", "c_diameter[2]", "c_diameter[3]")


@dataclass
class AniSOAPDescriptor:
    """Callable wrapper around AniSOAP's ellipsoidal density projection."""

    hypers: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_HYPERS))
    default_diameters: tuple[float, float, float] | None = None
    _projection: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.hypers = dict(self.hypers)
        if self.default_diameters is not None:
            diameters = tuple(float(value) for value in self.default_diameters)
            if len(diameters) != 3 or any(value <= 0 for value in diameters):
                raise ValueError(
                    "default_diameters must contain three positive values."
                )
            self.default_diameters = diameters

    @property
    def orientation_key(self) -> str:
        return str(self.hypers.get("rotation_key", "c_q"))

    def prepare_atoms(self, atoms: Atoms) -> Atoms:
        """Validate ellipsoidal metadata on a copy of an ASE structure."""
        prepared = atoms.copy()
        atom_count = len(prepared)

        if self.orientation_key not in prepared.arrays:
            raise ValueError(
                "AniSOAP requires per-particle orientations in "
                f"'{self.orientation_key}'."
            )

        orientations = np.asarray(prepared.arrays[self.orientation_key], dtype=float)
        if orientations.shape != (atom_count, 4):
            raise ValueError(
                f"'{self.orientation_key}' must have shape ({atom_count}, 4); "
                f"received {orientations.shape}."
            )
        if not np.all(np.isfinite(orientations)):
            raise ValueError(f"'{self.orientation_key}' contains non-finite values.")
        if np.any(np.linalg.norm(orientations, axis=1) == 0):
            raise ValueError(f"'{self.orientation_key}' contains a zero quaternion.")

        missing = [key for key in _DIAMETER_KEYS if key not in prepared.arrays]
        if missing:
            if self.default_diameters is None:
                joined = ", ".join(missing)
                raise ValueError(
                    "AniSOAP requires ellipsoid diameters. Missing arrays: " + joined
                )
            defaults = dict(zip(_DIAMETER_KEYS, self.default_diameters, strict=True))
            for key in missing:
                prepared.set_array(
                    key, np.full(atom_count, defaults[key], dtype=float)
                )

        for key in _DIAMETER_KEYS:
            values = np.asarray(prepared.arrays[key], dtype=float)
            if values.shape != (atom_count,):
                raise ValueError(
                    f"'{key}' must have shape ({atom_count},); received {values.shape}."
                )
            if not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(f"'{key}' must contain positive finite values.")

        return prepared

    def _get_projection(self) -> Any:
        if self._projection is None:
            try:
                from anisoap.representations import EllipsoidalDensityProjection
            except ImportError as exc:
                raise ImportError(
                    "AniSOAP is required for the default descriptor. Install the "
                    "pinned upstream dependency with 'bash scripts/bootstrap.sh'."
                ) from exc
            self._projection = EllipsoidalDensityProjection(**self.hypers)
        return self._projection

    def __call__(self, atoms: Atoms) -> np.ndarray:
        prepared = self.prepare_atoms(atoms)
        output = self._get_projection().power_spectrum([prepared])
        first = output[0]
        values = getattr(first, "values", first)
        features = np.asarray(values, dtype=float).reshape(-1)
        if features.size == 0:
            raise ValueError("AniSOAP returned an empty descriptor.")
        if not np.all(np.isfinite(features)):
            raise ValueError("AniSOAP returned non-finite descriptor values.")
        return features
