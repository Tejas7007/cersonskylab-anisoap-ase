"""ASE calculator adapter for AniSOAP descriptor-based energy models."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)

from .descriptors import AniSOAPDescriptor
from .model import evaluate_model

Descriptor = Callable[[Atoms], Any]


class AniSOAPCalculator(Calculator):
    """Connect an AniSOAP descriptor and scalar energy model to ASE.

    Energy evaluation is delegated to the supplied model. Forces are optional
    and, when requested, are computed by central finite differences of the
    complete energy pipeline. This class does not claim analytical or PyTorch
    autodifferentiation through AniSOAP.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model: Any,
        descriptor: Descriptor | None = None,
        *,
        force_method: str | None = None,
        finite_difference_step: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if model is None:
            raise ValueError("model is required.")
        if force_method not in (None, "central"):
            raise ValueError("force_method must be None or 'central'.")
        if not np.isfinite(finite_difference_step) or finite_difference_step <= 0:
            raise ValueError("finite_difference_step must be positive and finite.")

        self.model = model
        self.descriptor = descriptor if descriptor is not None else AniSOAPDescriptor()
        self.force_method = force_method
        self.finite_difference_step = float(finite_difference_step)

    def _evaluate_energy(self, atoms: Atoms) -> float:
        features = np.asarray(self.descriptor(atoms), dtype=float).reshape(-1)
        return evaluate_model(self.model, features)

    def _central_forces(self, atoms: Atoms) -> np.ndarray:
        step = self.finite_difference_step
        forces = np.empty((len(atoms), 3), dtype=float)

        for atom_index in range(len(atoms)):
            for axis in range(3):
                plus = atoms.copy()
                minus = atoms.copy()
                plus.calc = None
                minus.calc = None
                plus.positions[atom_index, axis] += step
                minus.positions[atom_index, axis] -= step
                energy_plus = self._evaluate_energy(plus)
                energy_minus = self._evaluate_energy(minus)
                forces[atom_index, axis] = -(energy_plus - energy_minus) / (2 * step)

        return forces

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Iterable[str] = ("energy",),
        system_changes: Iterable[str] = all_changes,
    ) -> None:
        requested = set(properties)
        unsupported = requested.difference(self.implemented_properties)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise PropertyNotImplementedError(f"Unsupported ASE properties: {names}")
        if "forces" in requested and self.force_method is None:
            raise PropertyNotImplementedError(
                "Forces are disabled. Set force_method='central' to enable "
                "finite-difference forces."
            )

        if atoms is None and self.atoms is None:
            raise ValueError("AniSOAPCalculator requires an ASE Atoms object.")

        super().calculate(atoms, properties, system_changes)
        working_atoms = self.atoms
        if working_atoms is None:
            raise RuntimeError("ASE did not provide a structure to calculate.")

        self.results["energy"] = self._evaluate_energy(working_atoms)
        if "forces" in requested:
            self.results["forces"] = self._central_forces(working_atoms)
