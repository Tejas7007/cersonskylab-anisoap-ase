from __future__ import annotations
from typing import Callable, Optional, Tuple
import numpy as np
from ase.calculators.calculator import (
    Calculator, all_changes, PropertyNotImplementedError, CalculatorSetupError
)

DescriptorFn = Callable[..., "np.ndarray | float | int"]
ModelFn = Callable[..., float]

class AniSOAPCalculator(Calculator):
    """
    ASE-compatible calculator that bridges AniSOAP → ASE.

    Status: energy-only
    - Units: energy returned/stored in eV (ASE convention).
    - Future: add "forces", "stress" once backend exposes them.

    Parameters
    ----------
    backend : {"numpy","torch"}, optional
        Hint for downstream descriptor/model (not enforced here).
    descriptor_fn : callable(atoms) -> any, optional
        Function that returns a descriptor object/array for the given Atoms.
    model : callable(descriptor) -> energy_eV, optional
        Function that maps descriptor -> scalar energy in eV.
    cache_results : bool, default True
        If True, skip recompute when atoms numbers+positions unchanged.
    energy_units_to_eV : float, default 1.0
        Multiply model energy by this factor before storing in results.
    length_units_to_A : float, default 1.0
        Reserved for future forces/stress conversions.
    label : str | None
        Optional ASE label.
    """

    implemented_properties = ["energy"]

    def __init__(
        self,
        backend: str = "numpy",
        descriptor_fn: Optional[DescriptorFn] = None,
        model: Optional[ModelFn] = None,
        cache_results: bool = True,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        label: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(label=label, **kwargs)
        self.backend = backend
        self.descriptor_fn = descriptor_fn
        self.model = model
        self.cache_results = cache_results
        self.energy_units_to_eV = float(energy_units_to_eV)
        self.length_units_to_A = float(length_units_to_A)
        self._last_state: Optional[Tuple[Tuple[int, ...], bytes]] = None

        # Help ASE parameter hashing/caching
        self.parameters.update(
            dict(
                backend=backend,
                cache_results=cache_results,
                energy_units_to_eV=self.energy_units_to_eV,
                length_units_to_A=self.length_units_to_A,
            )
        )

    # ---- ASE change detection polish (like MACE does for .info) ----
    def check_state(self, atoms, tol: float = 1e-15):
        state = super().check_state(atoms, tol=tol)
        if not state:
            if getattr(self.atoms, "info", None) != getattr(atoms, "info", None):
                state.append("info")
        return state

    # ---- internal helpers ----
    @staticmethod
    def _state(atoms) -> Tuple[Tuple[int, ...], bytes]:
        numbers = tuple(int(z) for z in atoms.numbers)
        pos = atoms.get_positions(wrap=False)
        return numbers, np.ascontiguousarray(pos, dtype=np.float64).tobytes()

    def _compute_descriptor(self, atoms):
        if self.descriptor_fn is None:
            # Minimal stub so the example runs; replace with real AniSOAP call.
            return np.asarray(atoms.numbers, dtype=np.float64)
        return self.descriptor_fn(atoms)

    def _compute_energy_eV(self, desc) -> float:
        if self.model is None:
            # Minimal stub model: scaled sum in eV (demo only).
            return float(np.sum(desc)) * 1.0e-3
        return float(self.model(desc))

    # ---- main ASE entrypoint ----
    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # Reject unsupported properties cleanly
        for p in properties:
            if p not in self.implemented_properties:
                raise PropertyNotImplementedError(p)

        # Cache: skip work if geometry unchanged
        state = self._state(atoms)
        if self.cache_results and self._last_state == state and hasattr(self, "results"):
            return  # results already set

        try:
            desc = self._compute_descriptor(atoms)
            energy_ev = self._compute_energy_eV(desc) * self.energy_units_to_eV
        except Exception as exc:
            raise CalculatorSetupError(
                f"AniSOAPCalculator failed to compute energy: {exc}"
            ) from exc

        self.results = {}
        if "energy" in properties:
            self.results["energy"] = float(energy_ev)

        if self.cache_results:
            self._last_state = state

    # explicit passthrough; ASE will call this either way
    def get_potential_energy(self, atoms=None, force_consistent: bool = False) -> float:
        return super().get_potential_energy(atoms=atoms, force_consistent=force_consistent)
