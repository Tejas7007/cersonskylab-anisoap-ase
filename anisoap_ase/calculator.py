from __future__ import annotations
from ase.calculators.calculator import Calculator, all_changes

class AniSOAPCalculator(Calculator):
    """Prototype ASE calculator for AniSOAP (energy-only version)."""
    implemented_properties = ["energy"]

    def __init__(self, backend: str = "numpy", **kwargs):
        super().__init__(**kwargs)
        self.backend = backend

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        # TODO: hook into actual AniSOAP backend
        self.results["energy"] = 0.0
