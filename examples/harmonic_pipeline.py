"""Minimal descriptor-to-model ASE example with finite-difference forces."""

import numpy as np
from ase import Atoms

from anisoap_ase import AniSOAPCalculator


def position_descriptor(atoms: Atoms) -> np.ndarray:
    return atoms.get_positions().reshape(-1)


def harmonic_energy(features: np.ndarray) -> float:
    return 0.5 * float(features @ features)


atoms = Atoms("H2", positions=[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
atoms.calc = AniSOAPCalculator(
    descriptor=position_descriptor,
    model=harmonic_energy,
    force_method="central",
)

print("Energy (eV):", atoms.get_potential_energy())
print("Forces (eV/Å):")
print(atoms.get_forces())
