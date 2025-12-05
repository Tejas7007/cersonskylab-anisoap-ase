import numpy as np
import pytest
from ase import Atoms

from anisoap_ase.calculator import AniSOAPCalculator


@pytest.mark.xfail(reason="Autodiff forces path not yet fully integrated", strict=False)
def test_autodiff_forces_shape():
    # Very small system just to check the plumbing works
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.75, 0, 0]])
    calc = AniSOAPCalculator(lr_path="lr.pkl")
    atoms.set_calculator(calc)

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert np.isfinite(energy)
    assert forces.shape == (2, 3)
    assert np.all(np.isfinite(forces))
