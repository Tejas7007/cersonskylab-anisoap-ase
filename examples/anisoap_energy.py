"""Illustrative AniSOAP descriptor integration with an injected energy model."""

import numpy as np
from ase import Atoms

from anisoap_ase import AniSOAPCalculator, AniSOAPDescriptor

atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
atoms.set_array("c_q", np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)))

descriptor = AniSOAPDescriptor(default_diameters=(4.0, 4.0, 0.5))

# Replace this illustrative reduction with a trained physical energy model.
def example_model(features: np.ndarray) -> float:
    return float(features.mean())


atoms.calc = AniSOAPCalculator(descriptor=descriptor, model=example_model)
print("Illustrative scalar output:", atoms.get_potential_energy())
