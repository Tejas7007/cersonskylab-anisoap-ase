from ase.build import molecule
from anisoap_ase import AniSOAPCalculator
from anisoap_ase.descriptors import anisoap_stub_descriptor
from anisoap_ase.model import linear_stub_model

atoms = molecule("H2O")
atoms.calc = AniSOAPCalculator(
    descriptor_fn=anisoap_stub_descriptor,
    model=linear_stub_model,
    cache_results=True,
)
print("Energy (eV):", atoms.get_potential_energy())
