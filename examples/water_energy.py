from ase.build import molecule
from anisoap_ase.calculator import AniSOAPCalculator

atoms = molecule("H2O")
atoms.calc = AniSOAPCalculator(backend="numpy")
print("Energy:", atoms.get_potential_energy())
