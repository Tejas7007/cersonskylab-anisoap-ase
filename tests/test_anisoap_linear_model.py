import numpy as np
from pathlib import Path

from ase.io import read

from anisoap_ase.calculator import AniSOAPCalculator


DATA_DIR = Path(__file__).resolve().parent / "data"
ELLIPSOIDS_XYZ = DATA_DIR / "ellipsoids.xyz"


def test_anisoap_linear_model_runs_on_ellipsoids():
    """
    Basic functionality test:

    - Load several ellipsoidal frames from ellipsoids.xyz
    - Attach AniSOAPCalculator
    - Ensure get_potential_energy() runs without error
      and returns finite energies.

    This verifies that the AniSOAP → linear model (lr.pkl)
    pipeline is correctly wired for the ellipsoid xyz files.
    """
    assert ELLIPSOIDS_XYZ.is_file(), f"Missing test data file: {ELLIPSOIDS_XYZ}"

    # Load all frames from the ellipsoids.xyz test file
    frames = read(str(ELLIPSOIDS_XYZ), ":")

    # Keep the test light: just the first few frames
    frames = frames[:5]

    for atoms in frames:
        # Attach the AniSOAP ASE calculator (which internally
        # computes AniSOAP descriptors and calls the linear model)
        calc = AniSOAPCalculator()
        atoms.calc = calc

        energy = atoms.get_potential_energy()

        # Sanity checks: energy must be a finite float
        assert isinstance(energy, float)
        assert np.isfinite(energy)

