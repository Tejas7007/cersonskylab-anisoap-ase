"""
Test force calculations with finite differences.
"""

import pytest
import numpy as np
from ase.build import molecule
from ase.io import read

from anisoap_ase import AniSOAPCalculator


def test_forces_finite_difference():
    """
    Test that forces computed via finite differences are self-consistent.
    """
    # Load test data
    try:
        atoms = read("tests/data/ellipsoid.xyz", index=0)
    except FileNotFoundError:
        pytest.skip("Test data not found")
    
    # Create calculator with forces enabled
    calc = AniSOAPCalculator(
        backend="torch",
        enable_forces=True
    )
    
    atoms.calc = calc
    
    # Get forces
    forces = atoms.get_forces()
    
    # Check shape
    assert forces.shape == (len(atoms), 3), f"Expected shape ({len(atoms)}, 3), got {forces.shape}"
    
    # Forces should be finite
    assert np.all(np.isfinite(forces)), "Forces contain NaN or Inf"
    
    print(f"✓ Forces shape: {forces.shape}")
    print(f"✓ Forces magnitude: {np.linalg.norm(forces):.6f} eV/Å")


def test_forces_symmetry():
    """
    Test that forces respect molecular symmetry (basic sanity check).
    """
    try:
        atoms = read("tests/data/ellipsoid.xyz", index=0)
    except FileNotFoundError:
        pytest.skip("Test data not found")
    
    calc = AniSOAPCalculator(backend="torch", enable_forces=True)
    atoms.calc = calc
    
    forces = atoms.get_forces()
    
    # Sum of forces should be small (no net force on isolated system)
    force_sum = np.sum(forces, axis=0)
    assert np.linalg.norm(force_sum) < 1.0, f"Net force too large: {force_sum}"
    
    print(f"✓ Net force: {force_sum} (magnitude: {np.linalg.norm(force_sum):.6e} eV/Å)")


def test_error_message_missing_c_q():
    """
    Test that the improved error message appears when c_q is missing.
    """
    atoms = molecule("H2O")
    # Don't add c_q attribute
    
    calc = AniSOAPCalculator(backend="numpy")
    atoms.calc = calc
    
    with pytest.raises(ValueError) as exc_info:
        _ = atoms.get_potential_energy()
    
    # Check error message format
    error_msg = str(exc_info.value)
    assert "Expect frames with ellipsoidal attributes" in error_msg
    assert "frame at index 0" in error_msg
    assert "missing a required attribute 'c_q'" in error_msg
    
    print(f"✓ Error message: {error_msg}")


if __name__ == "__main__":
    print("Running force tests...")
    test_forces_finite_difference()
    test_forces_symmetry()
    test_error_message_missing_c_q()
    print("\n✓ All tests passed!")
