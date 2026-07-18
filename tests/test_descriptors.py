from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from anisoap_ase import AniSOAPDescriptor


def oriented_atoms() -> Atoms:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.set_array("c_q", np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)))
    return atoms


def test_default_diameters_are_added_to_a_copy() -> None:
    atoms = oriented_atoms()
    descriptor = AniSOAPDescriptor(default_diameters=(4.0, 4.0, 0.5))
    prepared = descriptor.prepare_atoms(atoms)

    for key, expected in zip(
        ("c_diameter[1]", "c_diameter[2]", "c_diameter[3]"),
        (4.0, 4.0, 0.5),
        strict=True,
    ):
        np.testing.assert_allclose(prepared.arrays[key], expected)
        assert key not in atoms.arrays


def test_defaults_preserve_supplied_diameters() -> None:
    atoms = oriented_atoms()
    atoms.set_array("c_diameter[1]", np.full(2, 7.0))
    prepared = AniSOAPDescriptor(
        default_diameters=(4.0, 4.0, 0.5)
    ).prepare_atoms(atoms)
    np.testing.assert_allclose(prepared.arrays["c_diameter[1]"], 7.0)
    np.testing.assert_allclose(prepared.arrays["c_diameter[2]"], 4.0)
    np.testing.assert_allclose(prepared.arrays["c_diameter[3]"], 0.5)


def test_missing_orientation_is_reported() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    descriptor = AniSOAPDescriptor(default_diameters=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="per-particle orientations"):
        descriptor.prepare_atoms(atoms)


def test_missing_diameters_are_not_silently_invented() -> None:
    descriptor = AniSOAPDescriptor()
    with pytest.raises(ValueError, match="Missing arrays"):
        descriptor.prepare_atoms(oriented_atoms())


def test_invalid_quaternion_shape_is_reported() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.set_array("c_q", np.ones((1, 3)))
    descriptor = AniSOAPDescriptor(default_diameters=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="must have shape"):
        descriptor.prepare_atoms(atoms)


def test_nonpositive_diameter_is_rejected() -> None:
    atoms = oriented_atoms()
    atoms.set_array("c_diameter[1]", np.ones(2))
    atoms.set_array("c_diameter[2]", np.ones(2))
    atoms.set_array("c_diameter[3]", np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="positive finite"):
        AniSOAPDescriptor().prepare_atoms(atoms)
