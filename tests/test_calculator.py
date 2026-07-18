from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from anisoap_ase import AniSOAPCalculator


def position_descriptor(atoms: Atoms) -> np.ndarray:
    return atoms.get_positions().reshape(-1)


def harmonic_energy(features: np.ndarray) -> float:
    return 0.5 * float(features @ features)


def test_energy_from_custom_pipeline() -> None:
    atoms = Atoms("H", positions=[[1.0, -2.0, 0.5]])
    atoms.calc = AniSOAPCalculator(
        model=harmonic_energy, descriptor=position_descriptor
    )
    assert atoms.get_potential_energy() == pytest.approx(2.625)


def test_central_difference_forces_match_harmonic_gradient() -> None:
    atoms = Atoms("H2", positions=[[1.0, -0.5, 0.25], [-0.2, 0.4, 0.8]])
    atoms.calc = AniSOAPCalculator(
        model=harmonic_energy,
        descriptor=position_descriptor,
        force_method="central",
        finite_difference_step=1e-6,
    )
    np.testing.assert_allclose(atoms.get_forces(), -atoms.positions, atol=1e-7)


def test_forces_must_be_enabled_explicitly() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = AniSOAPCalculator(
        model=harmonic_energy, descriptor=position_descriptor
    )
    with pytest.raises(PropertyNotImplementedError, match="Forces are disabled"):
        atoms.get_forces()


def test_ase_cache_reuses_results_for_unchanged_atoms() -> None:
    calls = 0

    def descriptor(atoms: Atoms) -> np.ndarray:
        nonlocal calls
        calls += 1
        return atoms.get_positions().reshape(-1)

    atoms = Atoms("He", positions=[[0.1, 0.2, 0.3]])
    atoms.calc = AniSOAPCalculator(model=harmonic_energy, descriptor=descriptor)
    first = atoms.get_potential_energy()
    second = atoms.get_potential_energy()
    assert first == second
    assert calls == 1


def test_sklearn_style_predict_interface() -> None:
    class Model:
        def predict(self, batch: np.ndarray) -> np.ndarray:
            assert batch.shape == (1, 3)
            return batch.sum(axis=1)

    atoms = Atoms("H", positions=[[1.0, 2.0, 3.0]])
    atoms.calc = AniSOAPCalculator(model=Model(), descriptor=position_descriptor)
    assert atoms.get_potential_energy() == pytest.approx(6.0)


def test_model_must_return_one_scalar() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = AniSOAPCalculator(
        model=lambda _: np.array([1.0, 2.0]),
        descriptor=position_descriptor,
    )
    with pytest.raises(ValueError, match="exactly one scalar"):
        atoms.get_potential_energy()
