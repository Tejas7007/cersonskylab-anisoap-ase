import numpy as np
import pytest
from ase import Atoms
from anisoap_ase import AniSOAPCalculator
from ase.calculators.calculator import PropertyNotImplementedError

def test_energy_constant_mock():
    def desc_fn(a): return np.array([1.0, 2.0])
    def model(d): return 0.5
    a = Atoms("He", positions=[[0,0,0]])
    a.calc = AniSOAPCalculator(descriptor_fn=desc_fn, model=model)
    assert abs(a.get_potential_energy() - 0.5) < 1e-12

def test_cache_reuse():
    calls = {"n": 0}
    def desc_fn(a):
        calls["n"] += 1
        return np.array([1.0])
    def model(d): return float(d.sum())

    a = Atoms("H", positions=[[0,0,0]])
    a.calc = AniSOAPCalculator(descriptor_fn=desc_fn, model=model, cache_results=True)
    e1 = a.get_potential_energy()
    e2 = a.get_potential_energy()
    assert e1 == e2
    assert calls["n"] == 1

def test_property_not_implemented():
    a = Atoms("H", positions=[[0,0,0]])
    a.calc = AniSOAPCalculator()
    with pytest.raises(PropertyNotImplementedError):
        a.calc.calculate(a, properties=("forces",))
