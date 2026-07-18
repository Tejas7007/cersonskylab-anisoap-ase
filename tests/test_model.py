from __future__ import annotations

import numpy as np
import pytest

from anisoap_ase import LinearModel


def test_linear_model_round_trip(tmp_path) -> None:
    model = LinearModel(np.array([1.5, -2.0]), intercept=0.25)
    path = tmp_path / "model.npz"
    model.save(path)
    restored = LinearModel.load(path)
    assert restored(np.array([2.0, 3.0])) == pytest.approx(-2.75)


def test_linear_model_rejects_feature_length_mismatch() -> None:
    model = LinearModel(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="Feature length"):
        model(np.array([1.0]))
