from __future__ import annotations

import pathlib
import pickle
from typing import Optional

import numpy as np

# Path to the trained linear regressor (RidgeCV) for benzene.
# We assume lr.pkl lives in the repository root, i.e.
# cersonskylab-anisoap-ase/lr.pkl
_MODEL_PATH = pathlib.Path(__file__).resolve().parents[1] / "lr.pkl"

# Cached model instance
_LR: Optional[object] = None


def _load_lr_model():
    """
    Load the stored linear regressor (RidgeCV) from lr.pkl once and cache it.
    """
    global _LR

    if _LR is not None:
        return _LR

    if not _MODEL_PATH.is_file():
        raise FileNotFoundError(f"Could not find lr.pkl at {_MODEL_PATH}")

    with open(_MODEL_PATH, "rb") as f:
        _LR = pickle.load(f)

    return _LR


def linear_stub_model(desc) -> float:
    """
    Apply the stored linear regressor to a single descriptor vector.

    Parameters
    ----------
    desc
        Descriptor for a single frame. This may come in as a scalar,
        1D array, or higher-dimensional array. We:
          - flatten it,
          - pad or truncate to match lr.n_features_in_,
          - reshape to (1, n_features) for sklearn.

    Returns
    -------
    float
        Predicted energy (eV) as a Python float.
    """
    lr = _load_lr_model()

    # Flatten descriptor to 1D
    x = np.asarray(desc, dtype=float).ravel()

    # Target feature length from the trained model
    n_target = getattr(lr, "n_features_in_", x.size)

    # Pad or truncate to match expected number of features
    if x.size < n_target:
        x_full = np.zeros(n_target, dtype=float)
        x_full[: x.size] = x
        x = x_full
    elif x.size > n_target:
        x = x[:n_target]

    # Shape for sklearn: (n_samples, n_features)
    X = x.reshape(1, -1)  # (1, n_target)

    # RidgeCV.predict returns shape (n_samples, 1) or (n_samples,)
    y_pred = lr.predict(X)

    # Convert to scalar float
    return float(np.asarray(y_pred).ravel()[0])

