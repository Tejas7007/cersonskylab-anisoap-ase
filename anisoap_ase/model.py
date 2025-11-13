import numpy as np
import pickle
from pathlib import Path

# Path helpers
_THIS_DIR = Path(__file__).resolve().parent
# lr.pkl is in the repo root: ~/repos/cersonskylab-anisoap-ase/lr.pkl
_LR_PATH = _THIS_DIR.parent / "lr.pkl"

# Load the trained linear regressor once at import time
with open(_LR_PATH, "rb") as f:
    _LR = pickle.load(f)


def linear_stub_model(desc) -> float:
    """
    Replace the stub with a real linear model.

    Parameters
    ----------
    desc : array-like
        Descriptor / feature vector for a single structure.
        Expected shape: (n_features,) or (1, n_features).

    Returns
    -------
    float
        Predicted energy (e.g., per-atom energy) from the trained linear model.
    """
    x = np.array(desc)

    # Ensure 2D shape for sklearn: (1, n_features)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim > 2:
        raise ValueError(f"Descriptor has unexpected shape {x.shape}, expected 1D or 2D.")

    # Predict with the loaded RidgeCV model
    y_pred = _LR.predict(x)

    # Handle shape (1,) or (1, 1)
    y_pred = np.array(y_pred).reshape(-1)

    return float(y_pred[0])

