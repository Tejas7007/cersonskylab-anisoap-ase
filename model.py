from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import torch


class LinearEnergyModel(torch.nn.Module):
    """
    Simple linear energy model using pre-trained coefficients from lr.pkl.

    Energy = descriptors @ coef + intercept
    """

    def __init__(self, coef, intercept, device: str = "cpu") -> None:
        super().__init__()
        coef_tensor = torch.as_tensor(coef, dtype=torch.float32, device=device)
        intercept_tensor = torch.as_tensor(intercept, dtype=torch.float32, device=device)

        # store as non-trainable buffers (fixed model)
        self.register_buffer("coef", coef_tensor)
        self.register_buffer("intercept", intercept_tensor)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """
        descriptors: [n_cfg, n_features]
        returns: energies [n_cfg]
        """
        # ensure 2D
        if descriptors.ndim == 1:
            descriptors = descriptors.unsqueeze(0)
        return descriptors @ self.coef + self.intercept


def load_linear_model(
    lr_path: Union[str, Path] = "lr.pkl",
    device: str = "cpu",
) -> LinearEnergyModel:
    """
    Load scikit-learn linear regression from disk and wrap as a torch Module.
    """
    lr_path = Path(lr_path)
    lr = joblib.load(lr_path)

    coef = lr.coef_
    intercept = getattr(lr, "intercept_", 0.0)

    model = LinearEnergyModel(coef, intercept, device=device)
    model.to(device)
    return model
