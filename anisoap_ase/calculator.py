from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from ase.calculators.calculator import (
    Calculator,
    all_changes,
    PropertyNotImplementedError,
)

from anisoap.representations import EllipsoidalDensityProjection
from .model import linear_stub_model  # Our sklearn-loaded linear model

# Type aliases for optional custom hooks
DescriptorFn = Callable[..., "np.ndarray | float | int"]
ModelFn = Callable[..., float]


# ======================================================================================
# AniSOAP Hyperparameters — EXACTLY the ones used in the benzene example
# ======================================================================================
ANI_HYPERS_BENZENE = {
    "max_angular": 9,
    "max_radial": 6,
    "radial_basis_name": "gto",
    "subtract_center_contribution": True,
    "rotation_type": "quaternion",
    "rotation_key": "c_q",
    "cutoff_radius": 7.0,
    "radial_gaussian_width": 1.5,
    "basis_rcond": 1e-8,
    "basis_tol": 1e-3,
}


# ======================================================================================
# Automatically insert ellipsoid semiaxes if missing
# ======================================================================================
def _ensure_ellipsoid_semiaxes(atoms, a1=4.0, a2=4.0, a3=0.5):
    """
    Ensure c_diameter[1-3] exist on each frame, matching AniSOAP documentation.

    The ellipsoids.xyz file does NOT include shape information, so we must add it.
    """
    n = len(atoms)

    if "c_diameter[1]" not in atoms.arrays:
        atoms.arrays["c_diameter[1]"] = np.ones(n) * a1

    if "c_diameter[2]" not in atoms.arrays:
        atoms.arrays["c_diameter[2]"] = np.ones(n) * a2

    if "c_diameter[3]" not in atoms.arrays:
        atoms.arrays["c_diameter[3]"] = np.ones(n) * a3


# ======================================================================================
# AniSOAP ASE Calculator
# ======================================================================================
class AniSOAPCalculator(Calculator):
    """
    ASE-compatible calculator using AniSOAP descriptors + linear regressor.

    Currently computes:
      ✓ energy (eV)
    """

    implemented_properties = ["energy"]

    def __init__(
        self,
        descriptor_fn: Optional[DescriptorFn] = None,
        model_fn: Optional[ModelFn] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Either a custom descriptor function or AniSOAP
        self.descriptor_fn = descriptor_fn

        # Either a custom ML model or our lr.pkl model
        self.model_fn = model_fn if model_fn is not None else linear_stub_model

        # Create AniSOAP descriptor calculator
        self._anisoap = EllipsoidalDensityProjection(**ANI_HYPERS_BENZENE)

    # --------------------------------------------------------------------------
    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            raise ValueError("AniSOAPCalculator requires an Atoms object.")

        # ---------------------------------------------------------
        # Compute descriptors
        # ---------------------------------------------------------
        if self.descriptor_fn is not None:
            # User-supplied descriptor
            desc = self.descriptor_fn(atoms)

        else:
            # Ensure required ellipsoid attributes exist
            _ensure_ellipsoid_semiaxes(atoms)

            # Compute AniSOAP power spectrum
            x = self._anisoap.power_spectrum([atoms])

            # x has shape (1, n_features)
            desc = np.array(x[0]).ravel()

        # ---------------------------------------------------------
        # ML model prediction
        # ---------------------------------------------------------
        energy = float(self.model_fn(desc))

        # ASE stores energy in self.results
        self.results["energy"] = energy

