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
# Validate ellipsoidal attributes
# ======================================================================================
def _validate_ellipsoidal_attributes(atoms, frame_index=0):
    """
    Validate that required ellipsoidal attributes are present.
    
    Raises ValueError with clear message if attributes are missing.
    """
    # Check for quaternion attribute (required)
    if "c_q" not in atoms.arrays:
        raise ValueError(
            f"Expect frames with ellipsoidal attributes: "
            f"frame at index {frame_index} is missing a required attribute 'c_q'"
        )
    
    # Validate quaternion shape
    if atoms.arrays["c_q"].shape != (len(atoms), 4):
        raise ValueError(
            f"Attribute 'c_q' has incorrect shape at frame {frame_index}. "
            f"Expected ({len(atoms)}, 4), got {atoms.arrays['c_q'].shape}"
        )


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
      ✓ forces (eV/Å) - with PyTorch backend
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        descriptor_fn: Optional[DescriptorFn] = None,
        model: Optional[ModelFn] = None,  # Support both 'model' and 'model_fn'
        model_fn: Optional[ModelFn] = None,
        backend: str = "numpy",
        enable_forces: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Either a custom descriptor function or AniSOAP
        self.descriptor_fn = descriptor_fn

        # Support both 'model' and 'model_fn' for backwards compatibility
        if model is not None:
            self.model_fn = model
        elif model_fn is not None:
            self.model_fn = model_fn
        else:
            self.model_fn = linear_stub_model

        # Backend selection
        self.backend = backend
        self.enable_forces = enable_forces

        # Create AniSOAP descriptor calculator (only if using AniSOAP)
        if descriptor_fn is None:
            self._anisoap = EllipsoidalDensityProjection(**ANI_HYPERS_BENZENE)
        else:
            self._anisoap = None
        
        # PyTorch descriptor (lazy initialization)
        self._torch_descriptor = None

    # --------------------------------------------------------------------------
    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            raise ValueError("AniSOAPCalculator requires an Atoms object.")

        # Check if forces are requested but not available FIRST
        # This should happen before validation to match expected error behavior
        if "forces" in properties and not (self.enable_forces and self.backend == "torch"):
            raise PropertyNotImplementedError(
                "Forces require backend='torch' and enable_forces=True"
            )

        # ---------------------------------------------------------
        # Compute descriptors and energy/forces
        # ---------------------------------------------------------
        if "forces" in properties and self.enable_forces and self.backend == "torch":
            # Use PyTorch pathway with autodiff
            from .descriptors_torch import TorchAniSOAPDescriptor
            
            # Validate ellipsoidal attributes (only for AniSOAP descriptor)
            _validate_ellipsoidal_attributes(atoms, frame_index=0)
            
            if self._torch_descriptor is None:
                self._torch_descriptor = TorchAniSOAPDescriptor(ANI_HYPERS_BENZENE)
            
            _ensure_ellipsoid_semiaxes(atoms)
            energy, forces = self._torch_descriptor.compute_with_forces(atoms, self.model_fn)
            
            self.results["energy"] = energy
            self.results["forces"] = forces
            
        else:
            # NumPy pathway (energy only)
            if self.descriptor_fn is not None:
                # User-supplied descriptor - NO validation needed
                desc = self.descriptor_fn(atoms)
            else:
                # Using AniSOAP descriptor - validate ellipsoidal attributes
                _validate_ellipsoidal_attributes(atoms, frame_index=0)
                
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
