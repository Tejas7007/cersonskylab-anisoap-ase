from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from ase.calculators.calculator import (
    Calculator,
    all_changes,
    PropertyNotImplementedError,
)

from anisoap.representations import EllipsoidalDensityProjection
from .model import linear_stub_model

# Type aliases
DescriptorFn = Callable[..., "np.ndarray | float | int"]
ModelFn = Callable[..., float]


# AniSOAP hyperparameters from benzene example
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


class AniSOAPCalculator(Calculator):
    """
    ASE-compatible calculator that converts ellipsoidal frames into AniSOAP
    descriptors and evaluates energies using a trained linear regressor.
    """

    implemented_properties = ["energy"]

    def __init__(
        self,
        descriptor_fn: Optional[DescriptorFn] = None,
        model_fn: Optional[ModelFn] = None,
        anisoap_hypers: Optional[dict] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        descriptor_fn : callable, optional
            Custom hook: atoms -> descriptor vector.
            If None, we compute AniSOAP power spectrum.
        model_fn : callable, optional
            Custom hook: descriptor -> energy.
            Defaults to linear_stub_model (lr.predict wrapper).
        anisoap_hypers : dict, optional
            Hypers for EllipsoidalDensityProjection.
        """
        super().__init__(**kwargs)

        if anisoap_hypers is None:
            anisoap_hypers = ANI_HYPERS_BENZENE

        # AniSOAP featurizer
        self._anisoap = EllipsoidalDensityProjection(**anisoap_hypers)

        self.descriptor_fn = descriptor_fn
        self.model_fn = model_fn or linear_stub_model

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        """
        1. Convert the ellipsoidal frame (atoms) → AniSOAP descriptor.
        2. Apply trained linear model (lr.predict)
        3. Return energy in ASE format
        """
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            atoms = self.atoms
        else:
            self.atoms = atoms

        if "energy" not in properties:
            raise PropertyNotImplementedError(
                "AniSOAPCalculator currently only provides 'energy'"
            )

        # ----- Descriptor computation -----
        if self.descriptor_fn is not None:
            desc = self.descriptor_fn(atoms)
        else:
            # Power spectrum returns shape (1, n_features)
            x = self._anisoap.power_spectrum([atoms])
            desc = np.array(x[0]).ravel()

        # ----- ML prediction -----
        energy = float(self.model_fn(desc))

        # If needed: multiply by #atoms for total energy
        # energy = energy * len(atoms)

        # ----- Store result -----
        self.results["energy"] = energy

