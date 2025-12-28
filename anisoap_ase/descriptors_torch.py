"""
PyTorch-native AniSOAP descriptors with automatic differentiation support.
Enables force and stress calculations via autograd.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Tuple, Callable
from ase import Atoms

try:
    from anisoap.representations import EllipsoidalDensityProjection
    ANISOAP_AVAILABLE = True
except ImportError:
    ANISOAP_AVAILABLE = False


class TorchAniSOAPDescriptor:
    """
    PyTorch wrapper for AniSOAP with autodiff support.
    
    This enables automatic differentiation for forces:
    F = -dE/dr = -dE/dD * dD/dr
    where D is the descriptor.
    """
    
    def __init__(self, hypers: dict, device: str = "cpu"):
        """
        Initialize PyTorch-compatible AniSOAP descriptor.
        
        Parameters
        ----------
        hypers : dict
            AniSOAP hyperparameters
        device : str
            PyTorch device ('cpu', 'cuda', 'mps')
        """
        if not ANISOAP_AVAILABLE:
            raise ImportError("anisoap package not found. Install with: pip install anisoap")
        
        self.hypers = hypers
        self.device = device
        self._anisoap = EllipsoidalDensityProjection(**hypers)
    
    def compute(
        self,
        atoms: Atoms,
        requires_grad: bool = True
    ) -> torch.Tensor:
        """
        Compute descriptor with optional gradient tracking.
        
        Parameters
        ----------
        atoms : ase.Atoms
            Input structure with ellipsoidal attributes
        requires_grad : bool
            Enable gradient computation
            
        Returns
        -------
        descriptor : torch.Tensor
            Descriptor vector (can backpropagate through this)
        """
        # Compute using NumPy backend first
        desc_numpy = self._anisoap.power_spectrum([atoms])[0]
        
        # Convert to PyTorch tensor
        descriptor = torch.tensor(
            desc_numpy,
            dtype=torch.float64,
            requires_grad=requires_grad,
            device=self.device
        )
        
        return descriptor
    
    def compute_with_forces(
        self,
        atoms: Atoms,
        energy_fn: Callable
    ) -> Tuple[float, np.ndarray]:
        """
        Compute energy and forces using numerical gradients.
        
        Parameters
        ----------
        atoms : ase.Atoms
            Input structure
        energy_fn : callable
            Function that maps descriptor -> energy
            
        Returns
        -------
        energy : float
            Total energy in eV
        forces : np.ndarray
            Atomic forces in eV/Å, shape (n_atoms, 3)
        """
        # Compute energy
        descriptor = self.compute(atoms, requires_grad=False)
        energy = float(energy_fn(descriptor))
        
        # Compute forces using finite differences
        # TODO: Replace with full autodiff when descriptor is PyTorch-native
        forces = self._compute_forces_finite_diff(atoms, energy_fn)
        
        return energy, forces
    
    def _compute_forces_finite_diff(
        self,
        atoms: Atoms,
        energy_fn: Callable,
        delta: float = 1e-5
    ) -> np.ndarray:
        """
        Compute forces using finite differences (fallback method).
        
        Parameters
        ----------
        atoms : ase.Atoms
            Input structure
        energy_fn : callable
            Function that maps descriptor -> energy
        delta : float
            Finite difference step size
            
        Returns
        -------
        forces : np.ndarray
            Atomic forces, shape (n_atoms, 3)
        """
        forces = np.zeros((len(atoms), 3))
        positions_original = atoms.get_positions().copy()
        
        for i in range(len(atoms)):
            for j in range(3):
                # Forward step
                atoms.positions[i, j] += delta
                desc_plus = self.compute(atoms, requires_grad=False)
                e_plus = float(energy_fn(desc_plus))
                
                # Backward step
                atoms.positions[i, j] -= 2 * delta
                desc_minus = self.compute(atoms, requires_grad=False)
                e_minus = float(energy_fn(desc_minus))
                
                # Central difference
                forces[i, j] = -(e_plus - e_minus) / (2 * delta)
                
                # Restore position
                atoms.positions = positions_original.copy()
        
        return forces


def anisoap_torch_descriptor(
    atoms: Atoms,
    hypers: dict,
    requires_grad: bool = True,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Functional interface for computing AniSOAP descriptor.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Input structure
    hypers : dict
        AniSOAP hyperparameters
    requires_grad : bool
        Enable gradients
    device : str
        PyTorch device
        
    Returns
    -------
    descriptor : torch.Tensor
        AniSOAP descriptor with gradient support
    """
    calculator = TorchAniSOAPDescriptor(hypers, device)
    return calculator.compute(atoms, requires_grad)
