from __future__ import annotations

from typing import Iterable, Tuple

import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.atoms import Atoms

from .descriptors import AniSOAPDescriptorTorch, compute_descriptors_torch
from .model import load_linear_model


class AniSOAPCalculator(Calculator):
    """
    ASE calculator that uses AniSOAP descriptors + a linear model.

    Energies are computed with a torch Module, and forces are obtained via
    PyTorch autodiff: F = -dE/dr.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        lr_path: str = "lr.pkl",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.device = device

        # Descriptor wrapper (you must implement its __call__ in descriptors.py)
        self.descriptor = AniSOAPDescriptorTorch(device=device)

        # Linear regression model wrapped as torch Module
        self.model = load_linear_model(lr_path=lr_path, device=device)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Iterable[str] = ("energy", "forces"),
        system_changes: Iterable[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)

        assert atoms is not None, "AniSOAPCalculator.calculate requires atoms."

        # 1. positions + species as torch tensors
        positions = torch.tensor(
            atoms.positions,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        species = torch.tensor(
            atoms.numbers,
            dtype=torch.long,
            device=self.device,
        )

        # 2. descriptors in torch
        descriptors = compute_descriptors_torch(
            positions=positions,
            species=species,
            descriptor=self.descriptor,
        )

        # 3. energy from model
        energies = self.model(descriptors)  # [n_cfg]
        energy = energies.sum()

        # 4. autodiff forces: F = -dE/dr
        forces = -torch.autograd.grad(
            energy,
            positions,
            create_graph=False,
            retain_graph=False,
        )[0]

        # 5. move back to CPU / numpy for ASE
        self.results["energy"] = float(energy.detach().cpu().item())
        self.results["forces"] = forces.detach().cpu().numpy()
