from __future__ import annotations

from typing import Tuple

import torch

# TODO: import whatever you normally use to compute AniSOAP descriptors.
# For example (THIS IS JUST AN EXAMPLE, ADAPT IT):
# from anisoap import AniSOAPDescriptor


class AniSOAPDescriptorTorch:
    """
    Thin wrapper that computes AniSOAP descriptors using torch tensors.

    You MUST adapt the body of `__call__` to your existing descriptor code.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        # TODO: initialize your real AniSOAP descriptor object here.
        # Example:
        # self.descriptor = AniSOAPDescriptor(...)

    def __call__(
        self,
        positions: torch.Tensor,
        species: torch.Tensor,
    ) -> torch.Tensor:
        """
        positions: [n_atoms, 3], torch, requires_grad=True
        species: [n_atoms], torch.long
        returns: descriptors as torch.Tensor [n_cfg, n_features]
        """
        # TODO: replace this dummy implementation with your real code.

        # Example pattern if your existing code expects numpy:
        #
        # pos_np = positions.detach().cpu().numpy()
        # Z_np = species.detach().cpu().numpy()
        # desc_np = self.descriptor.compute(pos_np, Z_np)
        # desc = torch.from_numpy(desc_np).to(positions.device)
        #
        # NOTE: this will NOT give correct gradients w.r.t positions.
        # For true autodiff forces, your descriptor code itself
        # must be implemented in torch.
        #
        # For now we just raise to remind you to fill this in.
        raise NotImplementedError(
            "Implement AniSOAPDescriptorTorch.__call__ using your existing "
            "descriptor pipeline."
        )


def compute_descriptors_torch(
    positions: torch.Tensor,
    species: torch.Tensor,
    descriptor: AniSOAPDescriptorTorch,
) -> torch.Tensor:
    """
    Convenience function to compute descriptors with the given descriptor object.
    """
    return descriptor(positions, species)
