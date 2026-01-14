"""
Score-Based Guidance for Molecular Sampling.

During sampling, we can add gradient-based guidance to push generated structures
toward satisfying molecular constraints (bond lengths, angles, etc.).

This is similar to classifier guidance in image generation, but for
molecular validity constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable, List, Tuple


def compute_pairwise_distances(x):
    """
    Compute pairwise distance matrix.

    Args:
        x: (batch, n_atoms, 3) coordinates

    Returns:
        distances: (batch, n_atoms, n_atoms) pairwise distances
    """
    x_i = x.unsqueeze(2)
    x_j = x.unsqueeze(1)
    distances = torch.norm(x_j - x_i, dim=-1)
    return distances


def compute_bond_lengths(x, bond_indices):
    """
    Compute bond lengths for specified bonds.

    Args:
        x: (batch, n_atoms, 3) coordinates
        bond_indices: (n_bonds, 2) tensor of bonded atom pairs

    Returns:
        bond_lengths: (batch, n_bonds) bond lengths
    """
    src_idx = bond_indices[:, 0]
    dst_idx = bond_indices[:, 1]

    x_src = x[:, src_idx, :]  # (batch, n_bonds, 3)
    x_dst = x[:, dst_idx, :]  # (batch, n_bonds, 3)

    bond_lengths = torch.norm(x_dst - x_src, dim=-1)  # (batch, n_bonds)
    return bond_lengths


def compute_angles(x, angle_indices):
    """
    Compute angles for specified triplets.

    Args:
        x: (batch, n_atoms, 3) coordinates
        angle_indices: (n_angles, 3) tensor of [atom_i, atom_j, atom_k]
                      where angle is i-j-k (j is the central atom)

    Returns:
        angles: (batch, n_angles) angles in radians
    """
    i_idx = angle_indices[:, 0]
    j_idx = angle_indices[:, 1]  # Central atom
    k_idx = angle_indices[:, 2]

    x_i = x[:, i_idx, :]  # (batch, n_angles, 3)
    x_j = x[:, j_idx, :]
    x_k = x[:, k_idx, :]

    # Vectors from central atom
    v1 = x_i - x_j  # j -> i
    v2 = x_k - x_j  # j -> k

    # Normalize
    v1_norm = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)
    v2_norm = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-8)

    # Dot product gives cos(angle)
    cos_angle = (v1_norm * v2_norm).sum(dim=-1)
    cos_angle = torch.clamp(cos_angle, -1 + 1e-6, 1 - 1e-6)

    angles = torch.acos(cos_angle)
    return angles


class MolecularConstraints:
    """
    Collection of molecular constraints for guidance.

    Each constraint returns a scalar loss that is minimized when the
    constraint is satisfied.
    """

    def __init__(
        self,
        bond_indices: Optional[torch.Tensor] = None,
        target_bond_lengths: Optional[torch.Tensor] = None,
        angle_indices: Optional[torch.Tensor] = None,
        target_angles: Optional[torch.Tensor] = None,
        min_distance: float = 1.0,
        max_distance: float = 2.0
    ):
        """
        Args:
            bond_indices: (n_bonds, 2) bonded atom pairs
            target_bond_lengths: (n_bonds,) target bond lengths in Angstroms
            angle_indices: (n_angles, 3) angle triplets [i, j, k]
            target_angles: (n_angles,) target angles in radians
            min_distance: Minimum allowed distance between any atoms
            max_distance: Maximum allowed bond distance
        """
        self.bond_indices = bond_indices
        self.target_bond_lengths = target_bond_lengths
        self.angle_indices = angle_indices
        self.target_angles = target_angles
        self.min_distance = min_distance
        self.max_distance = max_distance

    def bond_length_loss(self, x):
        """
        Penalize deviation from target bond lengths.

        Args:
            x: (batch, n_atoms, 3) coordinates

        Returns:
            loss: scalar
        """
        if self.bond_indices is None:
            return 0.0

        bond_lengths = compute_bond_lengths(x, self.bond_indices)

        if self.target_bond_lengths is not None:
            # Penalize deviation from target
            target = self.target_bond_lengths.to(x.device)
            loss = F.mse_loss(bond_lengths, target.unsqueeze(0).expand_as(bond_lengths))
        else:
            # Penalize bonds outside typical range [1.2, 1.6]
            too_short = F.relu(1.2 - bond_lengths)
            too_long = F.relu(bond_lengths - 1.6)
            loss = (too_short ** 2 + too_long ** 2).mean()

        return loss

    def angle_loss(self, x):
        """
        Penalize deviation from target angles.

        Args:
            x: (batch, n_atoms, 3) coordinates

        Returns:
            loss: scalar
        """
        if self.angle_indices is None:
            return 0.0

        angles = compute_angles(x, self.angle_indices)

        if self.target_angles is not None:
            target = self.target_angles.to(x.device)
            loss = F.mse_loss(angles, target.unsqueeze(0).expand_as(angles))
        else:
            # Penalize very small or very large angles
            # Typical bond angles are 100-120 degrees
            target_angle = np.pi * 109.5 / 180  # Tetrahedral angle
            loss = ((angles - target_angle) ** 2).mean()

        return loss

    def steric_clash_loss(self, x):
        """
        Penalize atoms that are too close (steric clashes).

        Args:
            x: (batch, n_atoms, 3) coordinates

        Returns:
            loss: scalar
        """
        distances = compute_pairwise_distances(x)

        # Mask out diagonal (self-distances)
        n_atoms = x.shape[1]
        mask = ~torch.eye(n_atoms, dtype=torch.bool, device=x.device)

        # Get non-diagonal distances
        dist_masked = distances[:, mask].view(distances.shape[0], n_atoms, n_atoms - 1)

        # Penalize distances below minimum
        clash_penalty = F.relu(self.min_distance - dist_masked)
        loss = (clash_penalty ** 2).mean()

        return loss

    def connectivity_loss(self, x):
        """
        Penalize disconnected atoms (no neighbors within max_distance).

        Args:
            x: (batch, n_atoms, 3) coordinates

        Returns:
            loss: scalar
        """
        distances = compute_pairwise_distances(x)
        n_atoms = x.shape[1]

        # Mask diagonal
        mask = ~torch.eye(n_atoms, dtype=torch.bool, device=x.device)
        distances_masked = distances.clone()
        distances_masked[:, ~mask] = float('inf')

        # Find minimum distance to any other atom
        min_distances = distances_masked.min(dim=-1)[0]  # (batch, n_atoms)

        # Penalize atoms with no neighbor within max_distance
        disconnected_penalty = F.relu(min_distances - self.max_distance)
        loss = (disconnected_penalty ** 2).mean()

        return loss

    def total_loss(self, x, weights=None):
        """
        Compute total constraint loss.

        Args:
            x: (batch, n_atoms, 3) coordinates
            weights: dict of weights for each loss term

        Returns:
            loss: scalar
        """
        if weights is None:
            weights = {
                'bond': 1.0,
                'angle': 0.5,
                'clash': 1.0,
                'connectivity': 1.0
            }

        loss = 0.0
        loss += weights.get('bond', 1.0) * self.bond_length_loss(x)
        loss += weights.get('angle', 0.5) * self.angle_loss(x)
        loss += weights.get('clash', 1.0) * self.steric_clash_loss(x)
        loss += weights.get('connectivity', 1.0) * self.connectivity_loss(x)

        return loss


class GuidedSampler:
    """
    Sampler with gradient-based guidance for molecular constraints.

    During each sampling step, we:
    1. Compute the model's predicted update
    2. Compute gradient of constraint loss w.r.t. coordinates
    3. Combine both for the final update
    """

    def __init__(
        self,
        model: nn.Module,
        constraints: MolecularConstraints,
        guidance_scale: float = 1.0,
        constraint_weights: Optional[dict] = None
    ):
        """
        Args:
            model: The diffusion/flow model
            constraints: MolecularConstraints object
            guidance_scale: How strongly to apply guidance
            constraint_weights: Weights for different constraints
        """
        self.model = model
        self.constraints = constraints
        self.guidance_scale = guidance_scale
        self.constraint_weights = constraint_weights or {}

    def compute_guidance(self, x, scale=1.0):
        """
        Compute gradient-based guidance.

        Args:
            x: (batch, n_atoms, 3) current coordinates (requires_grad=True)
            scale: scaling factor for guidance

        Returns:
            guidance: (batch, n_atoms, 3) gradient direction to improve constraints
        """
        x_grad = x.clone().detach().requires_grad_(True)

        # Compute constraint loss
        loss = self.constraints.total_loss(x_grad, self.constraint_weights)

        # Compute gradient
        loss.backward()
        guidance = -x_grad.grad * scale  # Negative gradient to minimize loss

        return guidance

    @torch.no_grad()
    def sample_euler(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 50,
        device: str = 'mps',
        atom_types: Optional[torch.Tensor] = None,
        guidance_start: float = 0.3,
        guidance_end: float = 0.9
    ):
        """
        Sample with Euler method + constraint guidance.

        Guidance is applied during a window of the sampling process
        (not at the very beginning when structure is pure noise,
        and not at the very end to avoid instability).

        Args:
            shape: (batch_size, n_atoms, 3)
            num_steps: Number of integration steps
            device: Device
            atom_types: Atom type indices
            guidance_start: When to start applying guidance (fraction of process)
            guidance_end: When to stop applying guidance

        Returns:
            samples: (batch, n_atoms, 3) generated coordinates
        """
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.full((shape[0],), t_val, device=device)

            # Get model prediction
            v = self.model(x, t, atom_types=atom_types)

            # Apply constraint guidance during specified window
            if guidance_start <= t_val <= guidance_end and self.guidance_scale > 0:
                # Temporarily enable gradients for guidance computation
                with torch.enable_grad():
                    guidance = self.compute_guidance(x, scale=self.guidance_scale * dt)
                x = x + v * dt + guidance
            else:
                x = x + v * dt

        return x

    @torch.no_grad()
    def sample_with_refinement(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 50,
        device: str = 'mps',
        atom_types: Optional[torch.Tensor] = None,
        refinement_steps: int = 100,
        refinement_lr: float = 0.01
    ):
        """
        Sample normally, then refine with gradient descent on constraints.

        This is a two-stage process:
        1. Generate samples using the model
        2. Optimize the samples to satisfy constraints

        Args:
            shape: (batch_size, n_atoms, 3)
            num_steps: Steps for initial generation
            device: Device
            atom_types: Atom type indices
            refinement_steps: Gradient descent steps for refinement
            refinement_lr: Learning rate for refinement

        Returns:
            samples: (batch, n_atoms, 3) refined coordinates
        """
        # Stage 1: Generate samples
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((shape[0],), i * dt, device=device)
            v = self.model(x, t, atom_types=atom_types)
            x = x + v * dt

        # Stage 2: Refine with gradient descent
        x = x.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=refinement_lr)

        for step in range(refinement_steps):
            optimizer.zero_grad()
            loss = self.constraints.total_loss(x, self.constraint_weights)
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                print(f"Refinement step {step}, constraint loss: {loss.item():.4f}")

        return x.detach()


class AdaptiveGuidedSampler(GuidedSampler):
    """
    Guided sampler with adaptive guidance strength.

    Automatically adjusts guidance based on current constraint satisfaction.
    """

    def __init__(
        self,
        model: nn.Module,
        constraints: MolecularConstraints,
        min_guidance: float = 0.1,
        max_guidance: float = 2.0,
        target_loss: float = 0.1
    ):
        super().__init__(model, constraints, guidance_scale=1.0)
        self.min_guidance = min_guidance
        self.max_guidance = max_guidance
        self.target_loss = target_loss

    def adaptive_guidance_scale(self, x):
        """Compute adaptive guidance scale based on constraint violation."""
        with torch.no_grad():
            loss = self.constraints.total_loss(x)

        # Scale guidance based on how far we are from target
        if loss < self.target_loss:
            scale = self.min_guidance
        else:
            # Linear interpolation
            scale = self.min_guidance + (self.max_guidance - self.min_guidance) * min(loss / 1.0, 1.0)

        return scale

    @torch.no_grad()
    def sample_adaptive(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 50,
        device: str = 'mps',
        atom_types: Optional[torch.Tensor] = None
    ):
        """Sample with adaptive guidance."""
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.full((shape[0],), t_val, device=device)

            # Get model prediction
            v = self.model(x, t, atom_types=atom_types)

            # Adaptive guidance (after initial phase)
            if t_val > 0.2:
                scale = self.adaptive_guidance_scale(x)
                with torch.enable_grad():
                    guidance = self.compute_guidance(x, scale=scale * dt)
                x = x + v * dt + guidance
            else:
                x = x + v * dt

        return x


def infer_bonds_from_reference(reference_coords: torch.Tensor, max_bond_length: float = 1.8):
    """
    Infer bonds from a reference structure.

    Args:
        reference_coords: (n_atoms, 3) reference coordinates
        max_bond_length: Maximum distance to consider as bond

    Returns:
        bond_indices: (n_bonds, 2) tensor of bonded atom pairs
        bond_lengths: (n_bonds,) bond lengths from reference
    """
    n_atoms = reference_coords.shape[0]
    bond_indices = []
    bond_lengths = []

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = torch.norm(reference_coords[i] - reference_coords[j])
            if dist < max_bond_length:
                bond_indices.append([i, j])
                bond_lengths.append(dist.item())

    bond_indices = torch.tensor(bond_indices, dtype=torch.long)
    bond_lengths = torch.tensor(bond_lengths, dtype=torch.float32)

    return bond_indices, bond_lengths


def infer_angles_from_bonds(bond_indices: torch.Tensor, n_atoms: int):
    """
    Infer angle triplets from bond connectivity.

    For each atom that has 2+ bonds, create angle triplets.

    Args:
        bond_indices: (n_bonds, 2) bonded atom pairs
        n_atoms: Total number of atoms

    Returns:
        angle_indices: (n_angles, 3) angle triplets [i, j, k] where j is central
    """
    # Build adjacency list
    neighbors = {i: [] for i in range(n_atoms)}
    for bond in bond_indices:
        i, j = bond[0].item(), bond[1].item()
        neighbors[i].append(j)
        neighbors[j].append(i)

    # Find angles
    angle_indices = []
    for j in range(n_atoms):  # j is central atom
        nbrs = neighbors[j]
        if len(nbrs) >= 2:
            # All pairs of neighbors form angles through j
            for idx1 in range(len(nbrs)):
                for idx2 in range(idx1 + 1, len(nbrs)):
                    i, k = nbrs[idx1], nbrs[idx2]
                    angle_indices.append([i, j, k])

    return torch.tensor(angle_indices, dtype=torch.long)
