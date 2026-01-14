"""
Flow Matching Model for lipid conformations.

Flow matching provides an alternative to diffusion models with several advantages:
- Simpler training objective (direct velocity prediction)
- Faster sampling (straight paths through probability space)
- Better sample quality in many cases
- More stable training

Compatible with any backbone (Transformer, GNN, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def compute_pairwise_distances(coords):
    """
    Compute pairwise distance matrix for a batch of coordinates.

    Args:
        coords: (batch, n_atoms, 3) coordinates

    Returns:
        distances: (batch, n_atoms, n_atoms) pairwise distances
    """
    # coords_i: (batch, n_atoms, 1, 3)
    # coords_j: (batch, 1, n_atoms, 3)
    coords_i = coords.unsqueeze(2)
    coords_j = coords.unsqueeze(1)

    # Compute pairwise distances
    diff = coords_j - coords_i  # (batch, n_atoms, n_atoms, 3)
    distances = torch.norm(diff, dim=-1)  # (batch, n_atoms, n_atoms)

    return distances


class FlowMatchingModel:
    """
    Flow Matching Model (also known as Conditional Flow Matching or Rectified Flow).
    
    Instead of gradually adding/removing noise like diffusion, flow matching:
    1. Learns a velocity field that transforms noise → data
    2. Samples by integrating the ODE: dx/dt = v(x, t)
    3. Uses straight-line interpolation paths during training
    
    Key differences from DDPM:
    - Training: Predict velocity v_t instead of noise ε
    - Sampling: ODE integration instead of iterative denoising
    - Paths: Straight lines instead of noisy diffusion paths
    """
    
    def __init__(
        self,
        model: nn.Module,
        sigma_min: float = 1e-4,
        noise_schedule: str = 'vp',  # 'vp' (variance preserving) or 'ot' (optimal transport)
        distance_matrix_weight: float = 0.0,  # Weight for distance matrix preservation loss
    ):
        """
        Args:
            model: Neural network to predict velocity field v(x, t)
                   Must accept (x, t) and return same shape as x
            sigma_min: Minimum noise level for numerical stability
            noise_schedule: Type of interpolation schedule
                'vp': Variance preserving (similar to diffusion)
                'ot': Optimal transport (straight lines, better sample quality)
            distance_matrix_weight: Weight for distance matrix preservation loss (0 = disabled)
        """
        self.model = model
        self.sigma_min = sigma_min
        self.noise_schedule = noise_schedule
        self.distance_matrix_weight = distance_matrix_weight
        
    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        return self
    
    def get_interpolant(self, x_0, x_1, t):
        """
        Compute interpolation between noise (x_1) and data (x_0) at time t.
        
        For optimal transport (straight lines):
            x_t = t * x_0 + (1 - t) * x_1
            
        For variance preserving (similar to diffusion):
            x_t = alpha_t * x_0 + sigma_t * x_1
            where alpha_t = cos(π*t/2), sigma_t = sin(π*t/2)
        
        Args:
            x_0: (batch, n_atoms, 3) clean data
            x_1: (batch, n_atoms, 3) noise (usually Gaussian)
            t: (batch,) time in [0, 1]
            
        Returns:
            x_t: (batch, n_atoms, 3) interpolated data
            u_t: (batch, n_atoms, 3) conditional velocity (target for training)
        """
        # Expand time for broadcasting
        t_expanded = t[:, None, None]  # (batch, 1, 1)
        
        if self.noise_schedule == 'ot':
            # Optimal transport: straight line interpolation
            # x_t = (1-t) * x_1 + t * x_0
            x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
            
            # Velocity along straight line: dx/dt = x_0 - x_1
            u_t = x_0 - x_1
            
        elif self.noise_schedule == 'vp':
            # Variance preserving: similar to diffusion
            # alpha_t = cos(π*t/2), sigma_t = sin(π*t/2)
            alpha_t = torch.cos(t_expanded * np.pi / 2)
            sigma_t = torch.sin(t_expanded * np.pi / 2)
            
            x_t = alpha_t * x_0 + sigma_t * x_1
            
            # Velocity: dx/dt = -π/2 * sin(π*t/2) * x_0 + π/2 * cos(π*t/2) * x_1
            dalpha_dt = -np.pi / 2 * torch.sin(t_expanded * np.pi / 2)
            dsigma_dt = np.pi / 2 * torch.cos(t_expanded * np.pi / 2)
            u_t = dalpha_dt * x_0 + dsigma_dt * x_1
            
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")
        
        return x_t, u_t
    
    def compute_loss(self, x_0, t=None, atom_types=None):
        """
        Compute flow matching training loss.

        The loss is: ||v_θ(x_t, t) - u_t||^2 + λ * ||D(x_0) - D(x_pred)||^2
        where:
        - v_θ is the predicted velocity
        - u_t is the true conditional velocity
        - D(x) is the pairwise distance matrix
        - λ is the distance_matrix_weight
        - x_pred = x_t + v_pred * (1-t) is the predicted clean structure

        Args:
            x_0: (batch, n_atoms, 3) clean data samples
            t: (batch,) time samples in [0, 1], if None samples uniformly
            atom_types: (batch, n_atoms) or (n_atoms,) atom type indices (optional)

        Returns:
            loss: Scalar loss value
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample random times if not provided
        if t is None:
            t = torch.rand(batch_size, device=device)

        # Sample noise
        x_1 = torch.randn_like(x_0)

        # Get interpolation and target velocity
        x_t, u_t = self.get_interpolant(x_0, x_1, t)

        # Predict velocity (pass atom_types)
        v_pred = self.model(x_t, t, atom_types=atom_types)

        # MSE loss on velocity
        velocity_loss = F.mse_loss(v_pred, u_t)
        loss = velocity_loss

        # Distance matrix preservation loss
        if self.distance_matrix_weight > 0:
            # Compute reference distance matrix from clean data
            dist_ref = compute_pairwise_distances(x_0)

            # Predict clean structure by integrating velocity from x_t
            # For simplicity, use single-step prediction: x_pred = x_t + v_pred * (1-t)
            t_expanded = t[:, None, None]
            if self.noise_schedule == 'ot':
                # For OT: x_t = (1-t)*x_1 + t*x_0, v = x_0 - x_1
                # So: x_pred = x_1 + v_pred = x_t + v_pred*(1-t)
                x_pred = x_t + v_pred * (1 - t_expanded)
            elif self.noise_schedule == 'vp':
                # For VP: approximate with x_pred ≈ x_t / alpha_t
                alpha_t = torch.cos(t_expanded * np.pi / 2)
                x_pred = x_t / (alpha_t + 1e-8)
            else:
                x_pred = x_t + v_pred * (1 - t_expanded)

            # Compute predicted distance matrix
            dist_pred = compute_pairwise_distances(x_pred)

            # Distance matrix loss (only for nearby atoms to focus on bonds)
            # Use mask to focus on distances < 5 Å (bonded and nearby interactions)
            mask = (dist_ref < 5.0).float()
            dist_loss = F.mse_loss(dist_pred * mask, dist_ref * mask)

            loss = loss + self.distance_matrix_weight * dist_loss

        return loss
    
    @torch.no_grad()
    def sample_euler(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 50,
        device: str = 'mps',
        atom_types=None
    ):
        """
        Generate samples using Euler method for ODE integration.

        This is the simplest sampling method:
            x_{t+dt} = x_t + v(x_t, t) * dt

        Args:
            shape: (batch_size, n_atoms, 3) shape of samples
            num_steps: Number of integration steps (fewer steps = faster)
            device: Device to generate on
            atom_types: (n_atoms,) atom type indices (optional)

        Returns:
            samples: (batch_size, n_atoms, 3) generated conformations
        """
        # Start from noise
        x = torch.randn(shape, device=device)

        # Time steps from 0 to 1
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((shape[0],), i * dt, device=device)

            # Predict velocity (pass atom_types)
            v = self.model(x, t, atom_types=atom_types)

            # Euler step
            x = x + v * dt

        return x
    
    @torch.no_grad()
    def sample_heun(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 50,
        device: str = 'mps',
        atom_types=None
    ):
        """
        Generate samples using Heun's method (2nd order Runge-Kutta).

        More accurate than Euler, can use fewer steps:
            k1 = v(x_t, t)
            k2 = v(x_t + k1*dt, t+dt)
            x_{t+dt} = x_t + (k1 + k2)/2 * dt

        Args:
            shape: (batch_size, n_atoms, 3) shape of samples
            num_steps: Number of integration steps
            device: Device to generate on
            atom_types: (n_atoms,) atom type indices (optional)

        Returns:
            samples: (batch_size, n_atoms, 3) generated conformations
        """
        # Start from noise
        x = torch.randn(shape, device=device)

        # Time steps
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((shape[0],), i * dt, device=device)
            t_next = torch.full((shape[0],), (i + 1) * dt, device=device)

            # First velocity evaluation (pass atom_types)
            v1 = self.model(x, t, atom_types=atom_types)

            # Predict next position
            x_next = x + v1 * dt

            # Second velocity evaluation (pass atom_types)
            v2 = self.model(x_next, t_next, atom_types=atom_types)

            # Average velocity
            x = x + (v1 + v2) / 2 * dt

        return x
    
    @torch.no_grad()
    def sample_rk4(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 25,
        device: str = 'mps',
        atom_types=None
    ):
        """
        Generate samples using 4th order Runge-Kutta method.

        Most accurate, can use even fewer steps (25-50 typically sufficient):
            k1 = v(x_t, t)
            k2 = v(x_t + k1*dt/2, t+dt/2)
            k3 = v(x_t + k2*dt/2, t+dt/2)
            k4 = v(x_t + k3*dt, t+dt)
            x_{t+dt} = x_t + (k1 + 2*k2 + 2*k3 + k4)/6 * dt

        Args:
            shape: (batch_size, n_atoms, 3) shape of samples
            num_steps: Number of integration steps (25-50 usually sufficient)
            device: Device to generate on
            atom_types: (n_atoms,) atom type indices (optional)

        Returns:
            samples: (batch_size, n_atoms, 3) generated conformations
        """
        # Start from noise
        x = torch.randn(shape, device=device)

        # Time steps
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((shape[0],), i * dt, device=device)
            t_half = torch.full((shape[0],), (i + 0.5) * dt, device=device)
            t_next = torch.full((shape[0],), (i + 1) * dt, device=device)

            # RK4 stages (pass atom_types)
            k1 = self.model(x, t, atom_types=atom_types)
            k2 = self.model(x + k1 * dt / 2, t_half, atom_types=atom_types)
            k3 = self.model(x + k2 * dt / 2, t_half, atom_types=atom_types)
            k4 = self.model(x + k3 * dt, t_next, atom_types=atom_types)

            # Update
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt

        return x
    
    def sample(
        self,
        shape: Tuple[int, ...],
        device: str = 'mps',
        method: str = 'heun',
        num_steps: int = 50,
        atom_types=None
    ):
        """
        Generate samples using specified ODE solver.

        Args:
            shape: (batch_size, n_atoms, 3) shape of samples
            device: Device to generate on
            method: 'euler', 'heun', or 'rk4'
            num_steps: Number of integration steps
            atom_types: (n_atoms,) atom type indices (optional)

        Returns:
            samples: (batch_size, n_atoms, 3) generated conformations
        """
        if method == 'euler':
            return self.sample_euler(shape, num_steps, device, atom_types)
        elif method == 'heun':
            return self.sample_heun(shape, num_steps, device, atom_types)
        elif method == 'rk4':
            return self.sample_rk4(shape, num_steps, device, atom_types)
        else:
            raise ValueError(f"Unknown sampling method: {method}")


class StochasticFlowMatchingModel(FlowMatchingModel):
    """
    Stochastic Flow Matching with added noise during sampling.
    
    This can improve sample diversity by adding small Brownian motion
    during ODE integration, similar to stochastic differential equations (SDEs).
    """
    
    def __init__(
        self,
        model: nn.Module,
        sigma_min: float = 1e-4,
        noise_schedule: str = 'ot',
        stochasticity: float = 0.01,
        distance_matrix_weight: float = 0.0
    ):
        """
        Args:
            model: Neural network to predict velocity
            sigma_min: Minimum noise level
            noise_schedule: 'vp' or 'ot'
            stochasticity: Amount of noise to add during sampling (0 = deterministic)
            distance_matrix_weight: Weight for distance matrix preservation loss
        """
        super().__init__(model, sigma_min, noise_schedule, distance_matrix_weight)
        self.stochasticity = stochasticity
    
    @torch.no_grad()
    def sample_sde(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 50,
        device: str = 'mps',
        atom_types=None
    ):
        """
        Generate samples using stochastic ODE (adding Brownian motion).

        x_{t+dt} = x_t + v(x_t, t) * dt + σ * sqrt(dt) * ε
        where ε ~ N(0, I)

        Args:
            shape: (batch_size, n_atoms, 3) shape of samples
            num_steps: Number of integration steps
            device: Device to generate on
            atom_types: (n_atoms,) atom type indices (optional)

        Returns:
            samples: (batch_size, n_atoms, 3) generated conformations
        """
        x = torch.randn(shape, device=device)

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((shape[0],), i * dt, device=device)

            # Predict velocity (pass atom_types)
            v = self.model(x, t, atom_types=atom_types)

            # Deterministic step
            x = x + v * dt

            # Stochastic step (Brownian motion)
            if i < num_steps - 1:  # Don't add noise on last step
                noise = torch.randn_like(x)
                x = x + self.stochasticity * np.sqrt(dt) * noise

        return x


# Compatibility wrapper for training scripts
class FlowMatchingTrainer:
    """
    Wrapper to make FlowMatchingModel compatible with existing training scripts.
    Provides the same interface as DiffusionModel for drop-in replacement.
    """
    
    def __init__(
        self,
        model: nn.Module,
        noise_schedule: str = 'ot',
        stochastic: bool = False,
        stochasticity: float = 0.01,
        distance_matrix_weight: float = 0.0,
        # These params are for compatibility with DiffusionModel but not used
        n_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        """
        Args:
            model: Neural network backbone
            noise_schedule: 'ot' (optimal transport) or 'vp' (variance preserving)
            stochastic: Whether to use stochastic sampling
            stochasticity: Amount of stochasticity if enabled
            distance_matrix_weight: Weight for distance matrix preservation loss (e.g., 0.1)
            n_timesteps: Ignored (for compatibility)
            beta_start: Ignored (for compatibility)
            beta_end: Ignored (for compatibility)
        """
        if stochastic:
            self.flow_model = StochasticFlowMatchingModel(
                model,
                noise_schedule=noise_schedule,
                stochasticity=stochasticity,
                distance_matrix_weight=distance_matrix_weight
            )
        else:
            self.flow_model = FlowMatchingModel(
                model,
                noise_schedule=noise_schedule,
                distance_matrix_weight=distance_matrix_weight
            )

        self.model = model
        # Store dummy values for compatibility
        self.n_timesteps = n_timesteps
    
    def to(self, device):
        """Move to device."""
        self.flow_model.to(device)
        self.model = self.model.to(device)
        return self
    
    def p_losses(self, x_0, t=None, atom_types=None):
        """
        Compute loss - compatible with diffusion training interface.

        Args:
            x_0: (batch, n_atoms, 3) clean data
            t: Ignored - flow matching samples time internally
            atom_types: (batch, n_atoms) or (n_atoms,) atom type indices (optional)

        Returns:
            loss: Training loss
        """
        return self.flow_model.compute_loss(x_0, atom_types=atom_types)
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        device: str = 'mps',
        method: str = 'heun',
        num_steps: int = 50,
        atom_types=None
    ):
        """
        Generate samples.

        Args:
            shape: (batch_size, n_atoms, 3)
            device: Device
            method: 'euler', 'heun', 'rk4', or 'sde' (if stochastic)
            num_steps: Integration steps (fewer than diffusion!)
            atom_types: (n_atoms,) atom type indices (optional)

        Returns:
            samples: Generated conformations
        """
        if hasattr(self.flow_model, 'sample_sde') and method == 'sde':
            return self.flow_model.sample_sde(shape, num_steps, device, atom_types)
        else:
            return self.flow_model.sample(shape, device, method, num_steps, atom_types)
