"""
Training and sampling functions for diffusion models.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from ..models.diffusion import DiffusionModel


def train_diffusion_model(
    dataloader: DataLoader,
    model: torch.nn.Module,
    diffusion: DiffusionModel,
    n_epochs: int = 100,
    lr: float = 1e-4,
    device: str = 'cpu'
):
    """
    Train the diffusion model.
    
    Args:
        dataloader: DataLoader with lipid conformations
        model: Neural network model
        diffusion: DiffusionModel instance
        n_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        
    Returns:
        losses: List of average loss per epoch
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        
        for batch in dataloader:
            batch = batch.to(device)
            batch_size = batch.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.n_timesteps, (batch_size,), device=device)
            
            # Compute loss
            loss = diffusion.p_losses(batch, t)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
    
    return losses


def generate_samples(
    diffusion: DiffusionModel,
    n_samples: int,
    n_atoms: int,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Generate new lipid conformations.
    
    Args:
        diffusion: Trained DiffusionModel
        n_samples: Number of conformations to generate
        n_atoms: Number of atoms per lipid
        device: Device to generate on
        
    Returns:
        samples: (n_samples, n_atoms, 3) generated coordinates
    """
    shape = (n_samples, n_atoms, 3)
    samples = diffusion.sample(shape, device=device)
    return samples.cpu().numpy()
