"""
Training and sampling functions for diffusion models.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from ..models.diffusion import DiffusionModel


def train_diffusion_model(
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model: torch.nn.Module,
    diffusion: DiffusionModel,
    n_epochs: int = 100,
    lr: float = 1e-4,
    device: str = 'cpu',
    patience: int = 20,
    min_delta: float = 1e-4,
    save_path: str = 'outputs/best_model.pt'
):
    """
    Train the diffusion model with train/test split and early stopping.
    
    Args:
        train_dataloader: DataLoader with training lipid conformations
        test_dataloader: DataLoader with test lipid conformations
        model: Neural network model
        diffusion: DiffusionModel instance
        n_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in loss to qualify as improvement
        save_path: Path to save the best model
        
    Returns:
        train_losses: List of average training loss per epoch
        test_losses: List of average test loss per epoch
        best_epoch: Epoch number with best validation performance
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    
    best_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_epoch_loss = 0.0
        
        for batch in train_dataloader:
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
            
            train_epoch_loss += loss.item()
        
        avg_train_loss = train_epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        model.eval()
        test_epoch_loss = 0.0
        
        with torch.no_grad():
            for batch in test_dataloader:
                batch = batch.to(device)
                batch_size = batch.shape[0]
                
                # Sample random timesteps
                t = torch.randint(0, diffusion.n_timesteps, (batch_size,), device=device)
                
                # Compute loss
                loss = diffusion.p_losses(batch, t)
                test_epoch_loss += loss.item()
        
        avg_test_loss = test_epoch_loss / len(test_dataloader)
        test_losses.append(avg_test_loss)
        
        # Check for improvement
        # Best model: low train loss AND test loss hasn't started increasing
        # We use train loss as primary metric but monitor test loss for overfitting
        if avg_train_loss < best_loss - min_delta:
            best_loss = avg_train_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            
            # Save best model (only if test loss isn't diverging too much)
            if len(test_losses) < 2 or avg_test_loss <= min(test_losses[:-1]) * 1.1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'test_loss': avg_test_loss,
                }, save_path)
                print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.6f}, "
                      f"Test Loss: {avg_test_loss:.6f} - BEST MODEL SAVED")
        else:
            epochs_without_improvement += 1
            print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.6f}, "
                  f"Test Loss: {avg_test_loss:.6f}")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            print(f"Best epoch was {best_epoch+1} with train loss {best_loss:.6f}")
            break
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.6f}, "
                  f"Test Loss: {avg_test_loss:.6f} - "
                  f"No improvement for {epochs_without_improvement} epochs")
    
    return train_losses, test_losses, best_epoch


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
