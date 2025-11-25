"""
Main script for training and sampling POPC lipid diffusion model.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from lipid_diffusion.data.preprocessor import LipidCoordinatePreprocessor
from lipid_diffusion.data.dataset import LipidDataset
from lipid_diffusion.models.transformer import AtomwiseTransformer
from lipid_diffusion.models.diffusion import DiffusionModel
from lipid_diffusion.training.trainer import train_diffusion_model, generate_samples


def main():
    """Example pipeline for training and sampling."""
    
    # 1. Load and preprocess data
    print("Loading coordinate data...")
    preprocessor = LipidCoordinatePreprocessor()
    
    # Example: Load from multiple XYZ files
    # file_paths = [
    #     'data/popc_conf_001.xyz',
    #     'data/popc_conf_002.xyz',
    #     # ... add more files
    # ]
    # coords_normalized = preprocessor.process_dataset(file_paths, format='xyz')
    
    # Or generate synthetic data for testing
    n_conformations = 100
    n_atoms = 134  # POPC has 134 atoms
    synthetic_data = np.random.randn(n_conformations, n_atoms, 3) * 5.0
    
    # Normalize
    preprocessor.mean = synthetic_data.mean(axis=(0, 1))
    preprocessor.std = synthetic_data.std(axis=(0, 1))
    coords_normalized = (synthetic_data - preprocessor.mean) / (preprocessor.std + 1e-8)
    
    print(f"Loaded {len(coords_normalized)} conformations with {n_atoms} atoms each")
    
    # 2. Create dataset and dataloader
    dataset = LipidDataset(coords_normalized)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 3. Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = AtomwiseTransformer(
        n_atoms=n_atoms,
        hidden_dim=128,
        n_heads=4,
        n_layers=3
    )
    
    diffusion = DiffusionModel(
        model=model,
        n_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02
    )
    
    # 4. Train
    print("Training model...")
    losses = train_diffusion_model(
        dataloader=dataloader,
        model=model,
        diffusion=diffusion,
        n_epochs=50,
        lr=1e-4,
        device=device
    )
    
    # 5. Generate new samples
    print("Generating new conformations...")
    model.eval()
    new_conformations = generate_samples(
        diffusion=diffusion,
        n_samples=10,
        n_atoms=n_atoms,
        device=device
    )
    
    # Denormalize
    new_conformations_real = preprocessor.denormalize(new_conformations)
    
    print(f"Generated {len(new_conformations_real)} new conformations")
    print(f"Shape: {new_conformations_real.shape}")
    
    # 6. Save results
    np.save('outputs/generated_conformations.npy', new_conformations_real)
    torch.save(model.state_dict(), 'outputs/diffusion_model.pt')
    
    # 7. Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('outputs/training_loss.png')
    plt.close()
    
    print("Done! Results saved to 'outputs/' directory")


if __name__ == '__main__':
    main()
