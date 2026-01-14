"""
Training script to evaluate different architectural approaches for lipid generation.

Approaches:
1. EGNN - E(3)-equivariant graph neural network
2. BondGNN - Bond-aware message passing
3. LatentFlow - Flow matching in latent space
4. Guided - Base model + score-based guidance during sampling

Usage:
    python train_architectures.py --model egnn
    python train_architectures.py --model bond
    python train_architectures.py --model latent
    python train_architectures.py --model guided
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from natsort import natsorted
import random
import glob
import os
from tqdm import tqdm

from lipid_diffusion.data.preprocessor import LipidCoordinatePreprocessor
from lipid_diffusion.data.dataset import LipidDataset
from lipid_diffusion.models.flow_matching import FlowMatchingTrainer
from lipid_diffusion.models.gnn import LipidGraphNetwork
from lipid_diffusion.models.egnn import EGNN, EGNNWithRBF
from lipid_diffusion.models.bond_gnn import BondAwareGNN, infer_bonds_from_structure
from lipid_diffusion.models.latent_diffusion import LatentFlowMatcher
from lipid_diffusion.models.guided_sampling import (
    GuidedSampler, MolecularConstraints,
    infer_bonds_from_reference, infer_angles_from_bonds
)
from lipid_diffusion.training.trainer import train_diffusion_model, generate_samples


def analyze_structure(coords, reference_coords=None):
    """Analyze generated structure quality."""
    n_atoms = len(coords)

    # Compute pairwise distances
    dists = []
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            dists.append(dist)
    dists = np.array(dists)

    # Bond analysis
    bonds = dists[dists < 2.0]

    # Connectivity analysis
    disconnected = 0
    for i in range(n_atoms):
        neighbors = 0
        for j in range(n_atoms):
            if i != j:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < 3.0:
                    neighbors += 1
        if neighbors == 0:
            disconnected += 1

    # Structure extent
    extent = [
        coords[:, i].max() - coords[:, i].min()
        for i in range(3)
    ]

    results = {
        'num_bonds': len(bonds),
        'bond_mean': bonds.mean() if len(bonds) > 0 else 0,
        'bond_std': bonds.std() if len(bonds) > 0 else 0,
        'disconnected': disconnected,
        'extent_x': extent[0],
        'extent_y': extent[1],
        'extent_z': extent[2],
    }

    # Compare to reference if provided
    if reference_coords is not None:
        ref_dists = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(reference_coords[i] - reference_coords[j])
                ref_dists.append(dist)
        ref_dists = np.array(ref_dists)
        ref_bonds = ref_dists[ref_dists < 2.0]

        results['ref_num_bonds'] = len(ref_bonds)
        results['ref_bond_mean'] = ref_bonds.mean()

    return results


def train_model(model_type, device='mps', epochs=300, patience=30):
    """Train a specific model architecture."""

    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} model")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    preprocessor = LipidCoordinatePreprocessor()
    file_paths = natsorted(glob.glob('../pdbs/conf*.pdb'))
    random.seed(42)
    random.shuffle(file_paths)
    coords_normalized, atom_types = preprocessor.process_dataset(file_paths, format='pdb')
    n_atoms = coords_normalized.shape[1]

    # Get reference structure for bond inference
    reference_coords = preprocessor.load_coordinates(file_paths[0], format='pdb')

    # Create dataset
    dataset = LipidDataset(coords_normalized, atom_types)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"Train: {train_size}, Test: {test_size}")

    # Create model based on type
    if model_type == 'egnn':
        model = EGNN(
            n_atoms=n_atoms,
            hidden_dim=256,
            num_layers=4,
            num_atom_types=dataset.num_atom_types,
            use_atom_types=True,
            attention=True
        )

    elif model_type == 'egnn_rbf':
        model = EGNNWithRBF(
            n_atoms=n_atoms,
            hidden_dim=256,
            num_layers=4,
            num_atom_types=dataset.num_atom_types,
            use_atom_types=True,
            num_rbf=50,
            cutoff=5.0
        )

    elif model_type == 'bond':
        # Infer bonds from reference structure
        bonds = infer_bonds_from_structure(reference_coords)
        bond_indices = torch.tensor(bonds, dtype=torch.long)
        print(f"Inferred {len(bonds)} bonds from reference structure")

        model = BondAwareGNN(
            n_atoms=n_atoms,
            hidden_dim=256,
            num_layers=4,
            num_atom_types=dataset.num_atom_types,
            use_atom_types=True,
            bond_indices=bond_indices
        )

    elif model_type == 'latent':
        model = LatentFlowMatcher(
            n_atoms=n_atoms,
            latent_dim=64,
            hidden_dim=256,
            num_layers=3,
            num_atom_types=dataset.num_atom_types,
            use_atom_types=True
        )

    elif model_type == 'baseline':
        # Baseline GNN with improved RBF
        model = LipidGraphNetwork(
            n_atoms=n_atoms,
            hidden_dim=512,
            num_layers=4,
            num_heads=4,
            num_rbf=50,
            cutoff=5.0,
            dropout=0.05,
            num_atom_types=dataset.num_atom_types,
            use_atom_types=True
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Wrap in flow matching trainer
    diffusion = FlowMatchingTrainer(
        model=model,
        noise_schedule='vp',
        stochastic=False
    )
    diffusion.to(device)

    # Train
    print(f"\nTraining {model_type}...")
    save_path = f'outputs/{model_type}_best_model.pt'

    train_losses, test_losses, best_epoch = train_diffusion_model(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        diffusion=diffusion,
        n_epochs=epochs,
        lr=1e-4,
        device=device,
        patience=patience,
        min_delta=1e-4,
        save_path=save_path
    )

    print(f"\nBest epoch: {best_epoch+1}")
    print(f"Best train loss: {train_losses[best_epoch]:.6f}")
    print(f"Best test loss: {test_losses[best_epoch]:.6f}")

    # Load best model
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Generate samples
    print("\nGenerating samples...")

    if model_type == 'guided':
        # Use guided sampling
        reference_tensor = torch.tensor(reference_coords, dtype=torch.float32)
        bond_indices, bond_lengths = infer_bonds_from_reference(reference_tensor)
        angle_indices = infer_angles_from_bonds(bond_indices, n_atoms)

        constraints = MolecularConstraints(
            bond_indices=bond_indices.to(device),
            target_bond_lengths=bond_lengths.to(device),
            angle_indices=angle_indices.to(device) if len(angle_indices) > 0 else None
        )

        sampler = GuidedSampler(model, constraints, guidance_scale=0.5)
        new_conformations = sampler.sample_with_refinement(
            shape=(10, n_atoms, 3),
            num_steps=50,
            device=device,
            atom_types=dataset.atom_type_indices.to(device),
            refinement_steps=50,
            refinement_lr=0.01
        ).cpu()
    else:
        new_conformations = generate_samples(
            diffusion=diffusion,
            n_samples=10,
            n_atoms=n_atoms,
            device=device,
            atom_types=dataset.atom_type_indices.to(device)
        )

    # Denormalize
    new_conformations_real = preprocessor.denormalize(new_conformations)

    # Analyze results
    print("\nAnalyzing generated structures...")
    all_results = []
    for i, conf in enumerate(new_conformations_real):
        results = analyze_structure(conf, reference_coords)
        all_results.append(results)

    # Average results
    avg_results = {
        key: np.mean([r[key] for r in all_results])
        for key in all_results[0].keys()
    }

    print(f"\n{model_type.upper()} Results:")
    print(f"  Bonds < 2Å: {avg_results['num_bonds']:.1f} (ref: {avg_results.get('ref_num_bonds', 'N/A')})")
    print(f"  Bond mean: {avg_results['bond_mean']:.3f} Å (ref: {avg_results.get('ref_bond_mean', 'N/A'):.3f})")
    print(f"  Disconnected atoms: {avg_results['disconnected']:.1f}/52")
    print(f"  Extent: {avg_results['extent_x']:.1f} × {avg_results['extent_y']:.1f} × {avg_results['extent_z']:.1f} Å")

    # Save results
    np.save(f'outputs/{model_type}_generated.npy', new_conformations_real)

    # Save training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.axvline(best_epoch, color='g', linestyle='--', label=f'Best ({best_epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type.upper()} Training')
    plt.legend()
    plt.savefig(f'outputs/{model_type}_training.png', dpi=150)
    plt.close()

    return {
        'model_type': model_type,
        'best_epoch': best_epoch,
        'train_loss': train_losses[best_epoch],
        'test_loss': test_losses[best_epoch],
        **avg_results
    }


def train_with_guidance(device='mps', epochs=300, patience=30):
    """Train baseline model, then use guided sampling."""

    print(f"\n{'='*60}")
    print(f"Training GUIDED model (baseline + constraint guidance)")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    preprocessor = LipidCoordinatePreprocessor()
    file_paths = natsorted(glob.glob('../pdbs/conf*.pdb'))
    random.seed(42)
    random.shuffle(file_paths)
    coords_normalized, atom_types = preprocessor.process_dataset(file_paths, format='pdb')
    n_atoms = coords_normalized.shape[1]

    reference_coords = preprocessor.load_coordinates(file_paths[0], format='pdb')

    dataset = LipidDataset(coords_normalized, atom_types)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Use baseline model
    model = LipidGraphNetwork(
        n_atoms=n_atoms,
        hidden_dim=512,
        num_layers=4,
        num_heads=4,
        num_rbf=50,
        cutoff=5.0,
        dropout=0.05,
        num_atom_types=dataset.num_atom_types,
        use_atom_types=True
    )

    diffusion = FlowMatchingTrainer(
        model=model,
        noise_schedule='vp',
        stochastic=False
    )
    diffusion.to(device)

    # Train
    save_path = 'outputs/guided_best_model.pt'
    train_losses, test_losses, best_epoch = train_diffusion_model(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        diffusion=diffusion,
        n_epochs=epochs,
        lr=1e-4,
        device=device,
        patience=patience,
        min_delta=1e-4,
        save_path=save_path
    )

    # Load best model
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Set up guided sampling
    reference_tensor = torch.tensor(reference_coords, dtype=torch.float32)
    bond_indices, bond_lengths = infer_bonds_from_reference(reference_tensor)
    angle_indices = infer_angles_from_bonds(bond_indices, n_atoms)

    print(f"\nInferred {len(bond_indices)} bonds and {len(angle_indices)} angles")

    constraints = MolecularConstraints(
        bond_indices=bond_indices.to(device),
        target_bond_lengths=bond_lengths.to(device),
        angle_indices=angle_indices.to(device) if len(angle_indices) > 0 else None,
        min_distance=1.0,
        max_distance=2.5
    )

    sampler = GuidedSampler(
        model=model,
        constraints=constraints,
        guidance_scale=0.5,
        constraint_weights={'bond': 1.0, 'angle': 0.3, 'clash': 0.5, 'connectivity': 0.5}
    )

    # Generate with guidance
    print("\nGenerating with guided sampling...")
    new_conformations = sampler.sample_with_refinement(
        shape=(10, n_atoms, 3),
        num_steps=50,
        device=device,
        atom_types=dataset.atom_type_indices.to(device),
        refinement_steps=100,
        refinement_lr=0.01
    ).cpu()

    new_conformations_real = preprocessor.denormalize(new_conformations)

    # Analyze
    all_results = []
    for conf in new_conformations_real:
        results = analyze_structure(conf, reference_coords)
        all_results.append(results)

    avg_results = {
        key: np.mean([r[key] for r in all_results])
        for key in all_results[0].keys()
    }

    print(f"\nGUIDED Results:")
    print(f"  Bonds < 2Å: {avg_results['num_bonds']:.1f}")
    print(f"  Bond mean: {avg_results['bond_mean']:.3f} Å")
    print(f"  Disconnected atoms: {avg_results['disconnected']:.1f}/52")

    np.save('outputs/guided_generated.npy', new_conformations_real)

    return {
        'model_type': 'guided',
        'best_epoch': best_epoch,
        'train_loss': train_losses[best_epoch],
        'test_loss': test_losses[best_epoch],
        **avg_results
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all',
                       choices=['egnn', 'egnn_rbf', 'bond', 'latent', 'guided', 'baseline', 'all'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=30)
    args = parser.parse_args()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    os.makedirs('outputs', exist_ok=True)

    results = []

    if args.model == 'all':
        models_to_train = ['baseline', 'egnn', 'egnn_rbf', 'bond', 'latent']
    else:
        models_to_train = [args.model]

    for model_type in models_to_train:
        if model_type == 'guided':
            result = train_with_guidance(device, args.epochs, args.patience)
        else:
            result = train_model(model_type, device, args.epochs, args.patience)
        results.append(result)

    # Print comparison
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("COMPARISON OF RESULTS")
        print(f"{'='*80}")
        print(f"{'Model':<12} {'Bonds':<8} {'Bond Mean':<12} {'Disconn':<10} {'Test Loss':<12}")
        print("-" * 80)
        for r in results:
            print(f"{r['model_type']:<12} {r['num_bonds']:<8.1f} {r['bond_mean']:<12.3f} "
                  f"{r['disconnected']:<10.1f} {r['test_loss']:<12.4f}")


if __name__ == '__main__':
    main()
