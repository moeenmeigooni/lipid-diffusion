"""
PyTorch Dataset for lipid conformations.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class LipidDataset(Dataset):
    """PyTorch Dataset for lipid conformations with optional atom type information."""

    def __init__(self, coordinates: np.ndarray, atom_types: np.ndarray = None):
        """
        Args:
            coordinates: (n_samples, n_atoms, 3) array of coordinates
            atom_types: (n_atoms,) array of element symbols (e.g., ['C', 'N', 'O', 'P'])
                       If provided, will be converted to indices for embedding lookup
        """
        self.coordinates = torch.FloatTensor(coordinates)

        # Create atom type vocabulary and convert to indices
        if atom_types is not None:
            # Get unique atom types in sorted order for consistent indexing
            unique_types = sorted(set(atom_types))  # e.g., ['C', 'N', 'O', 'P']
            self.atom_type_vocab = {elem: idx for idx, elem in enumerate(unique_types)}
            self.atom_type_indices = torch.LongTensor([
                self.atom_type_vocab[elem] for elem in atom_types
            ])
            self.num_atom_types = len(unique_types)
            print(f"Created atom type vocabulary: {self.atom_type_vocab}")
        else:
            self.atom_type_indices = None
            self.atom_type_vocab = None
            self.num_atom_types = 0

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        if self.atom_type_indices is not None:
            return self.coordinates[idx], self.atom_type_indices
        return self.coordinates[idx]
