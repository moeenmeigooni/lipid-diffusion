"""
PyTorch Dataset for lipid conformations.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class LipidDataset(Dataset):
    """PyTorch Dataset for lipid conformations."""
    
    def __init__(self, coordinates: np.ndarray):
        """
        Args:
            coordinates: (n_samples, n_atoms, 3) array
        """
        self.coordinates = torch.FloatTensor(coordinates)
        
    def __len__(self):
        return len(self.coordinates)
    
    def __getitem__(self, idx):
        return self.coordinates[idx]
