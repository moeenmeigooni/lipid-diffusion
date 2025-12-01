"""
Lipid coordinate preprocessing utilities.
"""

import numpy as np
from typing import List


class LipidCoordinatePreprocessor:
    """
    Preprocesses lipid coordinate data for diffusion modeling.
    Handles alignment, normalization, and featurization.
    """
    
    def __init__(self, align_method='kabsch'):
        self.align_method = align_method
        self.mean = None
        self.std = None
        self.reference_structure = None
        
    def load_coordinates(self, file_path: str, format='xyz') -> np.ndarray:
        """
        Load lipid coordinates from various file formats.
        
        Args:
            file_path: Path to coordinate file
            format: 'xyz', 'pdb', or 'npy'
            
        Returns:
            coords: (n_atoms, 3) array of coordinates
        """
        if format == 'xyz':
            # XYZ format: atom_symbol x y z
            data = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines[2:]:  # Skip first two lines (count and comment)
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        data.append([float(parts[1]), float(parts[2]), float(parts[3])])
            return np.array(data)
            
        elif format == 'pdb':
            # Simple PDB parser for ATOM records
            coords = []
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        spl = line.split()
                        x = float(spl[6])
                        y = float(spl[7])
                        z = float(spl[8])
                        coords.append([x, y, z])
            return np.array(coords)
            
        elif format == 'npy':
            return np.load(file_path)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def kabsch_align(self, coords: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Align coordinates to reference using Kabsch algorithm.
        
        Args:
            coords: (n_atoms, 3) coordinates to align
            reference: (n_atoms, 3) reference coordinates
            
        Returns:
            aligned_coords: (n_atoms, 3) aligned coordinates
        """
        # Center both structures
        coords_centered = coords - coords.mean(axis=0)
        ref_centered = reference - reference.mean(axis=0)
        
        # Compute covariance matrix
        H = coords_centered.T @ ref_centered
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        
        # Apply rotation
        aligned = coords_centered @ R
        
        return aligned + reference.mean(axis=0)
    
    def process_dataset(self, 
                       file_paths: List[str], 
                       format='xyz',
                       align=True) -> np.ndarray:
        """
        Process multiple conformations into a dataset.
        
        Args:
            file_paths: List of paths to coordinate files
            format: File format
            align: Whether to align all structures
            
        Returns:
            dataset: (n_conformations, n_atoms, 3) array
        """
        conformations = []
        
        # Load first structure as reference
        first_coords = self.load_coordinates(file_paths[0], format)
        self.reference_structure = first_coords
        conformations.append(first_coords)
        
        # Load and optionally align remaining structures
        for file_path in file_paths[1:]:
            coords = self.load_coordinates(file_path, format)
            
            if align and self.align_method == 'kabsch':
                coords = self.kabsch_align(coords, self.reference_structure)
            
            conformations.append(coords)
        
        dataset = np.array(conformations)
        
        # Normalize
        self.mean = dataset.mean(axis=(0, 1))
        self.std = dataset.std(axis=(0, 1))
        dataset_normalized = (dataset - self.mean) / (self.std + 1e-8)
        
        return dataset_normalized
    
    def denormalize(self, coords: np.ndarray) -> np.ndarray:
        """Denormalize coordinates back to original scale."""
        return coords * (self.std + 1e-8) + self.mean
