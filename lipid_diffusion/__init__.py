"""
POPC Lipid Diffusion Model
===========================
A package for training diffusion models on lipid conformations.
"""

__version__ = "0.1.0"

from .data.preprocessor import LipidCoordinatePreprocessor
from .data.dataset import LipidDataset
from .models.transformer import AtomwiseTransformer
from .models.diffusion import DiffusionModel
from .training.trainer import train_diffusion_model, generate_samples

__all__ = [
    'LipidCoordinatePreprocessor',
    'LipidDataset',
    'AtomwiseTransformer',
    'DiffusionModel',
    'train_diffusion_model',
    'generate_samples',
]
