"""
Transformer-based architecture for atom coordinates.
"""

import torch
import torch.nn as nn
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings for diffusion timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AtomwiseTransformer(nn.Module):
    """
    Transformer-based architecture for atom coordinates.
    Uses self-attention to model interactions between atoms.
    """
    
    def __init__(self, 
                 n_atoms: int,
                 hidden_dim: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 time_dim: int = 64):
        super().__init__()
        
        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(3, hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)
        )
        
    def forward(self, x, t):
        """
        Args:
            x: (batch, n_atoms, 3) coordinates
            t: (batch,) timesteps
            
        Returns:
            noise_pred: (batch, n_atoms, 3) predicted noise
        """
        # Time embedding
        t_emb = self.time_mlp(t)  # (batch, hidden_dim)
        
        # Project input
        h = self.input_proj(x)  # (batch, n_atoms, hidden_dim)
        
        # Add time embedding to each atom
        h = h + t_emb.unsqueeze(1)
        
        # Transformer
        h = self.transformer(h)
        
        # Output projection
        noise_pred = self.output_proj(h)
        
        return noise_pred
