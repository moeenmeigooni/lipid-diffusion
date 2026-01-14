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
                 time_dim: int = 64,
                 num_atom_types: int = 4,
                 use_atom_types: bool = True):
        super().__init__()

        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim
        self.use_atom_types = use_atom_types

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Atom type embedding
        if use_atom_types:
            self.atom_type_embedding = nn.Embedding(num_atom_types, hidden_dim // 2)
            coord_input_dim = 3 + hidden_dim // 2
        else:
            coord_input_dim = 3

        # Input projection
        self.input_proj = nn.Linear(coord_input_dim, hidden_dim)
        
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
        
    def forward(self, x, t, atom_types=None):
        """
        Args:
            x: (batch, n_atoms, 3) coordinates
            t: (batch,) timesteps
            atom_types: (batch, n_atoms) or (n_atoms,) atom type indices (optional)

        Returns:
            noise_pred: (batch, n_atoms, 3) predicted noise
        """
        batch_size = x.shape[0]

        # Time embedding
        t_emb = self.time_mlp(t)  # (batch, hidden_dim)

        # Combine coordinates with atom type features
        if self.use_atom_types and atom_types is not None:
            # Handle both batched and unbatched atom_types
            if atom_types.dim() == 1:
                atom_types = atom_types.unsqueeze(0).expand(batch_size, -1)

            atom_type_feats = self.atom_type_embedding(atom_types)  # (batch, n_atoms, hidden_dim//2)
            coord_input = torch.cat([x, atom_type_feats], dim=-1)  # (batch, n_atoms, 3 + hidden_dim//2)
        else:
            coord_input = x

        # Project input
        h = self.input_proj(coord_input)  # (batch, n_atoms, hidden_dim)

        # Add time embedding to each atom
        h = h + t_emb.unsqueeze(1)
        
        # Transformer
        h = self.transformer(h)
        
        # Output projection
        noise_pred = self.output_proj(h)
        
        return noise_pred
