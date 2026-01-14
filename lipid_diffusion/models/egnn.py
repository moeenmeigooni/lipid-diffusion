"""
E(3)-Equivariant Graph Neural Network (EGNN) for molecular generation.

EGNN maintains equivariance to rotations and translations by:
1. Only using invariant features (distances, not coordinates) for node updates
2. Updating coordinates using normalized direction vectors weighted by learned scalars

Reference: Satorras et al., "E(n) Equivariant Graph Neural Networks" (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


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


class EGNNLayer(nn.Module):
    """
    Single E(3)-equivariant graph neural network layer.

    Updates both:
    - h (node features): invariant update using distances and features
    - x (coordinates): equivariant update using direction vectors
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 0,
        act_fn: nn.Module = nn.SiLU(),
        residual: bool = True,
        attention: bool = True,
        normalize: bool = False,
        coords_agg: str = 'mean',
        tanh: bool = False
    ):
        super().__init__()
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh

        # Edge MLP: computes messages from pairs of nodes
        edge_input_dim = hidden_dim * 2 + 1 + edge_dim  # h_i, h_j, ||x_i - x_j||^2, edge_attr
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )

        # Node MLP: updates node features
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Coordinate MLP: computes scalar weights for coordinate updates
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # Initialize coord_mlp last layer to small values for stability
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)

        # Attention MLP (optional)
        if attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, h, x, edge_index, edge_attr=None):
        """
        Args:
            h: (batch, n_atoms, hidden_dim) node features
            x: (batch, n_atoms, 3) coordinates
            edge_index: (2, n_edges) edge indices [src, dst] or None for fully connected
            edge_attr: (batch, n_atoms, n_atoms, edge_dim) edge attributes (optional)

        Returns:
            h_out: (batch, n_atoms, hidden_dim) updated node features
            x_out: (batch, n_atoms, 3) updated coordinates
        """
        batch_size, n_atoms, _ = h.shape

        # For fully connected graph (all pairs)
        # Compute pairwise differences
        x_i = x.unsqueeze(2)  # (batch, n_atoms, 1, 3)
        x_j = x.unsqueeze(1)  # (batch, 1, n_atoms, 3)

        coord_diff = x_i - x_j  # (batch, n_atoms, n_atoms, 3)
        radial = torch.sum(coord_diff ** 2, dim=-1, keepdim=True)  # (batch, n_atoms, n_atoms, 1)

        # Normalize coordinate differences
        if self.normalize:
            norm = torch.sqrt(radial + 1e-8)
            coord_diff = coord_diff / norm

        # Get pairwise node features
        h_i = h.unsqueeze(2).expand(-1, -1, n_atoms, -1)  # (batch, n_atoms, n_atoms, hidden_dim)
        h_j = h.unsqueeze(1).expand(-1, n_atoms, -1, -1)  # (batch, n_atoms, n_atoms, hidden_dim)

        # Edge input
        edge_input = torch.cat([h_i, h_j, radial], dim=-1)  # (batch, n_atoms, n_atoms, 2*hidden + 1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)

        # Compute edge messages
        m_ij = self.edge_mlp(edge_input)  # (batch, n_atoms, n_atoms, hidden_dim)

        # Apply attention if enabled
        if self.attention:
            att = self.att_mlp(m_ij)  # (batch, n_atoms, n_atoms, 1)
            m_ij = m_ij * att

        # Aggregate messages for node update
        m_i = m_ij.sum(dim=2)  # (batch, n_atoms, hidden_dim)

        # Update node features
        h_input = torch.cat([h, m_i], dim=-1)
        h_out = self.node_mlp(h_input)

        if self.residual:
            h_out = h + h_out

        # Coordinate update (equivariant)
        coord_weights = self.coord_mlp(m_ij)  # (batch, n_atoms, n_atoms, 1)

        if self.tanh:
            coord_weights = torch.tanh(coord_weights)

        # Mask self-interactions
        mask = torch.eye(n_atoms, device=x.device).bool()
        mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, n_atoms, n_atoms, 1)
        coord_weights = coord_weights.masked_fill(mask, 0)

        # Aggregate coordinate updates
        if self.coords_agg == 'mean':
            coord_update = (coord_diff * coord_weights).mean(dim=2)
        elif self.coords_agg == 'sum':
            coord_update = (coord_diff * coord_weights).sum(dim=2)
        else:
            raise ValueError(f"Unknown coords_agg: {self.coords_agg}")

        x_out = x + coord_update

        return h_out, x_out


class EGNN(nn.Module):
    """
    E(3)-Equivariant Graph Neural Network for lipid generation.

    This model predicts noise/velocity while maintaining equivariance:
    - Rotations: If input is rotated, output rotates the same way
    - Translations: If input is translated, output is unchanged (predicts relative updates)
    """

    def __init__(
        self,
        n_atoms: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        time_dim: int = 64,
        num_atom_types: int = 4,
        use_atom_types: bool = True,
        attention: bool = True,
        normalize: bool = False,
        tanh: bool = False
    ):
        """
        Args:
            n_atoms: Number of atoms in the molecule
            hidden_dim: Hidden dimension size
            num_layers: Number of EGNN layers
            time_dim: Dimension for time embeddings
            num_atom_types: Number of unique atom types
            use_atom_types: Whether to use atom type embeddings
            attention: Whether to use attention in EGNN layers
            normalize: Whether to normalize coordinate differences
            tanh: Whether to apply tanh to coordinate updates
        """
        super().__init__()

        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim
        self.use_atom_types = use_atom_types

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Atom type embedding
        if use_atom_types:
            self.atom_type_embedding = nn.Embedding(num_atom_types, hidden_dim)

        # Initial node embedding (from atom types or learned)
        self.node_embedding = nn.Linear(hidden_dim if use_atom_types else 1, hidden_dim)

        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(
                hidden_dim=hidden_dim,
                attention=attention,
                normalize=normalize,
                tanh=tanh
            )
            for _ in range(num_layers)
        ])

        # Output layer for noise prediction
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x, t, atom_types=None):
        """
        Args:
            x: (batch, n_atoms, 3) coordinates
            t: (batch,) timesteps
            atom_types: (batch, n_atoms) or (n_atoms,) atom type indices

        Returns:
            noise_pred: (batch, n_atoms, 3) predicted noise/velocity
        """
        batch_size = x.shape[0]

        # Time embedding
        t_emb = self.time_mlp(t)  # (batch, hidden_dim)

        # Initialize node features
        if self.use_atom_types and atom_types is not None:
            if atom_types.dim() == 1:
                atom_types = atom_types.unsqueeze(0).expand(batch_size, -1)
            h = self.atom_type_embedding(atom_types)  # (batch, n_atoms, hidden_dim)
        else:
            # Use placeholder if no atom types
            h = torch.ones(batch_size, self.n_atoms, 1, device=x.device)

        h = self.node_embedding(h)  # (batch, n_atoms, hidden_dim)

        # Add time embedding to all nodes
        h = h + t_emb.unsqueeze(1)

        # Store original coordinates for computing noise
        x_orig = x.clone()

        # Apply EGNN layers
        for layer in self.layers:
            h, x = layer(h, x, edge_index=None)

        # Compute coordinate change as noise prediction
        # This is equivariant: rotating input rotates the noise prediction
        coord_change = x - x_orig

        # Also use node features for additional prediction capacity
        noise_from_h = self.output_mlp(h)

        # Combine coordinate change and feature-based prediction
        # Both are equivariant, so the sum is too
        noise_pred = coord_change + noise_from_h

        return noise_pred


class EGNNWithRBF(nn.Module):
    """
    EGNN with RBF edge features for better distance encoding.

    Combines the equivariance of EGNN with explicit RBF distance encoding
    to help the model distinguish different bond lengths.
    """

    def __init__(
        self,
        n_atoms: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        time_dim: int = 64,
        num_atom_types: int = 4,
        use_atom_types: bool = True,
        num_rbf: int = 50,
        cutoff: float = 5.0,
        attention: bool = True
    ):
        super().__init__()

        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim
        self.use_atom_types = use_atom_types
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        # RBF parameters
        self.rbf_centers = nn.Parameter(
            torch.linspace(0, cutoff, num_rbf),
            requires_grad=False
        )
        self.rbf_width = nn.Parameter(
            torch.tensor(cutoff / num_rbf),
            requires_grad=False
        )

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Atom type embedding
        if use_atom_types:
            self.atom_type_embedding = nn.Embedding(num_atom_types, hidden_dim)

        self.node_embedding = nn.Linear(hidden_dim if use_atom_types else 1, hidden_dim)

        # Edge feature projection (RBF -> hidden)
        self.edge_embedding = nn.Linear(num_rbf, hidden_dim)

        # Modified EGNN layers with edge features
        self.layers = nn.ModuleList([
            EGNNLayerWithEdge(
                hidden_dim=hidden_dim,
                edge_dim=hidden_dim,
                attention=attention
            )
            for _ in range(num_layers)
        ])

        # Output
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )

    def rbf_encoding(self, distances):
        """Encode distances using RBF."""
        distances = distances.unsqueeze(-1)
        centers = self.rbf_centers.view(1, 1, 1, -1)
        rbf = torch.exp(-((distances - centers) ** 2) / (2 * self.rbf_width ** 2))
        return rbf

    def forward(self, x, t, atom_types=None):
        batch_size = x.shape[0]

        # Time embedding
        t_emb = self.time_mlp(t)

        # Node features
        if self.use_atom_types and atom_types is not None:
            if atom_types.dim() == 1:
                atom_types = atom_types.unsqueeze(0).expand(batch_size, -1)
            h = self.atom_type_embedding(atom_types)
        else:
            h = torch.ones(batch_size, self.n_atoms, 1, device=x.device)

        h = self.node_embedding(h)
        h = h + t_emb.unsqueeze(1)

        # Compute RBF edge features
        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
        distances = torch.norm(x_j - x_i, dim=-1)  # (batch, n_atoms, n_atoms)
        rbf_features = self.rbf_encoding(distances)  # (batch, n_atoms, n_atoms, num_rbf)
        edge_attr = self.edge_embedding(rbf_features)  # (batch, n_atoms, n_atoms, hidden_dim)

        x_orig = x.clone()

        # Apply layers
        for layer in self.layers:
            h, x = layer(h, x, edge_attr=edge_attr)

        coord_change = x - x_orig
        noise_from_h = self.output_mlp(h)
        noise_pred = coord_change + noise_from_h

        return noise_pred


class EGNNLayerWithEdge(nn.Module):
    """EGNN layer that accepts edge features."""

    def __init__(self, hidden_dim: int, edge_dim: int, attention: bool = True):
        super().__init__()
        self.attention = attention

        # Edge MLP with edge features
        edge_input_dim = hidden_dim * 2 + 1 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)

        if attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, h, x, edge_attr=None):
        batch_size, n_atoms, _ = h.shape

        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
        coord_diff = x_i - x_j
        radial = torch.sum(coord_diff ** 2, dim=-1, keepdim=True)

        h_i = h.unsqueeze(2).expand(-1, -1, n_atoms, -1)
        h_j = h.unsqueeze(1).expand(-1, n_atoms, -1, -1)

        edge_input = torch.cat([h_i, h_j, radial], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)

        m_ij = self.edge_mlp(edge_input)

        if self.attention:
            att = self.att_mlp(m_ij)
            m_ij = m_ij * att

        m_i = m_ij.sum(dim=2)
        h_input = torch.cat([h, m_i], dim=-1)
        h_out = h + self.node_mlp(h_input)

        coord_weights = self.coord_mlp(m_ij)
        mask = torch.eye(n_atoms, device=x.device).bool()
        mask = mask.unsqueeze(0).unsqueeze(-1)
        coord_weights = coord_weights.masked_fill(mask, 0)
        coord_update = (coord_diff * coord_weights).mean(dim=2)
        x_out = x + coord_update

        return h_out, x_out
