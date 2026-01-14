"""
Bond-Aware Graph Neural Network for molecular generation.

Instead of using distance-based edges, this model uses explicit molecular bonds
defined from the molecular topology. This ensures the model knows which atoms
are actually bonded and can preserve these relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings."""

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


# POPC bond connectivity (0-indexed atom pairs)
# This defines which atoms are bonded in the POPC molecule (non-hydrogen atoms only)
POPC_BONDS = [
    # Head group
    (0, 1),    # N - C12
    (0, 4),    # N - C13
    (0, 8),    # N - C14
    (0, 12),   # N - C15 (actually C11 in sequence)
    (1, 16),   # C12 - C11
    (16, 19),  # C11 - P
    (19, 20),  # P - O13
    (19, 21),  # P - O14
    (19, 22),  # P - O12
    (22, 23),  # O12 - O11
    (23, 24),  # O11 - C1
    (24, 27),  # C1 - C2
    (27, 29),  # C2 - O21
    (29, 30),  # O21 - C21
    (30, 31),  # C21 - O22
    (30, 32),  # C21 - C22
    (24, 35),  # C1 - C3
    (35, 38),  # C3 - O31
    (38, 39),  # O31 - C31
    (39, 40),  # C31 - O32
    (39, 41),  # C31 - C32
    # Acyl chain 1 (sn-2)
    (32, 44),  # C22 - C23
    (44, 47),  # C23 - C24
    (47, 50),  # C24 - C25
    # Continue chain...
]

# Alternative: Use a function to infer bonds from reference structure
def infer_bonds_from_structure(coords: np.ndarray, max_bond_length: float = 1.8) -> List[Tuple[int, int]]:
    """
    Infer bonds from a reference structure based on interatomic distances.

    Args:
        coords: (n_atoms, 3) coordinates of a reference structure
        max_bond_length: Maximum distance to consider as a bond (Angstroms)

    Returns:
        List of (i, j) tuples representing bonded atom pairs
    """
    n_atoms = len(coords)
    bonds = []

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < max_bond_length:
                bonds.append((i, j))

    return bonds


class BondMessagePassing(nn.Module):
    """
    Message passing layer that operates on bond graph.

    Messages are passed along defined molecular bonds rather than
    distance-based edges, ensuring topology preservation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Message computation
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),  # h_i, h_j, distance
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Attention for message weighting
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, num_heads),
            nn.Softmax(dim=-1)
        )

        # Node update
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, x, bond_indices, bond_mask=None):
        """
        Args:
            h: (batch, n_atoms, hidden_dim) node features
            x: (batch, n_atoms, 3) coordinates
            bond_indices: (n_bonds, 2) tensor of bonded atom pairs
            bond_mask: (batch, n_bonds) optional mask for bonds

        Returns:
            h_out: (batch, n_atoms, hidden_dim) updated features
        """
        batch_size, n_atoms, _ = h.shape
        n_bonds = bond_indices.shape[0]
        device = h.device

        # Get source and target indices
        src_idx = bond_indices[:, 0]  # (n_bonds,)
        dst_idx = bond_indices[:, 1]  # (n_bonds,)

        # Gather features for bonded pairs
        # h_src: (batch, n_bonds, hidden_dim)
        h_src = h[:, src_idx, :]
        h_dst = h[:, dst_idx, :]

        # Compute distances for bonded pairs
        x_src = x[:, src_idx, :]  # (batch, n_bonds, 3)
        x_dst = x[:, dst_idx, :]
        bond_distances = torch.norm(x_dst - x_src, dim=-1, keepdim=True)  # (batch, n_bonds, 1)

        # Compute messages
        message_input = torch.cat([h_src, h_dst, bond_distances], dim=-1)
        messages = self.message_mlp(message_input)  # (batch, n_bonds, hidden_dim)

        # Apply attention
        attn_weights = self.attention(messages)  # (batch, n_bonds, num_heads)
        messages = messages.unsqueeze(-1) * attn_weights.unsqueeze(-2)  # (batch, n_bonds, hidden_dim, num_heads)
        messages = messages.sum(dim=-1)  # (batch, n_bonds, hidden_dim)

        # Apply bond mask if provided
        if bond_mask is not None:
            messages = messages * bond_mask.unsqueeze(-1)

        # Aggregate messages to nodes (scatter add)
        # Need to aggregate messages to both src and dst (undirected bonds)
        aggregated = torch.zeros(batch_size, n_atoms, self.hidden_dim, device=device)

        # Add messages to destination nodes
        dst_idx_expanded = dst_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.hidden_dim)
        aggregated.scatter_add_(1, dst_idx_expanded, messages)

        # Also add to source nodes (undirected)
        src_idx_expanded = src_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.hidden_dim)
        aggregated.scatter_add_(1, src_idx_expanded, messages)

        # Update node features
        h_input = torch.cat([h, aggregated], dim=-1)
        h_out = self.node_update(h_input)
        h_out = self.dropout(h_out)
        h_out = self.norm(h + h_out)  # Residual connection

        return h_out


class BondAwareGNN(nn.Module):
    """
    Graph Neural Network that uses explicit molecular bonds for message passing.

    Key differences from distance-based GNN:
    1. Edges are defined by molecular topology, not distance cutoff
    2. Bond distances are used as edge features but don't determine connectivity
    3. This ensures the model "knows" which atoms should be bonded
    """

    def __init__(
        self,
        n_atoms: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        time_dim: int = 64,
        dropout: float = 0.1,
        num_atom_types: int = 4,
        use_atom_types: bool = True,
        bond_indices: Optional[torch.Tensor] = None
    ):
        """
        Args:
            n_atoms: Number of atoms
            hidden_dim: Hidden dimension
            num_layers: Number of message passing layers
            num_heads: Number of attention heads
            time_dim: Time embedding dimension
            dropout: Dropout rate
            num_atom_types: Number of atom types
            use_atom_types: Whether to use atom type embeddings
            bond_indices: (n_bonds, 2) tensor of bonded atom pairs
        """
        super().__init__()

        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim
        self.use_atom_types = use_atom_types

        # Store bond indices (will be set during initialization or first forward)
        if bond_indices is not None:
            self.register_buffer('bond_indices', bond_indices)
        else:
            self.bond_indices = None

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Atom type embedding
        if use_atom_types:
            self.atom_type_embedding = nn.Embedding(num_atom_types, hidden_dim // 2)
            input_dim = 3 + hidden_dim // 2
        else:
            input_dim = 3

        # Initial node embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Bond message passing layers
        self.layers = nn.ModuleList([
            BondMessagePassing(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Also add global attention layers for long-range interactions
        self.global_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.global_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )

    def set_bond_indices(self, bond_indices: torch.Tensor):
        """Set the bond indices from a reference structure."""
        self.register_buffer('bond_indices', bond_indices)

    def forward(self, x, t, atom_types=None):
        """
        Args:
            x: (batch, n_atoms, 3) coordinates
            t: (batch,) timesteps
            atom_types: (batch, n_atoms) or (n_atoms,) atom type indices

        Returns:
            noise_pred: (batch, n_atoms, 3) predicted noise
        """
        if self.bond_indices is None:
            raise ValueError("Bond indices not set. Call set_bond_indices() first.")

        batch_size = x.shape[0]

        # Time embedding
        t_emb = self.time_mlp(t)  # (batch, hidden_dim)

        # Prepare node input
        if self.use_atom_types and atom_types is not None:
            if atom_types.dim() == 1:
                atom_types = atom_types.unsqueeze(0).expand(batch_size, -1)
            atom_feats = self.atom_type_embedding(atom_types)
            node_input = torch.cat([x, atom_feats], dim=-1)
        else:
            node_input = x

        # Initial node features
        h = self.node_embedding(node_input)
        h = h + t_emb.unsqueeze(1)

        # Apply message passing layers
        for bond_layer, global_attn, global_norm in zip(
            self.layers, self.global_layers, self.global_norms
        ):
            # Bond-aware message passing
            h = bond_layer(h, x, self.bond_indices)

            # Global attention for long-range
            h_attn, _ = global_attn(h, h, h)
            h = global_norm(h + h_attn)

        # Predict noise
        noise_pred = self.output_head(h)

        return noise_pred


class BondAwareGNNWithCoordUpdate(nn.Module):
    """
    Bond-aware GNN that also updates coordinates like EGNN.

    Combines bond-aware message passing with equivariant coordinate updates.
    """

    def __init__(
        self,
        n_atoms: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        time_dim: int = 64,
        num_atom_types: int = 4,
        use_atom_types: bool = True,
        bond_indices: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim
        self.use_atom_types = use_atom_types

        if bond_indices is not None:
            self.register_buffer('bond_indices', bond_indices)
        else:
            self.bond_indices = None

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

        # Layers with coordinate updates
        self.layers = nn.ModuleList([
            BondAwareEGNNLayer(hidden_dim)
            for _ in range(num_layers)
        ])

        # Output
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )

    def set_bond_indices(self, bond_indices: torch.Tensor):
        self.register_buffer('bond_indices', bond_indices)

    def forward(self, x, t, atom_types=None):
        if self.bond_indices is None:
            raise ValueError("Bond indices not set.")

        batch_size = x.shape[0]

        t_emb = self.time_mlp(t)

        if self.use_atom_types and atom_types is not None:
            if atom_types.dim() == 1:
                atom_types = atom_types.unsqueeze(0).expand(batch_size, -1)
            h = self.atom_type_embedding(atom_types)
        else:
            h = torch.ones(batch_size, self.n_atoms, 1, device=x.device)

        h = self.node_embedding(h)
        h = h + t_emb.unsqueeze(1)

        x_orig = x.clone()

        for layer in self.layers:
            h, x = layer(h, x, self.bond_indices)

        coord_change = x - x_orig
        noise_from_h = self.output_mlp(h)
        noise_pred = coord_change + noise_from_h

        return noise_pred


class BondAwareEGNNLayer(nn.Module):
    """EGNN-style layer that operates on bond graph."""

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Coordinate update weights
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)

    def forward(self, h, x, bond_indices):
        batch_size, n_atoms, hidden_dim = h.shape
        n_bonds = bond_indices.shape[0]
        device = h.device

        src_idx = bond_indices[:, 0]
        dst_idx = bond_indices[:, 1]

        # Get bonded pairs
        h_src = h[:, src_idx, :]
        h_dst = h[:, dst_idx, :]
        x_src = x[:, src_idx, :]
        x_dst = x[:, dst_idx, :]

        # Compute messages
        coord_diff = x_dst - x_src
        dist = torch.norm(coord_diff, dim=-1, keepdim=True)

        message_input = torch.cat([h_src, h_dst, dist], dim=-1)
        messages = self.message_mlp(message_input)

        # Aggregate to nodes
        aggregated = torch.zeros(batch_size, n_atoms, hidden_dim, device=device)
        dst_expanded = dst_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, hidden_dim)
        src_expanded = src_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, hidden_dim)
        aggregated.scatter_add_(1, dst_expanded, messages)
        aggregated.scatter_add_(1, src_expanded, messages)

        # Update nodes
        h_input = torch.cat([h, aggregated], dim=-1)
        h_out = h + self.node_mlp(h_input)

        # Coordinate update (along bonds only)
        coord_weights = self.coord_mlp(messages)  # (batch, n_bonds, 1)

        # Normalize direction
        coord_diff_norm = coord_diff / (dist + 1e-8)

        # Compute coordinate updates for each bond
        coord_updates_to_dst = coord_diff_norm * coord_weights
        coord_updates_to_src = -coord_diff_norm * coord_weights

        # Aggregate coordinate updates
        coord_update = torch.zeros_like(x)
        dst_expanded_3 = dst_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 3)
        src_expanded_3 = src_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 3)
        coord_update.scatter_add_(1, dst_expanded_3, coord_updates_to_dst)
        coord_update.scatter_add_(1, src_expanded_3, coord_updates_to_src)

        # Average by number of bonds per atom
        bond_counts = torch.zeros(n_atoms, device=device)
        bond_counts.scatter_add_(0, dst_idx, torch.ones(n_bonds, device=device))
        bond_counts.scatter_add_(0, src_idx, torch.ones(n_bonds, device=device))
        bond_counts = bond_counts.clamp(min=1)

        coord_update = coord_update / bond_counts.view(1, -1, 1)
        x_out = x + coord_update

        return h_out, x_out
