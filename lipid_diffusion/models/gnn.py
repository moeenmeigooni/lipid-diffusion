"""
Graph Neural Network architecture for lipid conformations.
Uses message passing and attention mechanisms to model atomic interactions.
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


class EdgeFeatureExtractor(nn.Module):
    """
    Extract edge features between atoms based on distances and relative positions.
    """
    
    def __init__(self, hidden_dim: int, num_rbf: int = 16, cutoff: float = 10.0):
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        
        # Radial basis functions for distance encoding
        self.rbf_centers = nn.Parameter(
            torch.linspace(0, cutoff, num_rbf),
            requires_grad=False
        )
        self.rbf_width = nn.Parameter(
            torch.tensor(cutoff / num_rbf),
            requires_grad=False
        )
        
        # Edge feature MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(num_rbf + 3, hidden_dim),  # RBF + relative position
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def rbf_encoding(self, distances):
        """Encode distances using Radial Basis Functions."""
        # distances: (batch, n_atoms, n_atoms)
        # rbf_centers: (num_rbf,)
        distances = distances.unsqueeze(-1)  # (batch, n_atoms, n_atoms, 1)
        centers = self.rbf_centers.view(1, 1, 1, -1)  # (1, 1, 1, num_rbf)
        
        # Gaussian RBF
        rbf = torch.exp(-((distances - centers) ** 2) / (2 * self.rbf_width ** 2))
        return rbf  # (batch, n_atoms, n_atoms, num_rbf)
    
    def forward(self, coords):
        """
        Extract edge features from atomic coordinates.
        
        Args:
            coords: (batch, n_atoms, 3) atomic coordinates
            
        Returns:
            edge_features: (batch, n_atoms, n_atoms, hidden_dim)
            edge_mask: (batch, n_atoms, n_atoms) - True for edges within cutoff
        """
        batch_size, n_atoms = coords.shape[0], coords.shape[1]
        
        # Compute pairwise distances and relative positions
        # coords_i: (batch, n_atoms, 1, 3)
        # coords_j: (batch, 1, n_atoms, 3)
        coords_i = coords.unsqueeze(2)
        coords_j = coords.unsqueeze(1)
        
        relative_pos = coords_j - coords_i  # (batch, n_atoms, n_atoms, 3)
        distances = torch.norm(relative_pos, dim=-1)  # (batch, n_atoms, n_atoms)
        
        # Create edge mask (within cutoff distance)
        edge_mask = (distances < self.cutoff) & (distances > 0)  # Exclude self-loops
        
        # RBF encoding of distances
        rbf_features = self.rbf_encoding(distances)  # (batch, n_atoms, n_atoms, num_rbf)
        
        # Normalize relative positions
        relative_pos_normalized = relative_pos / (distances.unsqueeze(-1) + 1e-8)
        
        # Combine RBF and relative position features
        edge_input = torch.cat([
            rbf_features,
            relative_pos_normalized
        ], dim=-1)  # (batch, n_atoms, n_atoms, num_rbf + 3)
        
        # Apply edge MLP
        edge_features = self.edge_mlp(edge_input)  # (batch, n_atoms, n_atoms, hidden_dim)
        
        return edge_features, edge_mask


class GraphAttentionLayer(nn.Module):
    """
    Graph attention layer with edge features.
    Similar to Graph Attention Networks (GAT) but with edge information.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge feature projection
        self.edge_proj = nn.Linear(hidden_dim, num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, node_features, edge_features, edge_mask):
        """
        Args:
            node_features: (batch, n_atoms, hidden_dim)
            edge_features: (batch, n_atoms, n_atoms, hidden_dim)
            edge_mask: (batch, n_atoms, n_atoms)
            
        Returns:
            updated_features: (batch, n_atoms, hidden_dim)
        """
        batch_size, n_atoms, _ = node_features.shape
        
        # Project to Q, K, V
        Q = self.q_proj(node_features)  # (batch, n_atoms, hidden_dim)
        K = self.k_proj(node_features)
        V = self.v_proj(node_features)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, n_atoms, self.num_heads, self.head_dim)
        K = K.view(batch_size, n_atoms, self.num_heads, self.head_dim)
        V = V.view(batch_size, n_atoms, self.num_heads, self.head_dim)
        
        # Compute attention scores
        # Q: (batch, n_atoms, num_heads, head_dim)
        # K: (batch, n_atoms, num_heads, head_dim)
        Q = Q.transpose(1, 2)  # (batch, num_heads, n_atoms, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention: Q @ K^T
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        # attn_scores: (batch, num_heads, n_atoms, n_atoms)
        
        # Add edge features to attention
        edge_bias = self.edge_proj(edge_features)  # (batch, n_atoms, n_atoms, num_heads)
        edge_bias = edge_bias.permute(0, 3, 1, 2)  # (batch, num_heads, n_atoms, n_atoms)
        attn_scores = attn_scores + edge_bias
        
        # Apply edge mask
        attn_mask = edge_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)  # Handle NaN from all -inf
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (batch, num_heads, n_atoms, head_dim)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous()  # (batch, n_atoms, num_heads, head_dim)
        out = out.view(batch_size, n_atoms, self.hidden_dim)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class GraphConvBlock(nn.Module):
    """
    Graph convolutional block with attention and feedforward network.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.attention = GraphAttentionLayer(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, node_features, edge_features, edge_mask):
        """
        Args:
            node_features: (batch, n_atoms, hidden_dim)
            edge_features: (batch, n_atoms, n_atoms, hidden_dim)
            edge_mask: (batch, n_atoms, n_atoms)
            
        Returns:
            updated_features: (batch, n_atoms, hidden_dim)
        """
        # Attention with residual
        attn_out = self.attention(node_features, edge_features, edge_mask)
        node_features = self.norm1(node_features + attn_out)
        
        # Feedforward with residual
        ffn_out = self.ffn(node_features)
        node_features = self.norm2(node_features + ffn_out)
        
        return node_features


class LipidGraphNetwork(nn.Module):
    """
    Graph Neural Network for lipid conformations using message passing.
    
    Architecture:
    1. Initial node features from coordinates
    2. Extract edge features (distances, relative positions)
    3. Multiple graph convolution blocks with attention
    4. Predict noise for each atom
    """
    
    def __init__(
        self,
        n_atoms: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        num_rbf: int = 16,
        cutoff: float = 10.0,
        time_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Args:
            n_atoms: Number of atoms in lipid
            hidden_dim: Hidden dimension size
            num_layers: Number of graph conv layers
            num_heads: Number of attention heads
            num_rbf: Number of radial basis functions for distance encoding
            cutoff: Distance cutoff for edges (Angstroms)
            time_dim: Dimension for time embeddings
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initial node feature projection (from 3D coordinates)
        self.node_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge feature extractor
        self.edge_extractor = EdgeFeatureExtractor(hidden_dim, num_rbf, cutoff)
        
        # Graph convolution blocks
        self.conv_blocks = nn.ModuleList([
            GraphConvBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head to predict noise
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)  # Predict 3D noise vector
        )
        
    def forward(self, x, t):
        """
        Args:
            x: (batch, n_atoms, 3) coordinates
            t: (batch,) timesteps
            
        Returns:
            noise_pred: (batch, n_atoms, 3) predicted noise
        """
        batch_size = x.shape[0]
        
        # Time embedding
        t_emb = self.time_mlp(t)  # (batch, hidden_dim)
        
        # Initial node features from coordinates
        node_features = self.node_embedding(x)  # (batch, n_atoms, hidden_dim)
        
        # Add time embedding to all nodes
        node_features = node_features + t_emb.unsqueeze(1)
        
        # Extract edge features and mask
        edge_features, edge_mask = self.edge_extractor(x)
        # edge_features: (batch, n_atoms, n_atoms, hidden_dim)
        # edge_mask: (batch, n_atoms, n_atoms)
        
        # Apply graph convolution blocks
        for conv_block in self.conv_blocks:
            node_features = conv_block(node_features, edge_features, edge_mask)
        
        # Predict noise
        noise_pred = self.output_head(node_features)  # (batch, n_atoms, 3)
        
        return noise_pred


class EquivariantGraphNetwork(nn.Module):
    """
    E(3)-Equivariant Graph Network that respects rotational and translational symmetry.
    
    This is a more advanced version that ensures the output transforms correctly
    under rotations and translations of the input coordinates.
    
    Note: This is a simplified implementation. For production use, consider
    using libraries like e3nn or SEGNN.
    """
    
    def __init__(
        self,
        n_atoms: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_rbf: int = 16,
        cutoff: float = 10.0,
        time_dim: int = 64
    ):
        """
        Args:
            n_atoms: Number of atoms in lipid
            hidden_dim: Hidden dimension for scalar features
            num_layers: Number of message passing layers
            num_rbf: Number of radial basis functions
            cutoff: Distance cutoff for edges
            time_dim: Dimension for time embeddings
        """
        super().__init__()
        
        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        
        # Time embedding (scalar)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initial scalar features (invariant)
        self.scalar_embedding = nn.Linear(1, hidden_dim)  # Just a placeholder
        
        # RBF for distance encoding
        self.num_rbf = num_rbf
        self.rbf_centers = nn.Parameter(
            torch.linspace(0, cutoff, num_rbf),
            requires_grad=False
        )
        self.rbf_width = nn.Parameter(
            torch.tensor(cutoff / num_rbf),
            requires_grad=False
        )
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
        # Update layers for coordinates (equivariant)
        self.coord_update_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)  # Weight for coordinate update
            )
            for _ in range(num_layers)
        ])
        
        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def rbf_encoding(self, distances):
        """Encode distances using RBF."""
        distances = distances.unsqueeze(-1)
        centers = self.rbf_centers.view(1, 1, 1, -1)
        rbf = torch.exp(-((distances - centers) ** 2) / (2 * self.rbf_width ** 2))
        return rbf
    
    def forward(self, x, t):
        """
        Args:
            x: (batch, n_atoms, 3) coordinates
            t: (batch,) timesteps
            
        Returns:
            noise_pred: (batch, n_atoms, 3) predicted noise
        """
        batch_size, n_atoms = x.shape[0], x.shape[1]
        
        # Time embedding
        t_emb = self.time_mlp(t)  # (batch, hidden_dim)
        
        # Initialize scalar features
        h = torch.zeros(batch_size, n_atoms, self.hidden_dim, device=x.device)
        h = h + t_emb.unsqueeze(1)
        
        # Coordinate updates accumulator
        coord_updates = torch.zeros_like(x)
        
        # Message passing
        for msg_layer, coord_layer in zip(self.message_layers, self.coord_update_layers):
            # Compute pairwise distances and directions
            x_i = x.unsqueeze(2)  # (batch, n_atoms, 1, 3)
            x_j = x.unsqueeze(1)  # (batch, 1, n_atoms, 3)
            
            diff = x_j - x_i  # (batch, n_atoms, n_atoms, 3)
            dist = torch.norm(diff, dim=-1)  # (batch, n_atoms, n_atoms)
            
            # Edge mask
            edge_mask = (dist < self.cutoff) & (dist > 0)
            
            # RBF encoding
            rbf = self.rbf_encoding(dist)  # (batch, n_atoms, n_atoms, num_rbf)
            
            # Build messages (invariant features)
            h_i = h.unsqueeze(2).expand(-1, -1, n_atoms, -1)
            h_j = h.unsqueeze(1).expand(-1, n_atoms, -1, -1)
            
            msg_input = torch.cat([h_i, h_j, rbf], dim=-1)
            messages = msg_layer(msg_input)  # (batch, n_atoms, n_atoms, hidden_dim)
            
            # Apply edge mask and aggregate
            messages = messages * edge_mask.unsqueeze(-1)
            h = h + messages.sum(dim=2)  # Aggregate messages
            
            # Coordinate updates (equivariant)
            weights = coord_layer(messages)  # (batch, n_atoms, n_atoms, 1)
            weights = weights * edge_mask.unsqueeze(-1)
            
            # Normalize directions
            directions = diff / (dist.unsqueeze(-1) + 1e-8)
            coord_update = (weights * directions).sum(dim=2)
            coord_updates = coord_updates + coord_update
        
        # Final prediction combines coordinate updates
        # This maintains equivariance
        noise_pred = coord_updates
        
        return noise_pred
