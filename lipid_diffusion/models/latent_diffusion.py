"""
Latent Space Diffusion Model for molecular generation.

Instead of diffusing directly in coordinate space (where structure can break),
this model:
1. Encodes coordinates into a structured latent space
2. Performs diffusion/flow matching in latent space
3. Decodes back to coordinates

The encoder/decoder can learn a representation that preserves molecular structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


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


class InvariantEncoder(nn.Module):
    """
    Encoder that maps coordinates to a rotation-invariant latent space.

    Uses pairwise distances and angles as input to ensure the latent
    representation doesn't depend on the orientation of the molecule.
    """

    def __init__(
        self,
        n_atoms: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_atom_types: int = 4,
        use_atom_types: bool = True
    ):
        super().__init__()

        self.n_atoms = n_atoms
        self.latent_dim = latent_dim
        self.use_atom_types = use_atom_types

        # Number of pairwise distances
        self.n_pairs = n_atoms * (n_atoms - 1) // 2

        # Atom type embedding
        if use_atom_types:
            self.atom_type_embedding = nn.Embedding(num_atom_types, hidden_dim // 4)
            atom_feat_dim = hidden_dim // 4
        else:
            atom_feat_dim = 0

        # Distance encoder
        self.distance_encoder = nn.Sequential(
            nn.Linear(self.n_pairs, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Per-atom encoder (local geometry)
        # Each atom gets its distances to all other atoms
        self.atom_encoder = nn.Sequential(
            nn.Linear(n_atoms - 1 + atom_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Transformer for combining information
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection to latent space
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Global pooling for molecule-level latent
        self.global_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def compute_distance_matrix(self, x):
        """Compute pairwise distance matrix."""
        # x: (batch, n_atoms, 3)
        x_i = x.unsqueeze(2)  # (batch, n_atoms, 1, 3)
        x_j = x.unsqueeze(1)  # (batch, 1, n_atoms, 3)
        dist_matrix = torch.norm(x_j - x_i, dim=-1)  # (batch, n_atoms, n_atoms)
        return dist_matrix

    def forward(self, x, atom_types=None):
        """
        Encode coordinates to latent space.

        Args:
            x: (batch, n_atoms, 3) coordinates
            atom_types: (batch, n_atoms) or (n_atoms,) atom type indices

        Returns:
            z: (batch, n_atoms, latent_dim) per-atom latent codes
            z_global: (batch, latent_dim) molecule-level latent code
        """
        batch_size = x.shape[0]

        # Compute distance matrix
        dist_matrix = self.compute_distance_matrix(x)  # (batch, n_atoms, n_atoms)

        # Per-atom features: distances to all other atoms
        # Remove diagonal (self-distances)
        mask = ~torch.eye(self.n_atoms, dtype=torch.bool, device=x.device)
        atom_dists = dist_matrix[:, :, mask[0]]  # (batch, n_atoms, n_atoms-1)

        # Add atom type features
        if self.use_atom_types and atom_types is not None:
            if atom_types.dim() == 1:
                atom_types = atom_types.unsqueeze(0).expand(batch_size, -1)
            atom_feats = self.atom_type_embedding(atom_types)  # (batch, n_atoms, hidden//4)
            atom_input = torch.cat([atom_dists, atom_feats], dim=-1)
        else:
            atom_input = atom_dists

        # Encode per-atom features
        h = self.atom_encoder(atom_input)  # (batch, n_atoms, hidden_dim)

        # Apply transformer
        h = self.transformer(h)

        # Per-atom latent codes
        z = self.to_latent(h)  # (batch, n_atoms, latent_dim)

        # Global latent code
        h_pooled = h.mean(dim=1)  # (batch, hidden_dim)
        z_global = self.global_latent(h_pooled)  # (batch, latent_dim)

        return z, z_global


class EquivariantDecoder(nn.Module):
    """
    Decoder that maps latent codes back to coordinates.

    Uses a process similar to distance geometry to reconstruct
    3D coordinates from the latent representation.
    """

    def __init__(
        self,
        n_atoms: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_atom_types: int = 4,
        use_atom_types: bool = True
    ):
        super().__init__()

        self.n_atoms = n_atoms
        self.latent_dim = latent_dim
        self.use_atom_types = use_atom_types

        # Atom type embedding
        if use_atom_types:
            self.atom_type_embedding = nn.Embedding(num_atom_types, hidden_dim // 4)
            input_dim = latent_dim + hidden_dim // 4
        else:
            input_dim = latent_dim

        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Global latent projection
        self.global_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Transformer for combining information
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # Predict pairwise distances
        self.dist_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive distances
        )

        # Direct coordinate prediction (as backup/refinement)
        self.coord_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, z, z_global, atom_types=None):
        """
        Decode latent codes to coordinates.

        Args:
            z: (batch, n_atoms, latent_dim) per-atom latent codes
            z_global: (batch, latent_dim) molecule-level latent code
            atom_types: (batch, n_atoms) or (n_atoms,) atom type indices

        Returns:
            x: (batch, n_atoms, 3) reconstructed coordinates
            dist_pred: (batch, n_atoms, n_atoms) predicted distance matrix
        """
        batch_size = z.shape[0]

        # Add atom type features
        if self.use_atom_types and atom_types is not None:
            if atom_types.dim() == 1:
                atom_types = atom_types.unsqueeze(0).expand(batch_size, -1)
            atom_feats = self.atom_type_embedding(atom_types)
            z_input = torch.cat([z, atom_feats], dim=-1)
        else:
            z_input = z

        # Project to hidden space
        h = self.input_proj(z_input)  # (batch, n_atoms, hidden_dim)

        # Add global information
        g = self.global_proj(z_global).unsqueeze(1)  # (batch, 1, hidden_dim)
        h = h + g

        # Apply transformer
        h = self.transformer(h)

        # Predict distances between all pairs
        h_i = h.unsqueeze(2).expand(-1, -1, self.n_atoms, -1)
        h_j = h.unsqueeze(1).expand(-1, self.n_atoms, -1, -1)
        h_pair = torch.cat([h_i, h_j], dim=-1)
        dist_pred = self.dist_predictor(h_pair).squeeze(-1)  # (batch, n_atoms, n_atoms)

        # Make distance matrix symmetric
        dist_pred = (dist_pred + dist_pred.transpose(1, 2)) / 2

        # Direct coordinate prediction
        x = self.coord_predictor(h)  # (batch, n_atoms, 3)

        return x, dist_pred


class LatentSpaceVAE(nn.Module):
    """
    Variational Autoencoder for molecular coordinates.

    Encodes to a structured latent space where diffusion can be performed.
    """

    def __init__(
        self,
        n_atoms: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_atom_types: int = 4,
        use_atom_types: bool = True
    ):
        super().__init__()

        self.n_atoms = n_atoms
        self.latent_dim = latent_dim

        self.encoder = InvariantEncoder(
            n_atoms, latent_dim, hidden_dim, num_layers,
            num_atom_types, use_atom_types
        )

        self.decoder = EquivariantDecoder(
            n_atoms, latent_dim, hidden_dim, num_layers,
            num_atom_types, use_atom_types
        )

        # VAE components
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)
        self.fc_mu_global = nn.Linear(latent_dim, latent_dim)
        self.fc_var_global = nn.Linear(latent_dim, latent_dim)

    def encode(self, x, atom_types=None):
        """Encode to latent distribution parameters."""
        z, z_global = self.encoder(x, atom_types)

        mu = self.fc_mu(z)
        log_var = self.fc_var(z)
        mu_global = self.fc_mu_global(z_global)
        log_var_global = self.fc_var_global(z_global)

        return mu, log_var, mu_global, log_var_global

    def reparameterize(self, mu, log_var):
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, z_global, atom_types=None):
        """Decode from latent space."""
        return self.decoder(z, z_global, atom_types)

    def forward(self, x, atom_types=None):
        """Full forward pass."""
        mu, log_var, mu_global, log_var_global = self.encode(x, atom_types)

        z = self.reparameterize(mu, log_var)
        z_global = self.reparameterize(mu_global, log_var_global)

        x_recon, dist_pred = self.decode(z, z_global, atom_types)

        return x_recon, dist_pred, mu, log_var, mu_global, log_var_global


class LatentDiffusionModel(nn.Module):
    """
    Diffusion model that operates in the latent space of a VAE.

    The idea:
    1. Train VAE to encode/decode molecular structures
    2. Perform diffusion in the latent space (which is more structured)
    3. Decode samples from latent space to get valid molecules
    """

    def __init__(
        self,
        n_atoms: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        time_dim: int = 64,
        num_atom_types: int = 4,
        use_atom_types: bool = True
    ):
        super().__init__()

        self.n_atoms = n_atoms
        self.latent_dim = latent_dim
        self.use_atom_types = use_atom_types

        # VAE for encoding/decoding
        self.vae = LatentSpaceVAE(
            n_atoms, latent_dim, hidden_dim, num_layers,
            num_atom_types, use_atom_types
        )

        # Latent diffusion model (denoiser in latent space)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Atom type embedding for latent denoiser
        if use_atom_types:
            self.atom_type_embedding = nn.Embedding(num_atom_types, hidden_dim // 4)

        # Latent denoiser (operates on per-atom latents)
        self.latent_denoiser = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim + (hidden_dim // 4 if use_atom_types else 0), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Global latent denoiser
        self.global_denoiser = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def encode(self, x, atom_types=None):
        """Encode coordinates to latent."""
        mu, log_var, mu_global, log_var_global = self.vae.encode(x, atom_types)
        # Use mean for deterministic encoding during generation
        return mu, mu_global

    def decode(self, z, z_global, atom_types=None):
        """Decode latent to coordinates."""
        x, dist_pred = self.vae.decode(z, z_global, atom_types)
        return x

    def denoise_latent(self, z_noisy, z_global_noisy, t, atom_types=None):
        """
        Predict noise in latent space.

        Args:
            z_noisy: (batch, n_atoms, latent_dim) noisy latent codes
            z_global_noisy: (batch, latent_dim) noisy global latent
            t: (batch,) timesteps
            atom_types: atom type indices

        Returns:
            noise_pred: (batch, n_atoms, latent_dim) predicted noise
            noise_pred_global: (batch, latent_dim) predicted global noise
        """
        batch_size = z_noisy.shape[0]

        # Time embedding
        t_emb = self.time_mlp(t)  # (batch, hidden_dim)

        # Per-atom denoising
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, self.n_atoms, -1)

        if self.use_atom_types and atom_types is not None:
            if atom_types.dim() == 1:
                atom_types = atom_types.unsqueeze(0).expand(batch_size, -1)
            atom_feats = self.atom_type_embedding(atom_types)
            denoiser_input = torch.cat([z_noisy, t_emb_expanded, atom_feats], dim=-1)
        else:
            denoiser_input = torch.cat([z_noisy, t_emb_expanded], dim=-1)

        noise_pred = self.latent_denoiser(denoiser_input)

        # Global denoising
        global_input = torch.cat([z_global_noisy, t_emb], dim=-1)
        noise_pred_global = self.global_denoiser(global_input)

        return noise_pred, noise_pred_global

    def forward(self, x, t, atom_types=None):
        """
        Forward pass for training.

        For compatibility with existing training code, this performs
        diffusion directly on coordinates by:
        1. Encoding to latent
        2. Adding noise to latent
        3. Predicting noise
        4. Returning decoded prediction

        But for actual training, you'd want to train the VAE and
        latent diffusion separately.
        """
        # Encode to latent
        z, z_global = self.encode(x, atom_types)

        # Add noise (simplified - in practice use proper diffusion schedule)
        noise = torch.randn_like(z)
        noise_global = torch.randn_like(z_global)

        # For now, just predict the noise directly
        # (This is a simplified version - full implementation would
        # use proper diffusion schedules in latent space)
        noise_pred, noise_pred_global = self.denoise_latent(
            z + noise, z_global + noise_global, t, atom_types
        )

        # Decode to get coordinate prediction
        z_denoised = z + noise - noise_pred
        z_global_denoised = z_global + noise_global - noise_pred_global
        x_pred = self.decode(z_denoised, z_global_denoised, atom_types)

        # Return noise prediction in coordinate space for compatibility
        return x - x_pred


class LatentFlowMatcher(nn.Module):
    """
    Flow matching in latent space.

    Simpler than full latent diffusion - just learns to map noise to data
    in the latent space of a pretrained VAE.
    """

    def __init__(
        self,
        n_atoms: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_dim: int = 64,
        num_atom_types: int = 4,
        use_atom_types: bool = True
    ):
        super().__init__()

        self.n_atoms = n_atoms
        self.latent_dim = latent_dim
        self.use_atom_types = use_atom_types

        # Encoder (invariant)
        self.encoder = InvariantEncoder(
            n_atoms, latent_dim, hidden_dim, num_layers,
            num_atom_types, use_atom_types
        )

        # Decoder
        self.decoder = EquivariantDecoder(
            n_atoms, latent_dim, hidden_dim, num_layers,
            num_atom_types, use_atom_types
        )

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Velocity predictor in latent space
        if use_atom_types:
            self.atom_type_embedding = nn.Embedding(num_atom_types, hidden_dim // 4)
            vel_input_dim = latent_dim + hidden_dim + hidden_dim // 4
        else:
            vel_input_dim = latent_dim + hidden_dim

        # Per-atom velocity
        self.velocity_net = nn.Sequential(
            nn.Linear(vel_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Global velocity
        self.global_velocity_net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def encode(self, x, atom_types=None):
        """Encode to latent."""
        return self.encoder(x, atom_types)

    def decode(self, z, z_global, atom_types=None):
        """Decode from latent."""
        x, _ = self.decoder(z, z_global, atom_types)
        return x

    def predict_velocity(self, z, z_global, t, atom_types=None):
        """Predict velocity in latent space."""
        batch_size = z.shape[0]

        t_emb = self.time_mlp(t)
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, self.n_atoms, -1)

        if self.use_atom_types and atom_types is not None:
            if atom_types.dim() == 1:
                atom_types = atom_types.unsqueeze(0).expand(batch_size, -1)
            atom_feats = self.atom_type_embedding(atom_types)
            vel_input = torch.cat([z, t_emb_expanded, atom_feats], dim=-1)
        else:
            vel_input = torch.cat([z, t_emb_expanded], dim=-1)

        v = self.velocity_net(vel_input)

        global_input = torch.cat([z_global, t_emb], dim=-1)
        v_global = self.global_velocity_net(global_input)

        return v, v_global

    def forward(self, x, t, atom_types=None):
        """
        Forward pass compatible with existing training code.

        Returns velocity prediction in coordinate space.
        """
        # Encode to latent
        z, z_global = self.encode(x, atom_types)

        # Predict velocity in latent space
        v_z, v_z_global = self.predict_velocity(z, z_global, t, atom_types)

        # For compatibility, decode velocity to coordinate space
        # This is an approximation - proper training would work in latent space
        x_plus_v = self.decode(z + v_z * 0.1, z_global + v_z_global * 0.1, atom_types)

        # Return approximate coordinate velocity
        return (x_plus_v - x) / 0.1
