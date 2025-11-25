"""
Denoising Diffusion Probabilistic Model (DDPM) implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionModel:
    """
    Denoising Diffusion Probabilistic Model (DDPM) for lipid conformations.
    """
    
    def __init__(self,
                 model: nn.Module,
                 n_timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        """
        Args:
            model: Neural network to predict noise
            n_timesteps: Number of diffusion steps
            beta_start: Starting noise schedule value
            beta_end: Ending noise schedule value
        """
        self.model = model
        self.n_timesteps = n_timesteps
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: add noise to data.
        
        Args:
            x_0: (batch, n_atoms, 3) clean data
            t: (batch,) timesteps
            noise: Optional pre-generated noise
            
        Returns:
            x_t: (batch, n_atoms, 3) noisy data
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_0, t):
        """
        Training loss: predict the noise added to data.
        
        Args:
            x_0: (batch, n_atoms, 3) clean data
            t: (batch,) timesteps
            
        Returns:
            loss: MSE between true and predicted noise
        """
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise=noise)
        
        predicted_noise = self.model(x_t, t)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        Reverse diffusion: denoise one step.
        
        Args:
            x_t: (batch, n_atoms, 3) noisy data at timestep t
            t: (batch,) timesteps
            
        Returns:
            x_{t-1}: (batch, n_atoms, 3) less noisy data
        """
        # Model prediction
        predicted_noise = self.model(x_t, t)
        
        # Extract values for this timestep
        alpha_t = self.alphas[t][:, None, None]
        alpha_cumprod_t = self.alphas_cumprod[t][:, None, None]
        beta_t = self.betas[t][:, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        # Predict x_0
        pred_x_0 = (x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        
        # Compute mean of q(x_{t-1} | x_t, x_0)
        posterior_variance_t = self.posterior_variance[t][:, None, None]
        
        # Compute previous sample mean
        pred_prev_sample = (
            torch.sqrt(self.alphas_cumprod_prev[t][:, None, None]) * beta_t * pred_x_0 /
            (1.0 - alpha_cumprod_t)
        ) + (
            torch.sqrt(alpha_t) * (1.0 - self.alphas_cumprod_prev[t][:, None, None]) * x_t /
            (1.0 - alpha_cumprod_t)
        )
        
        # Add noise (except for last step)
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            pred_prev_sample = pred_prev_sample + torch.sqrt(posterior_variance_t) * noise
        
        return pred_prev_sample
    
    @torch.no_grad()
    def sample(self, shape, device='cpu'):
        """
        Generate new samples via reverse diffusion.
        
        Args:
            shape: (batch_size, n_atoms, 3) shape of samples
            device: Device to generate on
            
        Returns:
            samples: (batch_size, n_atoms, 3) generated conformations
        """
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Gradually denoise
        for t in reversed(range(self.n_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)
        
        return x
