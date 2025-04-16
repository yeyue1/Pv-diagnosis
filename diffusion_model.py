# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear1 = nn.Linear(embedding_dim // 4, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, t):
        half_dim = self.embedding_dim // 8
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        emb = self.linear1(emb)
        emb = F.gelu(emb)
        emb = self.linear2(emb)
        return emb

class DiffusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.GELU()
        )
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        
        # Time embedding
        t_emb = self.time_mlp(t_emb)
        h = h + t_emb.unsqueeze(-1)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.gelu(h)
        
        return h + self.res_conv(x)

class DiffusionModel(nn.Module):
    def __init__(self, in_channels=6, seq_length=80, time_emb_dim=128, hidden_dims=[64, 128, 256]):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)
        
        # Input projection
        self.input_conv = nn.Conv1d(in_channels, hidden_dims[0], kernel_size=3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.down_blocks.append(DiffusionBlock(hidden_dims[i], hidden_dims[i+1], time_emb_dim))
        
        # Middle
        self.middle_block = DiffusionBlock(hidden_dims[-1], hidden_dims[-1], time_emb_dim)
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)-1, 0, -1):
            self.up_blocks.append(DiffusionBlock(hidden_dims[i], hidden_dims[i-1], time_emb_dim))
        
        # Output projection
        self.output_conv = nn.Conv1d(hidden_dims[0], in_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        """
        x: [B, C, T] tensor of input sequence
        t: [B] tensor of diffusion timesteps
        """
        t_emb = self.time_embedding(t)
        
        h = self.input_conv(x)
        
        # Encoder
        h_down = []
        for block in self.down_blocks:
            h_down.append(h)
            h = block(h, t_emb)
        
        # Middle
        h = self.middle_block(h, t_emb)
        
        # Decoder
        for block in self.up_blocks:
            h = block(h, t_emb)
        
        # Output
        h = self.output_conv(h)
        
        return h

class DiffusionProcess:
    def __init__(self, model, beta_start=1e-4, beta_end=2e-2, timesteps=1000, device="cuda"):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
    
    def forward_diffusion(self, x_0, t):
        """Forward diffusion process: q(x_t | x_0)"""
        noise = torch.randn_like(x_0)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
        std = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return mean + std * noise, noise
    
    def reverse_diffusion(self, x_t, t):
        """Reverse diffusion process: p(x_{t-1} | x_t)"""
        noise_pred = self.model(x_t, t)
        return noise_pred
    
    def sample(self, shape, return_all_timesteps=False):
        """Generate samples by running the reverse diffusion process"""
        b = shape[0]
        device = self.device
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = [img]
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self._p_sample(img, t)
            if return_all_timesteps:
                imgs.append(img)
        
        return img if not return_all_timesteps else torch.stack(imgs)
    
    def _p_sample(self, x, t):
        """Sample from p(x_{t-1} | x_t)"""
        noise_pred = self.model(x, t)
        
        t_index = t[0]
        alpha = self.alphas[t_index]
        alpha_cumprod = self.alphas_cumprod[t_index]
        beta = self.betas[t_index]
        
        # Parametrization for mean of p(x_{t-1} | x_t)
        x_0_pred = (x - beta * noise_pred / torch.sqrt(1 - alpha_cumprod)) / torch.sqrt(alpha)
        
        if t_index == 0:
            return x_0_pred
        else:
            z = torch.randn_like(x) if t_index > 0 else 0
            
            # Compute mean and variance
            mean = x_0_pred * self.posterior_mean_coef1[t_index] + x * self.posterior_mean_coef2[t_index]
            var = self.posterior_variance[t_index]
            
            return mean + torch.sqrt(var) * z
    
    def _extract(self, a, t, x_shape):
        """Extract values from a according to t and reshape to x_shape"""
        batch_size = t.shape[0]
        # 修复设备不匹配问题 - 确保a和t在同一设备上
        out = a.to(t.device).gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
