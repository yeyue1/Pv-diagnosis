# coding:utf-8
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from diffusion_model import DiffusionModel, DiffusionProcess
from train_data import train_r

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def train_diffusion(lab, d_n):
    # Hyperparameters
    timesteps = 500
    epochs = 100
    batch_size = 8
    learning_rate = 2e-4
    
    # Create directories if they don't exist
    os.makedirs(f"Diffusion_model/{d_n}", exist_ok=True)
    
    # Load data
    raw_data = train_r(lab)
    print(f"raw_data shape: {raw_data.shape}")
    
    # Convert data to have channels first [B, T, C] -> [B, C, T]
    data_tensor = raw_data.permute(0, 2, 1).float()
    dataset = CustomDataset(data_tensor)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True
    )
    
    # Create model
    in_channels = data_tensor.shape[1]  # 6 channels for PV data
    seq_length = data_tensor.shape[2]   # 80 time steps
    
    model = DiffusionModel(
        in_channels=in_channels,
        seq_length=seq_length,
        time_emb_dim=128, 
        hidden_dims=[64, 128, 256]
    ).to(device)
    
    diffusion = DiffusionProcess(
        model=model,
        beta_start=1e-4,
        beta_end=2e-2,
        timesteps=timesteps,
        device=device
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        for i, x in enumerate(dataloader):
            x = x.to(device)
            batch_size = x.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            
            # Forward diffusion to add noise
            x_noisy, noise = diffusion.forward_diffusion(x, t)
            
            # Predict noise
            predicted_noise = model(x_noisy, t)
            
            # Calculate loss
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'diffusion_params': {
                        'beta_start': diffusion.betas[0].item(),
                        'beta_end': diffusion.betas[-1].item(),
                        'timesteps': timesteps
                    }
                },
                f"Diffusion_model/{d_n}/diffusion_{lab}.pth"
            )
            print(f"Model saved with loss {avg_loss:.6f}")

if __name__ == "__main__":
    for d in range(3):
        for lab in range(8):
            print(f"Training diffusion model for dataset {d}, label {lab}")
            train_diffusion(lab, d)
            time.sleep(5)  # Short break between training runs
