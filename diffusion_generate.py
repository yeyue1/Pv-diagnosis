# coding:utf-8
import os
import numpy as np
import pandas as pd
import torch
from diffusion_model import DiffusionModel, DiffusionProcess
from prdc import cal_mmd
from train_data import train_r

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(d_n, lab):
    # Load model parameters
    checkpoint = torch.load(f"Diffusion_model/{d_n}/diffusion_{lab}.pth", map_location=device)
    
    # Create model with same parameters
    model = DiffusionModel(
        in_channels=6,      # 6 channels for PV data
        seq_length=80,      # 80 time steps
        time_emb_dim=128,
        hidden_dims=[64, 128, 256]
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create diffusion process with same parameters
    diffusion_params = checkpoint['diffusion_params']
    diffusion = DiffusionProcess(
        model=model,
        beta_start=diffusion_params['beta_start'],
        beta_end=diffusion_params['beta_end'],
        timesteps=diffusion_params['timesteps'],
        device=device
    )
    
    return diffusion

def generate_samples(diffusion, num_samples=500):
    """Generate samples using the diffusion model"""
    batch_size = 50  # Generate in smaller batches to save memory
    all_samples = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            curr_batch_size = min(batch_size, num_samples - i)
            shape = (curr_batch_size, 6, 80)  # [B, C, T]
            
            # Generate samples
            samples = diffusion.sample(shape, return_all_timesteps=False)
            all_samples.append(samples)
    
    # Concatenate all batches
    all_samples = torch.cat(all_samples, dim=0)
    return all_samples

def save_samples(samples, d_n, lab, num_to_save=88):
    """Save generated samples to CSV file after selecting best ones"""
    # Get original data for comparison
    original_data = train_r(lab).to(device)
    
    # Calculate MMD distance for each generated sample
    mmd_distances = []
    for sample in samples:
        sample_reshaped = sample.permute(1, 0).unsqueeze(0)  # [1, T, C]
        mmd = cal_mmd(sample_reshaped, original_data, device)
        mmd_distances.append(mmd.item())
    
    # Select samples with lowest MMD distance (most similar to original)
    sorted_indices = np.argsort(mmd_distances)
    selected_indices = sorted_indices[:num_to_save]
    
    # Convert selected samples to numpy arrays
    selected_samples = []
    for idx in selected_indices:
        # Convert from [C, T] to [T, C]
        sample = samples[idx].permute(1, 0).cpu().numpy()
        selected_samples.append(sample)
    
    # Flatten samples to match expected format
    flattened_samples = np.empty(shape=[num_to_save * 80, 6])
    for i, sample in enumerate(selected_samples):
        start_idx = i * 80
        flattened_samples[start_idx:start_idx + 80] = sample
    
    # Reshape and combine with original data
    samples_reshaped = flattened_samples.reshape([num_to_save, -1])
    original_numpy = original_data.cpu().numpy()
    original_reshaped = original_numpy.reshape([original_numpy.shape[0], -1])
    
    combined_data = np.concatenate([original_reshaped, samples_reshaped], axis=0)
    
    # Save to CSV
    os.makedirs("Diffusion_data", exist_ok=True)
    pd.DataFrame(combined_data).to_csv(f'Diffusion_data/{lab}.csv', index=False, header=False)
    print(f"Saved {num_to_save} samples for label {lab} to Diffusion_data/{lab}.csv")

if __name__ == "__main__":
    for d_n in range(3):  # Test with first dataset only
        for lab in range(8):
            print(f"Generating samples for dataset {d_n}, label {lab}")
            
            # Load model
            diffusion = load_model(d_n, lab)
            
            # Generate samples
            samples = generate_samples(diffusion, num_samples=500)
            
            # Save best samples
            save_samples(samples, d_n, lab, num_to_save=88)
