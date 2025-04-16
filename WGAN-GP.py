# coding:utf-8
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader, Dataset
from train_data import train_r

cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WGANGPGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(WGANGPGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class WGANGPDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(WGANGPDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

class CustomDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data, gain=1.)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    alpha = alpha.expand(real_samples.size())

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), requires_grad=False, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_wgan_gp(lab, d_n):
    latent_dim = 100
    output_dim = 80 * 6
    epochs = 50  # Changed from 6000 to 50
    max_g = 10000.0
    lambda_gp = 10
    
    # Initialize models
    generator = WGANGPGenerator(latent_dim, output_dim)
    discriminator = WGANGPDiscriminator(input_dim=output_dim)
    generator.to(device)
    discriminator.to(device)
    
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.0, 0.9))
    
    # Data loading
    raw_data = train_r(lab)
    print(f"raw_data shape: {raw_data.shape}")  # 添加调试信息
    data_tensor = raw_data.reshape(raw_data.size(0), -1)
    dataset = CustomDataset(data_tensor)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=6,         # 调整为更小的 batch size
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
    
    for epoch in range(epochs):
        D_loss_total = 0
        G_loss_total = 0
        
        for i, real_batch in enumerate(dataloader):
            real_data = real_batch.to(device, non_blocking=True).float()
            batch_size = real_data.size(0)
            
            # Train Discriminator
            for _ in range(5):
                optimizer_D.zero_grad()
                noise = torch.randn(batch_size, latent_dim).to(device)
                fake_data = generator(noise)
                
                # Gradient penalty
                alpha = torch.rand(batch_size, 1).to(device)
                interpolates = (alpha * real_data + (1 - alpha) * fake_data.detach())
                interpolates.requires_grad_(True)
                d_interpolates = discriminator(interpolates)
                
                grad_outputs = torch.ones_like(d_interpolates)
                gradients = torch.autograd.grad(
                    outputs=d_interpolates,
                    inputs=interpolates,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                
                real_loss = -torch.mean(discriminator(real_data))
                fake_loss = torch.mean(discriminator(fake_data.detach()))
                d_loss = real_loss + fake_loss + gradient_penalty
                
                d_loss.backward()
                optimizer_D.step()
                D_loss_total += d_loss.item()
            
            # Train Generator
            optimizer_G.zero_grad()
            fake_data = generator(noise)
            g_loss = -torch.mean(discriminator(fake_data))
            g_loss.backward()
            optimizer_G.step()
            G_loss_total += g_loss.item()
        
        if len(dataloader) == 0:
            print(f"[Epoch {epoch+1}/{epochs}] 数据集为空！")
        else:
            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"D_loss: {D_loss_total/len(dataloader):.4f} "
                  f"G_loss: {G_loss_total/len(dataloader):.4f}")
        
        # Save best model in last 10 epochs
        if epoch >= epochs-10 and G_loss_total/len(dataloader) <= max_g:
            max_g = G_loss_total/len(dataloader)
            torch.save(generator, f"WGAN_GP_model/{d_n}/WGAN_GP_{lab}.pth")

if __name__ == "__main__":
    for d in range(3):
        for lab in range(8):
            train_wgan_gp(lab, d)