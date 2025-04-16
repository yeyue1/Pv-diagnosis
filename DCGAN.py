# coding:utf-8
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from train_data import train_r

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim=80):
        super(DCGANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class DCGANDiscriminator(nn.Module):
    def __init__(self, input_dim=80):
        super(DCGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

class CustomDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def train_dcgan(label, dataset_num):
    latent_dim = 100
    output_dim = 80 * 6
    epochs = 500  # Changed from 6000 to 500
    max_g = 10000.0
    
    # Initialize models
    generator = DCGANGenerator(latent_dim, output_dim)
    discriminator = DCGANDiscriminator(input_dim=output_dim)
    generator.to(device)
    discriminator.to(device)
    
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), 
                                 lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), 
                                 lr=0.0002, betas=(0.5, 0.999))
    
    # Data loading
    raw_data = train_r(label)
    data_tensor = raw_data.reshape(raw_data.size(0), -1)
    dataset = CustomDataset(data_tensor)
    dataloader = DataLoader(dataset=dataset, batch_size=6, shuffle=True)
    
    # Training loop
    for epoch in range(epochs):
        D_loss_total = 0
        G_loss_total = 0
        
        for i, real_batch in enumerate(dataloader):
            real_data = real_batch.to(device).float()
            batch_size = real_data.size(0)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            label_real = torch.ones(batch_size, 1).to(device)
            label_fake = torch.zeros(batch_size, 1).to(device)
            
            output_real = discriminator(real_data)
            d_loss_real = nn.BCELoss()(output_real, label_real)
            
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_data = generator(noise)
            output_fake = discriminator(fake_data.detach())
            d_loss_fake = nn.BCELoss()(output_fake, label_fake)
            
            d_loss = (d_loss_real + d_loss_fake)
            d_loss.backward()
            optimizer_D.step()
            D_loss_total += d_loss.item()
            
            # Train Generator
            optimizer_G.zero_grad()
            output_fake = discriminator(fake_data)
            g_loss = nn.BCELoss()(output_fake, label_real)
            g_loss.backward()
            optimizer_G.step()
            G_loss_total += g_loss.item()
        
        print(f"[Epoch {epoch+1}/{epochs}] "
              f"D_loss: {D_loss_total/len(dataloader):.4f} "
              f"G_loss: {G_loss_total/len(dataloader):.4f}")
        
        # Save best model in last 100 epochs
        if epoch >= epochs-100 and G_loss_total/len(dataloader) <= max_g:
            max_g = G_loss_total/len(dataloader)
            torch.save(generator, 
                      f"DCGAN_model/{dataset_num}/DCGAN_{label}.pth")

if __name__ == "__main__":
    for dataset_idx in range(3):
        for label_idx in range(8):
            print(f"\n=== Training Dataset {dataset_idx} - Label {label_idx} ===")
            train_dcgan(label_idx, dataset_idx)
            print(f"=== Completed Dataset {dataset_idx} - Label {label_idx} ===\n")