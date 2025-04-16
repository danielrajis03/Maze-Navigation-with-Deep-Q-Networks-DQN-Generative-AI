import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm

# In maze_gan.py, update the MazeGenerator class:
class MazeGenerator(nn.Module):
    """Neural network that generates maze layouts based on a difficulty parameter."""
    def __init__(self, latent_dim=100, maze_size=31):  # Change from 15 to 31
        super(MazeGenerator, self).__init__()
        self.maze_size = maze_size
        
        # Neural network that transforms latent vector into maze
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 1, 256),  # +1 for difficulty parameter
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, maze_size * maze_size),  # Now outputs 31x31 = 961 values
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, z, difficulty):
        # Concatenate the latent vector with difficulty parameter
        z_with_diff = torch.cat([z, difficulty.unsqueeze(1)], dim=1)
        maze_flat = self.model(z_with_diff)
        # Reshape into a square maze
        maze = maze_flat.view(-1, self.maze_size, self.maze_size)
        # Threshold to make binary (0 = path, 1 = wall)
        return maze

# In maze_gan.py, update the MazeDiscriminator class:
class MazeDiscriminator(nn.Module):
    """Neural network that evaluates maze validity and difficulty."""
    def __init__(self, maze_size=31):  # Change from 15 to 31
        super(MazeDiscriminator, self).__init__()
        
        input_dim = maze_size * maze_size  # Will be 31*31 = 961
        
        # Neural network that evaluates maze validity
        self.validity_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),  # Now correctly handles 31x31 input
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output between 0 and 1 (real or fake)
        )
        
        # Neural network that predicts difficulty
        self.difficulty_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),  # Now correctly handles 31x31 input
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output between 0 and 1 (difficulty)
        )

    def forward(self, maze):
        validity = self.validity_model(maze)
        difficulty = self.difficulty_model(maze)
        return validity, difficulty

def train_maze_gan(generator, discriminator, real_mazes, epochs=500, batch_size=32, save_dir='models'):
    """Train the GAN to generate mazes with controllable difficulty."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Add these debug prints
    print(f"Input real_mazes shape: {real_mazes.shape}")
    print(f"Generator input dimensions: {generator.maze_size}x{generator.maze_size}")
    print(f"Discriminator expected input dimensions: {discriminator.validity_model[1].in_features}")
    
    generator.to(device)
    discriminator.to(device)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    criterion_validity = nn.BCELoss()
    criterion_difficulty = nn.MSELoss()
    
    # Convert real mazes to torch tensors
    real_mazes = torch.FloatTensor(real_mazes).to(device)
    real_difficulties = torch.FloatTensor(np.random.rand(len(real_mazes))).to(device)  # Random difficulties for real mazes
    
    latent_dim = 100
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        # Train on batches
        permutation = torch.randperm(real_mazes.size(0))
        total_g_loss = 0
        total_d_loss = 0
        num_batches = real_mazes.size(0) // batch_size
        
        for i in tqdm(range(0, real_mazes.size(0), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            indices = permutation[i:i+batch_size]
            if len(indices) < batch_size:  # Skip last batch if it's smaller
                continue
                
            real_batch = real_mazes[indices]
            batch_difficulties = real_difficulties[indices]
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Real mazes
            validity_real, diff_pred_real = discriminator(real_batch)
            
            # Fake mazes
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_difficulties = torch.rand(batch_size).to(device)  # Random difficulties
            fake_mazes = generator(z, gen_difficulties)
            validity_fake, diff_pred_fake = discriminator(fake_mazes.detach())
            
            # Calculate loss
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            d_loss_real = criterion_validity(validity_real, real_labels)
            d_loss_fake = criterion_validity(validity_fake, fake_labels)
            d_loss_diff = criterion_difficulty(diff_pred_real, batch_difficulties.unsqueeze(1))
            
            d_loss = d_loss_real + d_loss_fake + d_loss_diff
            
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            
            # Generate new fake mazes
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_difficulties = torch.rand(batch_size).to(device)
            fake_mazes = generator(z, gen_difficulties)
            validity, diff_pred = discriminator(fake_mazes)
            
            # Try to fool the discriminator and match target difficulty
            g_loss_fool = criterion_validity(validity, real_labels)
            g_loss_diff = criterion_difficulty(diff_pred, gen_difficulties.unsqueeze(1))
            
            g_loss = g_loss_fool + g_loss_diff
            
            g_loss.backward()
            optimizer_G.step()
            
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
        
        # Print status
        avg_g_loss = total_g_loss / num_batches
        avg_d_loss = total_d_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
        
        # Save models periodically
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            torch.save(generator.state_dict(), f"{save_dir}/generator_epoch_{epoch+1}.pt")
            torch.save(discriminator.state_dict(), f"{save_dir}/discriminator_epoch_{epoch+1}.pt")
            
            # Generate sample mazes at different difficulties
            generator.eval()
            with torch.no_grad():
                z = torch.randn(3, latent_dim).to(device)
                difficulties = torch.tensor([0.2, 0.5, 0.8]).to(device)  # Low, medium, high difficulty
                samples = generator(z, difficulties)
                samples = (samples > 0.5).float().cpu().numpy()
            generator.train()
            
            # Save samples (Implementation depends on visualization method)
            np.save(f"{save_dir}/sample_mazes_epoch_{epoch+1}.npy", samples)
    
    return generator, discriminator