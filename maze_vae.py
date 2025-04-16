import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm

class MazeVAE(nn.Module):
    """Variational Autoencoder for maze generation with difficulty conditioning."""
    def __init__(self, maze_size=15, latent_dim=32):
        super(MazeVAE, self).__init__()
        self.maze_size = maze_size
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(maze_size * maze_size + 1, 512),  # +1 for difficulty
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # Mean and variance for latent space
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder_layers = nn.Sequential(
            nn.Linear(latent_dim + 1, 256),  # +1 for difficulty
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, maze_size * maze_size),
            nn.Sigmoid()
        )
        
    def encode(self, x, difficulty):
        """Encode maze to latent space."""
        # Concatenate difficulty
        x_flat = x.view(x.size(0), -1)
        x_with_diff = torch.cat([x_flat, difficulty.unsqueeze(1)], dim=1)
        
        # Get hidden representation
        hidden = self.encoder_layers(x_with_diff)
        
        # Get mu and log_var
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick for sampling from latent space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, difficulty):
        """Decode from latent space to maze."""
        # Concatenate difficulty
        z_with_diff = torch.cat([z, difficulty.unsqueeze(1)], dim=1)
        
        # Decode
        x_reconstructed = self.decoder_layers(z_with_diff)
        
        # Reshape to maze dimensions
        return x_reconstructed.view(-1, self.maze_size, self.maze_size)
    
    def forward(self, x, difficulty):
        """Forward pass through VAE."""
        mu, log_var = self.encode(x, difficulty)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, difficulty), mu, log_var
    
    def generate(self, difficulty, num_samples=1, device=None):
        """Generate mazes with specified difficulty."""
        if device is None:
            device = next(self.parameters()).device
        
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(num_samples, self.latent_dim).to(device)
            difficulty_tensor = torch.full((num_samples,), difficulty).to(device)
            
            # Decode
            mazes = self.decode(z, difficulty_tensor)
            
            # Threshold to get binary mazes
            binary_mazes = (mazes > 0.5).float()
            
        return binary_mazes

def vae_loss_function(reconstructed_x, x, mu, log_var, beta=1.0):
    """VAE loss: reconstruction loss + KL divergence."""
    # Reconstruction loss (binary cross entropy)
    BCE = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return BCE + beta * KLD

def train_maze_vae(vae, real_mazes, epochs=100, batch_size=32, save_dir='models'):
    """Train the VAE on maze data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    
    # Convert real mazes to torch tensors
    real_mazes = torch.FloatTensor(real_mazes).to(device)
    
    # Create difficulty labels based on some heuristic
    # Here we use a simple random assignment for demonstration
    real_difficulties = torch.rand(len(real_mazes)).to(device)
    
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        permutation = torch.randperm(real_mazes.size(0))
        num_batches = real_mazes.size(0) // batch_size
        
        for i in tqdm(range(0, real_mazes.size(0), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            indices = permutation[i:i+batch_size]
            if len(indices) < batch_size:
                continue
                
            batch = real_mazes[indices]
            difficulties = real_difficulties[indices]
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed_batch, mu, log_var = vae(batch, difficulties)
            
            # Compute loss
            loss = vae_loss_function(reconstructed_batch, batch, mu, log_var)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress
        avg_loss = total_loss / (num_batches * batch_size)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Save model and generate samples
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save(vae.state_dict(), f"{save_dir}/vae_epoch_{epoch+1}.pt")
            
            # Generate sample mazes at different difficulties
            vae.eval()
            difficulties = [0.2, 0.5, 0.8]  # Low, medium, high
            samples = []
            
            for diff in difficulties:
                sample = vae.generate(diff, num_samples=1, device=device)
                samples.append(sample.cpu().numpy())
            
            samples = np.concatenate(samples, axis=0)
            np.save(f"{save_dir}/vae_samples_epoch_{epoch+1}.npy", samples)
    
    return vae