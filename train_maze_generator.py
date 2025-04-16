import argparse
import torch
import numpy as np
import os
from maze_gan import MazeGenerator, MazeDiscriminator, train_maze_gan
from maze_vae import MazeVAE, train_maze_vae
from maze_utils import collect_training_data
from maze_curriculum import MazeCurriculum

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
   # Replace the collect_training_data call in main() with:
    maze_data = collect_training_data(
    num_mazes=args.num_train_mazes,
    maze_size=(15, 15)  # Fixed size for all mazes
)
    
     
    # Add these debug prints
    print(f"Maze data shape: {maze_data.shape}")
    if len(maze_data) > 0:
        print(f"First maze shape: {maze_data[0].shape}")
    
    print(f"Collected {len(maze_data)} mazes for training")
    
    # Choose model type
    # In main function of train_maze_generator.py
    if args.model_type == 'gan':
        print("Training GAN model...")
    # Initialize GAN models
        generator = MazeGenerator(latent_dim=100, maze_size=31)  # Change from 15 to 31
        discriminator = MazeDiscriminator(maze_size=31)  # Change from 15 to 31
        
        # Train GAN
        generator, discriminator = train_maze_gan(
            generator, 
            discriminator, 
            maze_data, 
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=args.output_dir
        )
        
        # Generate curriculum using trained GAN
        print("Generating curriculum using trained GAN...")
        curriculum = MazeCurriculum(
            num_levels=args.num_levels,
            mazes_per_level=args.mazes_per_level
        )
        curriculum.generate_with_gan(generator)
        curriculum.save_curriculum(os.path.join(args.output_dir, 'curriculum'))
        
    elif args.model_type == 'vae':
        print("Training VAE model...")
        # Initialize VAE model
        vae = MazeVAE(maze_size=15, latent_dim=32)
        
        # Train VAE
        vae = train_maze_vae(
            vae,
            maze_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=args.output_dir
        )
        
        # Generate curriculum using trained VAE
        print("Generating curriculum using trained VAE...")
        curriculum = MazeCurriculum(
            num_levels=args.num_levels,
            mazes_per_level=args.mazes_per_level
        )
        curriculum.generate_with_vae(vae)
        curriculum.save_curriculum(os.path.join(args.output_dir, 'curriculum'))
    
    print("Training and curriculum generation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a generative model for maze creation")
    parser.add_argument("--model_type", type=str, choices=['gan', 'vae'], default='gan',
                        help="Type of generative model to train (gan or vae)")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save models and generated mazes")
    parser.add_argument("--num_train_mazes", type=int, default=1000,
                        help="Number of mazes to generate for training")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_levels", type=int, default=10,
                        help="Number of difficulty levels in curriculum")
    parser.add_argument("--mazes_per_level", type=int, default=5,
                        help="Number of mazes to generate per difficulty level")
    
    args = parser.parse_args()
    main(args)