import numpy as np
import torch
from maze_utils import postprocess_generated_maze, find_path
import os
import json


class MazeCurriculum:
    """Manages a curriculum of increasingly difficult mazes for training an agent."""
    
    # Add this method to your MazeCurriculum class
    def generate_with_gan(self, generator, device=None):
        """Generate a curriculum using a trained GAN model with robust error handling."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        generator.to(device)
        generator.eval()
    
    # Clear any existing curriculum
        self.curriculum = []
    
    # Generate mazes for each difficulty level
        for level in range(self.num_levels):
            try:
                difficulty = level / (self.num_levels - 1)  # 0 to 1
                level_mazes = []
            
                print(f"Starting generation for difficulty level {level+1}/{self.num_levels} (target: {difficulty:.2f})")
            
            # Generate multiple mazes for this level
                try:
                    with torch.no_grad():
                    # Sample latent vectors
                        z = torch.randn(self.mazes_per_level, 100).to(device)
                        diff_tensor = torch.full((self.mazes_per_level,), difficulty).to(device)
                    
                    # Generate mazes
                        generated = generator(z, diff_tensor)
                        binary_mazes = (generated > 0.5).float().cpu().numpy()
                
                # Post-process each maze
                    for i, maze in enumerate(binary_mazes):
                        try:
                        # Ensure maze meets requirements
                            processed_maze, start, end, actual_diff = postprocess_generated_maze(
                                maze, 
                                min_difficulty=max(0, difficulty - 0.1),
                                max_difficulty=min(1, difficulty + 0.1)
                            )
                        
                        # Add to curriculum
                            level_mazes.append({
                                'maze': processed_maze,
                                'start': start,
                                'end': end,
                                'target_difficulty': difficulty,
                                'actual_difficulty': actual_diff
                            })
                            print(f"  Maze {i+1}/{len(binary_mazes)} processed successfully with difficulty {actual_diff:.2f}")
                        except Exception as e:
                            print(f"  Error processing maze {i+1}: {str(e)}")
                            continue  # Skip this maze but continue with others
                except Exception as e:
                    print(f"  Error generating mazes for level {level+1}: {str(e)}")
            
            # Only add level if we have at least one valid maze
                if level_mazes:
                    self.curriculum.append(level_mazes)
                    print(f"Generated {len(level_mazes)} mazes for difficulty level {level+1}/{self.num_levels} (target: {difficulty:.2f})")
                
                # Save progress after each successful level
                    try:
                        temp_dir = os.path.join("models", "curriculum_temp")
                        os.makedirs(temp_dir, exist_ok=True)
                        self.save_curriculum(temp_dir)
                        print(f"  Saved progress to {temp_dir}")
                    except Exception as e:
                        print(f"  Error saving progress: {str(e)}")
                else:
                    print(f"Could not generate any valid mazes for level {level+1}, skipping")
                
                # Add an empty placeholder to maintain level indexing
                    self.curriculum.append([])
                
            except Exception as e:
                print(f"Unexpected error at level {level+1}: {str(e)}")
            # Add an empty placeholder to maintain level indexing
                self.curriculum.append([])
                continue
    
    # At the end, save the full curriculum
        try:
            curriculum_dir = os.path.join("models", "curriculum")
            os.makedirs(curriculum_dir, exist_ok=True)
            self.save_curriculum(curriculum_dir)
            print(f"Saved final curriculum to {curriculum_dir}")
        except Exception as e:
         print(f"Error saving final curriculum: {str(e)}")
    
        return self.curriculum

    def save_curriculum(self, save_dir):
        """Save the curriculum to disk."""
        import os
        import json
        import numpy as np
    
        os.makedirs(save_dir, exist_ok=True)
    
        for level_idx, level_mazes in enumerate(self.curriculum):
            level_dir = os.path.join(save_dir, f"level_{level_idx}")
        os.makedirs(level_dir, exist_ok=True)
        
        for maze_idx, maze_data in enumerate(level_mazes):
            # Save maze as numpy array
            maze_path = os.path.join(level_dir, f"maze_{maze_idx}.npy")
            np.save(maze_path, maze_data['maze'])
            
            # Save metadata as JSON
            meta_path = os.path.join(level_dir, f"maze_{maze_idx}_meta.json")
            metadata = {
                'start': list(maze_data['start']),
                'end': list(maze_data['end']),
                'target_difficulty': float(maze_data['target_difficulty']),
                'actual_difficulty': float(maze_data['actual_difficulty'])
            }
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
    
        print(f"Saved curriculum with {len(self.curriculum)} difficulty levels to {save_dir}")
    def load_curriculum(self, load_dir):
        """Load a curriculum from disk."""
        self.curriculum = []
        level_idx = 0
        
        while True:
            level_dir = os.path.join(load_dir, f"level_{level_idx}")
            if not os.path.exists(level_dir):
                break
                
            level_mazes = []
            maze_idx = 0
            
            while True:
                maze_path = os.path.join(level_dir, f"maze_{maze_idx}.npy")
                meta_path = os.path.join(level_dir, f"maze_{maze_idx}_meta.json")
                
                if not os.path.exists(maze_path) or not os.path.exists(meta_path):
                    break
                
                # Load maze and metadata
                maze = np.load(maze_path)
                
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                # Reconstruct maze data
                maze_data = {
                    'maze': maze,
                    'start': tuple(metadata['start']),
                    'end': tuple(metadata['end']),
                    'target_difficulty': metadata['target_difficulty'],
                    'actual_difficulty': metadata['actual_difficulty']
                }
                
                level_mazes.append(maze_data)
                maze_idx += 1
            
            if level_mazes:
                self.curriculum.append(level_mazes)
            
            level_idx += 1
        
        print(f"Loaded curriculum with {len(self.curriculum)} difficulty levels from {load_dir}")
        return self.curriculum
    
    def __init__(self, num_levels=10, mazes_per_level=5, maze_size=15):
        self.num_levels = num_levels
        self.mazes_per_level = mazes_per_level
        self.maze_size = maze_size
        self.curriculum = []  # Will hold mazes at different difficulty levels
        
    

    
    def generate_with_vae(self, vae, device=None):
        """Generate a curriculum using a trained VAE model."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        vae.to(device)
        vae.eval()
        
        # Clear any existing curriculum
        self.curriculum = []
        
        # Generate mazes for each difficulty level
        for level in range(self.num_levels):
            difficulty = level / (self.num_levels - 1)  # 0 to 1
            level_mazes = []
            
            # Generate multiple mazes for this level
            with torch.no_grad():
                # Generate mazes
                mazes = vae.generate(difficulty, num_samples=self.mazes_per_level, device=device)
                binary_mazes = mazes.cpu().numpy()
            
            # Post-process each maze
            for maze in binary_mazes:
                # Ensure maze meets requirements
                processed_maze, start, end, actual_diff = postprocess_generated_maze(
                    maze, 
                    min_difficulty=max(0, difficulty - 0.1))