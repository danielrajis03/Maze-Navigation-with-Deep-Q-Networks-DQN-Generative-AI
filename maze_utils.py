import numpy as np
import queue
import random
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.generate.Sidewinder import Sidewinder
from mazelib.generate.BacktrackingGenerator import BacktrackingGenerator
from mazelib.generate.GrowingTree import GrowingTree


def collect_training_data(num_mazes=1000, maze_size=(15, 15)):
    """Collect training data from various maze generation algorithms using a fixed size."""
    maze_data = []
    algorithms = [Prims, Sidewinder, BacktrackingGenerator, GrowingTree]
    
    height, width = maze_size
    print(f"Generating mazes with size: {height}x{width}")
    
    for i in range(num_mazes):
        # Randomly select algorithm (but keep size fixed)
        algorithm = random.choice(algorithms)
        
        # Create maze
        maze = Maze()
        maze.generator = algorithm(height, width)
        maze.generate()
        maze.generate_entrances()
        
        # Print shape of each 10th maze
        if i % 100 == 0:
            print(f"Maze {i} shape: {maze.grid.shape}")
        
        # Append maze grid
        maze_data.append(maze.grid.astype(np.float32))
    
    return np.array(maze_data)

def ensure_solvable(maze, start=None, end=None):
    """Ensure the maze has at least one valid path from start to end."""
    # Convert to numpy if it's a torch tensor
    if not isinstance(maze, np.ndarray):
        maze = maze.detach().cpu().numpy()
    
    # Make a copy to modify
    maze_copy = maze.copy()
    
    # Make sure borders are walls
    maze_copy[0, :] = 1
    maze_copy[-1, :] = 1
    maze_copy[:, 0] = 1
    maze_copy[:, -1] = 1
    
    # Set default start and end if not provided
    if start is None:
        start = (1, 1)
    if end is None:
        end = (maze.shape[0]-2, maze.shape[1]-2)
    
    # Ensure start and end are open
    maze_copy[start[0], start[1]] = 0
    maze_copy[end[0], end[1]] = 0
    
    # Check if there's a path
    path = find_path(maze_copy, start, end)
    
    if path is None:
        # If no path exists, create one
        maze_copy = create_path(maze_copy, start, end)
    
    return maze_copy, start, end

def find_path(maze, start, end):
    """Find a path through the maze using BFS."""
    height, width = maze.shape
    visited = np.zeros_like(maze, dtype=bool)
    visited[start[0], start[1]] = True
    
    # Queue for BFS
    q = queue.Queue()
    q.put((start, []))
    
    # Directions: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    while not q.empty():
        (x, y), path = q.get()
        
        # Check if we reached the end
        if (x, y) == end:
            return path + [(x, y)]
        
        # Try each direction
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check if valid and not a wall
            if (0 <= nx < height and 0 <= ny < width and 
                maze[nx, ny] == 0 and not visited[nx, ny]):
                visited[nx, ny] = True
                q.put(((nx, ny), path + [(x, y)]))
    
    # No path found
    return None

def create_path(maze, start, end, min_length=None):
    """Create a path from start to end in the maze."""
    # Create a copy to modify
    maze_copy = maze.copy()
    
    # Set current position to start
    current = start
    path = [current]
    
    # Directions: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    # Continue until we reach the end
    while current != end:
        # Possible next steps (prioritizing moves toward the end)
        possible_next = []
        
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            
            # Check if within bounds
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1]:
                # Calculate Manhattan distance to end
                dist_to_end = abs(nx - end[0]) + abs(ny - end[1])
                possible_next.append((dist_to_end, (nx, ny)))
        
        # Sort by distance to end (closest first)
        possible_next.sort()
        
        # Try moves in order of preference
        moved = False
        for _, (nx, ny) in possible_next:
            # Check if this is a valid move
            if nx == end[0] and ny == end[1]:
                # We've reached the end
                current = (nx, ny)
                path.append(current)
                moved = True
                break
            
            # Check if it's a valid intermediate step
            if 0 < nx < maze.shape[0]-1 and 0 < ny < maze.shape[1]-1:
                # Either it's already open or we can make it open
                if maze_copy[nx, ny] == 0 or (nx, ny) not in path:
                    maze_copy[nx, ny] = 0  # Carve a path
                    current = (nx, ny)
                    path.append(current)
                    moved = True
                    break
        
        # If we couldn't move, we're stuck - add more randomness
        if not moved:
            # Pick a random direction
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 < nx < maze.shape[0]-1 and 0 < ny < maze.shape[1]-1:
                    maze_copy[nx, ny] = 0  # Carve a path
                    current = (nx, ny)
                    path.append(current)
                    break
    
    # If a minimum path length is specified, add complexity
    if min_length is not None and len(path) < min_length:
        # Keep adding branch-offs until we reach minimum length
        while len(path) < min_length:
            # Pick a random point on our existing path
            branch_point = random.choice(path[:-1])  # Don't branch from the end
            
            # Try to create a branch
            branch_length = min(5, min_length - len(path))
            current = branch_point
            
            for _ in range(branch_length):
                # Pick a random direction
                random.shuffle(directions)
                moved = False
                
                for dx, dy in directions:
                    nx, ny = current[0] + dx, current[1] + dy
                    
                    # Check if it's a valid branch point
                    if (0 < nx < maze.shape[0]-1 and 0 < ny < maze.shape[1]-1 and
                        (nx, ny) != end and (nx, ny) not in path):
                        maze_copy[nx, ny] = 0  # Carve a path
                        current = (nx, ny)
                        path.append(current)
                        moved = True
                        break
                
                if not moved:
                    break
    
    return maze_copy

def calculate_maze_difficulty(maze, start, end):
    """Calculate a difficulty score for a maze."""
    # Find the optimal path
    path = find_path(maze, start, end)
    
    if path is None:
        return 0.0  # Unsolvable mazes are not useful
    
    # Factors that contribute to difficulty:
    # 1. Path length relative to maze size
    maze_size = maze.shape[0] * maze.shape[1]
    path_length = len(path)
    length_factor = min(1.0, path_length / (maze_size * 0.5))
    
    # 2. Number of decision points (junctions)
    junctions = 0
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    for x, y in path:
        # Count open neighbors
        open_neighbors = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
                open_neighbors += 1
        
        # A junction has more than 2 open neighbors (path splits)
        if open_neighbors > 2:
            junctions += 1
    
    junction_factor = min(1.0, junctions / (maze_size * 0.1))
    
    # 3. Maze density (ratio of walls to open spaces)
    wall_ratio = np.sum(maze) / maze_size
    density_factor = wall_ratio * 0.5  # Scale to have less impact
    
    # Combine factors with weights
    difficulty = 0.5 * length_factor + 0.3 * junction_factor + 0.2 * density_factor
    
    return np.clip(difficulty, 0.0, 1.0)

def postprocess_generated_maze(maze, min_difficulty=None, max_difficulty=None):
    """Post-process a generated maze to ensure it meets requirements."""
    # Ensure binary values
    binary_maze = (maze > 0.5).astype(np.float32)
    
    # Make sure borders are walls
    binary_maze[0, :] = 1
    binary_maze[-1, :] = 1
    binary_maze[:, 0] = 1
    binary_maze[:, -1] = 1
    
    # Set start and end points
    start = (1, 1)
    end = (binary_maze.shape[0]-2, binary_maze.shape[1]-2)
    
    # Ensure maze is solvable
    solvable_maze, start, end = ensure_solvable(binary_maze, start, end)
    
    # Calculate difficulty
    difficulty = calculate_maze_difficulty(solvable_maze, start, end)
    
    # Adjust difficulty if needed
    if min_difficulty is not None and difficulty < min_difficulty:
        # Make maze more difficult by adding walls
        while difficulty < min_difficulty:
            # Add a random wall
            x = random.randint(1, solvable_maze.shape[0]-2)
            y = random.randint(1, solvable_maze.shape[1]-2)
            
            # Don't block start or end
            if (x, y) != start and (x, y) != end:
                # Check if adding a wall here preserves solvability
                solvable_maze[x, y] = 1
                path = find_path(solvable_maze, start, end)
                
                if path is None:
                    # Undo if it blocks the path
                    solvable_maze[x, y] = 0
                else:
                    # Recalculate difficulty
                    difficulty = calculate_maze_difficulty(solvable_maze, start, end)
    
    elif max_difficulty is not None and difficulty > max_difficulty:
        # Make maze easier by removing walls
        while difficulty > max_difficulty:
            # Find a random wall that's not on the border
            wall_positions = []
            for i in range(1, solvable_maze.shape[0]-1):
                for j in range(1, solvable_maze.shape[1]-1):
                    if solvable_maze[i, j] == 1:
                        wall_positions.append((i, j))
            
            if not wall_positions:
                break  # No more walls to remove
            
            # Remove a random wall
            x, y = random.choice(wall_positions)
            solvable_maze[x, y] = 0
            
            # Recalculate difficulty
            difficulty = calculate_maze_difficulty(solvable_maze, start, end)
    
    return solvable_maze, start, end, difficulty