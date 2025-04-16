import numpy as np
import heapq
import time
import matplotlib.pyplot as plt
from maze_env import MazeEnv

class PriorityQueue:
    """A priority queue implementation for A* algorithm."""
    def __init__(self):
        self.elements = []
        self.count = 0

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        # Priority comes first so heap operations work based on priority
        heapq.heappush(self.elements, (priority, self.count, item))
        self.count += 1

    def get(self):
        # Return only the item, not the priority or count
        return heapq.heappop(self.elements)[2]


def heuristic(a, b):
    """
    Manhattan distance heuristic for A* algorithm.
    Args:
        a: Position tuple (x, y)
        b: Position tuple (x, y)
    Returns:
        Manhattan distance between a and b
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star_search(maze, start, goal):
  
    start_time = time.time()
    
    # Initialize the priority queue
    frontier = PriorityQueue()
    frontier.put(start, 0)
    
    # For each position, which position it came from
    came_from = {start: None}
    
    # For each position, the cost to get there from the start
    cost_so_far = {start: 0}
    
    # Track visited nodes and steps
    visited = set()
    steps = 0
    
    while not frontier.empty():
        steps += 1
        current = frontier.get()
        visited.add(current)
        
        # Exit if we reached the goal
        if current == goal:
            break
        
        # Try all four directions: up, right, down, left
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            next_pos = (current[0] + dx, current[1] + dy)
            
            # Check if position is within bounds
            if (0 <= next_pos[0] < maze.shape[0] and 
                0 <= next_pos[1] < maze.shape[1]):
                
                # Check if position is a wall
                if maze[next_pos[0], next_pos[1]] == 1:
                    continue
                
                # Calculate new cost
                new_cost = cost_so_far[current] + 1  # Just movement cost
                
                # If we haven't visited this node or we've found a better path
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    # Priority is cost so far plus heuristic
                    priority = new_cost + heuristic(goal, next_pos)
                    frontier.put(next_pos, priority)
                    came_from[next_pos] = current
    
    # Reconstruct path
    path = []
    current = goal
    
    # Check if goal was reached
    if current not in came_from:
        print("No path found!")
        return [], visited, steps, time.time() - start_time
    
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    
    time_taken = time.time() - start_time
    return path, visited, steps, time_taken


def visualize_path(maze, path, visited=None):
    """
    Visualize the maze with the path and optionally the visited nodes.
    
    Args:
        maze: 2D numpy array representing the maze
        path: List of positions in the path
        visited: Set of visited positions (optional)
    """
    # Create a visualization grid
    viz = np.zeros(maze.shape, dtype=str)
    viz[maze == 1] = '#'  # Walls
    viz[maze == 0] = ' '  # Open spaces
    
    # Mark visited nodes
    if visited:
        for pos in visited:
            if maze[pos[0], pos[1]] == 0:  # Only mark if it's not a wall
                viz[pos[0], pos[1]] = '.'
    
    # Mark path
    for pos in path:
        viz[pos[0], pos[1]] = 'o'
    
    # Mark start and goal
    if path:
        viz[path[0][0], path[0][1]] = 'S'
        viz[path[-1][0], path[-1][1]] = 'G'
    
    # Print the visualization
    for row in viz:
        print(' '.join(row))


def test_a_star_on_maze(difficulty='easy', visualize=True):
    """
    Test the A* algorithm on mazes with different difficulty levels.
    
    Args:
        difficulty: Maze difficulty ('easy', 'moderate', 'difficult', 'very_difficult')
        visualize: Whether to visualize the maze with the path
    """
    # Create maze environment
    env = MazeEnv(difficulty=difficulty)
    
    print(f"\nTesting A* on {difficulty.capitalize()} Maze")
    print(f"Maze dimensions: {env.maze.shape}")
    print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
    
    # Run A* algorithm
    path, visited, steps, time_taken = a_star_search(
        env.maze, env.start_pos, env.goal_pos
    )
    
    # Print results
    if path:
        print(f"Path found with {len(path)} steps")
        print(f"A* algorithm took {steps} iterations and {time_taken:.6f} seconds")
        print(f"Visited {len(visited)} cells out of {np.sum(env.maze == 0)} open cells")
    else:
        print("No path found!")
    
    # Visualize if requested
    if visualize:
        print("\nMaze with Path:")
        visualize_path(env.maze, path, visited)
    
    return path, visited, steps, time_taken


def compare_all_difficulties(plot=True):
    """Compare A* algorithm performance on all maze difficulties."""
    results = {}
    difficulty_labels = ['Easy Maze', 'Moderate Maze', 'Higher difficulty', 'Very High Difficulty']
    difficulty_keys = ['easy', 'moderate', 'difficult', 'very_difficult']
    
    path_lengths = []
    time_takens = []
    
    for i, difficulty in enumerate(difficulty_keys):
        path, visited, steps, time_taken = test_a_star_on_maze(
            difficulty=difficulty, visualize=True
        )
        
        path_lengths.append(len(path))
        time_takens.append(time_taken)
        
        results[difficulty] = {
            'path_length': len(path),
            'steps': steps,
            'time_taken': time_taken,
            'visited': len(visited)
        }
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("A* Algorithm Performance Comparison")
    print("=" * 60)
    print(f"{'Difficulty':<15} {'Path Length':<15} {'Steps':<15} {'Time (s)':<15} {'Cells Visited':<15}")
    print("-" * 60)
    
    for i, difficulty in enumerate(difficulty_keys):
        metrics = results[difficulty]
        print(f"{difficulty_labels[i]:<15} {metrics['path_length']:<15} {metrics['steps']:<15} {metrics['time_taken']:<15.6f} {metrics['visited']:<15}")
    
    # Generate plot
    if plot:
        plt.figure(figsize=(10, 6))
        
        # Create bar plot for path length
        ax1 = plt.gca()
        ax1.set_xlabel('Maze')
        ax1.set_ylabel('Path Length', color='blue')
        ax1.bar(difficulty_labels, path_lengths, color='blue', alpha=0.7)
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Create line plot for time taken
        ax2 = ax1.twinx()
        ax2.set_ylabel('Time Taken (s)', color='red')
        ax2.plot(difficulty_labels, time_takens, color='red', marker='o')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title("A* Path Length and Computation Time Across Different Mazes")
        plt.tight_layout()
        plt.savefig('a_star_performance.png')
        plt.show()

    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='A* Maze Solver')
    parser.add_argument('--difficulty', type=str, default='all',
                        choices=['all', 'easy', 'moderate', 'difficult', 'very_difficult'],
                        help='Maze difficulty level')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the maze with the path')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Generate performance comparison plot')
    
    args = parser.parse_args()
    
    if args.difficulty == 'all':
        compare_all_difficulties(plot=args.plot)
    else:
        test_a_star_on_maze(difficulty=args.difficulty, visualize=args.visualize)