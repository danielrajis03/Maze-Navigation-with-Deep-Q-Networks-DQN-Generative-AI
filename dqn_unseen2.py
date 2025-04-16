import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from maze_env import MazeEnv
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.generate.Sidewinder import Sidewinder
from mazelib.generate.BacktrackingGenerator import BacktrackingGenerator
from mazelib.generate.GrowingTree import GrowingTree
from mazelib.generate.HuntAndKill import HuntAndKill

class MazeConverter:
    @staticmethod
    def convert_mazelib_to_dqn(mazelib_maze):
        """Convert a mazelib maze to the format our DQN understands"""
        # Get the mazelib grid
        grid = mazelib_maze.grid
        
        # In mazelib, 1 is wall and 0 is path
        # In our DQN environment, 1 is wall and 0 is path, so we're good
        converted_maze = grid.copy()
        
        return converted_maze, mazelib_maze.start, mazelib_maze.end

    @staticmethod
    def generate_test_mazes(seed=None):
        """Generate a variety of test mazes using different algorithms"""
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)
            
        test_mazes = {}
        
        # 1. Prim's Algorithm - Classic, perfect maze
        prim_maze = Maze()
        prim_maze.generator = Prims(15, 15)  # Size 15x15
        prim_maze.generate()
        prim_maze.generate_entrances()
        test_mazes["Prims"] = prim_maze
        
        # 2. Sidewinder - Biased in one direction with an open corridor
        side_maze = Maze()
        side_maze.generator = Sidewinder(15, 15)
        side_maze.generate()
        side_maze.generate_entrances()
        test_mazes["Sidewinder"] = side_maze
        
        # 3. Backtracker - High quality maze with few branches
        back_maze = Maze()
        back_maze.generator = BacktrackingGenerator(15, 15)
        back_maze.generate()
        back_maze.generate_entrances()
        test_mazes["Backtracking"] = back_maze
        
        # 4. Growing Tree - Configurable between random and backtracking
        grow_maze = Maze()
        grow_maze.generator = GrowingTree(15, 15, 0.5)  # 50% backtracking
        grow_maze.generate()
        grow_maze.generate_entrances()
        test_mazes["GrowingTree"] = grow_maze
        
        # 5. Hunt and Kill - Random walk with systematic backfill
        hunt_maze = Maze()
        hunt_maze.generator = HuntAndKill(20, 20)  # A bit larger
        hunt_maze.generate()
        hunt_maze.generate_entrances()
        test_mazes["HuntAndKill"] = hunt_maze
        
        return test_mazes

def test_agent_on_maze(agent, maze_data, maze_name, num_episodes=10, max_steps=700):
    """Test a DQN agent on a specific maze"""
    converted_maze, start, end = maze_data
    
    # Create a custom environment with the converted maze
    env = MazeEnv(custom_maze=converted_maze, 
                  custom_start=start,
                  custom_goal=end,
                  custom_max_steps=max_steps)
    
    # Debug output to confirm environment settings
    print(f"\nMaze Environment Settings:")
    print(f"  Maze shape: {env.maze.shape}")
    print(f"  Start position: {env.start_pos}")
    print(f"  Goal position: {env.goal_pos}")
    print(f"  Max steps: {env.max_steps}")
    
    # Testing metrics
    total_steps = []
    total_times = []
    successes = 0
    
    print(f"\nTesting on {maze_name} maze:")
    print(f"Running {num_episodes} episodes with {max_steps} max steps per episode")
    
    # Only print detailed logs for the first few episodes and then every 100th
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        start_time = time.time()
        
        while not done and steps < max_steps:
            # Get action from agent
            action = agent.act(state)
            
            # Take step in environment
            next_state, reward, done, _ = env.step(action)
            
            # Update state and step counter
            state = next_state
            steps += 1
        
        # Record results
        ep_time = time.time() - start_time
        total_steps.append(steps)
        total_times.append(ep_time)
        
        # Check if we reached the goal
        success = tuple(state) == env.goal_pos
        if success:
            successes += 1
        
        # Print progress less frequently for long runs
        if ep < 5 or ep % 100 == 0 or ep == num_episodes - 1:
            print(f"Episode {ep+1}/{num_episodes}: Steps: {steps}, Time: {ep_time:.4f}s, Success: {success}")
    
    # Calculate metrics
    avg_steps = np.mean(total_steps) if total_steps else 0
    avg_time = np.mean(total_times) if total_times else 0
    success_rate = (successes / num_episodes) * 100
    
    print(f"\nResults for {maze_name}:")
    print(f"  Average Steps: {avg_steps:.2f}")
    print(f"  Average Time: {avg_time:.4f}s")
    print(f"  Success Rate: {success_rate:.1f}%")
    
    return avg_steps, avg_time, success_rate

def run_unseen_maze_tests(model_path="maze_dqn_model.pt", num_episodes=1000, max_steps=700, seed=None):
    """Run tests on new, unseen mazes only"""
    # Set random seed if provided
    if seed is not None:
        print(f"Using random seed: {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Load the trained agent
    agent = DQNAgent(state_dim=2, action_dim=4)
    try:
        agent.load(model_path)
        print(f"Loaded trained model from {model_path} successfully")
    except Exception as e:
        print(f"Could not load trained model: {e}")
        print("Using untrained agent.")
    
    # Set agent to evaluation mode (minimal exploration)
    agent.epsilon = 0.05
    print(f"Set agent exploration rate (epsilon) to {agent.epsilon}")
    
    # Generate and test on new mazes
    print("\n--- Testing on New Unseen Mazes ---")
    test_mazes = MazeConverter.generate_test_mazes(seed=seed)
    
    results = {}
    for maze_name, maze in test_mazes.items():
        print(f"\n======= Testing on {maze_name} maze =======")
        maze_data = MazeConverter.convert_mazelib_to_dqn(maze)
        
        # Test agent on this maze
        avg_steps, avg_time, success_rate = test_agent_on_maze(
            agent, 
            maze_data, 
            maze_name,
            num_episodes=num_episodes,
            max_steps=max_steps
        )
        
        # Store results
        results[maze_name] = {
            'steps': avg_steps,
            'time': avg_time,
            'success_rate': success_rate
        }
    
    # Format and display results
    print("\n--- AGENT PERFORMANCE ON UNSEEN MAZES ---\n")
    print(f"{'Maze Type':<15} {'Steps Taken':<12} {'Time (s)':<10} {'Success Rate':<12}")
    print("-" * 60)
    
    for maze_type, metrics in results.items():
        success_symbol = "✅" if metrics['success_rate'] > 50 else "❌"
        print(f"{maze_type:<15} {metrics['steps']:<12.1f} {metrics['time']:<10.4f} {metrics['success_rate']:>10.1f}% {success_symbol}")
    
    # Generate visualization
    generate_maze_plots(results)
    
    return results

def generate_maze_plots(results):
    """Generate comparison plots for maze performance metrics"""
    # Extract data
    maze_names = list(results.keys())
    steps = [results[maze]['steps'] for maze in maze_names]
    times = [results[maze]['time'] for maze in maze_names]
    success_rates = [results[maze]['success_rate'] for maze in maze_names]
    
    # Create figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Steps plot
    axs[0].bar(maze_names, steps, color='teal', alpha=0.7)
    axs[0].set_title('Average Steps to Solution')
    axs[0].set_ylabel('Steps')
    plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    # Time plot
    axs[1].bar(maze_names, times, color='purple', alpha=0.7)
    axs[1].set_title('Average Time to Solution')
    axs[1].set_ylabel('Time (seconds)')
    plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    # Success rate plot
    axs[2].bar(maze_names, success_rates, color='orange', alpha=0.7)
    axs[2].set_title('Success Rate')
    axs[2].set_ylabel('Success Rate (%)')
    axs[2].set_ylim(0, 100)
    plt.setp(axs[2].xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig('dqn_unseen_maze_results.png')
    print("Saved results plot to 'dqn_unseen_maze_results.png'")
    plt.show()

def visualize_agent_on_maze(maze_name="Prims", model_path="maze_dqn_model.pt", max_steps=700, seed=42):
    """Visualize the agent solving a specific maze type"""
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load the agent
    agent = DQNAgent(state_dim=2, action_dim=4)
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except:
        print("Could not load model, using untrained agent")
    
    # Minimal exploration for visualization
    agent.epsilon = 0.01
    
    # Generate a maze of the specified type
    test_mazes = MazeConverter.generate_test_mazes(seed=seed)
    if maze_name not in test_mazes:
        print(f"Maze type '{maze_name}' not found. Available types: {list(test_mazes.keys())}")
        return
    
    maze = test_mazes[maze_name]
    maze_data = MazeConverter.convert_mazelib_to_dqn(maze)
    converted_maze, start, end = maze_data
    
    # Create environment
    env = MazeEnv(custom_maze=converted_maze, 
                  custom_start=start, 
                  custom_goal=end,
                  custom_max_steps=max_steps)
    
    # Run a single episode with rendering
    state = env.reset()
    done = False
    steps = 0
    total_reward = 0
    
    print(f"\nVisualizing agent on {maze_name} maze:")
    print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
    
    env.render()  # Initial state
    
    while not done and steps < max_steps:
        # Get action from agent
        action = agent.act(state)
        
        # Take step
        next_state, reward, done, _ = env.step(action)
        
        # Update state and tracking
        state = next_state
        total_reward += reward
        steps += 1
        
        # Print step info and render
        print(f"Step {steps}: Action {action}, Reward {reward:.2f}, Total Reward {total_reward:.2f}")
        env.render()
        time.sleep(0.1)  # Pause for visualization
      
    # Final result
    if tuple(state) == env.goal_pos:
        print(f"\n✅ Success! Reached goal in {steps} steps with total reward {total_reward:.2f}")
    else:
        print(f"\n❌ Failed to reach goal. Took {steps} steps with total reward {total_reward:.2f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DQN agent on unseen mazes')
    parser.add_argument('--model', type=str, default='maze_dqn_model.pt',
                        help='Path to trained model file')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to test per maze')
    parser.add_argument('--steps', type=int, default=700,
                        help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--visualize', type=str, default=None,
                        choices=['Prims', 'Sidewinder', 'Backtracking', 'GrowingTree', 'HuntAndKill'],
                        help='Visualize agent on a specific maze type')
    
    args = parser.parse_args()
    
    if args.visualize:
        visualize_agent_on_maze(
            maze_name=args.visualize,
            model_path=args.model,
            max_steps=args.steps,
            seed=args.seed if args.seed is not None else 42
        )
    else:
        run_unseen_maze_tests(
            model_path=args.model,
            num_episodes=args.episodes,
            max_steps=args.steps,
            seed=args.seed
        )