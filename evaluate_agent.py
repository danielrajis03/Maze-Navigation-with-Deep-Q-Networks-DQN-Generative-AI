import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from maze_env import MazeEnv
from maze_utils import collect_training_data, ensure_solvable, calculate_maze_difficulty
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.generate.Sidewinder import Sidewinder
from mazelib.generate.BacktrackingGenerator import BacktrackingGenerator
from mazelib.generate.GrowingTree import GrowingTree

def evaluate_on_test_mazes(agent, num_mazes=20, max_steps=500, seed=None):
    """Evaluate agent performance on a set of generated test mazes."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Generate test mazes
    print("Generating test mazes...")
    test_mazes = []
    algorithms = [
        ("Prims", Prims), 
        ("Sidewinder", Sidewinder), 
        ("Backtracking", BacktrackingGenerator), 
        ("GrowingTree", GrowingTree)
    ]
    
    for algo_name, algo_class in algorithms:
        for _ in range(num_mazes // len(algorithms)):
            maze = Maze()
            maze.generator = algo_class(15, 15)
            maze.generate()
            maze.generate_entrances()
            
            # Convert to format our environment understands
            maze_array = maze.grid.astype(np.float32)
            start = maze.start
            end = maze.end
            
            # Calculate difficulty
            difficulty = calculate_maze_difficulty(maze_array, start, end)
            
            test_mazes.append({
                'maze': maze_array,
                'start': start,
                'end': end,
                'algorithm': algo_name,
                'difficulty': difficulty
            })
    
    # Sort by difficulty
    test_mazes.sort(key=lambda x: x['difficulty'])
    
    # Evaluate on each maze
    results = []
    
    for i, maze_data in enumerate(test_mazes):
        print(f"\nEvaluating on test maze {i+1}/{len(test_mazes)} (Algorithm: {maze_data['algorithm']}, Difficulty: {maze_data['difficulty']:.2f})")
        
        # Create environment
        env = MazeEnv(
            custom_maze=maze_data['maze'],
            custom_start=maze_data['start'],
            custom_goal=maze_data['end'],
            custom_max_steps=max_steps
        )
        
        # Run multiple episodes
        successes = []
        steps_list = []
        rewards_list = []
        
        for episode in range(10):  # 10 episodes per maze
            state = env.reset()
            done = False
            steps = 0
            total_reward = 0
            
            # Set minimal exploration
            agent.epsilon = 0.05
            
            while not done and steps < max_steps:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward
                steps += 1
            
            # Check if goal was reached
            success = tuple(state) == env.goal_pos
            successes.append(success)
            steps_list.append(steps)
            rewards_list.append(total_reward)
            
            print(f"Episode {episode+1}: Steps: {steps}, Reward: {total_reward:.2f}, Success: {success}")
        
        # Calculate metrics
        success_rate = np.mean(successes) * 100
        avg_steps = np.mean(steps_list)
        avg_reward = np.mean(rewards_list)
        
        results.append({
            'algorithm': maze_data['algorithm'],
            'difficulty': maze_data['difficulty'],
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_reward': avg_reward
        })
        
        print(f"Results: Success Rate: {success_rate:.1f}%, Avg Steps: {avg_steps:.1f}, Avg Reward: {avg_reward:.2f}")
    
    return results

def main(args):
    # Load trained agent
    agent = DQNAgent(state_dim=2, action_dim=4)
    try:
        agent.load(args.model_path)
        print(f"Loaded agent from {args.model_path}")
    except:
        print(f"Could not load model from {args.model_path}. Using untrained agent.")
    
    # Evaluate
    results = evaluate_on_test_mazes(
        agent,
        num_mazes=args.num_test_mazes,
        max_steps=args.max_steps,
        seed=args.seed
    )
    
    # Analyze results
    algorithms = list(set(r['algorithm'] for r in results))
    
    # Plot results by algorithm
    plt.figure(figsize=(15, 10))
    
    # Success rate by algorithm
    plt.subplot(2, 2, 1)
    success_by_algo = {}
    for algo in algorithms:
        success_by_algo[algo] = [r['success_rate'] for r in results if r['algorithm'] == algo]
    
    plt.boxplot([success_by_algo[algo] for algo in algorithms], labels=algorithms)
    plt.title('Success Rate by Algorithm')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    
    # Steps by algorithm
    plt.subplot(2, 2, 2)
    steps_by_algo = {}
    for algo in algorithms:
        steps_by_algo[algo] = [r['avg_steps'] for r in results if r['algorithm'] == algo]
    
    plt.boxplot([steps_by_algo[algo] for algo in algorithms], labels=algorithms)
    plt.title('Average Steps by Algorithm')
    plt.ylabel('Steps')
    
    # Success rate vs difficulty
    plt.subplot(2, 2, 3)
    difficulties = [r['difficulty'] for r in results]
    success_rates = [r['success_rate'] for r in results]
    
    # Color code by algorithm
    colors = ['blue', 'green', 'red', 'purple']
    for i, algo in enumerate(algorithms):
        algo_results = [r for r in results if r['algorithm'] == algo]
        plt.scatter(
            [r['difficulty'] for r in algo_results],
            [r['success_rate'] for r in algo_results],
            label=algo,
            color=colors[i % len(colors)]
        )
    
    plt.title('Success Rate vs Difficulty')
    plt.xlabel('Maze Difficulty')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    plt.legend()
    
    # Steps vs difficulty
    plt.subplot(2, 2, 4)
    for i, algo in enumerate(algorithms):
        algo_results = [r for r in results if r['algorithm'] == algo]
        plt.scatter(
            [r['difficulty'] for r in algo_results],
            [r['avg_steps'] for r in algo_results],
            label=algo,
            color=colors[i % len(colors)]
        )
    
    plt.title('Steps vs Difficulty')
    plt.xlabel('Maze Difficulty')
    plt.ylabel('Average Steps')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('dqn_evaluation_results.png')
    plt.show()
    
    # Print overall results
    print("\n===== Overall Results =====")
    print(f"Average Success Rate: {np.mean([r['success_rate'] for r in results]):.1f}%")
    print(f"Average Steps: {np.mean([r['avg_steps'] for r in results]):.1f}")
    print(f"Average Reward: {np.mean([r['avg_reward'] for r in results]):.2f}")
    
    # Results by algorithm
    print("\n===== Results by Algorithm =====")
    for algo in algorithms:
        algo_results = [r for r in results if r['algorithm'] == algo]
        print(f"\n{algo}:")
        print(f"  Success Rate: {np.mean([r['success_rate'] for r in algo_results]):.1f}%")
        print(f"  Average Steps: {np.mean([r['avg_steps'] for r in algo_results]):.1f}")
        print(f"  Average Reward: {np.mean([r['avg_reward'] for r in algo_results]):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained DQN agent on test mazes")
    parser.add_argument("--model_path", type=str, default="models/dqn/dqn_final.pt",
                        help="Path to trained model")
    parser.add_argument("--num_test_mazes", type=int, default=20,
                        help="Number of test mazes to evaluate on")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    main(args)