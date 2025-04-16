import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from dqn_agent import DQNAgent
from maze_env import MazeEnv
from maze_curriculum import MazeCurriculum
import json



def train_dqn_with_curriculum(args):
    """Train a DQN agent on a curriculum of progressively harder mazes."""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load curriculum
    curriculum = MazeCurriculum()
    curriculum.load_curriculum(args.curriculum_dir)
    
    if not curriculum.curriculum:
        print("Error: No curriculum found. Please generate a curriculum first.")
        return
    
    # Initialize DQN agent
    agent = DQNAgent(state_dim=2, action_dim=4)
    
    # Training stats
    all_rewards = []
    all_success_rates = []
    all_steps = []
    level_success_rates = []
    
    # Train agent on curriculum
    for level, level_mazes in enumerate(curriculum.curriculum):
        print(f"\n===== Training on difficulty level {level+1}/{len(curriculum.curriculum)} =====")
        level_rewards = []
        level_steps = []
        level_successes = []
        
        # Continue to the next level only after achieving a minimum success rate
        level_complete = False
        
        while not level_complete:
            # Choose a random maze from this level for this training batch
            maze_data = np.random.choice(level_mazes)
            
            # Create environment with this maze
            env = MazeEnv(
                custom_maze=maze_data['maze'],
                custom_start=maze_data['start'],
                custom_goal=maze_data['end'],
                custom_max_steps=args.max_steps
            )
            
            print(f"Training on maze with difficulty {maze_data['actual_difficulty']:.2f}")
            print(f"Maze shape: {env.maze.shape}, Start: {env.start_pos}, Goal: {env.goal_pos}")
            
            # Train for some episodes on this maze
            for episode in tqdm(range(args.episodes_per_maze), desc=f"Level {level+1} training"):
                state = env.reset()
                done = False
                total_reward = 0
                steps = 0
                
                while not done and steps < args.max_steps:
                    # Select action
                    action = agent.act(state)
                    
                    # Take step in environment
                    next_state, reward, done, _ = env.step(action)
                    
                    # Store experience in replay memory
                    agent.remember(state, action, reward, next_state, done)
                    
                    # Update state and totals
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    # Train network
                    agent.replay()
                
                # Decay exploration rate
                agent.decay_epsilon()
                
                # Update target network periodically
                if episode % agent.target_update_freq == 0:
                    agent.update_target_network()
                
                # Track results
                level_rewards.append(total_reward)
                level_steps.append(steps)
                
                # Check if agent reached the goal
                success = tuple(state) == env.goal_pos
                level_successes.append(success)
                
                # Print progress occasionally
                if (episode + 1) % 100 == 0:
                    recent_rewards = level_rewards[-100:]
                    recent_success = level_successes[-100:]
                    recent_success_rate = np.mean(recent_success) * 100
                    
                    print(f"Episode {episode+1}/{args.episodes_per_maze} - "
                          f"Reward: {total_reward:.2f} - "
                          f"Steps: {steps}/{args.max_steps} - "
                          f"Success: {success} - "
                          f"Recent Success Rate: {recent_success_rate:.1f}% - "
                          f"Epsilon: {agent.epsilon:.4f}")
            
            # Evaluate on all mazes in this level
            print("\nEvaluating on all mazes in this level...")
            success_rates = []
            
            for eval_maze in level_mazes:
                eval_env = MazeEnv(
                    custom_maze=eval_maze['maze'],
                    custom_start=eval_maze['start'],
                    custom_goal=eval_maze['end'],
                    custom_max_steps=args.max_steps
                )
                
                # Run evaluation episodes
                eval_successes = []
                for _ in range(args.eval_episodes):
                    state = eval_env.reset()
                    done = False
                    steps = 0
                    
                    # Save epsilon and use minimal exploration for evaluation
                    eval_epsilon = agent.epsilon
                    agent.epsilon = 0.05
                    
                    while not done and steps < args.max_steps:
                        action = agent.act(state)
                        next_state, _, done, _ = eval_env.step(action)
                        state = next_state
                        steps += 1
                    
                    # Check if goal was reached
                    success = tuple(state) == eval_env.goal_pos
                    eval_successes.append(success)
                    
                    # Restore epsilon
                    agent.epsilon = eval_epsilon
                
                # Calculate success rate for this maze
                success_rate = np.mean(eval_successes) * 100
                success_rates.append(success_rate)
                
                print(f"Maze difficulty {eval_maze['actual_difficulty']:.2f} - Success Rate: {success_rate:.1f}%")
            
            # Calculate overall success rate for this level
            level_success_rate = np.mean(success_rates)
            print(f"\nOverall success rate for level {level+1}: {level_success_rate:.1f}%")
            
            # Move to next level if success rate is high enough
            if level_success_rate >= args.success_threshold:
                level_complete = True
                level_success_rates.append(level_success_rate)
                
                # Save checkpoint after completing level
                agent.save(os.path.join(args.output_dir, f"dqn_level_{level+1}.pt"))
                print(f"Saved model checkpoint for level {level+1}")
            else:
                print(f"Success rate {level_success_rate:.1f}% below threshold {args.success_threshold:.1f}%. Continuing training on this level.")
        
        # Add level stats to overall stats
        all_rewards.extend(level_rewards)
        all_steps.extend(level_steps)
        all_success_rates.append(level_success_rate)
    
    # Save final model
    agent.save(os.path.join(args.output_dir, "dqn_final.pt"))
    print("Saved final model")
    
    # Plot training progress
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(3, 1, 1)
    plt.plot(all_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot steps
    plt.subplot(3, 1, 2)
    plt.plot(all_steps)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # Plot success rates per level
    plt.subplot(3, 1, 3)
    plt.bar(range(1, len(all_success_rates) + 1), all_success_rates)
    plt.title('Success Rate by Difficulty Level')
    plt.xlabel('Difficulty Level')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'dqn_curriculum_training.png'))
    plt.show()
    
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent on maze curriculum")
    parser.add_argument("--curriculum_dir", type=str, default="models/curriculum",
                        help="Directory containing the curriculum")
    parser.add_argument("--output_dir", type=str, default="models/dqn",
                        help="Directory to save trained models")
    parser.add_argument("--episodes_per_maze", type=int, default=1000,
                        help="Number of episodes to train on each maze before evaluation")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--eval_episodes", type=int, default=20,
                        help="Number of episodes for evaluation")
    parser.add_argument("--success_threshold", type=float, default=80.0,
                        help="Success rate threshold to advance to next level")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    train_dqn_with_curriculum(args)