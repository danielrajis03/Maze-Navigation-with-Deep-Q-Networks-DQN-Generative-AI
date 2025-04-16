import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from maze_env import MazeEnv  # Import the fixed environment
from dqn_agent import DQNAgent  # Import the fixed agent




# Environment and visualization settings
visualize_training = True
render_eval = True
save_model = True
model_path = "maze_dqn_model.pt"
# Initialize environment with your chosen difficulty
env = MazeEnv(difficulty='very_difficult')  # Choose: 'easy', 'moderate', 'difficult', 'very_difficult'

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)



# Training hyperparameters
num_episodes = 1000
max_steps = env.max_steps
eval_interval = 50
eval_episodes = 10

# Training tracking
rewards_history = []
steps_history = []
success_history = []
avg_rewards = []
success_rates = []
eval_steps = []
training_times = []
losses = []
# Add time tracking for successful solves
solution_times = []  # Time to solve maze in successful episodes
episode_numbers = []  # Episode numbers for successful solves

# For visualization
window_size = 20

def evaluate(agent, env, num_episodes=10, render=False):
    """Evaluate agent performance"""
    total_rewards = []
    total_steps = []
    successes = 0
    total_times = []  # Track times for successful episodes
    
    # Save current epsilon and set to minimal for evaluation
    current_epsilon = agent.epsilon
    agent.epsilon = agent.epsilon_min * 0.3  # Use minimal exploration
    
    
    for i in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        start_time = time.time()  # Start timing
        
        while not done and steps < max_steps:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward
            steps += 1
            
            if render and i == 0:  # Only render first episode for clarity
                env.render()
                time.sleep(0.1)  # Pause to visualize
            
            if done and tuple(state) == env.goal_pos:
                successes += 1
                total_times.append(time.time() - start_time)  # Record time for successful episodes
        
        total_rewards.append(ep_reward)
        total_steps.append(steps)
    
    # Restore original epsilon
    agent.epsilon = current_epsilon
    
    # Calculate metrics
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    success_rate = successes / num_episodes
    avg_time = np.mean(total_times) if total_times else 0
    
    return avg_reward, avg_steps, success_rate, avg_time

# Main training loop
start_time = time.time()
best_success_rate = 0

print("Starting training...")
print(f"Maze shape: {env.maze.shape}, Start: {env.start_pos}, Goal: {env.goal_pos}")

for episode in range(1, num_episodes + 1):
    # Reset environment
    state = env.reset()
    episode_reward = 0
    episode_loss = 0
    num_updates = 0
    done = False
    steps = 0
    episode_start_time = time.time()  # Time the entire episode
    
    # Episode loop
    while not done and steps < max_steps:
        # Select and take action
        action = agent.act(state)

        
        next_state, reward, done, _ = env.step(action)
        
        # Store experience
        agent.remember(state, action, reward, next_state, done)
        
        # Update state and tracking
        state = next_state
        episode_reward += reward
        steps += 1
        
        # Train on batch
        loss = agent.replay()
        if loss > 0:
            episode_loss += loss
            num_updates += 1
    
    # Track success and time for solved episodes
    success = tuple(state) == env.goal_pos
    success_history.append(success)
    
    if success:
        episode_time = time.time() - episode_start_time
        solution_times.append(episode_time)
        episode_numbers.append(episode)
    
    # Decay exploration rate
    agent.decay_epsilon()
    
    # Update target network
    if episode % agent.target_update_freq == 0:
        agent.update_target_network()
        
    # Store metrics
    rewards_history.append(episode_reward)
    steps_history.append(steps)
    if num_updates > 0:
        losses.append(episode_loss / num_updates)
    else:
        losses.append(0)
    
    # Runtime tracking
    training_times.append(time.time() - start_time)
    
    # Print progress
    if episode % 10 == 0:
        recent_rewards = rewards_history[-20:] if len(rewards_history) >= 20 else rewards_history
        recent_success = success_history[-20:] if len(success_history) >= 20 else success_history
        recent_success_rate = np.mean(recent_success) * 100
        
        print(f"Episode {episode}/{num_episodes} - "
              f"Reward: {episode_reward:.2f} - "
              f"Steps: {steps}/{max_steps} - "
              f"Success: {success} - "
              f"Recent Success Rate: {recent_success_rate:.1f}% - "
              f"Epsilon: {agent.epsilon:.4f}")
    
    # Evaluation
    if episode % eval_interval == 0:
        eval_reward, avg_step, success_rate, avg_time = evaluate(agent, env, eval_episodes, render=render_eval and episode > 500)
        avg_rewards.append(eval_reward)
        eval_steps.append(avg_step)
        success_rates.append(success_rate)
        
        print(f"\nEVALUATION - Episode {episode}/{num_episodes}")
        print(f"Average Reward: {eval_reward:.2f}")
        print(f"Average Steps: {avg_step:.2f}")
        print(f"Success Rate: {success_rate*100:.1f}%")
        print(f"Average Solution Time: {avg_time*1000:.2f} ms\n")
        
        # Save best model
        if success_rate > best_success_rate and save_model:
            best_success_rate = success_rate
            agent.save(model_path)
            print(f"✅ New best model saved with success rate: {best_success_rate*100:.1f}%")

# Final evaluation
print("\n--- Training Complete ---")
final_reward, final_steps, final_success, final_time = evaluate(agent, env, 20, render=True)
print(f"Final Evaluation - Success Rate: {final_success*100:.1f}%, Avg Steps: {final_steps:.1f}, Avg Time: {final_time*1000:.2f} ms")

# Plot results
if visualize_training:
    plt.figure(figsize=(15, 15))  # Increased figure size to fit 7 subplots
    
    # Plot rewards
    plt.subplot(3, 3, 1)
    plt.plot(rewards_history, alpha=0.4, color='blue')
    plt.plot(np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid'), 
             color='blue', linewidth=2)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot steps
    plt.subplot(3, 3, 2)
    plt.plot(steps_history, alpha=0.4, color='green')
    plt.plot(np.convolve(steps_history, np.ones(window_size)/window_size, mode='valid'),
             color='green', linewidth=2)
    plt.axhline(y=max_steps, color='r', linestyle='--', alpha=0.3)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # Plot success rate
    plt.subplot(3, 3, 3)
    # Calculate moving success rate
    window_success = np.convolve(np.array(success_history, dtype=float), 
                               np.ones(window_size)/window_size, mode='valid')
    plt.plot(window_success, color='purple', linewidth=2)
    plt.title('Success Rate (Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.ylim(-0.05, 1.05)
    
    # Plot evaluation metrics
    plt.subplot(3, 3, 4)
    eval_x = np.arange(eval_interval, num_episodes + 1, eval_interval)
    plt.plot(eval_x, success_rates, 'bo-', label='Success Rate')
    plt.title('Evaluation Success Rate')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    
    # Plot losses
    plt.subplot(3, 3, 5)
    plt.plot(losses, alpha=0.4, color='red')
    plt.plot(np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
             if len(losses) >= window_size else losses, 
             color='red', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    # Plot training time
    plt.subplot(3, 3, 6)
    plt.plot(training_times, color='orange')
    plt.title('Cumulative Training Time')
    plt.xlabel('Episode')
    plt.ylabel('Time (seconds)')
    
    # New plot: Time Taken to Solve Maze
    plt.subplot(3, 3, 7)
    if solution_times:
        # Convert to milliseconds for better readability
        solution_times_ms = [t * 1000 for t in solution_times]
        plt.plot(episode_numbers, solution_times_ms, 'ro-', alpha=0.6, markersize=3)
        
        # Add moving average if we have enough points
        if len(solution_times_ms) >= window_size:
            window = min(window_size, len(solution_times_ms) // 2)
            if window > 1:
                # Use a proper window size for the moving average
                # We'll manually compute the moving average to ensure dimensions match
                moving_avg = []
                for i in range(len(solution_times_ms) - window + 1):
                    moving_avg.append(np.mean(solution_times_ms[i:i+window]))
                
                # Make sure x and y dimensions match
                valid_x = episode_numbers[:len(moving_avg)]
                
                # Now plot with matching dimensions
                plt.plot(valid_x, moving_avg, 'r-', linewidth=2)
        
        plt.title('Time to Solve Maze')
        plt.xlabel('Episode')
        plt.ylabel('Time (milliseconds)')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "No successful solutions recorded", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('dqn_training_results.png')
    plt.show()

# Load best model for demo
if save_model and best_success_rate > 0:
    print(f"\nLoading best model (Success Rate: {best_success_rate*100:.1f}%)")
    agent.load(model_path)
    
    # Final visualization with best model
    print("\nFinal demo with best model:")
    state = env.reset()
    done = False
    steps = 0
    total_reward = 0
    demo_start_time = time.time()
    
    while not done and steps < max_steps:
        env.render()
        time.sleep(0.3)  # Slow down for visualization
        
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        
        state = next_state
        total_reward += reward
        steps += 1
        
        print(f"Step {steps}: Action {action}, Reward {reward:.1f}, Total Reward {total_reward:.1f}")
    
    demo_time = time.time() - demo_start_time
    solution_status = 'Success!' if tuple(state) == env.goal_pos else 'Failed'
    print(f"\nDemo complete - {solution_status}")
    print(f"Steps: {steps}, Total Reward: {total_reward:.1f}, Time: {(demo_time - steps*0.3)*1000:.2f} ms (excluding visualization delays)")