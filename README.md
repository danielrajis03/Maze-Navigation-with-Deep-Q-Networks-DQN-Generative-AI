# 🧠 Deep Q-Network Maze Navigation #


# 📌 Overview #
This project explores the use of **Deep Q-Networks (DQN)** for autonomous maze navigation. The goal was to train an intelligent agent that learns to navigate increasingly complex mazes using reinforcement learning, outperforming classical search algorithms in adaptability and efficiency.

The project incorporates:
- **Curriculum Learning** for progressive difficulty.
- **Benchmarking with A\*** for comparative evaluation.
- **Procedural Maze Generation** using **VAE** and **GAN** models.
- **Evaluation on unseen mazes** to test generalisation.

# 🚀 Key Features #

## 🤖 Agent Architecture ##
**Deep Q-Network (DQN)** using:
- Input: `(x, y)` coordinates.
- Hidden Layers: Two fully-connected ReLU layers (128 neurons each).
- Output: Q-values for four actions (up/down/left/right).

**Training Techniques**:
- Replay Memory with 20,000 transitions.
- Epsilon-greedy exploration with decay.
- Mean Squared Error (MSE) loss function.

## 🌍 Maze Environment ##
Grid-based maze using OpenAI Gym.

**Reward Shaping**:
- +10 for reaching the goal
- −0.1 per step
- −1.0 if blocked
- +1.0 for moving closer to the goal

# 📚 Curriculum Learning #
- Maze difficulties: Easy → Moderate → Hard → Very Hard
- Automatic promotion based on consistent success rate
- Adaptive reward tuning for harder levels

# 🧮 Benchmarking #
**A\*** pathfinding used as a baseline with Manhattan heuristic.

Compared on:
- Path length
- Computation time
- Step efficiency

# 🎯 Stretch Goal: Generative AI #
**VAE & GAN** models trained to generate new maze layouts.

Filtered for:
- Solvability (via A*)
- Difficulty labelling (e.g., path length, wall density)
- Injected into curriculum pipeline to test generalisation on novel topologies

# 🧪 Skills & Technologies Used #

## ⚙️ Tools & Frameworks ##
- **Python** (NumPy, OpenAI Gym)
- **PyTorch** (Neural network implementation)
- **Matplotlib** (Training visualisation)
- **VAE/GAN** (for maze generation)
- **A\*** (Custom implementation for benchmarking)

# 📁 Repository Contents #
- `dqn_agent.py`: Core DQN logic
- `maze_env.py`: Custom OpenAI Gym maze environment
- `train_dqn_curriculum.py`: Training pipeline with curriculum logic
- `evaluate_agent.py`: Performance evaluation scripts
- `aStar.py`: Heuristic-based pathfinding baseline
- `maze_vae.py` & `maze_gan.py`: Generative model architectures

# Author #
Created by Daniel Rajis
