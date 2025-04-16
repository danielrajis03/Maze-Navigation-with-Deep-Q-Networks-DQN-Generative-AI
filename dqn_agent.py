import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    """Enhanced Deep Q-Network with better state representation"""
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights to break symmetry
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
            
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.output_layer(x)


class DQNAgent:
    """Improved DQN Agent with fixed implementation"""
    def __init__(self, state_dim, action_dim, maze_shape=(5, 7)):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.maze_shape = maze_shape
        
        # Enhanced state representation
        self.state_feature_dim = 6  # position (x,y) + goal (x,y) + distance + normalized distance
        
        # Hyperparameters - tuned for maze learning
        self.memory = deque(maxlen=10000)
        self.gamma = 0.98  # Discount factor
        self.epsilon = 1.0  # Start fully exploring
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Networks
        self.model = DQN(self.state_feature_dim, action_dim)
        self.target_model = DQN(self.state_feature_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_counter = 0
        self.target_update_freq = 20  # Update target network every 20 episodes
        
    def get_enhanced_state(self, state, goal_pos=(3, 5)):
        """Convert raw state to more informative representation"""
        x, y = state if isinstance(state, tuple) else state.tolist()
        goal_x, goal_y = goal_pos
        
        # Manhattan distance to goal
        distance = abs(x - goal_x) + abs(y - goal_y)
        
        # Normalized distance (divide by maximum possible distance in the maze)
        max_distance = self.maze_shape[0] + self.maze_shape[1]
        normalized_distance = distance / max_distance
        
        # Return enhanced state representation
        return np.array([
            x / self.maze_shape[0],  # Normalized x position
            y / self.maze_shape[1],  # Normalized y position
            goal_x / self.maze_shape[0],  # Normalized goal x
            goal_y / self.maze_shape[1],  # Normalized goal y
            distance,  # Raw distance
            normalized_distance  # Normalized distance
        ], dtype=np.float32)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        # Convert states to enhanced representation
        enhanced_state = self.get_enhanced_state(state)
        enhanced_next_state = self.get_enhanced_state(next_state)
        
        # Store transition
        self.memory.append((enhanced_state, action, reward, enhanced_next_state, done))

    def act(self, state):
        """Select action using epsilon-greedy policy"""
        enhanced_state = self.get_enhanced_state(state)
        
        # Exploration
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Exploitation - select best action based on Q-values
        enhanced_state_tensor = torch.FloatTensor(enhanced_state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(enhanced_state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        """Train on batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return 0
            
        # Sample random minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch tensors
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).unsqueeze(1)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))
        
        # Current Q values
        current_q_values = self.model(states).gather(1, actions).squeeze(1)
        
        # Next Q values from target network (Double DQN approach)
        with torch.no_grad():
            # Get actions from policy network
            best_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            # Get Q-values from target network
            next_q_values = self.target_model(next_states).gather(1, best_actions).squeeze(1)
            # Calculate target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q_values, target_q_values)  # Huber loss for stability
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network weights with current model weights"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self, path):
        """Save model"""
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        """Load model"""
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.model.state_dict())