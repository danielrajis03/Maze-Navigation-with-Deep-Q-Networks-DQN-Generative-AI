import numpy as np
import gym
from gym import spaces

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}


    # Different maze difficulties
    MAZES = {
        'easy': {
            'layout': [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Start at (1,1)
                [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ],
            'start': (1, 1),
            'goal': (8, 8),
            'max_steps': 50
        },
        
        'moderate': {
            'layout': [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # Start at (1,1)
                [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
                [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 0, 1, 0, 0, 0, 1],
                [1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ],
            'start': (1, 1),
            'goal': (8, 8),
            'max_steps': 70
        },
        
        'difficult': {
            'layout': [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
],
            'start': (1, 1),
            'goal': (15, 18),
            'max_steps': 600
        },
        
        'very_difficult': {
            'layout': [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ],
            'start': (1, 1),
            'goal': (19, 18),
            'max_steps': 600
        }
    }

    def __init__(self, difficulty='easy', custom_maze=None, custom_start=None, custom_goal=None, custom_max_steps=None):
        super(MazeEnv, self).__init__()

    # Check if we're using a custom maze or a predefined one
        if custom_maze is not None:
        # Use custom maze configuration
            self.maze = np.array(custom_maze)
            self.start_pos = custom_start
            self.goal_pos = custom_goal
            self.max_steps = custom_max_steps if custom_max_steps is not None else 600
            self.difficulty = 'custom'
        else:
        # Select maze based on difficulty
            if difficulty not in self.MAZES:
                print(f"Warning: Difficulty '{difficulty}' not found. Defaulting to 'easy'.")
            difficulty = 'easy'
            
        maze_config = self.MAZES[difficulty]
        
        # Set maze properties
        self.maze = np.array(maze_config['layout'])
        self.start_pos = maze_config['start']
        self.goal_pos = maze_config['goal']
        self.max_steps = maze_config['max_steps']
        self.difficulty = difficulty

    # Initialize agent position and step counter
        self.agent_pos = list(self.start_pos)
        self.steps = 0

    # Define observation and action spaces
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(
        low=np.array([0, 0]), 
        high=np.array([self.maze.shape[0]-1, self.maze.shape[1]-1]), 
        dtype=np.int32
    )

    def step(self, action):
        """Take a step in the environment"""
        self.steps += 1
        prev_pos = tuple(self.agent_pos)
        x, y = self.agent_pos
        done = False
        
        # Apply movement
        if action == 0 and self.maze[x - 1, y] == 0:  # Up
            x -= 1
        elif action == 1 and self.maze[x + 1, y] == 0:  # Down
            x += 1
        elif action == 2 and self.maze[x, y - 1] == 0:  # Left
            y -= 1
        elif action == 3 and self.maze[x, y + 1] == 0:  # Right
            y += 1

        self.agent_pos = [x, y]
        
        # Check goal
        if tuple(self.agent_pos) == self.goal_pos:
            reward = 30.0  # Big reward for reaching goal
            done = True
        else:
            # Meaningful reward structure
            goal_x, goal_y = self.goal_pos
            current_distance = abs(x - goal_x) + abs(y - goal_y)
            previous_distance = abs(prev_pos[0] - goal_x) + abs(prev_pos[1] - goal_y)
            
            # -0.1 as small step penalty
            reward = -0.1
            
            # +1 for moving closer to goal, -1 for moving away
            if current_distance < previous_distance:
                reward += 2.0
            elif current_distance > previous_distance:
                reward -= 2.0
                
            # -0.5 extra penalty for bumping into walls
            if prev_pos == tuple(self.agent_pos) and action in [0, 1, 2, 3]:
                reward -= 0.75 #was getting stuck between walls so increase penalty
        # End episode if max steps reached
        if self.steps >= self.max_steps:
            done = True
        
        return np.array(self.agent_pos, dtype=np.int32), reward, done, {'difficulty': self.difficulty}

    def reset(self):
        """Reset environment to start state"""
        self.agent_pos = list(self.start_pos)
        self.steps = 0
        return np.array(self.agent_pos, dtype=np.int32)
    
    def render(self, mode='human'):
        """Visualization of the environment"""
        if mode == 'human':
            maze_view = np.copy(self.maze).astype(str)
            
            # Replace 0s and 1s with more readable symbols
            maze_view[maze_view == '0'] = ' '
            maze_view[maze_view == '1'] = '#'
            
            # Mark start, goal and agent
            x, y = self.agent_pos
            sx, sy = self.start_pos
            gx, gy = self.goal_pos
            
            if (x, y) != (sx, sy) and (x, y) != (gx, gy):
                maze_view[x, y] = 'A'
            maze_view[sx, sy] = 'S'
            maze_view[gx, gy] = 'G'
            
            # Print the maze
            print("\n".join(["".join(row) for row in maze_view]))
            print(f"Agent position: {self.agent_pos}, Steps: {self.steps}/{self.max_steps}")
            print(f"Difficulty: {self.difficulty.capitalize()}")
    
    def set_difficulty(self, difficulty):
        """Change the maze difficulty"""
        if difficulty not in self.MAZES:
            print(f"Warning: Difficulty '{difficulty}' not found. Keeping current difficulty.")
            return False
            
        # Update maze configuration
        maze_config = self.MAZES[difficulty]
        self.maze = np.array(maze_config['layout'])
        self.start_pos = maze_config['start']
        self.goal_pos = maze_config['goal']
        self.max_steps = maze_config['max_steps']
        self.difficulty = difficulty
        
        # Reset position
        self.reset()
        return True