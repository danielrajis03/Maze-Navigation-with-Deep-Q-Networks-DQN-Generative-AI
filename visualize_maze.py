import time
import pygame
import sys
from maze_env import MazeEnv
from dqn_agent import DQNAgent

# Initialize environment and agent
env = MazeEnv(difficulty='easy')
agent = DQNAgent(state_dim=2, action_dim=4)

# Load the trained model
try:
    agent.load("maze_dqn_model.pt")  # Or large_maze_dqn_model.pt for the larger maze
    print("Loaded trained model successfully")
except:
    print("No trained model found. Using untrained agent.")

# Get screen info for fullscreen display
pygame.init()
screen_info = pygame.display.Info()
monitor_width, monitor_height = screen_info.current_w, screen_info.current_h

# Calculate cell size to fit the maze on screen with some padding
padding = 50  # Padding around the maze
max_cell_width = (monitor_width - 2 * padding) // env.maze.shape[1]
max_cell_height = (monitor_height - 2 * padding) // env.maze.shape[0]
CELL_SIZE = min(max_cell_width, max_cell_height)

# Calculate actual display size
WIDTH, HEIGHT = env.maze.shape[1] * CELL_SIZE, env.maze.shape[0] * CELL_SIZE

# Create the screen - either fullscreen or centered window
fullscreen = False  # Set to True for fullscreen, False for window
if fullscreen:
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    # Calculate offset to center the maze on screen
    offset_x = (monitor_width - WIDTH) // 2
    offset_y = (monitor_height - HEIGHT) // 2
else:
    screen = pygame.display.set_mode((WIDTH + 2 * padding, HEIGHT + 2 * padding))
    offset_x = padding
    offset_y = padding

pygame.display.set_caption("AI Maze Solver Visualization")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)

# Font for displaying information
pygame.font.init()
font = pygame.font.SysFont('Arial', 24)
small_font = pygame.font.SysFont('Arial', 16)

def draw_maze(agent_pos, step_count=0, path=None):
    """Draws the maze and AI's current position"""
    screen.fill(WHITE)
    
    # Draw maze grid with offset
    for row in range(env.maze.shape[0]):
        for col in range(env.maze.shape[1]):
            rect = pygame.Rect(
                offset_x + col * CELL_SIZE, 
                offset_y + row * CELL_SIZE, 
                CELL_SIZE, 
                CELL_SIZE
            )
            
            # Fill cells based on maze properties
            if env.maze[row, col] == 1:
                pygame.draw.rect(screen, BLACK, rect)  # Wall
            else:
                # Draw path cells
                if path and (row, col) in path:
                    pygame.draw.rect(screen, YELLOW, rect)  # Path
                
                # Draw special cells
                if (row, col) == env.start_pos:
                    pygame.draw.rect(screen, GREEN, rect)  # Start
                elif (row, col) == env.goal_pos:
                    pygame.draw.rect(screen, RED, rect)  # Goal
                elif (row, col) == tuple(agent_pos):
                    pygame.draw.rect(screen, BLUE, rect)  # Agent
            
            # Draw grid lines
            pygame.draw.rect(screen, GRAY, rect, 1)
    
    # Draw step counter and instructions
    step_text = font.render(f'Steps: {step_count}', True, BLACK)
    screen.blit(step_text, (offset_x, offset_y - 30))
    
    # Add instructions
    instructions = [
        "Controls: ESC to quit, F to toggle fullscreen, R to reset, SPACE to pause/resume",
        f"Maze size: {env.maze.shape[0]}x{env.maze.shape[1]}, Cell size: {CELL_SIZE}"
    ]
    
    for i, text in enumerate(instructions):
        text_surface = small_font.render(text, True, BLACK)
        screen.blit(text_surface, (offset_x, offset_y + HEIGHT + 10 + i * 20))

def main():
    global screen, offset_x, offset_y, fullscreen
    
    # Initialize game state
    state = env.reset()
    paused = False
    running = True
    step_count = 0
    path = []  # Store agent's path
    
    # Main game loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f:
                    # Toggle fullscreen
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        screen_info = pygame.display.Info()
                        monitor_width, monitor_height = screen_info.current_w, screen_info.current_h
                        offset_x = (monitor_width - WIDTH) // 2
                        offset_y = (monitor_height - HEIGHT) // 2
                    else:
                        screen = pygame.display.set_mode((WIDTH + 2 * padding, HEIGHT + 2 * padding))
                        offset_x = padding
                        offset_y = padding
                elif event.key == pygame.K_r:
                    # Reset maze
                    state = env.reset()
                    step_count = 0
                    path = []
                elif event.key == pygame.K_SPACE:
                    # Pause/resume
                    paused = not paused
        
        # Update game state if not paused
        if not paused:
            # Record current position in path
            path.append(tuple(state))
            
            # AI selects and takes action
            action = agent.act(state)
            next_state, _, done, _ = env.step(action)
            
            # Update position
            state = next_state
            step_count += 1
        
        # Draw everything
        draw_maze(state, step_count, path)
        pygame.display.flip()
        
        # Control speed
        time.sleep(0.3)
        
        # Check if maze is solved
        if done and tuple(state) == env.goal_pos:
            # Draw final state
            draw_maze(state, step_count, path)
            pygame.display.flip()
            
            # Show success message
            success_text = font.render("AI reached the goal! 🎉", True, BLACK)
            text_rect = success_text.get_rect(center=(offset_x + WIDTH//2, offset_y - 60))
            screen.blit(success_text, text_rect)
            pygame.display.flip()
            
            # Wait for a key press
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and 
                                                    (event.key == pygame.K_ESCAPE or 
                                                     event.key == pygame.K_RETURN or
                                                     event.key == pygame.K_SPACE)):
                        waiting = False
                        
            # Reset for another run
            state = env.reset()
            step_count = 0
            path = []

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()