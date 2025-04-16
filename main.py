import numpy as np
import pygame
import heapq
import time  # To control movement speed

# Initialize maze
maze = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1]
])

# Define start and goal positions
start = (1, 1)
goal = (3, 5)

# Pygame settings
CELL_SIZE = 50  # Keep the same cell size
WIDTH, HEIGHT = maze.shape[1] * CELL_SIZE, maze.shape[0] * CELL_SIZE

# Double the game screen size
SCALE_FACTOR = 2
DISPLAY_WIDTH = WIDTH * SCALE_FACTOR
DISPLAY_HEIGHT = HEIGHT * SCALE_FACTOR

pygame.init()
screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption("Maze Visualization")

# Colors
WHITE = (255, 255, 255)  # Open path
BLACK = (0, 0, 0)        # Walls
GREEN = (0, 255, 0)      # Start position
RED = (255, 0, 0)        # Goal position

# Create a smaller surface for drawing, then scale it
surface = pygame.Surface((WIDTH, HEIGHT))



def a_star_search(start, goal):
    """A* algorithm to find the shortest path"""
    open_list = []
    heapq.heappush(open_list, (0, start))  # Priority queue
    came_from = {start: None}  # Track movement
    cost_so_far = {start: 0}  # Cost tracking

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            break  # Stop when we reach the goal

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
            next_pos = (current[0] + dx, current[1] + dy)

            if maze[next_pos] == 1:  # Ignore walls
                continue
            
            new_cost = cost_so_far[current] + 1  # Each move costs 1
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + abs(goal[0] - next_pos[0]) + abs(goal[1] - next_pos[1])  # A* heuristic
                heapq.heappush(open_list, (priority, next_pos))
                came_from[next_pos] = current

    # Reconstruct path
    path = []
    temp = goal
    while temp != start:
        path.append(temp)
        temp = came_from[temp]
    path.reverse()
    return path

# Agent's initial position (same as start position)
agent_pos = list(start)

def draw_agent():
    """Draw the moving agent (player)"""
    agent_rect = pygame.Rect(
        agent_pos[1] * CELL_SIZE, agent_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE
    )
    pygame.draw.rect(surface, (0, 0, 255), agent_rect)  # Blue agent




def draw_maze(path):
    """Function to draw the maze and the A* path"""
    surface.fill(WHITE)  # Fill background

    for row in range(maze.shape[0]):
        for col in range(maze.shape[1]):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            if maze[row, col] == 1:
                pygame.draw.rect(surface, BLACK, rect)  # Draw walls
            elif (row, col) == start:
                pygame.draw.rect(surface, GREEN, rect)  # Draw start position
            elif (row, col) == goal:
                pygame.draw.rect(surface, RED, rect)  # Draw goal position
            elif (row, col) in path:
                pygame.draw.rect(surface, (255, 255, 0), rect)  # Yellow for A* path

            pygame.draw.rect(surface, (200, 200, 200), rect, 1)  # Grid lines




# Run A* algorithm
path = a_star_search(start, goal)

# Main loop with movement
running = True
path_index = 0  # Start at first step in the path

while running:
    surface.fill(WHITE)
    draw_maze(path)  # Show A* path
    draw_agent()  # Draw agent

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if path_index < len(path):  # Move agent along path
        agent_pos[0], agent_pos[1] = path[path_index]  # Move to next path position
        path_index += 1
        time.sleep(0.3)  # Delay for smooth movement

    scaled_surface = pygame.transform.scale(surface, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    screen.blit(scaled_surface, (0, 0))
    pygame.display.flip()  # Update display


pygame.quit()
