import pygame
import json
import random
import sys
from collections import deque
import heapq



# Initialize Pygame
pygame.init()

# Constants
GRID_ROWS = 5
GRID_COLS = 10
CELL_SIZE = 80
WINDOW_WIDTH = GRID_COLS * CELL_SIZE
WINDOW_HEIGHT = GRID_ROWS * CELL_SIZE + 100
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
BROWN = (139, 69, 19)

# Load team configuration
def load_config():
    try:
        with open('team_config.json', 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print("Error: team_config.json not found!")
        sys.exit(1)

# World class
class BangaloreWumpusWorld:
    def __init__(self, config):
        self.config = config
        self.seed = config['seed']
        random.seed(self.seed)

        # Initialize grid
        self.grid = [[{'type': 'empty', 'percepts': [], 'weight': random.randint(1,15)}
              for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]


        # Agent starts at bottom-left diagonal
        self.agent_start = (0, GRID_ROWS - 1)
        self.agent_pos = list(self.agent_start)
        self.agent_path = []

        # Game state
        self.game_over = False
        self.game_won = False
        self.message = ""

        # Generate world
        self._generate_world()

    def _generate_world(self):
        """Generate random world elements based on config"""
        num_traffic_lights = self.config['grid_config']['traffic_lights']
        num_cows = self.config['grid_config']['cows']
        num_pits = self.config['grid_config']['pits']

        # Generate available positions (exclude agent start)
        available_positions = [(x, y) for x in range(GRID_COLS) for y in range(GRID_ROWS)
                               if (x, y) != tuple(self.agent_start)]

        random.shuffle(available_positions)

        # Place traffic lights
        for i in range(num_traffic_lights):
            if available_positions:
                pos = available_positions.pop()
                self.grid[pos[1]][pos[0]]['type'] = 'traffic_light'

        # Place cows
        for i in range(num_cows):
            if available_positions:
                pos = available_positions.pop()
                self.grid[pos[1]][pos[0]]['type'] = 'cow'

        # Place pits
        for i in range(num_pits):
            if available_positions:
                pos = available_positions.pop()
                self.grid[pos[1]][pos[0]]['type'] = 'pit'

        # Place goal
        if available_positions:
            goal_pos = available_positions.pop()
            self.grid[goal_pos[1]][goal_pos[0]]['type'] = 'goal'
            self.goal_pos = goal_pos

        # Generate percepts
        self._generate_percepts()

    def _generate_percepts(self):
        """Generate percepts for all cells based on adjacent elements"""
        for y in range(GRID_ROWS):
            for x in range(GRID_COLS):
                neighbors = self._get_neighbors(x, y)

                for nx, ny in neighbors:
                    cell_type = self.grid[ny][nx]['type']

                    if cell_type == 'pit':
                        if 'breeze' not in self.grid[y][x]['percepts']:
                            self.grid[y][x]['percepts'].append('breeze')

                    elif cell_type == 'cow':
                        if 'moo' not in self.grid[y][x]['percepts']:
                            self.grid[y][x]['percepts'].append('moo')

                    elif cell_type == 'traffic_light':
                        if 'light' not in self.grid[y][x]['percepts']:
                            self.grid[y][x]['percepts'].append('light')

    def _get_neighbors(self, x, y):
        """Get valid adjacent neighbors (no diagonals - up, down, left, right only)"""
        neighbors = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_COLS and 0 <= ny < GRID_ROWS:
                neighbors.append((nx, ny))

        return neighbors

    def move_agent(self, new_x, new_y):
        """Move agent to new position and handle interactions"""
        if self.game_over or self.game_won:
            return

        # Check bounds
        if not (0 <= new_x < GRID_COLS and 0 <= new_y < GRID_ROWS):
            return

        # Only allow orthogonal movement (no diagonals)
        dx = abs(new_x - self.agent_pos[0])
        dy = abs(new_y - self.agent_pos[1])
        if dx + dy != 1:
            return

        self.agent_pos = [new_x, new_y]
        self.agent_path.append((new_x, new_y))

        cell_type = self.grid[new_y][new_x]['type']

        # Handle traffic light - simulate delay
        if cell_type == 'traffic_light':
            self.message = "Waiting at traffic signal..."
            self._simulate_traffic_delay()

        # Handle cow - reset to start
        elif cell_type == 'cow':
            self.message = "Moo! Cow encountered - returning to start!"
            self.agent_pos = list(self.agent_start)
            self.agent_path = []
            # TODO: Students need to handle this in their A* implementation
            # When cow is hit, pathfinding must restart from beginning

        # Handle pit - game over
        elif cell_type == 'pit':
            self.message = "Game Over - Fell into a pit!"
            self.game_over = True

        # Handle goal - win
        elif cell_type == 'goal':
            self.message = "Goal Reached! You won!"
            self.game_won = True

    def _simulate_traffic_delay(self):
        """Simulate traffic light delay using nested loop"""
        # This creates a time delay without using time.sleep
        delay = 0
        for i in range(1000):
            for j in range(10000):
                delay += 1

    def get_current_percepts(self):
        """Get percepts at current agent position"""
        x, y = self.agent_pos
        return self.grid[y][x]['percepts']

    # TODO: Students must implement A* pathfinding
    # def find_path_astar(self):
    #     """
    #     TODO: Implement A* pathfinding algorithm here

    #     Requirements:
    #     1. Find path from current agent position to goal
    #     2. Movement: Only up, down, left, right (NO diagonals)
    #     3. Handle obstacles:
    #        - Pits: Must avoid completely (game over if entered)
    #        - Traffic lights: Can pass through but have higher cost (simulate wait)
    #        - Cows: Must handle collision - if agent hits cow, it resets to start
    #                Your A* needs to account for this (hint: you might need replanning)

    #     4. Return path as list of (x, y) tuples
    #     5. If no path exists, return None and set self.message = "Path Not Found"

    #     Hints:
    #     - Use manhattan distance as heuristic
    #     - Traffic light cells should have cost > 1 (recommend cost = 5)
    #     - Cow cells: Either avoid them OR handle reset in your logic
    #     - Use self._get_neighbors(x, y) to get valid adjacent cells
    #     - Remember: only orthogonal movement allowed!

    #     Algorithm structure:
    #     - Priority queue (use heapq or implement your own)
    #     - Track: f_score, g_score, came_from
    #     - f(n) = g(n) + h(n)
    #     - g(n) = cost from start to n
    #     - h(n) = heuristic (manhattan distance to goal)

    #     Example return:
    #     return [(0, 4), (1, 4), (2, 4), (2, 3), ...]  # List of coordinates
    #     """
    #     # TODO: Your implementation here

    #     start = tuple(self.agent_pos)
    #     goal = self.goal_pos
    #     # Priority Queue for Open Set: stores (f_score, (x, y))
    #     open_set = []
    #     heapq.heappush(open_set, (0, start))
        
    #     # Tracking dictionaries
    #     came_from = {}          # To reconstruct path
    #     g_score = {start: 0}    # Cost from start to current node
        
    #     # f_score = g_score + h_score (heuristic)
    #     # Initial heuristic is Manhattan distance from start to goal
    #     start_h = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    #     f_score = {start: start_h}
        
    #     # Set to keep track of items in open_set for lookup efficiency
    #     open_set_hash = {start}

    #     while open_set:
    #         # Pop node with lowest f_score
    #         current_f, current = heapq.heappop(open_set)
    #         open_set_hash.remove(current)

    #         # 2. Goal Check
    #         if current == goal:
    #             # Reconstruct path backwards
    #             path = []
    #             while current in came_from:
    #                 path.append(current)
    #                 current = came_from[current]
    #             # Reverse path to go Start -> Goal
    #             return path[::-1]

    #         # 3. Explore Neighbors
    #         # Using existing _get_neighbors to ensure we stay in bounds
    #         for neighbor in self._get_neighbors(*current):
    #             nx, ny = neighbor
    #             cell_type = self.grid[ny][nx]['type']

    #             # 4. Determine Cost (The crucial logic)
    #             # Base cost for a move
    #             move_cost = 1
                
    #             # Traffic lights are "expensive" due to delay
    #             if cell_type == 'traffic_light':
    #                 move_cost = 5 
                
    #             # Pits and Cows are obstacles
    #             # Cows reset us (infinite loop risk), Pits kill us. Avoid both.
    #             elif cell_type == 'pit' or cell_type == 'cow':
    #                 continue 

    #             # Calculate tentative g_score
    #             tentative_g = g_score[current] + move_cost

    #             # If this path to neighbor is better than any previous one, record it
    #             if neighbor not in g_score or tentative_g < g_score[neighbor]:
    #                 came_from[neighbor] = current
    #                 g_score[neighbor] = tentative_g
                    
    #                 # Heuristic: Manhattan Distance ( |x1-x2| + |y1-y2| )
    #                 h = abs(nx - goal[0]) + abs(ny - goal[1])
    #                 f = tentative_g + h
                    
    #                 f_score[neighbor] = f
                    
    #                 if neighbor not in open_set_hash:
    #                     heapq.heappush(open_set, (f, neighbor))
    #                     open_set_hash.add(neighbor)
    #     # Example: Return None if not implemented
    #     self.message = "Path Not Found - A* not implemented yet"
    #     return None
    def find_path_astar(self):
        start = tuple(self.agent_pos)
        goal = self.goal_pos
        
        # Priority Queue: (f_score, x, y)
        # We store coordinates directly in the tuple for easier unpacking
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        
        # Heuristic: Manhattan distance
        start_h = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        f_score = {start: start_h}
        
        open_set_hash = {start}

        while open_set:
            current_f, current = heapq.heappop(open_set)
            open_set_hash.remove(current)

            # --- GOAL REACHED ---
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                # We do not include the start node, so the agent moves to the 
                # first step immediately.
                return path[::-1]

            # --- EXPLORE NEIGHBORS ---
            for neighbor in self._get_neighbors(*current):
                nx, ny = neighbor
                cell_data = self.grid[ny][nx]
                cell_type = cell_data['type']

                # --- 1. HANDLE HAZARDS (Pits & Cows) ---
                # Strictly avoid Pits (Death) and Cows (Reset)
                if cell_type == 'pit' or cell_type == 'cow':
                    continue

                # --- 2. CALCULATE COST (Terrain + Traffic) ---
                # Requirement: Handle environmental costs
                terrain_cost = cell_data['weight'] # Use the generated weight
                
                extra_cost = 0
                if cell_type == 'traffic_light':
                    extra_cost = 5 # Penalty for waiting
                
                move_cost = terrain_cost + extra_cost
                
                tentative_g = g_score[current] + move_cost

                # --- 3. CHECK IF PATH IS BETTER ---
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    h = abs(nx - goal[0]) + abs(ny - goal[1])
                    f = tentative_g + h
                    
                    f_score[neighbor] = f
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f, neighbor))
                        open_set_hash.add(neighbor)

        self.message = "Path Not Found - Goal unreachable!"
        return None

    def execute_path(self, path):
        """Execute a computed path step by step"""
        if path is None:
            return

        for x, y in path:
            self.move_agent(x, y)
            if self.game_over or self.game_won:
                break

# Pygame rendering
class GameRenderer:
    def __init__(self, world):
        self.world = world
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Bangalore Wumpus World - AI CODEFIX 2025")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def draw_grid(self):
        """Draw the grid lines"""
        for x in range(0, WINDOW_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (x, 0), (x, WINDOW_HEIGHT - 100), 2)
        for y in range(0, WINDOW_HEIGHT - 100, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (0, y), (WINDOW_WIDTH, y), 2)

    def draw_cell_contents(self):
        """Draw contents of each cell"""
        for y in range(GRID_ROWS):
            for x in range(GRID_COLS):
                cell = self.world.grid[y][x]
                px = x * CELL_SIZE
                py = y * CELL_SIZE

                # Draw cell type
                if cell['type'] == 'traffic_light':
                    pygame.draw.circle(self.screen, RED, (px + CELL_SIZE//2, py + CELL_SIZE//2), 20)
                    text = self.small_font.render("SIGNAL", True, WHITE)
                    self.screen.blit(text, (px + 15, py + 55))

                elif cell['type'] == 'cow':
                    pygame.draw.rect(self.screen, BROWN, (px + 20, py + 20, 40, 40))
                    text = self.small_font.render("COW", True, WHITE)
                    self.screen.blit(text, (px + 25, py + 30))

                elif cell['type'] == 'pit':
                    pygame.draw.circle(self.screen, BLACK, (px + CELL_SIZE//2, py + CELL_SIZE//2), 25)
                    text = self.small_font.render("PIT", True, WHITE)
                    self.screen.blit(text, (px + 28, py + 30))

                elif cell['type'] == 'goal':
                    pygame.draw.rect(self.screen, GREEN, (px + 15, py + 15, 50, 50))
                    text = self.small_font.render("GOAL", True, BLACK)
                    self.screen.blit(text, (px + 20, py + 30))

                # Draw percepts (small indicators)
                percept_y_offset = 10
                if 'breeze' in cell['percepts']:
                    text = self.small_font.render("~", True, BLUE)
                    self.screen.blit(text, (px + 5, py + percept_y_offset))
                    percept_y_offset += 15

                if 'moo' in cell['percepts']:
                    text = self.small_font.render("M", True, BROWN)
                    self.screen.blit(text, (px + 5, py + percept_y_offset))
                    percept_y_offset += 15

                if 'light' in cell['percepts']:
                    text = self.small_font.render("L", True, ORANGE)
                    self.screen.blit(text, (px + 5, py + percept_y_offset))

    def draw_agent(self):
        """Draw the agent"""
        x, y = self.world.agent_pos
        px = x * CELL_SIZE + CELL_SIZE // 2
        py = y * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(self.screen, YELLOW, (px, py), 15)
        pygame.draw.circle(self.screen, BLACK, (px, py), 15, 2)

        # Draw eyes
        pygame.draw.circle(self.screen, BLACK, (px - 5, py - 3), 3)
        pygame.draw.circle(self.screen, BLACK, (px + 5, py - 3), 3)

    def draw_info(self):
        """Draw info panel at bottom"""
        info_y = WINDOW_HEIGHT - 100
        pygame.draw.rect(self.screen, GRAY, (0, info_y, WINDOW_WIDTH, 100))

        # Current position
        pos_text = self.font.render(f"Position: {self.world.agent_pos}", True, BLACK)
        self.screen.blit(pos_text, (10, info_y + 10))

        # Percepts
        percepts = self.world.get_current_percepts()
        percept_text = self.font.render(f"Percepts: {', '.join(percepts) if percepts else 'None'}", True, BLACK)
        self.screen.blit(percept_text, (10, info_y + 35))

        # Message
        msg_text = self.font.render(self.world.message, True, RED if self.world.game_over else GREEN)
        self.screen.blit(msg_text, (10, info_y + 60))

    def render(self):
        """Main render function"""
        self.screen.fill(WHITE)
        self.draw_grid()
        self.draw_cell_contents()
        self.draw_agent()
        self.draw_info()
        pygame.display.flip()
        self.clock.tick(FPS)

# Main game loop
def main():
    config = load_config()
    world = BangaloreWumpusWorld(config)
    renderer = GameRenderer(world)

    print("=== Bangalore Wumpus World ===")
    print(f"Team ID: {config['team_id']}")
    print(f"Agent Start: {world.agent_start}")
    print(f"Goal Position: {world.goal_pos}")
    print("\nControls:")
    print("- Arrow keys: Manual movement")
    print("- SPACE: Execute A* pathfinding")
    print("- R: Reset world")
    print("- ESC: Quit")

    running = True
    
    # Variables for animation
    path_to_execute = deque()  # Store the path steps here
    step_delay = 0             # Timer to slow down movement

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    # Reset world and clear any pending path
                    world = BangaloreWumpusWorld(config)
                    renderer.world = world
                    path_to_execute.clear()

                # --- MANUAL CONTROLS ---
                # Only allow manual move if AI isn't currently walking
                elif not path_to_execute: 
                    if event.key == pygame.K_UP:
                        world.move_agent(world.agent_pos[0], world.agent_pos[1] - 1)
                    elif event.key == pygame.K_DOWN:
                        world.move_agent(world.agent_pos[0], world.agent_pos[1] + 1)
                    elif event.key == pygame.K_LEFT:
                        world.move_agent(world.agent_pos[0] - 1, world.agent_pos[1])
                    elif event.key == pygame.K_RIGHT:
                        world.move_agent(world.agent_pos[0] + 1, world.agent_pos[1])

                # --- A* TRIGGER ---
                if event.key == pygame.K_SPACE:
                    print("\n=== Executing A* Pathfinding ===")
                    path = world.find_path_astar()
                    if path:
                        print(f"Path found: {path}")
                        # Instead of moving instantly, add to queue
                        path_to_execute = deque(path)
                    else:
                        print("Path not found")

        # --- ANIMATION LOGIC ---
        # If there are steps in the queue, execute them one by one
        if path_to_execute and not world.game_over and not world.game_won:
            step_delay += 1
            # Adjust '10' to make it faster or slower
            if step_delay > 10:
                next_pos = path_to_execute.popleft()
                world.move_agent(next_pos[0], next_pos[1])
                step_delay = 0

        renderer.render()

    pygame.quit()

if __name__ == "__main__":
    main()
