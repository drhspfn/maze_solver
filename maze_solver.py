import random
import pygame
import sys
import heapq
from typing import List, Tuple, Dict, Optional
from scipy.stats import truncnorm

# Константы
GENERATION_MODE = "string"              # 'random', 'recursive_backtracking' or 'string'
FIELD_SIZES = (50, 50)
CELL_SIZE = 20
PLAYER_COLOR = (255, 0, 0)
WALL_COLOR = (0, 0, 0)
PATH_COLOR = (0, 255, 0)
EMPTY_COLOR = (255, 255, 255)
EXIT_COLOR = (0, 0, 255)
BUTTON_COLOR = (0, 128, 255)
BUTTON_HOVER_COLOR = (0, 102, 204)
TEXT_COLOR = (255, 255, 255)
PLAYER_MOVE_SPEED = 100  # Milliseconds per cell (higher is slower)
TEST_MAP = """
++++.++++++++++.++++
+.................+
+.+.+.+.+.+.+.+.+.+.+
+.+.+.+.+.+.+.+.+...+
+.................+
++++.+++++.+++++++++
"""
OWN_MAP = """"""


def create_maze_random(rows:int, cols:int) -> List[List[str]]:
    """
    This function generates a random maze represented as a 2D grid. The maze is created using a simple randomized algorithm.
    Each cell in the grid can be either a wall ('+') or an empty space ('.'). The probability of a cell being a wall is 30%.

    Parameters:
    rows (int): The number of rows in the maze grid.
    cols (int): The number of columns in the maze grid.

    Returns:
    List[List[str]]: A 2D list representing the generated maze. Each element in the list is a string ('+' or '.').
    """
    return [['.' if random.random() > 0.3 else '+' for _ in range(cols)] for _ in range(rows)]

def create_maze_recursive(rows:int, cols:int) -> List[List[str]]:
    """
    This function generates a random maze represented as a 2D grid using the Recursive Backtracking algorithm.
    Each cell in the grid can be either a wall ('+') or an empty space ('.'). The probability of a cell being a wall is not used in this algorithm.

    Parameters:
    rows (int): The number of rows in the maze grid. Must be a positive integer.
    cols (int): The number of columns in the maze grid. Must be a positive integer.

    Returns:
    List[List[str]]: A 2D list representing the generated maze. Each element in the list is a string ('+' or '.').
    """
    
    # Initializing a grid with filled walls
    maze = [['+' for _ in range(cols)] for _ in range(rows)]

    def carve_passages_from(row, col):
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if 0 <= new_row < rows and 0 <= new_col < cols and maze[new_row][new_col] == '+':
                maze[new_row][new_col] = '.'  # Paving the way
                maze[row + dr // 2][col + dc // 2] = '.'  # Laying a path between the cells
                carve_passages_from(new_row, new_col)

    # Start generating the maze from a random starting point
    start_row, start_col = random.randrange(1, rows, 2), random.randrange(1, cols, 2)
    maze[start_row][start_col] = '.'
    carve_passages_from(start_row, start_col)

    return maze

def find_random_start(maze: List[List[str]]) -> Tuple[int, int]:
    """
    This function finds a random starting position for the maze generation algorithm.
    It uses a truncated normal distribution to generate a more natural-looking starting point.
    If a suitable starting point cannot be found within a certain number of attempts, it falls back to the old method.

    Parameters:
    maze (List[List[str]]): The maze grid. Each cell contains a character representing the cell type: '.' for empty cells, '+' for walls.

    Returns:
    Tuple[int, int]: The row and column indices of the random starting position.
    """
    rows, cols = len(maze), len(maze[0])
    center_row, center_col = rows // 2, cols // 2
    
    std_dev_row = rows / 4
    std_dev_col = cols / 4
    
    row_dist = truncnorm((0 - center_row) / std_dev_row, (rows - 1 - center_row) / std_dev_row, loc=center_row, scale=std_dev_row)
    col_dist = truncnorm((0 - center_col) / std_dev_col, (cols - 1 - center_col) / std_dev_col, loc=center_col, scale=std_dev_col)
    
    attempts = 0
    max_attempts = rows * cols
    
    while attempts < max_attempts:
        row = int(row_dist.rvs())
        col = int(col_dist.rvs())
        
        if 0 <= row < rows and 0 <= col < cols and maze[row][col] == '.':
            return row, col
        
        attempts += 1
    
    return find_random_start_old(maze)

def find_random_start_old(maze:List[List[str]]) -> Tuple[int, int]:
    """
    This function finds a random starting position in the maze grid.
    It iterates through the grid until it finds an empty cell (represented by '.') and returns its coordinates.

    Parameters:
    maze (List[List[str]]): The maze grid. Each cell contains a character representing the cell type: '.' for empty cells, '+' for walls.

    Returns:
    Tuple[int, int]: The row and column indices of the random starting position.
    """
    
    while True:
        row = random.randint(0, len(maze) - 1)
        col = random.randint(0, len(maze[0]) - 1)
        if maze[row][col] == '.':
            return row, col

def import_maze_from_string(maze_string: str) -> List[List[str]]:
    """
    This function converts a string representation of a maze into a 2D list.
    The input string should contain the maze layout, where each character represents a cell in the maze.
    The maze can have different cell types, such as walls ('+') and empty spaces ('.').
    The function extends each row to have the same length by adding empty spaces ('.') at the end.

    Parameters:
    maze_string (str): A string representing the maze layout. Each character in the string represents a cell.

    Returns:
    List[List[str]]: A 2D list representing the maze. Each element in the list is a string ('+' or '.').
    """
    lines = [line.strip() for line in maze_string.strip().split('\n') if line.strip()]
    max_length = max(len(line) for line in lines)
    
    maze = []
    for line in lines:
        row = list(line)
        row.extend(['.'] * (max_length - len(line)))
        maze.append(row)
    
    return maze

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Calculates the Euclidean distance between two points in a 2D space.

    Parameters:
    a (Tuple[int, int]): The coordinates of the first point (x, y).
    b (Tuple[int, int]): The coordinates of the second point (x, y).

    Returns:
    float: The Euclidean distance between the two points.
    """
    return ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5  # Euclidean distance

def is_valid(maze:List[List[str]], row:int, col:int) -> bool:
    """
    Checks if a given position in the maze is valid.

    Parameters:
    maze (List[List[str]]): The 2D list representing the maze. Each element in the list is a string ('+' or '.').
    row (int): The row index of the position to check.
    col (int): The column index of the position to check.

    Returns:
    bool: True if the position is within the maze boundaries and represents an empty cell ('.'), False otherwise.
    """
    return 0 <= row < len(maze) and 0 <= col < len(maze[0]) and maze[row][col] == '.'

def is_exit(maze:List[List[str]], row:int, col:int) -> bool:
    """
    Checks if a given position in the maze represents an exit.

    Parameters:
    maze (List[List[str]]): The 2D list representing the maze. Each element in the list is a string ('+' or '.').
    row (int): The row index of the position to check.
    col (int): The column index of the position to check.

    Returns:
    bool: True if the position is an exit (top, bottom, left, or right boundary) and represents an empty cell ('.'), False otherwise.
    """
    return (row == 0 or row == len(maze) - 1 or col == 0 or col == len(maze[0]) - 1) and maze[row][col] == '.'

def is_valid_move(maze:List[List[str]], row:int, col:int) -> bool:
    """
    Check if a move is valid in the given maze.

    Parameters:
    maze (List[List[str]]): The 2D list representing the maze. Each element in the list is a string ('+' or '.').
    row (int): The row index of the position to check.
    col (int): The column index of the position to check.

    Returns:
    bool: True if the move is valid (i.e., the position is within the maze boundaries and is an empty cell '.'), False otherwise.
    """
    return 0 <= row < len(maze) and 0 <= col < len(maze[0]) and maze[row][col] == '.'

def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    Calculates the Manhattan distance between two points in a 2D grid.

    The Manhattan distance is the sum of the absolute differences of their coordinates.
    This function is useful for pathfinding algorithms in grid-based environments,
    where movement is restricted to horizontal and vertical directions.

    Parameters:
    a (Tuple[int, int]): The coordinates of the first point (x, y).
    b (Tuple[int, int]): The coordinates of the second point (x, y).

    Returns:
    int: The Manhattan distance between the two points.
    """
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

def find_exits(maze: List[List[str]]) -> List[Tuple[int, int]]:
    """
    This function identifies and returns all the exit points in a given maze represented as a 2D grid.
    An exit point is defined as an empty cell ('.') that is located on the top, bottom, left, or right boundary of the maze.

    Parameters:
    maze (List[List[str]]): A 2D list representing the maze. Each element in the list is a string ('+' or '.').
        The maze is assumed to be rectangular and have at least one row and one column.

    Returns:
    List[Tuple[int, int]]: A list of tuples, where each tuple represents the coordinates (row, column) of an exit point in the maze.
        The coordinates are 0-indexed. If no exit points are found, an empty list is returned.
    """
    
    rows, cols = len(maze), len(maze[0])
    exits = []
    for r in range(rows):
        if maze[r][0] == '.':
            exits.append((r, 0))
        if maze[r][cols-1] == '.':
            exits.append((r, cols-1))
    for c in range(1, cols-1):
        if maze[0][c] == '.':
            exits.append((0, c))
        if maze[rows-1][c] == '.':
            exits.append((rows-1, c))
    return exits

def find_exit(maze: List[List[str]], start_row: int, start_col: int) -> Optional[List[Tuple[int, int]]]:
    """
    This function uses the A* search algorithm to find the shortest path from the given start position to any exit in the maze.
    The maze is represented as a 2D grid, where '.' represents empty cells and '+' represents walls.

    Parameters:
    maze (List[List[str]]): A 2D list representing the maze. Each element in the list is a string ('+' or '.').
    start_row (int): The row index of the starting position in the maze.
    start_col (int): The column index of the starting position in the maze.

    Returns:
    Optional[List[Tuple[int, int]]]: A list of tuples representing the coordinates (row, column) of the shortest path from the start position to any exit.
        If no path to an exit is found, None is returned.
    """
    
    start = (start_row, start_col)
    exits = find_exits(maze)
    
    if not exits:
        print("No exits found in the maze!")
        return None
    
    def heuristic(pos: Tuple[int, int]) -> int:
        return min(manhattan_distance(pos, exit_pos) for exit_pos in exits)

    open_set = [(0, start)]
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    g_score: Dict[Tuple[int, int], int] = {start: 0}
    f_score: Dict[Tuple[int, int], int] = {start: heuristic(start)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current in exits and current != start:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dr, current[1] + dc)
            
            if not (0 <= neighbor[0] < len(maze) and 0 <= neighbor[1] < len(maze[0])) or maze[neighbor[0]][neighbor[1]] != '.':
                continue
            
            tentative_g_score = g_score[current] + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    print("No path to exit found! Regenerating maze...")
    return None

def regenerate_maze() -> List[List[str]]:
    """
    This function generates a new maze based on the specified generation mode.

    Parameters:
    None

    Returns:
    List[List[str]]: A 2D list representing the generated maze. Each element in the list is a string,
    where '+' represents a wall and '.' represents an empty cell.
    """
    
    
    if GENERATION_MODE == 'random':
        return create_maze_random(FIELD_SIZES[0], FIELD_SIZES[1])
    elif GENERATION_MODE == 'recursive_backtracking':
        return create_maze_recursive(FIELD_SIZES[0], FIELD_SIZES[1])
    elif GENERATION_MODE == 'string':
        return import_maze_from_string(OWN_MAP if OWN_MAP else TEST_MAP)

def draw_maze_player(screen:pygame.Surface, maze:List[List[str]], player_pos: Tuple[int, int]) -> None:
    """
    Draws the maze on the screen for the player mode.

    Parameters:
    screen (pygame.Surface): The surface on which the maze will be drawn.
    maze (List[List[str]]): A 2D list representing the maze. Each element in the list is a string,
        where '+' represents a wall and '.' represents an empty cell.
    player_pos (Tuple[int, int]): The row and column indices of the player's position in the maze.

    Returns:
    None
    """
    
    screen.fill(EMPTY_COLOR)
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            if maze[row][col] == '+':
                pygame.draw.rect(screen, WALL_COLOR, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            elif is_exit(maze, row, col) and (row, col) != player_pos:
                pygame.draw.rect(screen, EXIT_COLOR, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.draw.rect(screen, PLAYER_COLOR, 
                     (player_pos[1] * CELL_SIZE, 
                      player_pos[0] * CELL_SIZE, 
                      CELL_SIZE, 
                      CELL_SIZE))

def draw_maze_auto(screen:pygame.Surface, maze:List[List[str]], player_pos:Tuple[int, int], path:List[Tuple[int, int]]) -> None:
    """
    Draws the maze on the screen using the provided parameters.

    Parameters:
    screen (pygame.Surface): The surface on which the maze will be drawn.
    maze (List[List[str]]): A 2D list representing the maze. Each element in the list is a string ('+' or '.').
    player_pos (Tuple[int, int]): The row and column indices of the player's position in the maze.
    path (List[Tuple[int, int]]): A list of tuples representing the coordinates (row, column) of the shortest path from the start position to any exit.

    Returns:
    None
    """
    
    screen.fill(EMPTY_COLOR)
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            if maze[row][col] == '+':
                pygame.draw.rect(screen, WALL_COLOR, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            elif is_exit(maze, row, col) and (row, col) != player_pos:
                pygame.draw.rect(screen, EXIT_COLOR, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    if path:
        for row, col in path:
            pygame.draw.rect(screen, PATH_COLOR, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.draw.circle(screen, PLAYER_COLOR, 
                       (player_pos[1] * CELL_SIZE + CELL_SIZE // 2, 
                        player_pos[0] * CELL_SIZE + CELL_SIZE // 2), 
                       CELL_SIZE // 3)

def main_auto() -> None:
    """
    This function is the main game loop for the automatic mode. It generates a maze,
    finds a path to an exit, and simulates a player moving through the maze.

    Parameters:
    None

    Returns:
    None
    """
    maze = regenerate_maze()
    rows = len(maze)
    cols = len(maze[0])
    start_row, start_col = find_random_start(maze)
    path = find_exit(maze, start_row, start_col)

    pygame.init()
    screen = pygame.display.set_mode((cols * CELL_SIZE, rows * CELL_SIZE))
    pygame.display.set_caption("Maze Solver")

    clock = pygame.time.Clock()
    player_pos = [start_row, start_col]
    path_index = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:                         # R key processing
                    maze = regenerate_maze()
                    start_row, start_col = find_random_start(maze)
                    player_pos = [start_row, start_col]
                    path = find_exit(maze, start_row, start_col)
                    if path is None:
                        maze = regenerate_maze()
                        start_row, start_col = find_random_start(maze)
                        continue
                    path_index = 0
                    continue # Skip updating the player's position so you don't get stuck in the old maze
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    main_menu()
                    
                    
        # Движение по найденному пути
        if path and path_index < len(path):
            player_pos = list(path[path_index])
            path_index += 1

        # Проверка, достиг ли игрок выхода
        if path and path_index == len(path):
            maze = regenerate_maze()
            start_row, start_col = find_random_start(maze)
            player_pos = [start_row, start_col]
            path = find_exit(maze, start_row, start_col)
            path_index = 0

        draw_maze_auto(screen, maze, player_pos, path[:path_index] if path else None)
        pygame.display.flip()
        clock.tick(5)

def main_player() -> None:
    """
    This function is the main game loop for the player mode. It generates a maze,
    handles player movement, and checks if the player has reached the exit.

    Parameters:
    None

    Returns:
    None
    """
    
    maze = regenerate_maze()
    rows = len(maze)
    cols = len(maze[0])
    start_row, start_col = find_random_start(maze)
    player_pos = [start_row, start_col]

    pygame.init()
    screen = pygame.display.set_mode((cols * CELL_SIZE, rows * CELL_SIZE))
    pygame.display.set_caption("Maze Explorer")

    clock = pygame.time.Clock()
    last_move_time = pygame.time.get_ticks()
    
    while True:
        current_time = pygame.time.get_ticks()
        dt = current_time - last_move_time
        keys = pygame.key.get_pressed()  # Getting the status of all keys

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # R key processing
                    maze = regenerate_maze()
                    start_row, start_col = find_random_start(maze)
                    player_pos = [start_row, start_col]
                    continue  # Skip updating the player's position so you don't get stuck in the old maze
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    main_menu()
                    
        # Handling player movement
        move_x, move_y = 0, 0
        if keys[pygame.K_LEFT]:
            move_x = -1
        if keys[pygame.K_RIGHT]:
            move_x = 1
        if keys[pygame.K_UP]:
            move_y = -1
        if keys[pygame.K_DOWN]:
            move_y = 1
        
        if move_x != 0 or move_y != 0:
            if dt >= PLAYER_MOVE_SPEED:
                new_pos = [player_pos[0] + move_y, player_pos[1] + move_x]
                if is_valid_move(maze, new_pos[0], new_pos[1]):
                    player_pos = new_pos
                last_move_time = current_time

        # Verify whether the player has reached the exit
        if is_exit(maze, player_pos[0], player_pos[1]):
            maze = regenerate_maze()
            start_row, start_col = find_random_start(maze)
            player_pos = [start_row, start_col]
            last_move_time = pygame.time.get_ticks()

        draw_maze_player(screen, maze, player_pos)
        pygame.display.flip()
        clock.tick(30)  # Screen refresh rate

def draw_text(screen:pygame.Surface, text:str, font:pygame.font.Font, color:pygame.Color, x:int, y:int) -> None:
    """
    Draws the given text onto the provided screen using the specified font, color, and position.

    Parameters:
    screen (pygame.Surface): The surface on which the text will be drawn.
    text (str): The text to be drawn.
    font (pygame.font.Font): The font to be used for drawing the text.
    color (pygame.Color): The color to be used for drawing the text.
    x (int): The x-coordinate of the top-left corner of the text's bounding rectangle.
    y (int): The y-coordinate of the top-left corner of the text's bounding rectangle.

    Returns:
    None
    """
    
    textobj = font.render(text, True, color)
    textrect = textobj.get_rect()
    textrect.center = (x, y)
    screen.blit(textobj, textrect)
    
def draw_menu(screen) -> None:
    """
    Draws the main menu screen with buttons for playing as a player or in automatic mode.

    Parameters:
    screen (pygame.Surface): The surface on which the menu will be drawn.

    Returns:
    None
    """
    
    screen.fill((0, 0, 0))
    font = pygame.font.SysFont(None, 55)

    draw_text(screen, 'Maze Game Menu', font, TEXT_COLOR, screen.get_width() // 2, 100)
    
    button_width, button_height = 300, 60
    button_x, button_y = screen.get_width() // 2 - button_width // 2, 200
    
    pygame.draw.rect(screen, BUTTON_COLOR, (button_x, button_y, button_width, button_height))
    draw_text(screen, 'Play as Player', font, TEXT_COLOR, screen.get_width() // 2, button_y + button_height // 2)
    
    button_y += 80
    pygame.draw.rect(screen, BUTTON_COLOR, (button_x, button_y, button_width, button_height))
    draw_text(screen, 'Play Auto', font, TEXT_COLOR, screen.get_width() // 2, button_y + button_height // 2)
    
    pygame.display.flip()

def main_menu() -> None:
    """
    This function is the main menu of the maze game. It initializes the pygame library, sets up the game window,
    and handles user input to navigate between different game modes.

    Parameters:
    None

    Returns:
    None
    """
    
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Maze Game Menu")
    clock = pygame.time.Clock()

    while True:
        draw_menu(screen)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                # Check if Play as Player button was clicked
                if (300 <= mouse_x <= 500 and 200 <= mouse_y <= 260):
                    pygame.quit()
                    main_player()
                # Check if Play Auto button was clicked
                elif (300 <= mouse_x <= 500 and 280 <= mouse_y <= 340):
                    pygame.quit()
                    main_auto()

        clock.tick(30)


if __name__ == "__main__":
    main_menu()