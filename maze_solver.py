import random
import pygame
import heapq
from typing import List, Tuple, Dict, Optional
from scipy.stats import truncnorm
from PyQt5.QtWidgets import QMainWindow
from settings import GameSettings

settings = GameSettings()


def create_maze_random(rows: int, cols: int) -> List[List[str]]:
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


def create_maze_spiral(rows: int, cols: int) -> List[List[str]]:
    """
    Creates a maze using a spiral pattern.

    Parameters:
    rows (int): The number of rows in the maze.
    cols (int): The number of columns in the maze.

    Returns:
    List[List[str]]: A 2D list representing the maze. Each element in the list is a string ('+' or '.').
    """
    
    
    if rows != cols:
        raise ValueError("This function only works for square mazes")

    maze = [['+' for _ in range(cols)] for _ in range(rows)]

    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def is_empty(r, c):
        return maze[r][c] == '.'

    def carve_spiral(r, c, dr, dc, steps):
        for _ in range(steps):
            if in_bounds(r, c) and not is_empty(r, c):
                maze[r][c] = '.'
            r += dr
            c += dc

    def spiral_path(r, c, dr, dc, length):
        for _ in range(length):
            carve_spiral(r, c, dr, dc, 1)
            r += dr
            c += dc
        return r, c

    # Start from the center
    r, c = rows // 2, cols // 2
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    random.shuffle(directions)  # Randomize the initial direction

    step_size = 1
    while step_size < rows:
        for dr, dc in directions:
            # Randomize step size slightly to add variability
            r, c = spiral_path(r, c, dr, dc, step_size + random.randint(0, 1))
        directions = directions[1:] + directions[:1]  # Rotate directions
        step_size += 1

    # Create exits on all sides
    for i in range(rows):
        maze[i][0] = '.'  # Left exit
        maze[i][rows - 1] = '.'  # Right exit
    for j in range(cols):
        maze[0][j] = '.'  # Top exit
        maze[rows - 1][j] = '.'  # Bottom exit

    return maze


def create_maze_prims(rows: int, cols: int) -> List[List[str]]:
    """
    This function generates a maze using Prim's algorithm. The maze is represented as a 2D list of characters,
    where '+' represents walls and '.' represents empty cells.

    Parameters:
    rows (int): The number of rows in the maze. Must be an odd number greater than 1.
    cols (int): The number of columns in the maze. Must be an odd number greater than 1.

    Returns:
    List[List[str]]: A 2D list representing the generated maze. The maze is guaranteed to have a single path from the top-left corner to the bottom-right corner.
    """
    
    maze = [['+' for _ in range(cols)] for _ in range(rows)]

    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def add_walls(r, c):
        for dr, dc in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and maze[nr][nc] == '+':
                walls.append((nr, nc, r + dr // 2, c + dc // 2))

    # Initialize the maze
    start_row, start_col = random.randrange(
        1, rows, 2), random.randrange(1, cols, 2)
    maze[start_row][start_col] = '.'
    walls = []
    add_walls(start_row, start_col)

    while walls:
        nr, nc, pr, pc = random.choice(walls)
        if maze[nr][nc] == '+':
            maze[nr][nc] = '.'
            maze[pr][pc] = '.'
            add_walls(nr, nc)
        walls.remove((nr, nc, pr, pc))

    return maze


def create_maze_recursive(rows: int, cols: int) -> List[List[str]]:
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
                # Laying a path between the cells
                maze[row + dr // 2][col + dc // 2] = '.'
                carve_passages_from(new_row, new_col)

    # Start generating the maze from a random starting point on the border
    start_row, start_col = random.choice(
        [0, rows-1]), random.randrange(1, cols, 2)
    if start_row == 0:
        start_col = random.randrange(1, cols, 2)
    else:
        start_col = random.randrange(1, cols-1, 2)

    maze[start_row][start_col] = '.'
    carve_passages_from(start_row, start_col)

    # Ensure exits on the borders
    for i in range(rows):
        if maze[i][0] == '.':
            maze[i][0] = '.'
        if maze[i][cols-1] == '.':
            maze[i][cols-1] = '.'

    for j in range(cols):
        if maze[0][j] == '.':
            maze[0][j] = '.'
        if maze[rows-1][j] == '.':
            maze[rows-1][j] = '.'

    return maze


def create_maze_rooms_and_corridors(rows: int, cols: int) -> List[List[str]]:
    """
    This function generates a maze using the rooms and corridors technique.
    The maze is represented as a 2D grid of characters, where '+' represents walls and '.' represents empty cells.
    The function creates a specified number of rooms, connects them by corridors, and ensures that there is at least one exit.

    Parameters:
    rows (int): The number of rows in the maze grid.
    cols (int): The number of columns in the maze grid.

    Returns:
    List[List[str]]: A 2D list representing the generated maze. Each element in the list is a string ('+' or '.').
    """
    
    maze = [['+' for _ in range(cols)] for _ in range(rows)]
    rooms = []

    def create_room():
        room_height = random.randint(3, 7)
        room_width = random.randint(3, 7)
        start_row = random.randint(1, rows - room_height - 1)
        start_col = random.randint(1, cols - room_width - 1)

        for i in range(start_row, start_row + room_height):
            for j in range(start_col, start_col + room_width):
                maze[i][j] = '.'

        return (start_row + room_height // 2, start_col + room_width // 2)

    # Creating rooms
    for _ in range(random.randint(5, 10)):
        rooms.append(create_room())

    # Connecting rooms by corridors
    def connect_rooms(start, end):
        x, y = start
        while (x, y) != end:
            maze[x][y] = '.'
            if x < end[0]:
                x += 1
            elif x > end[0]:
                x -= 1
            elif y < end[1]:
                y += 1
            else:
                y -= 1

    for i in range(len(rooms) - 1):
        connect_rooms(rooms[i], rooms[i+1])

    # Обеспечение выхода
    exits = [(0, random.randint(0, cols-1)), (rows-1, random.randint(0, cols-1)),
             (random.randint(0, rows-1), 0), (random.randint(0, rows-1), cols-1)]

    for ex, ey in exits:
        maze[ex][ey] = '.'
        connect_rooms((ex, ey), random.choice(rooms))

    return maze


def create_maze_growing_tree(rows: int, cols: int) -> List[List[str]]:
    """
    Generates a maze using the growing tree algorithm.

    Parameters:
    rows (int): The number of rows in the maze.
    cols (int): The number of columns in the maze.

    Returns:
    maze (List[List[str]]): A 2D list representing the generated maze. Each element in the list is a string ('+' or '.').

    The function starts by creating a single cell in the center of the maze and then iteratively adds new cells to the maze, connecting them to existing cells using a random walk. The function ensures that the maze has a single path from the starting cell to any exit cell.
    The function also includes a probability of creating a new branch when adding a new cell to the maze. This can result in more complex and interesting mazes.
    The function returns the generated maze as a 2D list of characters.
    """
    maze = [['+' for _ in range(cols)] for _ in range(rows)]

    def is_valid(x, y):
        return 0 <= x < rows and 0 <= y < cols

    def carve_path(x, y):
        maze[x][y] = '.'
        stack = [(x, y)]

        while stack:
            current = stack[-1]
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = current[0] + dx*2, current[1] + dy*2
                if is_valid(nx, ny) and maze[nx][ny] == '+':
                    maze[nx][ny] = '.'
                    maze[current[0] + dx][current[1] + dy] = '.'
                    stack.append((nx, ny))

                    # Probability of creating a new branch
                    if random.random() < 0.3:
                        stack.append((nx, ny))
                    break
            else:
                stack.pop()

    start_x, start_y = random.randint(0, rows-1), random.randint(0, cols-1)
    carve_path(start_x, start_y)

    # Ensuring output
    exits = [(0, random.randint(0, cols-1)), (rows-1, random.randint(0, cols-1)),
             (random.randint(0, rows-1), 0), (random.randint(0, rows-1), cols-1)]

    for ex, ey in exits:
        if maze[ex][ey] == '+':
            maze[ex][ey] = '.'
            # Connecting an exit to the nearest aisle
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = ex + dx, ey + dy
                if is_valid(nx, ny) and maze[nx][ny] == '.':
                    break
            else:
                carve_path(ex, ey)

    return maze


def create_maze_from_string(maze_string: str) -> List[List[str]]:
    """
    This function converts a string representation of a maze into a 2D list.
    The input string should contain the maze layout, where each character represents a cell in the maze.
    The maze can have different cell types, such as walls ('+') and empty spaces ('.').
    The function extends each row to have the same length by adding empty spaces ('.') at the end.
    If any character other than '+' or '.' is found, the function returns None.

    Parameters:
    maze_string (str): A string representing the maze layout. Each character in the string represents a cell.

    Returns:
    List[List[str]]: A 2D list representing the maze. Each element in the list is a string ('+' or '.').
                     Returns None if any invalid character is found in the maze_string.
    """
    valid_chars = {'.', '+'}

    lines = [line.strip()
             for line in maze_string.strip().split('\n') if line.strip()]

    # Check for invalid characters
    for line in lines:
        if any(char not in valid_chars for char in line):
            return None

    max_length = max(len(line) for line in lines)

    maze = []
    for line in lines:
        row = list(line)
        row.extend(['.'] * (max_length - len(line)))
        maze.append(row)

    return maze


def create_maze_fractal(rows: int, cols: int) -> List[List[str]]:
    """
    Generates a maze using the fractal algorithm.

    Parameters:
    rows (int): The number of rows in the maze.
    cols (int): The number of columns in the maze.

    Returns:
    List[List[str]]: A 2D list representing the generated maze. Each element in the list is a string,
    where '+' represents a wall and '.' represents an empty cell.
    """
    
    
    maze = [['.' for _ in range(cols)] for _ in range(rows)]

    def divide_chamber(x1, y1, x2, y2, orientation):
        if x2 - x1 < 2 or y2 - y1 < 2:
            return

        if orientation == 'horizontal':
            wall_y = random.randint(y1 + 1, y2 - 1)
            passage = random.randint(x1, x2)
            for x in range(x1, x2 + 1):
                if x != passage:
                    maze[wall_y][x] = '+'

            divide_chamber(x1, y1, x2, wall_y - 1, 'vertical')
            divide_chamber(x1, wall_y + 1, x2, y2, 'vertical')
        else:
            wall_x = random.randint(x1 + 1, x2 - 1)
            passage = random.randint(y1, y2)
            for y in range(y1, y2 + 1):
                if y != passage:
                    maze[y][wall_x] = '+'

            divide_chamber(x1, y1, wall_x - 1, y2, 'horizontal')
            divide_chamber(wall_x + 1, y1, x2, y2, 'horizontal')

    # We start with vertical separation
    divide_chamber(1, 1, rows - 2, cols - 2, 'vertical')

    # Adding perimeter walls
    for i in range(rows):
        maze[i][0] = maze[i][-1] = '+'
    for j in range(cols):
        maze[0][j] = maze[-1][j] = '+'

    # Create input and output
    entrance = (random.randint(1, rows-2), 0)
    exit = (random.randint(1, rows-2), cols-1)
    maze[entrance[0]][entrance[1]] = '.'
    maze[exit[0]][exit[1]] = '.'

    # Making sure there's a path from the entrance to the exit.
    def find_path(start, end):
        queue = [start]
        visited = set([start])
        parent = {start: None}

        while queue:
            current = queue.pop(0)
            if current == end:
                path = []
                while current:
                    path.append(current)
                    current = parent[current]
                return path[::-1]

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_x, next_y = current[0] + dx, current[1] + dy
                if 0 <= next_x < rows and 0 <= next_y < cols and maze[next_x][next_y] == '.' and (next_x, next_y) not in visited:
                    queue.append((next_x, next_y))
                    visited.add((next_x, next_y))
                    parent[(next_x, next_y)] = current

        return None

    path = find_path(entrance, exit)
    if not path:
        # If there is no path, create one
        current = entrance
        while current != exit:
            x, y = current
            if x < exit[0]:
                x += 1
            elif x > exit[0]:
                x -= 1
            elif y < exit[1]:
                y += 1
            else:
                y -= 1
            maze[x][y] = '.'
            current = (x, y)

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

    row_dist = truncnorm((0 - center_row) / std_dev_row, (rows - 1 -
                         center_row) / std_dev_row, loc=center_row, scale=std_dev_row)
    col_dist = truncnorm((0 - center_col) / std_dev_col, (cols - 1 -
                         center_col) / std_dev_col, loc=center_col, scale=std_dev_col)

    attempts = 0
    max_attempts = rows * cols

    while attempts < max_attempts:
        row = int(row_dist.rvs())
        col = int(col_dist.rvs())

        if 0 <= row < rows and 0 <= col < cols and maze[row][col] == '.':
            return row, col

        attempts += 1

    return find_random_start_old(maze)


def find_random_start_old(maze: List[List[str]]) -> Tuple[int, int]:
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


def is_valid(maze: List[List[str]], row: int, col: int) -> bool:
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


def is_exit(maze: List[List[str]], row: int, col: int) -> bool:
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


def is_valid_move(maze: List[List[str]], row: int, col: int) -> bool:
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

    match settings.GENERATION_MODE:
        case 'random':
            return create_maze_random(settings.FIELD_SIZES[0], settings.FIELD_SIZES[1])
        case 'recursive_backtracking':
            return create_maze_recursive(settings.FIELD_SIZES[0], settings.FIELD_SIZES[1])
        case 'prim':
            return create_maze_prims(settings.FIELD_SIZES[0], settings.FIELD_SIZES[1])
        case 'spiral':
            return create_maze_spiral(settings.FIELD_SIZES[0], settings.FIELD_SIZES[1])
        case 'string':
            return create_maze_from_string(settings.OWN_MAP if settings.OWN_MAP else settings.TEST_MAP)
        case 'growing_tree':
            return create_maze_growing_tree(settings.FIELD_SIZES[0], settings.FIELD_SIZES[1])
        case 'rooms_and_corridors':
            return create_maze_rooms_and_corridors(settings.FIELD_SIZES[0], settings.FIELD_SIZES[1])
        case 'fractal':
            return create_maze_fractal(settings.FIELD_SIZES[0], settings.FIELD_SIZES[1])
        case _:
            raise ValueError(
                f"Unknown generation mode: {settings.GENERATION_MODE}")


def draw_maze_player(screen: pygame.Surface, maze: List[List[str]], player_pos: Tuple[int, int]) -> None:
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

    screen.fill(settings.EMPTY_COLOR)
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            if maze[row][col] == '+':
                pygame.draw.rect(
                    screen, settings.WALL_COLOR, (col * settings.CELL_SIZE, row * settings.CELL_SIZE, settings. CELL_SIZE, settings.CELL_SIZE))
            elif is_exit(maze, row, col) and (row, col) != player_pos:
                pygame.draw.rect(
                    screen, settings.EXIT_COLOR, (col * settings.CELL_SIZE, row * settings.CELL_SIZE, settings.CELL_SIZE, settings.CELL_SIZE))

    pygame.draw.rect(screen, settings.PLAYER_COLOR,
                     (player_pos[1] * settings.CELL_SIZE,
                      player_pos[0] * settings.CELL_SIZE,
                      settings.CELL_SIZE,
                      settings.CELL_SIZE))


def draw_maze_auto(screen: pygame.Surface, maze: List[List[str]], player_pos: Tuple[int, int], path: List[Tuple[int, int]]) -> None:
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

    screen.fill(settings.EMPTY_COLOR)
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            if maze[row][col] == '+':
                pygame.draw.rect(
                    screen, settings.WALL_COLOR, (col * settings.CELL_SIZE, row * settings.CELL_SIZE, settings.CELL_SIZE, settings.CELL_SIZE))
            elif is_exit(maze, row, col) and (row, col) != player_pos:
                pygame.draw.rect(
                    screen, settings.EXIT_COLOR, (col * settings.CELL_SIZE, row * settings.CELL_SIZE, settings.CELL_SIZE, settings.CELL_SIZE))

    if path:
        for row, col in path:
            pygame.draw.rect(screen, settings.PATH_COLOR, (col * settings.CELL_SIZE,
                             row * settings. CELL_SIZE, settings.CELL_SIZE, settings.CELL_SIZE))

    pygame.draw.rect(screen, settings.PLAYER_COLOR,
                     (player_pos[1] * settings.CELL_SIZE,
                      player_pos[0] * settings.CELL_SIZE,
                      settings.CELL_SIZE,
                      settings.CELL_SIZE))


def main_auto(main_menu: QMainWindow) -> None:
    """
    This function is the main game loop for the automatic mode. It generates a maze,
    finds a path to an exit, and simulates a player moving through the maze.

    Parameters:
    main_menu (PyQt5.QtWidgets.QMainWindow): The main menu window object. This parameter is used to show the main menu after the game loop ends.

    Returns:
    None
    """
    maze = regenerate_maze()
    rows = len(maze)
    cols = len(maze[0])
    start_row, start_col = find_random_start(maze)
    path = find_exit(maze, start_row, start_col)

    pygame.init()
    screen = pygame.display.set_mode(
        (cols * settings.CELL_SIZE, rows * settings.CELL_SIZE))
    pygame.display.set_caption("Maze Solver")

    clock = pygame.time.Clock()
    player_pos = [start_row, start_col]
    path_index = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                main_menu.show()
                return
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
                    continue  # Skip updating the player's position so you don't get stuck in the old maze
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    main_menu.show()
                    return

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

        draw_maze_auto(screen, maze, player_pos,
                       path[:path_index] if path else None)
        pygame.display.flip()
        clock.tick(5)


def main_player(main_menu: QMainWindow) -> None:
    """
    This function is the main game loop for the player mode. It generates a maze,
    handles player movement, and checks if the player has reached the exit.

    Parameters:
    main_menu (PyQt5.QtWidgets.QMainWindow): The main menu window object. This parameter is used to show the main menu after the game loop ends.

    Returns:
    None
    """

    maze = regenerate_maze()
    rows = len(maze)
    cols = len(maze[0])
    start_row, start_col = find_random_start(maze)
    player_pos = [start_row, start_col]

    pygame.init()
    screen = pygame.display.set_mode(
        (cols * settings.CELL_SIZE, rows * settings.CELL_SIZE))
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
                main_menu.show()
                return

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # R key processing
                    maze = regenerate_maze()
                    start_row, start_col = find_random_start(maze)
                    player_pos = [start_row, start_col]
                    continue  # Skip updating the player's position so you don't get stuck in the old maze
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    main_menu.show()  # Show the main menu
                    return

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
            if dt >= settings.PLAYER_MOVE_SPEED:
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


if __name__ == "__main__":
    from main_menu import main
    main()
