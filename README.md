# Maze Game

Maze Game is a fun and interactive game featuring two modes: Auto and Player. In Auto mode, the game level is solved automatically by an algorithm. In Player mode, you navigate through the maze yourself. The game also includes a main menu and level refresh functionality.

## Features

-   **Two Game Modes**:

    -   **Auto Mode**: The game automatically solves the maze using an algorithm and displays the solution path.
    -   **Player Mode**: You control the player and navigate through the maze manually using keyboard controls.

-   **Main Menu**:

    -   The game starts with a main menu where you can choose between Auto mode and Player mode.
    -   You can return to the main menu from both Auto and Player modes.

-   **Level Refresh**:
    -   In both Auto and Player modes, press the `R` key to regenerate the maze and start from a new level.

## Controls

-   **Player Mode**:

    -   Arrow keys (`←`, `→`, `↑`, `↓`): Move the player through the maze.
    -   `R` key: Regenerate the maze and restart from a new level.
    -   `ESC` key: Return to the main menu.

-   **Auto Mode**:
    -   The maze is solved automatically; no player input is required.
    -   `R` key: Regenerate the maze and start from a new level.

## Maze Generation Types

This project provides multiple algorithms for generating mazes, each offering a unique approach to maze construction:

1. **Random Maze Generation** (`create_maze_random`):
   - Generates a maze using a simple randomized algorithm where each cell has a 30% chance of being a wall. The result is a maze with a randomized pattern of walls and paths.

2. **Spiral Maze Generation** (`create_maze_spiral`):
   - Creates a spiral-patterned maze. The algorithm works best with square grids and forms a maze that spirals outward from the center, with random variations in the spiral’s direction.

3. **Prim's Algorithm** (`create_maze_prims`):
   - Utilizes Prim's algorithm to generate a maze, ensuring a single path exists from the top-left to the bottom-right corner. This method produces a maze with fewer dead ends and a more predictable layout.

4. **Recursive Backtracking** (`create_maze_recursive`):
   - Implements the Recursive Backtracking algorithm, which carves passages by exploring and backtracking. This method is known for creating mazes with a tree-like structure and intricate, winding paths.

5. **Rooms and Corridors** (`create_maze_rooms_and_corridors`):
   - Constructs a maze by first creating rooms at random positions, then connecting them with corridors. This method is ideal for mazes that need distinct areas connected by pathways.

6. **Growing Tree Algorithm** (`create_maze_growing_tree`):
   - Generates a maze by iteratively growing a tree structure, where cells are added randomly and connected to the existing maze. The algorithm can produce diverse patterns by varying the branching factor.

7. **Fractal Maze Generation** (`create_maze_fractal`):
   - Uses a fractal division algorithm to create a maze by recursively dividing the space and adding walls. This approach results in a maze with a highly structured, grid-like appearance.

Each algorithm can generate a maze of any specified size, providing flexibility in maze design for different use cases.


## Installation
You have two options for installing and running Maze Game:

### Option 1: Clone the Repository
1. **Clone the Repository:**

    ```bash
    git clone https://github.com/drhspfn/maze-solver.git
    cd maze-solver
    ```

2. **Install Dependencies:**
   Ensure you have Python and Pygame installed:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application:**
    ```bash
    python maze_solver.py
    ```
### Option 2: Download the Executable

1. **Download the Executable:**
Go to the [Releases page](https://github.com/drhspfn/maze_solver/releases) on GitHub and download the latest version of the executable file.

2. **Run the Executable:**
Double-click the downloaded `.exe` file to start the game.

## Usage

1. **Customize Maze Settings:**

    - You can modify the rows and cols variables to change the maze's size.
    - The maze is generated randomly each time the application runs, but you can also import a maze by uncommenting the `import_maze_from_string()` function in the `main()` method.

2. **Run the Application:**
    - When you run the application, a maze will be generated, and the A\* algorithm will start searching for the nearest exit.
    - The path will be displayed in green, and the player (starting point) is shown as a red circle.
    - The application will automatically animate the player's movement towards the exit.

## Algorithms and Techniques:

-   **Maze Generation:** A random maze is generated where each cell has a 70% chance of being open and a 30% chance of being a wall.
-   **A Pathfinding:** The A\* algorithm is used to find the shortest path to the nearest exit. This algorithm is efficient and ensures that the path found is the optimal one.
-   **Heuristics:** The algorithm uses the Manhattan distance as a - heuristic to estimate the cost of reaching the nearest exit.
-   **Random Start Point:** The starting point is chosen randomly, favoring positions closer to the center of the maze to increase the likelihood of finding an exit.

## Customization

-   **Maze String Import:** You can import a maze layout from a string by using the `import_maze_from_string()` function, which allows you to define specific scenarios for testing.
-   **Visual Settings:** Adjust the `CELL_SIZE`, `PLAYER_COLOR`, `WALL_COLOR`, `PATH_COLOR`, `EMPTY_COLOR`, and `EXIT_COLOR` constants to customize the appearance of the maze and the player.

## Contributing

If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
