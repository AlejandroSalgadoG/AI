# Search Algorithm Visualization

A Python-based interactive pathfinding visualization tool that demonstrates the Depth-First Search (DFS) algorithm on a customizable grid. Users can create obstacles and watch the algorithm find a path from one player to another in real-time.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Algorithm Details](#algorithm-details)
- [Customization](#customization)

## Overview

This project provides a visual representation of the Depth-First Search pathfinding algorithm. It allows users to interactively design a maze or obstacle course on a grid, then watch as the DFS algorithm navigates from a starting position (Player 1) to a goal position (Player 2).

## Features

- **Interactive Grid**: Click to add or remove obstacles on the grid
- **Real-time Visualization**: Watch the DFS algorithm explore the grid step-by-step
- **Customizable Parameters**: Easily adjust grid size, player positions, and colors
- **Visual Feedback**: Clear color-coded display:
  - Black: Empty space
  - Blue: Obstacles/Walls
  - Yellow: Starting player (Player 1)
  - Red: Goal player (Player 2)
- **Path Exploration**: See every step the algorithm takes to find the goal

## Requirements

- Python 3.x
- `graphics.py` library (John Zelle's graphics library)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Search
   ```

2. **Install the graphics library**:
   ```bash
   # Download graphics.py from:
   # https://mcsp.wartburg.edu/zelle/python/graphics.py
   # Place it in the Dfs/ directory or in your Python path
   ```

3. **Verify installation**:
   ```bash
   python -c "import graphics"
   ```

## Usage

1. **Run the program**:
   ```bash
   cd Dfs
   python Search.py
   ```

2. **Create obstacles**:
   - Click on any grid cell to create an obstacle (blue)
   - Click on an obstacle to remove it (turns black)
   - You cannot place obstacles on player positions

3. **Start the search**:
   - Press any key on the keyboard to start the DFS algorithm
   - The yellow player will start moving towards the red player

4. **Watch the visualization**:
   - The algorithm will explore the grid step-by-step
   - Each move is animated with a 1-second delay
   - Once the goal is reached, "Done!" will be printed in the console

5. **Close the program**:
   - Click anywhere in the window after the search completes

## Project Structure

```
Search/
└── Dfs/
    ├── Algorithms.py    # DFS algorithm implementation
    ├── Board.py         # Grid board management and visualization
    ├── Parameters.py    # Configuration parameters
    ├── Player.py        # Player class definition
    └── Search.py        # Main entry point
```

### File Descriptions

#### `Search.py`
The main entry point that:
- Creates the graphics window
- Initializes players and the board
- Handles user input for obstacle creation
- Starts the DFS algorithm execution

#### `Algorithms.py`
Contains the core DFS algorithm implementation:
- `executePlayer()`: Entry point for starting the search
- `dfs()`: Recursive DFS implementation
- `goal()`: Goal state checker
- `successors()`: Generates valid neighboring states
- Uses a closed set to avoid revisiting states

#### `Board.py`
Manages the grid board and visualization:
- Board representation (2D array)
- Obstacle creation/removal
- Player drawing and movement
- Grid rendering
- Board state values:
  - `0`: Empty space
  - `1`: Obstacle
  - `2`: Player 1 (start)
  - `3`: Player 2 (goal)

#### `Player.py`
Simple player class with:
- `id`: Unique identifier
- `color`: Display color
- `pos`: Current position (x, y)

#### `Parameters.py`
Configuration file containing:
- Grid dimensions (`numCol`, `numRow`)
- Cell size (`rectSize`)
- Window dimensions
- Initial player positions and colors

## How It Works

1. **Initialization**:
   - A grid window is created with specified dimensions
   - Two players are placed at their starting positions
   - The board is rendered with a white grid

2. **Setup Phase**:
   - User clicks cells to create obstacles
   - Each click toggles the cell between empty and obstacle
   - Pressing any key finalizes the board setup

3. **Search Phase**:
   - DFS algorithm starts from Player 1's position
   - The algorithm explores cells in this order: Right → Down → Left → Up
   - Visited cells are tracked in a closed set to prevent cycles
   - Each step is visualized with a 1-second delay
   - The search continues until Player 2 is reached

4. **Completion**:
   - When the goal is found, "Done!" is printed
   - The window waits for a mouse click to close

## Configuration

Edit `Parameters.py` to customize the visualization:

```python
# Grid dimensions
numCol = 5      # Number of columns
numRow = 5      # Number of rows
rectSize = 50   # Size of each cell in pixels

# Player 1 (Start) configuration
p1Pos = (1,3)        # Starting position (x, y)
p1Color = "yellow"   # Display color

# Player 2 (Goal) configuration
p2Pos = (3,1)        # Goal position (x, y)
p2Color = "red"      # Display color
```

## Algorithm Details

### Depth-First Search (DFS)

The implementation uses a recursive DFS approach:

**Key Characteristics**:
- **Search Strategy**: Explores as far as possible along each branch before backtracking
- **Memory**: Uses a closed set to track visited nodes
- **Completeness**: Will find a solution if one exists (in finite graphs)
- **Optimality**: Does not guarantee the shortest path

**Successor Generation Order**:
1. Right (x+1, y)
2. Down (x, y+1)
3. Left (x-1, y)
4. Up (x, y-1)

**Visited State Tracking**:
- Uses a global `closeSet` to prevent revisiting states
- States are added to the closed set before exploring successors
- Only unvisited, valid neighbors are considered

### Time Complexity
- **Time**: O(V + E) where V is vertices and E is edges
- **Space**: O(V) for the closed set and recursion stack

## Customization

### Changing Grid Size
Increase the grid for more complex mazes:
```python
numCol = 10
numRow = 10
```

### Adjusting Animation Speed
Modify the delay in `Algorithms.py`:
```python
time.sleep(0.5)  # Faster animation (0.5 seconds)
time.sleep(2)    # Slower animation (2 seconds)
```

### Adding Multiple Players
Extend the `players` list in `Search.py`:
```python
player3 = Player(4, "green", (4,4))
players = [player1, player2, player3]
```

### Changing Colors
Modify colors in `Parameters.py` or use any Tkinter-compatible color:
```python
p1Color = "orange"
p2Color = "purple"
```

### Implementing Other Search Algorithms
You can modify `Algorithms.py` to implement:
- **Breadth-First Search (BFS)**: Use a queue instead of recursion
- **A* Search**: Add heuristic function and priority queue
- **Dijkstra's Algorithm**: Add edge weights and priority queue
- **Greedy Best-First Search**: Use only heuristic for ordering

## Troubleshooting

### Graphics Library Not Found
```bash
# Download graphics.py and place it in the Dfs/ directory
wget https://mcsp.wartburg.edu/zelle/python/graphics.py -P Dfs/
```

### Window Not Responding
- Ensure you're running Python 3.x
- Check that Tkinter is installed (usually comes with Python)
- On macOS, you may need to run with `pythonw` instead of `python`

### Search Not Starting
- Make sure to press a key (not click) to start the search
- Verify that a valid path exists between the two players
