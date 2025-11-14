# Ghostbuster

A Bayesian inference-based game where players use probabilistic reasoning to locate a hidden ghost on a grid.

## Overview

Ghostbuster is an interactive probability puzzle game that demonstrates real-world applications of Bayesian inference. Players use a noisy sensor to gather evidence about a ghost's location, and the system updates probability distributions in real-time using Bayes' rule. The goal is to accurately predict the ghost's position based on accumulated sensor readings.

## Features

- **Probabilistic Inference**: Real-time Bayesian updates based on sensor readings
- **Interactive GUI**: Click-based interface with visual probability distributions
- **Noisy Sensor Model**: Distance-based color sensor with probabilistic accuracy
- **Educational**: Demonstrates practical applications of probability theory and inference

## How It Works

### Game Mechanics

1. **Setup**: A ghost is randomly placed on a 6×6 grid (hidden from the player)
2. **Sensor Readings**: Click any cell to activate the sensor at that position
3. **Color Feedback**: The sensor returns a color (green, yellow, orange, or red) based on Manhattan distance from the ghost
4. **Probability Updates**: The grid displays updated probabilities for each cell after each reading
5. **Final Guess**: Click the "BUST" button when ready, then click where you think the ghost is located

### Sensor Model

The sensor's accuracy depends on the Manhattan distance to the ghost:

| Distance | Green | Yellow | Orange | Red |
|----------|-------|--------|--------|-----|
| 0 (same cell) | 5% | 5% | 5% | **85%** |
| 1 (adjacent) | 5% | 5% | **85%** | 5% |
| 2 (two cells away) | 5% | **85%** | 5% | 5% |
| 3 (three cells away) | **85%** | 5% | 5% | 5% |
| 4+ (far away) | **85%** | 5% | 5% | 5% |

**Key Insight**: The sensor is noisy (85% accurate, 15% error), which makes the game challenging and requires multiple readings for confident predictions.

### Bayesian Inference

The game uses Bayes' rule to update probabilities:

```
P(ghost at position | sensor reading) ∝ P(sensor reading | ghost at position) × P(ghost at position)
```

For each sensor reading:
1. Calculate likelihood of that color for every possible ghost position
2. Multiply by the current probability distribution (prior)
3. Normalize to ensure probabilities sum to 1

This process accumulates evidence over multiple readings to narrow down the ghost's location.

## Installation

### Prerequisites

- Python 3.x
- `graphics.py` library (Zelle's graphics module)

### Install Graphics Library

```bash
# Download graphics.py from John Zelle's website
wget http://mcsp.wartburg.edu/zelle/python/graphics.py

# Or install via pip if available
pip install graphics.py
```

Alternatively, place the `graphics.py` file in the project directory.

### Clone and Run

```bash
git clone <repository-url>
cd Ghostbuster
python Main.py
```

## Usage

### Starting the Game

```bash
python Main.py
```

### Playing

1. **Initial State**: The grid shows equal probabilities (0.03 for each of 36 cells)

2. **Gather Evidence**:
   - Click any cell on the grid to get a sensor reading
   - The cell will be colored based on the sensor output (green/yellow/orange/red)
   - Probability values update across all cells

3. **Strategic Sampling**:
   - Start with cells spread across the grid
   - Focus subsequent readings near high-probability areas
   - Use multiple readings to reduce uncertainty

4. **Make Your Guess**:
   - Click the blue "BUST" button on the right
   - Click the cell where you believe the ghost is located
   - The result (HIT or MISS) will be displayed
   - The ghost's actual location is revealed in the console

### Tips for Success

- **Multiple Readings**: One reading is rarely enough due to sensor noise
- **Cover the Grid**: Sample different areas before focusing on one spot
- **High Probability ≠ Certainty**: Even high probabilities (>0.50) can be wrong
- **Distance Matters**: Remember the sensor model—red is close, green is far
- **Bayesian Thinking**: Each reading provides evidence, not absolute truth

## Project Structure

```
Ghostbuster/
│
├── Main.py          # Entry point and game loop
├── Gui.py           # Graphical user interface
├── Solver.py        # Bayesian inference engine
├── Model.py         # Sensor probability model
├── Parameters.py    # Configuration parameters
└── README.md        # This file
```

### File Descriptions

#### `Main.py`
- Initializes the GUI and game state
- Implements the main game loop
- Handles user interactions and coordinates between modules

#### `Gui.py`
- Manages the graphical window and grid display
- Renders probability distributions
- Handles mouse input
- Displays sensor readings and results

#### `Solver.py`
- Core Bayesian inference logic
- Ghost generation and position management
- Probability distribution calculations
- Sensor simulation (with noise)
- Distance calculations (Manhattan distance)

#### `Model.py`
- Defines the sensor model: `P(color | distance)`
- Maximum sensing distance (4 cells)
- Color-to-index translations

#### `Parameters.py`
- Window size (600×650 pixels)
- Grid size (6×6 cells)
- Button dimensions

## Configuration

Edit `Parameters.py` to customize the game:

```python
size = 600      # Window size in pixels
btnSize = 50    # Button width in pixels
numRow = 6      # Grid dimensions (6×6 = 36 cells)
```

Edit `Model.py` to modify the sensor model:

```python
maxDist = 4     # Maximum distance threshold

# Probability distribution: P(color | distance)
model = {
    0 : [0.05, 0.05, 0.05, 0.85],  # Distance 0: mostly red
    1 : [0.05, 0.05, 0.85, 0.05],  # Distance 1: mostly orange
    # ... etc
}
```

## Mathematical Background

### Manhattan Distance

The sensor uses Manhattan distance (L1 norm):

```
distance = |x₁ - x₂| + |y₁ - y₂|
```

This represents the minimum number of cell moves (horizontal + vertical) between two positions.

### Bayes' Rule Application

For each cell position and sensor reading:

1. **Prior**: `P(ghost at cell)` - current probability before reading
2. **Likelihood**: `P(color | ghost at cell)` - from the sensor model
3. **Posterior**: `P(ghost at cell | color)` - updated probability after reading

The posterior becomes the new prior for the next reading.

### Normalization

After computing unnormalized posteriors:

```
P(ghost at cell) = (likelihood × prior) / sum(all likelihoods × priors)
```

This ensures probabilities sum to 1.0 across all cells.

## Educational Value

This project demonstrates:

- **Bayesian Inference**: Real-world application of probability updates
- **Noisy Sensors**: Dealing with uncertainty and measurement error
- **Sequential Decision Making**: Choosing optimal sampling locations
- **Probability Visualization**: Understanding distributions and convergence
- **Manhattan Distance**: Common distance metric in grid-based problems

## Troubleshooting

### "No module named 'graphics'"

Install the graphics library:

```bash
pip install graphics.py
```

Or download `graphics.py` from [Zelle's website](http://mcsp.wartburg.edu/zelle/python/graphics.py) and place it in the project directory.

### Window Doesn't Open

Ensure you have a display environment configured:
- On macOS/Linux: X11 or native display
- On Windows: Should work out of the box
- For remote systems: Configure X forwarding

### Probabilities Don't Update

Check that:
- The sensor model in `Model.py` is properly defined
- All probabilities sum to 1.0
- Distance calculations are correct
