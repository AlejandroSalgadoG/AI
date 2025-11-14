# Ghost Tracker: Particle Filter HMM Simulation

An interactive probabilistic inference demonstration using particle filters and Hidden Markov Models (HMM) to track and locate a hidden ghost on a grid.

## Overview

This project implements a particle filter algorithm to estimate the position of a hidden "ghost" that moves randomly on a grid. The user interacts with a noisy sensor that provides color-coded readings based on the distance to the ghost. Through Bayesian inference and particle resampling, the system maintains a probability distribution over possible ghost locations, which is updated with each sensor reading and movement.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Algorithm Details](#algorithm-details)
- [Configuration](#configuration)
- [Game Controls](#game-controls)
- [Dependencies](#dependencies)

## Features

- **Interactive GUI**: Visual grid-based interface with probability displays
- **Particle Filter Algorithm**: Real-time probabilistic inference using 1000 particles
- **Sensor Model**: Distance-based noisy color sensor with 4 color states
- **Dynamic Updates**: Bayesian belief updates after each sensor reading
- **Ghost Movement**: Stochastic transition model with edge-aware probabilities
- **Visual Feedback**: Color-coded sensor readings and probability heat map

## Installation

### Prerequisites

- Python 3.x
- graphics.py library (Zelle's graphics library)

### Setup

1. Clone or download this repository:
```bash
git clone <repository-url>
cd Particles
```

2. Install the graphics library:
```bash
pip install graphics.py
```

Or download `graphics.py` from [John Zelle's website](https://mcsp.wartburg.edu/zelle/python/graphics.py) and place it in the project directory.

3. Run the program:
```bash
python Main.py
```

## Usage

### Starting the Game

Run the main script:
```bash
python Main.py
```

A window will open showing a 4×4 grid with probability values displayed in each cell.

### Game Flow

1. **Initial State**: The ghost is randomly placed (hidden) on the grid. All positions start with equal probability (0.06 or ~6.25% each).

2. **Sensor Reading**: Click on any grid cell to use the sensor at that position. The sensor will return a color based on the Manhattan distance to the ghost:
   - **Red**: Likely very close (distance 0)
   - **Orange**: Likely distance 1
   - **Yellow**: Likely distance 2
   - **Green**: Likely distance 3+

3. **Probability Update**: After each sensor reading, the probability distribution is updated using Bayesian inference, and new values are displayed on the grid.

4. **Moving the Ghost**: Click the **NEXT** button to:
   - Clear the grid colors
   - Move the ghost according to the transition model
   - Update particle positions
   - Recalculate probabilities

5. **Making a Guess**: Click the **BUST** button, then click on the grid cell where you think the ghost is located:
   - **HIT!** (red): You found the ghost
   - **MISS!** (blue): Wrong location
   - The actual ghost position is revealed in the console

## How It Works

### Particle Filter Algorithm

The system uses a **Sequential Monte Carlo** (particle filter) approach:

1. **Initialization**:
   - 1000 particles are distributed uniformly across the grid
   - Each particle represents a hypothesis about the ghost's location

2. **Sensor Update (Correction Step)**:
   - When a sensor reading is taken, each particle is weighted by how likely it would produce that reading
   - Particles are resampled based on weights (importance resampling)
   - High-weight particles (consistent with observation) are duplicated
   - Low-weight particles are eliminated

3. **Movement (Prediction Step)**:
   - Each particle moves according to the transition model
   - Movement is stochastic with position-dependent probabilities
   - Mirrors how the actual ghost moves

4. **Probability Estimation**:
   - The probability at each grid cell is proportional to the number of particles at that location

### Sensor Model

The sensor provides noisy color readings based on Manhattan distance:

| Distance | Green | Yellow | Orange | Red |
|----------|-------|--------|--------|-----|
| 0        | 5%    | 5%     | 5%     | **85%** |
| 1        | 5%    | 5%     | **85%** | 5% |
| 2        | 5%    | **85%** | 5%     | 5% |
| 3        | **85%** | 5%     | 5%     | 5% |
| 4+       | **85%** | 5%     | 5%     | 5% |

The sensor is noisy - there's always a 5% chance of getting any color, but the correct color for the distance appears 85% of the time.

### Transition Model

The ghost moves stochastically with probabilities depending on its position:

- **Corners**: 25% stay, 50% toward center (one direction), 25% toward center (other direction)
- **Edges**: 16% stay, 52% toward center (main direction), 16% each for other valid directions
- **Center**: 12% stay, 52% in one preferred direction, 12% in other three directions

The transition model is **biased** - certain directions have higher probabilities, making the ghost's movement somewhat predictable but still random.

## Project Structure

```
Particles/
│
├── Main.py          # Entry point and game loop
├── Solver.py        # Particle filter algorithms and probability calculations
├── Model.py         # Sensor and transition models
├── Gui.py           # Graphics interface using graphics.py
├── Parameters.py    # Configuration parameters
├── Standars.py      # Utility functions (coordinate conversions)
└── README.md        # This file
```

### File Descriptions

#### `Main.py`
The entry point that orchestrates the game:
- Initializes the GUI
- Generates the ghost
- Creates initial particle distribution
- Handles the main game loop
- Processes user clicks (sensor readings, move ghost, bust ghost)

#### `Solver.py`
Core algorithms and inference engine:
- `getInitialDist()`: Uniform initial probability distribution
- `generateGhost()`: Randomly places the ghost
- `useSensor()`: Simulates sensor reading at a position
- `distributeParticles()`: Initial particle distribution
- `redistributeParticles()`: Resampling based on weights
- `weightParticles()`: Assigns weights based on sensor likelihood
- `moveParticles()`: Applies transition model to all particles
- `getNewPosDist()`: Bayesian update P(pos|color) using P(color|pos) and P(pos)
- `getProbs()`: Estimates probability distribution from particle positions
- `normalize()`: Normalizes probability distributions
- `selectRandom()`: Samples from probability distribution

#### `Model.py`
Defines probabilistic models:
- `model`: Dictionary mapping distances to color probabilities P(color|distance)
- `transition()`: Returns transition probability distribution P(next_pos|current_pos)
- `translation`: Converts between color names and indices
- `maxDist`: Maximum distance considered (4)

#### `Gui.py`
Graphics interface class:
- Grid visualization
- Probability display
- Button rendering (BUST, NEXT)
- Color-coded sensor feedback
- Mouse input handling
- Result display (HIT/MISS)

#### `Parameters.py`
Configuration constants:
- `numRow`: Grid size (4×4)
- `particleNum`: Number of particles (1000)

#### `Standars.py`
Utility functions:
- `fromPosToIdx()`: Converts (row, col) to linear index
- `fromIdxToPos()`: Converts linear index to (row, col)
- `size`: Window size calculation
- `btnSize`: Button dimension

## Algorithm Details

### Bayesian Inference

The system implements Bayes' theorem for belief updates:

```
P(ghost_pos | sensor_color) ∝ P(sensor_color | ghost_pos) × P(ghost_pos)
```

Where:
- **P(ghost_pos)**: Prior belief (current probability distribution)
- **P(sensor_color | ghost_pos)**: Likelihood (from sensor model)
- **P(ghost_pos | sensor_color)**: Posterior belief (updated distribution)

### Particle Filter Steps

1. **Initialization**:
   ```python
   particles = distributeParticles(particleNum, uniform_distribution)
   ```

2. **Sensor Update**:
   ```python
   # Get likelihood distribution using Bayes' rule
   condProbs = getNewPosDist(sensor_pos, color, probs)

   # Weight particles by likelihood
   weights = weightParticles(particles, condProbs)

   # Resample particles
   particles = redistributeParticles(particles, normalize(weights))
   ```

3. **Prediction (Movement)**:
   ```python
   # Move each particle according to transition model
   particles = moveParticles(particles)
   ```

4. **Probability Estimation**:
   ```python
   # Count particles at each position
   probs = getProbs(particles)
   ```

### Advantages of Particle Filters

- **Non-parametric**: No assumption about probability distribution shape
- **Multi-modal**: Can represent multiple hypotheses simultaneously
- **Flexible**: Easy to incorporate complex models
- **Approximate**: Trades accuracy for computational efficiency

## Configuration

Modify `Parameters.py` to customize:

```python
numRow = 4           # Grid size (numRow × numRow)
particleNum = 1000   # Number of particles (more = better accuracy, slower)
```

### Effect of Parameters

- **Increasing `numRow`**: Larger grid, more complex problem
- **Increasing `particleNum`**: Better probability estimates, smoother convergence, but slower computation

## Game Controls

| Action | Control |
|--------|---------|
| Take sensor reading | Click on any grid cell |
| Move ghost | Click **NEXT** button (right side, bottom) |
| Make guess | Click **BUST** button (right side, top), then click grid cell |
| Close game | Click anywhere after revealing result |

### Button Locations

- **BUST**: Right panel, top half (blue when active)
- **NEXT**: Right panel, bottom half (black initially)

## Dependencies

### Required
- **Python 3.x**: Core language
- **graphics.py**: John Zelle's simple graphics library for GUI

### Optional
- **random**: Built-in Python module (no installation needed)

## Mathematical Background

### Hidden Markov Model (HMM)

The system models a Hidden Markov Model with:
- **States**: Grid positions (16 states for 4×4 grid)
- **Hidden state**: Ghost position (not directly observable)
- **Observations**: Sensor color readings (noisy measurements)
- **Transition model**: P(next_state | current_state)
- **Observation model**: P(observation | state)

### Manhattan Distance

Distance metric used for sensor model:
```
distance = |ghost_row - sensor_row| + |ghost_col - sensor_col|
```

This creates diamond-shaped distance contours rather than circular (Euclidean distance).

## Example Scenario

1. **Initial**: Ghost at (1,2), all positions show ~0.06 probability
2. **Sensor at (1,1)**: Returns "orange" → probabilities increase around (1,1)
3. **Sensor at (2,2)**: Returns "red" → high probability at (2,2) and nearby cells
4. **Click NEXT**: Ghost moves to (1,3), particles redistribute
5. **Sensor at (1,3)**: Returns "red" → probability spikes at (1,3)
6. **Click BUST → Click (1,3)**: **HIT!** 🎯

## Tips for Playing

- Take sensor readings from multiple locations before guessing
- Pay attention to sensor model - red means distance 0, not necessarily certainty
- The ghost prefers certain movement directions - watch patterns
- After several readings, probabilities should concentrate in a small region
- Moving the ghost (NEXT) tests the system's ability to track a dynamic target

## Technical Notes

- The typo in "Standars.py" is preserved for compatibility (should be "Standards.py")
- Ghost generation is limited to positions (0-2, 0-2) - first 9 cells of 16 total
- Particle resampling uses systematic resampling via `selectRandom()`
- GUI uses synchronous event handling (blocking on `getMouse()`)

## Topics used
- Bayesian inference in practice
- Monte Carlo methods
- Hidden Markov Models
- Sensor fusion
- Recursive state estimation
