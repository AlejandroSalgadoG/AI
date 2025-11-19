# Dual Perceptron

A Python implementation of the Dual Perceptron algorithm for multi-class classification with kernel support and visualization capabilities.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Algorithm](#algorithm)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Kernel Functions](#kernel-functions)
- [Visualization](#visualization)
- [Example](#example)
- [License](#license)

## Overview

This project implements the **Dual Perceptron** algorithm, a kernel-based online learning algorithm for multi-class classification. Unlike the standard (primal) perceptron that operates directly in feature space, the dual perceptron works in a dual representation using kernel functions, enabling non-linear decision boundaries through the kernel trick.

The dual perceptron is particularly useful for:
- Multi-class classification problems
- Non-linearly separable data (with appropriate kernels)
- High-dimensional feature spaces
- Online learning scenarios

## Features

- **Dual Perceptron Implementation**: Kernel-based perceptron algorithm
- **Multi-Class Support**: Handles classification with multiple classes
- **Multiple Kernel Functions**:
  - Linear kernel (K)
  - Polynomial kernel (K2)
- **Data Management**: Built-in data loading, parsing, and splitting utilities
- **Visualization**: Graphical interface for displaying data points and decision boundaries
- **Configurable Parameters**: Easy-to-modify configuration system

## Algorithm

### Dual Perceptron Learning

The dual perceptron maintains a weight vector **α** for each class. During training:

1. For each training sample:
   - Compute the predicted class using the kernel-based decision function
   - If the prediction is incorrect:
     - Increment α for the true class at the sample index
     - Decrement α for the predicted class at the sample index

2. Repeat until all samples are correctly classified (or convergence criteria met)

### Classification Rule

For a test sample **x**, the predicted class is:

```
y_pred = argmax_c Σ_i α_c,i · K(x_i, x)
```

Where:
- `c` is the class index
- `i` is the training sample index
- `α_c,i` is the weight for class c at sample i
- `K(·,·)` is the kernel function

## Installation

### Prerequisites

- Python 3.x
- graphics.py library (for visualization)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DualPerceptron.git
cd DualPerceptron
```

2. Install the graphics library:
```bash
pip install graphics.py
```

## Usage

### Basic Usage

Run the main script to train the perceptron on your dataset:

```bash
python Main.py
```

### Custom Dataset

1. Prepare your data in CSV format (see [Data Format](#data-format))
2. Update the `dataPath` in `Parameters.py`
3. Run the training script

```python
#!/bin/python

from Parameters import *
from Gui import *
from DataHandler import *
from Perceptron import *

def main():
    rawData = readDataset(dataPath)
    data = parseData(rawData)
    alphas = learn(data)

main()
```

## Project Structure

```
DualPerceptron/
│
├── Main.py           # Entry point and main execution flow
├── Perceptron.py     # Core dual perceptron algorithm
├── DataHandler.py    # Data loading, parsing, and splitting
├── Gui.py           # Visualization and graphical interface
├── Parameters.py     # Configuration parameters
├── Standars.py       # Constants and class mappings
├── data.dat         # Sample dataset
└── README.md        # This file
```

### File Descriptions

#### Main.py
Entry point of the application. Orchestrates data loading, parsing, and training.

#### Perceptron.py
Contains the core dual perceptron algorithm:
- `learn(data)`: Main training loop
- `perceptron(data, sample, alphas)`: Classification function
- `K(x1, x2)`: Linear kernel function
- `K2(x1, x2)`: Polynomial kernel function (degree 2)

#### DataHandler.py
Handles all data operations:
- `readDataset(path)`: Reads data from file
- `parseData(rawData)`: Converts raw data to numerical format
- `divideData(data)`: Splits data into training/validation/test sets

#### Gui.py
Visualization module using graphics.py:
- Grid-based display
- Data point rendering with color-coded classes
- Decision boundary visualization

#### Parameters.py
Configuration file for adjusting:
- Data path
- Number of classes
- Domain and range sizes
- Data split ratios (train/validation/test)

#### Standars.py
Constants for visualization and class mappings.

## Configuration

Edit `Parameters.py` to configure the system:

```python
# Data source
dataPath = "./data.dat"

# Classification settings
numClass = 2  # Number of classes

# Visualization domain
domain = 10   # X-axis range
rangeSz = 10  # Y-axis range

# Data split ratios
trainRate = 0.7      # 70% for training
heldoutRate = 0.15   # 15% for validation
testRate = 0.15      # 15% for testing
```

## Data Format

The data file should be in CSV format with the following structure:

```
feature1,feature2,...,featureN,class_label
```

### Example (data.dat)

```
1,2,red
2,1,blue
```

### Class Labels

Classes are mapped in `Standars.py`:

```python
classes = {
    "red"   : 0,
    "blue"  : 1,
    "green" : 2,
    0 : "red",
    1 : "blue",
    2 : "green"
}
```

You can add more classes by extending this dictionary.

## Kernel Functions

The project includes two kernel functions in `Perceptron.py`:

### Linear Kernel (K)

```python
def K(x1, x2):
    dot = 0
    size = len(x1) - 1
    for i in range(size):
        dot += x1[i] * x2[i]
    return dot
```

Computes the dot product: **K(x₁, x₂) = x₁ · x₂**

### Polynomial Kernel (K2)

```python
def K2(x1, x2):
    dot = 0
    size = len(x1) - 1
    for i in range(size):
        dot += x1[i] * x2[i]
    dot += 1
    return dot**2
```

Computes: **K(x₁, x₂) = (x₁ · x₂ + 1)²**

### Using Different Kernels

To switch between kernels, modify the `perceptron()` function in `Perceptron.py`:

```python
# Change this line:
ans += alpha * K(data[idx], sample)

# To:
ans += alpha * K2(data[idx], sample)  # For polynomial kernel
```

### Adding Custom Kernels

You can add custom kernel functions:

```python
def K_rbf(x1, x2, gamma=0.5):
    """Radial Basis Function (RBF) kernel"""
    diff = 0
    size = len(x1) - 1
    for i in range(size):
        diff += (x1[i] - x2[i]) ** 2
    return math.exp(-gamma * diff)
```

## Visualization

The `Gui` class provides visualization capabilities:

- **Grid Display**: Visual grid for coordinate reference
- **Data Points**: Color-coded circles representing different classes
- **Decision Boundaries**: Lines showing classification boundaries
- **Interactive**: Click-based coordinate input

### Visualization Parameters (Standars.py)

```python
rectSize = 50  # Size of each grid cell in pixels
radius = 7     # Radius of data point circles
thick = 4      # Line thickness for boundaries
sizeX = domain * rectSize   # Window width
sizeY = rangeSz * rectSize  # Window height
```

## Example

### Simple 2D Classification

```python
# data.dat
1,2,red
2,1,blue
3,3,red
4,2,blue
5,4,red

# Run training
python Main.py
```

The algorithm will:
1. Load and parse the dataset
2. Initialize alpha weights to zero
3. Iterate through samples, updating weights on misclassifications
4. Continue until all samples are correctly classified
5. Return the learned alpha weights

### Working with Iris Dataset

To use the Iris dataset:

1. Download iris.data
2. Update `Parameters.py`:
```python
dataPath = "/path/to/iris.data"
numClass = 3  # Iris has 3 classes
```

3. Update class mappings in `Standars.py`:
```python
classes = {
    "Iris-setosa"     : 0,
    "Iris-versicolor" : 1,
    "Iris-virginica"  : 2,
    0 : "Iris-setosa",
    1 : "Iris-versicolor",
    2 : "Iris-virginica"
}
```

## Algorithm Complexity

- **Time Complexity**: O(n² · m) per epoch
  - n: number of training samples
  - m: number of features
  - Epochs: depends on data separability

- **Space Complexity**: O(c · n)
  - c: number of classes
  - n: number of training samples

## Advantages of Dual Perceptron

1. **Kernel Trick**: Can learn non-linear decision boundaries
2. **Multi-Class**: Natural extension to multi-class problems
3. **Online Learning**: Updates incrementally with new data
4. **Simplicity**: Easy to implement and understand
5. **Interpretability**: Alpha weights show importance of training samples
