# K-Nearest Neighbors (K-NN) Classifier

A Python implementation of the K-Nearest Neighbors algorithm with graphical visualization for classifying data points.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Algorithm Details](#algorithm-details)
- [Data Format](#data-format)
- [How It Works](#how-it-works)
- [Example](#example)
- [Customization](#customization)
- [License](#license)

## Overview

This project implements a K-Nearest Neighbors classifier that:
- Reads data from a CSV-like file
- Visualizes data points on a graphical interface
- Splits data into training, held-out, and test sets
- Classifies samples using the K-NN algorithm with dot product similarity
- Displays results in the console

## Features

- **Interactive Visualization**: Displays data points on a grid-based GUI using different colors for different classes
- **Configurable K Value**: Easily adjust the number of neighbors to consider
- **Flexible Data Splitting**: Configurable train/held-out/test split ratios
- **Dot Product Similarity**: Uses dot product for similarity calculation between vectors
- **Majority Voting**: Classification based on majority vote from K nearest neighbors

## Requirements

- Python 3.x
- `graphics.py` library (for GUI visualization)

## Installation

1. Clone or download this repository:
```bash
git clone <repository-url>
cd Knn
```

2. Install the graphics library:

The project uses the `graphics.py` library by John Zelle. You can download it from:
- [graphics.py Documentation](https://mcsp.wartburg.edu/zelle/python/graphics.py)

Download `graphics.py` and place it in the project directory or install it in your Python path.

3. Ensure you have a data file ready (see [Data Format](#data-format))

## Usage

Run the main program:

```bash
python Main.py
```

Or make it executable:

```bash
chmod +x Main.py
./Main.py
```

The program will:
1. Open a graphical window displaying the data points
2. Split the data into training, held-out, and test sets
3. Classify each sample in the held-out set
4. Print classifications to the console
5. Wait for a mouse click to close

## Project Structure

```
Knn/
├── Main.py           # Entry point of the application
├── Knn.py            # Core K-NN algorithm implementation
├── DataHandler.py    # Data reading, parsing, and splitting functions
├── Gui.py            # Graphical user interface implementation
├── Parameters.py     # Configuration parameters
├── Standars.py       # GUI constants and standards
├── data.dat          # Sample data file
└── README.md         # This file
```

### File Descriptions

#### `Main.py`
The main entry point that orchestrates the entire workflow:
- Initializes the GUI
- Loads and parses data
- Visualizes data points
- Divides data into sets
- Performs classification on held-out samples

#### `Knn.py`
Contains the core K-NN algorithm:
- `classify(data, sample)`: Classifies a sample based on K nearest neighbors
- `similarity(vec1, vec2)`: Calculates dot product similarity between two vectors
- `insertInOrder(classification, new)`: Maintains sorted list of neighbors
- `vote(samples)`: Performs majority voting for final classification

#### `DataHandler.py`
Handles all data operations:
- `readDataset(path)`: Reads data from file
- `parseData(rawData)`: Parses CSV-like data into structured format
- `divideData(data)`: Randomly splits data into train/held-out/test sets
- `unlableData(rawData)`: Removes labels from data (for testing)

#### `Gui.py`
Manages graphical visualization:
- Creates a grid-based window
- Draws data points with color-coded classifications
- Handles mouse input for interaction

#### `Parameters.py`
Configuration file with adjustable parameters:
- Data file path
- K value for K-NN
- Domain and range sizes
- Train/held-out/test split ratios

#### `Standars.py`
GUI-related constants for rendering

## Configuration

Edit `Parameters.py` to customize the behavior:

```python
# Data source
dataPath = "./data.dat"

# Number of neighbors to consider
k = 2

# Grid dimensions for visualization
domain = 10
rangeSz = 10

# Data split ratios (must sum to 1.0)
trainRate = 0.8      # 80% for training
heldoutRate = 0.10   # 10% for held-out validation
testRate = 0.10      # 10% for testing
```

## Algorithm Details

### K-Nearest Neighbors

The K-NN algorithm classifies a sample based on the majority class among its K nearest neighbors in the feature space.

**Steps:**

1. **Calculate Similarity**: For each training sample, compute the dot product with the target sample
2. **Sort by Similarity**: Maintain a sorted list of samples by similarity score
3. **Select K Neighbors**: Extract the K samples with highest similarity
4. **Majority Vote**: Count the class labels of the K neighbors
5. **Classify**: Assign the most frequent class label

### Similarity Metric

This implementation uses **dot product** as the similarity measure:

```
similarity(v1, v2) = Σ(v1[i] * v2[i])
```

Higher dot product indicates greater similarity between vectors.

## Data Format

The data file should be in CSV format with:
- Each row representing a sample
- Comma-separated feature values
- Last column containing the class label

**Example (`data.dat`):**

```
1.5,2.3,4.1,ClassA
2.1,3.5,1.2,ClassB
3.4,2.8,3.9,ClassA
1.8,4.2,2.5,ClassB
```

**Format Requirements:**
- Features can be floating-point numbers
- Class labels should be strings
- No header row
- File must end with a newline

## How It Works

### Workflow

1. **Initialization**
   - GUI window is created with a grid
   - Data is loaded from the specified file path

2. **Data Processing**
   - Raw data is parsed into feature vectors and labels
   - Data is visualized on the grid (x, y coordinates with color-coded classes)

3. **Data Division**
   - Dataset is randomly split into three subsets:
     - **Training set**: Used to find nearest neighbors
     - **Held-out set**: Used for validation
     - **Test set**: Reserved for final evaluation

4. **Classification**
   - For each sample in the held-out set:
     - Calculate similarity to all training samples
     - Find K nearest neighbors
     - Determine classification via majority vote
     - Print results to console

5. **Interaction**
   - Program waits for mouse click before closing

### Classification Process

For each sample to classify:

1. Compute dot product with every training sample
2. Maintain sorted list of (sample, similarity) tuples
3. Extract top K samples
4. Count class labels among these K neighbors
5. Return the class with the highest count

## Example

### Sample Data File

```
1,2,red
2,3,red
3,4,blue
8,9,blue
7,8,blue
```

### Running with k=2

For a new sample `[2.5, 3.5]`:
1. Calculate similarity to all training samples
2. Find 2 nearest neighbors (highest dot products)
3. If both neighbors are "red", classify as "red"
4. If one is "red" and one is "blue", tie-breaking depends on implementation

### Visual Output

The GUI displays:
- A grid with dimensions specified in `Parameters.py`
- Colored circles representing data points
- Each color corresponds to a class label

## Customization

### Changing the K Value

Modify `k` in `Parameters.py`:

```python
k = 5  # Use 5 nearest neighbors
```

### Adjusting Data Split

Modify split ratios in `Parameters.py`:

```python
trainRate = 0.7      # 70% training
heldoutRate = 0.20   # 20% held-out
testRate = 0.10      # 10% test
```

### Using a Different Dataset

1. Prepare your data in CSV format
2. Update the path in `Parameters.py`:

```python
dataPath = "/path/to/your/data.csv"
```

### Changing Similarity Metric

To use a different similarity metric (e.g., Euclidean distance), modify the `similarity()` function in `Knn.py`:

```python
def similarity(vec1, vec2):
    # Example: Euclidean distance (lower is better, so negate)
    distance = sum((vec1[i] - vec2[i])**2 for i in range(len(vec2)))
    return -distance  # Negate so higher is better
```

### Visualization Customization

Modify `Standars.py` to change GUI appearance:

```python
rectSize = 50  # Size of each grid cell
radius = 7     # Size of data point circles
thick = 4      # Line thickness
```

## Troubleshooting

**Issue**: `ImportError: No module named graphics`
- **Solution**: Download and install `graphics.py` in your project directory or Python path

**Issue**: GUI window doesn't appear
- **Solution**: Ensure you have a graphical environment (X11 on Linux, native on macOS/Windows)

**Issue**: Data not displaying correctly
- **Solution**: Check that your data format matches the expected CSV format and coordinates are within the domain/range

**Issue**: Classification seems inaccurate
- **Solution**: Try adjusting the K value, or consider using Euclidean distance instead of dot product
