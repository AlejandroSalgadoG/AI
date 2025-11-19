# Decision Tree Classifier

A Python implementation of a Decision Tree classifier for supervised machine learning tasks. This project provides a framework for building decision trees using entropy-based information gain for feature selection.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Format](#data-format)
- [Configuration](#configuration)
- [Implementation Details](#implementation-details)
- [Examples](#examples)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)

## Overview

This project implements a decision tree algorithm from scratch using Python. Decision trees are a popular machine learning algorithm that can be used for both classification and regression tasks. This implementation focuses on classification problems and uses entropy-based information gain to select the best features for splitting.

The algorithm works by:
1. Reading and parsing dataset files
2. Building a feature table representation
3. Calculating entropy and information gain for each feature
4. Recursively splitting the data to construct the tree

## Project Structure

```
DesitionTree/
├── Main.py           # Main entry point and program orchestration
├── DataHandler.py    # Data loading, parsing, and preprocessing utilities
├── Solver.py         # Decision tree algorithm implementation
├── Parameters.py     # Configuration parameters and settings
├── Standars.py       # Class label definitions and mappings
├── data.dat          # Sample dataset file
└── README.md         # Project documentation
```

### File Descriptions

- **Main.py**: Orchestrates the entire workflow from data loading to tree construction
- **DataHandler.py**: Contains utility functions for data manipulation including:
  - Dataset reading and parsing
  - Data splitting (train/validation/test)
  - Data filtering and transformation
  - Table building for tree construction
- **Solver.py**: Core decision tree algorithm with:
  - Entropy calculation
  - Feature selection based on information gain
  - Tree building logic
- **Parameters.py**: Centralized configuration for:
  - Dataset paths
  - Train/validation/test split ratios
- **Standars.py**: Class label mappings for different datasets
- **data.dat**: Example dataset in CSV format

## Installation

### Prerequisites

- Python 3.x
- No external dependencies required (uses only standard library)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd DesitionTree
```

2. Ensure the data file is in the correct location or update the path in `Parameters.py`

3. Run the program:
```bash
python Main.py
```

## Usage

### Basic Usage

```python
from DataHandler import readDataset, parseData, buildTable
from Solver import desitionTree

# Load and parse data
rawData = readDataset("./data.dat")
data = parseData(rawData)

# Build feature table
treeTable = buildTable(data)

# Create decision tree
tree = desitionTree(treeTable)
```

### Running the Main Program

Simply execute:
```bash
python Main.py
```

## Data Format

The input data should be in CSV format with the following structure:

```
feature1,feature2,...,featureN,class_label
1,2,red
1,3,red
2,1,blue
3,1,blue
```

### Format Requirements

- **Features**: Numeric values (integers or floats)
- **Class Labels**: String identifiers (e.g., "red", "blue", "Iris-setosa")
- **Delimiter**: Comma (`,`)
- **Last Column**: Must be the class label
- **File Ending**: Must end with a newline

### Example Datasets

The project includes a sample dataset (`data.dat`) with:
- 2 features (numeric)
- Binary classification (red/blue)
- 4 samples

The code is also compatible with the Iris dataset (configuration commented out in `Standars.py`).

## Configuration

### Parameters.py

Configure the following parameters:

```python
# Dataset path
dataPath = "./data.dat"

# Data split ratios (must sum to 1.0)
trainRate = 0.7      # 70% for training
heldoutRate = 0.15   # 15% for validation
testRate = 0.15      # 15% for testing
```

### Standars.py

Define class label mappings:

```python
# Class name to numeric code mapping
classes = {
    "red"  : 0,
    "blue" : 1
}

# Numeric code to class name mapping
translate = {
    0 : "red",
    1 : "blue"
}
```

## Implementation Details

### Data Processing Pipeline

1. **Reading** (`readDataset`): Loads raw data from file
2. **Parsing** (`parseData`): Converts CSV to structured format
   - Splits lines and features
   - Converts features to floats
   - Maps class labels to numeric codes
3. **Table Building** (`buildTable`): Transposes data into column-wise format for efficient feature access

### Decision Tree Algorithm

The algorithm uses the following approach:

1. **Entropy Calculation**:
   ```
   H(S) = -Σ(p_i * log2(p_i))
   ```
   Where p_i is the probability of class i in the dataset

2. **Information Gain**: Measures the reduction in entropy after splitting on a feature

3. **Feature Selection**: Chooses the feature with the highest information gain (lowest entropy)

### Data Structures

- **Raw Data**: String from file
- **Parsed Data**: List of lists `[[feature1, feature2, ..., class_code], ...]`
- **Table**: Column-wise representation `[[all_feature1_values], [all_feature2_values], ..., [all_class_codes]]`

### Utility Functions

#### DataHandler.py

- `readDataset(path)`: Reads file content
- `parseData(rawData)`: Converts CSV to structured format
- `buildTable(dataset)`: Creates column-wise feature table
- `divideData(data)`: Randomly splits data into train/validation/test sets
- `getConsistentData(query, dataset)`: Filters data by class label
- `removeLabel(dataset)`: Removes class labels from dataset

#### Solver.py

- `desitionTree(table)`: Builds the decision tree (in progress)
- `entropy(probs)`: Calculates Shannon entropy
- `selectFeature(information)`: Selects feature with minimum entropy

## Examples

### Example 1: Basic Classification

```python
# data.dat
1,2,red
1,3,red
2,1,blue
3,1,blue

# The tree will learn to classify based on feature values
# Feature 0 or Feature 1 will be selected based on information gain
```

### Example 2: Using with Iris Dataset

1. Download the Iris dataset
2. Update `Parameters.py`:
   ```python
   dataPath = "/path/to/iris.data"
   ```
3. Update `Standars.py` (uncomment the Iris configuration):
   ```python
   classes = {
       "Iris-setosa" : 0,
       "Iris-versicolor" : 1,
       "Iris-virginica" : 2
   }
   ```
4. Run `python Main.py`

### Example 3: Data Splitting

```python
from DataHandler import readDataset, parseData, divideData

rawData = readDataset("./data.dat")
data = parseData(rawData)

# Split into training, validation, and test sets
training, heldout, test = divideData(data)

print(f"Training samples: {len(training)}")
print(f"Validation samples: {len(heldout)}")
print(f"Test samples: {len(test)}")
```

## Algorithm Theory

### Entropy

Entropy measures the impurity or uncertainty in a dataset. For a dataset with classes:

- **High entropy**: Classes are evenly distributed (high uncertainty)
- **Low entropy**: One class dominates (low uncertainty)
- **Zero entropy**: Only one class present (pure node)

### Information Gain

Information gain measures how much a feature reduces entropy:

```
IG(S, A) = H(S) - Σ(|S_v|/|S|) * H(S_v)
```

Where:
- S is the dataset
- A is the feature
- S_v is the subset of S where feature A has value v
- H is entropy

The feature with the highest information gain is selected for splitting.
