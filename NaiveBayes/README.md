# Naive Bayes Classifier

A Python implementation of the Naive Bayes classification algorithm with support for discrete feature classification.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Details](#algorithm-details)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)

## Overview

This project demonstrates the fundamental concepts of probabilistic classification using Bayes' theorem with the "naive" assumption of feature independence.

The classifier supports:
- Discrete feature classification
- Multiple classes
- Maximum Likelihood (ML) smoothing
- Laplace smoothing (implemented but not actively used)
- Modular architecture for easy extension

## Features

- **Simple and Clean Implementation**: Easy-to-understand code structure
- **Flexible Data Handling**: Support for CSV-like data format
- **Probability Smoothing**: ML and Laplace smoothing techniques to handle zero probabilities
- **Configurable Parameters**: Easy configuration through `Parameters.py` and `Standars.py`
- **Class Translation**: Human-readable class labels with numeric encoding
- **Logarithmic Probability Calculation**: Prevents numerical underflow for better stability

## Project Structure

```
NaiveBayes/
│
├── Main.py           # Entry point and main execution flow
├── Model.py          # Class probability definitions
├── Solver.py         # Core Naive Bayes algorithm and smoothing functions
├── DataHandler.py    # Data loading, parsing, and preprocessing utilities
├── Parameters.py     # Configuration parameters (paths, rates, smoothing)
├── Standars.py       # Class definitions and label mappings
├── data.dat          # Sample dataset
└── README.md         # This file
```

### File Descriptions

#### `Main.py`
The main entry point that orchestrates the entire classification pipeline:
- Loads and parses the dataset
- Separates data by class
- Calculates conditional probabilities for each feature
- Performs classification on sample data
- Outputs results

#### `Model.py`
Defines prior probabilities for each class (P(C)):
```python
classesProb = {
    "red"   : 0.5,
    "blue"  : 0.5,
}
```

#### `Solver.py`
Contains the core machine learning algorithms:
- `mlSmooth()`: Maximum likelihood smoothing for probability estimation
- `laplaceSmooth()`: Laplace smoothing (k-smoothing) implementation
- `naiveBayes()`: Main classification function using Bayes' theorem
- `getMaxPos()`: Utility to find the class with maximum posterior probability

#### `DataHandler.py`
Provides utilities for data manipulation:
- `readDataset()`: Reads data from file
- `parseData()`: Converts raw data into structured format
- `divideData()`: Splits data into training, validation, and test sets
- `getConsistentData()`: Filters data by class label
- `removeLabel()`: Strips labels from feature vectors

#### `Parameters.py`
Configuration file for tunable parameters:
- `dataPath`: Path to the dataset file
- `k`: Smoothing parameter for Laplace smoothing
- `trainRate`, `heldoutRate`, `testRate`: Dataset split ratios

#### `Standars.py`
Defines class encodings and translations:
- `classes`: Maps string labels to numeric IDs
- `translate`: Maps numeric IDs back to string labels

## Installation

### Prerequisites

- Python 3.x
- No external dependencies required (uses only standard library)

### Setup

1. Clone or download this repository:
```bash
git clone <repository-url>
cd NaiveBayes
```

2. Ensure Python 3 is installed:
```bash
python --version
```

3. Make the main script executable (optional):
```bash
chmod +x Main.py
```

## Usage

### Basic Usage

Run the classifier with the default sample data:

```bash
python Main.py
```

or

```bash
./Main.py
```

### Custom Dataset

1. Prepare your dataset in the required format (see [Data Format](#data-format))
2. Update `dataPath` in `Parameters.py`:
```python
dataPath = "./your_data.dat"
```
3. Update class definitions in `Standars.py`:
```python
classes = {
    "class1": 0,
    "class2": 1,
    "class3": 2
}

translate = {
    0: "class1",
    1: "class2",
    2: "class3"
}
```
4. Update prior probabilities in `Model.py`:
```python
classesProb = {
    "class1": 0.33,
    "class2": 0.33,
    "class3": 0.34
}
```
5. Run the classifier

## Algorithm Details

### Naive Bayes Theorem

The Naive Bayes classifier is based on Bayes' theorem with the "naive" assumption that features are conditionally independent given the class:

\[
P(C|F_1, F_2, ..., F_n) = \frac{P(C) \cdot P(F_1, F_2, ..., F_n|C)}{P(F_1, F_2, ..., F_n)}
\]

With the independence assumption:

\[
P(C|F_1, F_2, ..., F_n) \propto P(C) \cdot \prod_{i=1}^{n} P(F_i|C)
\]

### Implementation Details

1. **Training Phase**:
   - Data is loaded and parsed from the dataset file
   - Samples are separated by class
   - For each class and each feature, conditional probabilities P(F|C) are calculated
   - Probabilities are stored in a nested structure: `condProb[class][feature][value]`

2. **Smoothing**:
   - **ML Smoothing**: Calculates probabilities using maximum likelihood estimation:
     ```
     P(feature_value|class) = count(feature_value, class) / count(class)
     ```
   - **Laplace Smoothing**: Adds k to each count to avoid zero probabilities (implemented but not default)

3. **Classification Phase**:
   - For a new sample, calculate log probabilities to prevent underflow:
     ```
     log P(C|F) = log P(C) + Σ log P(F_i|C)
     ```
   - Select the class with the highest log probability

### Why Logarithms?

The implementation uses logarithms for numerical stability. When multiplying many small probabilities, the result can underflow (become too small for floating-point representation). Using logarithms converts multiplication to addition:

```
log(a × b × c) = log(a) + log(b) + log(c)
```

## Configuration

### Parameters.py

```python
dataPath = "./data.dat"          # Path to dataset
k = 1                            # Laplace smoothing parameter
trainRate = 0.7                  # Training set ratio (70%)
heldoutRate = 0.15               # Validation set ratio (15%)
testRate = 0.15                  # Test set ratio (15%)
```

### Standars.py

Define your classes and their encodings:

```python
classes = {
    "red"  : 0,
    "blue" : 1
}

translate = {
    0 : "red",
    1 : "blue"
}
```

### Model.py

Set prior probabilities for each class:

```python
classesProb = {
    "red"   : 0.5,   # P(red) = 0.5
    "blue"  : 0.5,   # P(blue) = 0.5
}
```

## Data Format

The dataset should be a CSV-like format with the following structure:

```
feature1,feature2,...,featureN,class_label
feature1,feature2,...,featureN,class_label
...
```

### Example (`data.dat`):

```
1,2,red
1,3,red
2,1,blue
3,1,blue
```

Each line represents a sample with:
- Features: Comma-separated numeric values
- Label: The last value is the class label (string)

### Requirements:
- Features must be numeric values
- The last column must be the class label
- Class labels must match those defined in `Standars.py`
- File should end with a newline

## Examples

### Example 1: Binary Classification

Given the sample dataset:
```
1,2,red
1,3,red
2,1,blue
3,1,blue
```

Classifying a new sample `[1, 2]`:

```python
sample = [1, 2]
classification = naiveBayes(condProb, sample)
print(translate[classification])  # Output: red
```

### Example 2: Multi-class Classification (Iris Dataset)

Uncomment the Iris dataset configuration in the files:

**Standars.py:**
```python
classes = {
    "Iris-setosa" : 0,
    "Iris-versicolor" : 1,
    "Iris-virginica" : 2
}
```

**Model.py:**
```python
classesProb = {
    "Iris-setosa"     : 0.33,
    "Iris-versicolor" : 0.33,
    "Iris-virginica"  : 0.33
}
```

**Parameters.py:**
```python
dataPath = "/path/to/iris.data"
```

### Example 3: Custom Sample Classification

Modify `Main.py` to classify your own samples:

```python
# After computing condProb
samples = [
    [1, 2],
    [3, 1],
    [2, 3]
]

for sample in samples:
    classification = naiveBayes(condProb, sample)
    print(f"Sample {sample} -> {translate[classification]}")
```

## Reference

### DataHandler.py

#### `readDataset(path)`
Reads dataset from file.
- **Parameters**: `path` (str) - Path to dataset file
- **Returns**: Raw data as string

#### `parseData(rawData)`
Parses raw data into structured format.
- **Parameters**: `rawData` (str) - Raw dataset string
- **Returns**: List of samples, each as `[feature1, feature2, ..., class_id]`

#### `divideData(data)`
Splits data into train, validation, and test sets.
- **Parameters**: `data` (list) - Parsed dataset
- **Returns**: Tuple of (train, heldout, test) lists

#### `getConsistentData(query, dataset)`
Filters dataset by class.
- **Parameters**:
  - `query` (int) - Class ID to filter
  - `dataset` (list) - Dataset to filter
- **Returns**: Filtered dataset containing only samples of specified class

#### `removeLabel(dataset)`
Removes class labels from dataset.
- **Parameters**: `dataset` (list) - Dataset with labels
- **Returns**: Dataset without labels (modifies in place and returns)

### Solver.py

#### `mlSmooth(dataset, feature)`
Calculates probability distribution for a feature using ML estimation.
- **Parameters**:
  - `dataset` (list) - Class-specific dataset
  - `feature` (int) - Feature index
- **Returns**: Dictionary mapping feature values to probabilities

#### `laplaceSmooth(dataset, k, feature)`
Calculates probability distribution using Laplace smoothing.
- **Parameters**:
  - `dataset` (list) - Class-specific dataset
  - `k` (int) - Smoothing parameter
  - `feature` (int) - Feature index
- **Returns**: Dictionary mapping feature values to probabilities

#### `naiveBayes(condProb, sample)`
Classifies a sample using Naive Bayes.
- **Parameters**:
  - `condProb` (list) - Conditional probability tables for all classes
  - `sample` (list) - Feature vector to classify
- **Returns**: Class ID (int) with highest posterior probability

#### `getMaxPos(probs)`
Finds index of maximum value.
- **Parameters**: `probs` (list) - List of probabilities
- **Returns**: Index of maximum value
