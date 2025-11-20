# Clustering Algorithms Implementation

A Python implementation of various clustering algorithms for image segmentation and data analysis. This project provides implementations of both classical and fuzzy clustering methods with visualization capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Visualization](#visualization)
- [Performance Metrics](#performance-metrics)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## 🎯 Overview

This project implements five clustering algorithms from scratch using NumPy, designed primarily for image segmentation tasks but applicable to any multidimensional data. The implementation focuses on:

- **Educational clarity**: Clean, readable code for understanding clustering algorithms
- **Flexibility**: Modular design allowing easy algorithm comparison
- **Visualization**: Rich 2D, 3D, and image-based visualization capabilities
- **Metrics**: Comprehensive evaluation using Silhouette, Davies-Bouldin, and Calinski-Harabasz scores

## ✨ Features

### Implemented Algorithms

1. **K-Means Clustering** - Fast partitional clustering
2. **Fuzzy C-Means** - Soft clustering with membership degrees
3. **Mountain Clustering** - Grid-based density estimation
4. **Subtractive Clustering** - Data-point-based density estimation
5. **Agglomerative Hierarchical Clustering** - Bottom-up hierarchical approach

### Capabilities

- ✅ Image color segmentation
- ✅ Custom distance metrics (Euclidean, Manhattan, Infinity norm)
- ✅ 2D/3D scatter plot visualization
- ✅ Image-based result visualization
- ✅ Performance metrics evaluation
- ✅ Support for dimensionality reduction (t-SNE)

## 🚀 Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Clustering.git
cd Clustering
```

2. Install dependencies:

```bash
pip install numpy matplotlib scikit-learn pandas seaborn
```

Or create a `requirements.txt`:

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
pandas>=1.2.0
seaborn>=0.11.0
```

## 🎬 Quick Start

### Basic Example

```python
from Data import X  # Load your data
from Distances import euclidean
from Kmeans import kmeans
from PlotResults import plot3d, plot_result_as_img

# Run K-Means clustering
C, M = kmeans(X, euclidean, num_c=7, iters=30)

# Visualize results in 3D
plot3d(X, M, C)

# For image data
img_n, img_m = 100, 100  # Your image dimensions
plot_result_as_img(img_n, img_m, X, M, C)
```

### Running the Main Script

```python
python Main.py
```

The script will cluster the image data and display visualization results.

## 📚 Algorithms

### 1. K-Means Clustering

**File**: `Kmeans.py`

Classic partitional clustering that minimizes within-cluster variance.

**Key Functions**:
- `kmeans(X, distance_func, num_c=2, iters=1, print_error=True)`

**Parameters**:
- `X`: Data matrix (n_samples × n_features)
- `distance_func`: Distance metric function
- `num_c`: Number of clusters
- `iters`: Number of iterations
- `print_error`: Print cost function at each iteration

**Algorithm Steps**:
1. Randomly initialize cluster centers
2. Assign points to nearest center
3. Update centers as mean of assigned points
4. Repeat until convergence or max iterations

**Example**:
```python
from Kmeans import kmeans
from Distances import euclidean

C, M = kmeans(X, euclidean, num_c=5, iters=50)
# C: cluster centers, M: membership labels
```

---

### 2. Fuzzy C-Means

**File**: `FuzzyKMeans.py`

Soft clustering where each point has a degree of membership to all clusters.

**Key Functions**:
- `fuzzy_kmeans(X, distance_func, m=2, num_c=2, iters=1, print_error=True)`

**Parameters**:
- `X`: Data matrix
- `distance_func`: Distance metric
- `m`: Fuzziness parameter (typically 2)
- `num_c`: Number of clusters
- `iters`: Number of iterations

**Algorithm Steps**:
1. Initialize fuzzy membership matrix randomly
2. Calculate cluster centers using weighted means
3. Update membership values based on distances
4. Repeat until convergence

**Example**:
```python
from FuzzyKMeans import fuzzy_kmeans
from Utils import fuzzy_to_membership

C, U = fuzzy_kmeans(X, euclidean, m=2, num_c=5, iters=50)
M, Umax = fuzzy_to_membership(U)
# C: centers, U: fuzzy membership, M: hard labels, Umax: membership degrees
```

---

### 3. Mountain Clustering

**File**: `Mountain.py`

Grid-based method using density estimation on a discretized feature space.

**Key Functions**:
- `mountain(X, distance_func, num_c=2, num_div=1, sigma=0.1, beta=0.1)`

**Parameters**:
- `X`: Data matrix
- `distance_func`: Distance metric
- `num_c`: Number of clusters
- `num_div`: Grid divisions per dimension
- `sigma`: Density function width
- `beta`: Destruction parameter

**Algorithm Steps**:
1. Create a grid over the feature space
2. Calculate mountain function (density) at each grid point
3. Select highest peak as cluster center
4. Reduce mountain function around selected center
5. Repeat for desired number of clusters

**Example**:
```python
from Mountain import mountain
from Utils import calculate_membership

C = mountain(X, euclidean, num_c=5, num_div=5)
M = calculate_membership(X, C, euclidean)
```

---

### 4. Subtractive Clustering

**File**: `Substractive.py`

Data-point-based density estimation without requiring grid discretization.

**Key Functions**:
- `substract(X, distance_func, num_c=2, num_div=1, ra=1.0, rb=None)`

**Parameters**:
- `X`: Data matrix
- `distance_func`: Distance metric
- `num_c`: Number of clusters
- `ra`: Radius defining neighborhood (acceptance radius)
- `rb`: Radius for removal (default: 1.5 × ra)

**Algorithm Steps**:
1. Calculate density potential at each data point
2. Select point with highest potential as center
3. Reduce potential around selected center
4. Repeat for desired number of clusters

**Example**:
```python
from Substractive import substract
from Utils import calculate_membership

C = substract(X, euclidean, num_c=5, ra=0.5)
M = calculate_membership(X, C, euclidean)
```

---

### 5. Agglomerative Hierarchical Clustering

**File**: `Agglomerative.py`

Bottom-up hierarchical clustering using centroid linkage.

**Key Functions**:
- `agglomerative(X, distance_func, num_c=2)`

**Parameters**:
- `X`: Data matrix
- `distance_func`: Distance metric
- `num_c`: Final number of clusters

**Algorithm Steps**:
1. Start with each point as a separate cluster
2. Calculate pairwise distances between clusters
3. Merge closest clusters
4. Update distance matrix
5. Repeat until desired number of clusters

**Example**:
```python
from Agglomerative import agglomerative
from Utils import calculate_center

M = agglomerative(X, euclidean, num_c=5)
C = calculate_center(X, M)
```

## 📁 Project Structure

```
Clustering/
├── Main.py                      # Main execution script
├── Data.py                      # Data loading and preprocessing
├── Utils.py                     # Utility functions
├── Distances.py                 # Distance metric functions
├── Colors.py                    # Color palette for visualization
├── PlotResults.py               # Visualization functions
│
├── Kmeans.py                    # K-Means implementation
├── FuzzyKMeans.py              # Fuzzy C-Means implementation
├── Mountain.py                  # Mountain clustering implementation
├── Substractive.py             # Subtractive clustering implementation
├── Agglomerative.py            # Agglomerative clustering implementation
│
├── clustering_metrics.ipynb    # Metrics evaluation notebook
│
├── images/                      # Sample images
│   ├── MNM.jpg
│   ├── MNM_small.jpg
│   └── MNM_ultrasmall.jpg
```

### File Descriptions

#### Core Algorithm Files

- **`Kmeans.py`**: K-Means clustering with cost function optimization
- **`FuzzyKMeans.py`**: Fuzzy C-Means with soft membership assignments
- **`Mountain.py`**: Grid-based density clustering
- **`Substractive.py`**: Point-based density clustering
- **`Agglomerative.py`**: Hierarchical clustering with cluster merging

#### Utility Files

- **`Data.py`**: Image loading and preprocessing utilities
- **`Utils.py`**: Common functions (membership calculation, center calculation, etc.)
- **`Distances.py`**: Distance metric implementations
- **`Colors.py`**: Color schemes for visualization
- **`PlotResults.py`**: Comprehensive visualization functions

#### Analysis Files

- **`Main.py`**: Example usage demonstrating all algorithms
- **`clustering_metrics.ipynb`**: Performance evaluation and comparison

## 💡 Usage Examples

### Example 1: Image Segmentation with K-Means

```python
import numpy as np
import matplotlib.image as matimg
from Kmeans import kmeans
from Distances import euclidean
from PlotResults import plot_result_as_img

# Load image
img = matimg.imread("images/MNM_small.jpg").astype(int)
img_n, img_m, _ = img.shape
X = img.reshape(img_n * img_m, 3)

# Cluster colors
C, M = kmeans(X, euclidean, num_c=7, iters=30)

# Visualize segmentation
plot_result_as_img(img_n, img_m, X, M, C)
```

### Example 2: Comparing Multiple Algorithms

```python
from Kmeans import kmeans
from FuzzyKMeans import fuzzy_kmeans
from Mountain import mountain
from Agglomerative import agglomerative
from Utils import fuzzy_to_membership, calculate_membership, calculate_center
from Distances import euclidean

num_clusters = 5

# K-Means
C1, M1 = kmeans(X, euclidean, num_c=num_clusters, iters=30)

# Fuzzy C-Means
C2, U = fuzzy_kmeans(X, euclidean, num_c=num_clusters, iters=30)
M2, Umax = fuzzy_to_membership(U)

# Mountain
C3 = mountain(X, euclidean, num_c=num_clusters, num_div=5)
M3 = calculate_membership(X, C3, euclidean)

# Agglomerative
M4 = agglomerative(X, euclidean, num_c=num_clusters)
C4 = calculate_center(X, M4)
```

### Example 3: Custom Distance Metric

```python
import numpy as np
from Kmeans import kmeans

# Define custom distance metric
def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))

# Use with any algorithm
C, M = kmeans(X, manhattan_distance, num_c=5, iters=30)
```

### Example 4: Fuzzy Clustering with Uncertainty Visualization

```python
from FuzzyKMeans import fuzzy_kmeans
from Utils import fuzzy_to_membership
from PlotResults import plot3d

# Run fuzzy clustering
C, U = fuzzy_kmeans(X, euclidean, m=2, num_c=7, iters=30)
M, Umax = fuzzy_to_membership(U)

# Visualize with membership degree as transparency
plot3d(X, M, C, Umax)
```

## 📊 Visualization

The project provides three types of visualization:

### 1. 2D Scatter Plot

```python
from PlotResults import plot_result2d

plot_result2d(X, M, C, features=[0, 1])
```

Features:
- Points colored by cluster
- Cluster centers marked with 'X'
- Optional transparency based on membership degree (fuzzy)

### 2. 3D Scatter Plot

```python
from PlotResults import plot3d

plot3d(X, M, C, features=[0, 1, 2])
```

Features:
- Interactive 3D visualization
- Color-coded clusters
- Center markers
- Fuzzy membership visualization

### 3. Image Visualization

```python
from PlotResults import plot_result_as_img

plot_result_as_img(img_n, img_m, X, M, C)
```

Features:
- Side-by-side original and segmented image
- Color quantization visualization
- Fuzzy segmentation with transparency

## 📈 Performance Metrics

The project includes comprehensive metric evaluation in `clustering_metrics.ipynb`.

### Supported Metrics

1. **Silhouette Score** (-1 to 1, higher is better)
   - Measures how similar points are to their own cluster vs. other clusters

2. **Davies-Bouldin Index** (0 to ∞, lower is better)
   - Ratio of within-cluster to between-cluster distances

3. **Calinski-Harabasz Score** (0 to ∞, higher is better)
   - Ratio of between-cluster to within-cluster dispersion

### Running Metrics Evaluation

```python
import sklearn.metrics as mt

def metrics_clusters(X, labels):
    return {
        "Silhouette": mt.silhouette_score(X, labels),
        "Davies-Bouldin": mt.davies_bouldin_score(X, labels),
        "Calinski-Harabasz": mt.calinski_harabasz_score(X, labels)
    }

# Evaluate clustering result
scores = metrics_clusters(X, M)
print(scores)
```

### Comparative Analysis

The notebook performs:
- Multi-algorithm comparison (K-Means, Fuzzy C-Means, Mountain, Subtractive, Agglomerative)
- Multiple distance metrics (Euclidean, Manhattan, Infinity norm)
- Varying cluster numbers (2-12)
- Normalized metric visualization
- Result export to CSV

## 📦 Dependencies

### Required Libraries

```python
numpy>=1.19.0          # Core numerical operations
matplotlib>=3.3.0      # Visualization
scikit-learn>=0.24.0   # Metrics and t-SNE
pandas>=1.2.0          # Data analysis (metrics notebook)
seaborn>=0.11.0        # Advanced plotting (metrics notebook)
```

### Installation Commands

```bash
# Using pip
pip install numpy matplotlib scikit-learn pandas seaborn

# Using conda
conda install numpy matplotlib scikit-learn pandas seaborn
```

## 🔧 Configuration

### Modifying Main.py

The `Main.py` file contains commented examples for all algorithms. Uncomment the desired algorithm:

```python
# K-Means (active by default)
C, M = kmeans(X, euclidean, num_c=7, iters=30)
plot3d(X, M, C)
plot_result_as_img(img_n, img_m, X, M, C)

# Fuzzy C-Means (uncomment to use)
# C, U = fuzzy_kmeans(X, euclidean, num_c=7, iters=30)
# M, Umax = fuzzy_to_membership(U)
# plot3d(X, M, C, Umax)
# plot_result_as_img(img_n, img_m, X, M, C, Umax)

# And so on for other algorithms...
```

### Custom Data Loading

Modify `Data.py` to load your own data:

```python
import numpy as np
import matplotlib.image as matimg

# For images
img = matimg.imread("path/to/your/image.jpg").astype(int)
img_n, img_m, _ = img.shape
X = img.reshape(img_n * img_m, 3)

# For CSV/other data
# X = np.loadtxt("data.csv", delimiter=",")
```

## 🎓 Algorithm Details and Theory

### Distance Metrics

The project supports any distance function with signature `distance(x, y) -> float`.

**Implemented**:
- **Euclidean**: `np.linalg.norm(x - y)` - L2 norm
- **Manhattan**: `np.sum(np.abs(x - y))` - L1 norm
- **Infinity**: `np.linalg.norm(x - y, ord=np.inf)` - L∞ norm

### Convergence Criteria

- **K-Means**: Fixed number of iterations (could add convergence check)
- **Fuzzy C-Means**: Fixed iterations (monitors cost function)
- **Mountain/Subtractive**: Number of clusters determines stopping
- **Agglomerative**: Stops when desired cluster count reached

### Computational Complexity

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| K-Means | O(n × k × d × i) | O(n + k × d) |
| Fuzzy C-Means | O(n × k × d × i) | O(n × k + k × d) |
| Mountain | O(g × n × d + k × g) | O(g + k × d) |
| Subtractive | O(n² × d + k × n) | O(n + k × d) |
| Agglomerative | O(n³) | O(n²) |

Where:
- n = number of data points
- k = number of clusters
- d = number of dimensions
- i = number of iterations
- g = grid size (num_div^d)

## 🐛 Troubleshooting

### Common Issues

**1. Memory Error with Large Images**

```python
# Resize image before processing
from PIL import Image

img = Image.open("large_image.jpg")
img = img.resize((200, 200))  # Reduce size
```

**2. Empty Clusters in K-Means**

This can happen with poor initialization. The current implementation may fail. Consider:
- Increasing iterations
- Using different random seed
- Implementing K-Means++ initialization

**3. Slow Performance with Agglomerative**

Agglomerative has O(n³) complexity. For large datasets:
- Subsample your data
- Use other algorithms (K-Means, Fuzzy C-Means)
- Consider approximate methods

**4. Mountain/Subtractive Grid Size**

If `num_div` is too large, grid becomes huge:
- Grid points = num_div^dimensions
- Start with num_div=3-5
- Increase gradually if needed
