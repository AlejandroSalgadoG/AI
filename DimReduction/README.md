# Dimensionality Reduction: LLE and MDS

A theoretical and practical implementation of non-linear dimensionality reduction algorithms using **Locally Linear Embedding (LLE)** and **Multidimensional Scaling (MDS)** in MATLAB.

## 📋 Table of Contents

- [Overview](#overview)
- [Algorithms](#algorithms)
  - [Locally Linear Embedding (LLE)](#locally-linear-embedding-lle)
  - [Multidimensional Scaling (MDS)](#multidimensional-scaling-mds)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Running LLE](#running-lle)
  - [Running MDS](#running-mds)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Results](#results)
- [Mathematical Background](#mathematical-background)
- [Documentation](#documentation)
- [Authors](#authors)
- [References](#references)

## Overview

This project presents a theoretical and practical approach to the problem of non-linear dimensionality reduction. It implements two powerful manifold learning algorithms:

1. **LLE (Locally Linear Embedding)**: A two-step procedure that first calculates a weighted data representation based on K-nearest neighbors, then minimizes the distance between the data representation and a low-dimensionality configuration.

2. **MDS (Multidimensional Scaling)**: Constructs a distance matrix that approximates the distance over a manifold by solving shortest path problems over a graph, which is then used to find a low-dimensional representation.

Both algorithms are tested on 3-dimensional manifolds including the Swiss Roll, S-Curve, and Sphere datasets.

## Algorithms

### Locally Linear Embedding (LLE)

LLE is a non-linear dimensionality reduction technique that preserves local neighborhood relationships. The algorithm works in three main steps:

1. **Find K-Nearest Neighbors**: For each data point, identify its K nearest neighbors
2. **Compute Reconstruction Weights**: Calculate weights that best reconstruct each point from its neighbors
3. **Compute Low-Dimensional Embedding**: Find low-dimensional coordinates that preserve the reconstruction weights

**Key Parameters:**
- `n`: Number of data points (e.g., 1000)
- `k`: Number of nearest neighbors (e.g., 12)
- `p`: Target dimensionality (e.g., 2)
- `tol`: Regularization tolerance (e.g., 0.001)

### Multidimensional Scaling (MDS)

MDS finds a low-dimensional representation that preserves pairwise distances between points. This implementation uses graph-based distances to approximate geodesic distances on the manifold:

1. **Construct K-Nearest Neighbors Graph**: Build a graph connecting each point to its K nearest neighbors
2. **Compute Geodesic Distances**: Use shortest paths on the graph to approximate manifold distances
3. **Classical MDS**: Apply eigenvalue decomposition to find the low-dimensional embedding

**Key Parameters:**
- `n`: Number of data points (e.g., 1500)
- `k`: Number of nearest neighbors (e.g., 14)
- `p`: Target dimensionality (e.g., 2)

## Features

- ✅ Complete implementation of LLE algorithm
- ✅ Complete implementation of MDS with geodesic distance approximation
- ✅ K-nearest neighbors search functionality
- ✅ Multiple 3D test manifolds (Swiss Roll, S-Curve, Sphere)
- ✅ Visualization utilities with color mapping
- ✅ Academic documentation with LaTeX sources
- ✅ Example results and figures

## Requirements

- **MATLAB** (R2016b or later recommended)
- Required MATLAB Toolboxes:
  - Statistics and Machine Learning Toolbox (for `eigs`)
  - Graph Theory Toolbox (for MDS shortest path computation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DimReduction.git
cd DimReduction
```

2. Open MATLAB and navigate to the project directory:
```matlab
cd('/path/to/DimReduction')
```

3. Add the relevant subdirectories to your MATLAB path:
```matlab
addpath('Lle')
addpath('Mds')
```

## Usage

### Running LLE

To run the Locally Linear Embedding algorithm on the Swiss Roll dataset:

```matlab
cd Lle
Lle  % Runs the main LLE script
```

**Example: Custom LLE Implementation**

```matlab
% Generate Swiss Roll dataset
n = 1000;
[x, y, z] = build_swiss_roll(n);
data = [x y z];

% Set parameters
k = 12;  % Number of neighbors
p = 2;   % Target dimensionality
tol = 0.001;  % Regularization parameter

% Compute weight matrix
W = zeros(n);
e = ones(k,1);
for i=1:n
    x_i = data(i,:);
    neighbors = k_neighbors(data, x_i, k);
    v_i = data(neighbors,:)';
    x_e = repmat(x_i, k, 1)';
    g = (x_e - v_i)' * (x_e - v_i);
    g = g + eye(k)*tol*trace(g);
    w_i = g\e;
    w_i = w_i / sum(w_i);
    W(neighbors,i) = w_i;
end

% Compute embedding
I = eye(n);
M = (I-W) * (I-W)';
[eigvec, ~] = eigs(M, p+1, 'SM');
Y = eigvec(:, 2:p+1);

% Visualize results
colors = get_cuadrant_colors(x, z);
figure;
scatter(Y(:,1), Y(:,2), 20, colors, 'filled');
title('LLE Embedding of Swiss Roll');
```

### Running MDS

To run the Multidimensional Scaling algorithm on the S-Curve dataset:

```matlab
cd Mds
Mds  % Runs the main MDS script
```

**Example: Custom MDS Implementation**

```matlab
% Generate S-Curve dataset
n = 1500;
[x, y, z] = build_s_curve(n);
data = [x y z];

% Set parameters
k = 14;  % Number of neighbors
p = 2;   % Target dimensionality

% Compute distance matrix
dist_matrix = get_distance(data);

% Build k-nearest neighbors graph
graph_matrix = zeros(n);
for i=1:n
    x_i = data(i,:);
    neighbors = k_neighbors(dist_matrix, i, k);
    graph_matrix(i, neighbors) = dist_matrix(i, neighbors);
    graph_matrix(neighbors, i) = dist_matrix(i, neighbors);
end

% Compute geodesic distances using shortest paths
manifold = graph(graph_matrix);
D = distances(manifold);

% Classical MDS
H = eye(n) - ones(n)/n;
A = -0.5 * D.^2;
B = H * A * H;
[V, L] = eigs(B, p+3, 'LR');
L_half = sqrt(L);
Y_full = V * L_half;
Y = Y_full(:, 1:p);

% Visualize results
colors = get_cuadrant_colors(x, z);
figure;
scatter(Y(:,1), Y(:,2), 20, colors, 'filled');
title('MDS Embedding of S-Curve');
```

## Project Structure

```
DimReduction/
├── README.md                          # This file
├── Documentation/                     # Main documentation
│   ├── figures/                      # Result figures and images
│   │   ├── logo.png
│   │   ├── s_curve_result.png
│   │   ├── s_curve.png
│   │   ├── sphere_result.png
│   │   ├── sphere.png
│   │   ├── swiss_roll_result_lle.png
│   │   ├── swiss_roll_result_mds.png
│   │   └── swiss_roll.png
│   ├── Makefile                      # LaTeX compilation makefile
│   ├── poster.bib                    # Bibliography for poster
│   └── poster.tex                    # Conference poster source
├── Lle/                              # LLE implementation
│   ├── Lle.m                         # Main LLE algorithm
│   ├── k_neighbors.m                 # K-nearest neighbors search
│   ├── get_n_smallest.m              # Helper for finding nearest neighbors
│   ├── build_swiss_roll.m            # Swiss roll dataset generator
│   ├── build_sphere.m                # Sphere dataset generator
│   ├── get_cuadrant_colors.m         # Color mapping for visualization
│   ├── paint_neighbors.m             # Neighbor visualization utility
│   ├── plot_swiss_roll.m             # Plotting utility
│   └── Documentation/                # LLE-specific documentation
│       ├── Article_lle.tex
│       ├── Article_lle.bib
│       ├── images/
│       └── Makefile
└── Mds/                              # MDS implementation
    ├── Mds.m                         # Main MDS algorithm
    ├── k_neighbors.m                 # K-nearest neighbors search
    ├── get_distance.m                # Distance matrix computation
    ├── build_swiss_roll.m            # Swiss roll dataset generator
    ├── build_s_curve.m               # S-curve dataset generator
    ├── get_cuadrant_colors.m         # Color mapping for visualization
    ├── plot_swiss_roll.m             # Plotting utility
    └── Documentation/                # MDS-specific documentation
        ├── Article_mds.tex
        ├── Article_mds.bib
        ├── images/
        └── Makefile
```

## Datasets

The project includes several classic manifold learning test datasets:

### Swiss Roll
A 2D manifold embedded in 3D space in the shape of a rolled sheet.

**Generation:**
```matlab
[x, y, z] = build_swiss_roll(n);
% where:
% t = 3*pi/2 * (1 + 2*rand(n,1))
% x = t.*cos(t)
% y = 5 * rand(n,1)
% z = t.*sin(t)
```

### S-Curve
A 2D manifold forming an S-shaped curve in 3D space.

**Generation:**
```matlab
[x, y, z] = build_s_curve(n);
% where:
% t = 3 * pi * (rand(n,1) - 0.5)
% x = sin(t)
% y = 2 * rand(n,1)
% z = t./abs(t) .* (cos(t) - 1)
```

### Sphere (Hemisphere)
Points sampled on a spherical surface.

**Generation:**
```matlab
[x, y, z] = build_sphere(n, r);
% where r is the radius
% Samples points uniformly on a hemisphere
```

## Results

The algorithms successfully reduce the dimensionality of complex 3D manifolds to 2D while preserving their intrinsic structure:

### LLE Results
- **Swiss Roll**: Successfully "unrolls" the manifold into a 2D plane
- **Sphere**: Preserves local neighborhood structure

### MDS Results
- **Swiss Roll**: Preserves geodesic distances along the manifold
- **S-Curve**: Successfully flattens the curve while maintaining distance relationships

Result images can be found in:
- `Documentation/figures/`
- `Lle/Documentation/images/`
- `Mds/Documentation/images/`

## Mathematical Background

### LLE Algorithm

The LLE algorithm minimizes the following cost function:

**Step 1: Reconstruction weights**
```
Φ(W) = Σᵢ ||xᵢ - Σⱼ wᵢⱼxⱼ||²
```

Subject to: Σⱼ wᵢⱼ = 1 and wᵢⱼ = 0 if xⱼ is not a neighbor of xᵢ

**Step 2: Low-dimensional embedding**
```
Ψ(Y) = Σᵢ ||yᵢ - Σⱼ wᵢⱼyⱼ||²
```

This is solved via eigendecomposition: `M = (I - W)ᵀ(I - W)`

### MDS Algorithm

Classical MDS finds coordinates Y that minimize:

```
Stress = √(Σᵢⱼ (dᵢⱼ - δᵢⱼ)²)
```

Where:
- `dᵢⱼ` is the Euclidean distance in the low-dimensional space
- `δᵢⱼ` is the original (geodesic) distance

The solution involves:
1. Centering matrix: `H = I - (1/n)11ᵀ`
2. Double centering: `B = -½HD²H`
3. Eigendecomposition: `B = VΛVᵀ`
4. Coordinates: `Y = V₊Λ₊^(1/2)`

## Documentation

Academic documentation is available in LaTeX format:

### Compile Documentation

```bash
# Compile main poster
cd Documentation
make

# Compile LLE article
cd Lle/Documentation
make

# Compile MDS article
cd Mds/Documentation
make
```

The documentation includes:
- Theoretical background
- Algorithm descriptions
- Implementation details
- Experimental results
- Comparative analysis

### Related Algorithms

- **Isomap**: Geodesic distance-based extension of MDS
- **t-SNE**: Probabilistic dimensionality reduction for visualization
- **UMAP**: Uniform Manifold Approximation and Projection
- **Laplacian Eigenmaps**: Spectral method using graph Laplacian
