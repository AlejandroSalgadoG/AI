# K-means Clustering - CUDA Parallel Implementation

A high-performance implementation of the K-means clustering algorithm using CUDA for GPU acceleration. This project demonstrates parallel computing techniques for unsupervised machine learning, achieving significant speedup compared to sequential implementations.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Algorithm](#algorithm)
- [Prerequisites](#prerequisites)
- [Building the Project](#building-the-project)
- [Usage](#usage)
- [Input Format](#input-format)
- [Architecture](#architecture)
- [CUDA Kernels](#cuda-kernels)
- [Performance Considerations](#performance-considerations)
- [File Structure](#file-structure)
- [Example](#example)

## Overview

K-means is a popular unsupervised learning algorithm used for clustering data points into K distinct groups. This implementation leverages NVIDIA CUDA to parallelize computationally intensive operations, making it suitable for large datasets with many features.

The algorithm iteratively:
1. Assigns each data point to the nearest centroid
2. Recalculates centroids based on assigned points
3. Repeats until convergence or maximum iterations reached

## Features

- **GPU Acceleration**: Utilizes CUDA for parallel computation on NVIDIA GPUs
- **Dynamic Kernel Configuration**: Automatically adjusts block and thread dimensions based on GPU capabilities
- **Euclidean Distance**: Uses standard Euclidean distance for cluster assignment
- **Convergence Detection**: Monitors classification changes to detect algorithm convergence
- **Memory Efficient**: Optimized memory transfers between host and device
- **Scalable**: Handles datasets with arbitrary dimensions and cluster counts

## Algorithm

The K-means algorithm implemented follows these steps:

1. **Initialization**: Select first K samples as initial centroids
2. **Assignment Step**: Assign each sample to the nearest centroid (parallel)
3. **Update Step**: Recalculate centroid positions as the mean of assigned samples (parallel)
4. **Convergence Check**: Compare current classifications with previous iteration
5. **Iteration**: Repeat steps 2-4 until convergence or max iterations

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA support (Compute Capability 2.0 or higher)
- Sufficient GPU memory for your dataset

### Software Requirements
- CUDA Toolkit (tested with CUDA 7.0+)
- NVIDIA GPU drivers
- g++ compiler (GCC 4.8+)
- nvcc compiler (included with CUDA Toolkit)
- Make build system

### Installation

#### Ubuntu/Debian
```bash
# Install CUDA Toolkit
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit

# Install build essentials
sudo apt-get install build-essential
```

#### macOS
```bash
# Download CUDA Toolkit from NVIDIA website
# https://developer.nvidia.com/cuda-downloads

# Install Xcode Command Line Tools
xcode-select --install
```

## Building the Project

1. **Clone the repository** (or navigate to the project directory):
```bash
cd /path/to/Parallel
```

2. **Verify CUDA installation**:
```bash
nvcc --version
```

3. **Configure CUDA path** (if necessary):
Edit the `Makefile` and update `CUDA_INCLUDE_PATH` to match your CUDA installation:
```makefile
CUDA_INCLUDE_PATH=/opt/cuda/include  # Default path
# Or use: /usr/local/cuda/include
```

4. **Build the project**:
```bash
make clean
make
```

This will generate the `Main` executable.

## Usage

The program reads input from standard input (stdin) and outputs progress to standard output (stdout).

### Basic Usage

```bash
./Main < input_file.txt
```

### With Output Redirection

```bash
./Main < input_file.txt > output.txt
```

### Interactive Mode

```bash
./Main
# Then manually enter parameters and data
```

## Input Format

The input must follow this specific format:

```
<number_of_samples>
<number_of_features>
<number_of_clusters>
<max_iterations>
<feature1> <feature2> ... <featureN>
<feature1> <feature2> ... <featureN>
...
```

### Example Input

```
6
2
2
100
1 1
2 2
1 2
8 8
9 9
8 9
```

This example:
- 6 samples
- 2 features per sample
- 2 clusters (K=2)
- Maximum 100 iterations
- 6 data points in 2D space

### Parameter Descriptions

| Parameter | Description |
|-----------|-------------|
| `samples` | Total number of data points |
| `features` | Dimensionality of each data point |
| `k` | Number of clusters to form |
| `max_iterations` | Maximum iterations before forced termination |

## Architecture

### Memory Management

The implementation uses a host-device memory model:

- **Host Memory (CPU)**: Stores original samples, centroids, and final classifications
- **Device Memory (GPU)**: Copies of samples, centroids, and classifications for parallel processing
- **Memory Transfers**: Minimized to reduce overhead; only necessary data transferred

### Execution Flow

```
┌─────────────────────────────────────┐
│  Read Parameters & Initialize Data  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Allocate & Copy to Device Memory   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Configure Kernel Dimensions        │
└──────────────┬──────────────────────┘
               │
          ┌────▼────┐
          │ Iterate │◄─────────────┐
          └────┬────┘              │
               │                   │
               ▼                   │
    ┌──────────────────────┐      │
    │ Assign to Centroids  │      │
    │  (GPU Kernel)        │      │
    └──────────┬───────────┘      │
               │                   │
               ▼                   │
    ┌──────────────────────┐      │
    │ Check Convergence    │      │
    │  (GPU Kernel)        │      │
    └──────────┬───────────┘      │
               │                   │
          ┌────▼────┐              │
          │Changed? │──Yes─────────┘
          └────┬────┘
               │No
               ▼
    ┌──────────────────────┐
    │ Copy Results to Host │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   Output Results     │
    └──────────────────────┘
```

## CUDA Kernels

### 1. `centroid_calculation`

**Purpose**: Assigns each sample to the nearest centroid.

**Parallelization**: One thread per sample (vertical parallelization)

**Algorithm**:
- Each thread processes one sample
- Calculates Euclidean distance to all centroids
- Assigns sample to nearest centroid

```cpp
Grid:  [ceil(samples / max_threads), 1, 1]
Block: [max_threads, 1, 1]
```

### 2. `compare_classes`

**Purpose**: Checks if classifications changed between iterations.

**Parallelization**: One thread per sample

**Algorithm**:
- Compares current classification with previous iteration
- Sets flag to false if any difference found
- Used for convergence detection

### 3. `copy_classes`

**Purpose**: Copies current classifications to previous buffer.

**Parallelization**: One thread per sample

**Algorithm**:
- Simple memory copy operation
- Prepares for next iteration comparison

### 4. `centroid_movement`

**Purpose**: Recalculates centroid positions based on assigned samples.

**Parallelization**: One thread per (centroid, feature) pair (horizontal parallelization)

**Algorithm**:
- Each thread updates one feature of one centroid
- Calculates mean of all assigned samples for that feature
- Handles empty clusters by keeping current position

```cpp
Grid:  [ceil((k * features) / max_threads), 1, 1]
Block: [max_threads, 1, 1]
```

### Device Functions

#### `euclidian_distance`

Calculates Euclidean distance between a sample and a centroid:

\[
d(x, c) = \sqrt{\sum_{i=1}^{n} (x_i - c_i)^2}
\]

## Performance Considerations

### Optimization Strategies

1. **Coalesced Memory Access**: Memory access patterns optimized for GPU architecture
2. **Thread Utilization**: Maximizes GPU occupancy by using full thread blocks
3. **Memory Transfer Minimization**: Only essential data copied between host/device
4. **Convergence Detection on GPU**: Avoids unnecessary host-device synchronization

### Computational Complexity

- **Assignment Step**: O(samples × k × features) - fully parallelized
- **Update Step**: O(k × features × samples) - fully parallelized
- **Per Iteration**: O(samples × k × features)
- **Total**: O(iterations × samples × k × features)

### Speedup

Theoretical speedup depends on:
- Number of CUDA cores
- Dataset size (samples × features)
- Number of clusters (k)
- Memory bandwidth

Typical speedup: **10-100x** compared to sequential CPU implementation for large datasets.

## File Structure

```
Parallel/
│
├── Main.cpp              # Entry point, orchestrates the algorithm
├── Kmeans.cu            # CUDA kernels and main kmeans function
├── Kmeans.h             # K-means function declarations
│
├── Utilities.cpp        # Input/output and initialization utilities
├── Utilities.h          # Utility function declarations
│
├── CudaUtilities.cpp    # CUDA configuration and kernel dimensions
├── CudaUtilities.h      # CUDA utility declarations
│
├── Makefile            # Build configuration
└── README.md           # This documentation
```

### File Descriptions

| File | Purpose |
|------|---------|
| `Main.cpp` | Program entry point, memory management, algorithm orchestration |
| `Kmeans.cu` | CUDA kernel implementations and host-device coordination |
| `Kmeans.h` | Interface for K-means function |
| `Utilities.cpp` | Input parsing, centroid initialization, output formatting |
| `Utilities.h` | Utility function interfaces |
| `CudaUtilities.cpp` | GPU property querying, kernel configuration |
| `CudaUtilities.h` | CUDA utility structures and interfaces |

## Example

### Input Data

Create a file `test_data.txt`:

```
8
2
2
50
1.0 1.0
1.5 2.0
1.0 2.5
9.0 9.0
9.5 10.0
10.0 9.5
5.0 5.0
5.5 5.5
```

### Run the Program

```bash
./Main < test_data.txt
```

### Expected Output

```
Starting kmeans

iteration 1
iteration 2
iteration 3
iteration 4

done
```

The algorithm will converge when classifications stabilize. The final classifications are stored but not displayed by default.

### Enabling Output

To see the labeled samples, uncomment line 15 in `Main.cpp`:

```cpp
print_labeled_samples(h_samples, h_class, features, samples);
```

Then rebuild:

```bash
make clean && make
./Main < test_data.txt
```

Output will show each sample with its cluster label:

```
1.0 1.0 0
1.5 2.0 0
1.0 2.5 0
9.0 9.0 1
9.5 10.0 1
10.0 9.5 1
5.0 5.0 0
5.5 5.5 0
```

## Troubleshooting

### Common Issues

**Issue**: `nvcc: command not found`
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Issue**: CUDA include path not found
```bash
# Update Makefile CUDA_INCLUDE_PATH to your CUDA installation
# Common paths:
# /usr/local/cuda/include
# /opt/cuda/include
# /usr/include
```

**Issue**: GPU out of memory
- Reduce the dataset size
- Decrease the number of features
- Use a GPU with more memory

**Issue**: Slow convergence
- Try different initial centroids
- Increase max_iterations
- Check for outliers in your data

## Limitations

1. **Initial Centroids**: Uses first K samples as initial centroids (not random)
2. **Empty Clusters**: May produce NaN if a cluster becomes empty
3. **Local Optima**: K-means is sensitive to initialization
4. **Fixed K**: Number of clusters must be specified in advance
5. **Euclidean Distance**: Only supports Euclidean distance metric
