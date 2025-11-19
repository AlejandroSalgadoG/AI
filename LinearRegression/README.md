# Linear Regression

An implementation of linear regression algorithms using gradient descent, featuring multiple programming languages and progressively complex models.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Mathematical Background](#mathematical-background)
- [Implementations](#implementations)
  - [One Parameter (y = θx)](#one-parameter-y--θx)
  - [Two Parameters (y = θ₀ + θ₁x)](#two-parameters-y--θ₀--θ₁x)
  - [Multiple Parameters](#multiple-parameters)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Building and Running C++ Code](#building-and-running-c-code)
  - [Running Octave Code](#running-octave-code)
  - [Running R Statistics](#running-r-statistics)
- [Data Format](#data-format)
- [Algorithm Details](#algorithm-details)
- [Documentation](#documentation)
- [Examples](#examples)
- [Contributing](#contributing)

## Overview

This project provides educational implementations of linear regression using gradient descent optimization. It demonstrates three levels of complexity:

1. **One Parameter Model**: Simple proportional relationship (y = θx)
2. **Two Parameter Model**: Linear relationship with intercept (y = θ₀ + θ₁x)
3. **Multiple Parameter Model**: Multivariate linear regression

Each implementation is available in both **C++** and **Octave/MATLAB**, allowing comparison of approaches and performance characteristics.

## Project Structure

```
LinearRegression/
├── Code/
│   ├── OneParam/          # Single parameter implementation
│   │   ├── C++/
│   │   │   ├── OneParam.C
│   │   │   ├── Data.txt
│   │   │   └── Makefile
│   │   └── Octave/
│   │       ├── OneParam.m
│   │       └── Data.txt
│   ├── TwoParam/          # Two parameter implementation
│   │   ├── C++/
│   │   │   ├── TwoParam.C
│   │   │   ├── Data.txt
│   │   │   └── Makefile
│   │   └── Octave/
│   │       ├── TwoParam.m
│   │       └── Data.txt
│   └── MultiParam/        # Multiple parameter implementation
│       ├── C++/
│       │   ├── MultiParam.C
│       │   ├── Data.txt
│       │   └── Makefile
│       └── Octave/
│           ├── MultiParam.m
│           └── Data.txt
├── Documentation/         # LaTeX documentation
│   ├── OneParam/
│   ├── TwoParam/
│   ├── MultiParam/
│   └── Derivative/
├── Statistic/            # Statistical analysis in R
│   └── LinearRegression.R
└── README.md
```

## Mathematical Background

### Linear Regression

Linear regression models the relationship between a dependent variable `y` and one or more independent variables `x` by fitting a linear equation to observed data.

**General Form:**
```
h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

Where:
- `h(x)` is the hypothesis function (predicted value)
- `θᵢ` are the parameters (coefficients) to be learned
- `xᵢ` are the input features

### Cost Function

The cost function (Mean Squared Error) measures how well our model fits the data:

```
J(θ) = (1/2m) Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

Where:
- `m` is the number of training examples
- `h(x⁽ⁱ⁾)` is the predicted value for the i-th example
- `y⁽ⁱ⁾` is the actual value for the i-th example

### Gradient Descent

Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize the cost function:

```
θⱼ := θⱼ - α · ∂J(θ)/∂θⱼ
```

Where:
- `α` is the learning rate (step size)
- `∂J(θ)/∂θⱼ` is the partial derivative of the cost function with respect to parameter θⱼ

**Derivative for each parameter:**
```
∂J(θ)/∂θⱼ = (1/m) Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) · xⱼ⁽ⁱ⁾
```

## Implementations

### One Parameter (y = θx)

The simplest model assumes a proportional relationship with no intercept.

**Hypothesis:** `h(x) = θx`

**Use case:** Data that passes through the origin (0,0)

**Files:**
- C++: `Code/OneParam/C++/OneParam.C`
- Octave: `Code/OneParam/Octave/OneParam.m`

### Two Parameters (y = θ₀ + θ₁x)

Standard simple linear regression with an intercept term.

**Hypothesis:** `h(x) = θ₀ + θ₁x`

**Parameters:**
- `θ₀`: intercept (bias term)
- `θ₁`: slope (coefficient of x)

**Use case:** Most common linear regression scenarios

**Files:**
- C++: `Code/TwoParam/C++/TwoParam.C`
- Octave: `Code/TwoParam/Octave/TwoParam.m`

### Multiple Parameters

Multivariate linear regression with multiple input features.

**Hypothesis:** `h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ`

**Use case:** Complex relationships with multiple independent variables

**Files:**
- C++: `Code/MultiParam/C++/MultiParam.C` (under development)
- Octave: `Code/MultiParam/Octave/MultiParam.m`

## Getting Started

### Prerequisites

#### For C++ implementations:
- g++ compiler (GCC)
- make

```bash
# On macOS
xcode-select --install

# On Ubuntu/Debian
sudo apt-get install build-essential

# On Fedora/RHEL
sudo dnf install gcc-c++ make
```

#### For Octave implementations:
- GNU Octave

```bash
# On macOS
brew install octave

# On Ubuntu/Debian
sudo apt-get install octave

# On Fedora/RHEL
sudo dnf install octave
```

#### For R statistics:
- R language

```bash
# On macOS
brew install r

# On Ubuntu/Debian
sudo apt-get install r-base

# On Fedora/RHEL
sudo dnf install R
```

### Building and Running C++ Code

#### One Parameter Example:

```bash
cd Code/OneParam/C++

# Build the executable
make

# Run with data file and number of iterations
./OneParam Data.txt 5

# Clean build files
make clean
```

#### Two Parameter Example:

```bash
cd Code/TwoParam/C++

# Build the executable
make

# Run with data file and number of iterations
./TwoParam Data.txt 10

# Clean build files
make clean
```

**Command line arguments:**
- First argument: path to data file
- Second argument: number of gradient descent iterations

### Running Octave Code

#### One Parameter Example:

```bash
cd Code/OneParam/Octave

# Run the script
octave OneParam.m
```

#### Two Parameter Example:

```bash
cd Code/TwoParam/Octave

# Run the script
octave TwoParam.m
```

The Octave scripts read from `Data.txt` in the same directory and run a predefined number of iterations.

### Running R Statistics

```bash
cd Statistic
Rscript LinearRegression.R
```

The R script demonstrates statistical linear regression analysis including:
- Parameter estimation
- Standard error calculation
- Confidence intervals
- Visualization with residuals

## Data Format

Training data is stored in plain text files with space-separated values:

```
x1 y1
x2 y2
x3 y3
...
```

**Example (Data.txt):**
```
1 4
2 5
3 6
```

Where:
- First column: input feature (x)
- Second column: output value (y)

## Algorithm Details

### Hyperparameters

Both C++ and Octave implementations use the following hyperparameters:

- **Learning Rate (α):** 0.1
  - Controls the step size in gradient descent
  - Smaller values: slower convergence but more stable
  - Larger values: faster convergence but risk overshooting

- **Initial Parameters (θ):** 1.0
  - Starting values for all parameters
  - Can be initialized randomly or with heuristic values

### Training Process

1. **Initialize** parameters (θ) and learning rate (α)
2. **Load** training data from file
3. **Iterate** for specified number of steps:
   - Calculate cost function J(θ)
   - Compute gradients ∂J(θ)/∂θⱼ for each parameter
   - Update parameters: θⱼ := θⱼ - α · ∂J(θ)/∂θⱼ
   - Display current parameters and error
4. **Output** final parameters

### Output Format

During training, the programs display:
- Learning rate (α)
- Current parameter values (θ, θ₀, θ₁, etc.)
- Current error (cost function value)

**Example output:**
```
alpha = 0.1

theta_0 = 1.000000
theta_1 = 1.000000
error = 2.083333

theta_0 = 0.900000
theta_1 = 0.866667
error = 1.465972

...
```

## Documentation

Detailed mathematical documentation is available in LaTeX format:

- **OneParam**: `Documentation/OneParam/OneParam.tex`
  - Derivation of gradient for single parameter model
  - Cost function analysis
  - Convergence properties

- **TwoParam**: `Documentation/TwoParam/TwoParam.tex`
  - Partial derivatives for two-parameter model
  - Simultaneous parameter updates
  - Geometric interpretation

- **MultiParam**: `Documentation/MultiParam/MultiParam.tex`
  - Vector notation and matrix formulation
  - Generalized gradient descent
  - Feature scaling considerations

- **Derivative**: `Documentation/Derivative/Derivative.tex`
  - Mathematical foundations
  - Chain rule applications
  - Proof of convergence

### Building Documentation

```bash
cd Documentation/OneParam
make

# Or manually with pdflatex
pdflatex OneParam.tex
```

## Examples

### Example 1: Simple Linear Fit

**Data:**
```
1 4
2 5
3 6
```

**Initial parameters:** θ₀ = 1, θ₁ = 1, α = 0.1

**Expected result:** θ₀ ≈ 3, θ₁ ≈ 1 (line: y = 3 + x)

### Example 2: Proportional Relationship

For one-parameter model with data passing through origin:

**Data:**
```
1 2
2 4
3 6
```

**Expected result:** θ ≈ 2 (line: y = 2x)

## Key Features

- ✅ **Multiple Languages**: Compare C++ performance with Octave simplicity
- ✅ **Progressive Complexity**: Start simple, build understanding
- ✅ **Complete Documentation**: LaTeX documents with mathematical details
- ✅ **Gradient Descent**: From-scratch implementation
- ✅ **Configurable**: Adjustable learning rate and iterations
- ✅ **Visualization Support**: R script for statistical analysis

## Algorithm Complexity

- **Time Complexity**: O(m·n·k)
  - m: number of training examples
  - n: number of features
  - k: number of iterations

- **Space Complexity**: O(m·n)
  - Storage for training data
