# Deep Neural Network (DNN) Implementation

A lightweight implementation of a Deep Neural Network from scratch in both **C++** and **Python**. This project demonstrates the core concepts of neural networks including forward propagation, backpropagation, and gradient descent without relying on high-level deep learning frameworks.

## 🎯 Features

- **Dual Implementation**: Complete implementations in both C++ and Python for performance comparison and educational purposes
- **Backpropagation Algorithm**: Full implementation of the backpropagation algorithm with gradient descent
- **Configurable Architecture**: Support for arbitrary network architectures (custom number of layers and neurons)
- **Activation Functions**: Sigmoid activation function with derivative support
- **Loss Functions**: Mean Squared Error (MSE) / Least Squares loss function
- **Training Features**:
  - Adjustable learning rate
  - Mini-batch gradient descent (Python)
  - Model persistence (save/load trained models in C++)
- **Visualization**: Real-time training progress plotting (Python)

## 📁 Project Structure

```
Dnn/
├── Cpp/                    # C++ Implementation
│   ├── NNet.h             # Neural network class header
│   ├── NNet.cpp           # Neural network implementation
│   ├── Functions.h        # Activation and loss functions header
│   ├── Functions.cpp      # Activation and loss functions implementation
│   ├── Main.cpp           # Example usage and training
│   ├── Makefile           # Build configuration
│   ├── Doc/               # LaTeX documentation
│   │   ├── Main.tex
│   │   ├── NNet.tex
│   │   ├── DataStruct.tex
│   │   ├── Formulas.tex
│   │   └── Example.tex
│   └── Plot/
│       └── Plotter.py     # Real-time training visualization
├── Python/                 # Python Implementation
│   ├── NNet.py            # Neural network class
│   ├── Functions.py       # Utility functions
│   └── Execute.py         # Example with heart disease dataset
└── README.md              # This file
```

## 🔧 Requirements

### C++ Version
- **Compiler**: g++ with C++11 support or later
- **Build Tool**: Make
- **Optional**: Python 3.x with matplotlib (for visualization)

### Python Version
- **Python**: 3.6 or later
- **Libraries**:
  - `numpy` - Numerical computing
  - `pandas` - Data manipulation (for examples)
  - `matplotlib` - Visualization
  - `scikit-learn` - Data splitting and utilities

## 📦 Installation

### C++ Setup

1. Navigate to the C++ directory:
```bash
cd Cpp
```

2. Build the project:
```bash
make
```

3. Run the executable:
```bash
make exe
# or
./Main
```

4. Clean build files:
```bash
make clean
```

### Python Setup

1. Install required packages:
```bash
pip install numpy pandas matplotlib scikit-learn
```

2. Navigate to the Python directory:
```bash
cd Python
```

3. Run the example:
```bash
python Execute.py
```

## 🚀 Usage

### C++ Example

```cpp
#include "NNet.h"

// Define network architecture: [input_size, hidden_size, output_size]
int num_layers = 3;
int layers[num_layers] = {2, 2, 2};  // 2 inputs, 2 hidden, 2 outputs

// Create network
NNet* nnet = new NNet(layers, num_layers);

// Prepare input (with bias)
double input[3] = {0.05, 0.10, 1};  // Two inputs plus bias

// Initialize weights
double weights_1[6] = {0.15, 0.20, 0.35,
                       0.25, 0.30, 0.35};
double weights_2[6] = {0.40, 0.45, 0.60,
                       0.50, 0.55, 0.60};

// Configure network
nnet->set_input(input);
nnet->set_weights(weights_1, 0);
nnet->set_weights(weights_2, 1);
nnet->set_activations(new Sigmoid(), 1);
nnet->set_activations(new Sigmoid(), 2);
nnet->set_loss(new LessSquare());
nnet->set_labels(labels);
nnet->set_learning_rate(0.5);

// Training loop
for(int i = 0; i < 10000; i++) {
    double* ans = nnet->forward();
    double loss = nnet->loss(ans);

    std::cout << "Epoch " << i << ", Loss: " << loss << std::endl;

    nnet->backward();
    nnet->update_weights();
}

// Save trained model
nnet->save("my_net.net");

// Load a saved model
// NNet* loaded_nnet = new NNet("my_net.net");

delete nnet;
```

### Python Example

```python
from NNet import NNet
from Functions import *
import numpy as np

# Define architecture: [input_size, hidden_layer1, hidden_layer2, output_size]
arch = [5, 5, 5, 2]  # 5 inputs, two hidden layers with 5 neurons each, 2 outputs

# Create network
nnet = NNet(arch)

# Initialize random weights
weights = nnet.init_random_weights()

# Training data
input_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Example input
label = np.array([0, 1])  # Example label (one-hot encoded)

# Training parameters
learning_rate = 0.2
epochs = 1000

# Training loop
for epoch in range(epochs):
    error, weights = nnet.train(input_data, label, weights, learning_rate)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Error: {error}")

# Make predictions
prediction = nnet.predict(input_data, weights)
print(f"Prediction: {prediction}")
```

### Complete Example with Dataset (Python)

The `Execute.py` file demonstrates training on a real dataset (heart disease prediction):

```python
# Load and preprocess data
dataset = pd.read_csv("heart.csv")
x = dataset[["age", "trtbps", "chol", "thalachh", "oldpeak"]]
y = dataset["output"]
x, y = normalize(x.values), classes2binary(y.values)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Train network
nnet, weights = train(
    epoch=500,
    tol=1e-2,
    arch=[5, 5, 5, 2],
    eta=0.2,
    x=x_train,
    y=y_train,
    batch_size=100
)

# Evaluate
y_pred = predict(nnet, weights, x_test)
y_true = binary2class(y_test)
plot_result(y_true, y_pred)
```

## 📊 Visualization

### Real-time Training (C++)

The C++ version can pipe its output to a Python plotter for real-time visualization:

```bash
cd Cpp
./Main | python Plot/Plotter.py
```

This will display a live plot of the training error over epochs.

## 🔬 Reference

### C++

#### `NNet` Class

**Constructors:**
- `NNet(int* layers, int num_layers)` - Create network from architecture array
- `NNet(char const* file_name)` - Load network from file

**Configuration Methods:**
- `void set_input(double* x)` - Set input data
- `void set_weights(double* w, int layer)` - Initialize weights for a layer
- `void set_activations(Activation* activation, int layer)` - Set activation function
- `void set_loss(Loss* loss)` - Set loss function
- `void set_labels(double* y)` - Set target labels
- `void set_learning_rate(double alpha)` - Set learning rate

**Training Methods:**
- `double* forward()` - Perform forward propagation
- `double loss(double* y_hat)` - Calculate loss
- `void backward()` - Perform backpropagation
- `void update_weights()` - Update weights using gradients

**Utility Methods:**
- `void save(char const* file_name)` - Save model to file
- `double* get_weights(int layer)` - Get weights for a layer

**Activation Functions:**
- `Sigmoid` - Sigmoid activation: σ(x) = 1/(1 + e^(-x))

**Loss Functions:**
- `LessSquare` - Mean Squared Error: L = 1/2 * Σ(y - ŷ)²

### Python

#### `NNet` Class

**Constructor:**
- `__init__(self, arch)` - Initialize with architecture list

**Methods:**
- `predict(self, data, weights)` - Make prediction (forward pass)
- `test_prediction(self, data, weights)` - Forward pass with intermediate values saved
- `train(self, data, label, weights, eta)` - Perform one training step (forward + backward)
- `init_random_weights(self, low=0, high=1)` - Initialize random weights

**Utility Functions (Functions.py):**
- `sigmoid(x)` - Sigmoid activation
- `d_sigmoid(x)` - Sigmoid derivative
- `mse(y, y_bar)` - Mean squared error
- `d_mse(y, y_bar)` - MSE derivative
- `normalize(x)` - Z-score normalization
- `classes2binary(y)` - Convert class labels to one-hot encoding
- `binary2class(y)` - Convert one-hot encoding to class labels
- `hard_classification(y_hat, threshold)` - Apply threshold to predictions
- `predict(nnet, weights, x)` - Batch prediction
- `plot_result(y_true, y_pred)` - Visualize predictions vs ground truth
- `get_batch(x, y, batch_size)` - Sample mini-batch

## 🧮 Mathematical Foundation

### Forward Propagation

For each layer l:

1. **Weighted sum**: z^(l) = W^(l) · a^(l-1) + b^(l)
2. **Activation**: a^(l) = σ(z^(l))

Where:
- W^(l) = weight matrix for layer l
- a^(l-1) = activations from previous layer
- b^(l) = bias vector
- σ = activation function (sigmoid)

### Backpropagation

The algorithm computes gradients layer by layer:

1. **Output layer error**: δ^(L) = (a^(L) - y) ⊙ σ'(z^(L))
2. **Hidden layer error**: δ^(l) = (W^(l+1))^T · δ^(l+1) ⊙ σ'(z^(l))
3. **Weight gradient**: ∂E/∂W^(l) = δ^(l) · (a^(l-1))^T
4. **Weight update**: W^(l) ← W^(l) - η · ∂E/∂W^(l)

Where:
- ⊙ = element-wise multiplication
- η = learning rate
- E = error/loss function

### Loss Function

**Mean Squared Error (MSE)**:

E = 1/2 · Σᵢ(yᵢ - ŷᵢ)²

**Derivative**:

∂E/∂ŷ = ŷ - y

## 📚 Implementation Details

### C++ Implementation

- **Memory Management**: Manual memory allocation and deallocation
- **Data Structures**: Arrays and pointers for efficiency
- **Weights Format**: Flattened arrays with bias included
- **Forward/Backward Buffers**: Separate structures (f, b) for intermediate values
- **Extensibility**: Factory pattern for activation and loss functions

### Python Implementation

- **NumPy Arrays**: Leverages NumPy for vectorized operations
- **Weight Format**: List of matrices (one per layer)
- **Bias Handling**: Appended to input vectors dynamically
- **Mini-batch Support**: Built-in batch sampling
- **Code Comments**: Extensive Spanish comments explaining the algorithm

## 🎓 Educational Resources

This implementation is based on the classic backpropagation example:
- [A Step-by-Step Backpropagation Example by Matt Mazur](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)

The LaTeX documentation in `Cpp/Doc/` provides detailed mathematical explanations:
- **Main.tex**: Main documentation structure
- **NNet.tex**: Network architecture explanation
- **DataStruct.tex**: Data structure details
- **Formulas.tex**: Mathematical formulas
- **Example.tex**: Worked examples

To compile the documentation:
```bash
cd Cpp/Doc
make
```
