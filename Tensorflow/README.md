# TensorFlow Deep Learning Examples

A collection of deep learning implementations using TensorFlow and Keras, covering fundamental neural network architectures and their applications.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Neural Network Types](#neural-network-types)
  - [Artificial Neural Networks (ANN)](#artificial-neural-networks-ann)
  - [Convolutional Neural Networks (CNN)](#convolutional-neural-networks-cnn)
  - [Recurrent Neural Networks (RNN)](#recurrent-neural-networks-rnn)
  - [Autoencoders](#autoencoders)
  - [Generative Adversarial Networks (GAN)](#generative-adversarial-networks-gan)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This repository contains practical implementations of various deep learning architectures and techniques. Each subdirectory focuses on a specific type of neural network with hands-on examples using real datasets. The project is designed for learning and experimentation with modern deep learning approaches.

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Tensorflow
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install tensorflow==2.17.0
pip install pandas==2.2.3
pip install seaborn==0.13.2
pip install scikit-learn==1.5.2
pip install lxml==5.3.0
```

## 📁 Project Structure

```
Tensorflow/
│
├── ann/                          # Artificial Neural Networks
│   ├── classification/           # Classification tasks
│   │   └── simple_classification.ipynb
│   └── regression/              # Regression tasks
│       ├── exploratory_analysis_houses.ipynb
│       ├── regression_houses.ipynb
│       └── simple_regression.ipynb
│
├── cnn/                          # Convolutional Neural Networks
│   ├── cifar_cnn.ipynb
│   ├── mnist_cnn.ipynb
│   └── convolution_example.py
│
├── rnn/                          # Recurrent Neural Networks
│   └── simple_rnn.ipynb
│
├── autoencoder/                  # Autoencoders
│   ├── dim_reduction.ipynb
│   └── noise_filter.ipynb
│
├── gan/                          # Generative Adversarial Networks
│   └── simple_gan.ipynb
│
└── README.md                     # This file
```

## 🧠 Neural Network Types

### Artificial Neural Networks (ANN)

Basic feedforward neural networks for classification and regression tasks.

#### Classification (`ann/classification/`)

**simple_classification.ipynb**
- **Dataset**: Cancer classification dataset (569 samples, 30 features)
- **Task**: Binary classification (benign vs malignant tumors)
- **Key Features**:
  - Data preprocessing with MinMaxScaler
  - Three model variations:
    1. **Overfitting model**: Basic architecture without regularization
    2. **Early stopping model**: Using EarlyStopping callback
    3. **Dropout model**: Using Dropout layers (0.5 rate) for regularization
  - TensorBoard integration for training visualization
  - Performance metrics: classification report and confusion matrix
- **Architecture**:
  - Input layer: 30 features
  - Hidden layers: 30 → 15 neurons
  - Output layer: 1 neuron (sigmoid activation)
  - Loss: Binary crossentropy
  - Optimizer: Adam

#### Regression (`ann/regression/`)

**simple_regression.ipynb**
- **Dataset**: Fake regression dataset with 2 features
- **Task**: Price prediction
- **Key Features**:
  - Feature scaling with MinMaxScaler
  - Pairplot visualization for feature analysis
  - LeakyReLU activation (alpha=0.05)
  - Performance evaluation with MAE and MSE
- **Architecture**:
  - Input layer: 2 features
  - Hidden layers: 3 layers with 4 neurons each
  - Output layer: 1 neuron (linear)
  - Loss: Mean Squared Error (MSE)
  - Optimizer: RMSprop

**regression_houses.ipynb** & **exploratory_analysis_houses.ipynb**
- House price prediction with comprehensive exploratory data analysis
- Feature engineering and correlation analysis
- Advanced preprocessing techniques

### Convolutional Neural Networks (CNN)

Specialized neural networks for image processing and computer vision tasks.

#### MNIST Digit Classification (`cnn/mnist_cnn.ipynb`)

- **Dataset**: MNIST handwritten digits (60,000 training, 10,000 test images)
- **Task**: Multi-class classification (10 digits: 0-9)
- **Key Features**:
  - Image preprocessing and normalization (0-1 range)
  - One-hot encoding for labels
  - Data reshaping for CNN input (28×28×1)
  - EarlyStopping callback
  - Performance visualization (accuracy and loss curves)
- **Architecture**:
  - Conv2D layer: 32 filters, 4×4 kernel, ReLU activation
  - MaxPooling2D: 2×2 pool size
  - Flatten layer
  - Dense layer: 128 neurons, ReLU activation
  - Output layer: 10 neurons, softmax activation
  - Loss: Categorical crossentropy
  - Optimizer: Adam

#### CIFAR-10 Classification (`cnn/cifar_cnn.ipynb`)

- **Dataset**: CIFAR-10 (colored images, 10 object classes)
- **Task**: Multi-class classification of real-world objects
- More complex CNN architecture for colored images

#### Convolution Example (`cnn/convolution_example.py`)

- **Purpose**: Demonstrates convolution operations on images
- **Key Concepts**:
  - Manual convolution using scipy's `convolve2d`
  - Face and eye image processing
  - Visualization of convolution results
  - Understanding kernel operations

### Recurrent Neural Networks (RNN)

Neural networks designed for sequential data and time series.

#### Simple RNN (`rnn/simple_rnn.ipynb`)

- **Dataset**: Synthetic sine wave data (501 points)
- **Task**: Time series prediction
- **Key Features**:
  - TimeseriesGenerator for sequence generation
  - Window size: 64 time steps
  - MinMaxScaler for data normalization
  - Train/test split with proper sequencing
  - Visualization of predictions vs actual values
- **Architecture**:
  - SimpleRNN layer: 64 units
  - Dense output layer: 1 neuron
  - Loss: Mean Squared Error (MSE)
  - Optimizer: Adam
- **Use Cases**:
  - Stock price prediction
  - Weather forecasting
  - Signal processing

### Autoencoders

Unsupervised learning for dimensionality reduction and feature learning.

#### Dimensionality Reduction (`autoencoder/dim_reduction.ipynb`)

- **Dataset**: MNIST digits
- **Task**: Compress 784-dimensional images to 25 dimensions and reconstruct
- **Key Features**:
  - Encoder-decoder architecture
  - Bottleneck layer for compressed representation
  - Image reconstruction quality assessment
- **Architecture**:
  - **Encoder**: 784 → 400 → 200 → 100 → 50 → 25 (ReLU activation)
  - **Decoder**: 25 → 50 → 100 → 200 → 400 → 784 (ReLU/Sigmoid)
  - Loss: Binary crossentropy
  - Optimizer: SGD with learning rate 1.5
- **Applications**:
  - Data compression
  - Feature extraction
  - Anomaly detection

#### Noise Filtering (`autoencoder/noise_filter.ipynb`)

- Denoising autoencoder for removing noise from images
- Training on corrupted data to learn robust representations

### Generative Adversarial Networks (GAN)

Two neural networks competing to generate realistic synthetic data.

#### Simple GAN (`gan/simple_gan.ipynb`)

- **Dataset**: MNIST zeros (single digit generation)
- **Task**: Generate synthetic images of the digit "0"
- **Key Features**:
  - Generator and discriminator networks
  - Adversarial training process
  - Random noise to image generation
  - TensorFlow Dataset API for efficient batching
- **Architecture**:
  - **Generator**:
    - Input: 100-dimensional noise vector
    - Hidden layers: 100 → 150 → 784 neurons (ReLU)
    - Output: 28×28 image (reshaped)
  - **Discriminator**:
    - Input: 28×28 image (flattened)
    - Hidden layers: 150 → 100 neurons (ReLU)
    - Output: 1 neuron (sigmoid) - real vs fake classification
  - Loss: Binary crossentropy
  - Optimizer: Adam
- **Training Process**:
  - Phase 1: Train discriminator on real and generated images
  - Phase 2: Train generator to fool discriminator
  - Alternating training for both networks

## 💻 Usage

### Running Jupyter Notebooks

1. Start Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
# or
jupyter lab
```

2. Navigate to the desired directory and open the notebook

3. Run cells sequentially to execute the code

### Running Python Scripts

For standalone Python scripts like `convolution_example.py`:

```bash
cd cnn/
python convolution_example.py
```

### TensorBoard Visualization

Some notebooks include TensorBoard logging. To view training metrics:

```bash
tensorboard --logdir=logs/fit
```

Then open `http://localhost:6006` in your browser.

## 🔧 Key Techniques Demonstrated

### Regularization Techniques
- **Dropout**: Randomly dropping neurons during training to prevent overfitting
- **Early Stopping**: Stopping training when validation loss stops improving
- **Batch Normalization**: Normalizing layer inputs for faster convergence

### Data Preprocessing
- **MinMaxScaler**: Scaling features to [0, 1] range
- **StandardScaler**: Standardizing features with zero mean and unit variance
- **One-hot Encoding**: Converting categorical labels to binary vectors
- **Data Augmentation**: Creating variations of training data

### Callbacks
- **EarlyStopping**: Prevent overfitting by monitoring validation metrics
- **TensorBoard**: Real-time training visualization and monitoring
- **ModelCheckpoint**: Save best models during training
- **ReduceLROnPlateau**: Adaptive learning rate adjustment

### Model Evaluation
- **Classification Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
- **Regression Metrics**: MAE, MSE, RMSE, R² score
- **Visualization**: Loss curves, accuracy plots, prediction vs actual plots

## 📊 Datasets Used

- **MNIST**: 28×28 grayscale images of handwritten digits (0-9)
- **CIFAR-10**: 32×32 colored images of 10 object classes
- **Cancer Classification**: Wisconsin breast cancer dataset
- **House Prices**: Real estate price prediction dataset
- **Synthetic Data**: Custom-generated data for specific demonstrations

## 🛠️ Dependencies

- **TensorFlow 2.17.0**: Core deep learning framework
- **Pandas 2.2.3**: Data manipulation and analysis
- **Seaborn 0.13.2**: Statistical data visualization
- **Scikit-learn 1.5.2**: Machine learning utilities and metrics
- **NumPy**: Numerical computing (dependency of above libraries)
- **Matplotlib**: Plotting and visualization (dependency of above libraries)
- **lxml 5.3.0**: XML and HTML parsing

## 📚 Learning Path

Recommended order for beginners:

1. **Start with ANNs**:
   - `ann/regression/simple_regression.ipynb` - understand basic neural networks
   - `ann/classification/simple_classification.ipynb` - learn classification and regularization

2. **Move to CNNs**:
   - `cnn/convolution_example.py` - understand convolution operations
   - `cnn/mnist_cnn.ipynb` - build your first CNN

3. **Explore RNNs**:
   - `rnn/simple_rnn.ipynb` - learn sequential data processing

4. **Advanced Topics**:
   - `autoencoder/dim_reduction.ipynb` - unsupervised learning
   - `gan/simple_gan.ipynb` - generative models

## 🎯 Best Practices Demonstrated

1. **Always split data**: Train/validation/test sets for proper evaluation
2. **Scale your features**: Normalize inputs for better convergence
3. **Monitor overfitting**: Use validation data and regularization techniques
4. **Start simple**: Begin with simple architectures and increase complexity
5. **Visualize everything**: Plot loss curves, predictions, and data distributions
6. **Use callbacks**: Implement early stopping and model checkpointing
7. **Proper activation functions**: ReLU for hidden layers, softmax/sigmoid for output
8. **Choose appropriate loss functions**: Match loss to your task type
