# Machine Learning Mini Projects

This repository contains a collection of small machine learning projects. All code is written in Python using NumPy, pandas, and Matplotlib. These are educational and experimental implementations.

## Projects

### 1. Recursive Least Squares (RLS) Regression

- Generates 15 noisy data points from a simple polynomial.
- Fits four models of increasing complexity using the RLS algorithm:
  - Model 1: constant  
  - Model 2: linear  
  - Model 3: quadratic  
  - Model 4: cubic
- Calculates error for each model.
- Plots each estimated curve.

### 2. Hopfield Network

- Creates 8x8 binary images for digits 1, 4, 7, and 9.
- Adds noise with three different noise levels.
- Trains a Hopfield network to memorize the original patterns.
- Tests recovery of original patterns from noisy inputs.
- Plots each iteration until convergence.

### 3. XOR Neural Network and Function Approximation

#### 3.1 XOR Problem

- Neural network with:
  - 2 input neurons  
  - 1 hidden layer (5 neurons)  
  - 1 output neuron
- Sigmoid activation, MSE loss.
- Trains with backpropagation to solve XOR.
- Plots training loss.

#### 3.2 Function Approximation

- Target: `y = sin(x) + 2*cos(x)`
- Neural network with:
  - 1 input  
  - 2 hidden layers (4 and 5 neurons)  
  - 1 output
- Uses tanh in hidden layers, linear output.
- Trains with MSE loss.
- Plots predictions and error.

#### 3.3 Handwritten Digit Recognition

- Uses optdigits dataset (CSV).
- Neural network with:
  - 64 input neurons  
  - 2 hidden layers (32 and 24 neurons)  
  - 10 output neurons (softmax)
- Cross-entropy loss.
- Classifies digits 0–9.
- Shows accuracy and predictions.

### 4. Radial Basis Function (RBF) Network

- 1D function: `y = 0.4*cos(2πx) + 0.6*sin(7πx) + 0.5 + noise`
- Uses 5 Gaussian basis functions (centers chosen manually).
- Trains using gradient descent.
- Plots predictions and loss.

## Requirements

- Python 3.x  
- NumPy  
- pandas  
- matplotlib  
- scikit-learn (only used for KMeans in RBF)

## How to Run

Clone the repo and run any `.py` or `.ipynb` file. Each file is self-contained.
