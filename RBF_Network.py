#!/usr/bin/env python
# coding: utf-8

# In[452]:


# EE550 - Project 4
# Radial Basis Function (RBF) Networks

import random

import numpy as np
from matplotlib import pyplot as plt

np.random.seed(2020802018)
random.seed(40)

# Part 1-) Generating Data Point Pairs

n_of_samples = 100  # Defining number of overall samples
train_size = 70
# Producing x and y (combined version)
# Producing samples from uniform distribution [0,1]
x = np.random.rand(n_of_samples, 1)
x = np.sort(x, axis=0)
e = np.random.normal(loc=0.0, scale=0.2, size=n_of_samples).reshape(
    n_of_samples, 1
)  # Producing errors with zero mean variance 0.2
y = (
    0.4 * np.cos(np.radians(2 * 180 * x))
    + 0.6 * np.sin(np.radians(7 * 180 * x))
    + 0.5
    + e
)  # Producing y values


# Visualizing Data Point Pairs
plt.title(
    "Generated Data Points (y = 0.4 cos(2*pi*x) + 0.6 sin(7*pi*x) + 0.5 + e)"
)
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x, y)
plt.savefig("generated_all.png")
plt.show()


# In[453]:


# Part 2-) Splitting Dataset into Train and Test
test_size = n_of_samples - train_size
# Producing random index numbers for train set
train_indices = random.sample(range(0, n_of_samples - 1), train_size)
train_indices = np.sort(train_indices, axis=0)

x_train = x[train_indices]
y_train = y[train_indices]

x_test = np.delete(x, train_indices).reshape(test_size, 1)
y_test = np.delete(y, train_indices).reshape(test_size, 1)

# Visualizing Train Data Point Pairs
plt.title(
    "Train Data Points (y = 0.4 cos(2*pi*x) + 0.6 sin(7*pi*x) + 0.5 + e)"
)
plt.xlabel("Train x")
plt.ylabel("Train y")
plt.scatter(x_train, y_train)
plt.savefig("train_data.png")
plt.show()


# In[541]:


# Part 3-) Designing Architecture

# K-Means Algorithm for detecting mean values
# kmeans = KMeans(n_clusters=5, random_state=0).fit(x_train)
# means = kmeans.cluster_centers_
# # This "means" array tried for the RBF network but it gave poor results
# Therefore, means are set manually by visual inspection
# (on the extremum points of train data)
# Defining Gaussian Means
k = 5  # Number of Gaussians
m1 = 0.24  # Mean 1
m2 = 0.34  # Mean 2
m3 = 0.50  # Mean 3
m4 = 0.67  # Mean 4
m5 = 0.76  # Mean 5

means = np.array([m1, m2, m3, m4, m5])  # Final means array


# Defining RBF Functions
def rbf(x, mean, sd):
    return np.exp(-(1 / (2 * sd**2)) * (x - mean) ** 2)


# In[542]:


# Part 4.i-) Defining Forward Propagation and Update Rules
# Forward Propagation
def compute_rbf_activations(x, means, sigma):
    return np.exp(-((x - means.reshape(1, -1)) ** 2) / (2 * sigma**2)).T


def forward(X, W, b, std, means):
    fi1 = rbf(X, means[0], std)
    fi2 = rbf(X, means[1], std)
    fi3 = rbf(X, means[2], std)
    fi4 = rbf(X, means[3], std)
    fi5 = rbf(X, means[4], std)

    fi = np.array([fi1, fi2, fi3, fi4, fi5]).reshape(
        k, 1
    )  # fi values concatenation
    F = np.dot(W.T, fi) + b  # Calculating output
    return (fi, F)


# Update


def update(y, F, fi, learning_rate, W, b):
    # Calculating Loss
    error = y - F
    L = 0.5 * error**2
    # Weight Update
    delta = learning_rate * np.multiply(error, fi).flatten()
    W = W + delta.reshape(k, 1)
    # Bias Update
    delta_b = learning_rate * error
    b = b + delta_b
    return (W, L, b)


# In[599]:


# Part 4.ii-) Training the
# Initializing the weight values


# Initialization of Weights

W = np.random.uniform(low=-1, high=1, size=(k, 1))
b = np.random.uniform(low=-1, high=1, size=(1, 1))

# Selection of Standard Deviation
# sigma = (means.max() - means.min())/np.sqrt(2*k)
# This sigma value gave poor results.

# Final sigma value is obtained after several trials
# since it is a hyperparameter for the network
sigma = 0.08

# Training Algorithm - Gradient Descent
epochs = 500
n = 0.01

Loss_2 = np.empty((0, 1))  # Loss function results for each epoch

for epoc in range(epochs):
    combined = np.column_stack((x_train, y_train))
    np.random.shuffle(combined)  # Shuffling the dataset for each epoc
    Loss = np.empty((0, 1))  # Loss function results for each iteration
    for i in range(len(combined)):
        X = combined[i, 0].reshape(1, 1)
        Y = combined[i, 1].reshape(1, 1)
        fi, F = forward(X, W, b, sigma, means)
        Loss = np.append(Loss, 0.5 * (Y - F) ** 2)
        W, L, b = update(Y, F, fi, n, W, b)
    Loss_2 = np.append(Loss_2, Loss.mean())
    if (epoc + 1) % 100 == 0:
        plt.title("Loss Function Change per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.scatter(np.arange(len(Loss_2)), Loss_2)
        plt.show()

    print("Epoc =", epoc, "--", "Loss (MSE) =", Loss_2[len(Loss_2) - 1])


# In[600]:


# Part 4.iii-) Train Data Predictions

preds = np.empty((0, 1))  # Predictions array
for i in range(len(x_train)):
    preds = np.append(preds, forward(x_train[i], W, b, sigma, means)[1])

print("Final MSE for training: ", Loss_2[epochs - 1])
plt.title("Predictions on Train Data")
plt.plot(x_train, y_train, "-o", label="Ground Truth")
plt.plot(x_train, preds, "-o", label="Prediction")
plt.xlabel("x")
plt.ylabel("y, Prediction")
plt.legend()
plt.savefig("preds_train.png")
plt.show()


# In[601]:


# Part 5-) Testing the Model on Test Data
preds = np.empty((0, 1))  # Predictions array

for i in range(len(x_test)):
    preds = np.append(preds, forward(x_test[i], W, b, sigma, means)[1])

print(
    "MSE for test data: ",
    (0.5 * (preds.flatten() - y_test.flatten()) ** 2).mean(),
)
plt.title("Predictions on Test Data")
plt.plot(x_test, y_test, "-o", label="Ground Truth")
plt.plot(x_test, preds, "-o", label="Prediction")
plt.xlabel("x")
plt.ylabel("y, Prediction")
plt.legend()
plt.savefig("preds_test.png")
plt.show()


# In[602]:


# Part 6-) Visualizing the Cost Function

plt.title("Loss Function Change per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.scatter(np.arange(len(Loss_2)), Loss_2)
plt.savefig("loss.png")
plt.show()
