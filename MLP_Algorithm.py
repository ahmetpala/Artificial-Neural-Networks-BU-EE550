#!/usr/bin/env python
# coding: utf-8

# EE550 - Project 3
# Question 1 (XOR Problem)
# 1.i) Generating XOR dataset
from random import randrange
import numpy as np
from matplotlib import pyplot as plt

# Defining the necessary functions


def xor(a, b):
    return (a+b-2*a*b)


def sigmoid(x):  # Sigmoid Function
    return 1 / (1 + np.exp(-x))


def derive_sigmoid(array):  # Derivation of natural logarithm
    n_array = np.copy(array)
    return np.multiply(sigmoid(n_array), (1 - sigmoid(n_array)))


def derive_tanh(array):
    n_array = np.copy(array)
    return 1 - np.tanh(array)**2


data_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Generating XOR dataset
labels_xor = xor(data_xor[:, 0], data_xor[:, 1]).reshape(
    len(data_xor), 1)  # Calculating Labels


# Network Architecture: 2 input neurons, 1 hidden layer with 5 hidden neurons, and 1 output neuron
# Activation function is sigmoid in the hidden layer and output layer
# MSE Loss Function


# Defining forward and back-propagation functions for XOR problem
def forward_propagation(x, W1, b1, W2, b2):
    # Forward Propagation
    # Layer 1 (Hidden Layer)
    s1 = np.dot(W1, x) + b1
    o1 = sigmoid(s1)
    # Layer 2 (Output Layer)
    s2 = np.dot(W2, o1) + b2
    o2 = sigmoid(s2)
    y_hat = o2
    return (o1, o2, s1, s2, y_hat)


def back_propagation(o1, o2, s1, s2, y_hat, y, n, W1, W2, b1, b2):
    # Back-Propagation
    L = 0.5*(y-y_hat)**2  # Calculating Loss function (MSE)
    g = y_hat - y
    # Layer 2 (Output Layer)
    g = np.multiply(g, derive_sigmoid(s2))
    delta_b2 = n*g
    delta_W2 = n*np.dot(g, o1.T)
    g = np.dot(W2.T, g)
    # Layer 1 (Hidden Layer)
    g = np.multiply(g, derive_sigmoid(s1))
    delta_b1 = n*g
    delta_W1 = n*np.dot(g, x.T)
    # Updates
    W1 = W1 - delta_W1
    b1 = b1 - delta_b1
    W2 = W2 - delta_W2
    b2 = b2 - delta_b2
    return (W1, W2, b1, b2, L)


# The Network Architecture with 10000 epoches and learning rate 0.09
# Initialization
np.random.seed(2020802018)
# Defining number of neurons in each layer
nof_l1 = 5  # Layer 1 (Hidden layer)
# Initialization
W1 = np.random.rand(nof_l1, 2)
W2 = np.random.rand(1, nof_l1)
b1 = np.random.rand(nof_l1, 1)
b2 = np.random.rand(1, 1)

n = 0.09  # Defining the learning rate
epoches = 20000  # Number of epoches for training algorithm

Loss = np.empty((0, 1))  # Loss function results for each iteration
Train_Acc = np.empty((0, 1))

for epoc in range(epoches):
    # n = n/(epoc+1)
    combined = np.column_stack((data_xor, labels_xor))
    np.random.shuffle(combined)  # Shuffling the dataset for each epoc
    for i in range(len(data_xor)):
        x = combined[i, [0, 1]].reshape(2, 1)
        y = combined[i, 2]
        o1, o2, s1, s2, y_hat = forward_propagation(x, W1, b1, W2, b2)
        W1, W2, b1, b2, L = back_propagation(
            o1, o2, s1, s2, y_hat, y, n, W1, W2, b1, b2)
    Loss = np.append(Loss, ((forward_propagation(data_xor.T, W1, b1, W2, b2)[
                     4].reshape(4, 1)-labels_xor)**2).sum())
    print("Epoc =", epoc, "--", "Loss (MSE) =", Loss[len(Loss)-1])
    if epoc >= 2500:  # Defining minimum numbers of epoches
        if (abs(Loss[len(Loss)-1] - Loss[len(Loss)-2])/Loss[len(Loss)-2])*100 <= 0.01:  # Threshold = 0.01 %
            break


# 1.ii) Plotting the Loss Function
# Plotting Loss Function
plt.title("Loss Function")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.plot(Loss)
plt.savefig('Q1_Loss.png')
plt.show()
print("Number of epoches =", len(Loss))


# 1.iii) Testing the model with original inputs

print("Original predictions", forward_propagation(
    data_xor.T, W1, b1, W2, b2)[4])  # Original predictions

print("Estimations (Rounded)", np.round(
    forward_propagation(data_xor.T, W1, b1, W2, b2)[4]).flatten())
print("True Labels", labels_xor.T.flatten())


# Question 2 (Function Approximation)
np.random.seed(2020802018)

# Producing train and test data
x_train = np.radians(np.random.rand(200, 1)*360)
y_train = np.sin(x_train) + 2*np.cos(x_train)

x_test = np.radians(np.random.rand(25, 1)*360)
y_test = np.sin(x_test) + 2*np.cos(x_test)

# Plotting the train dataset consisting of 60 data points
plt.title("Train Data (y = sin(x) + 2*cos(x))")
plt.xlabel("x (radians)")
plt.ylabel("y")
plt.scatter(x_train, y_train)
plt.savefig('Q2_train_data.png')
plt.show()


# Network Architecture: 1 input neuron, 4 and 5 hidden neurons in 1st and 2nd hidden layers, and 1 output neuron in output layer
# Activation function is tanh in the hidden layers and 2*tanh in output layer
# MSE Loss Function

# Defining forward and back-propagation functions for the function approximation problem
def forward_propagation_2(x, W1, b1, W2, b2, W3, b3):
    # Forward Propagation
    # Layer 1 (1st Hidden Layer)
    s1 = np.dot(W1, x) + b1
    o1 = np.tanh(s1)
    # Layer 2 (2nd Hidden Layer)
    s2 = np.dot(W2, o1) + b2
    o2 = np.tanh(s2)
    # Layer 3 (Output Layer)
    s3 = np.dot(W3, o2) + b3
    o3 = s3  # Linear Activation function
    y_hat = o3
    return (o1, o2, o3, s1, s2, s3, y_hat)


def back_propagation_2(o1, o2, o3, s1, s2, s3, y_hat, y, n, W1, W2, W3, b1, b2, b3):
    # Back-Propagation
    L = 0.5*(y-y_hat)**2  # Calculating Loss function (MSE)
    g = y_hat - y
    # Layer 3 (Output Layer)
    # g = np.multiply(g,3*derive_tanh(s3))
    delta_b3 = n*g
    delta_W3 = n*np.dot(g, o2.T)
    g = np.dot(W3.T, g)
    # Layer 2 (2nd Hidden Layer)
    g = np.multiply(g, derive_tanh(s2))
    delta_b2 = n*g
    delta_W2 = n*np.dot(g, o1.T)
    g = np.dot(W2.T, g)
    # Layer 1 (1st Hidden Layer)
    g = np.multiply(g, derive_tanh(s1))
    delta_b1 = n*g
    delta_W1 = n*np.dot(g, x.T)
    # Updates
    W1 = W1 - delta_W1
    b1 = b1 - delta_b1
    W2 = W2 - delta_W2
    b2 = b2 - delta_b2
    W3 = W3 - delta_W3
    b3 = b3 - delta_b3
    return (W1, W2, W3, b1, b2, b3, L)


# The Network Architecture with 2000 epoches (maximum) and learning rate 0.04 (after some trial)
# Initialization
np.random.seed(2020802018)
# Defining number of neurons in each layer
nof_l1 = 4  # Layer 1 (1st hidden layer)
nof_l2 = 5  # Layer 2 (2nd hidden layer)
# Initialization
W1 = np.random.rand(nof_l1, 1)
W2 = np.random.rand(nof_l2, nof_l1)
W3 = np.random.rand(1, nof_l2)
b1 = np.random.rand(nof_l1, 1)
b2 = np.random.rand(nof_l2, 1)
b3 = np.random.rand(1, 1)

n = 0.01  # Defining the learning rate
epoches = 2000  # Number of epoches for training algorithm

Loss_2 = np.empty((0, 1))  # Loss function results for each iteration

for epoc in range(epoches):
    # n = n/(epoc+1)
    combined = np.column_stack((x_train, y_train))
    np.random.shuffle(combined)  # Shuffling the dataset for each epoc
    for i in range(len(combined)):
        x = combined[i, 0].reshape(1, 1)
        y = combined[i, 1].reshape(1, 1)
        o1, o2, o3, s1, s2, s3, y_hat = forward_propagation_2(
            x, W1, b1, W2, b2, W3, b3)
        W1, W2, W3, b1, b2, b3, L = back_propagation_2(
            o1, o2, o3, s1, s2, s3, y_hat, y, n, W1, W2, W3, b1, b2, b3)
    Loss_2 = np.append(Loss_2, ((forward_propagation_2(
        x_train.T, W1, b1, W2, b2, W3, b3)[6].reshape(len(x_train), 1)-y_train)**2).sum())
    print("Epoc =", epoc, "--", "Loss (MSE) =", Loss_2[len(Loss_2)-1])
    if epoc >= 200:  # defining minimum number of epoches
        if (abs(Loss_2[len(Loss_2)-1] - Loss_2[len(Loss_2)-2])/Loss_2[len(Loss_2)-2])*100 <= 0.5:  # Threshold = 0.1 %
            break


# 2.i) Plotting the Loss Function
# Plotting Loss Function
plt.title("Loss Function")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.plot(Loss_2)
plt.savefig('Q2_Loss.png')
plt.show()
print("Number of epoches =", len(Loss_2))


# Plotting the train dataset consisting of 60 data points
predictions = forward_propagation_2(x_train.T, W1, b1, W2, b2, W3, b3)[
    6]  # Predictions for train data
plt.title("Predictions and Ground Truth Values on Train Data")
plt.xlabel("x (radians)")
plt.ylabel("y, Predictions")
plt.scatter(x_train, predictions.T, label="Predictions")
plt.scatter(x_train, y_train, label="Ground Truth values")
plt.legend(loc='upper center')
plt.savefig('Q2_Predictions_Train.png')
plt.show()


# 2.ii) Testing the model on test data
predictions = forward_propagation_2(x_test.T, W1, b1, W2, b2, W3, b3)[
    6]  # Predictions for train data
print("Test data", "Predictions for test data")
print(np.column_stack((x_test, y_test, predictions.T)))


# 2.iii) Plotting the predictions vs ground truth values on test data
plt.title("Predictions and Ground Truth Values on Test Data")
plt.xlabel("x (radians)")
plt.ylabel("y, Predictions")
plt.scatter(x_test, predictions.T, label="Predictions")
plt.scatter(x_test, y_test, label="Ground Truth values")
plt.legend(loc='upper center')
plt.savefig('Q2_Predictions_Test.png')
plt.show()


# Question 3 Recognition of Handwritten Digits
# Reading the datasets

my_data = np.genfromtxt('Q3_datasets/optdigits_test.csv', delimiter=',')


# Stratified Sampling
np.random.seed(2020802018)  # Setting seed
train_data = np.empty((0, my_data.shape[1]))  # Initial empty train array
test_data = np.empty((0, my_data.shape[1]))  # Initial empty test array

# Randomly stratified sampling
for k in range(10):
    deneme = my_data[my_data[:, 64] == k]  # Selecting each class
    random_indices = np.random.choice(
        # Selecting randomly 120 rows
        deneme.shape[0], size=120, replace=False)
    # Adding first 100 rows to train data for each class
    train_data = np.append(train_data, deneme[random_indices][:100, :], axis=0)
    # Adding last 20 rows to test data for each class
    test_data = np.append(test_data, deneme[random_indices][-20:, :], axis=0)

# Arranging train and test digits(x) and targets(y)
train_y = train_data[:, 64].reshape(len(train_data), 1)
train_x = np.delete(train_data, 64, axis=1)/16  # normalization

test_y = test_data[:, 64].reshape(len(test_data), 1)
test_x = np.delete(test_data, 64, axis=1)/16  # normalization


# Example patterns
plt.subplot(251)
plt.imshow(test_x[1, :].reshape(8, 8), cmap='gray')
plt.subplot(252)
plt.imshow(test_x[21, :].reshape(8, 8), cmap='gray')
plt.subplot(253)
plt.imshow(test_x[41, :].reshape(8, 8), cmap='gray')
plt.title("Example Patterns")
plt.subplot(254)
plt.imshow(test_x[61, :].reshape(8, 8), cmap='gray')
plt.subplot(255)
plt.imshow(test_x[81, :].reshape(8, 8), cmap='gray')
plt.subplot(256)
plt.imshow(test_x[101, :].reshape(8, 8), cmap='gray')
plt.subplot(257)
plt.imshow(test_x[121, :].reshape(8, 8), cmap='gray')
plt.subplot(258)
plt.imshow(test_x[141, :].reshape(8, 8), cmap='gray')
plt.subplot(259)
plt.imshow(test_x[161, :].reshape(8, 8), cmap='gray')
plt.subplot(2, 5, 10)
plt.imshow(test_x[181, :].reshape(8, 8), cmap='gray')
plt.savefig('Q3_Example_Patterns.png')
plt.show()


# Defining necessary functions

def one_hot(indice):  # One-hot representation function
    array = np.zeros(shape=(10, 1))
    for i in range(len(array)):
        array[indice, 0] = 1
    return array


def softmax(array):  # Softmax function
    n_array = np.copy(array)
    return (np.exp(n_array) / sum(np.exp(n_array)))


def relu(x):
    return np.maximum(0, x)


def derive_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


# Network Architecture: 64 input neurons, 32 and 24 hidden neurons in 1st and 2nd hidden layers, and 10 output neurons in output layer
# Activation function is tanh in the hidden layers and softmax in output layer
# Cross-Entrophy Loss Function

# Defining forward and back-propagation functions for the Recognition of Handwritten Digits
def forward_propagation_3(x, W1, b1, W2, b2, W3, b3):
    # Forward Propagation
    # Layer 1 (1st Hidden Layer)
    s1 = np.dot(W1, x) + b1
    o1 = np.tanh(s1)
    # Layer 2 (2nd Hidden Layer)
    s2 = np.dot(W2, o1) + b2
    o2 = np.tanh(s2)
    # Layer 3 (Output Layer)
    s3 = np.dot(W3, o2) + b3
    o3 = softmax(s3)
    y_hat = o3
    return (o1, o2, o3, s1, s2, s3, y_hat)


def back_propagation_3(o1, o2, o3, s1, s2, s3, y_hat, y, n, W1, W2, W3, b1, b2, b3):
    # Back-Propagation
    L = -np.dot(y.T, np.log(y_hat))  # Calculating Loss function Cross Entropy
    g = y_hat - y
    # Layer 3 (Output Layer)
    delta_b3 = n*g
    delta_W3 = n*np.dot(g, o2.T)
    g = np.dot(W3.T, g)
    # Layer 2 (2nd Hidden Layer)
    g = np.multiply(g, derive_tanh(s2))
    delta_b2 = n*g
    delta_W2 = n*np.dot(g, o1.T)
    g = np.dot(W2.T, g)
    # Layer 1 (1st Hidden Layer)
    g = np.multiply(g, derive_tanh(s1))
    delta_b1 = n*g
    delta_W1 = n*np.dot(g, x.T)
    # Updates
    W1 = W1 - delta_W1
    b1 = b1 - delta_b1
    W2 = W2 - delta_W2
    b2 = b2 - delta_b2
    W3 = W3 - delta_W3
    b3 = b3 - delta_b3
    return (W1, W2, W3, b1, b2, b3, L)

# Accuracy calculation function


def accuracy(x_ar, y_ar, W1, b1, W2, b2, W3, b3):
    return ((y_ar.astype('int').flatten() == np.argmax(forward_propagation_3(x_ar.T, W1, b1, W2, b2, W3, b3)[6], axis=0)).sum() / len(y_ar))


# The Network Architecture with 200 epoches (maximum) and learning rate 0.001 (after some trial)
# Initialization (weight and bias values in [-1,1])
np.random.seed(2020802018)
W1 = np.random.uniform(low=-1, high=1, size=(32, 64))
W2 = np.random.uniform(low=-1, high=1, size=(24, 32))
W3 = np.random.uniform(low=-1, high=1, size=(10, 24))
b1 = np.random.uniform(low=-1, high=1, size=(32, 1))
b2 = np.random.uniform(low=-1, high=1, size=(24, 1))
b3 = np.random.uniform(low=-1, high=1, size=(10, 1))


n = 0.001  # Defining the learning rate
epoches = 200  # Number of epoches for training algorithm

Loss_3 = np.empty((0, 1))  # Loss function results for each iteration
Accuracy = np.empty((0, 1))  # Loss function results for each iteration

for epoc in range(epoches):
    # n = n/(epoc+1) # Learning rate decay (if needed)
    combined = np.copy(train_data)
    np.random.shuffle(combined)  # Shuffling the dataset for each epoc
    y_ar = combined[:, 64].reshape(len(combined), 1)
    x_ar = np.delete(combined, 64, axis=1)/16  # normalization
    # Producing one-hot representations of labels
    hot_train_targets = np.zeros(shape=(10, len(y_ar)))
    for i in range(hot_train_targets.shape[1]):
        hot_train_targets[:, i] = one_hot(y_ar.astype('int')[i]).reshape(10)
    for j in range(len(y_ar)):
        x = x_ar[j, :].reshape(64, 1)
        y = hot_train_targets[:, j].reshape(10, 1)
        o1, o2, o3, s1, s2, s3, y_hat = forward_propagation_3(
            x, W1, b1, W2, b2, W3, b3)
        W1, W2, W3, b1, b2, b3, L = back_propagation_3(
            o1, o2, o3, s1, s2, s3, y_hat, y, n, W1, W2, W3, b1, b2, b3)
    Loss_3 = np.append(Loss_3, np.mean(-np.multiply(hot_train_targets, np.log(
        forward_propagation_3(x_ar.T, W1, b1, W2, b2, W3, b3)[6])).sum(axis=0)))
    Accuracy = np.append(Accuracy, accuracy(
        train_x, train_y, W1, b1, W2, b2, W3, b3))
    print("Epoc =", epoc, "--", "Loss =", Loss_3[len(Loss_3)-1], "--",
          "Training Accuracy = %", 100*accuracy(train_x, train_y, W1, b1, W2, b2, W3, b3))
    if epoc >= 75:  # defining minimum number of epoches
        # Defining minimum threshold
        if (abs(Loss_3[len(Loss_3)-1] - Loss_3[len(Loss_3)-2])/Loss_3[len(Loss_3)-2])*100 <= 1:
            break


# 3.i) Plotting the Loss Function
# Plotting Loss Function
plt.title("Loss Function")
plt.xlabel("Epoch")
plt.ylabel("Loss (Cross-Entropy)")
plt.plot(Loss_3)
plt.savefig('Q3_Loss.png')
plt.show()
print("Number of epoches =", len(Loss_3))


# 3.ii) Testing the network with test data
# Plotting Training Accuracy vs number of epoches
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy (%)")
plt.plot(100*Accuracy)
plt.savefig('Q3_Training_Accuracy.png')
plt.show()
# Calculating Test Accuracy
Test_Accuracy = 100*accuracy(test_x, test_y, W1, b1, W2, b2, W3, b3)
print("Training Accuracy  =", "%", Accuracy[len(Accuracy)-1])
print("Test Accuracy  =", "%", Test_Accuracy)


# 3.iii) Plotting each sample and estimations
plt.subplot(251)
plt.imshow(test_x[0, :].reshape(8, 8), cmap='gray')
plt.subplot(252)
plt.imshow(test_x[20, :].reshape(8, 8), cmap='gray')
plt.subplot(253)
plt.imshow(test_x[40, :].reshape(8, 8), cmap='gray')
plt.subplot(254)
plt.imshow(test_x[60, :].reshape(8, 8), cmap='gray')
plt.subplot(255)
plt.imshow(test_x[80, :].reshape(8, 8), cmap='gray')
plt.subplot(256)
plt.imshow(test_x[100, :].reshape(8, 8), cmap='gray')
plt.subplot(257)
plt.imshow(test_x[120, :].reshape(8, 8), cmap='gray')
plt.subplot(258)
plt.imshow(test_x[140, :].reshape(8, 8), cmap='gray')
plt.subplot(259)
plt.imshow(test_x[160, :].reshape(8, 8), cmap='gray')
plt.subplot(2, 5, 10)
plt.imshow(test_x[180, :].reshape(8, 8), cmap='gray')
plt.savefig('Q3_Sample_Estimations.png')


# Estimations for each class shown above (respectively)
for e in range(10):
    print(np.argmax(forward_propagation_3(
        test_x[20*e, :].reshape(64, 1), W1, b1, W2, b2, W3, b3)[6], axis=0))
# Each class is estimated correctly (Except 2)
