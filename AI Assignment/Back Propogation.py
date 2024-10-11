# Write an algorithm for back propagation training on multilayer feedforward network. The algorithm
# may take initial weight values, learning rate and input table from user (input table may be two input
# table of any logical gate). The algorithm must be capable of running perceptron learning algorithm
# and train the model. After training the user may be able to feed testing inputs and the algorithm
# may be able to generate output approximating the function. [Marks: 1]

import random
import math


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


# Create a matrix (list of lists) filled with random values
def create_matrix(rows, cols):
    return [[random.random() for _ in range(cols)] for _ in range(rows)]


# Perform dot product between two matrices
def dot_product(A, B):
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


# Element-wise addition of matrix and bias
def add_bias(matrix, bias):
    return [[matrix[i][j] + bias[0][j] for j in range(len(matrix[0]))] for i in range(len(matrix))]


# Element-wise activation of a matrix
def apply_activation(matrix, activation_fn):
    return [[activation_fn(matrix[i][j]) for j in range(len(matrix[0]))] for i in range(len(matrix))]


# Element-wise subtraction of two matrices
def subtract(Y, A):
    return [[Y[i][j] - A[i][j] for j in range(len(Y[0]))] for i in range(len(Y))]


# Element-wise multiplication of two matrices
def elementwise_multiply(A, B):
    return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


# Transpose a matrix
def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


# Forward pass
def forward_pass(X, W1, b1, W2, b2, W3, b3):
    Z1 = add_bias(dot_product(X, W1), b1)
    A1 = apply_activation(Z1, sigmoid)

    Z2 = add_bias(dot_product(A1, W2), b2)
    A2 = apply_activation(Z2, sigmoid)

    Z3 = add_bias(dot_product(A2, W3), b3)
    A3 = apply_activation(Z3, sigmoid)

    return Z1, A1, Z2, A2, Z3, A3


# Backpropagation
def backprop(X, Y, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, b1, b2, b3, learning_rate):
    m = len(X)

    # Output layer error
    dZ3 = subtract(A3, Y)
    dW3 = dot_product(transpose(A2), dZ3)
    db3 = [[sum(dZ3[i][j] for i in range(len(dZ3))) / m for j in range(len(dZ3[0]))]]

    # Hidden layer 2 error
    dA2 = dot_product(dZ3, transpose(W3))
    dZ2 = elementwise_multiply(dA2, apply_activation(A2, sigmoid_derivative))
    dW2 = dot_product(transpose(A1), dZ2)
    db2 = [[sum(dZ2[i][j] for i in range(len(dZ2))) / m for j in range(len(dZ2[0]))]]

    # Hidden layer 1 error
    dA1 = dot_product(dZ2, transpose(W2))
    dZ1 = elementwise_multiply(dA1, apply_activation(A1, sigmoid_derivative))
    dW1 = dot_product(transpose(X), dZ1)
    db1 = [[sum(dZ1[i][j] for i in range(len(dZ1))) / m for j in range(len(dZ1[0]))]]

    # Update weights and biases
    for i in range(len(W1)):
        for j in range(len(W1[0])):
            W1[i][j] -= learning_rate * dW1[i][j]

    for i in range(len(W2)):
        for j in range(len(W2[0])):
            W2[i][j] -= learning_rate * dW2[i][j]

    for i in range(len(W3)):
        for j in range(len(W3[0])):
            W3[i][j] -= learning_rate * dW3[i][j]

    for i in range(len(b1[0])):
        b1[0][i] -= learning_rate * db1[0][i]

    for i in range(len(b2[0])):
        b2[0][i] -= learning_rate * db2[0][i]

    for i in range(len(b3[0])):
        b3[0][i] -= learning_rate * db3[0][i]

    return W1, b1, W2, b2, W3, b3


# Train the network
def train(X, Y, hidden1_size, hidden2_size, output_size, epochs, learning_rate):
    input_size = len(X[0])

    # Initialize weights and biases
    W1 = create_matrix(input_size, hidden1_size)
    b1 = create_matrix(1, hidden1_size)

    W2 = create_matrix(hidden1_size, hidden2_size)
    b2 = create_matrix(1, hidden2_size)

    W3 = create_matrix(hidden2_size, output_size)
    b3 = create_matrix(1, output_size)

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        Z1, A1, Z2, A2, Z3, A3 = forward_pass(X, W1, b1, W2, b2, W3, b3)

        # Backpropagation
        W1, b1, W2, b2, W3, b3 = backprop(X, Y, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, b1, b2, b3, learning_rate)

        if epoch % 1000 == 0:
            # Calculate loss (mean squared error)
            loss = sum((Y[i][0] - A3[i][0]) ** 2 for i in range(len(Y))) / len(Y)
            print(f'Epoch {epoch}, Loss: {loss}')

    return W1, b1, W2, b2, W3, b3


# Example usage
if __name__ == "__main__":
    # XOR input data
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [[0], [1], [1], [0]]  # XOR problem

    # Hyperparameters
    hidden1_size = 4
    hidden2_size = 4
    output_size = 1
    epochs = 10000
    learning_rate = 0.1

    # Train the network
