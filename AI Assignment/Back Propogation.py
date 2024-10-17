# import numpy as np
#
# class MultilayerPerceptron:
#     def __init__(self,hl_w, out_w, learning_rate, input_size=2, hidden_size=1, output_size=1, epochs=10000):
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#
#         # Initialize weights and biases
#         self.parameters = self.initialize_parameters(hl_w, out_w)
#
#     @staticmethod
#     def sigmoid(z):
#         return 1 / (1 + np.exp(-z))
#
#     def initialize_parameters(self, hl_w, out_w):
#         np.random.seed(2)
#         W1 = np.full((self.hidden_size, self.input_size), hl_w)
#         b1 = np.zeros((self.hidden_size, 1))
#         W2 = np.full((self.output_size, self.hidden_size), out_w)
#         b2 = np.zeros((self.output_size, 1))
#         parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
#         return parameters
#
#     def forward_prop(self, X):
#         W1 = self.parameters["W1"]
#         b1 = self.parameters["b1"]
#         W2 = self.parameters["W2"]
#         b2 = self.parameters["b2"]
#
#         Z1 = np.dot(W1, X) + b1
#         A1 = np.tanh(Z1)
#         Z2 = np.dot(W2, A1) + b2
#         A2 = self.sigmoid(Z2)
#
#         cache = {"A1": A1, "A2": A2}
#         return A2, cache
#
#     def backward_prop(self, X, Y, cache):
#         A1 = cache["A1"]  # Hidden layer activation
#         A2 = cache["A2"]  # Output layer activation
#         W2 = self.parameters["W2"]  # Weights between hidden and output layer
#         m = X.shape[1]  # Number of training examples (4 in this case)
#
#         # Calculate gradients for output layer
#         dZ2 = A2 - Y  # Derivative of cost with respect to A2 (output)
#         dW2 = np.dot(dZ2, A1.T) / m  # Derivative with respect to W2
#         db2 = np.sum(dZ2, axis=1, keepdims=True) / m  # Derivative with respect to b2
#
#         # Calculate gradients for hidden layer
#         dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))  # Derivative of tanh
#         dW1 = np.dot(dZ1, X.T) / m  # Derivative with respect to W1
#         db1 = np.sum(dZ1, axis=1, keepdims=True) / m  # Derivative with respect to b1
#
#         grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
#         return grads
#
#     def update_parameters(self, grads):
#         self.parameters["W1"] -= self.learning_rate * grads["dW1"]
#         self.parameters["b1"] -= self.learning_rate * grads["db1"]
#         self.parameters["W2"] -= self.learning_rate * grads["dW2"]
#         self.parameters["b2"] -= self.learning_rate * grads["db2"]
#
#     def train(self, X, Y):
#         for i in range(self.epochs + 1):
#             A2, cache = self.forward_prop(X)
#             grads = self.backward_prop(X, Y, cache)
#             self.update_parameters(grads)
#
#     def predict(self, X):
#         A2, _ = self.forward_prop(X)
#         return A2 > 0.5  # Return 1 if >= 0.5, else 0
#
#
# hidden_neurons = int(input("Enter number of neurons in the hidden layer: "))
# learning_rate = float(input("Enter the learning rate: "))
# print()
# hl_w = float(input("Enter initial weights for the hidden layer: "))
# out_w = float(input("Enter initial weights for the output layer: "))
#
# print("\nXNOR IMPLEMENTATION\n")
# X_train = [[0,0],[1,0],[0,1],[1,1]]
# get = "Y"
# # while get.upper() == "Y":
# #     inp = eval(input("Enter two training inputs in the form e.g. [0, 0]: "))
# #     X_train.append(inp)
# #     get = input("More entries(Y/N)? ")
#
#
# print()
# Y_train = [0,1,1,0]
# # for i in range(len(X_train)):
# #     out = int(input(f"Enter output for {X_train[i]}: "))
# #     Y_train.append(out)
#
# X_train = np.array(X_train).T
# print(X_train)
# Y_train = np.array(Y_train).reshape(-1, 1)
#
#
# mlp = MultilayerPerceptron(input_size=2, hidden_size=hidden_neurons, output_size=1, hl_w=hl_w, out_w=out_w,
#                            learning_rate=learning_rate, epochs=10000)
# mlp.train(X_train, Y_train)


# X_test = []

# get = "Y"
# while get.upper() == "Y":
#     inp = eval(input("Enter two testing inputs in the form e.g. [0, 0]: "))
#     X_test.append(inp)
#     get = input("More entries(Y/N)? ")
# print()
# predictions = mlp.predict(X_train)
# rounded_predictions = np.round(predictions).astype(int)
# print("Predictions:", rounded_predictions.flatten())

#
# # XOR data
# X_train = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # Input data
# Y_train = np.array([[0, 1, 1, 0]])  # Output data for XOR
#
# # Model parameters
# n_x = 2  # Number of input neurons
# n_h = 2  # Number of hidden neurons
# n_y = 1  # Number of output neurons
# learning_rate = 0.3
# num_of_iters = 10000
#
# # Initialize and train the model
# nn = NeuralNetwork(n_x, n_h, n_y, learning_rate, num_of_iters)
# nn.train(X_train, Y_train)
#
# # Test the network
# X_test = np.array([[0], [0]])  # Testing XOR(0, 0)
# y_predict = nn.predict(X_test)
# print(f"Neural Network prediction for (0, 0): {y_predict[0][0]}")
#
# X_test = np.array([[1], [1]])  # Testing XOR(1, 1)
# y_predict = nn.predict(X_test)
# print(f"Neural Network prediction for (1, 1): {y_predict[0][0]}")
#
# X_test = np.array([[0], [1]])  # Testing XOR(0, 1)
# y_predict = nn.predict(X_test)
# print(f"Neural Network prediction for (0, 1): {y_predict[0][0]}")
#
# X_test = np.array([[1], [0]])  # Testing XOR(1, 0)
# y_predict = nn.predict(X_test)
# print(f"Neural Network prediction for (1, 0): {y_predict[0][0]}")
# # import numpy as np
# #
# #
# # class MultilayerPerceptron:
# #
# #     def __init__(self, input_size, hidden_size, output_size, hl_w, out_w, learning_rate=0.1, epochs=1000):
# #         self.input_size = input_size
# #         self.hidden_size = hidden_size
# #         self.output_size = output_size
# #         self.learning_rate = learning_rate
# #         self.epochs = epochs
# #
# #         self.weights_input_hidden = np.full((input_size, hidden_size), hl_w)
# #         self.weights_hidden_output = np.full((input_size, output_size), out_w)
# #
# #         # self.bias_hidden = np.random.rand(hidden_size)
#         # self.bias_output = np.random.rand(output_size)
#         self.bias_hidden = np.zeros(hidden_size)
#         self.bias_output = np.zeros(output_size)
#
#     @staticmethod
#     def sigmoid(x):
#         return 1 / (1 + np.exp(-x))
#
#     @staticmethod
#     def sigmoid_derivative(x):
#         return x * (1 - x)
#
#     def forward_propagation(self, X):
#         self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
#         self.hidden_output = np.tanh(self.hidden_input)
#
#         self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
#         self.final_output = self.sigmoid(self.final_input)
#
#         return self.final_output
#
#     def backpropagation(self, X, y, output):
#         m = X.shape[1]
#         error_output = y - output
#         delta_output = error_output * self.sigmoid_derivative(output) / m
#
#         error_hidden = np.dot(delta_output, self.weights_hidden_output.T)
#         delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output) / m
#
#         self.weights_hidden_output -= np.dot(self.hidden_output.T, delta_output) * self.learning_rate
#         self.bias_output -= np.sum(delta_output, axis=0) * self.learning_rate
#
#         self.weights_input_hidden -= np.dot(X.T, delta_hidden) * self.learning_rate
#         self.bias_hidden -= np.sum(delta_hidden, axis=0) * self.learning_rate
#
#     def train(self, X, Y):
#         for epoch in range(self.epochs):
#             y = self.forward_propagation(X)
#             self.backpropagation(X, Y, y)
#             if epoch % 1000 == 0:
#                 loss = np.mean(np.square(Y - y))
#                 print(f"Epoch {epoch}, Loss: {loss}")
#
#     def predict(self, X):
#         return self.forward_propagation(X)
#
#
# hidden_neurons = int(input("Enter number of neurons in the hidden layer: "))
# learning_rate = float(input("Enter the learning rate: "))
# print()
# hl_w = float(input("Enter initial weights for the hidden layer: "))
# out_w = float(input("Enter initial weights for the output layer: "))
#
# print("\nXNOR IMPLEMENTATION\n")
# X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
# get = "Y"
# # while get.upper() == "Y":
# # inp = eval(input("Enter two training inputs in the form e.g. [0, 0]: "))
# # X_train.append(inp)
# # get = input("More entries(Y/N)? ")
#
# print()
# Y_train = [1,0,0,1]
# # for i in range(len(X_train)):
# #     out = int(input(f"Enter output for {X_train[i]}: "))
# #     Y_train.append(out)
#
# X_train = np.array(X_train)
# Y_train = np.array(Y_train).reshape(-1, 1)
#
#
# mlp = MultilayerPerceptron(input_size=2, hidden_size=hidden_neurons, output_size=1, hl_w=hl_w, out_w=out_w,
#                            learning_rate=learning_rate, epochs=10000)
# mlp.train(X_train, Y_train)
#
#
# X_test = []
# get = "Y"
# # while get.upper() == "Y":
# #     inp = eval(input("Enter two testing inputs in the form e.g. [0, 0]: "))
# #     X_test.append(inp)
# #     get = input("More entries(Y/N)? ")
#
# X_test = np.array(X_train)
# print()
# predictions = mlp.predict(X_test)
# rounded_predictions = np.round(predictions).astype(int)
# print("Predictions:", rounded_predictions.flatten())


# import numpy as np
#
#
# class MultiLayerPerceptron:
#     def __init__(self, input_size, output_size, hl_size, lr):
#         self.input_size = input_size
#         self.output_size = output_size
#         self.hl_size = hl_size
#         self.lr = lr
#         self.input_hidden_weights = np.random.rand(input_size, hl_size)
#         self.hidden_output_weights = np.random.rand(hl_size, output_size)
#         self.bias_hidden = np.random.rand(hl_size)
#         self.bias_output = np.random.rand(output_size)
#         self.epochs = 1000
#
#     def train(self, X, y):
#         for epoch in range(self.epochs):
#             output = self.back_propogation()
#
#
# inp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# out = np.array([0, 1, 1, 0])

#
# import numpy as np
#
#
# class MultilayerPerceptron:
#     def __init__(self, input_size, hidden_size, output_size, hl_w, out_w, learning_rate=0.1, epochs=1000):
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#
#         self.weights_input_hidden = np.full((input_size, hidden_size), hl_w)
#         self.weights_hidden_output = np.full((hidden_size, output_size), out_w)
#
#         # self.bias_hidden = np.random.rand(hidden_size)
#         # self.bias_output = np.random.rand(output_size)
#         self.bias_hidden = np.zeros(hidden_size)
#         self.bias_output = np.zeros(output_size)
#
#
#     @staticmethod
#     def sigmoid(z):
#         return 1 / (1 + np.exp(-z))
#
#     # def initialize_parameters(self):
#     #     np.random.seed(2)
#     #     W1 = np.random.randn(self.n_h, self.n_x)
#     #     b1 = np.zeros((self.n_h, 1))
#     #     W2 = np.random.randn(self.n_y, self.n_h)
#     #     b2 = np.zeros((self.n_y, 1))
#     #     parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
#     #     return parameters
#
#     def forward_propagation(self, X):
#         # W1 = self.parameters["W1"]
#         # b1 = self.parameters["b1"]
#         # W2 = self.parameters["W2"]
#         # b2 = self.parameters["b2"]
#
#         Z1 = np.dot(self.weights_input_hidden, X) + self.bias_hidden
#         A1 = np.tanh(Z1)
#         Z2 = np.dot(self.weights_hidden_output, A1) + self.bias_output
#         A2 = self.sigmoid(Z2)
#
#         cache = {"A1": A1, "A2": A2}
#         return A2, cache
#
#     def calculate_cost(self, A2, Y):
#         m = Y.shape[1]
#         cost = -np.sum(np.multiply(Y, np.log(A2)) + np.multiply(1 - Y, np.log(1 - A2))) / m
#         cost = np.squeeze(cost)
#         return cost
#
#     def backward_prop(self, X, Y, cache):
#         A1 = cache["A1"]
#         A2 = cache["A2"]
#         # W2 = self.parameters["W2"]
#         m = X.shape[1]
#
#         dZ2 = A2 - Y
#         dW2 = np.dot(dZ2, A1.T) / m
#         db2 = np.sum(dZ2, axis=1, keepdims=True) / m
#         dZ1 = np.multiply(np.dot(self.weights_hidden_output, dZ2), 1 - np.power(A1, 2))
#         dW1 = np.dot(dZ1, X.T) / m
#         db1 = np.sum(dZ1, axis=1, keepdims=True) / m
#
#         self.weights_input_hidden -= learning_rate * dW2
#         self. weights_hidden_output -= learning_rate * dW1
#
#         self.bias_hidden -= learning_rate * db2
#         self.bias_output -= learning_rate * db1
#
#         # grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
#         # return grads
#
#     # def update_parameters(self, grads):
#     #     self.parameters["W1"] -= self.learning_rate * grads["dW1"]
#     #     self.parameters["b1"] -= self.learning_rate * grads["db1"]
#     #     self.parameters["W2"] -= self.learning_rate * grads["dW2"]
#     #     self.parameters["b2"] -= self.learning_rate * grads["db2"]
#
#     def train(self, X, Y):
#         for i in range(self.epochs + 1):
#             A2, cache = self.forward_propagation(X)
#             cost = self.calculate_cost(A2, Y)
#             self.backward_prop(X, Y, cache)
#             # self.update_parameters(grads)
#
#             if i % 100 == 0:
#                 print(f'Cost after iteration {i}: {cost}')
#
#     def predict(self, X):
#         A2, _ = self.forward_propagation(X)
#         return A2 > 0.5  # Return 1 if >= 0.5, else 0
#
#
# hidden_neurons = int(input("Enter number of neurons in the hidden layer: "))
# learning_rate = float(input("Enter the learning rate: "))
# print()
# hl_w = float(input("Enter initial weights for the hidden layer: "))
# out_w = float(input("Enter initial weights for the output layer: "))
#
# print("\nXNOR IMPLEMENTATION\n")
# X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
# get = "Y"
# # while get.upper() == "Y":
# # inp = eval(input("Enter two training inputs in the form e.g. [0, 0]: "))
# # X_train.append(inp)
# # get = input("More entries(Y/N)? ")
#
# print()
# Y_train = [1,0,0,1]
# # for i in range(len(X_train)):
# #     out = int(input(f"Enter output for {X_train[i]}: "))
# #     Y_train.append(out)
#
# X_train = np.array(X_train)
# Y_train = np.array(Y_train).reshape(-1, 1)
#
#
# mlp = MultilayerPerceptron(input_size=2, hidden_size=hidden_neurons, output_size=1, hl_w=hl_w, out_w=out_w,
#                            learning_rate=learning_rate, epochs=10000)
# mlp.train(X_train, Y_train)
#
#
# X_test = []
# get = "Y"
# # while get.upper() == "Y":
# #     inp = eval(input("Enter two testing inputs in the form e.g. [0, 0]: "))
# #     X_test.append(inp)
# #     get = input("More entries(Y/N)? ")
#
# X_test = np.array(X_train)
# print()
# predictions = mlp.predict(X_test)
# rounded_predictions = np.round(predictions).astype(int)
# print("Predictions:", rounded_predictions.flatten())


import numpy as np


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))
    parameters = {
    "W1": W1,
    "b1" : b1,
    "W2": W2,
    "b2" : b2
    }
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = {
    "A1": A1,
    "A2": A2
    }
    return A2, cache


def backward_propagation(X, Y, cache, parameters):
    A1 = cache["A1"]
    A2 = cache["A2"]
    W2 = parameters["W2"]
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m
    grads = {
    "dW1": dW1,
    "db1": db1,
    "dW2": dW2,
    "db2": db2
    }
    return grads


def update_parameters(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    new_parameters = {
    "W1": W1,
    "W2": W2,
    "b1" : b1,
    "b2" : b2
    }
    return new_parameters


def model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate):
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(0, num_of_iters+1):
        a2, cache = forward_propagation(X, parameters)
        grads = backward_propagation(X, Y, cache, parameters)
        parameters = update_parameters(parameters, grads,learning_rate)

    return parameters


def predict(X, parameters):
    y, _ = forward_propagation(X, parameters)
    yhat = np.squeeze(y)
    return int(yhat >= 0.5)


hidden_neurons = int(input("Enter number of neurons in the hidden layer: "))
learning_rate = float(input("Enter the learning rate: "))
print()
hl_w = float(input("Enter initial weights for the hidden layer: "))
out_w = float(input("Enter initial weights for the output layer: "))

print("\nXNOR IMPLEMENTATION\n")
X = []
get = "Y"
while get.upper() == "Y":
    inp = eval(input("Enter two training inputs in the form e.g. [0, 0]: "))
    X.append(inp)
    get = input("More entries(Y/N)? ")


print()
Y_train = []
for i in range(len(X)):
    out = int(input(f"Enter output for {X[i]}: "))
    Y_train.append(out)

X = np.array(X).T
Y = np.array([Y_train])
m = X.shape[1]

n_x = 2
n_h = 2
n_y = 1
epochs = 1000
trained_parameters = model(X, Y, 2, hidden_neurons, 1, epochs, learning_rate)

X_test = []
get = "Y"
while get.upper() == "Y":
    inp = eval(input("Enter two testing inputs in the form e.g. [0, 0]: "))
    X_test.append(inp)
    get = input("More entries(Y/N)? ")
print()

X_test = np.array(X_test)

for x1, x2 in X_test:
    predictions = predict([[x1], [x2]], trained_parameters)
    rounded_predictions = np.round(predictions).astype(int)
    print(f"Prediction for ({x1}, {x2}): ", rounded_predictions.flatten())
