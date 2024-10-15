# Write an algorithm for back propagation training on multilayer feedforward network. The algorithm
# may take initial weight values, learning rate and input table from user (input table may be two input
# table of any logical gate). The algorithm must be capable of running perceptron learning algorithm
# and train the model. After training the user may be able to feed testing inputs and the algorithm
# may be able to generate output approximating the function. [Marks: 1]

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class MultilayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)

    def forward_propagation(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)

        return self.final_output

    def backpropagation(self, X, y, output):
        error_output = y - output
        delta_output = error_output * sigmoid_derivative(output)

        error_hidden = np.dot(delta_output, self.weights_hidden_output.T)
        delta_hidden = error_hidden * sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += np.dot(self.hidden_output.T, delta_output) * self.learning_rate
        self.bias_output += np.sum(delta_output, axis=0) * self.learning_rate

        self.weights_input_hidden += np.dot(X.T, delta_hidden) * self.learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0) * self.learning_rate

    def train(self, X, y):
        for epoch in range(self.epochs):
            output = self.forward_propagation(X)
            self.backpropagation(X, y, output)

    def predict(self, X):
        return self.forward_propagation(X)


hidden_neurons = int(input("Enter number of neurons in the hidden layer: "))
learning_rate = float(input("Enter the learning rate: "))

X_train = []
get = "Y"
while get.upper() == "Y":
    inp = eval(input("Enter two training inputs in the form e.g. [0, 0]: "))
    X_train.append(inp)
    get = input("More entries(Y/N)? ")

Y_train = []
for i in range(len(X_train)):
    out = int(input(f"Enter output for {X_train[i]}: "))
    Y_train.append(out)

X_train = np.array(X_train)
Y_train = np.array(Y_train).reshape(-1, 1)

mlp = MultilayerPerceptron(input_size=2, hidden_size=hidden_neurons, output_size=1, learning_rate=learning_rate, epochs=10000)
mlp.train(X_train, Y_train)

X_test = []
get = "Y"
while get.upper() == "Y":
    inp = eval(input("Enter two testing inputs in the form e.g. [0, 0]: "))
    X_test.append(inp)
    get = input("More entries(Y/N)? ")

X_test = np.array(X_test)

predictions = mlp.predict(X_test)
rounded_predictions = np.round(predictions).astype(int)
print("Predictions:", rounded_predictions.flatten())


# X_train = np.array([
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ])
#
# # XOR output
# y_train = np.array([
#     [0],
#     [1],
#     [1],
#     [0]
# ])
#
# # Initialize the neural network with 2 inputs, 2 hidden layers (2 neurons each), and 1 output
# mlp.py = MultilayerPerceptron(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1, epochs=10000)
#
# # Train the model
# mlp.py.train(X_train, y_train)
#
# # Hardcoded testing inputs
# X_test = np.array([
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ])
#
# # Predict and print results for test data
# predictions = mlp.py.predict(X_test)
# # Round predictions to the nearest integer (since it's a logical gate, we expect 0 or 1)
# rounded_predictions = np.round(predictions).astype(int)
# print("Predictions:", rounded_predictions.flatten())
