# Write down a program for training a perceptron. The algorithm may take initial weight values,
# learning rate and input table from user (input table may be two input table of any logical gate). The
# algorithm must be capable of running perceptron learning algorithm and train the model. After
# training the user may be able to feed testing inputs and the algorithm may be able to generate output
# approximating the function. Show the practical demonstration to course instructor. [Marks: 1]

import numpy as np


class Perceptron:

    def __init__(self, weights, learning_rate, epochs):
        self.weights = weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias = 0

    @staticmethod
    def binary_step(x):
        predictions = x >= 0.5
        return predictions.astype(int)

    def train(self, X, Y):
        for epoch in range(self.epochs + 1):
            for x, y in zip(X, Y):
                z = self.predict(x)
                update = self.learning_rate * (y - z)
                self.weights += update * x
                self.bias += update

    def predict(self, X):
        Z = np.dot(X, self.weights) + self.bias
        return self.binary_step(Z)

    def test(self, X):
        return [self.predict(x) for x in X]


w1 = float(input("Enter initial weight 1: "))
w2 = float(input("Enter initial weight 2: "))
l_r = float(input("Enter the learning rate: "))

get = "Y"
X_train = []

print("\nOR GATE IMPLEMENTATION\n")
while get.upper() == "Y":
    inp = eval(input("Enter two training inputs in the form e.g. [0, 0]: "))
    X_train.append(inp)
    get = input("More entries(Y/N)? ")

print()
Y_train = []
for i in range(len(X_train)):
    out = int(input(f"Enter output for {X_train[i]}: "))
    Y_train.append(out)

X_train = np.array(X_train)
y_train = np.array(Y_train)


perceptron = Perceptron(weights=[w1, w2], learning_rate=l_r, epochs=10)

perceptron.train(X_train, y_train)

get = "Y"

print()
X_test = []
while get.upper() == "Y":
    inp = eval(input("Enter two testing inputs in the form e.g. [0, 0]: "))
    X_test.append(inp)
    get = input("More entries(Y/N)? ")


X_test = np.array(X_test)
predictions = perceptron.test(X_test)
print("\nPredictions:", predictions)

