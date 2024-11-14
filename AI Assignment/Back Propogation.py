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
    "b1": b1,
    "b2": b2
    }
    return new_parameters


def model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate):
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(0, num_of_iters+1):
        a2, cache = forward_propagation(X, parameters)
        grads = backward_propagation(X, Y, cache, parameters)
        parameters = update_parameters(parameters, grads, learning_rate)

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


X = [[0, 0], [0, 1], [1, 0], [1, 1]]


print()
Y_train = []


num = int(input('Which gate do you want to train the model for?\n1. XOR\n2. XNOR\n'))
if num == 1:
    Y_train = [0, 1, 1, 0]
else:
    Y_train = [1, 0, 0, 1]

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


# print("\nXNOR IMPLEMENTATION\n")
# for i in range(len(X)):
#     out = int(input(f"Enter output for {X[i]}: "))
#     Y_train.append(out)

# get = "Y"
# while get.upper() == "Y":
#     inp = eval(input("Enter two training inputs in the form e.g. [0, 0]: "))
#     X.append(inp)
#     get = input("More entries(Y/N)? ")
