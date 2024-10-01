from Lab1.perceptron import Perceptron

weight1 = 0.3
weight2 = 0.4
input1 = [0, 0, 1, 1]
input2 = [0, 1, 0, 1]
output = [int(not (a or b)) for a, b in zip(input1, input2)]

nor_output = Perceptron(2, [input1, input2], output, [weight1, weight2], bias=0.6, learning_rate=0.4)

print("epochs:", nor_output.training())

