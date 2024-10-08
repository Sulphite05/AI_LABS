from Lab1.perceptron import Perceptron

weight1 = -0.3
weight2 = -0.5
input1 = [-1, -1, 1, 1]
input2 = [-1, 1, -1, 1]
# output = [int(a or b) for a, b in zip(input1, input2)]
output = [-1, 1, 1, 1]
or_output = Perceptron(2, [input1, input2], output, [weight1, weight2])

print("epochs:", or_output.training())

