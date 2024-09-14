from Lab1.perceptron import Perceptron

weight1 = 0.1
weight2 = 0.1
input1 = [0, 0, 1, 1]
input2 = [0, 1, 0, 1]
output = [int(a and b) for a, b in zip(input1, input2)]

and_output = Perceptron(2, [input1, input2], output, [weight1, weight2], bias=.3)

print("epochs:", and_output.training())

