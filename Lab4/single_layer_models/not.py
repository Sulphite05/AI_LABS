from Lab1.perceptron import Perceptron

weight = 0.1
input = [0, 1]
output = [int(not a) for a in input]

not_output = Perceptron(1, [input], output, [weight], bias=0.5)

print("epochs:", not_output.training())

