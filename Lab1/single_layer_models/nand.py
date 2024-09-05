from Lab1.perceptron import Perceptron

weight0 = 0.1
weight1 = 0.1
weight2 = 0.1
input0 = [0, 0, 0, 0, 1, 1, 1, 1]
input1 = [0, 0, 1, 1, 0, 0, 1, 1]
input2 = [0, 1, 0, 1, 0, 1, 0, 1]
# output = [int(not (a and b)) for a, b in zip(input1, input2)]
output = [1, 1, 1, 1, 1, 1, 1, 0]

nand_output = Perceptron(3, [input0, input1, input2], output, [weight0, weight1, weight2], bias=1)

print("epochs:", nand_output.training())
