from Lab1.perceptron import Perceptron

weight1 = 0.1
weight2 = 0.1
input1 = [0, 0, 1, 1]
input2 = [0, 1, 0, 1]

first_term = [int(not a and not b) for a, b in zip(input1, input2)]  # equivalent to nor.py(apply bias)
second_term = [int(a and b) for a, b in zip(input1, input2)]

xnor_term = [int(a or b) for a, b in zip(first_term, second_term)]

first_out = Perceptron(2, [input1, input2], first_term, [weight1, weight2], bias=0.5)
second_out = Perceptron(2, [input1, input2], second_term, [weight1, weight2])
final_out = Perceptron(2, [first_term, second_term], xnor_term, [weight1, weight2])
first = first_out.training()
second = second_out.training()
final = final_out.training()
print(f"epochs\n\nA'B': {first}\nAB: {second}\nor: {final}\nxnor: {first+second+final}")