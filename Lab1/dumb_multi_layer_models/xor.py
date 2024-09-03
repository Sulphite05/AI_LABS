from Lab1.perceptron import Perceptron

weight1 = 0.1
weight2 = 0.1
input1 = [0, 0, 1, 1]
input2 = [0, 1, 0, 1]

first_term = [int(a and not b) for a, b in zip(input1, input2)]
second_term = [int(not a and b) for a, b in zip(input1, input2)]
xor_term = [int(a or b) for a, b in zip(first_term, second_term)]

# OR
# first_term = [int(not (a and b)) for a, b in zip(input1, input2)] # nand(apply bias)
# second_term = [int(a or b) for a, b in zip(input1, input2)]       # or
# xor_term = [int(a and b) for a, b in zip(first_term, second_term)]

first_out = Perceptron(2, [input1, input2], first_term, [weight1, weight2])
second_out = Perceptron(2, [input1, input2], second_term, [weight1, weight2])
final_out = Perceptron(2, [first_term, second_term], xor_term, [weight1, weight2])

first = first_out.training()
second = second_out.training()
final = final_out.training()

print(f"AB': {first}\nA'B: {second}\nor: {final}\nxor: {max(first,second)+final}\n")


