# Write down a program for training a perceptron. The algorithm may take initial weight values,
# learning rate and input table from user (input table may be two input table of any logical gate). The
# algorithm must be capable of running perceptron learning algorithm and train the model. After
# training the user may be able to feed testing inputs and the algorithm may be able to generate output
# approximating the function. Show the practical demonstration to course instructor.

class Perceptron:
    def __init__(self, num_of_inputs, inputs, output, weights, learning_rate=0.2, bias=0, threshold=0.5):
        self.num_of_inputs = num_of_inputs
        self.inputs = inputs
        self.output = output
        self.weights = weights
        self.learning_rate = learning_rate
        self.bias = bias
        self.threshold = threshold

    def testing(self):
        length = len(self.output)
        assert self.num_of_inputs == len(self.inputs) == len(self.weights)
        for i in range(self.num_of_inputs):
            assert len(self.inputs[i]) == length

    def training(self):
        length = len(self.output)
        self.testing()
        K = [0] * length
        Y = [0] * length
        D = [1] * length

        iterations = 0
        while set(D) != {0} and iterations < 17:
            for i in range(length):
                weighted_input = 0

                for input_num in range(self.num_of_inputs):
                    weighted_input += self.inputs[input_num][i] * self.weights[input_num]

                K[i] = weighted_input + self.bias
                Y[i] = 1 if K[i] >= self.threshold else 0
                D[i] = self.output[i] - Y[i]

                for input_num in range(self.num_of_inputs):
                    self.weights[input_num] = self.weights[input_num] + self.learning_rate * D[i] * \
                                              self.inputs[input_num][i]
                print(self.weights)
            iterations += 1
            print(K)
            print(D)
            print()

        return iterations
