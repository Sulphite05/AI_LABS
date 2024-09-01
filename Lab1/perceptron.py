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
        weights = []
        
        for i in range(len(self.weights)):
            weights.append([self.weights[i]] * length)

        iterations = 0
        while sum(D) and iterations < 10:

            for i in range(length):
                weighted_input = 0

                for input in range(self.num_of_inputs):
                    weighted_input += self.inputs[input][i] * weights[input][i]

                K[i] = weighted_input + self.bias
                Y[i] = 1 if K[i] >= self.threshold else 0
                D[i] = self.output[i] - Y[i]

                for input in range(self.num_of_inputs):
                    weights[input][i] = weights[input][i] + self.learning_rate * D[i] * self.inputs[input][i]

            iterations += 1
        
        return iterations
