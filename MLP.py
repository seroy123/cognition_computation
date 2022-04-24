import numpy as np


class MLP:
    def __init__(self, input_size: int, layer_num: list, actv_func: list):
        # actve_func - activation function per layer in a list.
        # layer_num is a list of number of neurons in each layer. f.e [80 80 80] is a 3 layered net of 80 neurons in
        # each layer.
        self.layer_num = layer_num
        self.actv_func = actv_func
        if len(self.actv_func) != len(self.layer_num):
            raise("number of activation functions must equal to number of layers")
        bias = lambda layer_size: np.zeros([1, layer_size])
        self.weights = [np.append(np.random.rand(input_size, layer_num[0]), bias(layer_num[0]))]+ \
                       [np.append(np.random.rand(layer_num[ind-1],
                        layer_num[ind]),bias(layer_num[0])) for ind in range(2, len(self.layer_num))]
        # + 1 for the n_neurons in each layer for bias

    def weight_initialization(self):
        # TODO: maybe move the whole weight initialization here so you will know the input size

        self.weights = self.weights

    def predict(self, test_data: np.ndarray):
        pass

    def train(self, train_data: np.ndarray, labels: np.ndarray):
        # TODO: maybe initialize weights here so you will have input size and make a field with bias initialization method
        # feed propagation

        # back propagation
        pass
