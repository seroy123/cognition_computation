import numpy as np


class MLP:
    def __init__(self, layer_num: list, actv_func: list):
        # actve_func - activation function per layer in a list.
        # layer_num is a list of number of neurons in each layer. f.e [80 80 80] is a 3 layered net of 80 neurons in
        # each layer.
        self.layer_num = layer_num
        self.actv_func = actv_func
        if len(self.actv_func) != len(self.layer_num):
            raise("number of activation functions must equal to number of layers")
        self.weights = np.zeros(layer_num)

    def weight_initialization(self):
        self.weights = self.weights

    def predict(self, test_data: np.ndarray):
        pass

    def train(self, train_data: np.ndarray, labels: np.ndarray):
        # feedforward
        pass
