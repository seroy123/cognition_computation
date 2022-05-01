import numpy as np
from numpy.random import rand
from numpy import mean


class WeightInitializations:
    def __init__(self, layer_num):
        """
        This function initialize the weights
        :param layer_num: list, number of neurons in each layer. f.e [80 80 80] is a 3 layered net of 80 neurons in each layer.
        """
        self.layer_num = layer_num

    def random_initialization(self):
        """
        This function initialize the weights and bias randomly.
        :return: tuple of lists, weights, bias
        """
        weights = []
        bias = [np.random.randn(next_layer, 1) for next_layer in (self.layer_num[1:])]
        for layer, next_layer in zip(self.layer_num[:-1], self.layer_num[1:]):
            weights += [np.random.randn(next_layer, layer)]
        return weights, bias

    def xavier_initialization(self):
        """
        This function initialize the weights and bias by xavier.
        :return: tuple of lists, weights, bias
        """
        weights = []
        bias = [np.random.randn(next_layer, 1) for next_layer in (self.layer_num[1:])]
        for layer, next_layer in zip(self.layer_num[:-1], self.layer_num[1:]):
            previous_layer_node_number = layer
            # calculate the range for the weights
            lower, upper = -(1.0 / np.sqrt(previous_layer_node_number)), (1.0 / np.sqrt(previous_layer_node_number))
            # generate random weights and scale them to the desired range
            weights += [lower + np.random.randn(next_layer, layer) * (upper - lower)]
        return weights, bias

    def he_initialization(self):
        """
        This function initialize the weights and bias by he.
        :return: tuple of lists, weights, bias
        """
        weights = []
        bias = [np.random.randn(next_layer, 1) for next_layer in (self.layer_num[1:])]
        for layer, next_layer in zip(self.layer_num[:-1], self.layer_num[1:]):
            previous_layer_node_number = layer
            # calculate the range for the weights
            std = np.sqrt(2.0 / previous_layer_node_number)
            # generate random weights and scale them to the desired range
            weights += [np.random.randn(next_layer, layer) * std]
        return weights, bias
