import numpy as np
from numpy.random import rand
from numpy import mean
from math import sqrt


class WeightInitializations:
    def __init__(self, layer_num):
        self.layer_num = layer_num

    def random_initialization(self):
        weights = []
        bias = [np.random.randn(next_layer, 1) for next_layer in (self.layer_num[1:])]
        for layer, next_layer in zip(self.layer_num[:-1], self.layer_num[1:]):
            weights += [np.random.randn(layer, next_layer)]
        return weights, bias

    def xavier_initialization(self):
        weights = []
        bias = [np.random.randn(next_layer, 1) for next_layer in (self.layer_num[1:])]
        for layer, next_layer in zip(self.layer_num[:-1], self.layer_num[1:]):
            previous_layer_node_number = layer
            # calculate the range for the weights
            lower, upper = -(1.0 / sqrt(previous_layer_node_number)), (1.0 / sqrt(previous_layer_node_number))
            # generate random weights and scale them to the desired range
            weights += [lower + np.random.randn(next_layer, layer) * (upper - lower)]
        return weights, bias

    def he_initialization(self):
        weights = []
        bias = [np.random.randn(next_layer, 1) for next_layer in (self.layer_num[1:])]
        for layer, next_layer in zip(self.layer_num[:-1], self.layer_num[1:]):
            previous_layer_node_number = layer
            # calculate the range for the weights
            std = sqrt(2.0 / previous_layer_node_number)
            # generate random weights and scale them to the desired range
            weights += [np.random.randn(layer, next_layer) * std]
        return weights, bias
