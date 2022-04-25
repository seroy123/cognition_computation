import copy

import numpy as np


class MLP:
    def __init__(self, layer_num: list, active_func: list):
        # active_func - activation function per layer in a list.
        # layer_num is a list of number of neurons in each layer. f.e [80 80 80] is a 3 layered net of 80 neurons in
        # each layer.
        self.layer_num = layer_num
        self.active_func = active_func
        if len(self.active_func) != len(self.layer_num):
            raise ("number of activation functions must equal to number of layers")
        self.weights = []
        self.bias = []

    def weight_initialization(self):
        # TODO: add an option for a different method (f.e. xavier)
        # TODO: look where yonatan explained how to do it
        # set the bias of each layer in a random way
        self.bias = [np.random.randn(1, 1) for layer in range(len(self.layer_num)-1)]
        # set random weights
        for layer, next_layer in zip(self.layer_num[:-1], self.layer_num[1:]):
            self.weights += [np.random.randn(layer, next_layer)]

    def gradient_descent(self, data: np.ndarray, label: np.ndarray, eta):
        # TODO: maybe initialize weights here so you will have input size and make a field with bias initialization method
        # do feedforward and back propagation
        Z, A = self.forward_prop(data)
        delta_B, delta_W = self.backward_prop(Z, A, data, label)
        # ΔW=-ηδ(w)
        # Delta_W[1] = Delta_W[1].T
        Delta_B = -eta * delta_B
        Delta_W = delta_W
        for i in range(len(delta_W)):
            Delta_W[i] = -eta * delta_W[i]
        # Delta_W[1] = Delta_W[1].T
        # update weights and bias
        # TODO: eta
        self.weights = [weight - current_Delta_W for weight, current_Delta_W in
                        zip(self.weights, Delta_W)]
        self.bias = [bias - current_Delta_B for bias, current_Delta_B in
                     zip(self.bias, Delta_B)]

    def forward_prop(self, X):
        activation_func_ans = []
        weights_and_input_multiplication = []
        # go over all the network (all the layers)
        for bias, weight, current_sigma in zip(self.bias, self.weights, self.active_func):
            # multiply each layer with the layer's input and add the bias
            current_weights_and_input_multiplication = weight.T @ current_activation_func_ans + bias \
                if weights_and_input_multiplication else weight.T @ X + bias
            # TODO: current_A = current_sigma(current_Z)
            # use the activation function on the multiplication answer
            current_activation_func_ans = self.ReLU(current_weights_and_input_multiplication)
            # save the multiplication answer and the activation function answer
            weights_and_input_multiplication.append(current_weights_and_input_multiplication)
            activation_func_ans.append(current_activation_func_ans)
        return weights_and_input_multiplication, activation_func_ans

    def backward_prop(self, weights_and_input_multiplication, activation_func_ans, X, Y):
        # TODO: create func_derivative method
        hidden_layers_number = len(self.layer_num) - 2  # removing input and output layer
        # initialize deltas as zeros
        delta_W = [np.zeros(W.shape) for W in self.weights]
        delta_B = [np.zeros(B.shape) for B in self.bias]
        # go over all layers from the hidden layers
        for layer in range(hidden_layers_number, -1, -1):
            # TODO: delta = func_derivative()
            # get delta
            delta = self.ReLU_deriv(weights_and_input_multiplication[layer]) * (
                    self.weights[layer + 1] @ delta.T) if layer != hidden_layers_number else \
                (activation_func_ans[layer] - Y).T * self.ReLU_deriv(weights_and_input_multiplication[layer])
            # update delta bias to be delta
            delta_B[layer] = delta
            # update delta weights by delta
            delta_W[layer] = np.ravel(activation_func_ans[layer - 1])[np.newaxis].T @ delta[
                np.newaxis] if layer != 0 else \
                np.ravel(X)[np.newaxis].T @ delta[np.newaxis]
        return delta_B, delta_W

    def train(self, epochs, training_data, labels, eta):
        self.weight_initialization()
        for i in range(epochs):
            for ind in range(len(labels)):
                X = training_data[ind, :]
                Y = labels[ind]
                self.gradient_descent(X, Y, eta)

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def ReLU_deriv(self, Z):
        return Z > 0

    def predict(self, test_data: np.ndarray):
        pass
