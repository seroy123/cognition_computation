import copy
import math

import WeightInitializations
import numpy as np
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, layer_num: list, active_func: list):
        # active_func - activation function per layer in a list.
        # layer_num is a list of number of neurons in each layer. f.e [80 80 80] is a 3 layered net of 80 neurons in
        # each layer.
        self.layer_num = layer_num
        self.active_func = active_func
        if len(self.active_func) != len(self.layer_num) - 1:
            raise ("number of activation functions must equal to number of layers")
        self.weights = []
        self.bias = []

    def weight_initialization(self):
        self.weights, self.bias = WeightInitializations.WeightInitializations(self.layer_num).random_initialization()

    def forward_prop(self, X):
        activation_func_ans = []
        weights_and_input_multiplication = []
        # go over all the network (all the layers)
        for bias, weight, current_sigma in zip(self.bias, self.weights, self.active_func):
            # multiply each layer with the layer's input and add the bias
            current_weights_and_input_multiplication = weight.T @ current_activation_func_ans +\
                                                       bias if weights_and_input_multiplication else\
                np.transpose((weight.T @ X)[np.newaxis]) + bias
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
            if layer != hidden_layers_number:
                temp = copy.copy(delta)
            delta = self.ReLU_deriv(weights_and_input_multiplication[layer]).T * (
                    temp @ self.weights[layer + 1].T) if layer != hidden_layers_number else \
                1/2*(Y-activation_func_ans[layer]) * self.ReLU_deriv(weights_and_input_multiplication[layer])
            # update delta bias to be delta
            delta_B[layer] = delta
            # update delta weights by delta
            delta_W[layer] = (activation_func_ans[layer - 1] @ delta).T\
                if layer != 0 else (np.ravel(X)[np.newaxis].T @ delta).T
        return delta_B, delta_W

    def gradient_descent(self, data: np.ndarray, label: np.ndarray, eta):
        # do feedforward and back propagation
        Z, A = self.forward_prop(data)
        delta_B, delta_W = self.backward_prop(Z, A, data, label)
        # ΔW=-ηδ(w)
        # Delta_W[1] = Delta_W[1].T
        Delta_B = delta_B
        for i in range(len(delta_B)):
            Delta_B[i] = -eta * delta_B[i]
        Delta_W = delta_W
        for i in range(len(delta_W)):
            Delta_W[i] = -eta * delta_W[i]

        # update weights and bias
        self.weights = [weight + current_Delta_W.T for weight, current_Delta_W in
                        zip(self.weights, Delta_W)]
        self.bias = [bias + current_Delta_B.T for bias, current_Delta_B in
                     zip(self.bias, Delta_B)]

    def train(self, epochs, training_data, labels, eta):
        self.weight_initialization()
        y_axis = []
        epoch_axis = []
        for i in range(epochs):
            for ind in range(len(labels)):
                X = training_data[ind, :]
                Y = labels[ind]
                self.gradient_descent(X, Y, eta)
            ans = [(0 if class_ans <= 0.5 else 1) for class_ans in np.ravel(self.get_classification(training_data))]
            accuracy = sum(ans == labels) / len(labels)
            print(accuracy)
            y_axis += [accuracy]
            epoch_axis += [i+1]
        plt.plot(epoch_axis, y_axis)
        plt.show()

    def ReLU(self, Z):
        return np.maximum(Z, 0)# np.ravel([math.sin(val) for val in np.ravel(Z)])[np.newaxis].T#

    def ReLU_deriv(self, Z):
        return Z > 0# np.ravel([math.cos(val) for val in np.ravel(Z)])[np.newaxis].T #

    def predict(self, test_data: np.ndarray):
        pass

    def get_classification(self, X):
        ans = []
        for example in X:
            ans.append(self.forward_prop(example)[1][1][0])
        return ans