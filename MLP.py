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
            raise("number of activation functions must equal to number of layers")
        self.weights = []
        self.bias = []

    def weight_initialization(self):
        # TODO: add an option for a different method (f.e. xavier)
        # TODO: look where yonatan explained how to do it
        self.bias = [np.random.randn(layer,1) for layer in self.layer_num]
        for layer, next_layer in zip(self.layer_num[:-1], self.layer_num[1:]):
            self.weights += [np.random.randn(layer, next_layer)]

    def gradient_descent(self, batch_data: np.ndarray, batch_labels: np.ndarray):
        # TODO: maybe initialize weights here so you will have input size and make a field with bias initialization method
        # ΔW=-ηδ(w)*s.T
        Delta_B= [np.zeros(b.shape) for b in self.bias]
        Delta_W = [np.zeros(W.shape) for W in self.weights]
        #TODO: make train data the size we want(batch)
        # batch_size = len(batch_labels)
        batch_size = 1
        for ind in range(batch_size):
            # x = batch_data[ind, :]
            # y0 = batch_labels[ind]
            x = batch_data
            y0 = batch_labels
            Z, A = self.forward_prop(x)
            delta_B, delta_W = self.backward_prop(Z, A, x, y0)
            # TODO: plaster... Delta of output should be row vector for adding the arrays
            Delta_W[1] = Delta_W[1].T
            Delta_B = [prev_Delta_B + current_Delta_B for prev_Delta_B, current_Delta_B in zip(Delta_B, delta_B)]
            Delta_W = [prev_Delta_W + current_Delta_W for prev_Delta_W, current_Delta_W in zip(Delta_W, delta_W)]
        Delta_W[1] = Delta_W[1].T
        self.weights = [weight - (1 / batch_size) * current_Delta_W for weight, current_Delta_W in
                        zip(self.weights, Delta_W)]
        self.bias = [bias - (1 / batch_size) * current_Delta_B for bias, current_Delta_B in
                     zip(self.bias, Delta_B)]


    def forward_prop(self, X):
        A = []
        Z = []
        # Z = [self.weights[0][:, :-1].dot(X) + self.weights[0][:, -1]]  # input layer
        # A = self.ReLU(Z)
        for bias, weight, current_sigma in zip(self.bias, self.weights, self.active_func):
            current_Z = weight.T @ current_A + bias[0] if Z else weight.T @ X + bias[0]
            # current_A = current_sigma(current_Z)
            current_A = self.ReLU(current_Z)
            Z.append(current_Z)
            A.append(current_A)
        return Z, A

    def backward_prop(self, Z, A, X, Y):
        #TODO: create func_derivative method
        hidden_layers_number = len(self.layer_num)-2  # removing input and output layer
        delta_W = [np.zeros(W.shape) for W in self.weights]
        delta_B = [np.zeros(B.shape) for B in self.bias]
        for layer in range(hidden_layers_number,-1,-1):
            # delta = func_derivative()
            delta = self.ReLU_deriv(Z[layer]) * (self.weights[layer+1] @ delta.T) if layer != hidden_layers_number else\
                                                 (A[layer]-Y).T * self.ReLU_deriv(Z[layer])
            delta_B[layer] = delta
            delta_W[layer] = np.ravel(A[layer - 1])[np.newaxis].T @ delta.T if layer != 0 else np.ravel(X)[np.newaxis].T @ delta[np.newaxis]

        return delta_B, delta_W

    def train(self, epochs, training_data, labels):
        self.weight_initialization()
        for i in range(epochs):
            for ind in range(len(labels)):
                X = training_data[ind,:]
                Y = labels[ind]
                #TODO: divide to batchs in a seprate function that will receive as input the batch size
                self.gradient_descent(X, Y)

    def ReLU(self,Z):
        return np.maximum(Z, 0)

    def ReLU_deriv(self, Z):
        return Z > 0

    def predict(self, test_data: np.ndarray):
        pass
