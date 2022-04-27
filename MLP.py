import copy
import math
import WeightInitializations
import numpy as np
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, layer_num: list, optimizer, regularization):
        # active_func - activation function per layer in a list.
        # layer_num is a list of number of neurons in each layer. f.e [80 80 80] is a 3 layered net of 80 neurons in
        # each layer.
        self.layer_num = layer_num
        self.active_func = [lambda Z: self.Sin(Z)] * (len(self.layer_num)-2)+[lambda Z: self.Sigmoid(Z)]
        self.active_func_derivatives = [lambda Z: self.Sin_deriv(Z)] * (len(self.layer_num)-2)+\
                                       [lambda Z: self.Sigmoid_deriv(Z)]
        if len(self.active_func) != len(self.layer_num) - 1:
            raise ("number of activation functions must equal to number of layers")
        self.weights = []
        self.bias = []
        self.optimizer = optimizer
        self.regularization = regularization

    def weight_initialization(self):
        self.weights, self.bias = WeightInitializations.WeightInitializations(self.layer_num).xavier_initialization()
        self.momentum_optimizer_previous_layer_weights = [np.zeros((np.shape(weight))) for weight in self.weights]

    def forward_prop(self, X):
        activation_func_ans = []
        weights_and_input_multiplication = []
        # go over all the network (all the layers)
        for bias, weight, current_sigma in zip(self.bias, self.weights, self.active_func):
            # multiply each layer with the layer's input and add the bias
            current_weights_and_input_multiplication = weight @ current_activation_func_ans +\
                                                       bias if weights_and_input_multiplication else\
                np.transpose((weight @ X)[np.newaxis]) + bias
            # use the activation function on the multiplication answer
            current_activation_func_ans = current_sigma(current_weights_and_input_multiplication)
            # save the multiplication answer and the activation function answer
            weights_and_input_multiplication.append(copy.deepcopy(current_weights_and_input_multiplication))
            activation_func_ans.append(copy.deepcopy(current_activation_func_ans))
        return weights_and_input_multiplication, activation_func_ans

    def backward_prop(self, weights_and_input_multiplication, activation_func_ans, X, Y):
        hidden_layers_number = len(self.layer_num) - 2  # removing input and output layer
        # initialize deltas as zeros
        delta_W = [np.zeros(W.shape) for W in self.weights]
        delta_B = [np.zeros(B.shape) for B in self.bias]
        # go over all layers from the hidden layers
        for layer in range(hidden_layers_number, -1, -1):
            # get delta
            if layer != hidden_layers_number:
                delta_of_last_layer = copy.deepcopy(delta)
            delta = self.active_func_derivatives[layer](weights_and_input_multiplication[layer]) * (
                    self.weights[layer + 1].T @ delta_of_last_layer) if layer != hidden_layers_number else \
                (activation_func_ans[layer] - Y) * self.active_func_derivatives[layer](weights_and_input_multiplication[layer])
            # update delta bias to be delta
            delta_B[layer] = delta
            # update delta weights by delta
            delta_W[layer] = (delta @ activation_func_ans[layer - 1].T)\
                if layer != 0 else (delta @ X[np.newaxis])
        return delta_B, delta_W

    def gradient_descent(self, data: np.ndarray, label: np.ndarray, eta):
        # do feedforward and back propagation
        Z, A = self.forward_prop(data)
        delta_B, delta_W = self.backward_prop(Z, A, data, label)
        # ΔW=-ηδ(w)
        Delta_B = delta_B
        for i in range(len(delta_B)):
            Delta_B[i] = -eta * delta_B[i]
        Delta_W = delta_W
        for i in range(len(delta_W)):
            Delta_W[i] = -eta * delta_W[i]
        # update weights and bias
        for ind, (weight, current_Delta_W, previous_iteration_Delta_W) in enumerate(zip(self.weights, Delta_W, self.momentum_optimizer_previous_layer_weights)):
            momentum = previous_iteration_Delta_W * self.optimizer[1]
            penalty = -eta*(1/1400)*self.regularization[2] * np.sign(weight) *\
                      (1 - self.regularization[2]) + -eta*(1/1400)*self.regularization[1] * weight  # l1+l2
            # 1/1400 is 1/m in regularization algorithm representing number of labels
            self.momentum_optimizer_previous_layer_weights[ind] = current_Delta_W
            self.weights[ind] = weight + current_Delta_W + penalty + momentum
        self.bias = [bias + current_Delta_B for bias, current_Delta_B in
                     zip(self.bias, Delta_B)]

    def train(self, epochs, training_data, labels, eta):
        self.weight_initialization()
        y_axis = []
        epoch_axis = []
        for i in range(1, epochs):
            for ind in range(len(labels)):
                X = training_data[ind, :]
                Y = labels[ind]
                self.gradient_descent(X, Y, eta)
            ans = [(0 if class_ans <= 0.5 else 1) for class_ans in np.ravel(self.get_classification(training_data))]
            accuracy = sum(ans == labels) / len(labels)
            print(accuracy)
            y_axis += [accuracy]
            epoch_axis += [i]
        plt.plot(epoch_axis, y_axis)
        plt.title(f"Convergence on training set with l1+l2 and {'out optimizer' if not self.optimizer[0] else self.optimizer[0]}")
        plt.show()

    @staticmethod
    def ReLU(Z):
        return np.maximum(Z, 0)

    @staticmethod
    def ReLU_deriv(Z):
        return Z > 0

    @staticmethod
    def Sin(Z):
        return np.ravel([math.sin(val) for val in np.ravel(Z)])[np.newaxis].T

    @staticmethod
    def Sin_deriv(Z):
        return np.ravel([math.cos(val) for val in np.ravel(Z)])[np.newaxis].T

    @staticmethod
    def Sigmoid(Z):
        return 1/(1 + np.exp(-Z))

    def Sigmoid_deriv(self, Z):
        return self.Sigmoid(Z)*(1-self.Sigmoid(Z))

    def get_classification(self, X):
        ans = []
        for example in X:
            ans.append(self.forward_prop(example)[1][1][0])
        return ans
