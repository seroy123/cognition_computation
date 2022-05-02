import copy

import WeightInitializations
import numpy as np
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, layer_num: list, optimizer, regularization):
        """
        This function initialize the network
        :param layer_num: list, number of neurons in each layer. f.e [80 80 80] is a 3 layered net of 80 neurons in each layer.
        :param optimizer: tuple, optimizer type and size.
        :param regularization: tuple, regularization type and size.
        """
        self.layer_num = layer_num
        self.active_func = [lambda Z: self.Sin(Z)] * (len(self.layer_num) - 2) + [lambda Z: self.Sigmoid(Z)]
        self.active_func_derivatives = [lambda Z: self.Sin_deriv(Z)] * (len(self.layer_num) - 2) + \
                                       [lambda Z: self.Sigmoid_deriv(Z)]
        ## previous attempts
        # self.active_func = [lambda Z: self.ReLU(Z)] * (len(self.layer_num) - 2) + [lambda Z: self.Sigmoid(Z)]
        # self.active_func_derivatives = [lambda Z: self.ReLU_deric(Z)] * (len(self.layer_num) - 2) + \
        #                                [lambda Z: self.Sigmoid_deriv(Z)]
        # self.active_func = [lambda Z: self.Sin(Z)] * (len(self.layer_num) - 2) + [lambda Z: self.ReLU(Z)]
        # self.active_func_derivatives = [lambda Z: self.Sin_deriv(Z)] * (len(self.layer_num) - 2) + \
        #                                [lambda Z: self.ReLU_deriv(Z)]
        if len(self.active_func) != len(self.layer_num) - 1:
            raise ("number of activation functions must equal to number of layers")
        self.weights = []
        self.bias = []
        self.optimizer = optimizer
        self.regularization = regularization

    def weight_initialization(self):
        """
        This function initialize the weights, the bias and the momentum optimizer previous layer weights.
        """
        self.weights, self.bias = WeightInitializations.WeightInitializations(self.layer_num).random_initialization()
        ## previous attempts
        # self.weights, self.bias = WeightInitializations.WeightInitializations(self.layer_num).random_initialization()
        # self.weights, self.bias = WeightInitializations.WeightInitializations(self.layer_num).he_initialization()
        self.momentum_optimizer_previous_layer_weights = [np.zeros((np.shape(weight))) for weight in self.weights]

    def train(self, epochs, training_data, labels, eta):
        """
        This function train the network on the training data by the labels.
        The network will go over the training data "epochs" times and update the weights.
        :param epochs: int, the number of times the network will go over the training data.
        :param training_data: np.array, the examples that the network will learn, each example is a dot (x, y).
        :param labels: np.array, the correct classification of each example (spiral 1 or spiral 0).
        :param eta: float, the learning rate.
        """
        self.weight_initialization()
        y_axis = []
        epoch_axis = []
        # go over all the epochs
        for i in range(1, epochs):
            # go over all examples
            for ind in range(len(labels)):
                # get current example
                X = training_data[ind, :]
                Y = labels[ind]
                # update the weights by gradient descent
                self.gradient_descent(X, Y, eta, len(labels))
            # get the network classification for each example in the training data
            ans = [class_ans for class_ans in np.ravel(self.get_classification(training_data))]
            # get the accuracy of classification
            accuracy = sum(ans == labels) / len(labels)
            print(f'epoch number {i}, accuracy={accuracy}')
            y_axis += [accuracy]
            epoch_axis += [i]
        # plot the accuracy lever of the network throw all the epochs
        plt.plot(epoch_axis, y_axis)
        # TODO: check with roey what does that title means?
        plt.title(f"Convergence on training set with l1+l2 and {'out optimizer' if not self.optimizer[0] else self.optimizer[0]}")
        plt.show()
        # plot the classification of the training data by the network
        plt.scatter(training_data[[val == 1 for val in ans], :][:, 0],
                    training_data[[val == 1 for val in ans], :][:, 1])
        plt.scatter(training_data[[val == 0 for val in ans], :][:, 0],
                    training_data[[val == 0 for val in ans], :][:, 1])
        plt.title(f"The prediction rate on the training set is: {accuracy}")
        plt.show()

    def forward_prop(self, X):
        """
        This function put the X input in to the network.
        The first layer is multiplied by the input and all subsequent layers are multiplied by the previous layer.
        :param X: np.array, the input.
        :return: tuple or np.array:
            weights_and_input_multiplication - all the multiplication results.
            activation_func_ans - all the multiplication results after the activation functions.
        """
        activation_func_ans = []
        weights_and_input_multiplication = []
        # go over all the network (all the layers)
        for bias, weight, current_sigma in zip(self.bias, self.weights, self.active_func):
            # multiply each layer with the layer's input and add the bias
            current_weights_and_input_multiplication = weight @ current_activation_func_ans + \
                                                       bias if weights_and_input_multiplication else \
                np.transpose((weight @ X)[np.newaxis]) + bias
            # use the activation function on the multiplication answer
            current_activation_func_ans = current_sigma(current_weights_and_input_multiplication)
            # save the multiplication answer and the activation function answer
            weights_and_input_multiplication.append(np.copy(current_weights_and_input_multiplication))
            activation_func_ans.append(np.copy(current_activation_func_ans))
        return weights_and_input_multiplication, activation_func_ans

    def backward_prop(self, weights_and_input_multiplication, activation_func_ans, X, Y):
        """
        This function calculate the δ signals of each layer using the δ of the previous layer.
        :param weights_and_input_multiplication: np.array, output of feedforward.
        :param activation_func_ans: np.array, output of feedforward.
        :param X: np.array, the example.
        :param Y: np.array, the example classification.
        :return:tuple of np.array: delta_B, delta_W the δ of the weights and the bias.
        """
        hidden_layers_number = len(self.layer_num) - 2  # removing input and output layer
        # initialize deltas as zeros
        delta_W = [np.zeros(W.shape) for W in self.weights]
        delta_B = [np.zeros(B.shape) for B in self.bias]
        # go over all layers from the hidden layers
        for layer in range(hidden_layers_number, -1, -1):
            # get delta
            if layer != hidden_layers_number:
                delta_of_last_layer = copy.deepcopy(delta)
                # δ(l)=g'(∑w_ik(l)s_k(l-1)) * ∑δ_n(l+1)w_ni(l+1)
            delta = (self.weights[layer + 1].T @ delta_of_last_layer) * self.active_func_derivatives[layer](weights_and_input_multiplication[layer])\
                     if layer != hidden_layers_number else \
                (activation_func_ans[layer] - Y) * self.active_func_derivatives[layer](
                    weights_and_input_multiplication[layer])
            # update delta bias to be delta
            delta_B[layer] = delta
            # update delta weights by delta
            delta_W[layer] = (delta @ activation_func_ans[layer - 1].T) \
                if layer != 0 else (delta @ X[np.newaxis])
        return delta_B, delta_W

    def gradient_descent(self, data: np.ndarray, label: np.ndarray, eta, total_labels_num):
        """
        This function update the weights and bias by gradient decent.
        :param data: np.array, the data we want the network to learn.
        :param label: np.array, the data classification.
        :param eta: float, learning rate.
        :param total_labels_num: int, the number of all the training labels for the regularization.
        """
        # do feedforward and back propagation with the data
        Z, A = self.forward_prop(data)
        delta_B, delta_W = self.backward_prop(Z, A, data, label)
        # update Delta_w (ΔW) and Delta_B (ΔB)  with delta_W (δ(W)) and delta_B (δ(B))
        # ΔW=-ηδ(W), ΔB=-ηδ(B)
        Delta_B = delta_B
        for i in range(len(delta_B)):
            Delta_B[i] = -eta * delta_B[i]
        Delta_W = delta_W
        for i in range(len(delta_W)):
            Delta_W[i] = -eta * delta_W[i]
        # update weights and bias with Delta_w (ΔW) and Delta_B (ΔB)
        # W(n) = W(n-1) + ΔW, B(n) = B(n-1) + ΔB
        for ind, (weight, current_Delta_W, previous_iteration_Delta_W) in enumerate(
                zip(self.weights, Delta_W, self.momentum_optimizer_previous_layer_weights)):
            momentum = previous_iteration_Delta_W * self.optimizer[1]
            penalty = -eta * (1 / total_labels_num) * self.regularization[2] * np.sign(weight) + -eta * (1 / total_labels_num) * self.regularization[1] * weight  # l1+l2
            self.momentum_optimizer_previous_layer_weights[ind] = current_Delta_W
            self.weights[ind] = weight + current_Delta_W + penalty + momentum
        self.bias = [bias + current_Delta_B for bias, current_Delta_B in
                     zip(self.bias, Delta_B)]

    @staticmethod
    def ReLU(Z):
        """
        This function is the ReLU function.
        :param Z: np.array, numbers to calculate thr ReLU result on.
        :return: np.array, ReLU result.
        """
        return np.maximum(Z, 0)

    @staticmethod
    def ReLU_deriv(Z):
        """
        This function is the ReLU derivative function.
        :param Z: np.array, numbers to calculate thr ReLU derivative result on.
        :return: np.array, ReLU derivative result.
        """
        return Z > 0

    @staticmethod
    def Sin(Z):
        """
        This function is the Sin function.
        :param Z: np.array, numbers to calculatete thr Sin result on.
        :return: np.array, Sin result.
        """
        return np.ravel([np.sin(val) for val in np.ravel(Z)])[np.newaxis].T

    @staticmethod
    def Sin_deriv(Z):
        """
        This function is the Sin derivative function.
        :param Z: np.array, numbers to calculate thr Sin derivative result on.
        :return: np.array, Sin derivative result.
        """
        return np.ravel([np.cos(val) for val in np.ravel(Z)])[np.newaxis].T

    @staticmethod
    def Sigmoid(Z):
        """
        This function is the Sigmoid function.
        :param Z: np.array, numbers to calculate thr Sigmoid result on.
        :return: np.array, Sigmoid result.
        """
        return 1 / (1 + np.exp(-Z))

    def Sigmoid_deriv(self, Z):
        """
        This function is the Sigmoid derivative function.
        :param Z: np.array, numbers to calculate thr Sigmoid derivative result on.
        :return: np.array, Sigmoid derivative result.
        """
        return self.Sigmoid(Z) * (1 - self.Sigmoid(Z))

    def get_classification(self, X):
        """
        This function return the network classification for X data.
        :param X: np.array, examples for classification.
        :return: np.array, the network classification for X data.
        """
        ans = []
        for example in X:
            current_ans = self.forward_prop(example)[1][1][0]
            ans.append(0 if current_ans <= 0.5 else 1)
        return ans
