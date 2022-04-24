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

    def weight_initialization(self, input_size):
        bias = lambda layer_size: np.zeros([1, layer_size])
        self.weights = [np.append(np.random.rand(input_size, self.layer_num[0]), bias(self.layer_num[0]),axis=0).T]+ \
                       [np.append(np.random.rand(self.layer_num[ind-1],
                        self.layer_num[ind]),bias(self.layer_num[ind]),axis=0).T for ind in range(1, len(self.layer_num))]
        # + 1 for the n_neurons in each layer for bias

    def train(self, train_data: np.ndarray, labels: np.ndarray):
        # TODO: maybe initialize weights here so you will have input size and make a field with bias initialization method
        input_size = train_data.size[1]
        self.weight_initialization(input_size)  # initialize weights
        # feed propagation

        # back propagation
        pass

    def forward_prop(self, X):
        Z = [self.weights[0][:, :-1].dot(X) + self.weights[0][:, -1]]  # input layer
        A = self.ReLU(Z)
        for ind, layer in enumerate(self.weights[1:]):
            Z += layer[:, :-1].dot(Z[ind])+layer[:,-1]
            A += self.ReLU(Z[-1])
        return Z, A

    def backward_prop(self, Z, A, X, Y):
        dz = [A[-1] - Y]
        dw = [dz[0].dot(A[-2].T)]
        #  loop through layers
        for ind, layer in enumerate(self.weights[:-1]):
            pass
        #one_hot_Y = one_hot(Y)
        dZ2 = A2  # - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)
        return dW1, db1, dW2, db2

    def ReLU(self,Z):
        return np.maximum(Z, 0)

    def ReLU_deriv(self, Z):
        return Z > 0

    def predict(self, test_data: np.ndarray):
        pass
