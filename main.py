import numpy as np
import MLP


def call_MLP(X,Y):
    mlp = MLP.MLP([15, 15, 1], ['ReLU']*3)
    for ind in range(len(X)):
        x = X[ind,:]
        y = Y[ind]
        input_size = len(x)
        mlp.weight_initialization(input_size)
        A,Z = mlp.forward_prop(x)
if __name__ == '__main__':
    with open("./DATA_TRAIN.csv") as file_name:
        array = np.loadtxt(file_name, delimiter=",")

    X = array[:,[0,1]]
    Y = array[:,2]
    call_MLP(X, Y)

