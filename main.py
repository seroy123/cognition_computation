import numpy as np
import MLP


def call_MLP(X,Y):
    mlp = MLP.MLP([2, 15, 1], [MLP.MLP.ReLU] * 3)
    mlp.train(20,X,Y)
if __name__ == '__main__':
    with open("./DATA_TRAIN.csv") as file_name:
        array = np.loadtxt(file_name, delimiter=",")

    X = array[:,[0,1]]
    Y = array[:,2]
    call_MLP(X, Y)

