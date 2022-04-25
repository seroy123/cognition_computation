import numpy as np
import MLP


def call_MLP(X, Y):
    mlp = MLP.MLP([2, 15, 1], [MLP.MLP.ReLU] * 2)
    mlp.train(20, X, Y, eta=lambda epoc: 1 / epoc + 1)
    return mlp


def get_classification(mlp, X):
    ans = []
    for example in X:
        ans.append(mlp.forward_prop(example)[1][1][0])
    return ans


if __name__ == '__main__':
    with open("./DATA_TRAIN.csv") as file_name:
        array = np.loadtxt(file_name, delimiter=",")

    X = array[:, [0, 1]]
    Y = array[:, 2]
    mlp = call_MLP(X, Y)
    ans = get_classification(mlp, X)
    print('yayy')
