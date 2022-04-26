import numpy as np
import MLP
from sklearn.utils import shuffle

def call_MLP(X, Y, layers):
    mlp = MLP.MLP(layers, [MLP.MLP.ReLU] * (len(layers)-1))
    mlp.train(500, X, Y, eta=-0.005)
    return mlp


def get_classification(mlp, X):
    ans = []
    for example in X:
        ans.append(mlp.forward_prop(example)[1][1][0])
    return ans


if __name__ == '__main__':
    with open("./DATA_TRAIN.csv") as file_name:
        array = np.loadtxt(file_name, delimiter=",")
    # np.random.shuffle(array)
    X = array[:, [0, 1]]/np.max(np.abs(array[:, [0, 1]]),axis=0)
    Y = array[:, 2]
    X, Y = shuffle(X, Y)
    mlp = call_MLP(X, Y, [2]+[12,12]+[1])
    ans = [(0 if class_ans <= 0.5 else 1) for class_ans in np.ravel(get_classification(mlp, X))]
    accuracy = sum(ans == Y)/len(Y)
    print(accuracy)
