import numpy as np
import MLP
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def call_MLP(X_train, Y_train, layers, optimizer, regularization):
    mlp = MLP.MLP(layers, optimizer, regularization)
    mlp.train(40, X_train, Y_train, eta=0.01)
    return mlp


def get_classification_over_validation(mlp):
    with open("./DATA_valid.csv") as file_name2:
        validation_set = np.loadtxt(file_name2, delimiter=",")
    X_validation = validation_set[:, [0, 1]]
    Y_validation = validation_set[:, 2]
    X_validation, Y_validation = shuffle(X_validation, Y_validation)

    ans = [(0 if class_ans <= 0.5 else 1) for class_ans in np.ravel(
        mlp.get_classification(X_validation))]
    plt.scatter(X_validation[[val == 1 for val in Y_validation], :][:, 0], X_validation[[val == 1 for val in Y_validation], :][:, 1])
    plt.scatter(X_validation[[val == 0 for val in Y_validation], :][:, 0], X_validation[[val == 0 for val in Y_validation], :][:, 1])
    plt.show()
    plt.scatter(X_validation[[val == 1 for val in ans], :][:, 0], X_validation[[val == 1 for val in ans], :][:, 1])
    plt.scatter(X_validation[[val == 0 for val in ans], :][:, 0], X_validation[[val == 0 for val in ans], :][:, 1])
    accuracy = sum(ans == Y_validation) / len(Y_validation)
    plt.title(f"The prediction rate on the validation set is: {accuracy}")
    plt.show()

if __name__ == '__main__':
    with open("./DATA_TRAIN.csv") as file_name:
        train_data = np.loadtxt(file_name, delimiter=",")
    np.random.shuffle(train_data)
    X = train_data[:, [0, 1]]
    Y = train_data[:, 2]
    X, Y = shuffle(X, Y)
    # select number of hidden layers
    number_of_hidden_layers = [30]
    # optimizer = (method, value(for momentum), regularization = (method, lambda value, penalty parameter value)
    trained_mlp = call_MLP(X, Y, [2] + number_of_hidden_layers + [1], optimizer=('momentum',0.75), regularization=('l1+l2',0.7,0.7))
    get_classification_over_validation(trained_mlp)
