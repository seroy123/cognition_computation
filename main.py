import numpy as np
import MLP
import matplotlib.pyplot as plt


def call_MLP(X_train, Y_train, layers, optimizer, regularization):
    # create MLP network
    mlp = MLP.MLP(layers, optimizer, regularization)
    # train the network
    mlp.train(epochs=80, training_data=X_train, labels=Y_train, eta=0.01)
    ## previous attempts
    # mlp.train(epochs=35, training_data=X_train, labels=Y_train, eta=0.001)
    # mlp.train(epochs=20, training_data=X_train, labels=Y_train, eta=0.05)
    # mlp.train(epochs=90, training_data=X_train, labels=Y_train, eta=0.02)
    # mlp.train(epochs=150, training_data=X_train, labels=Y_train, eta=0.01)
    return mlp


def get_classification_over_validation(mlp):
    # read the validation data
    with open("./DATA_valid.csv") as file_name2:
        validation_set = np.loadtxt(file_name2, delimiter=",")
    X_validation = validation_set[:, [0, 1]]
    Y_validation = validation_set[:, 2]
    # shuffle the validation examples
    order = np.random.permutation(len(Y_validation))
    X_validation = X_validation[order]
    Y_validation = Y_validation[order]
    # get the network classification for the validation data
    ans = [class_ans for class_ans in np.ravel(mlp.get_classification(X_validation))]
    # TODO: this is the Y0 plot, do we want to show it?
    # plt.scatter(X_validation[[val == 1 for val in Y_validation], :][:, 0], X_validation[[val == 1 for val in Y_validation], :][:, 1])
    # plt.scatter(X_validation[[val == 0 for val in Y_validation], :][:, 0], X_validation[[val == 0 for val in Y_validation], :][:, 1])
    # plt.show()
    # plot the classification of the training data by the network
    plt.scatter(X_validation[[val == 1 for val in ans], :][:, 0], X_validation[[val == 1 for val in ans], :][:, 1])
    plt.scatter(X_validation[[val == 0 for val in ans], :][:, 0], X_validation[[val == 0 for val in ans], :][:, 1])
    accuracy = sum(ans == Y_validation) / len(Y_validation)
    plt.title(f"The prediction rate on the validation set is: {accuracy}")
    plt.show()

if __name__ == '__main__':
    # read the training data
    with open("./DATA_TRAIN.csv") as file_name:
        train_data = np.loadtxt(file_name, delimiter=",")
    # shuffle the training examples
    X = train_data[:, [0, 1]]
    Y = train_data[:, 2]
    order = np.random.permutation(len(Y))
    X = X[order]
    Y = Y[order]
    # select sizes of hidden layers
    number_of_hidden_layers = [30]
    ## previous attempts
    # number_of_hidden_layers = [30, 15, 12, 15, 2]
    # number_of_hidden_layers = [12, 12]
    # number_of_hidden_layers = [3, 5, 8]
    # number_of_hidden_layers = [14, 8]
    # number_of_hidden_layers = [30, 30]
    # train the network
    # optimizer = (method, value(for momentum), regularization = (method, lambda value, penalty parameter value)
    trained_mlp = call_MLP(X_train=X, Y_train=Y, layers=[2] + number_of_hidden_layers + [1], optimizer=('momentum',0.5), regularization=('l1+l2',0.5,0.5))
    ## previous attempts
    # trained_mlp = call_MLP(X_train=X, Y_train=Y, layers=[2] + number_of_hidden_layers + [1],
    #                        optimizer=('momentum', 0.3), regularization=('l1+l2', 0.2, 0.2))
    # trained_mlp = call_MLP(X_train=X, Y_train=Y, layers=[2] + number_of_hidden_layers + [1],
    #                        optimizer=('momentum', 0.7), regularization=('l1+l2', 0.75, 0.75))
    # get the classification for the validation data
    get_classification_over_validation(trained_mlp)
