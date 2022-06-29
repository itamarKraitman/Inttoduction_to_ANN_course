import random

import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network._base import ACTIVATIONS
# import VisualizeNN as VisNN
from sklearn.neural_network import MLPClassifier
from mlxtend.classifier import Adaline
from adaline2 import *
import seaborn as sns


def crate_data_points_b(numer_of_points: int):  # type B data points
    '''
    Create data points for the Adaline learning algorithm
    '''
    data_points = []
    for _ in range(numer_of_points // 2):
        x = random.randint(-10000, 10000) / 100
        y = random.randint(-10000, 10000) / 100
        label = 1 if 4 <= (x ** 2 + y ** 2) <= 9 else -1
        data_points.append((x, y, label))
    for _ in range(numer_of_points // 2):  # oversampling in the middle of the circle
        x = random.randint(-30, 30) / 10
        y = random.randint(-30, 30) / 10
        label = 1 if (x ** 2 + y ** 2) <= 9 else -1
        data_points.append((x, y, label))
    df = pd.DataFrame(data_points, columns=['x', 'y', 'label'])
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    return X, Y


def part_c(X, y, x_test, y_test, classifier):
    for layer in range(2, classifier.n_layers_):
        layer_i = forward_d(classifier, X, layer)
        geometric_diagram(X, layer - 1, layer_i)
    plot_decision_regions(X, y, classifier)  # plot the decision regions
    plt.show()
    plt.title("Part C: MLP Algorithm")
    plt.legend(loc='upper left')
    'plot confusion matrix'
    cm = confusion_matrix(y_test, classifier.predict(x_test))
    cm_trian = confusion_matrix(y, classifier.predict(X))
    plot_confusion_matrix(cm, 'Part C Data type B Test Data Confusion Matrix ')
    plt.show()
    plot_confusion_matrix(cm_trian, 'Part C Data type B Train Data Confusion Matrix ')
    print(classification_report(y_test, classifier.predict(x_test)))
    print(classification_report(y, classifier.predict(X)))
    print(y)
    plt.show()

    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1), )

    pred = classifier.predict(np.array([xx1.flatten(), xx2.flatten()]).T)
    pred = pred.reshape(xx1.shape)
    colors = ListedColormap(('yellow', 'black'))
    plt.contourf(xx1, xx2, pred, alpha=0.4, )
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.scatter(X[:, 0], X[:, 1], marker=".", c=y * 2 - 1, s=50, cmap=colors)
    plt.legend(loc='upper left')
    plt.show()


def plot_confusion_matrix(cm, title):
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  # annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(['1', '-1'])
    ax.yaxis.set_ticklabels(['1', '-1'])
    plt.show()


def part_d(X, y, x_test, y_test, clf, ):
    last_hidden_layer = []  # last hidden layer for training
    for layer in range(1, clf.n_layers_):
        layer_i = forward_d(clf, X, layer)  # forward propagation
        if layer == clf.n_layers_ - 1:  # last layer
            last_hidden_layer = layer_i  # save last hidden layer
        geometric_diagram(X, layer - 1, layer_i)
    X_Adaline = np.array([last_hidden_layer[0], last_hidden_layer[1]]).T  # create X_Train for Adaline
    last_hidden_layer_t = []  # last hidden layer for testing
    for layer in range(1, clf.n_layers_):
        layer_i = forward_d(clf, x_test, layer)
        if layer == clf.n_layers_ - 1:
            last_hidden_layer_t = layer_i
    X_Adaline_t = np.array([last_hidden_layer_t[0], last_hidden_layer_t[1]]).T  # create X_Test for Adaline
    y[y < 0] = 0  # change labels to 0 and 1
    y_test[y_test < 0] = 0  # change labels to 0 and 1
    classifier_adaline = Adaline(epochs=15, eta=0.01, random_seed=0).fit(X_Adaline, y.astype(int))  # train Adaline
    predict_adaline = classifier_adaline.predict(X_Adaline_t)  # predict Adaline on test data
    print('acc', classifier_adaline.score(X_Adaline_t, y_test) * 100, "%")  # print accuracy
    print(classification_report(y_test, predict_adaline))  # print classification report
    # Geometric diagram of the combination between MLP and Adaline Algorithms
    # plt.scatter(x=x_test[predict_adaline == 0, 1], y=x_test[predict_adaline == 0, 0],
    #             alpha=0.9, c='green', marker='.', label=-1.0)
    # plt.scatter(x=x_test[predict_adaline == 1, 1], y=x_test[predict_adaline == 1, 0],
    #             alpha=0.9, c='purple', marker='.', label=1.0)
    # plt.title("Part D:Backpropagation & Adaline ")
    # plt.legend(loc='upper left')
    # plt.show()
    # confusion_matrix
    conf_matrix = confusion_matrix(predict_adaline, y_test)
    plot_confusion_matrix(conf_matrix, 'Part D:Backpropagation & Adaline Test Data Confusion Matrix')

    # sns.distplot(X_Adaline, fit=sp.stats.norm, kde=False, label="x")
    # sns.distplot(X_Adaline_t, fit=sp.stats.norm, kde=False, label="x1")
    # plt.show()

    x_min = X_Adaline[:, 0].min() - 1
    x_max = X_Adaline[:, 0].max() + 1
    y_min = X_Adaline[:, 1].min() - 1
    y_max = X_Adaline[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1), )

    pred = classifier_adaline.predict(np.array([xx1.flatten(), xx2.flatten()]).T)
    pred = pred.reshape(xx1.shape)
    colors = ListedColormap(('yellow', 'black'))
    plt.contourf(xx1, xx2, pred, alpha=0.4, )
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.scatter(X_Adaline[:, 0], X_Adaline[:, 1], marker=".", c=y * 2 - 1, s=50, cmap=colors)
    plt.legend(loc='upper left')
    plt.show()


def forward_d(classifier, X, layers=None):
    # function to forward propagate through the network basd on https://stackoverflow.com/questions/50869405/
    # python-sklearn-mlpclassifier- how-to-obtain-output-of-the-first-hidden-layer
    if layers is None or layers == 0:
        layers = classifier.n_layers_
    # Initialize first layer
    activation = X
    # Forward propagate
    classifier_activation = ACTIVATIONS[
        classifier.activation]  # Activation function of the classifier in our case is tanh
    for layer in range(layers - 1):  # -1 because we don't need to include the input layer
        layer_weight, layer_bias = classifier.coefs_[layer], classifier.intercepts_[layer]  # Get weights and biases
        activation = np.dot(activation, layer_weight) + layer_bias  # Calculate activation
        if layer != layers - 2:  # If not last layer
            classifier_activation(activation)  # Apply activation function
    if activation.shape[1] > 1:  # If more than one output
        neurons_output = []  # Initialize list
        for j in range(activation.shape[1]):  # For each output
            neurons_output.append(classifier._label_binarizer.inverse_transform(activation[:, j]))  # Get label
        return neurons_output  # Return list
    classifier_activation(activation)  # Apply activation function
    return classifier._label_binarizer.inverse_transform(activation)  # Return label


def geometric_diagram(X, index_layer, layer):
    index_n = 1
    for neuron in layer:
        plt.scatter(x=X[neuron == -1, 1], y=X[neuron == -1, 0],
                    alpha=0.9, c='red',
                    marker='2', label=-1.0)

        plt.scatter(x=X[neuron == 1, 1], y=X[neuron == 1, 0],
                    alpha=0.9, c='blue',
                    marker='>', label=1.0)

        plt.legend(loc='upper left')
        plt.title("Layer: " + str(index_layer) + " Neuron: " + str(index_n))
        plt.show()
        index_n += 1


if __name__ == '__main__':
    X, y = crate_data_points_b(2000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                     hidden_layer_sizes=(8, 2), random_state=1, max_iter=1000)

    classifier_mlp_a = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='tanh',
                                     hidden_layer_sizes=(8, 2), random_state=1, max_iter=1000).fit(X_train, y_train)
    print("\n------ Condition A ------")
    part_c(X_train, y_train, X_test, y_test, classifier_mlp_a, )
    # partD(X_train, y_train, X_test, y_test, classifier_mlp_a)

