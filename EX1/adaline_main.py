import numpy as np
import scipy as sp
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from adaline import Adaline
import seaborn as sns

sns.set_theme(style="ticks")

SIZE = 10000
MIN = -10000
MAX = 10000


def create_data(part, is_fit):
    X = np.empty((SIZE, 2), dtype=object)
    if is_fit:
        random.seed(10)
    y = np.zeros(SIZE)
    if part != "A" and part != "B":
        print("invalid input, enter A or B")
        return
    if part == "A":
        for i in range(SIZE):
            X[i, 0] = (random.randint(MIN, MAX) / 100)  # x
            X[i, 1] = (random.randint(MIN, MAX) / 100)  # y
        for i in range(SIZE):
            y[i] = 1 if X[i][1] > 1 else -1
    if part == "B":
        """manually add points to the dataset in order to get label 1, then fill the other part randomly"""
        for i in range(SIZE):
            X[i, 0] = (random.randint(MIN, MAX) / 1000)  # x
            X[i, 1] = (random.randint(MIN, MAX) / 1000)  # y
        for i in range(SIZE):
            y[i] = 1 if 4 <= (X[i][1] ** 2 + X[i][0] ** 2) <= 9 else -1
    return X, y


def main_a():
    "Part A"
    #create data
    X, y = create_data("A", True)
    X1, y1 = create_data("A", False)
    X2, y2 = create_data("A", False)

    # data distribution
    sns.distplot(X, fit=sp.stats.norm, kde=False, label="x", color="red")
    sns.distplot(X1, fit=sp.stats.norm, kde=False, label="x1", color="green")
    sns.distplot(X2, fit=sp.stats.norm, kde=False, label="x2", color="blue")

    # fit and predict
    adaline = Adaline().fit(X, y)
    pred1 = adaline.predict(X1)
    pred2 = adaline.predict(X2)

    # classification report
    print(classification_report(y1, pred1))
    print(classification_report(y2, pred2))

    # confusion matrix
    cm = confusion_matrix(pred1, y1)
    plt.subplots()
    sns.heatmap(cm, fmt=".0f", annot=True)
    plt.title("confusion matrix")
    plt.xlabel("Test")
    plt.ylabel("Predict")

    cm = confusion_matrix(pred2, y2)
    plt.subplots()
    sns.heatmap(cm, fmt=".0f", annot=True)
    plt.title("confusion matrix")
    plt.xlabel("Test")
    plt.ylabel("Predict")

    plt.show()

    # linear separation
    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1), )

    pred = adaline.predict(np.array([xx1.flatten(), xx2.flatten()]).T)
    pred = pred.reshape(xx1.shape)
    colors = ListedColormap(('red', 'blue'))
    plt.contourf(xx1, xx2, pred, alpha=0.4)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.scatter(X[:, 0], X[:, 1], marker=".", c=y * 2 - 1, s=50, cmap=colors)

    plt.show()

    x_min = X1[:, 0].min() - 1
    x_max = X1[:, 0].max() + 1
    y_min = X1[:, 1].min() - 1
    y_max = X1[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1), )

    pred = adaline.predict(np.array([xx1.flatten(), xx2.flatten()]).T)
    pred = pred.reshape(xx1.shape)
    colors = ListedColormap(('red', 'blue'))
    plt.contourf(xx1, xx2, pred, alpha=0.4, )
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.scatter(X1[:, 0], X1[:, 1], marker=".", c=y1 * 2 - 1, s=50, cmap=colors)

    plt.show()

    x_min = X2[:, 0].min() - 1
    x_max = X2[:, 0].max() + 1
    y_min = X2[:, 1].min() - 1
    y_max = X2[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1), )

    pred = adaline.predict(np.array([xx1.flatten(), xx2.flatten()]).T)
    pred = pred.reshape(xx1.shape)
    colors = ListedColormap(('red', 'blue'))
    plt.contourf(xx1, xx2, pred, alpha=0.4, )
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.scatter(X2[:, 0], X2[:, 1], marker=".", c=y2 * 2 - 1, s=50, cmap=colors)

    plt.show()

    # error-iteration
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(1, len(adaline.costs) + 1), adaline.costs, marker='o')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error')
    ax.set_title('error change per iteration')
    plt.show()


def main_b():
    "Part B"

    # create data
    X, y = create_data("B", True)
    X1, y1 = create_data("B", False)
    X2, y2 = create_data("B", False)

    # data distribution
    # sns.distplot(X, fit=sp.stats.norm, kde=False, label="x", color="red")
    # sns.distplot(X1, fit=sp.stats.norm, kde=False, label="x1", color="green")
    # sns.distplot(X2, fit=sp.stats.norm, kde=False, label="x2", color="blue")
    # plt.show()
    #
    # # fit and predict
    adaline = Adaline().fit(X, y)
    pred1 = adaline.predict(X1)
    #
    # # classification report
    # print(classification_report(y1, pred1))
    #
    # # confusion matrix
    # cm = confusion_matrix(pred1, y1)
    # plt.subplots()
    # sns.heatmap(cm, fmt=".0f", annot=True)
    # plt.title("confusion matrix")
    # plt.xlabel("Test")
    # plt.ylabel("Predict")
    #
    # plt.show()

    """non-linear separation 
        NOTE: SIZE var should be changed to 10,000 when required to execute with 10,000 samples"""
    x_min = X1[:, 0].min() - 1
    x_max = X1[:, 0].max() + 1
    y_min = X1[:, 1].min() - 1
    y_max = X1[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1), )

    pred = adaline.predict(np.array([xx1.flatten(), xx2.flatten()]).T)
    pred = pred.reshape(xx1.shape)
    colors = ListedColormap(('yellow', 'black'))
    plt.contourf(xx1, xx2, pred, alpha=0.4, )
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.scatter(X1[:, 0], X1[:, 1], marker=".", c=y1 * 2 - 1, s=50, cmap=colors)
    plt.legend(loc='upper left')
    plt.show()

    print(accuracy_score(y_true=y1, y_pred=pred1))



if __name__ == '__main__':
    # main_a()
    main_b()
