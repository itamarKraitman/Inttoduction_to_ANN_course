import numpy as np

class Adaline(object):

    def __init__(self, rate=0.0001, n_iter=24):
        self.l_rate = rate
        self.n_iter = n_iter
        self.weights = []
        self.costs = []

    def fit(self, X, y):
        """
        training the data- 1. adding bias. 2. shuffling. 3. training
        :param X: data
        :param y:label
        :return: trained model
        """
        row = X.shape[0]
        col = X.shape[1]
        # bias
        X = self._bias(X, (row, col))
        # weights
        np.random.seed(1)
        self.weights = np.random.rand(col + 1)
        # training
        for iter in range(self.n_iter):
            # shuffling
            X, y = self._shuffle(X, y)
            cost = []
            for sample, label in zip(X, y):
                cost.append(self._update_weights(sample, label))
            # computing the avg cost and adding to cost list
            avg = sum(cost) / len(y)
            self.costs.append(avg)
        return self

    def _update_weights(self, sample, label):
        """
        updating weights. private method
        :param sample:
        :param target:
        :return: cost
        """
        result = self.net_input(sample)
        error = label - result
        self.weights += self.l_rate * sample.dot(error)
        # cost = 0.5 * (error ** 2)
        return (error ** 2) / 2

    def _shuffle(self, X, y):
        """
        shuffling data
        :param X: data
        :param y: label
        :return: shuffled data
        """
        per = np.random.permutation(len(y))
        return X[per], y[per]

    def net_input(self, X):
        """
        calculating the net input using matrix multiplication
        :param X: data
        :return: net input
        """
        return X @ self.weights

    def predict(self, X):
        """
        predict sample label
        :param X: data
        :return: 1 or -1
        """
        """ if data and weights are not in the same shape
        we should add bias in order to perform matrix multiplication"""
        if len(X.T) != len(self.weights):
            X = self._bias(X, (X.shape[0], X.shape[1]))
        return np.where(self.net_input(X) > 0.0, 1, -1)

    def score(self, X, y):
        """
        computing the score of the net.
        :param X: data
        :param y: label
        :return: score of net
        """
        wrong_predictions = 0
        # count the wrong predictions
        predicted = self.predict(X)
        for pred, real in zip(predicted, y):
            if pred != real:
                wrong_predictions += 1
        # compute the score
        self.score_ = (len(y) - wrong_predictions) / len(y)
        return self.score_

    def _bias(self,X, size):
        """
        adding bias to the data
        :param X: data
        :param size: size of X
        :return: X after adding bias
        """
        bias = np.ones((size[0], size[1] + 1))
        bias[:, 1:] = X
        X = bias
        return X