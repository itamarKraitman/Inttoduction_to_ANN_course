# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:33:06 2022

@author: avida
"""
import random
import numpy as np
from matplotlib import pyplot as plt
from som_data import create_data


class Som:
    def __init__(self, data, n_iterations=1000, learning_rate=0.1, dima=15, dimb=15):
        self.n_iterations = n_iterations
        self.data = None
        self.learning_rate = learning_rate
        self.network_dimensions = np.array([dima, dimb])
        self.net, self.time_constant, self.init_radius = self.initalization(data)

    def initalization(self, data):
        init_radius = max(self.network_dimensions[0], self.network_dimensions[1]) / 2
        # radius decay parameter
        time_constant = self.n_iterations / np.log(init_radius)
        x_min = np.min(data[:, 0])
        y_min = np.min(data[:, 1])
        x_max = np.max(data[:, 0])
        y_max = np.max(data[:, 1])
        net = np.array([[(random.uniform(x_min + 0.001, x_max),
                          random.uniform(y_min + 0.001, y_max)) for i in range(self.network_dimensions[0])] for j in
                        range(self.network_dimensions[1])])
        # print(net.shape)
        return net, time_constant, init_radius

    def fit(self, data):
        x_min = np.min(data[:, 0])
        x_max = np.max(data[:, 0])
        net, time_constant, init_radius = self.initalization(data)
        # print(net.shape)
        for i in range(self.n_iterations + 1):
            # select a training example at random
            t = random.randint(0, len(data) - 1) # random index
            # find its Best Matching Unit
            bmu_idx = np.array([0, 0]) # index of the best matching unit
            min_dist = np.inf # initialize the minimum distance with infinity
            # calculate the distance between each neuron and the input
            for x in range(net.shape[0]):
                for y in range(net.shape[1]):
                    w = np.linalg.norm(data[t] - self.net[x, y]) # euclidean distance
                    if w < min_dist:
                        min_dist = w  # dist
                        bmu_idx = np.array([x, y])  # id
            # decay the SOM parameters
            r = init_radius * np.exp(-i / time_constant) # radius
            l = self.learning_rate * np.exp(-i / self.n_iterations) # learning rate
            # update weight vector to move closer to input
            # and move its neighbours in 2-D vector space closer
            for x in range(net.shape[0]):
                for y in range(net.shape[1]):
                    w = self.net[x, y]
                    w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)  # dist
                    w_dist = np.sqrt(w_dist)  # dist
                    if w_dist <= r:
                        # calculate the degree of influence (based on the 2-D distance)
                        influence = np.exp(-w_dist / (2 * (r ** 2)))
                        # new w = old w + (learning rate * influence * delta)
                        # where delta = input vector (t) - old w
                        new_w = w + (l * influence * (data[t] - w))
                        self.net[x, y] = new_w # update the weight vector
            if (i % 100 == 0): # plot the map every 100 iterations
                self.draw_map(x_max, x_min, i)
        return self.net

    def draw_map(self, max, min, t):
        xs = []  # x-coordinates of the neurons
        ys = []  # y-coordinates of the neurons
        for i in range(self.net.shape[0]):
            for j in range(self.net.shape[1]):
                xs.append(self.net[i, j, 0])
                ys.append(self.net[i, j, 1])
        fig, ax = plt.subplots()
        ax.scatter([xs], [ys], c='darkorange')  # plot the neurons
        ax.set_xlim(min, max)
        ax.set_ylim(min, max)
        ax.plot(xs, ys, 'b-')
        ax.scatter(self.data[:, 0], self.data[:, 1], alpha=0.3, c='mediumspringgreen')  # plot the data
        ax.set_title(f'Iteration: {t} / {self.n_iterations} | Network Dimensions: {self.network_dimensions[0]}x{self.network_dimensions[1]}')
        plt.show()


if __name__ == '__main__':
    data = create_data(1000, )
    s = Som(data, n_iterations=1000, learning_rate=0.1, dima=1, dimb=100).fit(data)
    # print(s)
    # "plot the result"
    # plt.figure(figsize=(5, 5))
    # plt.scatter(data[:, 0], data[:, 1], s=1, c='b')
    # for x in range(s.shape[0]):
    #     for y in range(s.shape[1]):
    #         plt.plot(s[x, y, 0], s[x, y, 1], 'go')
    # plt.show()
