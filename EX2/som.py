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
        self.net, self.time_constant, self.init_radius, self.x_min, self.x_max, self.y_min, self.y_max = \
            self.initialization(data)

    def initialization(self, data):
        init_radius = max(self.network_dimensions[0], self.network_dimensions[1]) / 2  # radius
        self.data = data
        # radius decay parameter
        time_constant = self.n_iterations / np.log(init_radius)
        x_min = np.min(data[:, 0])
        y_min = np.min(data[:, 1])
        x_max = np.max(data[:, 0])
        y_max = np.max(data[:, 1])
        net = np.array([[(random.uniform(x_min + 0.001, x_max),
                          random.uniform(y_min + 0.001, y_max)) for _ in range(self.network_dimensions[0])] for _ in
                        range(self.network_dimensions[1])])  # initialize the network
        return net, time_constant, init_radius, x_min, x_max, y_min, y_max

    def fit(self, data):
        for i in range(self.n_iterations + 1):
            # select a training example at random
            t = random.randint(0, len(data) - 1)  # random index
            # find its Best Matching Unit
            bmu_idx = np.array([0, 0])  # index of the best matching unit
            min_dist = np.inf  # initialize the minimum distance with infinity
            # calculate the distance between each neuron and the input
            for x in range(self.net.shape[0]):
                for y in range(self.net.shape[1]):
                    w = np.linalg.norm(data[t] - self.net[x, y])  # euclidean distance
                    if w < min_dist:
                        min_dist = w  # dist
                        bmu_idx = np.array([x, y])  # id
            # decay the SOM parameters
            r = self.init_radius * np.exp(-i / self.time_constant)  # radius
            l_rate = self.learning_rate * np.exp(-i / self.n_iterations)  # learning rate
            # update weight vector to move closer to input
            # and move its neighbours in 2-D vector space closer
            for x in range(self.net.shape[0]):
                for y in range(self.net.shape[1]):
                    w = self.net[x, y]
                    w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)  # dist
                    w_dist = np.sqrt(w_dist)  # dist
                    if w_dist <= r:
                        # calculate the degree of influence (based on the 2-D distance)
                        influence = np.exp(-w_dist / (2 * (r ** 2)))
                        # new w = old w + (learning rate * influence * delta)
                        # where delta = input vector (t) - old w
                        new_w = w + (l_rate * influence * (data[t] - w))
                        self.net[x, y] = new_w  # update the weight vector
            if i % 100 == 0:  # plot the map every 100 iterations
                if self.network_dimensions[0] == 1:
                    self.draw_map(self.x_max, self.x_min, i)
                else:
                    self.plot_net(i)

        return self.net

    def fit_donut(self, data):
        for i in range(self.n_iterations + 1):
            # select a training example at random
            t = random.randint(0, len(data) - 1)  # random index
            # find its Best Matching Unit
            bmu_idx = np.array([0, 0])  # index of the best matching unit
            min_dist = np.inf  # initialize the minimum distance with infinity
            # calculate the distance between each neuron and the input
            for x in range(self.net.shape[0]):
                for y in range(self.net.shape[1]):
                    w = np.linalg.norm(data[t] - self.net[x, y])  # euclidean distance
                    if w < min_dist:
                        min_dist = w  # dist
                        bmu_idx = np.array([x, y])  # id
            # decay the SOM parameters
            r = self.init_radius * np.exp(-i / self.time_constant)
            l_rate = self.learning_rate * np.exp(-i / self.n_iterations)  # learning rate
            # update weight vector to move closer to input
            # and move its neighbours in 2-D vector space closer
            if bmu_idx[0] == 0 and bmu_idx[1] == 0:
                for x in range(self.net.shape[0]):
                    for y in range(self.net.shape[1]):
                        w = self.net[x, y]
                        if x == self.net.shape[0] - 1:
                            w_dist = 1
                        else:
                            w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)  # dist
                            w_dist = np.sqrt(w_dist)  # dist
                        if w_dist <= r:
                            # calculate the degree of influence (based on the 2-D distance)
                            influence = np.exp(-w_dist / (2 * (r ** 2)))
                            # new w = old w + (learning rate * influence * delta)
                            # where delta = input vector (t) - old w
                            new_w = w + (l_rate * influence * (data[t] - w))
                            self.net[x, y] = new_w  # update the weight vector
            elif bmu_idx[0] == self.net.shape[0] - 1:
                for x in range(self.net.shape[0]):
                    for y in range(self.net.shape[1]):
                        w = self.net[x, y]
                        if x == 0:
                            w_dist = 1
                        else:
                            w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)  # dist
                            w_dist = np.sqrt(w_dist)  # dist
                        if w_dist <= r:
                            # calculate the degree of influence (based on the 2-D distance)
                            influence = np.exp(-w_dist / (2 * (r ** 2)))
                            # new w = old w + (learning rate * influence * delta)
                            # where delta = input vector (t) - old w
                            new_w = w + (l_rate * influence * (data[t] - w))
                            self.net[x, y] = new_w  # update the weight vector
            else:
                for x in range(self.net.shape[0]):
                    for y in range(self.net.shape[1]):
                        w = self.net[x, y]
                        w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)  # dist
                        w_dist = np.sqrt(w_dist)  # dist
                        if w_dist <= r:
                            # calculate the degree of influence (based on the 2-D distance)
                            influence = np.exp(-w_dist / (2 * (r ** 2)))
                            # new w = old w + (learning rate * influence * delta)
                            # where delta = input vector (t) - old w
                            new_w = w + (l_rate * influence * (data[t] - w))
                            self.net[x, y] = new_w  # update the weight vector
            if (i % 100 == 0):  # plot the map every 100 iterations
                if self.network_dimensions[0] == 1:
                    self.draw_map(self.x_max, self.x_min, i)
                else:
                    self.plot_net(i)

        return self.net

    def draw_map(self, max, min, t):
        x_n = []  # x-coordinates of the neurons
        y_n = []  # y-coordinates of the neurons
        for i in range(self.net.shape[0]):
            for j in range(self.net.shape[1]):
                x_n.append(self.net[i, j, 0])
                y_n.append(self.net[i, j, 1])
        fig, ax = plt.subplots()
        ax.set_xlim(min, max)
        ax.set_ylim(min, max)
        ax.plot(x_n, y_n, 'b-')
        ax.scatter(self.data[:, 0], self.data[:, 1], alpha=0.3, c='mediumspringgreen')  # plot the data
        ax.scatter([x_n], [y_n], c='darkorange')  # plot the neurons
        ax.set_title(
            f'Iteration: {t} / {self.n_iterations} | Network Dimensions: {self.network_dimensions[0]}x{self.network_dimensions[1]}')
        plt.show()

    def plot_net(self, t):
        neurons_x = self.net[:, :, 0]  # x-coordinates of the neurons
        neurons_y = self.net[:, :, 1]  # y-coordinates of the neurons
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for i in range(self.network_dimensions[0]):
            xh = []
            yh = []
            xs = []
            ys = []
            for j in range(self.network_dimensions[1]):
                xs.append(neurons_x[i, j])
                ys.append(neurons_y[i, j])
                xh.append(neurons_x[j, i])
                yh.append(neurons_y[j, i])
            ax.plot(xs, ys, 'b-', markersize=0, linewidth=1)
            ax.plot(xh, yh, 'b-', markersize=0, linewidth=1)
        ax.plot(neurons_x, neurons_y, color='darkorange', marker='o', linewidth=0, markersize=3)
        ax.scatter(self.data[:, 0], self.data[:, 1], c='mediumspringgreen', alpha=0.3)
        ax.set_title(
            f'Iteration: {t} / {self.n_iterations} | Network Dimensions: {self.network_dimensions[0]}x{self.network_dimensions[1]}')
        plt.show()


if __name__ == '__main__':
    data1 = create_data(1000, 3)
    s = Som(data1, n_iterations=1000, learning_rate=0.1, dima=10, dimb=10).fit(data1)
