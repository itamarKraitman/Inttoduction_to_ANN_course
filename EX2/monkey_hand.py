from som import *
from som_data import *


def create_data():
    data1 = np.zeros((1500, 2))
    data2 = np.zeros((1500, 2))
    random.seed(11)
    full_hand_data = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    three_fingers_data = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    full_hand_data = full_hand_data[::-1].T
    three_fingers_data = three_fingers_data[::-1].T
    random.seed(1)
    data1 = init_data(full_hand_data, data1)
    data2 = init_data(three_fingers_data, data2)
    return data1, data2


def init_data(mask, data):
    n = 0
    while n < 1500:
        x = random.uniform(0, 20)
        y = random.uniform(0, 20)
        i = int(x)
        j = int(y)
        if mask[i, j] == 1:
            data[n, 0] = x / 20
            data[n, 1] = y / 20
            n += 1
    return data


def plot_data(data, title):
    plt.figure(figsize=(5, 5))
    plt.scatter(data[:, 0], data[:, 1], s=1, alpha=1, c='g')
    plt.title(title)
    plt.show()


class monkey_hand():
    def __init__(self, data):
        self.som = Som(data=data, n_iterations=1000, learning_rate=0.1, dima=15, dimb=15)

    def fit_and_show_changes(self, data, data_type):
        net, time_constant, init_radius, x_min, x_max, y_min, y_max = self.som.initialization(data)
        # print(net.shape)
        for i in range(self.som.n_iterations + 1):
            # select a training example at random
            t = random.randint(0, len(data) - 1)
            # find its Best Matching Unit
            bmu_idx = np.array([0, 0])
            min_dist = np.inf
            # calculate the distance between each neuron and the input
            for x in range(net.shape[0]):
                for y in range(net.shape[1]):
                    w = np.linalg.norm(data[t] - self.som.net[x, y])
                    if w < min_dist:
                        min_dist = w  # dist
                        bmu_idx = np.array([x, y])  # id
            # decay the SOM parameters
            r = init_radius * np.exp(-i / time_constant)
            l = self.som.learning_rate * np.exp(-i / self.som.n_iterations)
            # update weight vector to move closer to input
            # and move its neighbours in 2-D vector space closer
            for x in range(net.shape[0]):
                for y in range(net.shape[1]):
                    w = self.som.net[x, y]
                    w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)  # dist
                    w_dist = np.sqrt(w_dist)  # dist
                    if w_dist <= r:
                        # calculate the degree of influence (based on the 2-D distance)
                        influence = np.exp(-w_dist / (2 * (r ** 2)))
                        # new w = old w + (learning rate * influence * delta)
                        # where delta = input vector (t) - old w
                        new_w = w + (l * influence * (data[t] - w))
                        # print(w.shape)
                        self.som.net[x, y] = new_w
            if (i % 100 == 0):
                self.som.plot_net(i)
        return self.som.net


if __name__ == '__main__':
    data = create_data()
    # plot_data(data[0], "Full Hand Data")
    plot_data(data[1], "Three fingers Data")
    # full hand
    classi = monkey_hand(data[0])
    classi.fit_and_show_changes(data[0], 0)
    # three fingers
    classi.fit_and_show_changes(data[1], 1)
