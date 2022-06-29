import numpy as np
import random


def create_data(data_size, data_kind=1):
    data = np.zeros((data_size, 2))
    if data_kind == 1:
        for i in range(data_size):
            data[i, 0] = random.uniform(0, 1000) / 1000
            data[i, 1] = random.uniform(0, 1000) / 1000
    elif data_kind == 2:
        # create values between 0.4 and 0.6
        for i in range(data_size // 2):
            data[i, 0] = random.uniform(0, 1000) / 1000
            data[i, 1] = random.uniform(0, 1000) / 1000
        for i in range(data_size // 2, data_size):
            data[i, 0] = random.uniform(0.4, 0.6)
            data[i, 1] = random.uniform(0.4, 0.6)
    elif data_kind == 3:
        for i in range(data_size // 2):
            data[i, 0] = random.uniform(0, 1000) / 1000
            data[i, 1] = random.uniform(0, 1000) / 1000
        for i in range(data_size // 2, data_size):
            data[i, 0] = random.uniform(500, 750) / 750
            data[i, 1] = random.uniform(500, 750) / 750
    elif data_kind == 4:  # create donut shape
        n = 0
        while n < data_size:
            x = random.uniform(-2, 2)
            y = random.uniform(-2, 2)
            if 1 <= x ** 2 + y ** 2 <= 2:
                data[n, 0] = x
                data[n, 1] = y
                n += 1
    return data
