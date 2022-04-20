import numpy as np
# from numpy.random.mtrand import random

DATA_SIZE = 1000


def PullSamples(data_size=DATA_SIZE, debug=False):
    np.random.seed(1)
    x_int = np.random.randint(-100, 99, size=(data_size, 2))
    x_frac = np.round(np.random.rand(data_size, 2), 2)
    x = x_int + x_frac

    y = np.empty(shape=[data_size, 1])

    for i in range(data_size):
        temp = ((x[i][0] ** 2) + (x[i][1] ** 2))
        y[i] = 1 if temp <= 9 and temp >= 4 else -1

    data = np.append(x, y, axis=1)

    if debug:
        X1 = x[:, 0] ** 2
        X2 = x[:, 1] ** 2
        checking = np.add(X1, X2)
        print(tuple(zip(checking, y)))
        # print("data: \n", data, "\n")

    return data, x, y


PullSamples(data_size=100, debug=True)
