import numpy as np
# from numpy.random.mtrand import random
from Adaline import Adaline

DATA_SIZE = 1000


def PullSamples(data_size=DATA_SIZE, debug=False):
    np.random.seed(1)
    x_int = np.random.randint(-100, 99, size=(data_size, 2))
    x_frac = np.round(np.random.rand(data_size, 2), 2)
    x = x_int + x_frac

    y = np.empty(shape=[data_size ])

    for i in range(data_size):
        y[i] = 1 if x[i][1] > 1 else -1

    # data = np.append(x, y, axis=1)

    if debug:
        print("x: \n", x, "\n")
        print("y: \n", y, "\n")
        # print("data: \n", data, "\n")

    return  x, y
    # k = 0
    # for i in range(-H_RANGE, H_RANGE + 1):
    #     for j in range(-H_RANGE, H_RANGE + 1):
    #         data[k, 0] = i / 10
    #         data[k, 1] = j / 10
    #         data[k, 2] = 1 if data[k, 1] > 1 else -1
    #         k += 1
    # print(k)
    # print(data)


_x, _y = PullSamples(data_size = 100000, debug=False)
model = Adaline().train(_x, _y)
print("weights: ",model.weights)
score = model.score(_x, _y)
print(score)

