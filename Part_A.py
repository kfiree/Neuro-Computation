import numpy as np
from mlxtend.classifier import Adaline
from numpy.random.mtrand import random

DATA_SIZE = 1000

def createData(debug=False):

    # Syntax : np.random.randint(the range for ex if you choose 100 then your array elements will be within the range 0 to 100, size = (row size, col size)
    np.random.seed(1)
    x_int = np.random.randint(-100, 99, size=(DATA_SIZE, 2))  # a is a variable(object)
    x_frac = np.round(np.random.rand(DATA_SIZE, 2), 2)
    x = x_int + x_frac

    y = np.empty(shape=[DATA_SIZE, 1])



    for i in range(DATA_SIZE):
        y[i] = 1 if x[i][1] > 1 else -1
        # y[i]

    data = np.append(x, y, axis=1)

    if debug:
        print("x: \n", x, "\n")
        print("y: \n", y, "\n")
        print("data: \n", data, "\n")

    return data
    # k = 0
    # for i in range(-H_RANGE, H_RANGE + 1):
    #     for j in range(-H_RANGE, H_RANGE + 1):
    #         data[k, 0] = i / 10
    #         data[k, 1] = j / 10
    #         data[k, 2] = 1 if data[k, 1] > 1 else -1
    #         k += 1
    # print(k)
    # print(data)


createData(True)
