import numpy as np
# from numpy.random.mtrand import random
from Adaline import Adaline
from Adaline import Part
import matplotlib.pyplot as plt


DATA_SIZE = 1000


def PullSamples(data_size=DATA_SIZE, debug=False, part = Part.A):
    np.random.seed(1)
    x_int = np.random.randint(-100, 99, size=(data_size, 2))
    x_frac = np.round(np.random.rand(data_size, 2), 2)
    x = x_int + x_frac

    # x[0][0] = 2
    # x[0][1] = 2

    y = np.empty(shape=[data_size])
    if part == Part.A:
        for i in range(data_size):
            y[i] = 1 if x[i][1] > 1 else -1

    elif part == Part.B:
        X1 = x[:, 0] ** 2
        X2 = x[:, 1] ** 2
        sum = np.add(X1, X2)
        for i in range(data_size):
            y[i] = 1 if sum[i] >= 4 and sum[i] <= 9 else -1

    if debug:
        print("x: \n", x, "\n")
        print("y: \n", y, "\n")
        # print("data: \n", data, "\n")

    return x, y



def partA(_x, _y, debug=False):
    model = Adaline().train(_x, _y, debug=debug)
    score = model.test(_x, _y)
    print("     PART A      ")
    if debug:
        print("weights: ", model.weights)
    print("score: ", score*100, "% success")

def partB(_x, _y, debug=False):
    model = Adaline(part = 'B').train(_x, _y, debug=debug)
    score = model.test(_x, _y)
    print("     PART B      ")
    if debug:
        print("weights: ", model.weights)
    print("score: ", score*100, "% success")

_x, _y = PullSamples(debug=False, part=Part.B)
model = Adaline(part='B').train(_x, _y)
neg_x, neg_y, pos_x, pos_y = model.splitAsShouldBe(_x, _y)
plt.scatter(pos_x, pos_y, label="stars", color="green",
            marker= "*", s=30)
plt.scatter(neg_x, neg_y, label="stars", color="red",
            marker= "*", s=30)
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title("just showing points")
plt.show()


# _x, _y = PullSamples(debug=False)
# partA(_x, _y)
_x, _y = PullSamples(debug=False, part=Part.B)
partB(_x, _y)
