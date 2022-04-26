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

    y = np.empty(shape=[data_size])
    if part == Part.A:
        for i in range(data_size):
            y[i] = 1 if x[i][1] > 1 else -1

    elif part == Part.B:
        X1 = x[:, 0] ** 2
        X2 = x[:, 1] ** 2
        sum = np.add(X1, X2)
        for i in range(data_size):
            y[i] = 1 if sum[i] >= 0.04 and sum[i] <= 0.09 else -1

    if debug:
        print("x: \n", x, "\n")
        print("y: \n", y, "\n")
        # print("data: \n", data, "\n")

    return x, y

def calculateResults(_x, _y, debug=False, part = Part.A):
    model = Adaline(part = part).train(_x, _y, debug=debug)
    score = model.test(_x, _y)
    print("     PART ",part.name,"      ")
    if debug:
        print("weights: ", model.weights)
    print("score: ", score*100, "% success")
    return model


def showResults(_x, _y, p = Part.A):
    model = calculateResults(_x, _y, part=p, debug=False)
    b = -model.weights[0]/ model.weights[2]
    m = -model.weights[1]/ model.weights[2]

    x = [-100, 100]
    y = [-100*m +b, 100*m +b]
    plt.plot(x, y)
    neg_x, neg_y, pos_x, pos_y = model.splitAsShouldBe(_x, _y)
    plt.scatter(pos_x, pos_y, label="stars", color="green",
                marker= "*", s=30)
    plt.scatter(neg_x, neg_y, label="stars", color="red",
                marker= "*", s=30)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    s= "PART " + p.name + ": Adaline Algorithm\n" + str(model.data_size) + " samples " + str(model.learning_rate) + " learning rate"
    plt.title(s)

    plt.xlim(_x[: , 0].min(), _x[: , 0].max())
    plt.ylim(_x[: , 1].min(), _x[: , 1].max())
    plt.show()

part = Part.B
_x, _y = PullSamples(debug=False, part=part)
showResults(_x, _y, part)

