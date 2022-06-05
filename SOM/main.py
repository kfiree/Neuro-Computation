import random
import numpy as np

from som import SOM

DATA_SIZE = 1000


def generateData(part, data_size=DATA_SIZE, debug=False, ):
    np.random.seed(1)
    if part == 1:
        return np.random.rand(data_size, 2)#.astype(np.float64)
    return np.zeros(data_size, 2)#.astype(np.float64)


def PartA():
    data = generateData(1)

    model = SOM(rows=100, cols=2, radius=2, learning_rate=0.3)
    model.train(data, debug=True)
# def train(self, input, epochs=50, times=100, neurons=100, radius=.1, debug=False):


if __name__ == '__main__':

    PartA()
