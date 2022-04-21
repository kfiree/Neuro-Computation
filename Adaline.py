import numpy as np
from tqdm import trange
from enum import Enum

DATA_SIZE = 1000

class Part(Enum):
    A = 1
    B = 2
"""
TODO:
- add the parts as enum
"""
class Adaline:
    """
            === ADALINE ALGORITHM ===
    adaline algorithm with stochastic gradient descent
    """

    def __init__(self, learning_rate=0.01, epochs=500, data_size=DATA_SIZE, part=Part.A):
        self.data_size = data_size
        self.part = part
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        # self.cost_ = []

    def train(self, X, Y, debug=False):
        """
        train model

        :param X: data samples
        :param Y: targets
        :param debug: verbose mode
        :return: adaline class with updated weights
        """

        ''' INIT WEIGHTS '''
        col = X.shape[1]
        np.random.seed(2)
        self.weights = np.random.rand(col + 1)

        X = self.data_preparation(X)

        ''' STOCHASTIC GRADIENT DESCENT
            for 'epochs' times, run over the data and update weights with each data sample '''
        for _ in trange(self.epochs, desc="Training model", unit="epochs"):
            for i, (x_i, y_i) in enumerate(zip(X, Y)):
                net_output_i = self.net_input(x_i)
                error_i = np.subtract(y_i, net_output_i)
                if debug:
                    print("weights: ", self.weights, ", addition: ", self.learning_rate * x_i.dot(error_i))
                self.weights += self.learning_rate * x_i.dot(error_i)

        return self

    def data_preparation(self, X):
        """
        1) normalize data.
        2) add '1' to the first coordinate of each data sample. <x,y>  ->  <1,x,y>

        :param X: data.
        :return: new data.
        """
        X = X / 100

        bias = np.ones((X.shape[0], 1))
        X = np.append(bias, X, axis=1)

        return X

    def net_input(self, X):
        """
        multiply the data sample x with weights.

        :param X: data sample
        :return: prediction in percentage
        """
        return np.dot(X, self.weights)

    def activation(self, X):
        """
        linear activation function - does nothing
        :return: X
        """
        return X

    def predict(self, X):
        """
        predict output '-1' or '1' for the net output of data samples.

        :param X: net output
        :return: class prediction
        """

        bias = np.ones((X.shape[0], 1))
        X = np.append(bias, X, axis=1)
        if self.part == Part.A:
            return np.where(self.activation(self.net_input(X)) > 0.01, 1, -1)
        elif Part.B:
            X1 = X[:, 0] ** 2
            X2 = X[:, 1] ** 2
            sum = np.add(X1, X2)
            y = np.empty(shape=[self.data_size])
            for i in range(self.data_size):
                y[i] = 1 if sum[i] >= 4 and sum[i] <= 9 else -1
            return y


    def test(self, X, y):
        """

        :param X: data samples.
        :param y: targets
        :return: percentage of success
        """
        missed_class = 0
        for pred_i, y_i in zip(self.predict(X), y):
            if y_i != pred_i:
                missed_class += 1

        return (y.shape[0] - missed_class) / (X.shape[0])

    # def error(self, n, Y):
    #     # E = Σ (wi - xi)^2
    #     return np.square(np.subtract(n, Y))


