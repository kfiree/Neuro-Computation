# from mlxtend.classifier import Adaline
import numpy as np

'''
Net Input is sum of weighted input signals
'''


class Adaline:

    def __init__(self, learning_rate=0.01, ephocs=50, random_state=11):
        self.learning_rate = learning_rate
        self.ephocs = ephocs
        self.random_state = random_state
        self.weights = []
        self.cost_ = []

    def train(self, X, Y):
        # === NORMALIZE DATA ===
        # X = (X + 100) / 200
        Y[Y==-1] = 0
        X = X/100

        row = X.shape[0]
        col = X.shape[1]

        #  for matrices multiplication - add 1 for each data sample,
        #  <x,y> -> <1,x,y>,
        bias = np.ones((row, 1))
        X = np.append(bias, X, axis=1)

        # init weights random values
        np.random.seed(2)
        self.weights = np.random.rand(col + 1)

        for _ in range(self.ephocs):
            # === NET INPUT ===
            net_output = self.net_input(X)

            # === ACTIVATION ===
            activation_output = self.activation(net_output)

            # === ERRORS ===
            square_error = self.error(activation_output, Y)

            # === UPDATE WEIGHTS ===

            for i, (x_i, y_i) in enumerate(zip(X, Y)):
                e = np.dot(x_i, self.weights) - y_i
                # print("weights: ", self.weights)
                # print("a's addition", self.learning_rate * X.T.dot(np.subtract(activation_output, Y)))
                # print("addition", self.learning_rate * x_i.dot(square_error[i]))
                print("weights: ", self.weights, ", addition: ", self.learning_rate * x_i.dot(e))
                self.weights += self.learning_rate * x_i.dot(e)

            print(self.weights, "\n")
            # e = Y - net_output

            # print("addition normal error ", self.learning_rate * X.T.dot(e))

            # self.weights = self.weights + self.learning_rate * X.T.dot(square_error)
            # self.weights = self.weights + self.learning_rate * X.T.dot(e)

            # a = 0
            # for i, (x_i, y_i) in enumerate(zip(X, Y)):
            #     a += ((net_output[i] - y_i) ** 2) * x_i
            #     # a += ((net_output[i] - y_i)**2)*x_i
            #
            # a = a/((X.shape[0]))
            # a = a/(2*(X.shape[0]))


            # print()

            # self.weights = self.weights + self.learning_rate * a

            # self.weights += self.learning_rate*(a)

            # self.weights = self.weights + self.learning_rate * hx.dot(e.sum())
            # self.weights = self.weights + self.learning_rate*X.T.dot(square_error)

        return self

    def error(self, n, Y):
        # E = Σ (wi - xi)^2
        return np.square(np.subtract(n, Y))
        # return (np.square(np.subtract(n, Y))/1000)

    # def error(self, n, Y):
    #     # E = Σ (wi - xi)^2
    #     return np.square(np.subtract(n, Y))


    def net_input(self, X):
        return np.dot(X, self.weights)

    def predict(self, X):
        row = X.shape[0]
        bias = np.ones((row, 1))
        X = np.append(bias, X, axis=1)
        return np.where(self.activation(self.net_input(X)) > 0.0, 1, 0)
        # return np.transpose(np.dot(x, self.weights))

    # def _shuffle(self, X, y):
    #     per = np.random.permutation(len(y))
    #     return X[per], y[per]

    def activation(self, X):
        return X


    def score(self, X, y):
        missed_class = self.predict(X).sum()
        return (y.shape[0] - missed_class)/y.shape[0]
        # wrong_prediction = abs((self.predict(X) - y) / 2).sum()
        # self.score_ = (len(X) - wrong_prediction) / len(X)
        # return self.score_


# new_weights[1:] = self.weights[1:] + self.learning_rate * X[1:].T.dot(square_error)

            # for i, (x_i, y_i, e_i) in enumerate(zip(X, Y, square_error)):
            #     new_weights[i] = self.weights[i] + self.learning_rate * np.xi
            #     print(i, ", ", x_i, ", ", y_i, ", ", e_i)

            # new_weights[1:] = self.weights[1:] + self.learning_rate * np.dot(X[1:], square_error)
            # new_weights[1:] = self.weights[1:] + self.learning_rate * X.T.dot(square_error)
            # his[1:] = self.weights[1:] + self.learning_rate * np.dot(Y, square_error)
            # self.his[0] = self.weights[0] + self.learning_rate * square_error.sum()