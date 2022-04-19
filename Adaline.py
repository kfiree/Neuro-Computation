from mlxtend.classifier import Adaline
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

    def run(self, X, Y):
        # init bias to [1...1]
        # bias = np.ones()
        # self.forward(X)

        row = X.shape[0]
        col = X.shape[1]

        #  add bias to X
        bias = np.ones((row, 1))
        X = np.append(X, bias, axis=1)

        # init weights random values
        np.random.seed(1)
        self.weight = np.random.rand(col + 1)

        # train
        for _ in range(self.ephocs):
            cost = []

            for x_i, y_i in zip(X, Y):
                # calculate error
                error = self.LSE(x_i, y_i)

                # calculate new Weights
                self.weights += self.learning_rate * np.dot(x_i, error)
                cost = 0.5 * (error ** 2)
            MSE = sum(cost) / len(Y)
            self.cost_.append(MSE)

        return self

    def LSE(self, x, y):
        # u = w · x = Σ wi xi
        np.square(np.subtract(y, self.predict(x))).mean()

    def update_weights(self, error, x_i):
        self.weights += self.learning_rate*error
        self.weight += self.learning_rate * error
        cost = 0.5 * (error ** 2)
        return cost

    def predict(self, x):
        return self.weights @ x

    def forward(self, X):
        return None

    def fit(self, X, y):
        row = X.shape[0]
        col = X.shape[1]

        #  add bias to X
        X_bias = np.ones((row, col + 1))
        X_bias[:, 1:] = X
        X = X_bias

        # initialize weights
        np.random.seed(1)
        self.weight = np.random.rand(col + 1)

        # training
        for _ in range(self.niter):
            if self.shuffle:
                X, y = self._shuffle(X, y)

            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = target - output

        self.weight += self.learning_rate * xi.dot(error)
        cost = 0.5 * (error ** 2)
        return cost

    def _shuffle(self, X, y):
        per = np.random.permutation(len(y))
        return X[per], y[per]

    def net_input(self, X):
        return X @ self.weight

    def activation(self, X):
        return self.net_input(X)

    # def predict(self, X):
    #     # if x is list instead of np.array
    #     if type(X) is list:
    #         X = np.array(X)
    #
    #     # add bias to x if he doesn't exist
    #     if len(X.T) != len(self.weight):
    #         X_bias = np.ones((X.shape[0], X.shape[1] + 1))
    #         X_bias[:, 1:] = X
    #         X = X_bias
    #
    #     return np.where(self.activation(X) > 0.0, 1, -1)

    def score(self, X, y):
        wrong_prediction = abs((self.predict(X) - y) / 2).sum()
        self.score_ = (len(X) - wrong_prediction) / len(X)
        return self.score_

    # def fit(self, X, y, biased_X=False):
    #     """ Fit training data to our model """
    #     X = self._add_bias(X)
    #     self._initialise_weights(X)
    #
    #     self.errors = []
    #
    #     for cycle in range(self.iterations):
    #         trg_error = 0
    #         for x_i, output in zip(X, y):
    #             output_pred = self.predict(x_i, biased_X=True)
    #             trg_update = self.learn_rate * (output - output_pred)
    #             self.weights += trg_update * x_i
    #             trg_error += int(trg_update != 0.0)
    #         self.errors.append(trg_error)
    #     return self
    #
    # def predict(self, X, biased_X=False):
    #     """ Make predictions for the given data, X, using unit step function """
    #     if not biased_X:
    #         X = self._add_bias(X)
    #     return np.where(np.dot(X, self.weights) >= 0.0, 1, 0)
    #
    # def _add_bias(self, X):
    #     """ Add a bias column of 1's to our data, X """
    #     bias = np.ones((X.shape[0], 1))
    #     biased_X = np.hstack((bias, X))
    #     return biased_X
    #
    # def initW(self, X):
    #     """ Initialise weigths - normal distribution sample with standard dev 0.01 """
    #     random_gen = np.random.RandomState(1)
    #     self.weights = random_gen.normal(loc=0.0, scale=0.01, size=X.shape[1])
    #     return self
    # def product(self, X):
    #     prediction = np.dot(X, self.weights[1:]+self.w[0])
    #
    # def active(self,X):
    #     return X

#
#
#
# class CustomAdaline(object):
#
#     def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01):
#         self.n_iterations = n_iterations
#         self.random_state = random_state
#         self.learning_rate = learning_rate
#
#     '''
#     Batch Gradient Descent
#
#     1. Weights are updated considering all training examples.
#     2. Learning of weights can continue for multiple iterations
#     3. Learning rate needs to be defined
#     '''
#
#
#     def fit(self, X, y):
#         rgen = np.random.RandomState(self.random_state)
#         self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
#         for _ in range(self.n_iterations):
#             activation_function_output = self.activation_function(self.net_input(X))
#             errors = y - activation_function_output
#             self.coef_[1:] = self.coef_[1:] + self.learning_rate * X.T.dot(errors)
#             self.coef_[0] = self.coef_[0] + self.learning_rate * errors.sum()
#
#     '''
#     Net Input is sum of weighted input signals
#     '''
#
#     def net_input(self, X):
#         weighted_sum = np.dot(X, self.coef_[1:]) + self.coef_[0]
#         return weighted_sum
#
#
#     '''
#     Activation function is fed the net input. As the activation function is
#     an identity function, the output from activation function is same as the
#     input to the function.
#     '''
#
#     def activation_function(self, X):
#         return X
#
#     '''
#     Prediction is made on the basis of output of activation function
#     '''
#
#     def predict(self, X):
#         return np.where(self.activation_function(self.net_input(X)) >= 0.0, 1, 0)
#
#     '''
#     Model score is calculated based on comparison of
#     expected value and predicted value
#     '''
#
#     def score(self, X, y):
#         misclassified_data_count = 0
#         for xi, target in zip(X, y):
#             output = self.predict(xi)
#             if (target != output):
#                 misclassified_data_count += 1
#         total_data_count = len(X)
#         self.score_ = (total_data_count - misclassified_data_count) / total_data_count
#         return self.score_
