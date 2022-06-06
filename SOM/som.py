import numpy as np
import matplotlib.pyplot as plt


class SOM(object):

    def __init__(self, rows, cols, radius=3, learning_rate=0.5):
        self.shape = (rows, cols)
        self.initial_learning_rate = learning_rate
        self.initial_radius = radius
        self.data = None
        self.epochs = None
        self.neurons = np.array(np.random.rand(rows, cols))  # , dtype=object)

    def find_BMU(self, vector):
        min_dis_vector = np.inf
        # bmu = None
        

        for index, neuron in enumerate(self.neurons):
            dist = np.linalg.norm(vector - neuron)

            if min_dis_vector > dist:
                min_dis_vector = dist
                bmu = index
                # ret_index = index
        # for x in range(self.neurons.shape[0]):
        #     for y in range(self.neurons.shape[1]):
        # n = self.neurons[x, y]
        # dist = np.linalg.norm(vector - n)
        #
        # if min_dis_vector > dist:
        #     min_dis_vector = dist
        #     bmu_index = np.array([x, y])

        return bmu

    def update_neurons(self, bmu, sample_i, iter):
        curr_lr = self.initial_learning_rate * np.exp(-iter / 300)
        for index, neuron in enumerate(self.neurons):
            dis = np.linalg.norm(bmu - index)
            radius = np.exp(-np.power(dis, 2) / self.initial_radius)
            self.neurons[index] += curr_lr * radius * (sample_i - self.neurons[index])

    def train(self, input, epochs=5, times=100, neurons=100, radius=.1, debug=False):
        self.data = input
        self.epochs = epochs

        for epoch in np.arange(epochs):
            np.random.shuffle(self.data)

            for index, sample in enumerate(self.data):
                bmu = self.find_BMU(sample)

                if debug and index % 200 == 0:
                    self.plot(index)  # , epoch)

                self.update_neurons(bmu, sample, index)

            if debug:
                print(str(int(epoch / times * 100)) + '%')  # Progress percentage
        self.plot(epochs, None)

    def plot(self, iter, sample):  # , epoch):

        x_axis, y_axis = np.hsplit(self.neurons, 2)

        x_axis = x_axis.reshape((self.neurons.shape[0],))
        y_axis = y_axis.reshape((self.neurons.shape[0],))

        fig, ax = plt.subplots()
        ax.scatter(x_axis, y_axis)

        # ax.plot(xs, ys)
        ax.scatter(self.data[:, 0], self.data[:, 1], alpha=0.3)

        if sample is not None:
            ax.scatter(sample[0], sample[1], c='red')
        ax.set_title("Data size: " + str(len(self.data)) + " | "
                     + "Iteration: " + str(iter) + " | "
                     # + "Epoch :" + str(epoch) +
                                                   "\n" +
                     "lr: " + str(self.initial_learning_rate) + " | "
                     + "Radius:" + str(self.initial_radius))
        plt.show()