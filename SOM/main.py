import random
import numpy as np
from matplotlib import pyplot as plt

from som import SOM

DATA_SIZE = 1000


def generate_unbalanced(data_size=DATA_SIZE, a=True):
    np.random.seed(1)
    if a:
        x_1 = np.random.uniform(0.5, 1, int(data_size / 4 * 3))
        x_2 = np.random.uniform(0, 0.5, int(data_size / 4 * 1))
        y = np.random.uniform(0, 1, data_size)
        x = np.concatenate((x_1, x_2), dtype=float)
    else:
        x_1 = np.random.uniform(0.5, 1, int(data_size / 2))
        x_2 = np.random.uniform(0, 0.5, int(data_size / 2))
        y_1 = np.random.uniform(0.5, 1, int(data_size / 2))
        y_2 = np.random.uniform(0, 0.5, int(data_size / 2))
        x = np.concatenate((x_1, x_2), dtype=float)
        y = np.concatenate((y_1, y_2), dtype=float)
    return np.array([x, y]).T


def generateData(part, data_size=DATA_SIZE, debug=False):
    np.random.seed(1)
    if part == 1:
        return np.random.rand(data_size, 2)  # .astype(np.float64)
    return np.zeros(data_size, 2)  # .astype(np.float64)


def generate_donut(data_size=DATA_SIZE):
    # np.random.seed(1)
    donut = []

    while len(donut) != data_size:
        p = np.random.uniform(-2, 2, 2)
        if 4 >= np.square(p).sum() >= 2:
            donut.append(p)
    return np.array(donut)


def generateDisk(data_size=DATA_SIZE):
    np.random.seed(1)
    radius = np.random.uniform(0, 1, data_size)
    theta = np.random.uniform(0, 2 * np.pi, data_size)
    x = np.sqrt(radius) * np.cos(theta)
    y = np.sqrt(radius) * np.sin(theta) * 0.5
    return np.array([x, y]).T


def PartA():
    # Q1.a
    data = generateDisk()
    model = SOM(rows=1, cols=100, radius=2, learning_rate=0.9)
    model.train(data, debug=False, epochs=10,
                fig_path='E:\\Git\\Neuro-Computation\\SOM\\partA\\disk_images\\100_neurons_in_line')

    # Q1.b
    model = SOM(rows=10, cols=10, radius=2, learning_rate=0.9)
    model.train(data, debug=False, epochs=10,
                fig_path='E:\\Git\\Neuro-Computation\\SOM\\partA\\disk_images\\10x10_neurons')

    # Q1.c
    data = generate_unbalanced()
    model = SOM(rows=10, cols=10, radius=2, learning_rate=0.9)
    model.train(data, debug=False, epochs=10,
                fig_path='E:\\Git\\Neuro-Computation\\SOM\\partA\\disk_images\\unbalanced\\a')
    data = generate_unbalanced(a=False)
    model = SOM(rows=10, cols=10, radius=2, learning_rate=0.4)
    model.train(data, debug=False, epochs=40,
                fig_path='E:\\Git\\Neuro-Computation\\SOM\\partA\\disk_images\\unbalanced\\b')

    # Q1.d
    data = generate_donut()
    model = SOM(rows=1, cols=30, radius=2, learning_rate=0.4)
    model.train(data, debug=False, epochs=10,
                fig_path='E:\\Git\\Neuro-Computation\\SOM\\partA\\disk_images\\donut')

def PartB():


    
if __name__ == '__main__':
    PartA()
