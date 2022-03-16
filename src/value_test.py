import random

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import mnist_loader
import network

sizes = [2,3,1]

def random_gaussion(sizes):
    return [np.random.randn(y, 1) for y in sizes[1:]]

# print([np.random.randn(y, 1) for y in sizes[1:]])
# print([np.random.randn(y, x)
#                         for x, y in zip(sizes[:-1], sizes[1:])])

# for i in range(3):
#     print(np.random.randn(5,1))

# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()

training_data, validation_data, test_data=mnist_loader.load_data_wrapper()
net = network.Network([784,100,10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)