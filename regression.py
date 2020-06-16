from neural_network import NeuralNetwork
from activation_function import Sigmoid, Softmax, Linear, ReLU, ReLU6, LeakyRelu6
from loss_function import Quadratic, CrossEntropy

import numpy as np
import matplotlib.pyplot as plt

n = 1000
X = np.linspace(-12, 12, num=n)
noise = np.random.normal(0, 0.1, n)
y = -(3*np.cos(X) - X) + noise

plt.scatter(X, y)
plt.show()

trainingSetSize = int(0.8*n)
trainIdxs = np.random.choice(n, trainingSetSize, replace=False)
valIdxs = np.array(list(set(np.arange(n)) - set(trainIdxs)))

train_X, train_y = X[trainIdxs][:, None], y[trainIdxs][:, None]
val_X, val_y = X[valIdxs][:, None], y[valIdxs][:, None]
# plt.scatter(train_X, train_y)
# plt.scatter(val_X, val_y, color="red")
# plt.show()

model = NeuralNetwork(size=[1, 50, 50, 1],
                      activation=[ReLU6(), ReLU6(), Linear()],
                      loss=Quadratic(),
                      regression=True)

model.SGD(train_X, train_y, val_X, val_y, epochs=500,
          batch_percent=0.05, eta=0.002, lmbda=0.5, verbose=False)
modelY = model.feedforward(val_X)

plt.scatter(train_X, train_y)
# plt.scatter(val_X, val_y, color="red")
plt.scatter(val_X, modelY, color="yellow")
plt.show()
