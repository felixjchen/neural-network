from lib.neural_network import NeuralNetwork
from lib.activation_function import Sigmoid, Softmax, Linear, ReLU, ReLU6, LeakyRelu, LeakyRelu6

from loss_function import Quadratic, CrossEntropy

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

n = 10000
X = np.linspace(-15, 15, num=n)
noise = np.random.normal(0, 0.1, n)
y = 4*np.sin(X/3) + 3*np.cos(X) - X + noise


trainingSetSize = int(0.8*n)
trainIdxs = np.random.choice(n, trainingSetSize, replace=False)
valIdxs = np.array(list(set(np.arange(n)) - set(trainIdxs)))

train_X, train_y = X[trainIdxs][:, None], y[trainIdxs][:, None]
val_X, val_y = X[valIdxs][:, None], y[valIdxs][:, None]

model = NeuralNetwork(size=[1, 50, 50, 1],
                      activation=[LeakyRelu(), LeakyRelu(), Linear()],
                      loss=Quadratic(),
                      regression=True)

model.SGD(train_X, train_y, val_X, val_y, epochs=200,
          batch_percent=0.0005, eta=0.001, lmbda=0.5, verbose=True)
modelY = model.feedforward(val_X)

plt.title("f(x) = 3cosx + 4sin(x/3) - x + Normal(0, 0.1)")
plt.scatter(train_X, train_y, label="data")
plt.scatter(val_X, modelY, color="yellow", label='model')
plt.legend(loc="upper right")
plt.show()
