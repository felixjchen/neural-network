from neural_network import NeuralNetwork
from activation_function import Sigmoid, Softmax, Linear, ReLU, ReLU6, LeakyRelu, LeakyRelu6
from loss_function import Quadratic, CrossEntropy

from pickle import _Unpickler
import gzip
import numpy as np

np.random.seed(0)

f = gzip.open('dataset/mnist.pkl.gz', 'rb')
u = _Unpickler(f)
u.encoding = 'latin1'

TRAINING, VALIDATION, TESTING = u.load()


def preprocess(data):
    x, y = data[0], data[1]
    # One hot
    y = np.eye(10)[y]
    return x, y


train_X, train_y = preprocess(TRAINING)
val_X, val_y = preprocess(VALIDATION)

model = NeuralNetwork(size=[784, 30, 10],
                      activation=[ReLU(), Sigmoid()],
                      loss=CrossEntropy())

# Eta = 0.5, lmbda=0.5 for crossentropy loss
# Eta = 3 for quadratic loss
model.SGD(train_X, train_y, val_X, val_y, epochs=10,
          batch_percent=0.0002, eta=0.05, lmbda=0.5)
