from neural_network import NeuralNetwork
from pickle import _Unpickler
import gzip
import numpy as np

f = gzip.open('mnist.pkl.gz', 'rb')
u = _Unpickler(f)
u.encoding = 'latin1'

TRAINING, VALIDATION, TESTING = u.load()


def preprocess(data):
    x, y = data[0], data[1]
    # One hot
    y = np.eye(10)[y]
    # Make into 2d array
    data = [(nx[:,None], ny[:,None]) for nx, ny in zip(x, y)]
    return data

TRAINING = preprocess(TRAINING)
VALIDATION = preprocess(VALIDATION)

model = NeuralNetwork([784, 784, 784, 10])
model.SGD(TRAINING, VALIDATION)
