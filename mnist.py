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
    return x, y


train_X, train_y = preprocess(TRAINING)
val_X, val_y = preprocess(TRAINING)

model = NeuralNetwork([784, 30,  10])
model.SGD(train_X, train_y, val_X, val_y)
