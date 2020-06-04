import numpy as np


class Sigmoid():
    def f(self, z):
        """ f: Real -> (0,1) """
        return 1 / (1 + np.exp(-z))

    def f_prime(self, z):
        return self.f(z) * (1-self.f(z))


class Softmax():
    def f(self, z):
        """ f: Real -> [0,1] """
        _, k = z.shape

        ez = np.exp(z)
        normalize_factor = np.sum(ez, axis=1)[:, None]
        normalize_factor = np.repeat(normalize_factor, k, axis=1)
        return ez / normalize_factor

    def f_prime(self, z):
        return self.f(z) * (1-self.f(z))


class Linear():
    def __init__(self, c=1):
        self.c = c

    def f(self, z):
        """ f: Real -> (-inf,inf) """
        return self.c*z

    def f_prime(self, z):
        return self.c


class ReLU():
    def __init__(self, c=1):
        self.c = c

    def f(self, z):
        """ f: Real -> [0,inf) """
        z = z.copy()
        z[z < 0] = 0
        return self.c*z

    def f_prime(self, z):
        """ f: Real -> [0,inf) """
        z = z.copy()
        z[z >= 0] = self.c
        z[z < 0] = 0
        return z
