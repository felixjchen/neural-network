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
    def f(self, z):
        """ f: Real -> (-inf,inf) """
        return z

    def f_prime(self, z):
        return 1


class ReLU():
    def f(self, z):
        """ f: Real -> [0,inf) """
        z = z.copy()
        z[z < 0] = 0
        return z

    def f_prime(self, z):
        """ f: Real -> [0,inf) """
        z = z.copy()
        z[z >= 0] = 1
        z[z < 0] = 0
        return z
