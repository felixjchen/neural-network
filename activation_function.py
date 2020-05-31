import numpy as np


class Sigmoid():
    def f(self, z):
        """ f: Real -> (0,1) """
        return 1 / (1 + np.exp(-z))

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
