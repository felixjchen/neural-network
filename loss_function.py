import numpy as np
from helpers import sigmoid, sigmoid_prime


class Quadratic():
    @staticmethod
    def get_loss(a_last, y):
        """ Returns the quadratic loss over all training examples,a_last nxk is the activations in the last layer and y nxk are the corresponding labels. For n inputs and label size k"""
        n, _ = y.shape
        norms = np.linalg.norm(a_last-y, axis=1) ** 2
        return np.sum(norms) / (n * 2)

    @staticmethod
    def get_error(a_last,  z_last, y):
        """ Given activations for final layer a_last, z_last for final layer values and y labels, return the error in the last layer"""

        return (a_last - y) * sigmoid_prime(z_last)


class CrossEntropy():
    @staticmethod
    def get_loss(a_last, y):
        """ Returns the CrossEntropy loss over all training examples,a_last nxk is the activations in the last layer and y nxk are the corresponding labels. For n inputs and label size k"""
        n, k = y.shape
        lna = np.log(a_last)
        ln1_a = np.log(np.ones((n, k)) - a_last)
        C = (y * lna) + (ln1_a) - (y * ln1_a)
        C = np.sum(C)
        return round(-C/n, 3)

    @staticmethod
    def get_error(a_last,  z_last, y):
        """ Given activations for final layer a_last, z_last for final layer values and y labels, return the error in the last layer"""

        return (a_last - y)
