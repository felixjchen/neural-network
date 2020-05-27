import numpy as np

np.random.seed(0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))


class NeuralNetwork():

    def __init__(self, size):

        self.num_layers = len(size)
        self.size = size

        self.bias = [np.random.randn(i, 1) for i in size[1:]]
        self.weights = [np.random.randn(i, o)
                        for i, o in zip(size[1:], size[:-1])]

    def feedforward(self, A):
        for w, b in zip(self.weights, self.bias):
            A = A @ w.T + b.T
        return A

    def predict(self, A):
        pred = self.feedforward(A)

        return np.argmax(pred, axis=1)

    def SGD(self, train_X, train_y, val_X, val_y, epochs=30, batch_percent=0.0002, eta=3):
        """ Stochastic gradient descent, """
        assert 0 < batch_percent <= 1, "batch percent invalid"

        n = len(train_X)
        num_batches = 1 / batch_percent

        for e in range(epochs):

            p = np.random.permutation(n)
            train_X, train_y = train_X[p], train_y[p]
            batches_X, batches_y = np.array_split(
                train_X, num_batches), np.array_split(train_y, num_batches)

            for batch_X, batch_y in zip(batches_X, batches_y):
                self.update(batch_X, batch_y, eta)

            pred = self.predict(val_X)
            actual = np.argmax(val_y, axis=1)
            print(sum(pred == actual) / len(val_X) * 100)

    def update(self, X, y, eta):

        c = eta/len(X)

        grad_b, grad_w = self.backprop(X, y)

        self.bias = [b - c*db for b, db in zip(self.bias, grad_b)]
        self.weights = [w - c*dw for w, dw in zip(self.weights, grad_w)]

    def backprop(self, X, y):
        """ For all mxj training inputs and mxk labels, compute the sum of grads w.r.t to bias and weights for each node"""
        # size of minibatch
        m = len(X)

        # Feed forward, if a layer has k nodes, Z is a list of nxk z values and A is a list of nxk activation values
        Z = []
        A = [X]
        for w, b in zip(self.weights, self.bias):
            z = A[-1] @ w.T + b.T
            Z += [z]
            A += [sigmoid(z)]

        grad_b = [None for b in self.bias]
        grad_w = [None for w in self.weights]

        # Find error in last layer
        n = self.num_layers

        # Deltas is a list of nxk errors for a layer with k nodes
        deltas = [None for _ in range(n-1)]
        deltas[-1] = (A[-1] - y) * sigmoid_prime(Z[-1])

        grad_b[-1] = deltas[-1]
        grad_b[-1] = np.sum(grad_b[-1], axis=0)[:, None]
        grad_w[-1] = deltas[-1][:, :, None] @ A[-2][:, None, :]
        grad_w[-1] = np.sum(grad_w[-1], axis=0)

        # Propagate error backward and solve for gradients
        for l in range(2, n):
            deltas[-l] = (deltas[-l+1] @ self.weights[-l+1]) * \
                sigmoid_prime(Z[-l])

            grad_b[-l] = deltas[-l]
            grad_b[-l] = np.sum(grad_b[-l], axis=0)[:, None]
            grad_w[-l] = deltas[-l][:, :, None] @ A[-l-1][:, None, :]
            grad_w[-l] = np.sum(grad_w[-l], axis=0)

        # for i in range(0, n-1):
        #     print(grad_b[i].shape, self.bias[i].shape)
        #     print(grad_w[i].shape, self.weights[i].shape)
        return grad_b, grad_w
