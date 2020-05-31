import numpy as np

np.random.seed(0)


class NeuralNetwork():

    def __init__(self, size, activation, loss):
        """
        Initialize a neural network with shape defined by size, layer activations defined by activation and loss function by loss.

        size :: [Int]
        - index 0 has size of input layer, index -1 has size of output layer, indexs [1, -2] have hidden layer sizes

        activation :: [ActivationFunction]
        - index -1 has activation function for output layer, indexs [0, -2] have activation functions for hidden layers

        loss :: LossFunction
        """

        self.num_layers = len(size)
        self.size = size

        self.loss_function = loss

        self.bias = [np.random.randn(i, 1) for i in size[1:]]
        self.weights = [np.random.randn(i, o)/np.sqrt(i)
                        for i, o in zip(size[1:], size[:-1])]
        self.activation_functions = activation

    def feedforward(self, A):
        # A is activation values in network at current layer
        for a, w, b in zip(self.activation_functions, self.weights, self.bias):
            A = a.f(A @ w.T + b.T)
        return A

    def predict(self, X):
        """
        For n training examples, feedforward each example and return the index with maximum activation. 
        """
        pred = self.feedforward(X)
        return np.argmax(pred, axis=1)

    def get_accuracy(self, X, y):
        """
        Get's the accuracy of the 
        """
        pred = self.predict(X)
        actual = np.argmax(y, axis=1)
        return round(sum(pred == actual) / len(X) * 100, 3)

    def SGD(self, train_X, train_y, val_X, val_y, epochs=30, batch_percent=0.0002, eta=1, lmbda=0.5):
        """ Stochastic gradient descent, 

        Note when batch_perecent = 1, this would be considered gradient descent.
        """
        assert 0 < batch_percent <= 1, "batch percent invalid"

        n = len(train_X)
        num_batches = 1 / batch_percent

        for e in range(epochs):

            p = np.random.permutation(n)
            train_X, train_y = train_X[p], train_y[p]
            batches_X, batches_y = np.array_split(
                train_X, num_batches), np.array_split(train_y, num_batches)

            for batch_X, batch_y in zip(batches_X, batches_y):
                self.update(batch_X, batch_y, eta, n, lmbda)

            # Get regularized loss
            loss = self.loss_function.get_loss(self.feedforward(
                val_X), val_y) + (lmbda/(2*n))*np.sum(np.linalg.norm(w)**2 for w in self.weights)
            loss = round(loss, 3)

            print(
                f"Epoch {e}: Loss {loss}, Validation accuracy {self.get_accuracy(val_X, val_y)}%")

    def update(self, X, y, eta, n, lmbda):

        c = eta/len(X)

        grad_b, grad_w = self.backprop(X, y)

        self.bias = [b - c*db for b, db in zip(self.bias, grad_b)]
        # Regularize
        self.weights = [(1-(eta*lmbda)/n)*w - c*dw for w,
                        dw in zip(self.weights, grad_w)]

    def backprop(self, X, y):
        """ For all mxj training inputs and mxk labels, compute the sum of grads w.r.t to bias and weights for each node"""

        # Feed forward, if a layer has k nodes, Z is a list of nxk z values and A is a list of nxk activation values
        Z = []
        A = [X]
        for a, w, b in zip(self.activation_functions, self.weights, self.bias):
            z = A[-1] @ w.T + b.T
            Z += [z]
            A += [a.f(z)]

        n = self.num_layers
        grad_b = [None for b in self.bias]
        grad_w = [None for w in self.weights]

        # Find error in last layer
        # Deltas is a list of nxk errors for a layer with k nodes
        deltas = [None for _ in range(n-1)]
        deltas[-1] = self.loss_function.last_layer_error(
            A[-1], y, Z[-1], self.activation_functions[-1])

        grad_b[-1] = deltas[-1]
        grad_b[-1] = np.sum(grad_b[-1], axis=0)[:, None]
        grad_w[-1] = deltas[-1][:, :, None] @ A[-2][:, None, :]
        grad_w[-1] = np.sum(grad_w[-1], axis=0)

        # Propagate error backward and solve for gradients
        for l in range(2, n):
            deltas[-l] = (deltas[-l+1] @ self.weights[-l+1]) * \
                self.activation_functions[-l].f_prime(Z[-l])

            grad_b[-l] = deltas[-l]
            grad_b[-l] = np.sum(grad_b[-l], axis=0)[:, None]
            grad_w[-l] = deltas[-l][:, :, None] @ A[-l-1][:, None, :]
            grad_w[-l] = np.sum(grad_w[-l], axis=0)

        return grad_b, grad_w
