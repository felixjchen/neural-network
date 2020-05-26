import numpy as np

np.random.seed(1)


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

    def feedforward(self, a):

        for w, b in zip(self.weights, self.bias):
            a = sigmoid(w @ a + b)
        return a

    def SGD(self, training_set, validation_set, epochs=10, minibatch_size=1000, eta=10):

        n = len(training_set)

        for e in range(epochs):

            np.random.shuffle(training_set)
            minibatches = [training_set[m:m+minibatch_size]
                           for m in range(0, n, minibatch_size)]

            for batch in minibatches:
                self.update(batch, eta)

                r = [(np.argmax(self.feedforward(x)) == np.argmax(y))
                     for (x, y) in validation_set]
                print(sum(r)/len(validation_set) * 100)

    def update(self, batch, eta):

        partial_b = [np.zeros(b.shape) for b in self.bias]
        partial_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            add_partial_b, add_partial_w = self.backprop(x, y)
            partial_b = [b + db for b, db in zip(partial_b, add_partial_b)]
            partial_w = [w + dw for w, dw in zip(partial_w, add_partial_w)]

        m = len(batch)
        c = eta/m

        self.bias = [b - c*db for b, db in zip(self.bias, partial_b)]
        self.weights = [w - c*dw for w, dw in zip(self.weights, partial_w)]

    def backprop(self, x, y):

        partial_b = [np.zeros(b.shape) for b in self.bias]
        partial_w = [np.zeros(w.shape) for w in self.weights]

        # Feed forward, caching each layer's activation a in A and z in Z
        Z = []
        activations = [x]
        for w, b in zip(self.weights, self.bias):
            z = w @ activations[-1] + b

            Z += [z]
            activations += [sigmoid(z)]

        # Find error in last layer
        n = self.num_layers

        deltas = [None for _ in range(n-1)]
        deltas[-1] = (activations[-1] - y) * sigmoid_prime(Z[-1])
        partial_b[-1] = deltas[-1]
        partial_w[-1] = deltas[-1] @ activations[-2].T

        for l in range(self.num_layers - 3, -1, -1):
            deltas[l] = self.weights[l+1].T @ deltas[l+1] * sigmoid_prime(Z[l])
            partial_b[l] = deltas[l]
            partial_w[l] = deltas[l] * activations[l].T

        return partial_b, partial_w
