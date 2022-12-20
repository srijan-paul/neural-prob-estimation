from math import ceil
from typing import List, Tuple
import numpy as np


class Network:
    num_layers: int
    layer_sizes: List[int]
    learn_rate: float
    weights: np.ndarray
    biases: np.ndarray

    def __init__(self, layer_sizes: List[int], learn_rate=0.01):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.learn_rate = learn_rate

        # initialize weights
        # self.weights[0] = weights between first hidden layer and input layer
        self.weights = []
        for l in range(1, self.num_layers):
            nrows, ncols = layer_sizes[l], layer_sizes[l - 1]
            ws = np.random.rand(nrows, ncols)
            self.weights.append(ws)

        # initialize biases
        # self.biases[0] = biases for the first hidden layer
        self.biases = []
        for n_neurons in layer_sizes[1:]:
            bs = np.random.rand(n_neurons, 1)
            self.biases.append(bs)

    def activation(self, z: np.ndarray) -> np.ndarray:
        return 1 / 1 + np.exp(-z)

    def generate_batch(
        xs: np.ndarray, ys: np.ndarray, np: np.ndarray, batch_size: int
    ) -> Tuple[int, int]:
        """Generate a batch for stochastic gradient descent.
        Calling this on a numpy array will shuffle the array in-place,
        then return the first `batch_size` items in it"""
        assert batch_size > 0 and batch_size < len(xs) and len(xs) == len(ys)
        np.random.shuffle(xs)
        return (xs[:batch_size], ys[:batch_size])

    def train(self, xs: np.ndarray, ys: np.ndarray, num_epochs=1000):
        assert isinstance(num_epochs, int), "#epochs must be an integer"
        for epoch in range(num_epochs):
            for (x_, y_) in zip(xs, ys):
                (x, y) = self.generate_batch(x_, y_)
                print(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        a = x  # current activation at layer 0 (input layer)
        for l in range(self.num_layers - 1):
            weights_l = self.weights[l]
            biases_l = self.biases[l]
            z = np.dot(weights_l, a) + biases_l
            a = self.activation(z)
        return a


nn = Network([3, 2, 2])
print(nn.predict(np.array([[1], [2], [3]])))
