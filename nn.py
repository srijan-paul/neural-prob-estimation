from typing import List, Tuple
import numpy as np


def sigmoid(z: np.ndarray):
    return 1 / (1 + np.exp(-z))


def d_sigmoid(z: np.ndarray):
    sigmoid_z = sigmoid(z)
    return sigmoid_z * (1 - sigmoid_z)


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
        # The input layer does not have any weights
        self.weights = []
        for l in range(1, self.num_layers):
            nrows, ncols = layer_sizes[l], layer_sizes[l - 1]
            ws = np.random.rand(nrows, ncols)
            self.weights.append(ws)

        # initialize biases
        # self.biases[0] = biases for the first hidden layer
        # Input layer does not have any biases
        self.biases = []
        for n_neurons in layer_sizes[1:]:
            bs = np.random.rand(n_neurons, 1)
            self.biases.append(bs)

    def activation(self, z: np.ndarray) -> np.ndarray:
        return sigmoid(z)

    def d_activation(self, z: np.ndarray) -> np.ndarray:
        return d_sigmoid(z)

    def generate_batch(
        xs: np.ndarray, ys: np.ndarray, np: np.ndarray, batch_size: int
    ) -> Tuple[int, int]:
        """Generate a batch for stochastic gradient descent.
        Calling this on an array will shuffle the array in-place,
        then return the first `batch_size` items in it"""

        assert batch_size > 0 and batch_size < len(xs) and len(xs) == len(ys)
        np.random.shuffle(xs)
        return (xs[:batch_size], ys[:batch_size])

    def cost(self, pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Quadratic cost function"""
        assert (
            pred.shape == y.shape
        ), "prediction and target must have the same dimensions"
        return ((pred - y) ** 2) / 2

    def loss_grad_pred(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        return target - pred

    def loss_grad_z(self, weighted_sum_l: np.ndarray, loss_grad_a: np.ndarray):
        """
        Gradient of the loss with respect to weighted-sum input to the final layer

        Parameters
        ----------
        `weighted_sum_l` : Weighted sum input to the final layer
        `loss_grad_a` : Gradient of the loss w.r.t the final layer activations
        """
        return d_sigmoid(weighted_sum_l) * loss_grad_a

    def train(self, xs: np.ndarray, ys: np.ndarray, num_epochs=1000):
        assert (
            isinstance(num_epochs, int) and num_epochs > 0
        ), "#epochs must be a positive integer"

        for _ in range(num_epochs):
            (x_batch, y_batch) = self.generate_batch(xs, ys)
            for (train_sample, target) in zip(x_batch, y_batch):
                prediction, activations, weighted_sums = self.forward_pass(train_sample)
                # C (cost)
                loss = self.cost(prediction, target)
                # dC/da_L
                loss_grad_a = self.loss_grad_pred(prediction, target)
                # dC/dz_L
                loss_grad_zL = self.loss_grad_z(weighted_sums[-1], loss_grad_a)


    def forward_pass(self, x: np.ndarray, store_values=True) -> np.ndarray:
        # activation at layer 0 (input layer) are the features of the training sample
        # itself.
        current_activation = x
        activations = [x]
        weighted_sums = []
        for l in range(self.num_layers - 1):
            weights_l = self.weights[l]
            biases_l = self.biases[l]
            z = np.dot(weights_l, current_activation) + biases_l
            if store_values:
                weighted_sums.append(z)
            current_activation = self.activation(z)
            if store_values:
                activations.append(current_activation)
        return current_activation, activations, weighted_sums

    def backprop_losses(self):
        """
        """
        pass


