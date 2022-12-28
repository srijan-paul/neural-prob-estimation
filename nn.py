from typing import List, Tuple
import numpy as np
import json


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

    def __init__(self, layer_sizes: List[int], learn_rate=0.1):
        self.num_layers = len(layer_sizes)
        assert self.num_layers >= 2
        self.layer_sizes = layer_sizes
        self.learn_rate = learn_rate

        # initialize weights
        # self.weights[0] = weights between first hidden layer and input layer
        # The input layer does not have any weights
        self.weights = []
        for l in range(1, self.num_layers):
            nrows, ncols = layer_sizes[l], layer_sizes[l - 1]
            ws = np.zeros((nrows, ncols))
            self.weights.append(ws)

        # initialize biases
        # self.biases[0] = biases for the first hidden layer
        # Input layer does not have any biases
        self.biases = []
        for n_neurons in layer_sizes[1:]:
            bs = np.zeros((n_neurons, 1))
            self.biases.append(bs)

    def activation(self, z: np.ndarray) -> np.ndarray:
        return sigmoid(z)

    def d_activation(self, z: np.ndarray) -> np.ndarray:
        return d_sigmoid(z)

    def generate_batch(
        self, xs: np.ndarray, ys: np.ndarray, batch_size: int
    ) -> Tuple[int, int]:
        """Generate a batch for stochastic gradient descent.
        Calling this on an array will shuffle the array in-place,
        then return the first `batch_size` items in it"""

        assert batch_size > 0 and batch_size <= len(xs) and len(xs) == len(ys)
        np.random.shuffle(xs)
        return (xs[:batch_size], ys[:batch_size])

    def cost(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Quadratic cost function"""
        assert (
            pred.shape == target.shape
        ), f"prediction and target must have the same dimensions ({pred.shape}, {target.shape})"
        return ((target - pred) ** 2) / 2

    def loss_grad_pred(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        return pred - target

    def loss_grad_z(self, weighted_sum_l: np.ndarray, loss_grad_a: np.ndarray):
        """
        Gradient of the loss with respect to weighted-sum input to the final layer

        Parameters
        ----------
        `weighted_sum_l` : Weighted sum input to the final layer
        `loss_grad_a` : Gradient of the loss w.r.t the final layer activations
        """
        assert weighted_sum_l.shape == loss_grad_a.shape
        return d_sigmoid(weighted_sum_l) * loss_grad_a

    def update_weights(self, layerwise_losses: np.ndarray, activations: np.ndarray):
        for l in range(self.num_layers - 2, -1, -1):
            loss_l = layerwise_losses[l]
            a_l_prev = activations[l]

            del_w_l = self.learn_rate * np.dot(loss_l, np.transpose(a_l_prev))
            self.weights[l] = self.weights[l] - del_w_l

            del_b_l = self.learn_rate * loss_l
            self.biases[l] = self.biases[l] - del_b_l

    def train(self, xs: np.ndarray, ys: np.ndarray, batch_size: int, num_epochs=1000):
        assert (
            isinstance(num_epochs, int) and num_epochs > 0
        ), "#epochs must be a positive integer"

        for _ in range(num_epochs):
            (x_batch, y_batch) = self.generate_batch(xs, ys, batch_size)
            avg_loss = 0
            for (train_sample, target) in zip(x_batch, y_batch):
                output, activations, weighted_sums = self.forward_pass(
                    train_sample)
                assert (
                    output.shape == target.shape
                ), "Network output and target have different dimensions"
                avg_loss += (batch_size / len(xs)) * self.cost(output, target)
                layerwise_losses = self.backprop_losses(
                    output, weighted_sums, target)
                self.update_weights(layerwise_losses, activations)
            print(f"Epoch #{_ + 1}: loss = {avg_loss}")

    def forward_pass(self, x: np.ndarray, store_values=True) -> np.ndarray:
        """
        Runs the forward pass of the network in the input featureset `x`,
        then returns the activations of the final layer.
        If the `store_values` parameter is set to `true`, it also returns the weighted-sum inputs and activations of each layer.

        Parameters
        ----------
        `x`: An input training sample. (An np.ndarray of features)

        `store_values`: If set to `True`, this function will also return two additional lists:
            1. A list `z` where `z[l]` is an np.ndarray representing the weighted sum inputs of neurons in layer l.
            2. A list `a` where `a[l]` is an np.ndarray representing the activations of the neurons in layer l.
        """
        # activation at layer 0 (input layer) are
        # the features of the training sample itself.

        assert x.shape[1] == 1, "input layer must have dimension n x 1 for n > 0"

        current_activation = x
        activations = [x]
        weighted_sums = []
        for l in range(self.num_layers - 1):
            weights_l = self.weights[l]
            biases_l = self.biases[l]
            z = np.dot(weights_l, current_activation) + biases_l

            assert z.shape[1] == 1

            current_activation = self.activation(z)
            if store_values:
                weighted_sums.append(z)
                activations.append(current_activation)

        if store_values:
            return current_activation, activations, weighted_sums
        return current_activation

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward_pass(x, False)

    def backprop_losses(
        self,
        prediction: np.ndarray,
        weighted_sums: List[np.ndarray],
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Applies the backpropagation algorithm by following these steps:
        [TODO: write the steps]

        Parameters
        ----------
        `loss`: Output of the cost function based on the network's prediction
        `prediction`: Prediction made by the neural network.
        `weighted_sums`: A list where the `lth` element is an np.ndarray representing z_l (weighted sum input to neurons at layer l).
        `activations`: A list where the `lth` element is an np.ndarray representing a_l (activation values of inputs to neurons at layer l).
        `target`: Target output
        """
        loss_grad_a = self.loss_grad_pred(prediction, target)

        # calculate gradient of loss w.r.t weighted-sum input of the current layer.
        # We start with the final layer as the current layer and work backwards.
        loss_grad_zl = self.loss_grad_z(weighted_sums[-1], loss_grad_a)

        losses = [loss_grad_zl]
        # Now, we go backwards from layer L to layer 1, and
        # calculate the loss vector for every layer.
        for l in range(self.num_layers - 3, -1, -1):
            w_lnext = self.weights[l + 1]
            z_l = weighted_sums[l]
            loss_grad_zl = self.d_activation(z_l) * np.dot(
                np.transpose(w_lnext),
                loss_grad_zl)
            losses.append(loss_grad_zl)

        losses.reverse()
        return losses


domain = np.linspace(-0.5, 0.5, 500)
train_xs = np.array([[[i]] for i in domain])
train_ys = np.array([[[i * 2]] for i in domain])

# TODO: Using [1, 2, 3, 1] as the layer sizes crashes the training loop.
# Investigate why
nn = Network([1, 2, 1], 1)
nn.train(train_xs, train_ys, 100, 1000)
print(nn.predict(np.array([[0.2]])))
