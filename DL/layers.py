import numpy as np
from utils import transfrom_1Darray_to_2D
from math import sqrt
from copy import deepcopy


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''

    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    assert isinstance(predictions, np.ndarray)

    initial_num_of_dimensions = predictions.ndim
    x = transfrom_1Darray_to_2D(predictions)

    max_ = np.max(x, axis=1)
    x = x - max_[:, np.newaxis]
    x = np.exp(x)
    sum_ = np.sum(x, axis=1)
    x = x / sum_[:, np.newaxis]
    if (initial_num_of_dimensions == 1):
        x = np.squeeze(x, axis=0)
    # gc.collect()
    return x


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    assert isinstance(probs, np.ndarray)
    assert isinstance(target_index, np.ndarray)

    X = transfrom_1Darray_to_2D(probs)
    y = target_index
    batch_size, number_of_features = X.shape



    loss = X[(np.arange(batch_size), y)]
    loss = -np.log(loss)
    loss = np.sum(loss) / batch_size

    #gc.collect()
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    assert predictions.ndim <= 2, "predictions must be a numpy array with no more than two dimensions"
    assert isinstance(predictions, np.ndarray)
    initial_num_of_dimensions = predictions.ndim

    probs = softmax(predictions)
    probs = transfrom_1Darray_to_2D(probs)
    loss = cross_entropy_loss(probs, target_index)

    predictions = transfrom_1Darray_to_2D(predictions)
    batch, N = predictions.shape

    sum_exp = np.sum(probs, axis=1)[:, np.newaxis]
    dprediction1 = np.zeros((batch, N))
    dprediction1[(np.arange(batch), target_index)] = probs[(np.arange(batch), target_index)].copy()
    dprediction2 = probs.copy()
    dprediction2 *= probs[(np.arange(batch), target_index)][:, np.newaxis]
    dprediction = dprediction1 - dprediction2
    dprediction = -dprediction / probs[(np.arange(batch), target_index)][:, np.newaxis]
    dprediction /= batch

    if (initial_num_of_dimensions == 1):
        dprediction = np.squeeze(dprediction, axis=0)

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self._value = value
        self._grad = np.zeros_like(value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, var):
        self._value = var

    @value.deleter
    def value(self):
        del self._value

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, var):
        assert self.grad.shape == var.shape, "Shape of gradient must not be changed. However, shape of " \
                                             f"gradient={self.grad.shape}, shape of the value you tried to" \
                                             f" assign={var.shape}"

        self._grad = var

    @grad.deleter
    def grad(self):
        del self._grad


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.zero_grad = X >= 0
        return np.maximum(X, np.zeros_like(X))

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = d_out * self.zero_grad
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(np.random.normal(0, sqrt(2 / n_input), (n_input, n_output)))
        self.B = Param(np.random.normal(0, sqrt(2 / n_input), n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = deepcopy(X)
        return np.matmul(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        batch_size, n_output = d_out.shape
        _, n_input = self.X.shape

        self.B.grad += np.sum(d_out, axis=0)

        d_input = np.zeros((batch_size, n_input))
        for i in range(batch_size):
            self.W.grad += np.repeat(self.X[i, :][:, np.newaxis], n_output, axis=1) * d_out[[i], :]
            d_input[i, :] += np.sum(self.W.value * d_out[[i], :], axis=1)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
