import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization
from copy import deepcopy


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.layers = ['FullyConnectedLayer1', 'ReLULayer', 'FullyConnectedLayer2']
        self.FullyConnectedLayer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.ReLULayer = ReLULayer()
        self.FullyConnectedLayer2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for name_of_layer in self.layers:
            layer = getattr(self, name_of_layer)
            for param in layer.params().values():
                param.grad = np.zeros_like(param.value)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        input_value = X.copy()
        for name_of_layer in self.layers:
            layer = getattr(self, name_of_layer)
            input_value = deepcopy(layer.forward(input_value))
        loss_cross_entropy, dpred = softmax_with_cross_entropy(input_value, y)

        # backward pass for cross entropy
        input_grad = deepcopy(dpred)
        for name_of_layer in self.layers[::-1]:
            layer = getattr(self, name_of_layer)
            input_grad = layer.backward(input_grad)

        loss_reg = 0
        for name_of_layer in self.layers:
            layer = getattr(self, name_of_layer)
            for param in layer.params().values():
                param.grad += 2 * self.reg * param.value
                loss_reg += np.sum(param.value ** 2)
        loss_reg *= self.reg

        loss = loss_cross_entropy + loss_reg
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        input_value = X
        for name_of_layer in self.layers:
            layer = getattr(self, name_of_layer)
            input_value = layer.forward(input_value)

        pred = np.argmax(input_value, axis=1)
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        for name_of_layer in self.layers:
            layer = getattr(self, name_of_layer)
            for name, value in layer.params().items():
                name = f"{name_of_layer}_{name}"
                result[name] = value

        return result
