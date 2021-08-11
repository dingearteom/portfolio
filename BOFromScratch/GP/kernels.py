import numpy as np
from abc import ABCMeta
import functools
from typing import final
from GP.utils import check_arrays


def get_bounds(self, other):
    return {**self.get_bounds(), **other.get_bounds()}


def get_initial_values(self, other):
    return {**self.get_initial_values(), **other.get_initial_values()}


class Kernel(metaclass=ABCMeta):
    @final
    def __init__(self, *args, **kwargs):
        self.params = kwargs
        self.kernels = args

    def apply(self, X, Y=None, diag=False):
        raise NotImplemented

    @final
    def set_params(self, theta):
        self.params.update(theta)
        for kernel in self.kernels:
            kernel.set_params(theta)

    @final
    def get_params(self):
        return self.params.copy()

    @final
    def get_params_names(self):
        return list(self.params.copy().keys())

    @staticmethod
    def get_bounds():
        raise NotImplemented

    @staticmethod
    def get_initial_values():
        raise NotImplemented

    def __add__(self, other):
        res = Kernel(self, other, **self.params, **other.params)

        def apply(*args, **kwargs):
            return self.apply(*args, **kwargs) + other.apply(*args, **kwargs)

        res.apply = apply
        res.get_bounds = functools.partial(get_bounds, self, other)
        res.get_initial_values = functools.partial(get_initial_values, self, other)
        return res

    def __mul__(self, other):
        res = Kernel(self, other, **self.params, **other.params)

        def apply(*args, **kwargs):
            return self.apply(*args, **kwargs) * other.apply(*args, **kwargs)

        res.apply = apply
        res.get_bounds = functools.partial(get_bounds, self, other)
        res.get_initial_values = functools.partial(get_initial_values, self, other)
        return res


class WhiteKernel(Kernel):
    def apply(self, X, Y=None, diag=False):
        sigma_noise = self.params['sigma_noise']

        check_arrays(X, Y)

        if Y is not None:
            if diag:
                assert X.shape[0] == Y.shape[0], 'If diag is True first dimensions of X and Y must be the same'
                return np.zeros((X.shape[0]))
            else:
                # X and Y are considered to be distinct
                return np.zeros((X.shape[0], Y.shape[0]))
        else:
            if diag:
                return np.ones((X.shape[0])) * (sigma_noise ** 2)
            else:
                return np.eye(X.shape[0]) * (sigma_noise ** 2)

    @staticmethod
    def get_bounds():
        return {'sigma_noise': (1e-7, None)}

    @staticmethod
    def get_initial_values():
        return {'sigma_noise': 1}


class RBF(Kernel):
    def apply(self, X, Y=None, diag=False):
        """
        :param X: matrix of size (batch_x, num_feat)
        :param Y: matrix of size (batch_y, num_feat)
        :return: K - RBF kernel of size (batch_x, batch_y)
        """
        sigma = self.params['sigma']
        length_scale = self.params['length_scale']

        if Y is None:
            Y = X

        check_arrays(X, Y)
        if diag:
            assert X.shape[0] == Y.shape[0], 'If diag is True first dimensions of X and Y must be the same'
            Z = np.sum(X ** 2, 1) + np.sum(Y ** 2, 1) - 2 * np.einsum('ij,ij->i', X, Y)
        else:
            Z = np.sum(X ** 2, 1).reshape(-1, 1) + np.sum(Y ** 2, 1) - 2 * np.dot(X, Y.T)
        Z = -0.5 * (1 / (length_scale ** 2)) * Z
        Z = (sigma ** 2) * np.exp(Z)
        return Z

    @staticmethod
    def get_bounds():
        return {'sigma': (1e-5, None), 'length_scale': (1e-5, None)}

    @staticmethod
    def get_initial_values():
        return {'sigma': 1, 'length_scale': 1}