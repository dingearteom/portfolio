import numpy as np
from scipy.linalg import solve_triangular, cholesky
from scipy.optimize import minimize
from typing import Tuple, List

from GP.kernels import RBF, WhiteKernel
from sortedcontainers import SortedList
from .utils import Point


class GaussianProcessRegression:
    def __init__(self, noisy):
        if noisy:
            self.kernel = RBF(sigma=1, length_scale=1) + WhiteKernel(sigma_noise=0.1)
        else:
            self.kernel = RBF(sigma=1, length_scale=1)
        self.X_train = None
        self.y_train = None
        self.sortedData = SortedList()

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert len(X.shape) == 2, 'X must be two-dimensional array'
        assert len(y.shape) == 1, 'y must be one-dimensional array'

        if self.X_train is None:
            self.X_train = X
        else:
            self.X_train = np.concatenate((self.X_train, X), axis=0)
        if self.y_train is None:
            self.y_train = y
        else:
            self.y_train = np.concatenate((self.y_train, y), axis=0)

        for x_point, y_value in zip(X, y):
            self.sortedData.add(Point(x_point, y_value))

        self.K = self.kernel.apply(X) + 1e-7 * np.eye(X.shape[0])
        self.K_cholesky = cholesky(self.K, lower=True)
        self.K_inv = np.linalg.inv(self.K)
        self.optimize()

    def predict(self, X: np.ndarray, return_cov=False):
        X_dim = len(X.shape)
        X = X.copy()
        X = np.atleast_2d(X)

        Kx = self.kernel.apply(self.X_train, X)

        mean = np.dot(np.dot(Kx.T, self.K_inv), self.y_train).flatten()

        if return_cov:
            Kxx = self.kernel.apply(X)
            cov = Kxx - np.dot(np.dot(Kx.T, self.K_inv), Kx)
            if X_dim == 2:
                return mean, cov
            else:
                return mean[0], cov[0]
        else:
            std = self.kernel.apply(X, diag=True) - np.einsum('ij,ji->i', np.dot(Kx.T, self.K_inv), Kx)
            if X_dim == 2:
                return mean, std
            else:
                return mean[0], std[0]

    def log_likelihood(self, theta: dict):
        old_theta = self.kernel.get_params()
        self._set_kernel_params(theta)
        S1 = solve_triangular(self.K_cholesky, self.y_train, lower=True)
        S2 = solve_triangular(self.K_cholesky.T, S1, lower=False)

        res = np.sum(np.log(np.diagonal(self.K_cholesky))) + \
              0.5 * self.y_train.dot(S2) + \
              0.5 * len(self.X_train) * np.log(2 * np.pi)
        self._set_kernel_params(old_theta)
        return res

    def optimize(self, num_run=5):

        def to_dict(x):
            return dict(zip(self.kernel.get_params_names(), x))

        def to_array(x):
            return np.array(list(x.values()))

        bounds_to_sample_from = {}
        initial_values = self.kernel.get_initial_values()
        bounds = self.kernel.get_bounds()
        param_names = self.kernel.get_params_names()
        for key in param_names:
            bounds_to_sample_from[key] = (max(0, bounds[key][0]) if bounds[key][0] is not None else 0,
                                          min(2*initial_values[key], bounds[key][1])
                                          if bounds[key][1] is not None else 2 * initial_values[key])

        best_log_likelihood = None
        best_theta = None
        for i in range(num_run):
            start_point = np.array([np.random.uniform(bounds_to_sample_from[key][0], bounds_to_sample_from[key][1])
                                    for key in param_names])
            res = minimize(lambda x: self.log_likelihood(to_dict(x)),
                           start_point,
                           bounds=to_array(self.kernel.get_bounds()),
                           method='L-BFGS-B')

            if best_log_likelihood is None or self.log_likelihood(to_dict(res.x)) < best_log_likelihood:
                best_log_likelihood = self.log_likelihood(to_dict(res.x))
                best_theta = to_dict(res.x)

        self._set_kernel_params(best_theta)

    def _set_kernel_params(self, theta: dict):
        self.kernel.set_params(theta)
        self.K = self.kernel.apply(self.X_train) + 1e-7 * np.eye(self.X_train.shape[0])
        self.K_cholesky = cholesky(self.K, lower=True)
        self.K_inv = np.linalg.inv(self.K)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X_train, self.y_train

    def get_sorted_data(self):
        return self.sortedData

    def get_best_points(self, num: int) -> List[Point]:
        assert num <= len(self.sortedData), 'There is not enough points to return'
        points = []
        for i in range(num):
            points.append(self.sortedData.__getitem__(i))
        return points

    @property
    def x_opt(self) -> np.ndarray:
        return self.sortedData.__getitem__(0).x

    @property
    def f_opt(self) -> float:
        return self.sortedData.__getitem__(0).y

    @property
    def size(self) -> int:
        return self.X_train.shape[0]