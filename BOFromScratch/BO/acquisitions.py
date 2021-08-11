from abc import ABCMeta
from abc import abstractmethod
from typing import final
from GP.gpr import GaussianProcessRegression
from scipy.special import erfc
import numpy as np
from scipy.optimize import minimize


class AcquisitionBase(metaclass=ABCMeta):
    @final
    def __init__(self, model: GaussianProcessRegression):
        self.model = model

    @abstractmethod
    def acquisition_function(self, x):
        pass

    def optimize(self, bounds, num_run=5, num_min_init=3) -> np.ndarray:
        assert num_run >= num_min_init

        best_x = None
        best_y = None

        num_min_init = min(num_min_init, self.model.size)
        d = self.model.get_best_points(num_min_init)

        for i in range(num_run):
            if i < num_min_init:
                start_point = d[i].x
            else:
                start_point = []
                for (low, high) in bounds:
                    start_point.append(np.random.uniform(low, high))
                start_point = np.array(start_point)

            res = minimize(lambda x: -self.acquisition_function(x), start_point, bounds=bounds, method='L-BFGS-B')
            y = self.acquisition_function(res.x)
            if best_y is None or y > best_y:
                best_y = y
                best_x = res.x

        return best_x


class AcquisitionEI(AcquisitionBase):

    def acquisition_function(self, x: np.ndarray):
        m, s = self.model.predict(x)
        if s < 1e-10:
            s = 1e-10

        delta = self.model.f_opt - m
        u = delta / s

        phi = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
        Phi = 0.5 * erfc(-u / np.sqrt(2))
        return s * (u * Phi + phi)
