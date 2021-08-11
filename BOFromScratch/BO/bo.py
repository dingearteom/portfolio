import numpy as np
from GP.gpr import GaussianProcessRegression
from BO.acquisitions import AcquisitionEI


class BO:
    def __init__(self, f, bounds, noisy=False, num_init=5):
        self.f = f
        self.bounds = bounds
        self.gr = GaussianProcessRegression(noisy=noisy)
        self.acquisition = AcquisitionEI(self.gr)
        self.initialize(num_init)

    def suggest_next_location(self):
        return self.acquisition.optimize(self.bounds)

    def run_optimization(self, num_run):
        for i in range(num_run):
            x = self.suggest_next_location()
            self.gr.fit(np.atleast_2d(x), np.atleast_1d(self.f(x)))

    def initialize(self, num_init):
        X = []
        for (low, high) in self.bounds:
            X.append(np.random.uniform(low, high, size=num_init)[:, np.newaxis])
        X = np.hstack(X)
        y = np.apply_along_axis(self.f, 1, X).squeeze()
        self.gr.fit(X, y)

