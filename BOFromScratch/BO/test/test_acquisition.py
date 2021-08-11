import unittest
import numpy as np
from GP.gpr import GaussianProcessRegression
from BO.acquisitions import AcquisitionEI
from scipy.integrate import quad
from scipy.stats import norm


class TestAcquisitionEI(unittest.TestCase):
    def test_acquisition_function_one_dimensional(self):
        sigma_noise = 0.2

        X = np.random.uniform(0, 10, size=3)
        y = np.sin(X) + np.random.normal(0, sigma_noise ** 2, X.shape[0])
        X = X[:, np.newaxis]

        gr = GaussianProcessRegression(noisy=False)
        gr.fit(X, y)

        num_run = 10
        samples = np.random.uniform(0, 10, size=num_run)
        acquisition = AcquisitionEI(gr)
        for i in range(num_run):
            x = np.array(samples[i])
            improvement = acquisition.acquisition_function(x)

            m, v = gr.predict(x)
            f_opt = gr.f_opt

            def f(x):
                return max(f_opt - x, 0) * norm.pdf(x, loc=m, scale=v)

            self.assertAlmostEqual(quad(f, m - 10 * v, f_opt)[0], improvement)


if __name__ == "__main__":
    unittest.main()