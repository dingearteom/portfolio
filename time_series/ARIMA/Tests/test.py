import unittest
import pandas as pd
import numpy as np
from ARIMA.transform import Transform


class TransformTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv('../../data/train.csv')
        self.df['ds'] = pd.to_datetime(self.df['ds'], infer_datetime_format=True)
        self.df = self.df.set_index('ds')

        self.assertAlmostEqualVectors = np.vectorize(self.assertAlmostEqual)

    def test_log(self):
        y_initial = self.df.y.copy()
        y_log = Transform.transform_log(self.df.y)

        self.assertAlmostEqualVectors(y_initial, self.df.y, delta=6)

        y_log_initial = y_log.copy()
        y = Transform.inv_transform_log(y_log)

        self.assertAlmostEqualVectors(y_log_initial, y_log, delta=6)

        self.assertAlmostEqualVectors(y, self.df.y, delta=6)

    def test_diff(self):
        y0 = self.df.y[0]

        y_initial = self.df.y.copy()
        y_diff = Transform.transform_diff(self.df.y)

        self.assertAlmostEqualVectors(y_initial, self.df.y, delta=6)

        y_diff_initial = y_diff.copy()
        y = Transform.inv_transform_diff(y_diff, y0)

        self.assertAlmostEqualVectors(y_diff_initial, y_diff, delta=6)

        self.assertAlmostEqualVectors(y, self.df.y, delta=6)

    def test_boxcox(self):
        y_initial = self.df.y.copy()
        y_boxcox, lmbda = Transform.transform_boxcox(self.df.y)

        self.assertAlmostEqualVectors(y_initial, self.df.y, delta=6)

        y_boxcox_initial = y_boxcox.copy()
        y = Transform.inv_transform_boxcox(y_boxcox, lmbda)

        self.assertAlmostEqualVectors(y_boxcox_initial, y_boxcox, delta=6)

        self.assertAlmostEqualVectors(y, self.df.y, delta=6)

    def test_boxcox_diff(self):
        y0 = Transform.transform_boxcox_ignore(self.df.y)[0]

        y_initial = self.df.y.copy()
        y_boxcox_diff, lmbda = Transform.transform_boxcox_diff(self.df.y)

        self.assertAlmostEqualVectors(y_initial, self.df.y, delta=6)

        y_boxcox_diff_initial = y_boxcox_diff.copy()
        y = Transform.inv_transform_boxcox_diff(y_boxcox_diff, lmbda, y0)

        self.assertAlmostEqualVectors(y_boxcox_diff_initial, y_boxcox_diff, delta=6)

        self.assertAlmostEqualVectors(y, self.df.y, delta=6)

    def test_boxcox_diff2(self):
        y0 = Transform.transform_boxcox_ignore(self.df.y)[0]
        y1 = Transform.transform_boxcox_diff_ignore(self.df.y)[0]

        y_initial = self.df.y.copy()
        y_boxcox_diff2, lmbda = Transform.transform_boxcox_diff2(self.df.y)

        self.assertAlmostEqualVectors(y_initial, self.df.y, delta=6)

        y_boxcox_diff2_initial = y_boxcox_diff2.copy()
        y = Transform.inv_transform_boxcox_diff2(y_boxcox_diff2, lmbda, y0, y1)

        self.assertAlmostEqualVectors(y_boxcox_diff2_initial, y_boxcox_diff2, delta=6)

        self.assertAlmostEqualVectors(y, self.df.y, delta=6)

    def test_log_diff(self):
        y0 = Transform.transform_log(self.df.y)[0]

        y_initial = self.df.y.copy()
        y_log_diff = Transform.transform_log_diff(self.df.y)

        self.assertAlmostEqualVectors(y_initial, self.df.y, delta=6)

        y_log_diff_initial = y_log_diff.copy()
        y = Transform.inv_transform_log_diff(y_log_diff, y0)

        self.assertAlmostEqualVectors(y_log_diff_initial, y_log_diff, delta=6)

        self.assertAlmostEqualVectors(y, self.df.y, delta=6)




if __name__ == '__main__':
    unittest.main()

