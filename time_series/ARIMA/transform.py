import numpy as np
from scipy.stats import boxcox


class Transform:
    @staticmethod
    def transform_log(y):
        return np.log(y)

    @staticmethod
    def transform_boxcox(y):
        y = y.copy()
        y, lmbda = boxcox(y)
        return y, lmbda

    @staticmethod
    def transform_boxcox_ignore(y):
        y = y.copy()
        y, lmbda = Transform.transform_boxcox(y)
        return y

    @staticmethod
    def transform_diff(y):
        return np.append([0], np.diff(y))

    @staticmethod
    def transform_boxcox_diff(y):
        y = y.copy()
        y, lmbda = Transform.transform_boxcox(y)
        return Transform.transform_diff(y), lmbda

    @staticmethod
    def transform_boxcox_diff_ignore(y):
        y = y.copy()
        y, lmbda = Transform.transform_boxcox_diff(y)
        return y

    @staticmethod
    def transform_boxcox_diff2(y):
        y = y.copy()
        y, lmbda = Transform.transform_boxcox_diff(y)
        return Transform.transform_diff(y), lmbda

    @staticmethod
    def transform_boxcox_diff2_ignore(y):
        y = y.copy()
        y, lmbda = Transform.transform_boxcox_diff2(y)
        return y

    @staticmethod
    def transform_log_diff(y):
        return Transform.transform_diff(Transform.transform_log(y))

    # inverse transform

    @staticmethod
    def inv_transform_log(y):
        return np.exp(y)

    @staticmethod
    def inv_transform_boxcox(y, lmbda):
        if lmbda == 0:
            return (np.exp(y))
        else:
            return (np.exp(np.log(lmbda * y + 1) / lmbda))

    @staticmethod
    def inv_transform_diff(y, y0):
        return np.cumsum(y) + y0

    @staticmethod
    def inv_transform_boxcox_diff(y, lmbda, y0):
        return Transform.inv_transform_boxcox(Transform.inv_transform_diff(y, y0), lmbda)

    @staticmethod
    def inv_transform_boxcox_diff2(y, lmbda, y0, y1):
        return Transform.inv_transform_boxcox_diff(Transform.inv_transform_diff(y, y1), lmbda, y0)

    @staticmethod
    def inv_transform_log_diff(y, y0):
        return Transform.inv_transform_log(Transform.inv_transform_diff(y, y0))
