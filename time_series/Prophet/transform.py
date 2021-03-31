import numpy as np
from scipy.stats import boxcox


class Transform:
    @staticmethod
    def transform_id(y):
        return y.copy()
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
    def inv_transform_id(y):
        return y.copy()
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

    @staticmethod
    def params_transform(name, df, fold, TEST_SIZE=70):
        assert name in ['id', 'log', 'boxcox', 'diff', 'boxcox_diff', 'boxcox_diff2', 'log_diff']

        TRAIN_SIZE = df.shape[0] - TEST_SIZE * fold
        df_train = df[:TRAIN_SIZE]
        params_transform = dict()

        if name == 'id':
            pass
        elif name == 'log':
            pass
        elif name == 'boxcox':
            _, params_transform['lmbda'] = Transform.transform_boxcox(df_train.y)
        elif name == 'diff':
            params_transform['y0'] = df_train.y[0]
        elif name == 'log_diff':
            params_transform['y0'] = Transform.transform_log(df_train.y)[0]
        elif name == 'boxcox_diff':
            y, params_transform['lmbda'] = Transform.transform_boxcox(df_train.y)
            params_transform['y0'] = y[0]
        elif name == 'boxcox_diff2':
            y, params_transform['lmbda'] = Transform.transform_boxcox(df_train.y)
            params_transform['y1'] = y[0]
            y = Transform.transform_diff(y)
            params_transform['y0'] = y[0]
        return params_transform

    @staticmethod
    def transform_by_name(name):
        assert name in ['id', 'log', 'boxcox', 'diff', 'boxcox_diff', 'boxcox_diff2', 'log_diff']

        if name == 'id':
            return Transform.transform_id, Transform.inv_transform_id
        elif name == 'log':
            return Transform.transform_log, Transform.inv_transform_log
        elif name == 'boxcox':
            return Transform.transform_boxcox_ignore, Transform.inv_transform_boxcox
        elif name == 'diff':
            return Transform.transform_diff, Transform.inv_transform_diff
        elif name == 'boxcox_diff':
            return Transform.transform_boxcox_diff_ignore, Transform.inv_transform_boxcox_diff
        elif name == 'boxcox_diff2':
            return Transform.transform_boxcox_diff2_ignore, Transform.inv_transform_boxcox_diff2
        elif name == 'log_diff':
            return Transform.transform_log_diff, Transform.inv_transform_log_diff


