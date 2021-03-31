import numpy as np


def MAPE(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    return np.sum(np.abs((y_true - y_pred) / y_pred)) / n