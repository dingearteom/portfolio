import numpy as np
from functools import total_ordering


def check_arrays(X, Y=None):
    if Y is None:
        Y = X
    assert isinstance(X, np.ndarray), 'X must be np.ndarray'
    assert isinstance(Y, np.ndarray), 'Y must be np.ndarray'
    assert X.shape[1] == Y.shape[1], 'Second shape of X and Y must be equal.'
    assert len(X.shape) == 2, 'X must have 2 dimensions'
    assert len(Y.shape) == 2, 'Y must have 2 dimensions'


@total_ordering
class Point:
    def __init__(self, x: np.ndarray, y: float):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return tuple(self.x) == tuple(other.x) and self.y == other.y

    def __le__(self, other):
        return self.y < other.y or (self.y == other.y and tuple(self.x) <= tuple(other.x))



