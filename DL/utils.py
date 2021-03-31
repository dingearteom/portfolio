import numpy as np


def transfrom_1Darray_to_2D(array):
    if array.ndim == 1:
        return array[np.newaxis]
    else:
        return array.copy()
