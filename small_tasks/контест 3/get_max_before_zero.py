import numpy as np


def get_max_before_zero(x):
    if not any(x[0: -1] == 0):
        return None
    else:
        y = x[np.where(x[0: -1] == 0)[0] + 1]
        return np.max(y)
