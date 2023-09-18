import numpy as np


def get_nonzero_diag_product(X):
    Y = X.diagonal()
    if np.all(Y == 0):
        return None
    return np.prod(Y[np.where(Y != 0)])
