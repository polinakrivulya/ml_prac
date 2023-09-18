import numpy as np


def calc_expectations(h, w, X, Q):
    if np.size(X) == 0:
        return X
    Y = np.cumsum(Q, axis=1)
    Y[:, w:] -= Y[:, :-w]
    Y = np.cumsum(Y, axis=0)
    Y[h:, :] -= Y[:-h, :]
    return X*Y
