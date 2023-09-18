import numpy as np


def replace_nan_to_means(X):
    Y = np.copy(X)
    nulls = np.isnan(Y)
    m = np.nanmean(Y, axis=0)
    index = np.where(nulls)
    Y[index] = np.take(m, index[1])
    Y[np.where(np.isnan(Y))] = 0
    return Y
