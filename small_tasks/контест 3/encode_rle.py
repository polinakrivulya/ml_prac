import numpy as np


def encode_rle(x):
    if (np.all(x == x[0])):
        return (np.array([x[0]]), np.array([x.shape[0]]))
    ind1 = np.where((x - np.append(x[1:], x[-1] - 1)) != 0)[0]
    ans1 = x[ind1]
    ind2 = np.copy(ind1[:-1])
    ind2 = np.append(-1, ind2)
    return (ans1, ind1 - ind2)
