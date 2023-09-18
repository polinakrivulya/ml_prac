import numpy as np
from scipy.spatial.distance import cdist


def euclidean_distance(X, Y):
    distance = np.linalg.norm(X, axis=1)[:, None] ** 2 + \
        np.linalg.norm(Y, axis=1)[None, :] ** 2 - \
        (2.0 * X) @ Y.T
    distance[np.where(abs(distance) < 1e-12)] = 0
    return np.sqrt(distance)


def cosine_distance(X, Y):
    # где нулевые элементы, сделаем нули
    with np.errstate(divide='ignore', invalid='ignore'):
        distance = np.divide(X @ Y.T, (
            np.linalg.norm(X, axis=1)[:, None] *
            np.linalg.norm(Y, axis=1)[None, :]
        ))
    distance[np.where(np.isnan(distance))] = 1
    distance[np.where(abs(distance) < 1e-12)] = 0
    return 1 - distance
