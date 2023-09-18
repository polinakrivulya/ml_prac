import numpy as np
import distances as my_own
from sklearn.neighbors import NearestNeighbors


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.strategy = strategy
        self.k = k
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        if strategy == 'my_own':
            self.X_train = None
            self.y_train = None
        else:
            if (strategy == 'kd_tree') | (strategy == 'ball_tree'):
                self.metric = 'euclidean'
            self.Neigh = NearestNeighbors(n_neighbors=k, metric=self.metric, algorithm=self.strategy)

    def fit(self, X, y):  # обучение strategy
        if self.strategy != "my_own":
            self.X_train = self.Neigh.fit(X, y)
            self.y_train = y
        else:
            self.X_train = X
            self.y_train = y

    def find_kneighbors(self, X, return_distance):
        if self.strategy != "my_own":
            return self.Neigh.kneighbors(X, n_neighbors=self.k, return_distance=return_distance)
        if self.metric == 'euclidean':
            matr = my_own.euclidean_distance(X, self.X_train)
        else:
            matr = my_own.cosine_distance(X, self.X_train)
        index_nearest = np.argpartition(matr, kth=self.k, axis=1)[:, :self.k]
        matr = np.take_along_axis(matr, index_nearest, axis=1)
        sorted_index = np.argsort(matr, axis=1)
        index_nearest = np.take_along_axis(index_nearest, sorted_index, axis=1)
        if not return_distance:
            return index_nearest
        matr = np.take_along_axis(matr, sorted_index, axis=1)
        return matr, index_nearest

    def predict(self, X):
        def answer_func(ouri, ourd):
            maximum = -1
            for i in set(ouri):
                t = np.sum(ourd[np.where(ouri == i)])
                if t > maximum:
                    maximum = t
                    ans = i
            return ans

        if not self.weights:
            prediction_all = np.empty(X.shape[0], dtype=int)
            distance_kneighbors = np.ones(X.shape[1])
            for j in range(
                    self.test_block_size, X.shape[0] + self.test_block_size, self.test_block_size
            ):
                index_kneighbors = self.find_kneighbors(X[j - self.test_block_size:j], False)
                prediction = np.empty(index_kneighbors.shape[0], dtype=int)
                for i in range(index_kneighbors.shape[0]):
                    prediction[i] = answer_func(self.y_train[index_kneighbors][i], distance_kneighbors)
                prediction_all[j - self.test_block_size:j] = prediction
        else:
            prediction_all = np.empty(X.shape[0], dtype=int)
            eps = 1e-5
            for j in range(
                    self.test_block_size, X.shape[0] + self.test_block_size, self.test_block_size
            ):
                distance_kneighbors, index_kneighbors = self.find_kneighbors(
                    X[j - self.test_block_size:j], True
                )
                distance_kneighbors = (distance_kneighbors + eps) ** (-1)
                prediction = np.empty(index_kneighbors.shape[0], dtype=int)
                for i in range(index_kneighbors.shape[0]):
                    prediction[i] = answer_func(self.y_train[index_kneighbors][i], distance_kneighbors[i])
                prediction_all[j - self.test_block_size:j] = prediction
        return prediction_all
