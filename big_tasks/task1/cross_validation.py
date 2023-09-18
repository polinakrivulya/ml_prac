import numpy as np
import nearest_neighbors
from random import randint


def kfold(n, n_folds):
    '''
    Функция возвращает список длины n_folds,
     каждый элемент списка — кортеж из двух одномерных np.ndarray.
     Первый массив содержит индексы обучающей подвыборки, а второй валидационной.
    '''
    arr = np.arange(n)
    # arr = np.random.choice(np.arange(n), size=n, replace=False)
    long = (n % n_folds) * (n // n_folds + 1)
    arr1 = arr[:long]
    arr2 = arr[long:]
    arr1 = arr1.reshape(-1, n // n_folds + 1)
    arr2 = arr2.reshape(-1, n // n_folds)
    lst = []
    mask = np.ones_like(arr1)
    for i in range(arr1.shape[0]):
        mask[i] = 0
        tup = (np.concatenate((arr1[np.where(mask == 1)], arr2), axis=None), arr1[i])
        mask[i] = 1
        lst.append(tup)
    mask = np.ones_like(arr2)
    for i in range(arr2.shape[0]):
        mask[i] = 0
        tup = (np.concatenate((arr2[np.where(mask == 1)], arr1), axis=None), arr2[i])
        mask[i] = 1
        lst.append(tup)
    return lst


def knn_cross_val_score(X, y, k_list, score, cv=None, **kwargs):
    if cv is not None:
        n_folds = len(cv)
    else:
        n_folds = 3
        cv = kfold(len(X), n_folds)
    first, second = zip(*cv)
    KNN = nearest_neighbors.KNNClassifier(k_list[-1], **kwargs)
    d = dict.fromkeys(k_list)
    for i in k_list:
        d[i] = np.empty(n_folds)
    for i in range(n_folds):
        KNN.fit(X[first[i]], y[first[i]])
        cls = y[first[i]]
        if not KNN.weights:
            distance_kneighbors = np.ones((len(y[second[i]]), k_list[-1]))
            index_kneighbors = KNN.find_kneighbors(X[second[i]], False)
        else:
            distance_kneighbors, index_kneighbors = KNN.find_kneighbors(X[second[i]], True)
            eps = 1e-5
            distance_kneighbors = (distance_kneighbors + eps) ** (-1)
        predict = np.zeros((second[i].shape[0], len(k_list)))
        for s in range(second[i].shape[0]):
            lst = dict()
            jj = -1
            for j in range(k_list[-1]):
                if cls[index_kneighbors[s][j]] in lst:
                    lst[cls[index_kneighbors[s][j]]] += distance_kneighbors[s][j]
                else:
                    lst[cls[index_kneighbors[s][j]]] = distance_kneighbors[s][j]
                if j + 1 in k_list:
                    jj += 1
                    predict[s][jj] = max(lst, key=lst.get)
        if score == 'accuracy':
            num_true_predict = np.sum(predict == y[second[i]][:, None], axis=0)
            num_all_predict = len(predict)
            num_true_predict = num_true_predict / num_all_predict
            for j, k in enumerate(k_list):
                d[k][i] = num_true_predict[j]
    return d
