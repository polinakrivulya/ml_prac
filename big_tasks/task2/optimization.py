import numpy as np
import oracles
import time
from scipy.special import expit


def euclidean_distance_sqr(X, Y):
    distance = np.linalg.norm(X, axis=1)[:, None] ** 2 + \
        np.linalg.norm(Y, axis=1)[None, :] ** 2 - \
        (2.0 * X) @ Y.T
    distance[np.where(abs(distance) < 1e-12)] = 0
    return distance


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
            self, loss_function, step_alpha=1, step_beta=0,
            tolerance=1e-5, max_iter=1000, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций
        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_function = loss_function
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.kwargs = kwargs
        self.__w = 0
        self.__accuracy = []

    def fit(self, X, y, w_0=None, trace=False, accuracy=False):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        trace - переменная типа bool
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        w_now = w_0
        n = self.step_alpha
        my_oracle = oracles.BinaryLogistic(self.kwargs['l2_coef'])
        history = dict.fromkeys(['func', 'time'])
        history['func'] = []
        history['time'] = []
        weights = []

        for i in range(1, self.max_iter + 1):
            start_time = time.time()
            f_grad = my_oracle.grad(X, y, w_now)
            f = my_oracle.func(X, y, w_now)
            history['func'].append(f)

            if len(history['func']) > 1:
                if abs(history['func'][-1] - history['func'][-2]) < self.tolerance:
                    history['time'].append(time.time() - start_time)
                    self.__w = w_now
                    if accuracy:
                        return self.__accuracy, history
                    elif trace:
                        return history

            w_now = np.ravel(w_now - n * f_grad)
            weights.append(w_now)
            self.__w = w_now
            ans = self.predict(X)
            ans = np.sum(ans == y) / len(y)
            self.__accuracy.append(ans)
            n = self.step_alpha / (i ** self.step_beta)
            history['time'].append(time.time() - start_time)

        f = my_oracle.func(X, y, w_now)
        history['func'].append(f)
        if accuracy:
            return self.__accuracy, history
        elif trace:
            return history

    def predict(self, X):
        """
        Получение меток ответов на выборке X
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: одномерный numpy array с предсказаниями
        """
        p = self.predict_proba(X)
        pred = np.full(X.shape[0], 1)
        pred[np.where(p[:, 1] < 0.5)] = -1

        return pred

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        ans = np.zeros((X.shape[0], 2))
        if not isinstance(X, type(ans)):
            m = X.dot(self.__w)
        else:
            m = np.dot(self.__w, X.T)
        p = expit(m)
        ans[:, 1] = p
        ans[:, 0] = 1 - p
        return ans

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: float
        """
        my_oracle = oracles.BinaryLogistic(self.kwargs['l2_coef'])
        return my_oracle.func(X, y, self.__w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: numpy array, размерность зависит от задачи
        """
        my_oracle = oracles.BinaryLogistic(self.kwargs['l2_coef'])
        return my_oracle.grad(X, y, self.__w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.__w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
            self, loss_function, batch_size, step_alpha=1, step_beta=0,
            tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        batch_size - размер подвыборки, по которой считается градиент
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций (эпох)
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        **kwargs - аргументы, необходимые для инициализации
        """
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.__w = 0
        self.__accuracy = []
        self = GDClassifier.__init__(self, loss_function, step_alpha, step_beta, tolerance, max_iter, **kwargs)

    def fit(self, X, y, w_0=None, trace=False, log_freq=1, accuracy=False):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        np.random.seed(self.random_seed)
        batch_index = np.random.permutation(X.shape[0])

        n = self.step_alpha
        my_oracle = oracles.BinaryLogistic(self.kwargs['l2_coef'])
        history = dict.fromkeys(['func', 'time', 'weights_diff', 'epoch_num'])
        history['func'] = []
        history['time'] = []
        history['weights_diff'] = []
        history['epoch_num'] = []
        batch_index_first = 0
        num_ep_last = num_ep_now = 0
        objects_num = 0

        start_time = time.time()
        w_now = w_0
        weights_last = w_now
        while num_ep_now < self.max_iter:
            indexs = batch_index[batch_index_first:batch_index_first + self.batch_size]
            if len(indexs) < self.batch_size / 10:
                batch_index = np.random.permutation(X.shape[0])
                batch_index_first = 0
                indexs = batch_index[batch_index_first:batch_index_first + self.batch_size]
            f_grad = my_oracle.grad(X[indexs],
                                    y[indexs],
                                    w_now)
            batch_index_first += self.batch_size

            objects_num += self.batch_size

            w_now = np.ravel(w_now - n * f_grad)

            num_ep_now = objects_num / len(y)
            if num_ep_now - num_ep_last >= log_freq:
                f = my_oracle.func(X, y, w_now)
                history['func'].append(f)
                history['time'].append(time.time() - start_time)
                history['epoch_num'].append(num_ep_now)
                history['weights_diff'].append(np.linalg.norm(w_now - weights_last) ** 2)
                self.__w = w_now
                ans = self.predict(X)
                ans = np.sum(ans == y) / len(y)
                self.__accuracy.append(ans)
                weights_last = w_now
                start_time = time.time()
                num_ep_last = num_ep_now
                batch_index = np.random.permutation(X.shape[0])
                batch_index_first = 0
                n = self.step_alpha / (num_ep_now ** self.step_beta)

            if len(history['func']) > 1:
                if abs(history['func'][-1] - history['func'][-2]) < self.tolerance:
                    break

        f = my_oracle.func(X, y, w_now)
        history['func'].append(f)
        history['time'].append(time.time() - start_time)
        history['epoch_num'].append(num_ep_now)
        history['weights_diff'].append(np.linalg.norm(w_now - weights_last) ** 2)
        self.__w = w_now
        ans = self.predict(X)
        ans = np.sum(ans == y) / len(y)
        self.__accuracy.append(ans)
        if accuracy:
            return self.__accuracy, history
        elif trace:
            return history

    def predict(self, X):
        """
        Получение меток ответов на выборке X
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: одномерный numpy array с предсказаниями
        """

        return GDClassifier.predict(self, X)

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        ans = np.zeros((X.shape[0], 2))
        if not isinstance(X, type(ans)):
            m = X.dot(self.__w)
        else:
            m = np.dot(self.__w, X.T)
        p = expit(m)
        ans[:, 1] = p
        ans[:, 0] = 1 - p
        return ans

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: float
        """
        my_oracle = oracles.BinaryLogistic(self.kwargs['l2_coef'])
        return my_oracle.func(X, y, self.__w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: numpy array, размерность зависит от задачи
        """
        my_oracle = oracles.BinaryLogistic(self.kwargs['l2_coef'])
        return my_oracle.grad(X, y, self.__w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.__w
