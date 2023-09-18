import numpy as np
from scipy.special import expit, logit


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.
    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef):
        """
        Задание параметров оракула.
        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w - одномерный numpy array
        """
        if not isinstance(X, type(y)):
            m = X.dot(w) * y
        else:
            m = np.dot(w, X.T) * y
        lf = expit(m)
        return 0.5 * self.l2_coef * np.linalg.norm(w) ** 2 - np.log(lf + 0.000000001).mean(axis=0)

        return super().func(w)

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w - одномерный numpy array
        """
        if not isinstance(X, type(y)):
            m = X.dot(w) * y
            lf = - X.multiply((expit(-m) * y)[:, None])
        else:
            m = np.dot(w, X.T) * y
            lf = - (expit(-m) * y)[:, None] * X
        return lf.mean(axis=0) + self.l2_coef * w

        return super().grad(w)
