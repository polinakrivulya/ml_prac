import numpy as np


def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    grad = []
    e_i = np.eye(len(w[2])) * eps
    for i in range(0, len(w[2])):
        grad.append((function(w[0], w[1], w[2] + e_i[i]) - function(w[0], w[1], w[2])) / eps)
    return np.array(grad)
