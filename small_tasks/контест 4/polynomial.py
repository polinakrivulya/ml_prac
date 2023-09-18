import numpy as np


class Polynomial:
    def __init__(self, *args):
        self.coef = np.array(args)

    def __call__(self, item):
        if item == 0:
            return self.coef[0]
        if item == 1:
            return sum(self.coef)
        arr1 = np.full_like(self.coef, item, dtype=type(item))
        arr1[0] = 1
        arr1 = np.cumprod(arr1)
        return np.sum(arr1 * self.coef)

    def get_coefs(self):
        return self.coef

    def set_coefs(self, lst):
        if not isinstance(lst, (list, tuple)):
            raise TypeError
        self.coef = np.array(lst)

    coefs = property(get_coefs, set_coefs)

    def __setitem__(self, *args):
        # print("help...")
        # print(args)
        set = args[0]
        item = args[1]
        self.coef[set] = item

    def __getitem__(self, item):
        return self.coef[item]


class IntegerPolynomial(Polynomial):
    def __init__(self, *args):
        Polynomial.__init__(self, *np.round(args).astype(int))

    def get_coefs(self):
        return self.coef

    def set_coefs(self, lst):
        v1 = [round(x) for x in lst]
        Polynomial.set_coefs(self, v1)

    coefs = property(get_coefs, set_coefs)

    def __setitem__(self, *args):
        set = args[0]
        item = np.round(args[1])
        Polynomial.__setitem__(self, set, item)
