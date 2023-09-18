

class CooSparseMatrix:
    def __init__(self, ijx_list, shape):
        if (len(shape) != 2):
            raise TypeError
        self.size = shape
        self.elem = dict()
        for k in ijx_list:
            if (type(k[0]) != int) | (type(k[1]) != int) \
                  | (k[0] > (shape[0] - 1)) | (k[1] > (shape[1] - 1)):
                raise TypeError
            if k[0] in self.elem:
                if k[1] in self.elem[k[0]]:
                    raise TypeError
            if k[2]:
                if k[0] in self.elem:
                    self.elem[k[0]][k[1]] = k[2]
                else:
                    self.elem[k[0]] = dict()
                    self.elem[k[0]][k[1]] = k[2]

    def __getitem__(self, item):
        if (type(item) != int):
            x = item[0]
            y = item[1]
            if (x > self.size[0] - 1) | (y > self.size[1] - 1) \
                    | (type(x) != int) | (type(y) != int):
                raise TypeError
            if x in self.elem:
                if y in self.elem[x]:
                    return self.elem[x][y]
            return 0
        else:
            x = item
            if (x > self.size[0] - 1) | (type(x) != int):
                raise TypeError
            ans = CooSparseMatrix([], (1, self.size[1]))
            if x in self.elem:
                ans.elem = dict()
                ans.elem[0] = self.elem[x]
            return ans

    def __setitem__(self, key, value):
        x = key[0]
        y = key[1]
        if (x > self.size[0] - 1) | (y > self.size[1] - 1) \
                | (type(x) != int) | (type(y) != int):
            raise TypeError
        if x in self.elem:
            if y in self.elem[x]:
                if (value != 0):
                    self.elem[x][y] = value
                else:
                    del self.elem[x][y]
            else:
                if (value != 0):
                    self.elem[x][y] = value
        else:
            if (value != 0):
                self.elem[x] = dict()
                self.elem[x][y] = value

    def __mul__(self, other):
        ans = CooSparseMatrix([], self.shape)
        for x in self.elem:
            for y in self.elem[x]:
                ans[x, y] = self.elem[x][y] * other
        return ans

    __rmul__ = __mul__

    def __add__(self, other):
        if self.size != other.size:
            raise TypeError
        ans = CooSparseMatrix([], self.size)
        for x in self.elem:
            for y in self.elem[x]:
                ans[x, y] = self.elem[x][y]
        for x in other.elem:
            for y in other.elem[x]:
                ans[x, y] = ans[x, y] + other.elem[x][y]
        return ans

    def __sub__(self, other):
        if self.size != other.size:
            raise TypeError
        ans = CooSparseMatrix([], self.shape)
        if (self.elem == other.elem):
            return ans
        for x in self.elem:
            for y in self.elem[x]:
                ans[x, y] = self.elem[x][y]
        for x in other.elem:
            for y in other.elem[x]:
                ans[x, y] = ans[x, y] - other.elem[x][y]
        return ans

    def get_shape(self):
        return self.size

    def setshape(self, tup):
        if (type(tup) != tuple):
            raise TypeError
        x1 = tup[0]
        y1 = tup[1]
        x0 = self.size[0]
        y0 = self.size[1]
        if (x1 * y1) != (x0 * y0):
            raise TypeError
        if (x1 <= 0) | (y1 <= 0):
            raise TypeError
        if (type(x1) != int) | (type(y1) != int):
            raise TypeError
        ans = CooSparseMatrix([], (x1, y1))
        for x in self.elem:
            for y in self.elem[x]:
                ans[(x * y0 + y) // y1, (x * y0 + y) % y1] = self.elem[x][y]
        self.size = ans.size
        self.elem = ans.elem

    shape = property(get_shape, setshape)

    def getT(self):
        ans = CooSparseMatrix([], (self.size[1], self.size[0]))
        for x in self.elem:
            for y in self.elem[x]:
                ans[y, x] = self.elem[x][y]
        return ans

    T = property(getT)
