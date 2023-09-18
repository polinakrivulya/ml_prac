import numpy as np


class RleSequence:
    def __init__(self, input_sequence):
        if (np.all(input_sequence == input_sequence[0])):
            self.times = np.array([input_sequence.shape[0]])
            self.numbers = np.array([input_sequence[0]])
        else:
            ind1 = np.where((input_sequence - np.append(input_sequence[1:], input_sequence[-1] - 1)) != 0)[0]
            self.numbers = input_sequence[ind1]
            ind2 = np.copy(ind1[:-1])
            ind2 = np.append(-1, ind2)
            self.times = ind1 - ind2
            self.len = input_sequence.shape[0]

    def __getitem__(self, item):
        if (type(item) == int):
            if (item < 0):
                item = self.len + item
            sum = 0
            i = -1
            while (sum <= item):
                i += 1
                sum += self.times[i]
            return self.numbers[i]
        # START:STOP:STEP
        start = item.start
        stop = item.stop
        step = item.step
        if (step is None):
            step = 1
        if (stop is None):
            stop = self.len
        if (start is None):
            start = 0
        if (stop < 0):
            stop = self.len + stop
        if (start < 0):
            start = self.len + start
        if (stop > self.len):
            stop = self.len
        lst = []
        s = 0
        k = -1
        for i in range(start, stop, step):
            while (s <= i):
                k += 1
                s += self.times[k]
            lst.append(self.numbers[k])
        return np.array(lst)

    def __iter__(self):
        for i in range(0, self.numbers.shape[0]):
            for j in range(0, self.times[i]):
                yield self.numbers[i]

    def __contains__(self, item):
        return item in self.numbers
