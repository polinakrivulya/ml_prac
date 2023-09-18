

class WordContextGenerator:
    def __init__(self, words, window_size):
        self.lst = []
        self.pos = -1
        for i in range(0, len(words)):
            for j in range(0 - window_size, window_size + 1):
                if (i + j >= 0) & (i + j < len(words)) & (j != 0):
                    self.lst.append([words[i], words[i + j]])

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos < len(self.lst) - 1:
            self.pos += 1
            return self.lst[self.pos]
        else:
            raise StopIteration
