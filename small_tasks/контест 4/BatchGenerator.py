import numpy as np


class BatchGenerator:
    def __init__(self, list_of_sequences, batch_size, shuffle=False):
        self.arr = np.array(list_of_sequences)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        n_samples = len(self.arr[0])
        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples, self.batch_size):
            A = []
            end = min(start + self.batch_size, n_samples)
            batch_idx = indices[start:end]
            for i in range(0, self.arr.shape[0]):
                A.append(list(np.take_along_axis(self.arr[i], batch_idx, axis=0)))
            yield A
