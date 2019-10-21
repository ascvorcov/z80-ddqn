import numpy as np
import struct

class SumTree:
    write = 0
    size = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total_priority(self):
        return self.tree[0] # root node contains sum of all priorities

    def min_priority(self):
        return np.min(self.tree[-self.capacity:]) # take 'tail' of tree, which contains actual priorities, non-aggregated

    def max_priority(self):
        return np.max(self.tree[-self.capacity:]) # take 'tail' of tree, which contains actual priorities, non-aggregated

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.size < self.capacity:
            self.size += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    def save(self, f, datawriter):
        f.write(struct.pack("III",self.capacity,self.write,self.size))
        np.save(f, self.tree)
        for i in range(self.size):
            datawriter(f, self.data[i])

    def load(self, f, datareader):
        hdr = struct.Struct("III")
        self.capacity,self.write,self.size = hdr.unpack_from(f.read(hdr.size))
        self.tree = np.load(f)
        for i in range(self.size):
            self.data[i] = datareader(f)
