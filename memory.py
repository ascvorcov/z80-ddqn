from sumtree import SumTree
import random
import struct
import math

class Memory:

    def __init__(self, capacity, e = 0.01, a = 0.6, b = 0.4):
        self.tree = SumTree(capacity)
        self.e = e
        self.a = a
        self.b = b

    def _getPriority(self, error):
        return min(error + self.e, 1.0) ** self.a

    def size(self):
        return self.tree.size

    def append(self, sample):
        p = self.tree.max_priority()
        self.tree.add(1.0 if p == 0 else p, sample) 

    def get_data(self,n):
        return self.tree.data[n]

    def sample(self, n):
        batch = []
        segment = self.tree.total_priority() / n
        self.b = min(1., self.b + 0.001)
        mp = self.tree.min_priority()
        mp = self.e if mp == 0 else mp

        p_min = mp / self.tree.total_priority()
        max_weight = (n * p_min) ** -self.b
        for i in range(n):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            sampling_probabilities = p / self.tree.total_priority()
            w = ((n * sampling_probabilities) ** -self.b) / max_weight
            batch.append( (idx, data, w) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        assert p > 0
        self.tree.update(idx, p)

    def save(self, f, datawriter):
        f.write(struct.pack("ddd",self.e,self.a,self.b))
        self.tree.save(f, datawriter)

    def load(self, f, datareader):
        hdr = struct.Struct("ddd")
        self.e,self.a,self.b = hdr.unpack_from(f.read(hdr.size))
        self.tree.load(f, datareader)
