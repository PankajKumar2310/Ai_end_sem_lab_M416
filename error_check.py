import numpy as np

class HopNet:
    def __init__(self, size):
        self.size = size
        self.wt = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            self.wt += np.outer(p, p)
        np.fill_diagonal(self.wt, 0)
        self.wt /= len(patterns)

    def recall(self, pat, steps=10):
        res = np.copy(pat)
        for _ in range(steps):
            for i in range(self.size):
                act = np.dot(self.wt[i], res)
                res[i] = 1 if act >= 0 else -1
        return res

pats = [np.array([1, -1, 1, -1, 1]), np.array([-1, 1, -1, 1, -1])]
testPat = np.array([1, -1, -1, -1, 1]) 

net = HopNet(size=5)
net.train(pats)

out = net.recall(testPat)
print("Recalled Pattern:", out)
