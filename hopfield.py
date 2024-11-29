import numpy as np

class HopNet:
    def __init__(self, size):
        self.size = size
        self.wt = np.zeros((size, size))

    def train(self, pats):
        for pat in pats:
            self.wt += np.outer(pat, pat)
        np.fill_diagonal(self.wt, 0)

    def retrieve(self, pat, maxIter=100):
        state = pat.copy()
        for _ in range(maxIter):
            newState = np.sign(self.wt @ state)
            newState[newState == 0] = 1
            if np.array_equal(state, newState):
                break
            state = newState
        return state

def genPat(numPats, size):
    return [np.random.choice([-1, 1], size=size) for _ in range(numPats)]

if __name__ == "__main__":
    size = 100
    numPats = 15
    pats = genPat(numPats, size)

    net = HopNet(size)
    net.train(pats)

    success = 0
    for pat in pats:
        noisyPat = pat.copy()
        noisyPat[:5] *= -1
        retrieved = net.retrieve(noisyPat)
        if np.array_equal(retrieved, pat):
            success += 1

    print(f"Patterns retrieved successfully: {success}/{numPats}")
