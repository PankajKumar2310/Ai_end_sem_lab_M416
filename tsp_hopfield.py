import numpy as np

class TspHop:
    def __init__(self, nCities):
        self.nCities = nCities
        self.nNeurons = nCities * nCities
        self.wt = np.zeros((self.nNeurons, self.nNeurons))
        self.distMat = None

    def initWt(self, dists, A=500, B=500, C=200, D=200):
        self.distMat = dists
        N = self.nCities

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        idx1 = i * N + j
                        idx2 = k * N + l
                        if i == k and j != l:
                            self.wt[idx1][idx2] -= A
                        if i != k and j == l:
                            self.wt[idx1][idx2] -= B
                        if i == k and j == l:
                            self.wt[idx1][idx2] -= C
                        if (i + 1) % N == k:
                            self.wt[idx1][idx2] -= D * dists[j][l]

    def opt(self, initState, steps=100):
        state = np.copy(initState)
        for _ in range(steps):
            for i in range(self.nNeurons):
                act = np.dot(self.wt[i], state)
                state[i] = 1 if act > 0 else 0
        return state

nCities = 10
dists = np.random.randint(1, 100, size=(nCities, nCities))
tsp = TspHop(nCities)
tsp.initWt(dists)

initState = np.random.choice([0, 1], size=(nCities * nCities))
finalState = tsp.opt(initState)
print("Optimized State:", finalState)
