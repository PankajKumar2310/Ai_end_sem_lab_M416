import numpy as np

class RookHop:
    def __init__(self, size=8):
        self.size = size
        self.wt = np.zeros((size, size))

    def initWt(self):
        for i in range(self.size):
            for j in range(self.size):
                if i != j:
                    self.wt[i][j] = -1

    def opt(self, initState, steps=10):
        state = np.copy(initState)
        for _ in range(steps):
            for i in range(self.size):
                act = np.dot(self.wt[i], state)
                state[i] = 1 if act > 0 else -1
        return state

initState = np.random.choice([-1, 1], size=(8,))
rook = RookHop(size=8)
rook.initWt()
finalState = rook.opt(initState)
print("Final State:", finalState)
