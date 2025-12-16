import numpy as np

class InitialCondition:
    def __init__(self, func):
        self.func = func

    def loss(self, model, x):
        t = np.zeros_like(x)
        X = np.vstack([x, t])
        u, _ = model.forward(X)
        return np.mean((u - self.func(x))**2)
