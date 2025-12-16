import numpy as np

class BoundaryCondition:
    def __init__(self, x_val, func):
        self.x_val = x_val
        self.func = func

    def loss(self, model, t):
        x = self.x_val * np.ones_like(t)
        X = np.vstack([x, t])
        u, _ = model.forward(X)
        return np.mean((u - self.func(t))**2)
