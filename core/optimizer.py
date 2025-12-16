import numpy as np

class SGD_Momentum:
    def __init__(self, params, lr=1e-3, gamma=0.9):
        self.params = params
        self.lr = lr
        self.gamma = gamma
        self.vel = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for i in range(len(self.params)):
            self.vel[i] = self.gamma * self.vel[i] + grads[i]
            self.params[i] -= self.lr * self.vel[i]
