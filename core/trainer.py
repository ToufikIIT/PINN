import numpy as np

class PINNTrainer:
    def __init__(self, model, pde, optimizer,
                 ic_func, bc_func,
                 ic_weight=10.0, bc_weight=10.0):
        self.model = model
        self.pde = pde
        self.opt = optimizer
        self.ic_func = ic_func
        self.bc_func = bc_func
        self.ic_weight = ic_weight
        self.bc_weight = bc_weight

    def loss(self, Xf):
        R = self.pde.residual(self.model, Xf)
        loss_pde = np.mean(R**2)

        x = Xf[0:1]
        t = Xf[1:2]

        u0, _, _, _ = self.model.forward_with_derivatives(
            np.vstack([x, np.zeros_like(x)])
        )
        loss_ic = np.mean((u0 - self.ic_func(x))**2)

        uL, _, _, _ = self.model.forward_with_derivatives(
            np.vstack([np.zeros_like(t), t])
        )
        uR, _, _, _ = self.model.forward_with_derivatives(
            np.vstack([np.ones_like(t), t])
        )
        loss_bc = np.mean(uL**2) + np.mean(uR**2)

        return loss_pde + self.ic_weight*loss_ic + self.bc_weight*loss_bc

    def finite_difference_grads(self, Xf, eps=1e-6):
        grads = []
        base_loss = self.loss(Xf)

        for p in self.model.params():
            g = np.zeros_like(p)
            it = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                p[idx] += eps
                l1 = self.loss(Xf)
                p[idx] -= eps
                g[idx] = (l1 - base_loss) / eps
                it.iternext()
            grads.append(g)

        return grads

    def step(self, Xf):
        grads = self.finite_difference_grads(Xf)
        self.opt.step(grads)
        return self.loss(Xf)
