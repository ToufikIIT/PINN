class Burgers1D:
    def __init__(self, nu):
        self.nu = nu

    def residual(self, model, X):
        u, ux, ut, uxx = model.forward_with_derivatives(X)
        return ut + u * ux - self.nu * uxx
