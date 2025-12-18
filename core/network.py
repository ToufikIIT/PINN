import numpy as np
from core.activations import tanh, d_tanh, dd_tanh

class NeuralField:
    def __init__(self, layers):
        self.layers = layers
        self.W = []
        self.b = []

        for i in range(len(layers)-1):
            self.W.append(np.random.randn(layers[i+1], layers[i]) * np.sqrt(2/layers[i]))
            self.b.append(np.zeros((layers[i+1], 1)))

    def params(self):
        return self.W + self.b

    def forward_with_derivatives(self, X):
        A = X
        dA_dx = np.array([[1.0], [0.0]])
        dA_dt = np.array([[0.0], [1.0]])
        d2A_dxx = np.zeros_like(dA_dx)

        for i in range(len(self.W)-1):
            Z = self.W[i] @ A + self.b[i]
            
            sp = d_tanh(Z)
            spp = dd_tanh(Z)

            A_new = tanh(Z)

            dZ_dx = self.W[i] @ dA_dx
            dZ_dt = self.W[i] @ dA_dt
            d2Z_dxx = self.W[i] @ d2A_dxx

            dA_dx = sp * dZ_dx
            dA_dt = sp * dZ_dt
            d2A_dxx = spp * (dZ_dx**2) + sp * d2Z_dxx

            A = A_new

        u = self.W[-1] @ A + self.b[-1]
        u_x = self.W[-1] @ dA_dx
        u_t = self.W[-1] @ dA_dt
        u_xx = self.W[-1] @ d2A_dxx

        return u, u_x, u_t, u_xx

def forward_ode(x, params):
    w, b, v = params
    z = w @ x + b  
    
    h = tanh(z)
    sp = d_tanh(z)
    y = v @ h  
    y_x = v @ (sp * w)  
    
    return y, y_x, z, h, sp