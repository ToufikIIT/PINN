import numpy as np
import matplotlib.pyplot as plt

from core.network import forward_ode
from core.optimizer import SGD_Momentum
from physics.linear_differential import ode_residual
from utils.sampling import sample_ode

np.random.seed(0)

n_hidden = 30
epochs = 30000
n_collocation = 100
lr = 1e-3
gamma = 0.9
bc_weight = 30.0

w = np.random.randn(n_hidden, 1)
b = np.zeros((n_hidden, 1))
v = np.random.randn(1, n_hidden)

optimizer = SGD_Momentum([w, b, v], lr, gamma)

for epoch in range(epochs):
    w, b, v = optimizer.params
    x_bc = np.zeros((1, 1))
    y_bc, _, z_bc, h_bc, sp_bc = forward_ode(x_bc, [w, b, v])
    diff_bc = y_bc - 1.0
    loss_bc = np.mean(diff_bc**2)

    dv_bc = 2 * diff_bc @ h_bc.T
    dw_bc = 2 * diff_bc * v.T * sp_bc @ x_bc.T
    db_bc = 2 * diff_bc * v.T * sp_bc

    x_col = sample_ode(n_collocation, 0, 2)
    R = ode_residual(x_col, [w, b, v], forward_ode)
    loss_pde = np.mean(R**2)

    g = (2 / n_collocation) * R
    y, y_x, z, h, sp = forward_ode(x_col, [w, b, v])
    spp = -2 * h * sp

    dv_pde = g @ ((w * sp) + 3 * h).T

    dy_dw = v.T * sp * x_col
    dyx_dw = v.T * (sp + w * spp * x_col)
    dw_pde = np.sum(g * (dyx_dw + 3 * dy_dw), axis=1, keepdims=True)

    db_pde = np.sum(
        g * (v.T * w * spp + 3 * v.T * sp),
        axis=1,
        keepdims=True
    )
    optimizer.step([
        dw_pde + bc_weight * dw_bc,
        db_pde + bc_weight * db_bc,
        dv_pde + bc_weight * dv_bc
    ])

    if epoch % 5000 == 0:
        total_loss = loss_pde + bc_weight * loss_bc
        print(
            f"Epoch {epoch:5d} | "
            f"PDE Loss = {loss_pde:.6e} | "
            f"BC Loss = {loss_bc:.6e} | "
            f"Total Loss = {total_loss:.6e}"
        )

x_test = np.linspace(0, 2, 200).reshape(1, -1)
y_pred, *_ = forward_ode(x_test, optimizer.params)

y_exact = (11/9)*np.exp(-3*x_test) + (2/3)*x_test - 2/9

plt.figure(figsize=(6,4))
plt.plot(x_test.flatten(), y_exact.flatten(), 'k-', label="Exact")
plt.plot(x_test.flatten(), y_pred.flatten(), 'r--', label="PINN")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend()
plt.grid(True)
plt.show()
