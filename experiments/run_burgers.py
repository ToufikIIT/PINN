import numpy as np
import matplotlib.pyplot as plt
from core.network import NeuralField
from core.optimizer import SGD_Momentum
from core.trainer import PINNTrainer
from physics.burgers_1d import Burgers1D
from utils.sampling import sample_collocation

np.random.seed(0)
model = NeuralField([2, 20, 20, 1])        
optimizer = SGD_Momentum(model.params(),lr=1e-4,gamma=0.9)
pde = Burgers1D(nu=0.01)
def initial_condition(x):
    return -np.sin(np.pi * x)

trainer = PINNTrainer(model=model,pde=pde,optimizer=optimizer,ic_func=initial_condition,bc_func=None,ic_weight=20.0,bc_weight=20.0)
epochs = 3000
n_collocation = 50

print("\nStarting training...\n")

for epoch in range(epochs):
    Xf = sample_collocation(n_collocation, (0, 1), (0, 1))
    loss = trainer.step(Xf)
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss = {loss:.6e}")
print("\nTraining finished.\n")

def evaluate(model, x, t):
    X = np.vstack([x, t])
    u, _, _, _ = model.forward_with_derivatives(X)
    return u

x_plot = np.linspace(0, 1, 200).reshape(1, -1)
t0 = np.zeros_like(x_plot)
u_pred_0 = evaluate(model, x_plot, t0)
u_exact_0 = initial_condition(x_plot)

plt.figure(figsize=(6, 4))
plt.plot(x_plot.flatten(), u_exact_0.flatten(),'k-', linewidth=2, label="Exact")
plt.plot(x_plot.flatten(), u_pred_0.flatten(),'r--', linewidth=2, label="PINN")
plt.xlabel("x")
plt.ylabel("u(x, 0)")
plt.title("Exact vs PINN (Initial Condition)")
plt.legend()
plt.grid(True)
plt.show()

""" 
t_fixed = 0.5 * np.ones_like(x_plot)
u_pred_t = evaluate(model, x_plot, t_fixed)

plt.figure(figsize=(6, 4))
plt.plot(x_plot.flatten(), u_pred_t.flatten(),
         'b-', linewidth=2, label="PINN")
plt.xlabel("x")
plt.ylabel("u(x, 0.5)")
plt.title("PINN Prediction at t = 0.5")
plt.legend()
plt.grid(True)
plt.show() 

nx, nt = 100, 100
x = np.linspace(0, 1, nx)
t = np.linspace(0, 1, nt)

X, T = np.meshgrid(x, t)
XT = np.vstack([X.flatten(), T.flatten()])

U, _, _, _ = model.forward_with_derivatives(XT)
U = U.reshape(nt, nx)

plt.figure(figsize=(7, 4))
plt.contourf(X, T, U, 50)
plt.colorbar(label="u(x, t)")
plt.xlabel("x")
plt.ylabel("t")
plt.title("PINN Solution of Burgers’ Equation")
plt.show()
"""