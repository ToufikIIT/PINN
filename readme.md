## Physics-Informed Neural Network Playground

Lightweight, NumPy-only implementation of a Physics-Informed Neural Network (PINN) for solving partial differential equations. The included example trains a PINN on the 1D viscous Burgers equation.

### Features
- Minimal dependencies (NumPy + Matplotlib for plotting)
- Fully manual PINN forward/derivative pass (`core/network.py`)
- Finite-difference gradient estimation in the trainer (easy to read/modify)
- SGD with momentum optimizer
- Example experiment for 1D Burgers’ equation with initial/boundary losses

### Project Structure
- `experiments/run_burgers.py` – end-to-end training script for Burgers 1D
- `core/network.py` – feedforward network with analytical derivatives wrt x, t
- `core/trainer.py` – PINN loss (PDE + IC/BC) and finite-difference grads
- `core/optimizer.py` – SGD with momentum
- `physics/burgers_1d.py` – Burgers equation residual
- `utils/sampling.py` – Latin-style random collocation sampling
- `constraints/` – simple IC/BC helpers (not used in the current script)

### Requirements
- Python 3.9+ (tested with recent CPython)
- Packages: `numpy`, `matplotlib`

Install deps:
```bash
pip install -r requirements.txt  # if you create one
# or
pip install numpy matplotlib
```

### Quick Start (Burgers 1D)
From the repo root:
```bash
python experiments/run_burgers.py
```

What happens:
- Builds a PINN with layer sizes `[2, 20, 20, 1]` using `tanh` activations.
- Samples random collocation points in `(x, t) ∈ [0,1] × [0,1]`.
- Applies initial condition `u(x,0) = -sin(πx)` and zero Dirichlet boundaries.
- Trains for 3000 epochs with SGD + momentum (`lr=1e-4`, `gamma=0.9`).
- Prints loss every 200 epochs and plots the predicted vs exact initial profile.

### Customization Guide
- **Network architecture**: edit the layer list in `experiments/run_burgers.py`.
- **Optimizer**: tweak `lr` or `gamma` in `SGD_Momentum` instantiation.
- **Physics**: change viscosity `nu` in `Burgers1D(nu=...)` or swap in another PDE class implementing `residual(model, X)`.
- **Initial/Boundary conditions**: adjust `initial_condition` or modify `PINNTrainer.loss` for different BCs/weights (`ic_weight`, `bc_weight`).
- **Sampling**: change domain or number of collocation points via `sample_collocation`.

### How It Works
1. **Forward + derivatives**: `NeuralField.forward_with_derivatives` returns `u`, `u_x`, `u_t`, `u_xx` via analytic differentiation of the network.
2. **PDE residual**: `Burgers1D.residual` computes `u_t + u u_x - ν u_xx`.
3. **Loss**: `PINNTrainer.loss` combines PDE residual MSE with IC/BC penalties.
4. **Gradients**: `PINNTrainer.finite_difference_grads` perturbs each parameter to approximate gradients (simple but slow).
5. **Update**: `SGD_Momentum.step` applies momentum updates.

### Tips and Limitations
- Finite-difference gradients are **slow**; for larger nets switch to autograd/JAX/PyTorch.
- Current BCs are hardcoded to zero at `x=0` and `x=1`; adjust as needed.
- Training is stochastic; set `np.random.seed` for reproducibility (already set in the script).
- Increase `n_collocation` and epochs for better accuracy, but expect longer runtimes.

### Next Steps
- Add autograd-based gradients to speed up training.
- Implement additional PDEs in `physics/` and configurable boundary types.
- Log metrics/plots to disk and add unit tests for residuals and derivatives.

