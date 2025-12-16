## Physics-Informed Neural Network Playground

Lightweight, NumPy-only implementation of a Physics-Informed Neural Network (PINN) for solving partial differential equations and ordinary differential equations. The included examples train PINNs on:
- **1D viscous Burgers equation** (PDE)
- **Linear ODE** (y' + 3y = 2x)

### Features
- Minimal dependencies (NumPy + Matplotlib for plotting)
- Fully manual PINN forward/derivative pass (`core/network.py`)
- Two network architectures: `NeuralField` for PDEs (2D input: x, t) and `forward_ode` for ODEs (1D input: x)
- Finite-difference gradient estimation in the trainer (easy to read/modify)
- SGD with momentum optimizer
- Example experiments for both PDE and ODE problems with boundary/initial condition losses

### Project Structure
- `experiments/run_burgers.py` – end-to-end training script for Burgers 1D (PDE)
- `experiments/run_linear.py` – end-to-end training script for linear ODE
- `core/network.py` – feedforward network with analytical derivatives:
  - `NeuralField` class for PDEs (2D input: x, t)
  - `forward_ode` function for ODEs (1D input: x)
- `core/trainer.py` – PINN loss (PDE + IC/BC) and finite-difference grads
- `core/optimizer.py` – SGD with momentum
- `physics/burgers_1d.py` – Burgers equation residual
- `physics/linear_differential.py` – Linear ODE residual (y' + 3y = 2x)
- `utils/sampling.py` – Random collocation sampling for PDEs and ODEs
- `constraints/` – simple IC/BC helpers (not used in the current scripts)

### Requirements
- Python 3.9+ (tested with recent CPython)
- Packages: `numpy`, `matplotlib`

Install deps:
```bash
pip install -r requirements.txt  # if you create one
# or
pip install numpy matplotlib
```

### Quick Start

#### Burgers 1D (PDE Example)
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

#### Linear ODE Example
From the repo root:
```bash
python experiments/run_linear.py
```

What happens:
- Builds a PINN with 30 hidden units for solving `y' + 3y = 2x`.
- Uses boundary condition `y(0) = 1` and domain `x ∈ [0, 2]`.
- Samples 100 random collocation points.
- Trains for 30000 epochs with SGD + momentum (`lr=1e-3`, `gamma=0.9`).
- Prints loss every 5000 epochs and plots the predicted vs exact solution.
- Exact solution: `y(x) = (11/9)exp(-3x) + (2/3)x - 2/9`

### Customization Guide
- **Network architecture**: 
  - For PDEs: edit the layer list in `experiments/run_burgers.py`.
  - For ODEs: adjust `n_hidden` in `experiments/run_linear.py`.
- **Optimizer**: tweak `lr` or `gamma` in `SGD_Momentum` instantiation.
- **Physics**: 
  - Change viscosity `nu` in `Burgers1D(nu=...)` or swap in another PDE class implementing `residual(model, X)`.
  - For ODEs: modify `ode_residual` in `physics/linear_differential.py` to implement different ODEs.
- **Initial/Boundary conditions**: 
  - For PDEs: adjust `initial_condition` or modify `PINNTrainer.loss` for different BCs/weights (`ic_weight`, `bc_weight`).
  - For ODEs: modify the boundary condition in the training loop and adjust `bc_weight`.
- **Sampling**: 
  - For PDEs: change domain or number of collocation points via `sample_collocation`.
  - For ODEs: change domain via `sample_ode(n, xmin, xmax)`.

### How It Works

**For PDEs (Burgers example):**
1. **Forward + derivatives**: `NeuralField.forward_with_derivatives` returns `u`, `u_x`, `u_t`, `u_xx` via analytic differentiation of the network.
2. **PDE residual**: `Burgers1D.residual` computes `u_t + u u_x - ν u_xx`.
3. **Loss**: `PINNTrainer.loss` combines PDE residual MSE with IC/BC penalties.
4. **Gradients**: `PINNTrainer.finite_difference_grads` perturbs each parameter to approximate gradients (simple but slow).
5. **Update**: `SGD_Momentum.step` applies momentum updates.

**For ODEs (Linear example):**
1. **Forward + derivatives**: `forward_ode` returns `y`, `y_x` (and intermediate values) via analytic differentiation.
2. **ODE residual**: `ode_residual` computes `y_x + 3y - 2x`.
3. **Loss**: Manual computation combining ODE residual MSE with boundary condition penalty.
4. **Gradients**: Manual gradient computation using chain rule.
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

