import numpy as np

def tanh(z):
    return np.tanh(z)

def d_tanh(z):
    return 1.0 - np.tanh(z)**2

def dd_tanh(z):
    return -2.0 * np.tanh(z) * (1.0 - np.tanh(z)**2)
