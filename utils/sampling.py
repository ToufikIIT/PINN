import numpy as np

def sample_collocation(n, xlim=(0,1), tlim=(0,1)):
    x = np.random.uniform(xlim[0], xlim[1], (1, n))
    t = np.random.uniform(tlim[0], tlim[1], (1, n))
    return np.vstack([x, t])
