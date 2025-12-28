import numpy as np
from typing import Tuple, List, Union

def rand_sample(x: np.ndarray, y: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly sample n_samples from x and y."""
    indices = np.random.choice(len(x), size=n_samples, replace=False)
    return x[indices], y[indices]


def filter_samples(x_samples: np.ndarray, y_samples: np.ndarray, A_limits: Union[Tuple, List]) -> Tuple[np.ndarray, np.ndarray]:
    """Filter samples to only include those within A_limits."""
    n_dims = x_samples.shape[1] if x_samples.ndim > 1 else 1
    
    if n_dims == 1:
        mask = (x_samples.flatten() >= A_limits[0]) & (x_samples.flatten() <= A_limits[1])
    else:
        if not isinstance(A_limits[0], (list, tuple)):
            A_limits = [A_limits] * n_dims
        
        mask = np.ones(len(x_samples), dtype=bool)
        for i in range(n_dims):
            mask &= (x_samples[:, i] >= A_limits[i][0]) & (x_samples[:, i] <= A_limits[i][1])
    
    return x_samples[mask], y_samples[mask]