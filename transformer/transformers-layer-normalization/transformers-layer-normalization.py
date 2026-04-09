import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.

    Args:
        x: Input array of shape (..., d_model)
        gamma: Scale parameter of shape (d_model,)
        beta: Shift parameter of shape (d_model,)
        eps: Small constant for numerical stability

    Returns:
        Normalized array of same shape as x
    """
    # Your code here

    mu=np.mean(x,axis=-1,keepdims=True)
    var=np.mean((x-mu)**2,axis=-1,keepdims=True)
    std=np.sqrt(var+eps)
    x_hat=(x-mu)/std
    out=gamma*x_hat+beta
    return out