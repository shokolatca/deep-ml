import numpy as np

def activation(x):
    '''
    Apply an activation function element-wise.
    
    Args:
        x: numpy array of any shape (raw neuron outputs)
    
    Returns:
        numpy array of same shape (activated outputs)
    
    Requirements:
        - Must be non-linear (not just returning x)
        - Must work on arrays of any shape
        - Must be deterministic
    '''
    # TODO: Implement your activation function
    alpha = 0.1
    sigmoid = 1 / (1 + np.exp(-np.clip(x, -60, 60)))
    return alpha * x + (1 - alpha) * x * sigmoid
