import torch
import torch.nn.functional as F

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x
    using PyTorch autograd.
    
    Args:
        x: Input value
        
    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    # Your code here - use autograd!
    # Hint: Create tensors with requires_grad=True, apply activation, call .backward()
    x_tensor = torch.tensor(x, dtype = torch.float32, requires_grad=True)

    result = {}

    for name, activation in [
        ("sigmoid" , torch.sigmoid),
        ("tanh" , torch.tanh),
        ("relu" , F.relu)
    ]:
        x_tensor.grad = None 

        y = activation(x_tensor)
        y.backward()

        result[name] = x_tensor.grad.item()
    return result