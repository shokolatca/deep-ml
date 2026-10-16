import torch

def relu(z: float) -> torch.Tensor:
    """
    Implements the ReLU activation function using PyTorch.
    
    Args:
        z: A float input value.
    
    Returns:
        A torch.Tensor with ReLU applied (max(0, z)).
    """
    z_tensor = torch.tensor(z, dtype=torch.float32) if not isinstance(z, torch.Tensor) else z

    return torch.max(torch.tensor(0.0, dtype = torch.float32), z_tensor)