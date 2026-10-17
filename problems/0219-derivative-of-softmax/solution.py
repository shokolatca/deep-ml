import torch

def softmax_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Jacobian matrix of the softmax function using PyTorch.
    
    Args:
        x: Input tensor
        
    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """
    # Your code here - you can use torch.autograd.functional.jacobian
    # or compute it directly from softmax output
    z_tensors = x - x.max()
    softmax_scores = torch.exp(z_tensors) / torch.exp(z_tensors).sum()
    n = len(softmax_scores)
    jacobian_matrix = torch.zeros((n, n), dtype=x.dtype, device=x.device)

    for i in range(n):
        for j in range(n):
            if i == j:
                jacobian_matrix[i, j] = softmax_scores[i] * (1 - softmax_scores[i])
            else:
                jacobian_matrix[i, j] = -softmax_scores[i] * softmax_scores[j]

    return jacobian_matrix