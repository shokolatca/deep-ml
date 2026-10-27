import torch

def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    # TODO: subtract max along dim, exp, then normalize
    
    max_values = t.max(dim = dim, keepdim = True).values
    exp_values = torch.exp(t - max_values)
    return exp_values / exp_values.sum(dim = dim, keepdim = True)
