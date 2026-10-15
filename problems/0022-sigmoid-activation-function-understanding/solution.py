import torch

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    # Your implementation here
    z_val = float(z)
    # Стабильная схема: разделяем вычисления для разных знаков z
    if z_val >= 0:
        res = 1.0 / (1.0 + torch.exp(torch.tensor(-z_val)))
    else:
        # Для глубоких отрицательных чисел, чтобы e^(-z) не улетело в бесконечность
        exp_z = torch.exp(torch.tensor(z_val))
        res = exp_z / (1.0 + exp_z)
        
    return round(float(res), 4)
