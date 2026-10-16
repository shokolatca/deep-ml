import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    # Your implementation here

    logits = torch.tensor(scores, dtype = torch.float32)
    logits = logits - logits.max()
    exp_scores = torch.exp(logits)
    probabilities = exp_scores / exp_scores.sum()
    return probabilities.tolist()
