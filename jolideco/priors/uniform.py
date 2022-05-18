import torch

class UniformPrior:
    """Uniform prior"""
    def __init__(self):
        pass

    def __call__(self, flux):
        return torch.tensor(0)
