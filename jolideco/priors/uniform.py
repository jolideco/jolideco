import torch

__all__ = [
    "UniformPrior",
]


class UniformPrior:
    """Uniform prior"""
    def __init__(self):
        pass

    def __call__(self, flux):
        return torch.tensor(0)


class ImagePrior:
    """Image prior"""
    def __init__(self, flux_prior):
        self.flux_prior = flux_prior

    def __call__(self, flux):

        return (flux - flux_prior) ** 2
