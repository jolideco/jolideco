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
    """Image prior

    Parameters
    ----------
    flux_prior : `~numpy.ndarray`
        Prior image
    beta : float
        Weight factor
    """
    def __init__(self, flux_prior, flux_prior_error=None, beta=1e-6):
        self.flux_prior = flux_prior
        self.flux_prior_error = flux_prior_error
        self.beta = beta

    def __call__(self, flux):

        return self.beta * (flux - flux_prior) ** 2


class LIRAPrior:
    """LIRA multiscale prior

    Parameters
    ----------
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def cycle_spin(self):
        pass

    def multiscale(self):
        pass

    def __call__(self, flux):
        return 0
