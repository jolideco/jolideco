import torch
from jolideco.utils.torch import convolve_fft_torch
from astropy.convolution import Gaussian2DKernel

__all__ = [
    "UniformPrior",
    "ImagePrior",
    "LIRAPrior",
    "SmoothnessPrior",
]


class UniformPrior:
    """Uniform prior"""
    def __init__(self):
        pass

    def __call__(self, flux):
        """Evaluate the prior

        Parameters
        ----------
        flux : `~pytorch.Tensor`
            Reconstructed flux

        Returns
        -------
        log_prior ; `~torch.tensor`
            Log prior set to zero.
        """
        return torch.tensor(0)


class ImagePrior:
    """Image prior

    Parameters
    ----------
    flux_prior : `~pytorch.Tensor`
        Prior image
    flux_prior_error : `~pytorch.Tensor`
        Prior error image
    beta : float
        Weight factor
    """
    def __init__(self, flux_prior, flux_prior_error=None, beta=1e-6):
        self.flux_prior = flux_prior
        self.flux_prior_error = flux_prior_error
        self.beta = beta

    def __call__(self, flux):
        """Evaluate the prior

        Parameters
        ----------
        flux : `~pytorch.Tensor`
            Reconstructed flux
        """
        return self.beta * (flux - flux_prior) ** 2


class SmoothnessPrior:
    """Gradient based smoothness prior"""
    def __init__(self, width=2):
        kernel = Gaussian2DKernel(width)
        self.kernel = torch.from_numpy(kernel.array[None, None])

    def __call__(self, flux):
        smooth = convolve_fft_torch(flux, self.kernel)
        return -torch.sum(flux * smooth)

