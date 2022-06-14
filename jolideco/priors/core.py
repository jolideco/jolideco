import torch
import torch.nn as nn
from jolideco.utils.torch import convolve_fft_torch
from astropy.convolution import Gaussian2DKernel

__all__ = [
    "Prior" "UniformPrior",
    "ImagePrior",
    "SmoothnessPrior",
]


class Prior(nn.Module):
    pass


class UniformPrior(Prior):
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


class ImagePrior(Prior):
    """Image prior

    Parameters
    ----------
    flux_prior : `~pytorch.Tensor`
        Prior image
    flux_prior_error : `~pytorch.Tensor`
        Prior error image
    """

    def __init__(self, flux_prior, flux_prior_error=None):
        self.flux_prior = flux_prior
        self.flux_prior_error = flux_prior_error

    def __call__(self, flux):
        """Evaluate the prior

        Parameters
        ----------
        flux : `~pytorch.Tensor`
            Reconstructed flux
        """
        return (flux - self.flux_prior) ** 2


class SmoothnessPrior(Prior):
    """Gradient based smoothness prior"""

    def __init__(self, width=2):
        kernel = Gaussian2DKernel(width)
        self.kernel = torch.from_numpy(kernel.array[None, None])

    def __call__(self, flux):
        smooth = convolve_fft_torch(flux, self.kernel)
        return -torch.sum(flux * smooth)
