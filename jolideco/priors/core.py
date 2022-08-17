import torch
import torch.nn as nn
import scipy
from jolideco.utils.torch import convolve_fft_torch, cycle_spin_subpixel
from astropy.convolution import Gaussian2DKernel
from astropy.utils import lazyproperty

__all__ = [
    "Prior",
    "Priors",
    "UniformPrior",
    "ImagePrior",
    "SmoothnessPrior",
    "PointSourcePrior",
]


class Prior(nn.Module):
    """Prior base class"""

    pass


class Priors(nn.ModuleDict):
    """Dict of mutiple priors"""

    def __call__(self, fluxes):
        """Evaluate all priors

        Parameters
        ----------
        fluxes : dict of `~torch.Tensor`
            Dict of flux tensors

        Returns
        -------
        log_prior : dict of `~torch.tensor`
            Dict of log priors
        """
        values = {}

        for name, flux in fluxes.items():
            values[name] = self[name](flux=flux)

        return values


class UniformPrior(Prior):
    """Uniform prior"""

    def __init__(self):
        super().__init__()

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


class PointSourcePrior(Prior):
    """Sparse prior for point sources

    Defined by a product of inverse Gamma distributions. See e.g. [ref]_

    .. [ref] https://doi.org/10.1051/0004-6361/201323006


    Parameters
    ----------
    alpha : float
        Alpha parameter
    beta : float
        Beta parameter

    """

    def __init__(self, alpha, beta=3 / 2, cycle_spin_subpix=False, generator=None):
        super().__init__()
        self.alpha = torch.tensor([alpha])
        self.beta = torch.tensor([beta])

        self.cycle_spin_subpix = cycle_spin_subpix

        if generator is None:
            generator = torch.Generator()

        self.generator = generator

    @lazyproperty
    def mean(self):
        """Mean of the distribution"""
        return self.beta / (self.alpha - 1)

    @lazyproperty
    def mode(self):
        """Mean of the distribution"""
        return self.beta / (self.alpha + 1)

    @lazyproperty
    def log_constant_term(self):
        """Log constant term"""
        value = self.alpha * torch.log(self.beta)
        value -= torch.lgamma(self.alpha)
        return float(value)

    def __call__(self, flux):
        """Evaluate the prior

        Parameters
        ----------
        flux : `~pytorch.Tensor`
            Reconstructed point source flux

        Returns
        -------
        log_prior ; `~torch.tensor`
            Log prior value.
        """
        if self.cycle_spin_subpix:
            flux = cycle_spin_subpixel(image=flux, generator=self.generator)

        value = -self.beta / flux
        value += (-self.alpha - 1) * torch.log(flux)
        value_sum = torch.sum(value) + flux.numel() * self.log_constant_term
        return value_sum


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
        super().__init__()
        self.flux_prior = flux_prior
        self.flux_prior_error = flux_prior_error

    def __call__(self, flux):
        """Evaluate the prior

        Parameters
        ----------
        flux : `~pytorch.Tensor`
            Reconstructed flux
        """
        return ((flux - self.flux_prior) / self.flux_prior_error) ** 2


class SmoothnessPrior(Prior):
    """Gradient based smoothness prior"""

    def __init__(self, width=2):
        super().__init__()
        kernel = Gaussian2DKernel(width)
        self.kernel = torch.from_numpy(kernel.array[None, None])

    def __call__(self, flux):
        smooth = convolve_fft_torch(flux, self.kernel)
        return -torch.sum(flux * smooth)
