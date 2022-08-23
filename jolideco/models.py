import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.torch import convolve_fft_torch

__all__ = ["FluxComponent", "NPredModel"]


class FluxComponent(nn.Module):
    """Flux component

    Attributes
    ----------
    flux_init : `~torch.Tensor`
        Initial flux tensor
    use_log_flux : bool
        Use log scaling for flux
    """

    def __init__(self, flux_init, use_log_flux=True):
        super().__init__()

        if use_log_flux:
            flux_init = torch.log(flux_init)

        self._flux = nn.Parameter(flux_init)
        self._use_log_flux = use_log_flux

    @property
    def shape(self):
        """Shape of the flux component"""
        return self._flux.shape

    @property
    def use_log_flux(self):
        """Use log flux (`bool`)"""
        return self._use_log_flux

    @property
    def flux(self):
        """Flux (`torch.Tensor`)"""
        if self.use_log_flux:
            return torch.exp(self._flux)
        else:
            return self._flux


class FluxComponents(nn.ModuleDict):
    """Flux components"""

    def to_dict(self):
        """Fluxes of the components ()"""
        fluxes = {}

        for name, component in self.items():
            fluxes[name] = component.flux

        return fluxes

    def to_numpy(self):
        """Fluxes of the components ()"""
        fluxes = {}

        for name, component in self.items():
            flux_cpu = component.flux.detach().cpu()
            fluxes[name] = np.squeeze(flux_cpu.numpy())

        return fluxes

    def to_tuple(self):
        """Fluxes as tuple"""
        return tuple(self.values())

    def evaluate(self):
        """Total flux"""
        # if not self:
        #    return

        values = list(self.values())

        flux = torch.zeros(values[0].shape)

        for component in values:
            flux += component.flux

        return flux

    def read(self, filename):
        """Read flux components"""
        raise NotImplementedError

    def write(self, filename, overwrite=False):
        """Write flux components"""
        raise NotImplementedError


class NPredModel(nn.Module):
    """Predicted counts model with mutiple components

    Attributes
    ----------
    components : dict of `FluxModel`
        Dict of flux model components
    upsampling_factor : None
        Spatial upsampling factor for the flux

    """

    def __init__(self, components, upsampling_factor=None):
        super().__init__()
        self.components = components
        self.upsampling_factor = upsampling_factor
        self.background_norm = nn.Parameter(torch.tensor([1.0]))

    def forward(self, background, exposure, psf=None, rmf=None):
        """Forward folding model evaluation.

        Parameters
        ----------
        background : `~torch.Tensor`
            Background tensor
        exposure : `~torch.Tensor`
            Exposure tensor
        psf : `~torch.Tensor`
            Point spread function
        rmf : `~torch.Tensor`
            Energy redistribution matrix.

        Returns
        -------
        npred : `~torch.Tensor`
            Predicted number of counts
        """
        flux = self.components.evaluate()
        npred = (flux + self.background_norm * background) * exposure

        if psf is not None:
            npred = convolve_fft_torch(npred, psf)

        if self.upsampling_factor:
            npred = F.avg_pool2d(
                npred, kernel_size=self.upsampling_factor, divisor_override=1
            )

        if rmf is not None:
            # TODO: simplify if possible
            npred = torch.matmul(npred[0].T, rmf).T[None]

        return npred
