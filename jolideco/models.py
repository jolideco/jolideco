import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.torch import convolve_fft_torch

__all__ = ["FluxComponent", "FluxComponents", "NPredModel"]


class FluxComponent(nn.Module):
    """Flux component

    Attributes
    ----------
    flux_upsampled : `~torch.Tensor`
        Initial flux tensor
    use_log_flux : bool
        Use log scaling for flux
    upsampling_factor : None
        Spatial upsampling factor for the flux.
    """

    def __init__(self, flux_upsampled, use_log_flux=True, upsampling_factor=None):
        super().__init__()

        if use_log_flux:
            flux_upsampled = torch.log(flux_upsampled)

        self._flux_upsampled = nn.Parameter(flux_upsampled)
        self._use_log_flux = use_log_flux
        self.upsampling_factor = upsampling_factor

    @classmethod
    def from_flux_init_numpy(cls, flux_init, **kwargs):
        """Create flux component

        Parameters
        ----------
        flux_init : `~numpy.ndarray`
            Flux init array
        **kwargs : dict
            Keyword arguments passsed to `FluxComponent`

        Returns
        -------
        flux_component : `FluxComponent`
            Flux component
        """
        upsampling_factor = kwargs.get("upsampling_factor", None)

        # convert to pytorch tensors
        flux_init = torch.from_numpy(flux_init[np.newaxis, np.newaxis])

        if upsampling_factor:
            flux_init = F.interpolate(
                flux_init, scale_factor=upsampling_factor, mode="bilinear"
            )

        return cls(flux_upsampled=flux_init, **kwargs)

    @property
    def shape(self):
        """Shape of the flux component"""
        return self._flux_upsampled.shape

    @property
    def use_log_flux(self):
        """Use log flux (`bool`)"""
        return self._use_log_flux

    @property
    def flux_upsampled(self):
        """Flux (`torch.Tensor`)"""
        if self.use_log_flux:
            return torch.exp(self._flux_upsampled)
        else:
            return self._flux_upsampled

    @property
    def flux(self):
        """Flux (`torch.Tensor`)"""
        flux = self.flux_upsampled

        if self.upsampling_factor:
            flux = F.avg_pool2d(
                self.flux_upsampled,
                kernel_size=self.upsampling_factor,
                divisor_override=1,
            )
        return flux


class FluxComponents(nn.ModuleDict):
    """Flux components"""

    def to_dict(self):
        """Fluxes of the components ()"""
        fluxes = {}

        for name, component in self.items():
            fluxes[name] = component.flux_upsampled

        return fluxes

    def to_numpy(self):
        """Fluxes of the components ()"""
        fluxes = {}

        for name, component in self.items():
            flux_cpu = component.flux_upsampled.detach().cpu()
            fluxes[name] = np.squeeze(flux_cpu.numpy())

        return fluxes

    def to_flux_tuple(self):
        """Fluxes as tuple"""
        return tuple([_.flux_upsampled for _ in self.values()])

    def evaluate(self):
        """Total flux"""
        values = list(self.values())

        flux = torch.zeros(values[0].shape)

        for component in values:
            flux += component.flux_upsampled

        return flux

    def read(self, filename):
        """Read flux components"""
        raise NotImplementedError

    def write(self, filename, overwrite=False):
        """Write flux components"""
        raise NotImplementedError


class NPredModel(nn.Module):
    """Predicted counts model with mutiple components

    Parameters
    ----------
    flux : `~torch.Tensor`
        Flux tensor
    background : `~torch.Tensor`
        Background tensor
    exposure : `~torch.Tensor`
        Exposure tensor
    psf : `~torch.Tensor`
        Point spread function
    rmf : `~torch.Tensor`
        Energy redistribution matrix.

    """

    def __init__(
        self, background, exposure, psf=None, rmf=None, upsampling_factor=None
    ):
        super().__init__()
        self.background = background
        self.exposure = exposure
        self.psf = psf
        self.rmf = rmf
        self.background_norm = nn.Parameter(torch.tensor([1.0]))
        self.upsampling_factor = upsampling_factor

    @property
    def shape(self):
        """Shape of the NPred model"""
        return self.background.shape

    @classmethod
    def from_dataset_numpy(
        cls,
        dataset,
        upsampling_factor=None,
        correct_exposure_edges=True,
    ):
        """Convert dataset to dataset of pytorch tensors

        Parameters
        ----------
        dataset : dict of `~numpy.ndarray`
            Dict containing `"counts"`, `"psf"` and optionally `"exposure"` and `"background"`
        upsampling_factor : int
            Upsampling factor for exposure, background and psf.
        correct_exposure_edges : bool
            Correct psf leakage at the exposure edges.

        Returns
        -------
        npred_model : `NPredModel`
            Predicted counts model
        """
        dims = (np.newaxis, np.newaxis)

        kwargs = {}

        for name in ["psf", "exposure", "background"]:
            value = dataset[name]
            tensor = torch.from_numpy(value[dims])

            if upsampling_factor:
                tensor = F.interpolate(
                    tensor, scale_factor=upsampling_factor, mode="bilinear"
                )

            if name in ["psf", "background", "flux"] and upsampling_factor:
                tensor = tensor / upsampling_factor**2

            kwargs[name] = tensor

        if correct_exposure_edges:
            exposure = kwargs["exposure"]
            weights = convolve_fft_torch(
                image=torch.ones_like(exposure), kernel=kwargs["psf"]
            )
            kwargs["exposure"] = exposure / weights

        return cls(**kwargs)

    def forward(self, flux):
        """Forward folding model evaluation.

        Parameters
        ----------
        flux : `~torch.Tensor`
            Flux tensor

        Returns
        -------
        npred : `~torch.Tensor`
            Predicted number of counts
        """
        npred = (flux + self.background_norm * self.background) * self.exposure

        if self.psf is not None:
            npred = convolve_fft_torch(npred, self.psf)

        if self.upsampling_factor:
            npred = F.avg_pool2d(
                npred, kernel_size=self.upsampling_factor, divisor_override=1
            )

        if self.rmf is not None:
            # TODO: simplify if possible
            npred = torch.matmul(npred[0].T, self.rmf).T[None]

        return npred


class NPredModels(nn.ModuleDict):
    """Flux components"""

    def evaluate(self, fluxes):
        """Evaluate npred model"""
        values = list(self.values())

        npred_total = torch.zeros(values[0].shape)

        for npred_model, flux in zip(values, fluxes):
            npred = npred_model(flux=flux)
            npred_total += npred

        return npred_total

    @classmethod
    def from_dataset_nunpy(cls, dataset, components):
        """Create multiple npred models."""
        values = []

        for name, component in components.items():
            npred_model = NPredModel.from_dataset_numpy(
                dataset=dataset, upsampling_factor=component.upsampling_factor
            )
            values.append((name, npred_model))

        return cls(values)
