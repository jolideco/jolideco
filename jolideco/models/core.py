import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.coordinates import SkyCoord
from astropy.utils import lazyproperty
from astropy.visualization import simple_norm
from astropy.wcs import WCS

from jolideco.priors.core import Prior, Priors, UniformPrior
from jolideco.utils.io import (
    IO_FORMATS_FLUX_COMPONENT_READ,
    IO_FORMATS_FLUX_COMPONENT_WRITE,
    IO_FORMATS_FLUX_COMPONENTS_READ,
    IO_FORMATS_FLUX_COMPONENTS_WRITE,
    IO_FORMATS_SPARSE_FLUX_COMPONENT_READ,
    IO_FORMATS_SPARSE_FLUX_COMPONENT_WRITE,
    document_io_formats,
    get_reader,
    get_writer,
)
from jolideco.utils.misc import format_class_str
from jolideco.utils.plot import add_cbar
from jolideco.utils.torch import grid_weights

log = logging.getLogger(__name__)


__all__ = [
    "SpatialFluxComponent",
    "FluxComponents",
    "SparseSpatialFluxComponent",
]


class SparseSpatialFluxComponent(nn.Module):
    """Sparse flux component to represent a list of point sources

    Attributes
    ----------
    flux : `~torch.Tensor`
        Initial flux tensor
    x_pos : `~torch.Tensor`
        x position in pixel coordinates
    y_pos : `~torch.Tensor`
        y position in pixel coordinates
    shape : tuple of int
        Image shape
    use_log_flux : bool
        Use log scaling for flux
    prior : `Prior`
        Prior for this flux component.
    frozen : bool
        Whether to freeze component.
    wcs : `~astropy.wcs.WCS`
        World coordinate transform object
    """

    is_sparse = True
    upsampling_factor = 1

    _shape_eval = (-1, 1, 1, 1, 1)
    _shape_eval_x = (1, 1, 1, 1, -1)
    _shape_eval_y = (1, 1, 1, -1, 1)

    _registry_write = IO_FORMATS_SPARSE_FLUX_COMPONENT_WRITE
    _registry_read = IO_FORMATS_SPARSE_FLUX_COMPONENT_READ

    def __init__(
        self,
        flux,
        x_pos,
        y_pos,
        shape,
        use_log_flux=True,
        prior=None,
        frozen=False,
        wcs=None,
    ):
        super().__init__()

        if prior is None:
            prior = UniformPrior()

        if use_log_flux:
            flux = torch.log(flux)

        self.prior = prior
        self.frozen = frozen
        self._wcs = wcs
        self._shape = shape
        self._flux = nn.Parameter(flux.type(torch.float32))
        self.x_pos = nn.Parameter(x_pos.type(torch.float32))
        self.y_pos = nn.Parameter(y_pos.type(torch.float32))
        self._use_log_flux = use_log_flux

    def parameters(self, recurse=True):
        """Parameter list"""
        if self.frozen:
            return []
        else:
            return super().parameters(recurse)

    @property
    def x_pos_numpy(self) -> np.ndarray:
        """x pos as numpy array"""
        return self.x_pos.detach().cpu().numpy()

    @property
    def y_pos_numpy(self) -> np.ndarray:
        """y pos as numpy array"""
        return self.y_pos.detach().cpu().numpy()

    @property
    def sky_coord(self) -> SkyCoord:
        """Positions as SkyCoord"""
        return SkyCoord.from_pixel(
            xp=self.x_pos_numpy, yp=self.y_pos_numpy, wcs=self.wcs
        )

    @classmethod
    def from_numpy(cls, flux, x_pos, y_pos, **kwargs):
        """Create sparse flux component from numpy arrays

        Attributes
        ----------
        flux : `~numpy.ndarray`
            Initial flux tensor
        x_pos : `~numpy.ndarray`
            x position in pixel coordinates
        y_pos : `~numpy.ndarray`
            y position in pixel coordinates
        **kwargs : dict
            Keyword arguments forwarded to `SparseFluxComponent`

        Returns
        -------
        sparse_flux_component : `SparseFluxComponent`
            Sparse flux component
        """
        flux = np.atleast_1d(flux)
        x_pos = np.atleast_1d(x_pos)
        y_pos = np.atleast_1d(y_pos)

        flux = torch.from_numpy(flux.astype(np.float32))
        x_pos = torch.from_numpy(x_pos.astype(np.float32))
        y_pos = torch.from_numpy(y_pos.astype(np.float32))

        return cls(flux=flux, x_pos=x_pos, y_pos=y_pos, **kwargs)

    @classmethod
    def from_sky_coord(cls, skycoord, wcs, **kwargs):
        """Create sparse flux component from sky coordinates

        Parameters
        ----------
        skycoord: `~astropy.coordinates.SkyCoord`
            Sky coordinates
        wcs : `~astropy.wcs.WCS`
            World coordinate transform object

        Returns
        -------
        sparse_flux_component : `SparseFluxComponent`
            Sparse flux component
        """
        y_pos, x_pos = skycoord.to_pixel(wcs=wcs)
        return cls.from_numpy(x_pos=x_pos, y_pos=y_pos, **kwargs)

    @property
    def wcs(self) -> WCS:
        """Flux error"""
        return self._wcs

    @property
    def shape(self) -> tuple:
        """Shape of the flux component"""
        return (1, 1) + self._shape

    @lazyproperty
    def indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Shape of the flux component"""
        idx = torch.arange(self._shape[1], dtype=torch.float32)
        idy = torch.arange(self._shape[0], dtype=torch.float32)
        return idx.reshape(self._shape_eval_x), idy.reshape(self._shape_eval_y)

    @property
    def use_log_flux(self) -> bool:
        """Use log flux"""
        return self._use_log_flux

    @property
    def flux_numpy(self) -> np.ndarray:
        """Flux as numpy array"""
        flux_cpu = self.flux.detach().cpu()
        return flux_cpu.numpy()[0, 0]

    @property
    def flux(self) -> torch.Tensor:
        """Flux (`~torch.Tensor`)"""
        y, x = self.indices
        x0 = self.x_pos.reshape(self._shape_eval)
        y0 = self.y_pos.reshape(self._shape_eval)

        weights = grid_weights(x=x, y=y, x0=x0, y0=y0)

        if self.use_log_flux:
            flux = torch.exp(self._flux)
        else:
            flux = self._flux

        flux = weights * flux.reshape(self._shape_eval)

        return flux.sum(axis=0)

    @property
    def flux_upsampled(self) -> torch.Tensor:
        """Upsampled flux"""
        return self.flux

    def plot(self, ax=None, kwargs_norm=None, **kwargs):
        """Plot flux component as sky image

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Plotting axes
        kwargs_norm: dict
            Keyword arguments passed to `~astropy.visualization.simple_norm`
        **kwargs : dict
            Keywords forwarded to `~matplotlib.pyplot.imshow`

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
            Plotting axes
        """
        if ax is None:
            ax = plt.subplot(projection=self.wcs)

        kwargs_norm = kwargs_norm or {"min_cut": 0, "stretch": "asinh", "asinh_a": 0.01}

        flux = self.flux_numpy

        norm = simple_norm(flux, **kwargs_norm)

        kwargs.setdefault("norm", norm)
        kwargs.setdefault("interpolation", "None")

        im = ax.imshow(flux, origin="lower", **kwargs)
        add_cbar(im=im, ax=ax, fig=ax.figure)
        return ax

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert sparse flux component configuration to dict, with simple data types.

        Returns
        -------
        data : dict
            Parameter dict.
        """
        # TODO: add all parameters, flux_upsampled could be filename
        data = {}
        data["use_log_flux"] = self.use_log_flux
        data["frozen"] = self.frozen
        data["shape"] = self.shape

        if self.use_log_flux:
            flux = torch.exp(self._flux)
        else:
            flux = self._flux

        data["flux"] = flux.detach().cpu().numpy()
        data["x_pos"] = self.x_pos_numpy
        data["y_pos"] = self.y_pos_numpy
        data["prior"] = self.prior.to_dict()
        return data

    def __str__(self):
        """String representation"""
        return format_class_str(instance=self)

    @document_io_formats(registry=_registry_write)
    def write(self, filename, format=None, overwrite=False, **kwargs):
        """Write flux component to file

        Parameters
        ----------
        filename : str or `Path`
            Output filename
        overwrite : bool
            Overwrite file.
        format : {formats}
            Format to use.
        """
        writer = get_writer(
            filename=filename, format=format, registry=self._registry_write
        )

        return writer(
            flux_component=self, filename=filename, overwrite=overwrite, **kwargs
        )

    @classmethod
    @document_io_formats(registry=_registry_read)
    def read(cls, filename, format=None):
        """Read sparse flux component from file

        Parameters
        ----------
        filename : str or `Path`
            Output filename
        format : {formats}
            Format to use.

        Returns
        -------
        flux_component : `SparseFluxComponent`
            Flux component
        """
        reader = get_reader(
            filename=filename, format=format, registry=cls._registry_read
        )
        return reader(filename)


def freeze_mask(module, grad_input, grad_output):
    """Freeze masked parameters"""

    if module.mask:
        grad_input = grad_input * module.mask

    return grad_input


class SpatialFluxComponent(nn.Module):
    """Flux component

    Attributes
    ----------
    flux_upsampled : `~torch.Tensor`
        Initial flux tensor
    flux_upsampled_error : `~torch.Tensor`
        Flux tensor error
    use_log_flux : bool
        Use log scaling for flux
    upsampling_factor : None
        Spatial upsampling factor for the flux.
    prior : `Prior`
        Prior for this flux component.
    frozen : bool
        Whether to freeze component.
    wcs : `~astropy.wcs.WCS`
        World coordinate transform object
    """

    is_sparse = False
    _registry_read = IO_FORMATS_FLUX_COMPONENT_READ
    _registry_write = IO_FORMATS_FLUX_COMPONENT_WRITE

    def __init__(
        self,
        flux_upsampled,
        flux_upsampled_error=None,
        mask=None,
        use_log_flux=True,
        upsampling_factor=1,
        prior=None,
        frozen=False,
        wcs=None,
    ):
        super().__init__()

        if not flux_upsampled.ndim == 4:
            raise ValueError(
                f"Flux tensor must be four dimensional. Got {flux_upsampled.ndim}"
            )

        if use_log_flux:
            flux_upsampled = torch.log(flux_upsampled)

        self._flux_upsampled = nn.Parameter(flux_upsampled)
        self._flux_upsampled_error = flux_upsampled_error

        if mask is not None and not mask.shape == flux_upsampled.shape:
            raise ValueError(
                "Flux and mask need to have the same shape, got "
                f"{flux_upsampled.shape} and {mask.shape}"
            )

        self.mask = mask

        self._use_log_flux = use_log_flux
        self.upsampling_factor = int(upsampling_factor)

        if prior is None:
            prior = UniformPrior()

        self.prior = prior
        self.frozen = frozen
        self._wcs = wcs
        self.register_full_backward_hook(freeze_mask)

    def to_dict(self, include_data=None) -> dict[str, Any]:
        """Convert flux component configuration to dict, with simple data types.

        Parameters
        ----------
        include_data : None or {"numpy"}
            Optionally include data array in the given format

        Returns
        -------
        data : dict
            Parameter dict.
        """
        # TODO: add all parameters, flux_upsampled could be filename
        data = {}
        data["use_log_flux"] = self.use_log_flux
        data["upsampling_factor"] = int(self.upsampling_factor)
        data["frozen"] = self.frozen
        data["prior"] = self.prior.to_dict()

        if include_data == "numpy":
            data["flux_upsampled"] = self.flux_upsampled_numpy

        return data

    @classmethod
    def from_dict(cls, data):
        """Create flux component from dict

        Parameters
        ----------
        data : dict
            Parameter dict.

        Returns
        -------
        flux_component : `FluxComponent`
            Flux component
        """
        kwargs = data.copy()
        prior_data = kwargs.pop("prior", None)

        if prior_data:
            kwargs["prior"] = Prior.from_dict(data=prior_data)

        value = kwargs["flux_upsampled"]

        if isinstance(value, str):
            filename = Path(value)
            flux = cls.read(filename).flux_upsampled
        elif not isinstance(value, torch.Tensor):
            flux = torch.from_numpy(value[np.newaxis, np.newaxis].astype(np.float32))
        else:
            flux = value

        kwargs["flux_upsampled"] = flux
        return cls(**kwargs)

    def __str__(self):
        """String representation"""
        return format_class_str(instance=self)

    @property
    def wcs(self) -> WCS:
        """Flux error"""
        return self._wcs

    def parameters(self, recurse=True):
        """Parameter list"""
        if self.frozen:
            return []
        else:
            return super().parameters(recurse)

    @classmethod
    def from_numpy(cls, flux, mask=None, **kwargs):
        """Create flux component from downsampled data.

        Parameters
        ----------
        flux : `~numpy.ndarray`
            Flux init array with 2 dimensions
        **kwargs : dict
            Keyword arguments passed to `FluxComponent`

        Returns
        -------
        flux_component : `FluxComponent`
            Flux component
        """
        upsampling_factor = kwargs.get("upsampling_factor", None)

        # convert to pytorch tensors
        flux = torch.from_numpy(flux[np.newaxis, np.newaxis].astype(np.float32))

        if upsampling_factor:
            flux = F.interpolate(flux, scale_factor=upsampling_factor, mode="bilinear")

        if mask is not None:
            mask = torch.from_numpy(mask[np.newaxis, np.newaxis].astype(bool))

            if upsampling_factor:
                mask = F.interpolate(
                    mask.type(torch.float32),
                    scale_factor=upsampling_factor,
                    mode="bilinear",
                )
                mask = mask > 0.5

        return cls(flux_upsampled=flux, mask=mask, **kwargs)

    @classmethod
    def from_flux_init_datasets(cls, datasets, **kwargs):
        """Compute flux init from datasets by averaging over the raw flux estimate.

        Parameters
        ----------
        datasets : list of dict
            List of dictionaries containing, "counts", "psf",
            "background" and "exposure".
        **kwargs : dict
            Keyword arguments passed to `FluxComponent`

        Returns
        -------
        flux_init : `~numpy.ndarray`
            Initial flux estimate.
        """
        fluxes = []

        for dataset in datasets:
            flux = dataset["counts"] / dataset["exposure"] - dataset["background"]
            fluxes.append(flux)

        flux_init = np.nanmean(fluxes, axis=0)
        return cls.from_numpy(flux=flux_init, **kwargs)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Shape of the flux component"""
        return self._flux_upsampled.shape

    @property
    def shape_image(self) -> tuple[int, int]:
        """Image shape of the flux component"""
        return self.shape[-2:]

    @property
    def use_log_flux(self) -> bool:
        """Use log flux"""
        return self._use_log_flux

    @property
    def flux_upsampled(self) -> torch.Tensor:
        """Flux"""
        flux = self._flux_upsampled

        if self.use_log_flux:
            flux = torch.exp(flux)

        if self.mask is not None:
            flux = flux * self.mask

        return flux

    @property
    def flux(self) -> torch.Tensor:
        """Flux as torch tensor"""
        flux = self.flux_upsampled

        if self.upsampling_factor:
            flux = F.avg_pool2d(
                flux,
                kernel_size=self.upsampling_factor,
                divisor_override=1,
            )
        return flux

    @property
    def flux_upsampled_error(self) -> torch.Tensor:
        """Flux error as torch tensor"""
        return self._flux_upsampled_error

    @property
    def flux_numpy(self) -> np.ndarray:
        """Flux as numpy array"""
        flux_cpu = self.flux.detach().cpu()
        return flux_cpu.numpy()[0, 0]

    @property
    def flux_upsampled_numpy(self) -> np.ndarray:
        """Flux upsampled as numpy array"""
        flux_cpu = self.flux_upsampled.detach().cpu()
        return flux_cpu.numpy()[0, 0]

    @property
    def flux_upsampled_error_numpy(self) -> np.ndarray:
        """Flux error upsampled as numpy array"""
        flux_error_cpu = self.flux_upsampled_error.detach().cpu()
        return flux_error_cpu.numpy()[0, 0]

    @classmethod
    @document_io_formats(registry=_registry_read)
    def read(cls, filename, format=None):
        """Read flux component from file

        Parameters
        ----------
        filename : str or `Path`
            Output filename
        format : {formats}
            Format to use.

        Returns
        -------
        flux_component : `FluxComponent`
            Flux component
        """
        reader = get_reader(
            filename=filename, format=format, registry=cls._registry_read
        )
        return reader(filename)

    @document_io_formats(registry=_registry_write)
    def write(self, filename, format=None, overwrite=False, **kwargs):
        """Write flux component to file

        Parameters
        ----------
        filename : str or `Path`
            Output filename
        overwrite : bool
            Overwrite file.
        format : {formats}
            Format to use.
        """
        writer = get_writer(
            filename=filename, format=format, registry=self._registry_write
        )

        return writer(
            flux_component=self, filename=filename, overwrite=overwrite, **kwargs
        )

    def plot(self, ax=None, kwargs_norm=None, **kwargs):
        """Plot flux component as sky image

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Plotting axes
        kwargs_norm: dict
            Keyword arguments passed to `~astropy.visualization.simple_norm`
        **kwargs : dict
            Keywords forwarded to `~matplotlib.pyplot.imshow`

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
            Plotting axes
        """
        if ax is None:
            ax = plt.subplot(projection=self.wcs)

        kwargs_norm = kwargs_norm or {"min_cut": 0, "stretch": "asinh", "asinh_a": 0.01}

        flux = self.flux_upsampled_numpy

        norm = simple_norm(flux, **kwargs_norm)

        kwargs.setdefault("norm", norm)
        kwargs.setdefault("interpolation", "None")
        ax.imshow(flux, origin="lower", **kwargs)
        return ax

    def as_gp_map(self):
        """Convert to Gammapy map

        Returns
        -------
        map : `~gammapy.maps.WcsNDmap`
            Gammapy WCS map
        """
        from gammapy.maps import Map, WcsGeom

        geom = WcsGeom(wcs=self.wcs, npix=self.shape_image)
        return Map.from_geom(geom=geom, data=self.flux_numpy)


class FluxComponents(nn.ModuleDict):
    """Flux components"""

    _registry_read = IO_FORMATS_FLUX_COMPONENTS_READ
    _registry_write = IO_FORMATS_FLUX_COMPONENTS_WRITE

    def parameters(self):
        """Parameter list"""
        parameters = []

        for component in self.values():
            if not component.frozen:
                parameters.extend(component.parameters())

        return parameters

    @property
    def priors(self):
        """Priors associated with the componenet"""
        priors = Priors()

        for name, component in self.items():
            priors[name] = component.prior

        return priors

    @property
    def flux_upsampled_total(self):
        """Total summed flux (`~torch.tensor`)"""
        values = list(self.values())

        flux = torch.zeros(values[0].shape)

        for component in values:
            flux += component.flux_upsampled

        return flux

    @property
    def fluxes_numpy(self):
        """Fluxes (`dict` of `~numpy.ndarray`)"""
        fluxes = {}

        for name, component in self.items():
            fluxes[name] = component.flux_numpy

        return fluxes

    @property
    def fluxes_upsampled_numpy(self):
        """Upsampled fluxes (`dict` of `~numpy.ndarray`)"""
        return self.to_numpy()

    @property
    def flux_upsampled_total_numpy(self):
        """Usampled total flux"""
        return np.sum([flux for flux in self.fluxes_upsampled_numpy.values()], axis=0)

    @property
    def flux_total_numpy(self):
        """Usampled total flux"""
        return np.sum([flux for flux in self.fluxes_numpy.values()], axis=0)

    def to_dict(self, include_data=None):
        """Convert flux component configuration to dict, with simple data types.

        Parameters
        ----------
        include_data : None or {"numpy"}
            Optionally include data array in the given format

        Returns
        -------
        data : dict
            Parameter dict.
        """
        fluxes = {}

        for name, component in self.items():
            fluxes[name] = component.to_dict(include_data=include_data)

        return fluxes

    @classmethod
    def from_dict(cls, data):
        """Create flux components from dict

        Parameters
        ----------
        data : dict
            Parameter dict.

        Returns
        -------
        flux_components : `FluxComponents`
            Flux components
        """
        components = []

        for name, component_data in data.items():
            component = SpatialFluxComponent.from_dict(data=component_data)
            components.append((name, component))

        return cls(components)

    def to_numpy(self):
        """Fluxes of the components (dict of `~numpy.ndarray`)"""
        fluxes = {}

        for name, component in self.items():
            flux_cpu = component.flux_upsampled.detach().cpu()
            fluxes[name] = np.squeeze(flux_cpu.numpy())

        return fluxes

    def to_flux_tuple(self):
        """Fluxes as tuple (tuple of `~torch.tensor`)"""
        return tuple([_.flux_upsampled for _ in self.values()])

    def set_flux_errors(self, flux_errors):
        """Set flux errors"""
        for name, flux_error in flux_errors.items():
            self[name]._flux_upsampled_error = flux_error

    @classmethod
    @document_io_formats(registry=_registry_read)
    def read(cls, filename, format=None):
        """Read flux components from file

        Parameters
        ----------
        filename : str or `Path`
            Output filename
        format : {formats}
            Format to use.

        Returns
        -------
        flux_components : `FluxComponents`
            Flux components
        """
        reader = get_reader(
            filename=filename, format=format, registry=cls._registry_read
        )
        return reader(filename=filename)

    @document_io_formats(registry=_registry_write)
    def write(self, filename, overwrite=False, format=None, **kwargs):
        """Write flux components to file

        Parameters
        ----------
        filename : str or `Path`
            Output filename
        overwrite : bool
            Overwrite file.
        format : {formats}
            Format to use.
        """
        writer = get_writer(
            filename=filename, format=format, registry=self._registry_write
        )
        return writer(
            flux_components=self, filename=filename, overwrite=overwrite, **kwargs
        )

    def plot(self, figsize=None, kwargs_norm=None, **kwargs):
        """Plot images of the flux components

        Parameters
        ----------
        fisize : tuple of int
            Figure size.
        kwargs_norm: dict
            Keyword arguments passed to `~astropy.visualization.simple_norm`
        **kwargs : dict
            Keywords forwarded to `~matplotlib.pyplot.imshow`

        Returns
        -------
        axes : list of `~matplotlib.pyplot.Axes`
            Plot axes
        """
        ncols = len(self) + 1

        if figsize is None:
            figsize = (ncols * 5, 5)

        fig, axes = plt.subplots(
            nrows=1,
            ncols=ncols,
            subplot_kw={"projection": list(self.values())[0].wcs},
            figsize=figsize,
        )

        kwargs_norm = kwargs_norm or {"min_cut": 0, "stretch": "asinh", "asinh_a": 0.01}

        flux = self.flux_total_numpy

        norm = simple_norm(flux, **kwargs_norm)
        im = axes[0].imshow(flux, origin="lower", norm=norm, **kwargs)

        axes[0].set_title("Total")

        for ax, name in zip(axes[1:], self.fluxes_numpy):
            component = self[name]
            component.plot(ax=ax, kwargs_norm=kwargs_norm, **kwargs)
            ax.set_title(name.title())

        add_cbar(im=im, ax=ax, fig=fig)
        return axes

    def __str__(self):
        return format_class_str(instance=self)
