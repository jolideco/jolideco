import logging
from pathlib import Path
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.utils import lazyproperty
from astropy.visualization import simple_norm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from jolideco.priors.core import Prior, Priors, UniformPrior
from .utils.io import (
    IO_FORMATS_FLUX_COMPONENT_READ,
    IO_FORMATS_FLUX_COMPONENT_WRITE,
    IO_FORMATS_FLUX_COMPONENTS_READ,
    IO_FORMATS_FLUX_COMPONENTS_WRITE,
    document_io_formats,
    get_reader,
    get_writer,
)
from .utils.misc import format_class_str
from .utils.plot import add_cbar
from .utils.torch import convolve_fft_torch, grid_weights, transpose

log = logging.getLogger(__name__)


__all__ = [
    "FluxComponent",
    "FluxComponents",
    "SparseFluxComponent",
    "NPredModel",
    "NPredModels",
]


class SparseFluxComponent(nn.Module):
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

    upsampling_factor = 1

    _shape_eval = (-1, 1, 1, 1, 1)
    _shape_eval_x = (1, 1, 1, 1, -1)
    _shape_eval_y = (1, 1, 1, -1, 1)

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
        self._flux = nn.Parameter(flux)
        self.x_pos = nn.Parameter(x_pos)
        self.y_pos = nn.Parameter(y_pos)
        self._use_log_flux = use_log_flux

    def parameters(self, recurse=True):
        """Parameter list"""
        if self.frozen:
            return []
        else:
            return super().parameters(recurse)

    @property
    def x_pos_numpy(self):
        """x pos as numpy array"""
        return self.x_pos.detach().cpu().numpy()

    @property
    def y_pos_numpy(self):
        """y pos as numpy array"""
        return self.y_pos.detach().cpu().numpy()

    @property
    def sky_coord(self):
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
            Keyword arguments forwared to `SparseFluxComponent`

        Returns
        -------
        sparse_flux_component : `SparseFluxComponent`
            Sparse flux component
        """
        flux = torch.from_numpy(flux)
        x_pos = torch.from_numpy(x_pos)
        y_pos = torch.from_numpy(y_pos)

        return cls(flux=flux, x_pos=x_pos, y_pos=y_pos, **kwargs)

    @classmethod
    def from_sky_coord(cls, skycoord, wcs, **kwargs):
        """Create sparse flux component from sky coordinates

        Parameters
        ----------
        skycoord: `~astropy.coodinates.SkyCoord`
            Sky coordinates
        wcs : `~astropy.wcs.WCS`
            World coordinate transform object

        Returns
        -------
        sparse_flux_component : `SparseFluxComponent`
            Sparse flux component
        """
        x_pos, y_pos = skycoord.to_pixel(wcs=wcs)
        return cls.from_numpy(x_pos=x_pos, y_pos=y_pos, **kwargs)

    @property
    def wcs(self):
        """Flux error"""
        return self._wcs

    @property
    def shape(self):
        """Shape of the flux component"""
        return (1, 1) + self._shape

    @lazyproperty
    def indices(self):
        """Shape of the flux component"""
        idx = torch.arange(self._shape[1])
        idy = torch.arange(self._shape[0])
        return idx.reshape(self._shape_eval_x), idy.reshape(self._shape_eval_y)

    @property
    def use_log_flux(self):
        """Use log flux (`bool`)"""
        return self._use_log_flux

    @property
    def flux_numpy(self):
        """Flux (`~numpy.ndarray`)"""
        flux_cpu = self.flux.detach().cpu()
        return flux_cpu.numpy()[0, 0]

    @property
    def flux(self):
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
    def flux_upsampled(self):
        """Upsampled flux"""
        return self.flux

    def plot(self, ax=None, **kwargs):
        """Plot flux component as sky image

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Plotting axes
        **kwargs : dict
            Keywords forwarded to `~matplotlib.pyplot.imshow`

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
            Plotting axes
        """
        if ax is None:
            ax = plt.subplot(projection=self.wcs)

        kwargs.setdefault("interpolation", "None")

        flux = self.flux_numpy
        _ = ax.imshow(flux, origin="lower", **kwargs)
        return ax

    def to_dict(self, **kwargs):
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


class NPredCalibration(nn.Module):
    """Dataset position calibration

    Attributes
    ----------
    shift_x : `~torch.Tensor`
        Shift in x direction
    shift_y: `~torch.Tensor`
        Shift in y direction
    grid_sample_kwargs : dict
        Keyword arguments passed to `~torch.nn.functional.grid_sample`

    """

    def __init__(self, shift_x=0, shift_y=0, grid_sample_kwargs=None):
        super().__init__()
        self.shifts = nn.Parameter(torch.Tensor([[shift_x, shift_y]]))
        self.grid_sample_kwargs = grid_sample_kwargs or {}

    def __call__(self, flux, scale):
        """Apply affine transform to calibrate position.

        Parameters
        ----------
        flux : `~torch.Tensor`
            Flux tensor
        scale : float
            Upsampling factor scale.

        Returns
        -------
        flux : `~torch.Tensor`
            Flux tensor
        """
        size = flux.size()

        diag = torch.eye(2)
        theta = torch.cat([diag, self.shifts.T], dim=1)[None]

        grid = F.affine_grid(theta=theta, size=size)
        flux_shift = F.grid_sample(flux, grid=grid, **self.grid_sample_kwargs)
        return flux_shift


class FluxComponent(nn.Module):
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

    _registry_read = IO_FORMATS_FLUX_COMPONENT_READ
    _registry_write = IO_FORMATS_FLUX_COMPONENT_WRITE

    def __init__(
        self,
        flux_upsampled,
        flux_upsampled_error=None,
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
        self._use_log_flux = use_log_flux
        self.upsampling_factor = int(upsampling_factor)

        if prior is None:
            prior = UniformPrior()

        self.prior = prior
        self.frozen = frozen
        self._wcs = wcs

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
    def wcs(self):
        """Flux error"""
        return self._wcs

    def parameters(self, recurse=True):
        """Parameter list"""
        if self.frozen:
            return []
        else:
            return super().parameters(recurse)

    @classmethod
    def from_numpy(cls, flux, **kwargs):
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

        return cls(flux_upsampled=flux, **kwargs)

    @classmethod
    def from_flux_init_datasets(cls, datasets, **kwargs):
        """Compute flux init from datasets by averaging over the raw flux estimate.

        Parameters
        ----------
        datasets : list of dict
            List of dictionaries containing, "counts", "psf", "background" and "exposure".
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
    def shape(self):
        """Shape of the flux component"""
        return self._flux_upsampled.shape

    @property
    def use_log_flux(self):
        """Use log flux (`bool`)"""
        return self._use_log_flux

    @property
    def flux_upsampled(self):
        """Flux (`~torch.Tensor`)"""
        if self.use_log_flux:
            return torch.exp(self._flux_upsampled)
        else:
            return self._flux_upsampled

    @property
    def flux(self):
        """Flux (`~torch.Tensor`)"""
        flux = self.flux_upsampled

        if self.upsampling_factor:
            flux = F.avg_pool2d(
                flux,
                kernel_size=self.upsampling_factor,
                divisor_override=1,
            )
        return flux

    @property
    def flux_upsampled_error(self):
        """Flux error (`~torch.Tensor`)"""
        return self._flux_upsampled_error

    @property
    def flux_numpy(self):
        """Flux (`~numpy.ndarray`)"""
        flux_cpu = self.flux.detach().cpu()
        return flux_cpu.numpy()[0, 0]

    @property
    def flux_upsampled_numpy(self):
        """Flux (`~numpy.ndarray`)"""
        flux_cpu = self.flux_upsampled.detach().cpu()
        return flux_cpu.numpy()[0, 0]

    @property
    def flux_upsampled_error_numpy(self):
        """Flux error (`~numpy.ndarray`)"""
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
        """Write flux component fo file

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

    def plot(self, ax=None, **kwargs):
        """Plot flux component as sky image

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Plotting axes
        **kwargs : dict
            Keywords forwarded to `~matplotlib.pyplot.imshow`

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
            Plotting axes
        """
        if ax is None:
            ax = plt.subplot(projection=self.wcs)

        flux = self.flux_upsampled_numpy
        _ = ax.imshow(flux, origin="lower", **kwargs)
        return ax


class FluxComponents(nn.ModuleDict):
    """Flux components"""

    _registry_read = IO_FORMATS_FLUX_COMPONENTS_READ
    _registry_write = IO_FORMATS_FLUX_COMPONENTS_WRITE

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
            component = FluxComponent.from_dict(data=component_data)
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

    def plot(self, figsize=None, **kwargs):
        """Plot images of the flux components

        Parameters
        ----------
        **kwargs : dict
            Keywords forwared to `~matplotlib.pyplot.imshow`

        Returns
        -------
        axes : list of `~matplotlib.pyplot.Axes`
            Plot axes
        """
        ncols = len(self) + 1

        if figsize is None:
            figsize = (ncols * 5, 5)

        norm = simple_norm(
            self.flux_upsampled_total_numpy, min_cut=0, stretch="asinh", asinh_a=0.01
        )

        kwargs.setdefault("norm", norm)

        fig, axes = plt.subplots(
            nrows=1,
            ncols=ncols,
            subplot_kw={"projection": list(self.values())[0].wcs},
            figsize=figsize,
        )

        im = axes[0].imshow(self.flux_upsampled_total_numpy, origin="lower", **kwargs)
        axes[0].set_title("Total")

        for ax, name in zip(axes[1:], self.fluxes_upsampled_numpy):
            component = self[name]
            component.plot(ax=ax, **kwargs)
            ax.set_title(name.title())

        add_cbar(im=im, ax=ax, fig=fig)
        return axes

    def __str__(self):
        return format_class_str(instance=self)


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
    upsampling_factor : int
            Upsampling factor.
    """

    def __init__(
        self, background, exposure, psf=None, rmf=None, upsampling_factor=None
    ):
        super().__init__()
        self.register_buffer("background", background)
        self.register_buffer("exposure", exposure)
        self.register_buffer("psf", psf)
        self.register_buffer("rmf", rmf)
        self.upsampling_factor = upsampling_factor

    @property
    def shape_upsampled(self):
        """Shape of the NPred model"""
        return tuple(self.background.shape)

    @property
    def shape(self):
        """Shape of the NPred model"""
        shape = list(self.shape_upsampled)
        shape[-1] //= self.upsampling_factor
        shape[-2] //= self.upsampling_factor
        return tuple(shape)

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

        kwargs = {
            "upsampling_factor": upsampling_factor,
        }

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
        npred = (flux + self.background) * self.exposure

        if self.psf is not None:
            npred = convolve_fft_torch(npred, self.psf)

        if self.upsampling_factor:
            npred = F.avg_pool2d(
                npred, kernel_size=self.upsampling_factor, divisor_override=1
            )

        if self.rmf is not None:
            npred_T = transpose(npred[0])
            npred = torch.matmul(npred_T, self.rmf)
            npred = transpose(npred)[None]

        return torch.clip(npred, 0, torch.inf)


class NPredModels(nn.ModuleDict):
    """Flux components

    Parameters
    ----------
    calibration : `NPredCalibration`
        Calibration model.
    """

    def __init__(self, calibration=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calibration = calibration

    def evaluate(self, fluxes):
        """Evaluate npred model

        Parameters
        ----------
        fluxes : tuple of  `~torch.tensor`
            Flux components

        Returns
        -------
        npred_total : `~torch.tensor`
            Predicted counts tensort
        """
        values = list(self.values())

        npred_total = torch.zeros(values[0].shape, device=fluxes[0].device)

        for npred_model, flux in zip(values, fluxes):
            if self.calibration is not None:
                flux = self.calibration(flux=flux, scale=npred_model.upsampling_factor)

            npred = npred_model(flux=flux)
            npred_total += npred

        return npred_total

    @classmethod
    def from_dataset_numpy(cls, dataset, components):
        """Create multiple npred models.

        Parameters
        ----------
        dataset : dict of `~torch.tensor`
            Dataset
        components : `FluxComponents`
            Flux components

        Returns
        -------
        npred_models : `NPredModel`
            NPredModels
        """
        values = []

        calibration = dataset.get("calibration", None)

        for name, component in components.items():
            npred_model = NPredModel.from_dataset_numpy(
                dataset=dataset, upsampling_factor=component.upsampling_factor
            )
            values.append((name, npred_model))

        return cls(calibration, values)
