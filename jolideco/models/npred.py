import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jolideco.utils.io import (
    IO_FORMATS_NPRED_CALIBRATIONS_READ,
    IO_FORMATS_NPRED_CALIBRATIONS_WRITE,
    document_io_formats,
    get_reader,
    get_writer,
)
from jolideco.utils.misc import format_class_str
from jolideco.utils.torch import convolve_fft_torch, transpose

__all__ = [
    "NPredModel",
    "NPredModels",
    "NPredCalibration",
    "NPredCalibrations",
]


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
    def from_numpy(
        cls, background, exposure, psf, upsampling_factor, correct_exposure_edges=True
    ):
        """Create NPred model from numpy arrays

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
        upsampling_factor : int
                Upsampling factor.
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
            "background": torch.from_numpy(background[dims]),
            "exposure": torch.from_numpy(exposure[dims]),
            "psf": torch.from_numpy(psf[dims]),
        }

        for name in ["psf", "exposure", "background"]:
            tensor = kwargs[name]
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

    @classmethod
    def from_dataset_numpy(
        cls,
        dataset,
        upsampling_factor=None,
        correct_exposure_edges=True,
    ):
        """Create NPred model from dataset

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
        return cls.from_numpy(
            background=dataset["background"],
            exposure=dataset["exposure"],
            psf=dataset["psf"],
            upsampling_factor=upsampling_factor,
            correct_exposure_edges=correct_exposure_edges,
        )

    def forward(self, flux, background_norm=1):
        """Forward folding model evaluation.

        Parameters
        ----------
        flux : `~torch.Tensor`
            Flux tensor
        background_norm : `~torch.Tensor`
            Background norm

        Returns
        -------
        npred : `~torch.Tensor`
            Predicted number of counts
        """
        npred = (flux + self.background * background_norm) * self.exposure

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
                background_norm = self.calibration.background_norm
            else:
                background_norm = 1.0

            npred = npred_model(flux=flux, background_norm=background_norm)
            npred_total += npred

        return npred_total

    @classmethod
    def from_dataset_numpy(cls, dataset, components, calibration=None):
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

        for name, component in components.items():
            # TODO: select component specific PSF and RMF here
            npred_model = NPredModel.from_dataset_numpy(
                dataset=dataset, upsampling_factor=component.upsampling_factor
            )
            values.append((name, npred_model))

        return cls(calibration, values)


class NPredCalibration(nn.Module):
    """Dataset calibration parameters

    Attributes
    ----------
    shift_x : `~torch.Tensor`
        Shift in x direction
    shift_y: `~torch.Tensor`
        Shift in y direction
    background_norm: `~torch.Tensor`
        Background normalisation parameter
    frozen : bool
        Whether to freeze component.
    """

    _grid_sample_kwargs = {"align_corners": False}

    def __init__(
        self,
        shift_x=0.0,
        shift_y=0.0,
        background_norm=1.0,
        frozen=False,
    ):
        super().__init__()
        self.shift_xy = nn.Parameter(torch.Tensor([[shift_x, shift_y]]))
        value = torch.log(torch.Tensor([background_norm]))
        self._background_norm = nn.Parameter(value)
        self.frozen = frozen

    @property
    def background_norm(self):
        """Background norm"""
        return torch.exp(self._background_norm)

    def parameters(self, recurse=True):
        """Parameter list"""
        if self.frozen:
            return []
        else:
            return super().parameters(recurse)

    def to_dict(self):
        """Convert calibration model to dict, with simple data types.

        Returns
        -------
        data : dict
            Parameter dict.
        """
        data = {}
        shift_xy = self.shift_xy.detach().cpu().numpy()
        data["shift_x"] = float(shift_xy[0, 0])
        data["shift_y"] = float(shift_xy[0, 1])
        data["background_norm"] = float(self.background_norm.detach().cpu().numpy())
        data["frozen"] = self.frozen
        return data

    @classmethod
    def from_dict(cls, data):
        """Create calibration model from dict

        Parameters
        ----------
        data : dict
            Parameter dict.

        Returns
        -------
        calibration : `NPredCalibration`
            Calibration model.
        """
        return cls(**data)

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

        scale = 2 * scale / torch.Tensor([[size[-1]], [size[-2]]])

        diag = torch.eye(2)
        theta = torch.cat([diag, scale * self.shift_xy.T], dim=1)[None]

        grid = F.affine_grid(theta=theta, size=size)
        flux_shift = F.grid_sample(flux, grid=grid, **self._grid_sample_kwargs)
        return flux_shift

    def __str__(self):
        """String representation"""
        return format_class_str(instance=self)


class NPredCalibrations(nn.ModuleDict):
    """Calibration components

    Parameters
    ----------
    calibration : `NPredCalibration`
        Calibration model.
    """

    _registry_read = IO_FORMATS_NPRED_CALIBRATIONS_READ
    _registry_write = IO_FORMATS_NPRED_CALIBRATIONS_WRITE

    def parameters(self, recurse=True):
        """Parameter list"""
        parameters = []

        for model in self.values():
            if not model.frozen:
                pars = list(model.parameters())
                parameters.extend(pars)

        return parameters

    def to_dict(self):
        """Convert calibration configuration to dict, with simple data types.

        Returns
        -------
        data : dict
            Parameter dict.
        """
        data = {}

        for name, model in self.items():
            data[name] = model.to_dict()

        return data

    @classmethod
    def from_dict(cls, data):
        """Create calibration models from dict

        Parameters
        ----------
        data : dict
            Parameter dict.

        Returns
        -------
        calibrations : `NPredCalibrations`
            Calibrations
        """
        components = []

        for name, component_data in data.items():
            component = NPredCalibration.from_dict(data=component_data)
            components.append((name, component))

        return cls(components)

    @classmethod
    @document_io_formats(registry=_registry_read)
    def read(cls, filename, format=None):
        """Read npred calibrations from file

        Parameters
        ----------
        filename : str or `Path`
            Output filename
        format : {formats}
            Format to use.

        Returns
        -------
        calibrations : `NPredCalibrations`
            Calibrations
        """
        reader = get_reader(
            filename=filename, format=format, registry=cls._registry_read
        )
        return reader(filename)

    @document_io_formats(registry=_registry_write)
    def write(self, filename, format=None, overwrite=False, **kwargs):
        """Write npred calibrations fo file

        Parameters
        ----------
        filename : str or `Path`
            Output filename
        format : {formats}
            Format to use.
        overwrite : bool
            Overwrite file.
        """
        writer = get_writer(
            filename=filename, format=format, registry=self._registry_write
        )

        return writer(
            npred_calibrations=self, filename=filename, overwrite=overwrite, **kwargs
        )

    def __str__(self):
        """String representation"""
        return format_class_str(instance=self)
