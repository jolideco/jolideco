import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jolideco.utils.torch import convolve_fft_torch, transpose

__all__ = [
    "NPredModel",
    "NPredModels",
    "NPredCalibration",
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

            npred = npred_model(
                flux=flux, background_norm=self.calibration.background_norm
            )
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
            # TODO: select component specific PSF and RMF here
            npred_model = NPredModel.from_dataset_numpy(
                dataset=dataset, upsampling_factor=component.upsampling_factor
            )
            values.append((name, npred_model))

        return cls(calibration, values)


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

    def __init__(
        self, shift_x=0, shift_y=0, background_norm=1, grid_sample_kwargs=None
    ):
        super().__init__()
        self.shifts = nn.Parameter(torch.Tensor([[shift_x, shift_y]]))
        self.background_norm = nn.Parameter(torch.Tensor(background_norm))
        self.grid_sample_kwargs = grid_sample_kwargs or {}

    def __call__(self, flux):
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
