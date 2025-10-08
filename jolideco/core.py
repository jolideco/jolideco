import copy
import logging
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.table import Table
from astropy.utils import lazyproperty
from astropy.visualization import simple_norm
from tqdm.auto import tqdm

from .loss import TotalLoss
from .models import FluxComponents, SpatialFluxComponent
from .utils.io import (
    IO_FORMATS_MAP_RESULT_READ,
    IO_FORMATS_MAP_RESULT_WRITE,
    get_reader,
    get_writer,
)
from .utils.misc import format_class_str
from .utils.plot import add_cbar
from .utils.torch import TORCH_DEFAULT_DEVICE

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)

__all__ = ["MAPDeconvolver", "MAPDeconvolverResult"]


def optimizer_to_dict(optim):
    """Create dict seialization of an optimizer instance"""
    lookup = {value: key for key, value in OPTIMIZER.items()}
    return lookup[optim.__class__]


OPTIMIZER = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}

COMPILE_MODEL = (platform.system() != "Windows")

class MAPDeconvolver:
    """Maximum A-Posteriori deconvolver

    Attributes
    ----------
    n_epochs : int
        Number of epochs to train
    beta : float
        Scale factor for the prior.
    learning_rate : float
        Learning rate
    compute_error : bool
        Whether to compute flux error
    stop_early : bool
        Stop training early, once the average results on the last n test
        datasets do not improve any more.
    stop_early_n_average: int
        Number of iterations to avergae over.
    device : `~pytorch.Device`
        Pytorch device
    display_progress : bool
        Whether to display a progress bar
    optimizer_type : {"adam", "sgd"}
        Optimizer to use
    optimizer_kwargs: dict
        Optimizer keywords
    checkpoint_path : str
        Path to save checkpoints
    """

    _default_flux_component = "flux"
    _default_checkpoint_filename = "checkpoint-epoch-{epoch}.asdf"

    def __init__(
        self,
        n_epochs=1_000,
        beta=1,
        learning_rate=0.1,
        compute_error=False,
        stop_early=False,
        stop_early_n_average=10,
        device=TORCH_DEFAULT_DEVICE,
        display_progress=True,
        optimizer_type="adam",
        optimizer_kwargs=None,
        checkpoint_path=None,
    ):
        self.n_epochs = n_epochs
        self.beta = beta
        self.learning_rate = learning_rate
        self.compute_error = compute_error
        self.stop_early = stop_early
        self.stop_early_n_average = stop_early_n_average
        self.display_progress = display_progress

        if "cuda" in device and not torch.cuda.is_available():
            log.warning(
                f"Device {device} not available, falling back to {TORCH_DEFAULT_DEVICE}"
            )
            device = TORCH_DEFAULT_DEVICE

        self.device = torch.device(device)

        if optimizer_type not in OPTIMIZER:
            raise ValueError(
                f"Unknown optimizer: {optimizer_type}, must be one of {OPTIMIZER}"
            )
            
        
        self.optimizer_type = optimizer_type

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_kwargs.setdefault("lr", self.learning_rate)
        
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.mkdir(exist_ok=True, parents=True)

        self.checkpoint_path = checkpoint_path

    def to_dict(self):
        """Convert deconvolver configuration to dict, with simple data types.

        Returns
        -------
        data : dict
            Parameter dict.
        """
        data = {}
        data.update(self.__dict__)
        data["device"] = str(self.device)
        data["checkpoint_path"] = str(self.checkpoint_path)
        data.pop("optimizer", None)
        data.pop("optimizer_kwargs", None)
        return data

    def __str__(self):
        """String representation"""
        return format_class_str(instance=self)

    def run(
        self, datasets, datasets_validation=None, components=None, calibrations=None
    ):
        """Run the MAP deconvolver

        Parameters
        ----------
        datasets : dict of [str, dict]
            Dictionary containing a name of the dataset as key and a dictionary
            containing, the data like "counts", "psf", "background" and "exposure".
        datasets_validation : dict of [str, dict]
            Dictionary containing a name of the validation dataset as key and a
            dictionary containing, the data like "counts", "psf", "background"
            and "exposure".
        components : `FluxComponents` or `FluxComponent`
            Flux components.
        calibrations : `NPredCalibrations`
            Optional model calibrations.

        Returns
        -------
        flux : `~numpy.ndarray`
            Reconstructed flux.
        """
        if self.stop_early and datasets_validation is None:
            raise ValueError("Early stopping requires providing test datasets")

        if isinstance(components, SpatialFluxComponent):
            components = {self._default_flux_component: components}

        components = FluxComponents(components)
        components_init = copy.deepcopy(components)
        calibrations_init = copy.deepcopy(calibrations)

        if COMPILE_MODEL:
            components = torch.compile(components)

        components = components.to(self.device)

        total_loss = TotalLoss.from_datasets_and_components(
            datasets=datasets,
            datasets_validation=datasets_validation,
            components=components,
            calibrations=calibrations,
            beta=self.beta,
            device=self.device,
        )

        parameters = list(components.parameters())

        if calibrations:
            parameters.extend(calibrations.parameters())

        self.optimizer = OPTIMIZER[self.optimizer_type](
            params=parameters, **self.optimizer_kwargs
        )

        disable = not self.display_progress

        with tqdm(total=self.n_epochs * len(datasets), disable=disable) as pbar:
            for epoch in range(self.n_epochs):
                pbar.set_description(f"Epoch {epoch + 1}")

                components.train()

                for counts, npred_model in total_loss.poisson_loss.iter_by_dataset:
                    self.optimizer.zero_grad()
                    # evaluate npred model
                    fluxes = components.to_flux_tuple()
                    npred = npred_model.evaluate(fluxes=fluxes)

                    # compute Poisson loss
                    loss = total_loss.poisson_loss.loss_function(npred, counts)

                    # compute prior losses
                    loss_prior = total_loss.prior_loss(fluxes=fluxes)

                    loss_total = loss - self.beta * loss_prior / total_loss.prior_weight

                    loss_total.backward()
                    self.optimizer.step()
                    pbar.update(1)

                components.eval()

                if self.checkpoint_path:
                    filename = self._default_checkpoint_filename.format(epoch=epoch)
                    checkpoint = MAPDeconvolverResult(
                        config=self.to_dict(),
                        trace_loss=total_loss.trace,
                        components=components,
                        calibrations=calibrations,
                    )
                    log.info(f"Writing checkpoint to {self.checkpoint_path / filename}")
                    checkpoint.write(filename=self.checkpoint_path / filename)
                else:
                    filename = ""

                total_loss.append_trace(fluxes=fluxes, filename=filename)

                row = total_loss.trace[-1]

                if (
                    self.stop_early
                    and len(total_loss.trace) > self.stop_early_n_average
                ):
                    range_mean = slice(-self.stop_early_n_average, None)
                    trace_loss_validation = total_loss.trace[
                        "datasets-validation-total"
                    ]
                    loss_test_average = np.mean(trace_loss_validation[range_mean])
                    if row["datasets-validation-total"] > loss_test_average:
                        break

                pbar.set_postfix(
                    total=row["total"],
                    datasets_total=row["datasets-total"],
                    priors_total=row["priors-total"],
                )

        if self.compute_error:
            flux_errors = total_loss.fluxes_error(fluxes=fluxes)
            components.set_flux_errors(flux_errors=flux_errors)

        config = self.to_dict()
        return MAPDeconvolverResult(
            config=config,
            components=components._orig_mod,
            components_init=components_init,
            trace_loss=total_loss.trace,
            calibrations=calibrations,
            calibrations_init=calibrations_init,
            wcs=None,
        )


class MAPDeconvolverResult:
    """MAP deconvolver result

    Parameters
    ----------
    config : `dict`
        Configuration from the `LIRADeconvolver`
    components: `FluxComponents`
        Flux components.
    trace_loss : `~astropy.table.Table` or dict
        Trace of the total loss.
    components_init : `FluxComponents`
        Initial flux components.
    calibrations : `NPredCalibrations`
        Optional model calibrations.
    calibrations_init : `NPredCalibrations`
        Initial model calibrations.
    wcs : `astropy.wcs.WCS`
        World coordinate system.
    """

    def __init__(
        self,
        config,
        components,
        trace_loss,
        components_init=None,
        calibrations=None,
        calibrations_init=None,
        wcs=None,
    ):
        self._components = components
        self._components_init = components_init
        self.trace_loss = trace_loss
        self._calibrations = calibrations
        self._calibrations_init = calibrations_init
        self._config = config
        self._wcs = wcs

    @property
    def checkpoint_path(self):
        """Path to checkpoints"""
        return Path(self.config.get("checkpoint_path", None))

    def read_checkpoint(self, epoch):
        """Read checkpoint

        Parameters
        ----------
        epoch : int
            Epoch number

        Returns
        -------
        checkpoint : `MAPDeconvolverResult`
            Checkpoint object
        """
        filename = self.checkpoint_path / self.trace_loss["filename"][epoch]
        return self.__class__.read(filename=filename)

    @property
    def components(self):
        """Flux components (`FluxComponents`)"""
        return self._components

    @property
    def components_init(self):
        """Initial flux components (`FluxComponents`)"""
        return self._components_init

    @property
    def calibrations(self):
        """Calibrations (`NPredCalibrations`)"""
        return self._calibrations

    @property
    def calibrations_init(self):
        """Initial calibrations (`NPredCalibrations`)"""
        return self._calibrations_init

    @property
    def flux_total(self):
        """Total flux"""
        return self.components.flux_total_numpy

    @property
    def flux_upsampled_total(self):
        """Total flux"""
        return self.components.flux_upsampled_total_numpy

    @lazyproperty
    def config(self):
        """Configuration data (`dict`)"""
        return self._config

    def plot_trace_loss(self, ax=None, which=None, **kwargs):
        """Plot trace loss

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Plot axes
        which : list of str
            Which traces to plot.

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
            Plot axes
        """
        from .utils.plot import plot_trace_loss

        ax = plt.gca() if ax is None else ax

        plot_trace_loss(ax=ax, trace_loss=self.trace_loss, which=which, **kwargs)
        return ax

    def peek(self, figsize=(12, 5), kwargs_norm=None):
        """Plot the result and the trace of the loss function

        Parameters
        ----------
        figsize : tuple
            Figure size
        """
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        self.plot_trace_loss(ax=axes[0])

        kwargs_norm = kwargs_norm or {"vmin": 0, "stretch": "asinh", "asinh_a": 0.01}

        flux = self.components.flux_total_numpy

        norm = simple_norm(flux, **kwargs_norm)

        kwargs = {"norm": norm, "interpolation": "None"}

        im = axes[1].imshow(flux, origin="lower", **kwargs)
        add_cbar(im=im, ax=axes[1], fig=fig)

    @property
    def config_table(self):
        """Configuration data as table (`~astropy.table.Table`)"""
        config = Table()

        for key, value in self.config.items():
            config[key] = [value]

        return config

    def write(self, filename, overwrite=False, format=None):
        """Write result to file

        Parameters
        ----------
        filename : str or `Path`
            Output filename
        overwrite : bool
            Overwrite file.
        format : {"fits", "asdf"}
            Format to use.
        """
        writer = get_writer(
            filename=filename, format=format, registry=IO_FORMATS_MAP_RESULT_WRITE
        )
        writer(result=self, filename=filename, overwrite=overwrite)

    @classmethod
    def read(cls, filename, format=None):
        """Write result to file

        Parameters
        ----------
        filename : str or `Path`
            Output filename
        format : {"fits", "asdf"}
            Format to use.

        Returns
        -------
        result : `~MAPDeconvolverResult`
            Result object
        """
        reader = get_reader(
            filename=filename, format=format, registry=IO_FORMATS_MAP_RESULT_READ
        )
        return reader(filename=filename)
