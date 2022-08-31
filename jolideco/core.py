import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.table import Table
from astropy.utils import lazyproperty

from .loss import PoissonLoss, PriorLoss, TotalLoss
from .models import FluxComponent, FluxComponents
from .priors import PRIOR_REGISTRY, Priors, UniformPrior
from .utils.io import IO_FORMATS_READ, IO_FORMATS_WRITE
from .utils.torch import TORCH_DEFAULT_DEVICE

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)


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
    fit_background_norm : bool
        Whether to fit background norm.
    stop_early : bool
        Stop training early, once the average results on the last n test
        datasets do not improve any more.
    stop_early_n_average: int
        Number of iterations to avergae over.
    device : `~pytorch.Device`
        Pytorch device
    """

    _default_flux_component = "flux"

    def __init__(
        self,
        n_epochs,
        beta=1,
        learning_rate=0.1,
        compute_error=False,
        fit_background_norm=False,
        stop_early=False,
        stop_early_n_average=10,
        device=TORCH_DEFAULT_DEVICE,
    ):
        self.n_epochs = n_epochs
        self.beta = beta
        self.learning_rate = learning_rate
        self.compute_error = compute_error
        self.fit_background_norm = fit_background_norm
        self.stop_early = stop_early
        self.stop_early_n_average = stop_early_n_average
        self.device = torch.device(device)

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
        return data

    def __str__(self):
        """String representation"""
        cls_name = self.__class__.__name__
        info = cls_name + "\n"
        info += len(cls_name) * "-" + "\n\n"
        data = self.to_dict()

        for key, value in data.items():
            info += f"\t{key:21s}: {value}\n"

        return info.expandtabs(tabsize=4)

    def run(self, datasets, datasets_validation=None, components=None):
        """Run the MAP deconvolver

        Parameters
        ----------
        datasets : list of dict
            List of dictionaries containing, "counts", "psf", "background" and "exposure".
        datasets_validation : list of dict
            List of validation datasets. List of dictionaries containing,
            "counts", "psf", "background" and "exposure".
        components : `FluxComponents` or `FluxComponent`
            Flux components.

        Returns
        -------
        flux : `~numpy.ndarray`
            Reconstructed flux.
        """
        if self.stop_early and datasets_validation is None:
            raise ValueError("Early stopping requires providing test datasets")

        if isinstance(components, FluxComponent):
            components = {self._default_flux_component: components}

        components = FluxComponents(components)

        components_init = copy.deepcopy(components)

        parameters = components.parameters()

        optimizer = torch.optim.Adam(
            params=parameters,
            lr=self.learning_rate,
        )

        poisson_loss = PoissonLoss.from_datasets(
            datasets=datasets, components=components
        )

        if datasets_validation:
            poisson_loss_validation = PoissonLoss.from_datasets(
                datasets=datasets_validation, components=components
            )
        else:
            poisson_loss_validation = None

        prior_loss = PriorLoss(priors=components.priors)

        total_loss = TotalLoss(
            poisson_loss=poisson_loss,
            poisson_loss_validation=poisson_loss_validation,
            prior_loss=prior_loss,
            beta=self.beta,
        )

        for epoch in range(self.n_epochs):
            for counts, npred_model in poisson_loss.iter_by_dataset:
                optimizer.zero_grad()
                # evaluate npred model
                fluxes = components.to_flux_tuple()
                npred = npred_model.evaluate(fluxes=fluxes)

                # compute Poisson loss
                loss = poisson_loss.loss_function(npred, counts)

                # compute prior losses
                loss_prior = prior_loss(fluxes=fluxes)

                loss_total = loss - self.beta * loss_prior / total_loss.prior_weight

                loss_total.backward()
                optimizer.step()

            total_loss.append_trace(fluxes=fluxes)

            row = total_loss.trace[-1]

            if self.stop_early and len(total_loss.trace) > self.stop_early_n_average:
                range_mean = slice(-self.stop_early_n_average, None)
                trace_loss_validation = total_loss.trace["datasets-validation-total"]
                loss_test_average = np.mean(trace_loss_validation[range_mean])
                if row["datasets-validation-total"] > loss_test_average:
                    break

            message = (
                f'Epoch: {epoch}, {row["total"]}, '
                f'{row["datasets-total"]}, {row["priors-total"]}'
            )
            log.debug(message)

        if self.compute_error:
            flux_errors = total_loss.fluxes_error(fluxes=fluxes)
            components.set_flux_errors(flux_errors=flux_errors)

        config = self.to_dict()
        return MAPDeconvolverResult(
            config=config,
            components=components,
            components_init=components_init,
            trace_loss=total_loss.trace,
        )


class MAPDeconvolverResult:
    """MAP deconvolver result

    Parameters
    ----------
    config : `dict`
        Configuration from the `LIRADeconvolver`
    components: `FluxComponents`
        Flux components.
    components_init : `FluxComponents`
        Initial flux components.
    trace_loss : `~astropy.table.Table` or dict
        Trace of the total loss.
    """

    def __init__(self, config, components, components_init, trace_loss, wcs=None):
        self._components = components
        self.components_init = components_init
        self.trace_loss = trace_loss
        self._config = config
        self._wcs = wcs

    @property
    def components(self):
        """Flux components (`FluxComponents`)"""
        return self._components

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

    @property
    def config_table(self):
        """Configuration data as table (`~astropy.table.Table`)"""
        config = Table()

        for key, value in self.config.items():
            config[key] = [value]

        return config

    def write(self, filename, overwrite=False, format="fits"):
        """Write result fo file

        Parameters
        ----------
        filename : str or `Path`
            Output filename
        overwrite : bool
            Overwrite file.
        format : {"fits"}
            Format to use.
        """
        filename = Path(filename)

        if format not in IO_FORMATS_WRITE:
            raise ValueError(
                f"Not a valid format '{format}', choose from {list(IO_FORMATS_WRITE)}"
            )

        writer = IO_FORMATS_WRITE[format]
        writer(result=self, filename=filename, overwrite=overwrite)

    @classmethod
    def read(cls, filename, format="fits"):
        """Write result fo file

        Parameters
        ----------
        filename : str or `Path`
            Output filename
        format : {"fits"}
            Format to use.

        Returns
        -------
        result : `~MAPDeconvolverResult`
            Result object
        """
        filename = Path(filename)

        if format not in IO_FORMATS_READ:
            raise ValueError(
                f"Not a valid format '{format}', choose from {list(IO_FORMATS_READ)}"
            )

        reader = IO_FORMATS_READ[format]
        kwargs = reader(filename=filename)
        return cls(**kwargs)
