import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.nddata import block_reduce
from astropy.table import Table
from astropy.utils import lazyproperty
from astropy.visualization import simple_norm

from .loss import PoissonLoss, PriorLoss, TotalLoss
from .models import FluxComponent, FluxComponents
from .priors import PRIOR_REGISTRY, Priors, UniformPrior
from .utils.io import IO_FORMATS_READ, IO_FORMATS_WRITE
from .utils.plot import add_cbar
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
    loss_function_prior : `~jolideco.priors.Priors`
        Loss functions for the priors (optional).
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
        loss_function_prior=None,
        learning_rate=0.1,
        compute_error=False,
        fit_background_norm=False,
        stop_early=False,
        stop_early_n_average=10,
        device=TORCH_DEFAULT_DEVICE,
    ):
        self.n_epochs = n_epochs
        self.beta = beta

        if loss_function_prior is None:
            loss_function_prior = Priors()
            loss_function_prior[self._default_flux_component] = UniformPrior()

        for prior in loss_function_prior.values():
            prior.to(device=device)

        self.loss_function_prior = loss_function_prior
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

        for key, value in PRIOR_REGISTRY.items():
            if isinstance(self.loss_function_prior, value):
                data["loss_function_prior"] = key

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

        prior_loss = PriorLoss(priors=self.loss_function_prior)

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
                loss_prior = self.loss_function_prior(fluxes=fluxes)

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
            log.info(message)

        if self.compute_error:
            flux_errors = total_loss.fluxes_error(fluxes=fluxes)
            components.set_flux_errors(flux_errors=flux_errors)

        return MAPDeconvolverResult(
            config=self.to_dict(),
            fluxes_upsampled=components.to_numpy(),
            fluxes_init=components_init.to_numpy(),
            trace_loss=total_loss.trace,
        )


class MAPDeconvolverResult:
    """MAP deconvolver result

    Parameters
    ----------
    config : `dict`
        Configuration from the `LIRADeconvolver`
    fluxes_upsampled : `~numpy.ndarray`
        Flux array
    fluxes_init : `~numpy.ndarray`
        Flux init array
    trace_loss : `~astropy.table.Table` or dict
        Trace of the total loss.
    wcs : `~astropy.wcs.WCS`
        World coordinate transform object
    """

    def __init__(self, config, fluxes_upsampled, fluxes_init, trace_loss, wcs=None):
        self._fluxes_upsampled = fluxes_upsampled
        self.fluxes_init = fluxes_init
        self.trace_loss = trace_loss
        self._config = config
        self._wcs = wcs

    @lazyproperty
    def fluxes_upsampled(self):
        """Upsampled fluxes (`dict` of `~numpy.ndarray`)"""
        return self._fluxes_upsampled

    @lazyproperty
    def flux_upsampled_total(self):
        """Usampled total flux"""
        return np.sum([flux for flux in self.fluxes_upsampled.values()], axis=0)

    @lazyproperty
    def fluxes(self):
        """Fluxes (`dict` of `~numpy.ndarray`)"""
        fluxes = {}
        block_size = self._config.get("upsampling_factor", 1)

        for name, flux in self.fluxes_upsampled.items():
            fluxes[name] = block_reduce(flux, block_size=block_size)

        return fluxes

    @lazyproperty
    def flux_total(self):
        """Usampled total flux"""
        return np.sum([flux for flux in self.fluxes.values()], axis=0)

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

    def plot_fluxes(self, figsize=None, **kwargs):
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
        ncols = len(self.fluxes) + 1

        if figsize is None:
            figsize = (ncols * 5, 5)

        norm = simple_norm(
            self.flux_upsampled_total, min_cut=0, stretch="asinh", asinh_a=0.01
        )

        kwargs.setdefault("norm", norm)

        fig, axes = plt.subplots(
            nrows=1,
            ncols=ncols,
            subplot_kw={"projection": self.wcs},
            figsize=figsize,
        )

        im = axes[0].imshow(self.flux_upsampled_total, origin="lower", **kwargs)
        axes[0].set_title("Total")

        for ax, name in zip(axes[1:], self.fluxes_upsampled):
            flux = self.fluxes_upsampled[name]
            im = ax.imshow(flux, origin="lower", **kwargs)
            ax.set_title(name.title())

        add_cbar(im=im, ax=ax, fig=fig)
        return axes

    @property
    def config_table(self):
        """Configuration data as table (`~astropy.table.Table`)"""
        config = Table()

        for key, value in self.config.items():
            config[key] = [value]

        return config

    @property
    def wcs(self):
        """Optional wcs"""
        return self._wcs

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
