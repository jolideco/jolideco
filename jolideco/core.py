import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.nddata import block_reduce
from astropy.table import Table
from astropy.utils import lazyproperty
from astropy.visualization import simple_norm

from .models import FluxComponent, FluxComponents, NPredModel
from .priors import PRIOR_REGISTRY, Priors, UniformPrior
from .utils.io import IO_FORMATS_READ, IO_FORMATS_WRITE
from .utils.plot import add_cbar
from .utils.torch import TORCH_DEFAULT_DEVICE, dataset_to_torch

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
    upsampling_factor : int
        Internal spatial upsampling factor for the reconstructed flux.
    use_log_flux : bool
        Use log scaling for flux
    freeze : set
        Components to freeze.
    fit_background_norm : bool
        Whether to fit background norm.
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
        upsampling_factor=1,
        use_log_flux=True,
        freeze={},
        fit_background_norm=False,
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
        self.upsampling_factor = upsampling_factor
        self.use_log_flux = use_log_flux
        self.freeze = freeze
        self.fit_background_norm = fit_background_norm
        self.device = torch.device(device)

        self.loss_function = nn.PoissonNLLLoss(
            log_input=False, reduction="sum", eps=1e-25, full=True
        )

    @property
    def freeze(self):
        """Model components to freeze"""
        return self._freeze

    @freeze.setter
    def freeze(self, value):
        """Set model components to freeze"""
        valid_names = set(self.loss_function_prior)

        diff_names = set(value).difference(valid_names)

        if diff_names:
            raise ValueError(
                f"Not a valid model component to freeze {diff_names}."
                f"Choose from {valid_names}."
            )

        self._freeze = value

    def to_dict(self):
        """Convert deconvolver configuration to dict, with simple data types.

        Returns
        -------
        data : dict
            Parameter dict.
        """
        data = {}
        data.update(self.__dict__)
        data.pop("loss_function")

        for key, value in PRIOR_REGISTRY.items():
            if isinstance(self.loss_function_prior, value):
                data["loss_function_prior"] = key

        data["device"] = str(self.device)
        data["freeze"] = ",".join(data.pop("_freeze"))
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

    def fluxes_init_from_datasets(self, datasets):
        """Compute flux init from datasets by averaging over the raw uncolvved flux estimate.

        Parameters
        ----------
        datasets : list of dict
            List of dictionaries containing, "counts", "psf", "background" and "exposure".

        Returns
        -------
        flux_init : `~numpy.ndarray`
            Initial flux estimate.
        """
        fluxes = []

        for dataset in datasets:
            flux = dataset["counts"] / dataset["exposure"] - dataset["background"]
            fluxes.append(flux)

        return np.nanmean(fluxes, axis=0)

    def prepare_flux_init(self, flux_init):
        """Prepare flux init

        Parameters
        ----------
        flux_init : `~numpy.ndarray`
            Initial flux estimate.

        Returns
        -------
        flux_init : `~torch.Tensor`
            Initial flux estimate.
        """
        # convert to pytorch tensors
        flux_init = torch.from_numpy(flux_init[np.newaxis, np.newaxis])

        flux_init = F.interpolate(
            flux_init, scale_factor=self.upsampling_factor, mode="bilinear"
        )

        flux_init = flux_init.to(self.device)
        return flux_init

    def prepare_datasets(self, datasets):
        """Prepare datasets by upsampling

        Parameters
        ----------
        datasets : list of dict
            List of dictionaries containing, "counts", "psf", "background" and "exposure".

        Returns
        -------
        datasets_torch : list of dict
            List of dictionaries containing, "counts", "psf", "background" and "exposure".

        """
        datasets_torch = []

        for dataset in datasets:
            dataset_torch = dataset_to_torch(
                dataset=dataset,
                upsampling_factor=self.upsampling_factor,
                device=self.device,
            )
            datasets_torch.append(dataset_torch)

        return datasets_torch

    def prepare_trace_loss_init(self, datasets):
        """Prepare trace loss init

        Parameters
        ----------
        datasets : list of dict
            List of dictionaries containing, "counts", "psf", "background" and "exposure".


        Returns
        -------
        tarce_loss : `~astropy.table.Table`
            Trace loss table.
        """
        names = ["total", "datasets-total", "priors-total"]
        names += [f"prior-{name}" for name in self.loss_function_prior]
        names += [f"dataset-{idx}" for idx in range(len(datasets))]
        return Table(names=names)

    def get_trace_loss_row(self, datasets, npred_model, components):
        """Append traceloss table"""

        loss_datasets = []

        for data in datasets:
            npred = npred_model(
                flux=components.evaluate(),
                exposure=data["exposure"],
                background=data["background"],
                psf=data.get("psf", None),
            )
            loss = self.loss_function(npred, data["counts"])
            loss_datasets.append(loss.item())

        prior_weight = len(datasets) * self.upsampling_factor**2

        loss_priors = []

        for name, prior in self.loss_function_prior.items():
            flux = components[name].flux
            value = prior(flux) / prior_weight
            loss_priors.append(value.item())

        loss_datasets_total = sum(loss_datasets)
        loss_priors_total = sum(loss_priors)

        loss_total = loss_datasets_total - self.beta * loss_priors_total

        row = {
            "total": loss_total,
            "datasets-total": loss_datasets_total,
            "priors-total": self.beta * loss_priors_total,
        }

        for name, value in zip(self.loss_function_prior, loss_priors):
            row[f"prior-{name}"] = value

        for idx, value in enumerate(loss_datasets):
            row[f"dataset-{idx}"] = value

        return row

    def run(self, datasets, fluxes_init=None):
        """Run the MAP deconvolver

        Parameters
        ----------
        datasets : list of dict
            List of dictionaries containing, "counts", "psf", "background" and "exposure".
        fluxes_init : dict of `~numpy.ndarray`
            Initial flux estimates.

        Returns
        -------
        flux : `~numpy.ndarray`
            Reconstructed flux.
        """
        if fluxes_init is None:
            fluxes_init = self.fluxes_init_from_datasets(datasets=datasets)

        components = FluxComponents()

        for name, flux_init in fluxes_init.items():
            flux_init = self.prepare_flux_init(flux_init=flux_init)
            flux_model = FluxComponent(
                flux_init=flux_init,
                use_log_flux=self.use_log_flux,
            )
            components[name] = flux_model

        datasets = self.prepare_datasets(datasets=datasets)
        trace_loss = self.prepare_trace_loss_init(datasets=datasets)

        npred_model = NPredModel(
            upsampling_factor=self.upsampling_factor,
        ).to(self.device)

        parameters = list(self.loss_function_prior.parameters())

        for name, component in components.items():
            if name not in self.freeze:
                parameters += list(component.parameters())

        if self.fit_background_norm:
            parameters += [npred_model.background_norm]

        optimizer = torch.optim.Adam(
            params=parameters,
            lr=self.learning_rate,
        )

        prior_weight = len(datasets) * self.upsampling_factor**2

        for epoch in range(self.n_epochs):
            for data in datasets:
                optimizer.zero_grad()

                # evaluate npred model
                npred = npred_model(
                    flux=components.evaluate(),
                    exposure=data["exposure"],
                    background=data["background"],
                    psf=data.get("psf", None),
                )

                # compute Poisson loss
                loss = self.loss_function(npred, data["counts"])

                # compute prior losses
                loss_prior = self.loss_function_prior(
                    fluxes=components.to_dict()
                )

                loss_total = loss - self.beta * loss_prior / prior_weight

                loss_total.backward()
                optimizer.step()

            row = self.get_trace_loss_row(
                    datasets=datasets, npred_model=npred_model, components=components
                )

            message = (
                f'Epoch: {epoch}, {row["total"]}, '
                f'{row["datasets-total"]}, {row["priors-total"]}'
            )
            log.info(message)

            trace_loss.add_row(row)

        return MAPDeconvolverResult(
            config=self.to_dict(),
            fluxes_upsampled=components.to_numpy(),
            fluxes_init=fluxes_init,
            trace_loss=trace_loss,
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
