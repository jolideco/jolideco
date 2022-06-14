import numpy as np
import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
from astropy.table import Table
from .models import SimpleNPredModel
from .priors import UniformPrior, PRIOR_REGISTRY
from .utils.torch import dataset_to_torch, TORCH_DEFAULT_DEVICE

logging.basicConfig(level=logging.INFO)


class MAPDeconvolver:
    """Maximum A-Posteriori deconvolver

    Parameters
    ----------
    n_epochs : int
        Number of epochs to train
    beta : float
        Scale factor for the prior.
    loss_function_prior : `~jolideco.priors.Prior`
        Loss function for the prior (optional).
    learning_rate : float
        Learning rate
    upsampling_factor : int
        Internal spatial upsampling factor for the reconstructed flux.
    use_log_flux : bool
        Use log scaling for flux
    device : `~pytorch.Device`
        Pytorch device
    """

    def __init__(
        self,
        n_epochs,
        beta=1e-6,
        loss_function_prior=None,
        learning_rate=0.1,
        upsampling_factor=None,
        use_log_flux=True,
        device=TORCH_DEFAULT_DEVICE,
    ):
        self.n_epochs = n_epochs
        self.beta = beta

        if loss_function_prior is None:
            loss_function_prior = UniformPrior()

        self.loss_function_prior = loss_function_prior.to(device)
        self.learning_rate = learning_rate
        self.upsampling_factor = upsampling_factor
        self.use_log_flux = use_log_flux
        self.log = logging.getLogger(__name__)
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

        data.pop("log")
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

    def run(self, flux_init, datasets):
        """Run the MAP deconvolver

        Parameters
        ----------
        flux_init : `~numpy.ndarray`
            Initial flux estimate.
        datasets : list of dict
            List of dictionaries containing, "counts", "psf", "background" and "exposure".

        Returns
        -------
        flux : `~numpy.ndarray`
            Reconstructed flux.
        """
        # convert to pytorch tensors
        flux_init = torch.from_numpy(flux_init[np.newaxis, np.newaxis])
        flux_init = flux_init.to(self.device)

        datasets = [
            dataset_to_torch(
                _, upsampling_factor=self.upsampling_factor, device=self.device
            )
            for _ in datasets
        ]

        names = ["total", "prior"]
        names += [f"dataset-{idx}" for idx in range(len(datasets))]

        trace_loss = Table(names=names)

        npred_model = SimpleNPredModel(
            flux_init=flux_init,
            upsampling_factor=self.upsampling_factor,
            use_log_flux=self.use_log_flux,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            params=npred_model.parameters(),
            lr=self.learning_rate,
        )

        loss_function = nn.PoissonNLLLoss(log_input=False, reduction="sum", eps=1e-25)

        for epoch in range(self.n_epochs):  # loop over the dataset multiple times
            value_loss_total = value_loss_prior = 0

            npred_model.train(True)

            loss_datasets = []

            for data in datasets:
                optimizer.zero_grad()

                npred = npred_model(
                    exposure=data["exposure"],
                    background=data["background"],
                    psf=data.get("psf", None),
                )

                loss = loss_function(npred, data["counts"])
                loss_datasets.append(loss.item())
                loss_prior = self.loss_function_prior(flux=npred_model.flux)
                loss_total = loss - self.beta * loss_prior

                value_loss_total += loss_total.item()
                value_loss_prior += self.beta * loss_prior.item()

                loss_total.backward()
                optimizer.step()

            value_loss = value_loss_total + value_loss_prior
            message = (
                f"Epoch: {epoch}, {value_loss_total}, {value_loss}, {value_loss_prior}"
            )
            self.log.info(message)

            row = {
                "total": value_loss_total,
                "prior": value_loss_prior,
            }

            for idx, value in enumerate(loss_datasets):
                row[f"dataset-{idx}"] = value

            trace_loss.add_row(row)

        flux = npred_model.flux.detach().cpu()

        return MAPDeconvolverResult(
            config=self.to_dict(),
            flux=flux.numpy()[0][0],
            trace_loss=trace_loss,
        )


class MAPDeconvolverResult:
    """MAP deconvolver result"""

    def __init__(self, config, flux, trace_loss):
        self.flux = flux
        self.trace_loss = trace_loss
        self._config = config

    @property
    def config(self):
        """Configuration data (`dict`)"""
        return self._config

    def plot_trace_loss(self, ax=None, **kwargs):
        """Plot traces"""
        from .utils.plot import plot_trace_loss

        ax = plt.gca() if ax is None else ax

        plot_trace_loss(ax=ax, trace_loss=self.trace_loss, **kwargs)
        return ax
