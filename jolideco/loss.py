import numpy as np
from astropy.table import Table
from astropy.utils import lazyproperty
import torch
import torch.nn as nn
from .models import NPredModels
from .utils.torch import TORCH_DEFAULT_DEVICE

__all__ = ["PoissonLoss", "PriorLoss", "TotalLoss"]


class PoissonLoss:
    """Poisson loss functions

    Attributes
    ----------
    counts_all : list of `~torch.tensor`
        List of counts
    npred_models_all : list of `~NPredModels`
        List of predicted counts models
    """

    def __init__(self, counts_all, npred_models_all, names_all):
        self.counts_all = counts_all
        self.npred_models_all = npred_models_all
        self.loss_function = nn.PoissonNLLLoss(
            log_input=False, reduction="mean", eps=1e-25, full=True
        )
        self.names_all = names_all

    @property
    def n_datasets(self):
        """Number of datasets"""
        return len(self.counts_all)

    def evaluate(self, fluxes):
        """Evaluate loss per dataset

        Parameters
        ----------
        fluxes : tuple of  `~torch.tensor`
            Flux components
        """
        loss_datasets = []

        for counts, npred_model in zip(self.counts_all, self.npred_models_all):
            npred = npred_model.evaluate(fluxes=fluxes)
            loss = self.loss_function(npred, counts)
            loss_datasets.append(loss)

        return loss_datasets

    @property
    def iter_by_dataset(self):
        """Iterate by counts and predicted counts"""
        for data in zip(self.counts_all, self.npred_models_all):
            yield data

    @classmethod
    def from_datasets(
        cls, datasets, components, calibrations=None, device=TORCH_DEFAULT_DEVICE
    ):
        """Create loss function from datasets

        Parameters
        ----------
        datasets : dict of [str, dict]
            Dictionary containing a name of the dataset as key and a dictionary containing,
            the data like "counts", "psf", "background" and "exposure".
        components : `FluxComponents`
            Flux components
        calibrations: `NPredCalibrations`
            NPred calibrations
        device : `~pytorch.Device`
            Pytorch device

        Returns
        -------
        poisson_loss : `PoissonLoss`
            Poisson loss function.
        """
        npred_models_all, counts_all = [], []

        for name, dataset in datasets.items():
            if calibrations:
                calibration = calibrations[name]
            else:
                calibration = None

            npred_models = NPredModels.from_dataset_numpy(
                dataset=dataset, components=components, calibration=calibration
            )
            npred_models_all.append(npred_models.to(device))

            counts = torch.from_numpy(dataset["counts"][np.newaxis, np.newaxis]).to(
                device
            )
            counts_all.append(counts)

        return cls(
            counts_all=counts_all,
            npred_models_all=npred_models_all,
            names_all=list(datasets),
        )

    def __call__(self, fluxes):
        """Evaluate and sum all losses"""
        losses = self.evaluate(fluxes=fluxes)
        return torch.sum(losses)


class PriorLoss:
    """Prior loss function

    Attributes
    ----------
    priors : `Priors`
        Priors for each model componenet

    """

    def __init__(self, priors):
        self.priors = priors

    def evaluate(self, fluxes):
        """Evaluate loss per flux componenet

        Parameters
        ----------
        fluxes : tuple of  `~torch.tensor`
            Flux components
        """
        loss_priors = []

        for flux, prior in zip(fluxes, self.priors.values()):
            value = prior(flux=flux)
            loss_priors.append(value)

        return loss_priors

    def __call__(self, fluxes):
        """Evaluate and sum all losses"""
        losses = self.evaluate(fluxes=fluxes)
        return sum(losses)


class TotalLoss:
    """Total loss function

    Attributes
    ----------
    poisson_loss : `PoissonLoss`
        Poisson dataset loss
    prior_loss : `PriorLoss`
        Prior loss
    poisson_loss_validation : `PoissonLoss`
        Poisson validation dataset loss
    beta : float
        Relative weight of the prior.
    """

    def __init__(self, poisson_loss, prior_loss, poisson_loss_validation=None, beta=1):
        self.poisson_loss = poisson_loss
        self.poisson_loss_validation = poisson_loss_validation
        self.prior_loss = prior_loss
        self.beta = beta

    @lazyproperty
    def trace(self):
        """Trace of the total loss

        Returns
        -------
        trace : `~astroy.table.Table`
            Trace table
        """
        names = ["total", "datasets-total", "priors-total"]
        names += [f"prior-{name}" for name in self.prior_loss.priors]
        names += [f"dataset-{name}" for name in self.poisson_loss.names_all]

        if self.poisson_loss_validation:
            names += ["datasets-validation-total"]

        return Table(names=names)

    @torch.no_grad()
    def append_trace(self, fluxes):
        """Append trace

        Parameters
        ----------
        fluxes : tuple of  `~torch.tensor`
            Flux components
        """
        loss_datasets = [_.item() for _ in self.poisson_loss.evaluate(fluxes=fluxes)]
        loss_priors = [_.item() for _ in self.prior_loss.evaluate(fluxes=fluxes)]

        loss_datasets_total = sum(loss_datasets)
        loss_priors_total = self.beta * sum(loss_priors)

        loss_total = loss_datasets_total - loss_priors_total

        row = {
            "total": loss_total,
            "datasets-total": loss_datasets_total,
            "priors-total": -loss_priors_total,
        }

        for name, value in zip(self.prior_loss.priors, loss_priors):
            row[f"prior-{name}"] = -self.beta * value

        for name, value in zip(self.poisson_loss.names_all, loss_datasets):
            row[f"dataset-{name}"] = value

        if self.poisson_loss_validation:
            loss_datasets_total_test = [
                _.item() for _ in self.poisson_loss_validation.evaluate(fluxes=fluxes)
            ]
            row["datasets-validation-total"] = sum(loss_datasets_total_test)

        self.trace.add_row(row)

    @lazyproperty
    def prior_weight(self):
        """Prior weight"""
        return len(self.poisson_loss.counts_all)

    def __call__(self, fluxes):
        """Evaluate total loss"""
        loss_datasets = self.poisson_loss.evaluate(fluxes=fluxes)
        loss_priors = self.prior_loss.evaluate(fluxes=fluxes)
        return sum(loss_datasets) - self.beta * sum(loss_priors)

    def hessian_diagonals(self, fluxes):
        """Compute Hessian diagonal

        Parameters
        ----------
        fluxes : tuple of  `~torch.tensor`
            Flux components

        Returns
        -------
        hessian_diagonals : tuple of  `~torch.tensor`
            Hessian diagonals
        """
        shapes = tuple([_.size() for _ in fluxes])
        unit_vectors = tuple([torch.ones(shape) for shape in shapes])
        results = torch.autograd.functional.hvp(self, inputs=fluxes, v=unit_vectors)[1]
        return tuple(results)

    def fluxes_error(self, fluxes):
        """Compute flux errors

        Parameters
        ----------
        fluxes : tuple of  `~torch.tensor`
            Flux components

        Returns
        -------
        fluxes_error : tuple of  `~torch.tensor`
            Flux errors
        """
        fluxes_error = {}
        hessian_diagonals = self.hessian_diagonals(fluxes=fluxes)

        for name, hessian in zip(self.prior_loss.priors, hessian_diagonals):
            fluxes_error[name] = torch.sqrt(1 / hessian)

        return fluxes_error
