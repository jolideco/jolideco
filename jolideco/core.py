import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

logging.basicConfig(level=logging.INFO)


def dataset_to_pytorch(dataset, scale_factor=None):
    """Convert to pytorch tensors"""
    dims = (np.newaxis, np.newaxis)

    dataset_torch = {}

    for key, value in dataset.items():
        tensor = torch.from_numpy(value[dims])

        if key in ["psf", "exposure", "background", "flux"] and scale_factor:
            tensor = F.interpolate(tensor, scale_factor=scale_factor)

        if key in ["psf", "background", "flux"] and scale_factor:
            tensor = tensor / scale_factor**2

        dataset_torch[key] = tensor

    return dataset_torch


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
    """

    def __init__(
        self,
        n_epochs,
        beta=1e-6,
        loss_function_prior=None,
        learning_rate=0.1,
        upsampling_factor=None,
    ):
        self.n_epochs = n_epochs
        self.beta = beta
        self.loss_function_prior = loss_function_prior
        self.learning_rate = learning_rate
        self.upsampling_factor = upsampling_factor
        self.log = logging.getLogger(__name__)

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
        datasets = [dataset_to_pytorch(_, scale_factor=self.upsampling_factor) for _ in datasets]

        trace_loss = []
        trace_validation_loss = []

        npred_model = SimpleNPredModel(
            flux_init=flux_init, upsampling_factor=self.upsampling_factor
        )

        optimizer = torch.optim.Adam(
            params=npred_model.parameters(),
            lr=self.learning_rate,
        )

        loss_function = nn.PoissonNLLLoss(log_input=False, reduction="sum", eps=1e-25)

        for epoch in range(self.n_epochs):  # loop over the dataset multiple times
            value_loss_total = value_loss_prior = 0

            npred_model.train(True)

            for data in datasets:
                optimizer.zero_grad()

                npred = npred_model(
                    psf=data["psf"],
                    exposure=data["exposure"],
                    background=data["background"],
                )

                loss = loss_function(npred, data["counts"])
                loss_prior = self.loss_function_prior(flux=npred_model.flux)
                loss_total = loss - self.beta * loss_prior

                value_loss_total += loss_total.item() / len(datasets)
                value_loss_prior += self.beta * loss_prior.item()

                loss_total.backward()
                optimizer.step()

            value_loss = value_loss_total + value_loss_prior
            message = (
                f"Epoch: {epoch}, {value_loss_total}, {value_loss}, {value_loss_prior}"
            )
            self.log.info(message)

            trace_loss.append(value_loss_total)

        return {
            "flux": npred_model.flux.detach().numpy()[0][0],
            "trace-loss": trace_loss,
            "trace-validation-loss": trace_validation_loss,
        }
