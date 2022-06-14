import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from scipy import linalg
from astropy.utils import lazyproperty
from astropy.table import Table
from jolideco.core import DEVICE_TORCH

__all__ = ["GaussianMixtureModel"]


class GaussianMixtureModel(nn.Module):
    """Gaussian mixture model

    Attributes
    ----------
    means : `~numpy.ndarray`
        Means
    covariances : `~numpy.ndarray`
        Covariances
    weights : `~numpy.ndarray`
        Weights
    device : `~pytorch.Device`
        Pytorch device
    """

    def __init__(self, means, covariances, weights, device=DEVICE_TORCH):
        super().__init__()

        # TODO: assert shapes
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.device = device

    @lazyproperty
    def means_torch(self):
        """Number of features"""
        return torch.from_numpy(self.means.astype(np.float32)).to(self.device)

    @lazyproperty
    def n_features(self):
        """Number of features"""
        _, n_features, _ = self.covariances.shape
        return n_features

    @lazyproperty
    def n_components(self):
        """Number of features"""
        n_components, _, _ = self.covariances.shape
        return n_components

    @lazyproperty
    def eigen_images(self):
        """Eigen images"""
        eigen_images = []

        for idx in range(self.n_components):
            w, v = linalg.eigh(self.covariances[idx])
            data = (v @ w).reshape((8, 8))
            eigen_images.append(data)

        return np.stack(eigen_images)

    def plot_eigen_images(self, ncols=20, figsize=(16, 10)):
        """Plot images"""
        nrows = self.n_components // ncols

        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

        for idx, ax in enumerate(axes.flat):
            data = self.eigen_images[idx]
            ax.imshow(data)
            ax.set_axis_off()
            ax.set_title(f"{idx}")

    @lazyproperty
    def precisions_cholesky(self):
        """Cholesky decomposition of the precision matrix"""
        shape = (self.n_components, self.n_features, self.n_features)
        precisions_chol = np.empty(shape)

        for k, covariance in enumerate(self.covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(f"Cholesky decomposition failed for {covariance}")

            precisions_chol[k] = linalg.solve_triangular(
                cov_chol, np.eye(self.n_features), lower=True
            ).T

        return precisions_chol

    @lazyproperty
    def precisions_cholesky_torch(self):
        """Precisison matrix pytoch"""
        return torch.from_numpy(self.precisions_cholesky.astype(np.float32)).to(
            self.device
        )

    @lazyproperty
    def means_precisions_cholesky_torch(self):
        """Precisison matrix pytoch"""
        means_precisions = []

        iterate = zip(self.means_torch, self.precisions_cholesky_torch)

        for mu, prec_chol in iterate:
            y = torch.matmul(mu, prec_chol)
            means_precisions.append(y)

        return means_precisions

    @lazyproperty
    def log_det_cholesky(self):
        """Compute the log-det of the cholesky decomposition of matrices"""
        reshaped = self.precisions_cholesky.reshape(self.n_components, -1)
        reshaped = reshaped[:, :: self.n_features + 1]
        return np.sum(np.log(reshaped), axis=1)

    @lazyproperty
    def log_det_cholesky_torch(self):
        """Precisison matrix pytoch"""
        return torch.from_numpy(self.log_det_cholesky.astype(np.float32)).to(
            self.device
        )

    def estimate_log_prob(self, x):
        """Compute log likelihood for given feature vector"""
        n_samples, n_features = x.shape

        log_prob = np.empty((n_samples, self.n_components))

        for k, (mu, prec_chol) in enumerate(zip(self.means, self.precisions_cholesky)):
            y = np.dot(x, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

        # Since we are using the precision of the Cholesky decomposition,
        # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
        return (
            -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + self.log_det_cholesky
        )

    def estimate_log_prob_torch(self, x):
        """Compute log likelihood for given feature vector, assumes means = 0"""
        n_samples, n_features = x.shape

        log_prob = torch.empty((n_samples, self.n_components)).to(self.device)

        iterate = zip(
            self.means_precisions_cholesky_torch, self.precisions_cholesky_torch
        )

        for k, (mu_prec, prec_chol) in enumerate(iterate):
            y = torch.matmul(x, prec_chol) - mu_prec
            log_prob[:, k] = torch.sum(torch.square(y), axis=1)

        # Since we are using the precision of the Cholesky decomposition,
        # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
        two_pi = torch.tensor(2 * np.pi).to(self.device)
        return (
            -0.5 * (n_features * torch.log(two_pi) + log_prob)
            + self.log_det_cholesky_torch
        )

    @classmethod
    def from_sklearn_gmm(cls, gmm):
        """Create from sklearn GMM"""
        return cls(
            means=gmm.means_,
            covariances=gmm.covariances_,
            weights=gmm.weights_,
        )

    @classmethod
    def read(cls, filename, format="epll-matlab", device=DEVICE_TORCH):
        """Read from matlab file

        Parameters
        ----------
        filename : str or Path
            Filename
        format : {"epll-matlab", "table"}
            Format
        device : `~pytorch.Device`
            Pytorch device

        Returns
        -------
        gmm : `GaussianMixtureModel`
            Gaussian mixture model.
        """
        if format == "epll-matlab":
            gmm_dict = sio.loadmat(filename)
            gmm_data = gmm_dict["GS"]

            means = gmm_data["means"][0][0].T
            covariances = gmm_data["covs"][0][0].T
            weights = gmm_data["mixweights"][0][0][:, 0]
        elif format == "table":
            table = Table.read(filename)
            means = table["means"].data
            weights = table["weights"].data
            covariances = table["covariances"].data
        else:
            raise ValueError(f"Not a supported format {format}")

        return cls(means=means, covariances=covariances, weights=weights, device=device)
