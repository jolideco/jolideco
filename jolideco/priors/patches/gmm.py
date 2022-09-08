import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from astropy.table import Table
from astropy.utils import lazyproperty

from jolideco.utils.misc import format_class_str
from jolideco.utils.numpy import get_pixel_weights
from jolideco.utils.torch import TORCH_DEFAULT_DEVICE

__all__ = ["GaussianMixtureModel", "GMM_REGISTRY"]


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
    stride : int
        Stride of the patch. Will be used to compute a correction factor for overlapping patches.
        Overlapping pixels are down-weighted in the log-likelihood computation.
    """

    def __init__(
        self, means, covariances, weights, device=TORCH_DEFAULT_DEVICE, stride=None
    ):
        super().__init__()

        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.device = device
        self.stride = stride

    @lazyproperty
    def patch_shape(self):
        """Patch shape (tuple)"""
        shape_mean = self.means.shape
        npix = int((shape_mean[-1]) ** 0.5)
        return npix, npix

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
        from scipy import linalg

        eigen_images = []

        for idx in range(self.n_components):
            w, v = linalg.eigh(self.covariances[idx])
            data = (v @ w).reshape(self.patch_shape)
            eigen_images.append(data)

        return np.stack(eigen_images)

    def plot_eigen_images(self, ncols=20, figsize=(16, 10)):
        """Plot images"""
        nrows = self.n_components // ncols

        _, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

        for idx, ax in enumerate(axes.flat):
            data = self.eigen_images[idx]
            ax.imshow(data)
            ax.set_axis_off()
            ax.set_title(f"{idx}")

    @lazyproperty
    def precisions_cholesky(self):
        """Cholesky decomposition of the precision matrix"""
        from scipy import linalg

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
        """Precision matrices pytorch"""
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
        """Precision matrices pytorch"""
        return torch.from_numpy(self.log_det_cholesky.astype(np.float32)).to(
            self.device
        )

    def estimate_log_prob(self, x):
        """Compute log likelihood for given feature vector"""
        n_samples, n_features = x.shape

        log_prob = np.empty((n_samples, self.n_components))

        for k, (mu, prec_chol) in enumerate(zip(self.means, self.precisions_cholesky)):
            y = np.dot(x, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y) * self.pixel_weights, axis=1)

        # Since we are using the precision of the Cholesky decomposition,
        # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
        return (
            -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + self.log_det_cholesky
        )

    @lazyproperty
    def pixel_weights_torch(self):
        """Pixel weights"""
        return torch.from_numpy(self.pixel_weights)

    @lazyproperty
    def pixel_weights(self):
        """Pixel weights"""
        if self.stride is None:
            weights = np.ones(self.patch_shape)
        else:
            weights = get_pixel_weights(
                patch_shape=self.patch_shape, stride=self.stride
            )
        return weights.reshape((1, -1))

    def estimate_log_prob_torch(self, x):
        """Compute log likelihood for given feature vector"""
        n_samples, n_features = x.shape

        log_prob = torch.empty((n_samples, self.n_components)).to(self.device)

        iterate = zip(
            self.means_precisions_cholesky_torch, self.precisions_cholesky_torch
        )

        for k, (mu_prec, prec_chol) in enumerate(iterate):
            y = torch.matmul(x, prec_chol) - mu_prec
            log_prob[:, k] = torch.sum(
                torch.square(y) * self.pixel_weights_torch, axis=1
            )

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
    def from_registry(cls, name, **kwargs):
        """Create GMM from registry

        Parameters
        ----------
        name : str
            Name of the registered GMM.

        Returns
        -------
        gmm : `GaussianMixtureModel`
            Gaussian mixture model.
        """
        from jolideco.priors.patches.gmm import GMM_REGISTRY

        available_names = list(GMM_REGISTRY.keys())

        if name not in available_names:
            raise ValueError(
                f"Not a supported GMM {name}, choose from {available_names}"
            )

        kwargs.update(GMM_REGISTRY[name])
        return cls.read(**kwargs)

    @classmethod
    def read(cls, filename, format="epll-matlab", **kwargs):
        """Read from matlab file

        Parameters
        ----------
        filename : str or Path
            Filename
        format : {"epll-matlab", "epll-matlab-16x16", "table"}
            Format
        **kwargs : dict
            Keyword arguments passed to GaussianMixtureModel

        Returns
        -------
        gmm : `GaussianMixtureModel`
            Gaussian mixture model.
        """
        import scipy.io as sio

        filename = str(Path(os.path.expandvars(filename)))

        if format == "epll-matlab":
            gmm_dict = sio.loadmat(filename)
            gmm_data = gmm_dict["GS"]

            means = gmm_data["means"][0][0].T
            covariances = gmm_data["covs"][0][0].T
            weights = gmm_data["mixweights"][0][0][:, 0]
        elif format == "epll-matlab-16x16":
            gmm_dict = sio.loadmat(filename)
            gmm_data = gmm_dict["GMM"]

            means = np.zeros((200, 256))
            covariances = gmm_data["covs"][0][0].T
            weights = gmm_data["mixweights"][0][0][:, 0]
        elif format == "table":
            table = Table.read(filename)
            means = table["means"].data
            weights = table["weights"].data
            covariances = table["covariances"].data
        else:
            raise ValueError(f"Not a supported format {format}")

        npix = int((means.shape[-1]) ** 0.5)
        kwargs.setdefault("stride", npix // 2)
        return cls(means=means, covariances=covariances, weights=weights, **kwargs)

    @lazyproperty
    def covariance_det(self):
        """Covariance determinant"""
        covar = self.covariances[0]
        return np.linalg.det(covar)

    def kl_divergence(self, other):
        """Compute KL divergence with respect to another GMM"

        See https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/

        Parameters
        ----------
        other : `~GaussianMixtureModel`
            Other GMM

        Returns
        -------
        value : float
            KL divergence
        """
        if not (self.n_components == 1 and other.n_components == 1):
            raise ValueError(
                "KL divergence can onlyy be computed for single component GMM"
            )

        k = self.means.shape[1]

        diff = self.means[0] - other.means[0]
        term_mean = diff.T @ other.precisons_cholesky[0] @ diff
        term_trace = np.trace(other.precisions_cholesky[0] * self.covariances[0])
        term_log = np.log(other.covariance_det / self.covariance_det)
        return 0.5 * (term_log - k + term_mean + term_trace)

    def is_equal(self, other):
        # TODO: improve check here?
        if not self.covariances.shape == other.covariances.shape:
            return False
        else:
            return np.allclose(self.covariances, other.covariances)

    def symmetric_kl_divergence(self, other):
        """Symmetric KL divergence"""
        return other.kl_divergence(other=self) + self.kl_divergence(other=other)

    def to_dict(self):
        """To dict"""
        data = {}

        from jolideco.priors.patches.gmm import GMM_REGISTRY

        for name in GMM_REGISTRY:
            gmm = GaussianMixtureModel.from_registry(name=name)
            if gmm.is_equal(self):
                break

        data["type"] = name
        data["stride"] = self.stride
        return data

    @classmethod
    def from_dict(cls, data):
        """Create from dict

        Parameters
        ----------
        data : dict
            Data dictionary

        Returns
        -------
        gmm : `~GaussianMixtureModel`
            Gaussian mixture model
        """
        name, stride = data["type"], data["stride"]
        return cls.from_registry(name=name, stride=stride)

    def __str__(self):
        return format_class_str(instance=self)


GMM_REGISTRY = {
    "zoran-weiss": {
        "filename": "$JOLIDECO_GMM_LIBRARY/GSModel_8x8_200_2M_noDC_zeromean.mat",
        "format": "epll-matlab",
    },
    "gleam-v0.1": {
        "filename": "$JOLIDECO_GMM_LIBRARY/patch-priors-gleam-v0.1.fits",
        "format": "table",
    },
    "gleam-v0.2": {
        "filename": "$JOLIDECO_GMM_LIBRARY/patch-priors-gleam-v0.2.fits",
        "format": "table",
    },
}
