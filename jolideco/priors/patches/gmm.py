import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from astropy.table import Table
from astropy.utils import lazyproperty

from jolideco.utils.misc import format_class_str
from jolideco.utils.numpy import compute_precision_cholesky, get_pixel_weights

__all__ = ["GaussianMixtureModel", "GMM_REGISTRY"]


class GaussianMixtureModel(nn.Module):
    """Gaussian mixture model

    Attributes
    ----------
    means : `~torch.Tensor`
        Means
    covariances : `~torch.Tensor`
        Covariances
    weights : `~torch.Tensor`
        Weights
    precisions_cholesky : `~torch.Tensor`
        Precision matrices
    stride : int
        Stride of the patch. Will be used to compute a correction factor for overlapping patches.
        Overlapping pixels are down-weighted in the log-likelihood computation.
    """

    def __init__(self, means, covariances, weights, precisions_cholesky, stride=None):
        super().__init__()
        self.register_buffer("means", means)
        self.register_buffer("covariances", covariances)
        self.register_buffer("weights", weights)
        self.register_buffer("precisions_cholesky", precisions_cholesky)
        self.stride = stride

    @lazyproperty
    def means_numpy(self):
        """Means (~numpy.ndarray)"""
        return self.means.detach().cpu().numpy()

    @lazyproperty
    def covariances_numpy(self):
        """Covariances (~numpy.ndarray)"""
        return self.covariances.detach().cpu().numpy()

    @lazyproperty
    def weights_numpy(self):
        """Weights (~numpy.ndarray)"""
        return self.weights.detach().cpu().numpy()

    @classmethod
    def from_numpy(cls, means, covariances, weights, stride=None):
        """Gaussian mixture model

        Parameters
        ----------
        means : `~numpy.ndarray`
            Means
        covariances : `~numpy.ndarray`
            Covariances
        weights : `~numpy.ndarray`
            Weights
        stride : int
            Stride of the patch. Will be used to compute a correction factor for
            overlapping patches. Overlapping pixels are down-weighted in the
            log-likelihood computation.

        Returns
        -------
        gmm : `GaussianMixtureModel`
            Gaussian mixture model.
        """
        precisions_cholesky = compute_precision_cholesky(covariances=covariances)

        return cls(
            means=torch.from_numpy(means.astype(np.float32)),
            covariances=torch.from_numpy(covariances.astype(np.float32)),
            weights=torch.from_numpy(weights.astype(np.float32)),
            precisions_cholesky=torch.from_numpy(
                precisions_cholesky.astype(np.float32)
            ),
            stride=stride,
        )

    @lazyproperty
    def patch_shape(self):
        """Patch shape (tuple)"""
        shape_mean = self.means.shape
        npix = int((shape_mean[-1]) ** 0.5)
        return npix, npix

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
            w, v = linalg.eigh(self.covariances_numpy[idx])
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
    def means_precisions_cholesky(self):
        """Precision matrices pytorch"""
        means_precisions = []

        iterate = zip(self.means, self.precisions_cholesky)

        for mu, prec_chol in iterate:
            y = torch.matmul(mu, prec_chol)
            means_precisions.append(y)

        return means_precisions

    @lazyproperty
    def log_det_cholesky_numpy(self):
        """Compute the log-det of the cholesky decomposition of matrices"""
        return self.log_det_cholesky.detach().cpu().numpy()

    @lazyproperty
    def log_det_cholesky(self):
        """Precision matrices pytorch"""
        reshaped = self.precisions_cholesky.reshape(self.n_components, -1)
        reshaped = reshaped[:, :: self.n_features + 1]
        return torch.sum(torch.log(reshaped), axis=1)

    def estimate_log_prob_numpy(self, x):
        """Compute log likelihood for given feature vector"""
        n_samples, n_features = x.shape

        log_prob = np.empty((n_samples, self.n_components))

        for k, (mu, prec_chol) in enumerate(
            zip(self.means_numpy, self.precisions_cholesky_numpy)
        ):
            y = np.dot(x, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y) * self.pixel_weights_numpy, axis=1)

        # Since we are using the precision of the Cholesky decomposition,
        # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
        return (
            -0.5 * (n_features * np.log(2 * np.pi) + log_prob)
            + self.log_det_cholesky_numpy
        )

    @lazyproperty
    def pixel_weights(self):
        """Pixel weights"""
        return torch.from_numpy(self.pixel_weights_numpy)

    @lazyproperty
    def pixel_weights_numpy(self):
        """Pixel weights"""
        if self.stride is None:
            weights = np.ones(self.patch_shape)
        else:
            weights = get_pixel_weights(
                patch_shape=self.patch_shape, stride=self.stride
            )
        return weights.reshape((1, -1))

    def estimate_log_prob(self, x):
        """Compute log likelihood for given feature vector"""
        n_samples, n_features = x.shape

        log_prob = torch.empty((n_samples, self.n_components))

        iterate = zip(self.means_precisions_cholesky, self.precisions_cholesky)

        for k, (mu_prec, prec_chol) in enumerate(iterate):
            y = torch.matmul(x, prec_chol) - mu_prec
            log_prob[:, k] = torch.sum(torch.square(y) * self.pixel_weights, axis=1)

        # Since we are using the precision of the Cholesky decomposition,
        # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
        two_pi = torch.tensor(2 * np.pi)
        return (
            -0.5 * (n_features * torch.log(two_pi) + log_prob) + self.log_det_cholesky
        )

    @classmethod
    def from_sklearn_gmm(cls, gmm):
        """Create from sklearn GMM"""
        return cls.from_numpy(
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
        return cls.from_numpy(
            means=means, covariances=covariances, weights=weights, **kwargs
        )

    @lazyproperty
    def covariance_det(self):
        """Covariance determinant"""
        covar = self.covariances_numpy[0]
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

        k = self.means_numpy.shape[1]

        diff = self.means_numpy[0] - other.means[0]
        term_mean = diff.T @ other.precisons_cholesky[0] @ diff
        term_trace = np.trace(other.precisions_cholesky[0] * self.covariances_numpy[0])
        term_log = np.log(other.covariance_det / self.covariance_det)
        return 0.5 * (term_log - k + term_mean + term_trace)

    def is_equal(self, other):
        # TODO: improve check here?
        if not self.covariances.shape == other.covariances.shape:
            return False
        else:
            return np.allclose(self.covariances_numpy, other.covariances_numpy)

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
