from math import sqrt
from matplotlib import patches
import numpy as np
from astropy.utils import lazyproperty
from astropy.convolution import Gaussian2DKernel
import torch
import torch.nn.functional as F
from jolideco.utils.torch import (
    view_as_overlapping_patches_torch,
    view_as_random_overlapping_patches_torch,
    convolve_fft_torch,
    cycle_spin,
)
from jolideco.utils.numpy import reconstruct_from_overlapping_patches
from jolideco.utils.norms import MaxImageNorm
from .gmm import GaussianMixtureModel
from .gmm_tree import BinaryTreeGaussianMixtureModel
from ..core import Prior


__all__ = ["GMMPatchPrior", "MultiScalePrior"]


class GMMPatchPrior(Prior):
    """Patch prior

    Attributes
    ----------
    gmm : `GaussianMixtureModel`
        Gaussian mixture model.
    stride : int or "random"
        Stride of the patches. By default it is half of the patch size.
    cycle_spin : bool
        Apply cycle spin.
    generator : `~torch.Generator`
        Random number generator
    norm : `~jolideco.utils.ImageNorm`
        Image normalisation applied before the GMM patch prior.
    jitter : bool
        Jitter patch positions.
    """

    def __init__(
        self, gmm, stride=None, cycle_spin=True, generator=None, norm=None, jitter=False
    ):
        super().__init__()

        self.gmm = gmm
        self.stride = stride
        self.cycle_spin = cycle_spin

        if generator is None:
            generator = torch.Generator()

        self.generator = generator

        if norm is None:
            norm = MaxImageNorm()

        self.norm = norm
        self.jitter = jitter

    @lazyproperty
    def use_gmm_tree(self):
        """Use binary tree GMM"""
        return isinstance(self.gmm, BinaryTreeGaussianMixtureModel)

    def prior_image(self, flux):
        """Compute a patch image from the eigenimages of the best fittign patches.

        Parameters
        ----------
        flux : `~pytorch.Tensor`
            Reconstructed flux

        Returns
        -------
        prior_image : `~numpy.ndarray`
            Average prior image.
        """
        patches, mean, shift = self.split_into_patches(flux=flux)
        loglike = self.gmm.estimate_log_prob_torch(x=patches)
        idx = torch.argmax(loglike, dim=1)

        eigen_images = self.gmm.eigen_images
        patches = eigen_images[idx] + mean.detach().numpy().reshape((-1, 1, 1))

        reco = reconstruct_from_overlapping_patches(
            patches=patches, image_shape=flux.shape[2:], stride=self.stride
        )
        image = np.roll(reco, shift=-1 * np.array(shift), axis=(0, 1))
        image = torch.from_numpy(image)
        scaled = self.norm.inverse(image=image)
        return scaled.detach().numpy()

    def prior_image_average(self, flux, n_average=100):
        """Compute an average patch image by averaging using cycle spinning.

        Parameters
        ----------
        flux : `~pytorch.Tensor`
            Reconstructed flux
        n_average: int
            Number of image to average over using cycle spinning.

        Returns
        -------
        prior_image : `~numpy.ndarray`
            Average prior image.
        """
        flux = torch.from_numpy(flux[None, None])
        images = []

        for _ in range(n_average):
            image = self.prior_image(flux)
            images.append(image)

        return np.mean(images, axis=0)

    @lazyproperty
    def overlap(self):
        """Patch overlap"""
        return max(self.patch_shape) - self.stride

    @lazyproperty
    def patch_shape(self):
        """Patch shape (tuple)"""
        if self.use_gmm_tree:
            shape_mean = self.gmm.gmm.means.shape
        else:
            shape_mean = self.gmm.means.shape
        npix = int(sqrt(shape_mean[-1]))
        return npix, npix

    def split_into_patches(self, flux):
        """Split into patches"""
        normed = self.norm(flux)

        if self.cycle_spin:
            normed, shifts = cycle_spin(
                image=normed, patch_shape=self.patch_shape, generator=self.generator
            )
        else:
            shifts = (0, 0)

        if self.jitter:
            patches = view_as_random_overlapping_patches_torch(
                image=normed,
                shape=self.patch_shape,
                stride=self.stride,
                generator=self.generator,
            )
        else:
            patches = view_as_overlapping_patches_torch(
                image=normed, shape=self.patch_shape, stride=self.stride
            )

        mean = torch.mean(patches, dim=1, keepdims=True)
        patches = patches - mean
        return patches, mean, shifts

    @lazyproperty
    def log_like_weight(self):
        """Log likelihood weight"""
        return self.stride**2 / np.multiply(*self.patch_shape)

    def __call__(self, flux):
        """Evaluate the prior

        Parameters
        ----------
        flux : `~pytorch.Tensor`
            Reconstructed flux

        Returns
        -------
        log_prior : float
            Summed log prior over all overlapping patches.
        """
        patches, _, _ = self.split_into_patches(flux=flux)

        if self.use_gmm_tree:
            max_loglike = self.gmm.estimate_log_prob_max_torch(x=patches)
        else:
            loglike = self.gmm.estimate_log_prob_torch(x=patches)
            max_loglike = torch.max(loglike, dim=1).values

        return torch.sum(max_loglike) * self.log_like_weight


class MultiScalePrior(Prior):
    """Multiscale prior

    Apply a given prior per resolution level and sum up the log likelihood contributions
    across all resolution levels.

    Parameters
    ----------
    prior : `Prior`
        Prior instance
    n_levels : int, optional
        Number of multiscale levels
    """

    def __init__(self, prior, n_levels=2, weights=None, cycle_spin=False):
        super().__init__()
        self.n_levels = n_levels

        #        if not isinstance(prior, GMMPatchPrior):
        #            raise ValueError("Multi scale prior only supports `GMMPatchPrior`")

        self.cycle_spin = cycle_spin
        self.prior = prior

        if weights is None:
            weights = [1 / n_levels] * n_levels

        self.weights = weights

    def __call__(self, flux):
        """Evaluate the prior

        Parameters
        ----------
        flux : `~pytorch.Tensor`
            Reconstructed flux

        Returns
        -------
        log_prior : float
            Summed log prior over all overlapping patches.
        """
        log_like = 0

        if self.cycle_spin:
            flux, _ = cycle_spin(
                image=flux,
                patch_shape=self.prior.patch_shape,
                generator=self.prior.generator,
            )

        for idx, weight in enumerate(self.weights):
            if weight == 0:
                continue

            factor = 2**idx
            sigma = 2 * factor / 6.0
            kernel = Gaussian2DKernel(sigma).array[None, None]
            kernel = torch.from_numpy(kernel.astype(np.float32))

            flux = convolve_fft_torch(flux, kernel=kernel)
            flux_downsampled = F.avg_pool2d(flux, kernel_size=factor)
            log_like_level = self.prior(flux=flux_downsampled)
            log_like += factor**2 * weight * log_like_level

        return log_like
