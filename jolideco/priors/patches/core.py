from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from astropy.convolution import Gaussian2DKernel
from astropy.utils import lazyproperty

from jolideco.priors.patches.gmm import GaussianMixtureModel
from jolideco.utils.norms import ImageNorm, MaxImageNorm
from jolideco.utils.numpy import reconstruct_from_overlapping_patches
from jolideco.utils.torch import (
    convolve_fft_torch,
    cycle_spin,
    cycle_spin_subpixel,
    view_as_overlapping_patches_torch,
    view_as_random_overlapping_patches_torch,
)

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
    cycle_spin_subpix : bool
        Apply subpixel cycle spin.
    generator : `~torch.Generator`
        Random number generator
    norm : `~jolideco.utils.ImageNorm`
        Image normalisation applied before the GMM patch prior.
    jitter : bool
        Jitter patch positions.
    """

    def __init__(
        self,
        gmm=None,
        stride=None,
        cycle_spin=True,
        cycle_spin_subpix=False,
        generator=None,
        norm=None,
        jitter=False,
    ):
        super().__init__()

        if gmm is None:
            gmm = GaussianMixtureModel.from_registry(name="zoran-weiss")

        print(gmm)

        self.gmm = gmm
        self.stride = gmm.stride
        self.cycle_spin = cycle_spin

        if generator is None:
            generator = torch.Generator()

        self.generator = generator

        if norm is None:
            norm = MaxImageNorm()

        self.norm = norm
        self.jitter = jitter
        self.cycle_spin_subpix = cycle_spin_subpix

    def to_dict(self):
        """To dict"""
        data = super().to_dict()
        data["stride"] = int(self.stride)
        data["cycle_spin"] = bool(self.cycle_spin)
        data["cycle_spin_subpix"] = bool(self.cycle_spin_subpix)
        data["jitter"] = bool(self.jitter)
        data["gmm"] = self.gmm.to_dict()
        data["norm"] = self.norm.to_dict()
        return data

    @classmethod
    def from_dict(cls, data):
        """Create from dict"""
        kwargs = data.copy()

        gmm_config = kwargs.pop("gmm")
        kwargs["gmm"] = GaussianMixtureModel.from_dict(gmm_config)

        norm_config = kwargs.pop("norm")
        kwargs["norm"] = ImageNorm.from_dict(norm_config)

        return cls(**kwargs)

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
        if self.jitter:
            raise ValueError("Computing prior images with jittering is not supported.")

        loglike, mean, shift = self._evaluate_log_like(flux=flux)
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
        shape_mean = self.gmm.means.shape
        npix = int(sqrt(shape_mean[-1]))
        return npix, npix

    def _evaluate_log_like(self, flux):
        normed = self.norm(flux)

        if self.cycle_spin:
            normed, shifts = cycle_spin(
                image=normed, patch_shape=self.patch_shape, generator=self.generator
            )

        if self.cycle_spin_subpix:
            normed = cycle_spin_subpixel(image=normed, generator=self.generator)

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
        loglike = self.gmm.estimate_log_prob_torch(patches)
        return loglike, mean, shifts

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
        loglike, _, _ = self._evaluate_log_like(flux=flux)
        max_loglike = torch.max(loglike, dim=1)
        return torch.sum(max_loglike.values) * self.log_like_weight / flux.numel()


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
    weights : list of floats
        Weight to be applied per level.
    cycle_spin : bool
        Apply cycle spin.
    anti_alias : bool
        Apply Gaussian smoothing before downsampling.
    """

    def __init__(
        self, prior, n_levels=2, weights=None, cycle_spin=False, anti_alias=False
    ):
        super().__init__()
        self.n_levels = n_levels
        self.cycle_spin = cycle_spin
        self.prior = prior

        if weights is None:
            weights = [1 / n_levels] * n_levels

        self.weights = weights
        self.anti_alias = anti_alias

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

            if self.anti_alias:
                factor = 2**idx
                sigma = 2 * factor / 6.0
                kernel = Gaussian2DKernel(sigma).array[None, None]
                kernel = torch.from_numpy(kernel.astype(np.float32))
                flux = convolve_fft_torch(flux, kernel=kernel)

            flux_downsampled = F.avg_pool2d(flux, kernel_size=factor)
            log_like_level = self.prior(flux=flux_downsampled)
            log_like += factor**2 * weight * log_like_level

        return log_like
