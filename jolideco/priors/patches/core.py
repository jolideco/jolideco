import logging
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from astropy.convolution import Gaussian2DKernel
from astropy.utils import lazyproperty

from jolideco.priors.patches.gmm import GaussianMixtureModel
from jolideco.utils.norms import IdentityImageNorm, ImageNorm
from jolideco.utils.numpy import reconstruct_from_overlapping_patches
from jolideco.utils.torch import (
    TORCH_DEFAULT_DEVICE,
    convolve_fft_torch,
    cycle_spin,
    cycle_spin_subpixel,
    get_default_generator,
    view_as_overlapping_patches_torch,
    view_as_random_overlapping_patches_torch,
)

from ..core import Prior

__all__ = ["GMMPatchPrior", "MultiScalePrior"]

log = logging.getLogger(__name__)


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
    patch_norm : `~jolideco.utils.PatchNorm`
        Patch normalisation.
    jitter : bool
        Jitter patch positions.
    device : `~pytorch.Device`
        Pytorch device
    """

    def __init__(
        self,
        gmm=None,
        stride=None,
        cycle_spin=True,
        cycle_spin_subpix=False,
        generator=None,
        norm=IdentityImageNorm(),
        patch_norm=None,
        jitter=False,
        marginalize=False,
        device=TORCH_DEFAULT_DEVICE,
    ):
        super().__init__()

        if gmm is None:
            gmm = GaussianMixtureModel.from_registry(name="zoran-weiss")

        self.gmm = gmm

        if stride is None:
            stride = gmm.meta.stride

        self.stride = stride
        self.cycle_spin = cycle_spin

        if generator is None:
            generator = get_default_generator(device=device)

        self.generator = generator
        self.norm = norm

        if patch_norm is None:
            patch_norm = gmm.meta.patch_norm

        self.patch_norm = patch_norm

        self.jitter = jitter
        self.cycle_spin_subpix = cycle_spin_subpix
        self.marginalize = marginalize
        self.device = torch.device(device)

    def to_dict(self):
        """To dict"""
        data = super().to_dict()
        data["stride"] = int(self.stride)
        data["cycle_spin"] = bool(self.cycle_spin)
        data["cycle_spin_subpix"] = bool(self.cycle_spin_subpix)
        data["jitter"] = bool(self.jitter)
        data["gmm"] = self.gmm.to_dict()
        data["norm"] = self.norm.to_dict()
        data["patch_norm"] = self.patch_norm.to_dict()
        data["device"] = str(self.device)
        return data

    @classmethod
    def from_dict(cls, data):
        """Create from dict"""
        kwargs = data.copy()

        gmm_config = kwargs.pop("gmm")
        gmm = GaussianMixtureModel.from_dict(gmm_config)
        kwargs["gmm"] = gmm

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
        shape_mean = self.gmm.means_numpy.shape
        npix = int(sqrt(shape_mean[-1]))
        return npix, npix

    def _evaluate_log_like(self, flux, mask=None):
        normed = self.norm(flux)
        shifts = (0, 0)

        if self.cycle_spin:
            normed, shifts = cycle_spin(
                image=normed, patch_shape=self.patch_shape, generator=self.generator
            )

        if self.cycle_spin_subpix:
            normed = cycle_spin_subpixel(image=normed, generator=self.generator)
            # TODO: how to compute the sub-pixel shift here?

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

        # filter patches with zero flux
        # TODO: use mask for this....
        selection = torch.all(patches > -1e5, dim=1, keepdims=False)
        patches = patches[selection, :]

        patches = self.patch_norm(patches)
        loglike = self.gmm.estimate_log_prob(patches)
        return loglike, None, shifts

    @lazyproperty
    def log_like_weight(self):
        """Log likelihood weight"""
        return self.stride**2 / np.multiply(*self.patch_shape)

    def __call__(self, flux, mask=None):
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
        loglike, _, _ = self._evaluate_log_like(flux=flux, mask=mask)

        if self.marginalize:
            values = torch.logsumexp(loglike, dim=1)
        else:
            values = torch.max(loglike, dim=1).values
        return torch.sum(values) * self.log_like_weight / flux.numel()


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
        self, prior, n_levels=2, weights=None, cycle_spin=True, anti_alias=True
    ):
        super().__init__()
        self.n_levels = n_levels
        self.cycle_spin = cycle_spin
        self.prior = prior

        if weights is None:
            weights = torch.tensor([1 / n_levels] * n_levels)

        self._log_weights = torch.nn.Parameter(torch.log(weights))
        self.anti_alias = anti_alias

    @property
    def weights(self):
        """Weights"""
        return torch.exp(self._log_weights) / torch.sum(torch.exp(self._log_weights))

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
            factor = 2**idx

            if weight == 0:
                continue

            if self.anti_alias:
                sigma = 2 * factor / 6.0
                kernel = Gaussian2DKernel(sigma).array[None, None]
                kernel = torch.from_numpy(kernel.astype(np.float32))
                flux = convolve_fft_torch(flux, kernel=kernel)

            flux_downsampled = F.avg_pool2d(flux, kernel_size=factor)
            log_like_level = self.prior(flux=flux_downsampled)
            log_like += factor**2 * weight * log_like_level

        return log_like

    def to_dict(self):
        """Convert to dict"""
        data = dict(
            n_levels=self.n_levels,
            weights=self.weights.detach().numpy().tolist(),
            cycle_spin=self.cycle_spin,
            anti_alias=self.anti_alias,
            prior=self.prior.to_dict(),
        )
        return data
