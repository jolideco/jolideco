from math import sqrt
import numpy as np
from astropy.utils import lazyproperty
from astropy.convolution import Gaussian2DKernel
import torch
import torch.nn.functional as F
from jolideco.utils.torch import view_as_overlapping_patches_torch, convolve_fft_torch
from jolideco.utils.numpy import reconstruct_from_overlapping_patches
from jolideco.utils.norms import MaxImageNorm
from ..core import Prior


__all__ = ["GMMPatchPrior", "MultiScalePrior"]


def cycle_spin(image, patch_shape, generator):
    """Cycle spin

    Parameters
    ----------
    image : `~pytorch.Tensor`
        Image tensor
    patch_shape : tuple of int
        Patch shape
    generator : `~torch.Generator`
        Random number generator

    Returns
    -------
    image, shifts: `~pytorch.Tensor`, tuple of `~pytorch.Tensor`
        Shifted tensor
    """
    x_max, y_max = patch_shape
    x_width, y_width = x_max // 4, y_max // 4
    shift_x = torch.randint(-x_width, x_width + 1, (1,), generator=generator)
    shift_y = torch.randint(-y_width, y_width + 1, (1,), generator=generator)
    shifts = (int(shift_x), int(shift_y))
    return torch.roll(image, shifts=shifts, dims=(2, 3)), shifts


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
    """

    def __init__(self, gmm, stride=None, cycle_spin=True, generator=None, norm=None):
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
        loglike, mean, shift = self._evaluate_log_like(flux=flux)
        idx = torch.argmax(loglike, dim=1)

        eigen_images = self.gmm.eigen_images
        patches = eigen_images[idx] + mean.detach().numpy().reshape((-1, 1, 1))

        reco = reconstruct_from_overlapping_patches(
            patches=patches, image_shape=flux.shape[2:], stride=self.stride
        )
        return np.roll(reco, shift=-1 * np.array(shift), axis=(0, 1))

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
        else:
            shifts = (0, 0)

        patches = view_as_overlapping_patches_torch(
            image=normed, shape=self.patch_shape, stride=self.stride
        )

        mean = torch.mean(patches, dim=1, keepdims=True)
        patches = patches - mean
        loglike = self.gmm.estimate_log_prob_torch(patches)
        return loglike, mean, shifts

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
        return torch.sum(max_loglike.values)


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

        if not isinstance(prior, GMMPatchPrior):
            raise ValueError("Multi scale prior only supports `GMMPatchPrior`")

        if prior.cycle_spin:
            raise ValueError("`GMMPatchPrior.cycle_spin` must be false.")

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
            flux = cycle_spin(
                image=flux,
                patch_shape=self.prior.patch_shape,
                generator=self.prior.generator,
            )

        for idx, weight in enumerate(self.weights):
            if weight == 0:
                continue
            kernel = Gaussian2DKernel(2**idx).array[None, None]
            flux = convolve_fft_torch(flux, torch.from_numpy(kernel.astype(np.float32)))
            flux_downsampled = F.avg_pool2d(flux, kernel_size=2**idx)
            log_like_level = self.prior(flux=flux_downsampled)
            log_like += (2**idx) ** 2 * weight * log_like_level

        return log_like
