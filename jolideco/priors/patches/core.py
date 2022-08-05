from math import sqrt
from astropy.utils import lazyproperty
import torch
import torch.nn.functional as F
from jolideco.utils.torch import view_as_overlapping_patches_torch
from jolideco.utils.norms import MaxImageNorm
from ..core import Prior


__all__ = ["GMMPatchPrior", "MultiScaleGMMPatchPrior"]


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
    image : `~pytorch.Tensor`
        Shifted tensor
    """

    x_max, y_max = patch_shape
    shift_x = torch.randint(0, x_max // 2, (1,), generator=generator)
    shift_y = torch.randint(0, y_max // 2, (1,), generator=generator)
    shifts = (int(shift_x), int(shift_y))
    return torch.roll(image, shifts=shifts, dims=(2, 3))


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

    @lazyproperty
    def patch_shape(self):
        """Patch shape (tuple)"""
        shape_mean = self.gmm.means.shape
        npix = int(sqrt(shape_mean[-1]))
        return npix, npix

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
        normed = self.norm(flux)

        if self.cycle_spin:
            normed = cycle_spin(
                image=normed, patch_shape=self.patch_shape, generator=self.generator
            )

        patches = view_as_overlapping_patches_torch(
            image=normed, shape=self.patch_shape, stride=self.stride
        )

        mean = torch.mean(patches, dim=1, keepdims=True)
        patches = patches - mean
        loglike = self.gmm.estimate_log_prob_torch(patches)
        max_loglike = torch.max(loglike, dim=1)
        return torch.sum(max_loglike.values)


class MultiScaleGMMPatchPrior(Prior):
    """Multiscale patch prior

    Parameters
    ----------
    patch_prior : `GMMPatchPrior`
        Patch prior instance
    n_levels : int, optional
        Number of multiscale levels
    """

    def __init__(self, patch_prior, n_levels=2, weights=None, cycle_spin=True):
        super().__init__()
        self.n_levels = n_levels

        if patch_prior.cycle_spin:
            raise ValueError("Cycle spin on GMMPatchPrior must be false.")

        self.patch_prior = patch_prior
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
                patch_shape=self.patch_prior.patch_shape,
                generator=self.patch_prior.generator,
            )

        for idx, weight in enumerate(self.weights):
            flux_downsampled = F.avg_pool2d(flux, kernel_size=2**idx)
            log_like_level = self.patch_prior(flux=flux_downsampled)
            log_like += weight * log_like_level

        return log_like
