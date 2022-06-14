from math import sqrt
from astropy.utils import lazyproperty
import torch
from jolideco.utils.torch import view_as_overlapping_patches_torch
from ..core import Prior


__all__ = ["GMMPatchPrior"]


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

    def __init__(self, gmm, stride=None, cycle_spin=True, generator=None):
        super().__init__()

        self.gmm = gmm
        self.stride = stride
        self.cycle_spin = cycle_spin

        if generator is None:
            generator = torch.Generator()

        self.generator = generator

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
        normed = flux / flux.max()

        if self.cycle_spin:
            x_max, y_max = self.patch_shape
            shift_x = torch.randint(0, x_max // 2, (1,), generator=self.generator)
            shift_y = torch.randint(0, y_max // 2, (1,), generator=self.generator)
            shifts = (int(shift_x), int(shift_y))
            normed = torch.roll(normed, shifts=shifts, dims=(2, 3))

        patches = view_as_overlapping_patches_torch(
            image=normed, shape=self.patch_shape, stride=self.stride
        )
        mean = torch.mean(patches, dim=1, keepdims=True)
        patches = patches - mean
        loglike = self.gmm.estimate_log_prob_torch(patches)
        max_loglike = torch.max(loglike, dim=1)
        return torch.sum(max_loglike.values)
