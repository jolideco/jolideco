from math import sqrt
from astropy.utils import lazyproperty
import torch

__all__ = [
    "view_as_overlapping_patches_torch",
    "GMMPatchPrior",
]


def view_as_overlapping_patches_torch(image, shape):
    """View tensor as overlapping rectangular patches

    Parameters
    ----------
    image : `~torch.Tensor`
        Image tensor
    shape : tuple
        Shape of the patches.

    Returns
    -------
    patches : `~torch.Tensor`
        Tensor of overlapping patches of shape
        (n_patches, patch_shape_flat)

    """
    step = shape[0] // 2
    patches = image.unfold(2, shape[0], step)
    patches = patches.unfold(3, shape[0], step)
    ncols = shape[0] * shape[1]
    return torch.reshape(patches, (-1, ncols))


class GMMPatchPrior:
    """Patch prior

    Attributes
    ----------
    gmm : `GaussianMixtureModel`
        Gaussian mixture model.
    """
    def __init__(self, gmm):
        self.gmm = gmm

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
        """
        # TODO: how to norm? per patch?
        normed = flux / flux.max()
        patches = view_as_overlapping_patches_torch(
            image=normed, shape=self.patch_shape
        )
        mean = torch.mean(patches, dim=1, keepdims=True)
        patches = patches - mean
        loglike = self.gmm.estimate_log_prob_torch(patches)
        max_loglike = torch.max(loglike, dim=1)
        return torch.sum(max_loglike.values)
