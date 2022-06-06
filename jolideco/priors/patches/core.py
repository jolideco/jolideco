from math import sqrt
from astropy.utils import lazyproperty
import torch
from jolideco.utils.torch import view_as_overlapping_patches_torch


__all__ = ["GMMPatchPrior"]


class GMMPatchPrior:
    """Patch prior

    Attributes
    ----------
    gmm : `GaussianMixtureModel`
        Gaussian mixture model.
    stride : int
        Stride of the patches. By default it is half of the patch size.
    """

    def __init__(self, gmm, stride=None):
        self.gmm = gmm
        self.stride = stride

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
            image=normed, shape=self.patch_shape, stride=self.stride
        )
        mean = torch.mean(patches, dim=1, keepdims=True)
        patches = patches - mean
        loglike = self.gmm.estimate_log_prob_torch(patches)
        max_loglike = torch.max(loglike, dim=1)
        return torch.sum(max_loglike.values)
