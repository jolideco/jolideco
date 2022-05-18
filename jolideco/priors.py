import torch
from .utils.torch import view_as_overlapping_patches_torch


class GMMPatchPrior:
    """Patch prior"""
    def __init__(self, gmm):
        self.gmm = gmm

    def __call__(self, flux):
        # TODO: how to norm? per patch?
        normed = flux / flux.max()
        patches = view_as_overlapping_patches_torch(normed, shape=(8, 8))
        mean = torch.mean(patches, dim=1, keepdims=True)
        patches = patches - mean
        loglike = self.gmm.estimate_log_prob_torch(patches)
        max_loglike = torch.max(loglike, dim=1)
        return torch.sum(max_loglike.values)


class UniformPrior:
    """Uniform prior"""
    def __init__(self):
        pass

    def __call__(self, flux):
        return torch.tensor(0)
