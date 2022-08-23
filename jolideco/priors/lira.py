import torch
from torch.distributions import Dirichlet

from jolideco.utils.torch import view_as_overlapping_patches_torch

from .core import Prior


class LIRAPrior(Prior):
    """LIRA multiscale prior

    Parameters
    ----------
    """

    def __init__(self, alphas, random_state=None):
        self.alphas = alphas
        self.random_state = random_state

    def cycle_spin(self, flux):
        """Cycle spin"""
        shift_x, shift_y = self.random_state, 0
        shifted = torch.roll(flux, shifts=(shift_y, shift_x))
        return shifted

    def multiscale(self):
        pass

    def __call__(self, flux):
        flux = self.cycle_spin(flux)

        log_prior = 0

        for alpha in self.alphas:
            # TODO: add downsampling...
            patches = view_as_overlapping_patches_torch(flux, shape=(2, 2), stride=2)
            patches = patches / torch.sum(patches, dim=1, keepdims=True)
            dirichlet = Dirichlet(patches)
            values = dirichlet.log_prob(...)
            log_prior += torch.sum(values)

        return log_prior
