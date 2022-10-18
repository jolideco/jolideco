import torch
from torch.distributions import Dirichlet
from jolideco.utils.torch import cycle_spin, view_as_overlapping_patches_torch
from .core import Prior


class LIRAPrior(Prior):
    """LIRA multiscale prior

    Parameters
    ----------
    alphas : list of float
        Alpha values
    """

    def __init__(self, alphas, cycle_spin=True, random_state=None, generator=None):
        self.alphas = alphas
        self.random_state = random_state
        self.cycle_spin = cycle_spin

        if generator is None:
            generator = torch.Generator()

        self.generator = generator

    def __call__(self, flux):
        if self.cycle_spin:
            flux = cycle_spin(image=flux, patch_shape=(2, 2), generator=self.generator)

        log_prior = 0

        for alpha in self.alphas:
            # TODO: add downsampling...
            patches = view_as_overlapping_patches_torch(flux, shape=(2, 2), stride=2)
            patches = patches / torch.sum(patches, dim=1, keepdims=True)
            dirichlet = Dirichlet(patches)
            values = dirichlet.log_prob(...)
            log_prior += torch.sum(values)

        return log_prior

    def to_dict(self):
        """To dict"""
        raise NotImplementedError
