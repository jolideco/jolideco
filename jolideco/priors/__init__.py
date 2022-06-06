from .patches import GMMPatchPrior, GaussianMixtureModel
from .core import UniformPrior, ImagePrior, SmoothnessPrior
from .lira import LIRAPrior

PRIOR_REGISTRY = {
    "uniform": UniformPrior,
    "image": ImagePrior,
    "gmm-patches": GMMPatchPrior,
    "smooth": SmoothnessPrior,
    "lira": LIRAPrior,
}

__all__ = [
    "GaussianMixtureModel",
    "GMMPatchPrior",
    "UniformPrior",
    "SmoothnessPrior",
    "ImagePrior",
    "LIRAPrior",
]
