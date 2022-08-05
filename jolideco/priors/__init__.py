from .patches import GMMPatchPrior, GaussianMixtureModel, MultiScaleGMMPatchPrior
from .core import UniformPrior, ImagePrior, SmoothnessPrior
from .lira import LIRAPrior

PRIOR_REGISTRY = {
    "uniform": UniformPrior,
    "image": ImagePrior,
    "gmm-patches": GMMPatchPrior,
    "multiscale-gmm-patches": MultiScaleGMMPatchPrior,
    "smooth": SmoothnessPrior,
    "lira": LIRAPrior,
}

__all__ = [
    "GaussianMixtureModel",
    "GMMPatchPrior",
    "MultiScaleGMMPatchPrior",
    "UniformPrior",
    "SmoothnessPrior",
    "ImagePrior",
    "LIRAPrior",
]
