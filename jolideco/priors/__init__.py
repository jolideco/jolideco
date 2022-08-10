from .patches import GMMPatchPrior, GaussianMixtureModel, MultiScalePrior
from .core import UniformPrior, ImagePrior, SmoothnessPrior, PointSourcePrior
from .lira import LIRAPrior

PRIOR_REGISTRY = {
    "uniform": UniformPrior,
    "image": ImagePrior,
    "gmm-patches": GMMPatchPrior,
    "multiscale-prior": MultiScalePrior,
    "smooth": SmoothnessPrior,
    "lira": LIRAPrior,
    "point": PointSourcePrior,
}

__all__ = [
    "GaussianMixtureModel",
    "GMMPatchPrior",
    "MultiScalePrior",
    "UniformPrior",
    "SmoothnessPrior",
    "ImagePrior",
    "LIRAPrior",
    "PointSourcePrior",
]
