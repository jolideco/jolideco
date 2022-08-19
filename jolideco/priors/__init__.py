from .patches import GMMPatchPrior, GaussianMixtureModel, MultiScalePrior
from .core import (
    ExponentialPrior,
    UniformPrior,
    ImagePrior,
    SmoothnessPrior,
    InverseGammaPrior,
    Priors,
)
from .lira import LIRAPrior


PRIOR_REGISTRY = {
    "uniform": UniformPrior,
    "image": ImagePrior,
    "gmm-patches": GMMPatchPrior,
    "multiscale-prior": MultiScalePrior,
    "smooth": SmoothnessPrior,
    "lira": LIRAPrior,
    "point": InverseGammaPrior,
    "exponetial": ExponentialPrior,
}

__all__ = [
    "GaussianMixtureModel",
    "GMMPatchPrior",
    "MultiScalePrior",
    "ExponentialPrior",
    "UniformPrior",
    "SmoothnessPrior",
    "ImagePrior",
    "LIRAPrior",
    "InverseGammaPrior",
    "Priors",
]
