from .core import (
    ExponentialPrior,
    ImagePrior,
    InverseGammaPrior,
    Priors,
    SmoothnessPrior,
    UniformPrior,
)
from .lira import LIRAPrior
from .patches import GaussianMixtureModel, GMMPatchPrior, MultiScalePrior

PRIOR_REGISTRY = {
    "uniform": UniformPrior,
    "gmm-patches": GMMPatchPrior,
    "smooth": SmoothnessPrior,
    "inverse-gamma": InverseGammaPrior,
    "exponential": ExponentialPrior,
    # TODO: those are currently not fully supported, implement if needed...
    #    "image": ImagePrior,
    #    "multiscale-prior": MultiScalePrior,
    #    "lira": LIRAPrior,
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
