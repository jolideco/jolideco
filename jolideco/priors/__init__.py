from .gmm import *
from .patches import *
from .core import *


PRIOR_REGISTRY = {
    "uniform": UniformPrior,
    "image": ImagePrior,
    "gmm-patches": GMMPatchPrior,
    "smooth": SmoothnessPrior,
}

