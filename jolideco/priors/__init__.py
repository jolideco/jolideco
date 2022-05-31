from .gmm import *
from .patches import *
from .core import *
from .lira import *


PRIOR_REGISTRY = {
    "uniform": UniformPrior,
    "image": ImagePrior,
    "gmm-patches": GMMPatchPrior,
    "smooth": SmoothnessPrior,
}

