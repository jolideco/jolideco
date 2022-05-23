from .gmm import *
from .patches import *
from .core import *


PRIOR_REGISTRY = {
    "uniform": UniformPrior,
    "image": ImagePrior,
    "gmmp-patches": GMMPatchPrior
}

