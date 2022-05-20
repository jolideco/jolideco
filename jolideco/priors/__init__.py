from .gmm import *
from .patches import *
from .uniform import *


PRIOR_REGISTRY = {
    "uniform": UniformPrior,
    "image": ImagePrior,
    "gmmp-patches": GMMPatchPrior
}

