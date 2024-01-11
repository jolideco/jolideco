"""
First Steps with Jolideco
=========================

In this short tutorial we will demonstrate how to use Jolideco to
perform image deconvolution using uniform prior and a patch prior.
This will motivate the usage of the patch prior and set the stage for
later more complex examples.
"""

import numpy as np
from jolideco import MAPDeconvolver, MAPDeconvolverResult
from jolideco.models import SpatialFluxComponent
from jolideco.data import gauss_and_point_sources_gauss_psf
from jolideco.utils.plot import plot_example_dataset
from jolideco.priors import GMMPatchPrior, GaussianMixtureModel
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

random_state = np.random.RandomState(428723)


######################################################################
# We first create some synthetic data:
# 

data = gauss_and_point_sources_gauss_psf(random_state=random_state, source_level=5000)
print(data.keys())


######################################################################
# The data variable is a dict containing the `"counts"`, `"psf"`,
# `"exposure"`, `"background"` and ground truth for flux. We can
# illustrate the data using:
# 

plot_example_dataset(data)


######################################################################
# The example data contains four point sources, each with a different flux
# and an extended source with a Gaussian shape in the center of the image.
# The point spread function (PSF) is a Gaussian with a sigma of 2 pixels.
# The important thing to note here is that the image contains both point
# sources and an extended source. This is a typical situation in real
# astronomical images and we will use this example to demonstrate how to
# this challenges the deconvolution process.
# 
# To deconvolve the image we first create a model component with a random
# initial guess:
# 

flux_init = random_state.gamma(30, size=(32, 32))

component = SpatialFluxComponent.from_numpy(flux=flux_init)

plt.imshow(flux_init, origin="lower")
plt.show()


######################################################################
# The model component is an image with the same shape as the data.
#
# Next we create the deconvolver object and define the number of epochs to
# use:

deconvolver = MAPDeconvolver(n_epochs=1000)
print(deconvolver)


######################################################################
# We use the convention common in machine learning and define an epoch as
# one pass through the data. In this case we only have one observation,
# but we will later simulate multiple observations.
#
# Before running the deconvolution we have to assign the PSF to the model
# component and finally we run the deconvolution by using: 

data["psf"] = {"flux": data["psf"]}
result = deconvolver.run(datasets={"obs-1": data}, components=component)


######################################################################
# We can plot the trace of the loss function and verify that the
# optimization has converged:
# 

result.peek()
plt.show()


######################################################################
# The result does not look very convincing. The extended component 
# has decomposed into multiple point sources and the point sources
# are not well recovered. This is because we have not used any
# regularization and the correlation between the pixels is not
# taken into account.
#
# To improve on this we need to define a prior. Jolideco provides
# many different built-in priors. Here we will use the so called
# "patch prior", which is the most flexible one. The patch prior
# was learned on astronomical images and encodes the correlation
# between the pixels as patches. The distribution of the patches
# is modeled using a Gaussian mixture model (GMM). 
#
# Jolideco provides multiple builtin pre-trained GMM, which can be
# used by accessing the registry:

gmm = GaussianMixtureModel.from_registry("gleam-v0.1")
print(gmm)

######################################################################
# Here we use the GMM trained on the `GLEAM survey <https://www.mwatelescope.org/science/galactic-science/gleam/>`_.
#
# Next we can define the patch prior:
#
patch_prior = GMMPatchPrior(gmm=gmm)
print(patch_prior)

######################################################################
# Beside the choice of the GMM the patch prior has a few more
# parameters, which we will ignore for now.
#
# Finally we can redefine the model component including the prior:

component_patch_prior = SpatialFluxComponent.from_numpy(
    flux=flux_init, prior=patch_prior, upsampling_factor=1.0
)
print(component_patch_prior)

######################################################################
# Now we can run the deconvolution again:

result = deconvolver.run(datasets={"obs-1": data}, components=component_patch_prior)

######################################################################
# And plot the result:

result.peek()
plt.show()


######################################################################
# The result already looks smoother, however the prior is to strong and
# over-smoothes the result. For this we could reduce the strength of the
# prior using the beta parameter. 
#
# However for this tutorial let us pretend we had more data
# available, by simulating multiple observations:
# 

n_obs = 5
datasets = {}

for idx in range(n_obs):
    dataset = gauss_and_point_sources_gauss_psf(
        random_state=random_state, source_level=5000
    )
    # re-assign psf
    dataset["psf"] = {"flux": dataset["psf"]}
    datasets[f"obs-{idx}"] = dataset

######################################################################
# Now we re-run the deconvolution:
    
result = deconvolver.run(datasets=datasets, components=component_patch_prior)

######################################################################
# And plot again the trace of the loss function and the result:

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

flux_ref = datasets["obs-0"]["flux"]
norm = simple_norm(flux_ref, stretch="asinh", asinh_a=0.01)

counts = np.sum([_["counts"] for _ in datasets.values()], axis=0)
axes[0].imshow(counts, origin="lower")
axes[0].set_title("Data")

axes[1].imshow(flux_ref, norm=norm, origin="lower")
axes[1].set_title("Ground Truth")

result.components["flux"].plot(ax=axes[2], norm=norm)
axes[2].set_title("Jolideco Deconvolved")

plt.show()


######################################################################
# Finally we can also write the result to a file:
# 

result.write("jolideco-result.fits", overwrite=True)

######################################################################
# And we can read the result back in:
# 

result = MAPDeconvolverResult.read("jolideco-result.fits")
