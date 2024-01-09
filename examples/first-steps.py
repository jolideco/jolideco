"""
First Steps with Jolideco
=========================

In this short tutorial we will demonstrate how to use Jolideco to
perform image deconvolution using uniform priors. We will motivate the
usage of the patch prior.

"""

import numpy as np
from jolideco import MAPDeconvolver
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
# 
# To deconvolve the image we first create a model component with a random
# initial guess:
# 

flux_init = random_state.gamma(30, size=(32, 32))

component = SpatialFluxComponent.from_numpy(flux=flux_init)

plt.imshow(flux_init, origin="lower")
plt.show()


######################################################################
# Next we create the deconvolver object and define the number of epochs to
# use:
# 

deconvolver = MAPDeconvolver(n_epochs=1000)
print(deconvolver)


######################################################################
# Finally we run the deconvolution by using:
# 
# re-assign the PSF to the default flux component named "flux"
data["psf"] = {"flux": data["psf"]}
result = deconvolver.run(datasets={"obs-1": data}, components=component)


######################################################################
# We can plot the trace of the loss function and verify that the
# optimization has converged:
# 

result.peek()
plt.show()


######################################################################
# The result does not look very good. This is because we have not used any
# regularization. To do so we need to define a prior. Jolideco provides
# many different built-in priors. Here we will use the patch prior.
# For this we have to define a Gaussian mixture model (GMM) first:

gmm = GaussianMixtureModel.from_registry("gleam-v0.1")
print(gmm)

######################################################################
# Then we can define the patch prior:
#
patch_prior = GMMPatchPrior(gmm=gmm)
print(patch_prior)

######################################################################
# And redefine the model component including the prior:

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
# The result looks smoother, however the prior is to strong and
# over-smoothes the result. Now, let us pretend we had more data
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

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

flux_ref = datasets["obs-0"]["flux"]
norm = simple_norm(flux_ref, stretch="asinh", asinh_a=0.01)

axes[0].imshow(flux_ref, norm=norm, origin="lower")
axes[0].set_title("Ground Truth")

result.components["flux"].plot(ax=axes[1], norm=norm)
axes[1].set_title("Deconvolved")
plt.show()


######################################################################
# You can also write the result to a file:
# 

result.write("jolideco-result.fits", overwrite=True)
