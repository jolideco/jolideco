# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: jolideco-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # First Steps with Jolideco
#
# In this short tutorial we will demonstrate how to use Jolideco to perform image
# deconvolution using uniform priors. We will motivate the usage of the patch prior.
#

# %%
import numpy as np
from jolideco import MAPDeconvolver
from jolideco.models import SpatialFluxComponent
from jolideco.data import gauss_and_point_sources_gauss_psf
from jolideco.utils.plot import plot_example_dataset
from jolideco.priors import GMMPatchPrior, GaussianMixtureModel
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

random_state = np.random.RandomState(428723)

# %% [markdown]
# We first create some synthetic data:

# %%
data = gauss_and_point_sources_gauss_psf(random_state=random_state, source_level=5000)
print(data.keys())

# %% [markdown]
# The data variable is a dict containing the `"counts"`, `"psf"`, `"exposure"`,
# `"background"` and ground truth for flux. We can illustrate the data using:

# %%
plot_example_dataset(data)

# %% [markdown]
# The example data contains four point sources, each with a different flux and an
# extended source with a Gaussian shape in the center of the image. The point spread 
# function (PSF) is a Gaussian with a sigma of 2 pixels.
#
# To deconvolve the image we first create a model component with a random initial guess:

# %%
flux_init = random_state.gamma(30, size=(32, 32))

component = SpatialFluxComponent.from_numpy(flux=flux_init)

plt.imshow(flux_init)

# %% [markdown]
# Next we create the deconvolver object and define the number of epochs to use:

# %%
deconvolver = MAPDeconvolver(n_epochs=1000)
print(deconvolver)

# %% [markdown]
# Finally we run the deconvolution by using:

# %%
result = deconvolver.run(datasets={"obs-1": data}, components=component)

# %% [markdown]
# We can plot the trace of the loss function and verify that the optimization
# has converged:

# %%
result.plot_trace_loss()

# %% [markdown]
# Now we take a look at the deconvolved image:

# %%
result.components["flux"].plot()

# %% [markdown]
# The result does not look very good. This is because we have not used any
# regularization. To do so we need to define a prior.

# %%
gmm = GaussianMixtureModel.from_registry("gleam-v0.1")
print(gmm)

# %%
patch_prior = GMMPatchPrior(gmm=gmm, cycle_spin=False, cycle_spin_subpix=True)
print(patch_prior)

# %%
component_patch_prior = SpatialFluxComponent.from_numpy(
    flux=flux_init, prior=patch_prior, upsampling_factor=1.0
)
print(component_patch_prior)

# %%
result = deconvolver.run(datasets={"obs-1": data}, components=component_patch_prior)

# %%
result.plot_trace_loss()

# %%
result.components["flux"].plot()

# %% [markdown]
# The result looks smoother, however the prior is to strong and over-smoothes
# the result. Now, let us pretend we had more data available, by simulating
# multiple observations:

# %%
n_obs = 5
datasets = {}

for idx in range(n_obs):
    datasets[f"obs-{idx}"] = gauss_and_point_sources_gauss_psf(
        random_state=random_state, source_level=5000
    )

# %%
result = deconvolver.run(datasets=datasets, components=component_patch_prior)

# %%
result.plot_trace_loss()

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

flux_ref = datasets["obs-0"]["flux"]
norm = simple_norm(flux_ref, stretch="asinh", asinh_a=0.01)
axes[0].imshow(flux_ref, norm=norm, origin="lower")

result.components["flux"].plot(ax=axes[1], norm=norm)

# %% [markdown]
# You can also write the result to a file:

# %%
result.write("jolideco-result.fits")

# %%

# %% [markdown]
#
