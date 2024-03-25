"""
Fermi Data Analysis with Jolideco
=================================

In this tutorial we will demonstrate how to use Jolideco together with Gammapy
to perform image deconvolution of Fermi-LAT data.

If you start from a full Gammapy dataset, you can use the following code to reduce it to a 2D image:

https://github.com/jolideco/jolideco-fermi-examples/blob/main/workflow/scripts/prepare-datasets.py


For Fermi-LAT data in general, you can use the following workflow to prepare the data:

https://github.com/adonath/snakemake-workflow-fermi-lat

This tutorial will require to have `Gammapy installed <https://docs.gammapy.org/1.2/getting-started/index.html>`_. 

Let's start with the following imports:
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.data import download_file
from astropy.visualization import simple_norm
from gammapy.maps import Map, Maps

from jolideco import MAPDeconvolver, MAPDeconvolverResult
from jolideco.models import (
    FluxComponents,
    NPredCalibration,
    NPredCalibrations,
    NPredModels,
    SpatialFluxComponent,
)
from jolideco.priors import (
    GaussianMixtureModel,
    GMMPatchPrior,
)

RUN_DECONVOLUTION = False  # Set to False to use precomputed result

random_state = np.random.RandomState(428723)

######################################################################
# First we download the data:

URL_BASE = "https://zenodo.org/records/10856342/files/"

filenames = [
    "vela-junior-above-10GeV-data-psf0-maps.fits",
    "vela-junior-above-10GeV-data-psf1-maps.fits",
    "vela-junior-above-10GeV-data-psf2-maps.fits",
    "vela-junior-above-10GeV-data-psf3-maps.fits",
    "vela-junior-above-10GeV-jolideco-result.fits",
]

path = Path("").absolute() / "fermi-vela-junior-above-10GeV"
path.mkdir(exist_ok=True)

for filename in filenames:
    if not (path / filename).exists():
        src = download_file(URL_BASE + filename, cache=True)
        shutil.copyfile(src, path / filename)


######################################################################
# Now we load the maps into a dictionary, where the key is the dataset name
datasets = {}


for filename in filenames[:-1]:
    maps = Maps.read(path / filename)
    name = Path(filename).stem.replace("-maps", "")
    datasets[name] = maps


######################################################################
# Here we define a function to convert the Gammapy maps to plain numpy arrays
# that can be used by Jolideco and apply the function to the datasets:


def to_jolideco_dataset(maps, dtype=np.float32):
    """Convert Gammapy maps to Jolideco dataset."""
    return {
        "counts": maps["counts"].data[0].astype(dtype),
        "background": maps["background"].data[0].astype(dtype),
        "psf": {"vela-junior": maps["psf"].data[0].astype(dtype)},
        "exposure": maps["exposure"].data[0].astype(dtype),
    }


datasets_jolideco = {name: to_jolideco_dataset(maps) for name, maps in datasets.items()}

######################################################################
# Counts
# ------
# Let's plot the counts for each dataset:
wcs = datasets["vela-junior-above-10GeV-data-psf0"]["counts"].geom.wcs

fig, axes = plt.subplots(
    ncols=2, nrows=2, subplot_kw={"projection": wcs}, figsize=(9, 9)
)

for ax, (name, maps) in zip(axes.flat, datasets.items()):
    counts = maps["counts"].sum_over_axes()
    counts.plot(ax=ax, cmap="viridis", add_cbar=True)
    ax.set_title(f"{name}")

plt.show()

######################################################################
# Background
# -----------
# Let's plot the prediced counts for the background for each dataset:

fig, axes = plt.subplots(
    ncols=2, nrows=2, subplot_kw={"projection": wcs}, figsize=(9, 9)
)

for ax, (name, maps) in zip(axes.flat, datasets.items()):
    background = maps["background"].sum_over_axes()
    background.plot(ax=ax, cmap="viridis", add_cbar=True, stretch="log")
    ax.set_title(f"{name}")

plt.show()

######################################################################
# PSF
# ---
#
# And finally we plot the PSF for each dataset:
wcs = datasets["vela-junior-above-10GeV-data-psf0"]["psf"].geom.wcs

fig, axes = plt.subplots(
    ncols=2, nrows=2, subplot_kw={"projection": wcs}, figsize=(9, 9)
)

for ax, (name, maps) in zip(axes.flat, datasets.items()):
    psf = maps["psf"].sum_over_axes()
    psf.plot(ax=ax, cmap="viridis", add_cbar=True, stretch="log")
    ax.set_title(f"{name}")

plt.show()

######################################################################
# We can see that the PSF varies between the datasets. This is something
# we can handle with Jolideco.
#
# Now we will define the prior for the flux components. We will use a Gaussian
# Mixture Model (GMM) as patch prior. We will use the GMM from the Chandra SNRs.

gmm = GaussianMixtureModel.from_registry("chandra-snrs-v0.1")
gmm.stride = 4
print(gmm)

######################################################################
# Let's plot the mean images of the Gaussian Mixture Model
gmm.plot_mean_images(ncols=16, figsize=(12, 8))
plt.show()

######################################################################
# Now we define the flux components.
#
# As the Fermi data is really low statistics we will not use any
# upsampled flux components.

patch_prior = GMMPatchPrior(gmm=gmm, cycle_spin=True, stride=4)


shape = datasets_jolideco["vela-junior-above-10GeV-data-psf1"]["counts"].shape
flux_init = np.random.normal(loc=0.1, scale=0.01, size=shape).astype(np.float32)

component = SpatialFluxComponent.from_numpy(
    flux=flux_init,
    prior=patch_prior,
    use_log_flux=True,
    upsampling_factor=1,
)


components = FluxComponents()
components["vela-junior"] = component

print(components)

######################################################################
# Now we define the calibration model for each dataset

calibrations = NPredCalibrations()

for name, value in zip(datasets, [0.5, 1.2, 1.2, 1.2]):
    calibration = NPredCalibration(background_norm=value, frozen=False)
    calibrations[name] = calibration

print(calibrations)


######################################################################
# We will freeze the shift parameters for the PSF0 and Fermi-GC dataset

calibrations["vela-junior-above-10GeV-data-psf0"].shift_xy.requires_grad = False

######################################################################
# Now we define the deconvolver
deconvolver = MAPDeconvolver(n_epochs=250, learning_rate=0.1)
print(deconvolver)

######################################################################
# And run the deconvolution
filename_result = "vela-junior-above-10GeV-jolideco-result.fits"

if RUN_DECONVOLUTION:
    result = deconvolver.run(
        datasets=datasets_jolideco,
        components=components,
        calibrations=calibrations,
    )
    result.write(path / filename_result)

######################################################################
# Let's read the result from the file

result = MAPDeconvolverResult.read(path / filename_result)

######################################################################
# And plot the trace of the loss function to verify the convergence:

plt.figure(figsize=(12, 8))
result.plot_trace_loss()
plt.show()


######################################################################
# Let's plot the deconvolved image and counts for comparison:

counts = np.sum([_["counts"] for _ in datasets_jolideco.values()], axis=0)

fig, axes = plt.subplots(ncols=2, subplot_kw={"projection": wcs}, figsize=(14, 6))

norm_asinh = simple_norm(
    counts,
    min_cut=0.1,
    max_cut=0.5,
    stretch="power",
    power=1.0,
)

im = axes[0].imshow(counts, origin="lower", interpolation="None")
axes[0].set_title("Counts")
plt.colorbar(im)

im = axes[1].imshow(
    result.components.flux_total_numpy,
    origin="lower",
    interpolation="bicubic",
)
axes[1].set_title("Deconvolved")
plt.colorbar(im)
plt.show()

######################################################################
# We have achieved a good deconvolution of the Fermi data with Jolideco.
# The processed image is much sharper and less noisy than the original
# counts image!
#
# Residuals
# ---------
# To check the quality of the deconvolution we can plot the residuals
# between the counts and the predicted counts. For this we first need
# to compute the predicted counts for each dataset:

geom = datasets["vela-junior-above-10GeV-data-psf0"]["counts"].geom.to_image()

npreds = {}

for name, dataset in datasets_jolideco.items():
    model = NPredModels.from_dataset_numpy(
        dataset=dataset,
        components=result.components,
        calibration=result.calibrations[name],
    )

    fluxes = result.components.to_flux_tuple()
    npred = model.evaluate(fluxes=fluxes).detach().numpy()[0, 0]
    npreds[name] = Map.from_geom(data=npred, geom=geom)

######################################################################
# Then we proceed to plot the residuals for each dataset:


fig, axes = plt.subplots(
    ncols=2,
    nrows=2,
    subplot_kw={"projection": wcs},
    gridspec_kw={"wspace": 0.2},
    figsize=(10, 10),
)


for (name, dataset), ax in zip(datasets.items(), axes.flat):
    counts = dataset["counts"].sum_over_axes(keepdims=False).smooth(5)
    npred = npreds[name].smooth(5)

    residual = (counts - npred) / np.sqrt(npred)

    residual.plot(ax=ax, vmin=-0.5, vmax=0.5, cmap="RdBu", add_cbar=True)
    ax.set_title(f"{name}")

plt.show()

######################################################################
# The residuals are consistent with zero, indicating that the model
# is a good fit to the data.
#
# For comparison we can also plot the residuals for the non calibrated model:

npreds_non_calibrated = {}

for name, dataset in datasets_jolideco.items():
    model = NPredModels.from_dataset_numpy(
        dataset=dataset, components=result.components
    )

    fluxes = result.components.to_flux_tuple()
    npred = model.evaluate(fluxes=fluxes).detach().numpy()[0, 0]
    npreds_non_calibrated[name] = Map.from_geom(data=npred, geom=geom)


fig, axes = plt.subplots(
    ncols=2,
    nrows=2,
    subplot_kw={"projection": wcs},
    gridspec_kw={"wspace": 0.2},
    figsize=(10, 10),
)


for (name, dataset), ax in zip(datasets.items(), axes.flat):
    counts = dataset["counts"].sum_over_axes(keepdims=False).smooth(5)
    npred = npreds_non_calibrated[name].smooth(5)

    residual = (counts - npred) / np.sqrt(npred)

    residual.plot(ax=ax, vmin=-0.5, vmax=0.5, cmap="RdBu", add_cbar=True)
    ax.set_title(f"{name}")

plt.show()

######################################################################
# We can see that the residuals are much larger for the non calibrated model.
# This indicates that the calibration is important for the quality of the
# deconvolution.
#
# We hope this tutorial was useful to get you started with Fermi data analysis
# using Jolideco. If you have any questions or feedback, please don't hesitate
# to contact us on GitHub.
