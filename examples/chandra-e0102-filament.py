"""
Chandra Data Analysis With Jolideco
===================================

In this tutorial we will demonstrate how to use Jolideco together with an example
Chandra dataset to perform image deconvolution. We will use 24 observations of 
a small region with an elomgated filament of the supernova remnant E0102. The
dataset is available from Zenodo: https://zenodo.org/records/10849740

To prepare Chandra data for analysis with Joldideco you can use the
following workflow: https://github.com/adonath/snakemake-workflow-chandra

A similar analysis from the Jolideco paper can be found in the following repository:

https://github.com/jolideco/jolideco-chandra-e0102-zoom-a


Let's start with the following imports:
"""

import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.visualization import simple_norm

from jolideco.core import MAPDeconvolver, MAPDeconvolverResult
from jolideco.models import (
    FluxComponents,
    NPredCalibration,
    NPredCalibrations,
    SpatialFluxComponent,
)
from jolideco.priors import GaussianMixtureModel, GMMPatchPrior
from jolideco.utils.norms import IdentityImageNorm

random_state = np.random.RandomState(428723)


URL = "https://zenodo.org/records/10849740/files/chandra-e0102-filament-all.tar.gz"

# Run decomvolution or use precomputed result
RUN_DECONVOLUTION = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"

######################################################################
# First we download and extract the data files:

path = Path("").absolute() / "chandra-e0102-filament-all"

if not path.exists():
    filename = download_file(URL, cache=True)
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path.parent, filter=lambda member, path: member)


######################################################################
# Next we extract the observation IDs from the filenames:

filenames_counts = path.glob("e0102-zoom-a-*-counts.fits")

obs_ids = [int(filename.stem.split("-")[3]) for filename in filenames_counts]

######################################################################
# The exampled dataset contains 24 observations. We will use the first
# observation as a reference for the calibration. By convention the reference
# observation has an exposure of unity, while all the other observations
# have an exposure that is relative to the reference observation.

obs_id_ref = 8365

######################################################################
# Next we load the data and create a dictionary with the datasets:


def read_data(filename, dtype=np.float32):
    return fits.getdata(path / filename).astype(dtype)


datasets = {}

for obs_id in obs_ids:
    dataset = {}
    dataset["counts"] = read_data(f"e0102-zoom-a-{obs_id}-counts.fits")

    psf_slice = slice(254 - 64, 254 + 64)
    psf = read_data(f"e0102-zoom-a-{obs_id}-e0102-zoom-a-marx-psf.fits")
    dataset["psf"] = {"filament-flux": psf[psf_slice, psf_slice]}

    dataset["exposure"] = read_data(f"e0102-zoom-a-{obs_id}-exposure.fits")
    datasets[f"obs-{obs_id}"] = dataset


######################################################################
# Let's plot the counts for each observation:

fig, axes = plt.subplots(4, 6, figsize=(12, 8))

for ax, (name, dataset) in zip(axes.flat, datasets.items()):
    ax.imshow(dataset["counts"], origin="lower")
    ax.set_title(name)

plt.tight_layout()
plt.show()


######################################################################
# As you can see the number of counts vary between the observations.
# Indicating the different exposure times.
#
# Now we can plot the PSF images as well:

fig, axes = plt.subplots(4, 6, figsize=(12, 8))

for ax, (name, dataset) in zip(axes.flat, datasets.items()):
    psf = dataset["psf"]["filament-flux"]
    norm = simple_norm(psf, stretch="log")
    ax.imshow(psf, origin="lower", norm=norm)
    ax.set_title(name)

plt.tight_layout()
plt.show()

######################################################################
# We can see again that the PSF varies between the observations. However
# this is something we can handle with Jolideco.
#
# In addition to the counts, PSF and exposure we will also
# provide a background. For now we will just use a constant background:

for dataset in datasets.values():
    dataset["background"] = 0.1 * np.ones_like(dataset["counts"])


######################################################################
# To run Jolideco we first need to define the Gaussian Mixture Model (GMM)
# to be used with patch prior. As we have sufficient data we can use the
# the GMM learned from the JWST Cas A image, which imposes rather strong
# correlation between the pixels.

gmm = GaussianMixtureModel.from_registry("jwst-cas-a-v0.1")
gmm.meta.stride = 4
print(gmm)

######################################################################
# For illustration we can also plot the mean images of the GMM:

gmm.plot_mean_images(ncols=16, figsize=(12, 8))
plt.show()

######################################################################
# Now we can define the patch prior:

patch_prior = GMMPatchPrior(
    gmm=gmm,
    cycle_spin=True,
    norm=IdentityImageNorm(),
    device=device,
)

######################################################################
# We have specified to use cycle spinning, which is a technique to
# reduce the impact of the fixed patch grid on the result.

shape = datasets[f"obs-{obs_id_ref}"]["counts"].shape
flux_init = random_state.normal(loc=3, scale=0.01, size=shape).astype(np.float32)

######################################################################
# Now we can define the spatial flux component. More specifically
# we defined the initial flux, the prior and the upsampling factor.
# We also specify that the internal flux representation is in log space.

component = SpatialFluxComponent.from_numpy(
    flux=flux_init,
    prior=patch_prior,
    use_log_flux=True,
    upsampling_factor=2,
)

components = FluxComponents()
components["filament-flux"] = component

print(components)

######################################################################
# When working with a real dataset it is important to "calibrate" the
# expected number of counts. This is done by defining a calibration
# model for each dataset. This model include three additional parameters
# that are used to adjust the expected number of counts. The background
# normalization and absolute shift in the x and y direction.

calibrations = NPredCalibrations()

for name in datasets:
    calibration = NPredCalibration(background_norm=1.0, frozen=False)
    calibrations[name] = calibration

######################################################################
# We freeze the shift parameters for the reference observation:

calibrations[f"obs-{obs_id_ref}"].shift_xy.requires_grad = False

print(calibrations)

######################################################################
# And we define the deconvolver. We will use a learning rate of 0.1 and
# a beta of 1.0. The number of epochs is set to 250. We also specify
# that the computation should be done on the CPU. If you have a GPU
# you can set the device to "cuda:0" or any other valid PyTorch device.

deconvolve = MAPDeconvolver(n_epochs=250, learning_rate=0.1, beta=1.0, device=device)
print(deconvolve)

######################################################################
# Now we can run the deconvolution. This will take a while (~30 min. on an
# M1 cpu), so we will not run it in this notebook. But if you have GPU
# acceleration it should not take more than a few minutes.

filename_result = path / "chandra-e0102-filament-jolideco.fits"

if RUN_DECONVOLUTION:
    result = deconvolve.run(
        components=components,
        calibrations=calibrations,
        datasets=datasets,
    )
    result.write(filename_result, overwrite=True)

######################################################################
# It is very good practice to always write the result to disk, after
# running the deconvolution. This is especially important for large
# datasets, as it allows to continue the analysis at a later time.
#
# Thus we just continue with the precomputed result:

result = MAPDeconvolverResult.read(filename_result)


######################################################################
# Now we can plot the result:

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

counts_all = np.sum([_["counts"] for _ in datasets.values()], axis=0)

axes[0].imshow(counts_all, origin="lower")
axes[0].set_title("Counts")

axes[1].imshow(result.flux_total, origin="lower")
axes[1].set_title("Flux Jolideco")

plt.show()

######################################################################
# The result looks very promising. The filament is clearly visible in the
# deconvolved image. However, the result is not perfect. There are still
# some artifacts visible in the deconvolved image, which comes from the
# fact that there is an individual shift per observation and currently
# there is no dedicated boundary handling for this.
