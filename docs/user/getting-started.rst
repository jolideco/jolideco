***************
Getting Started
***************


Data Reduction
==============

Jolideco is a deconvolution package for any astronomical image affected by Poisson noise.
It is not designed to work with any specific instrument, but rather a combination of
instruments. However this also means Jolideco only operates on already reduced data,
typically as `FITS images <https://en.wikipedia.org/wiki/FITS>`_.

So before you can run your first Jolideco analysis, for each observation
you need to have the following data ready:

===================== =================================================
Quantity              Definition
===================== =================================================
counts                An image containing the binned events
psf                   An image with a model of the point spread function (PSF)
exposure (optional)   An image containing the exposure, typically as product of the lifetime and effective area
background (optional) An image of the instrumental background, typically as predicted counts
===================== =================================================

As Jolideco does no provide any data reduction functionality you have to
reduce the data using the software provided by the observatory you got
the data from. However to simplify the process for Fermi-LAT and Chandra 
you can use the following snakemake pipelines:

- `Chandra Snakemake Workflow <https://github.com/adonath/snakemake-workflow-chandra>`_
- `Fermi-LAT Snakemake Workflow <https://github.com/adonath/snakemake-workflow-fermi-lat>`_

Both workflows will produce the required image files for Jolideco.

For TeV gamma-ray data reduction you can use `Gammapy <https://gammapy.org>`_.
Especially check out the `Tutorial on image data reduction <https://docs.gammapy.org/1.1/tutorials/analysis-2d/modeling_2D.html#sphx-glr-tutorials-analysis-2d-modeling-2d-py>`_.

Usage
=====
This is how to use Jolideco:

.. code::

    import numpy as np
    from jolideco import MAPDeconvolver
    from jolideco.models import FluxComponent
    from jolideco.data import point_source_gauss_psf

    data = point_source_gauss_psf()
    component = FluxComponent.from_numpy(
        flux=np.ones((32, 32))
    )
    deconvolve = MAPDeconvolver(n_epochs=1_000)
    result = deconvolve.run(data=data, components=component)


The ``MAPDeconvolver`` is the main API of Jolideco. It runs the reconstruction 
algorithm and returns a ``MapDeconvolverResult`` object.

The main data types and classes are:

- ``FluxComponents``: a collection of flux components, they hold the model parameters.
- ``NPPredCalibrations``: a collection of calibration models, that hold the parameters
  for the calibratuion, including the background norm as well as positional shift per 
  observation.
- ``data``: a list of dictionaries with the required data for each observation (see below)


The ``data`` object is a simple Python ``dict`` containing the following quantities:

===================== =================================================
Quantity              Definition
===================== =================================================
counts                2D Numpy array containing the counts image
psf                   2D Numpy array containing an image of the PSF
exposure (optional)   2D Numpy array containing the exposure image
background (optional) 2D Numpy array containing the background / baseline image
===================== =================================================

From these quantities the predicted number of counts is computed like:

.. math::

    N_{Pred} = \mathrm{PSF} \circledast (\mathcal{E} \cdot (F + B))

Where :math:`\mathcal{E}` is the exposure, :math:`F` the deconvovled
flux image, :math:`B` the background and :math:`PSF` the PSF image.

For more detailed analysis example check out the `tutorials page`_.


Patch Prior Library
===================
For reconstruction Jolideco relies on the patch prior. The patch prior is a learned from
astronomical images at other wavelengths. The patch distribution is parametrized by a
Gaussian mixture model (GMM). During optimization Jolideco also adapts the Prior
parameters to the data. For convenience we provide a set of pre-trained GMM priors
to use:

.. list-table:: Pre-trained GMM priors
   :widths: 25 25 25 50
   :header-rows: 1

   * - Name
     - Data Origin
     - GMM components
     - Analysis Scenario
   * - `"gleam"``
     - GLEAM Survey
     - 128
     - Multipurpose, Galactic Structure
   * - `"jwst-cas-a"``
     - JWST PR image of Cas A
     - 128
     - Multipurpose, Galactic Structure, SNRs
   * - `"zoran-weiss"``
     - Zoran et al. 2011, every day images
     - 256
     - No point sources, not recommended to use. But might work for extended structures.
    

To make them available, just follow the installation instructions below.

Tips and Tricks
===============

- Set `beta` such that the prior is not too strong. Aim for a value that makes the prior around 20% of the likelihood.
- You can check the prior strength by looking at result objects `.trace_loss`
- Start with the `gleam-v0.1` prior which is a good all purpose prior.
- For bright sources you can work with an oversampled flux image. Typically a 2x2 oversampling is sufficient.
