***************
Getting Started
***************


Data Preparation
================

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
Once you have prepared the data you can use Jolideco to deconvolve it. 


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
- ``data``: a list of dictionaries with the required data for each observation (see above)


From these quantities the predicted number of counts is computed like:

.. math::

    N_{Pred} = \mathrm{PSF} \circledast (\mathcal{E} \cdot (F + B))

Where :math:`\mathcal{E}` is the exposure, :math:`F` the deconvovled
flux image, :math:`B` the background and :math:`PSF` the PSF image.

For more detailed analysis example check out the :doc:`tutorials/index`.
    
Tips and Tricks
===============

Here is a list of tips and tricks that might help you to get started with Jolideco. 
They have been collected from the experience of the developers and users of Jolideco.

- Start with a uniform prior for the flux and "overfit" to the data. Convergence should be really fast.
- This image you can use as a starting point for the patch prior later.
- The patch prior is normalized such that it is roughly equally informative as one observation.
- If you only have few data (low statistics), ``beta`` should be chosen such that the prior is not too strong. Aim for a value that makes the prior around 20% of the likelihood.
- You can check the prior strength by looking at ``MapDeconvolverResult.trace_loss``
- Start with the ``"gleam-v0.1"`` prior which is a good all purpose prior.
- For bright sources you can work with an oversampled flux image. Typically a 2x2 oversampling is sufficient.
