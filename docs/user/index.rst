**********
User Guide
**********

Overview
========

Given a set of observations from the same region of the sky, Jolideco can reconstruct
a single flux image from all observations. The assumption is that the underlying flux
does not change between the observations.

* Different observations with one instrument or telescope at different times and observation conditions.
  For example, multiple observations of Chandra of the same astrophysical object with different offset
  angles and exposure times
* Different observations from different instruments or telescopes, which operate in the same wavelength range.
  For example, a Chandra and XMM observation of the same region in the sky
* A single observation with one telescope with different data quality categories and different associated
  instrument response functions, such as event classes for Fermi.

Using all available data Jolideco then reconstructs the flux estimate by deconvolution of the
data. This requires a model of the point spread function (PSF) of the instrument and optionally
and estimate for the exposure and background.

A naive reconstruction of the flux, for example using the Richardson Lucy algorithm,
results in very spiky and poor images. So the most important part of the method is the
undelyong prior assumption on the structure of the images. For this Jolideco uses a
patch prior; the prior is learned from astronomical images at other wavelength. 

In general Jolideco behaves rather stable during reconstruction, however there are a few
parameters that can be tuned to improve the reconstruction.


Installation
============
To install Jolideco, you can use pip:

.. code:: bash
    
    pip install jolideco


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
