********
Overview
********

Given a set of observations from the same region of the sky, Jolideco can reconstruct
a single flux image from all observations. The assumption is that the underlying flux
does not change between the observations. This includes the following scenarios:

* Different observations with one instrument or telescope at different times and observation conditions.
  For example, multiple observations of Chandra or an IACT of the same astrophysical object with different
  offset angles and exposure times
* Different observations from different instruments or telescopes, which operate in the same wavelength range.
  For example, a Chandra and XMM observation of the same region in the sky
* A single observation with one telescope with different data quality categories and different associated
  instrument response functions, such as event classes for Fermi.


.. image:: ../jolideco-illustration.png
    :width: 600
    :alt: Jolideco illustration
    :align: center
    :caption: Illustration of the Jolideco method. The input data are multiple images of the observed counts,
              a model of the PSF and the exposure map. The output is a single reconstructed flux image.


Using all available data Jolideco then reconstructs the flux estimate by deconvolution of the
data. This requires a model of the point spread function (PSF) of the instrument and optionally
and estimate for the exposure and background.

A naive reconstruction of the flux, for example using the `Richardson Lucy algorithm <https://en.wikipedia.org/wiki/Richardson–Lucy_deconvolution>`_,
results in very spiky and poor images. So the most important part of the method is the
undelyong prior assumption on the structure of the images. For this Jolideco uses a
patch prior; the prior is learned from astronomical images at other wavelength. 

In general Jolideco behaves rather stable during reconstruction, however there are a few
parameters that can be tuned to improve the reconstruction.


Jolideco builds on ideas of :cite:t:`Zoran2011`, :cite:t:`Bouman2016` and :cite:t:`Ingaramo2014`.