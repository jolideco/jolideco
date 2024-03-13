********
Overview
********

.. _jolideco_illustration:
.. figure:: ../jolideco-illustration.png
    :width: 600
    :alt: Jolideco illustration
    :align: center
    
    Illustration of the Jolideco method. The input data are multiple images of the observed counts,
    a model of the PSF and the exposure map. The output is a single reconstructed flux image.


Given a set of observations from the same region of the sky, Jolideco can reconstruct
a single flux image from all observations. The assumption is that the underlying flux
does not change between the observations. This includes, but is not limited,
to the following scenarios:

* Different observations with one instrument or telescope at different times and observation conditions.
  For example, multiple observations of Chandra or an IACT of the same astrophysical object with different
  offset angles and exposure times
* Different observations from different instruments or telescopes, which operate in the same wavelength range.
  For example, a Chandra and XMM observation of the same region in the sky
* A single observation with one telescope with different data quality categories and different associated
  instrument response functions, such as event classes for Fermi.
* In principle also images at different energies, for example from the Fermi-LAT. However in this case
  the assumption is that the flux does not change with energy, which is not always fulfilled.

For each of the observations **Jolideco can take the specific instrument response functions into account**.
The approach is illustrated in the figure above. Using all available data Jolideco then reconstructs 
a common flux image using a maximum a posterior estimate. This requires a model of the point spread 
function (PSF) of the instrument and optionally and estimate for the exposure and background.

A naive reconstruction of the flux, for example using the `Richardson Lucy algorithm <https://en.wikipedia.org/wiki/Richardson–Lucy_deconvolution>`_,
results in very `spiky and poor images <_images/sphx_glr_first-steps_003.png>`_. So the most important part of the method is the
undelyong prior assumption on the structure of the images. For this Jolideco uses a
patch prior :cite:p:`Zoran2011,Bouman2016`  and the prior is learned from
astronomical images at other wavelength. 

If you are interested in more details of the method, please have a look at the `Jolideco paper <https://github.com/adonath/jolideco-paper/raw/main-pdf/ms.pdf>`_.