**********
User Guide
**********

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
