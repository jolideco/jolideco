************
Installation
************

Jolideco is mostly build on `Pytorch <https://pytorch.org>`_, `Numpy <https://numpy.org>`_ and `Astropy <https://www.astropy.org>`_.

It is recommended to first install Pytorch, using the instruction on the `Pytorch webpage <https://pytorch.org/get-started/locally/#start-locally>`_.


Then to install Jolideco, you can use pip:

.. code:: bash
    
    pip install jolideco

Jolideco relies on pre-train Gaussian Mixture Models (GMM) to estimate the
prior distribution of the parameters of the model. These GMM are stored in a
separate repository.

To install the patch prior library, just clone the following  repository:

.. code:: bash

    git clone https://github.com/jolideco/jolideco-gmm-prior-library.git
    export JOLIDECO_GMM_PRIOR_LIBRARY=/path/to/jolideco-gmm-prior-library

And define the environment variable ``JOLIDECO_GMM_PRIOR_LIBRARY`` to point to the
folder where you cloned the repository.