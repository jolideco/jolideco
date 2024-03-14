************
Installation
************

This page provides information on how to install Jolideco. In general it is
recommended to use a virtual environment to install Jolideco. This can be done
using `conda <https://docs.conda.io/en/latest/>`_ or `virtualenv <https://virtualenv.pypa.io/en/latest/>`_.

Then to install Jolideco, you can use pip:

.. code:: bash
    
    python -m pip install jolideco

When you encounter problems during the installation especially with 
Pytorch, it is recommended to first install Pytorch, using the 
more detailed instructions on the `Pytorch webpage <https://pytorch.org/get-started/locally/#start-locally>`_.


Jolideco relies on pre-train Gaussian Mixture Models (GMM) to estimate the
prior distribution of the parameters of the model. These GMM are stored in a
separate repository.

To install the patch prior library, just clone the following  repository:

.. code:: bash

    git clone https://github.com/jolideco/jolideco-gmm-prior-library.git
    export JOLIDECO_GMM_PRIOR_LIBRARY=/path/to/jolideco-gmm-prior-library

And define the environment variable ``JOLIDECO_GMM_PRIOR_LIBRARY`` to point to the
folder where you cloned the repository.

You can verify that the installation was successful by running the following
command:

.. code:: bash

    jolideco test

If all tests pass, the installation was successful and you are ready to go!


.. admonition:: What's next?

   |:books:| If you would like to first learn Jolideco on example datasets checkout 
   the :doc:`tutorials/index`. 

   |:rocket:| If you would like to directly start with your own data, rather checkout
   the :doc:`getting-started` page.