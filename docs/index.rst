.. image:: jolideco-logo.png
    :width: 500
    :align: center

|


.. image:: https://github.com/jolideco/jolideco/actions/workflows/ci_tests.yml/badge.svg?style=flat
    :target: https://github.com/jolideco/jolideco/actions
    :alt: GitHub actions CI

.. image:: https://img.shields.io/badge/community-Github%20Discussions-violet
   :target: https://github.com/jolideco/jolideco/discussions

.. image:: https://img.shields.io/pypi/v/jolideco

.. raw:: html

   <hr>

Welcome to Jolideco's documentation! Jolideco [#]_ is a **Python library for Joint Likelihood 
deconvolution** of a set of observations  in the presence of Poisson noise.
It can be used to **combine data from multiple x-ray instruments**
such as `Chandra <https://cxc.harvard.edu/index.html>`_, 
`XMM-Newton <https://www.cosmos.esa.int/web/xmm-newton>`_ or **gamma-ray instruments** such as
`Fermi-LAT <https://fermi.gsfc.nasa.gov/>`_ or `H.E.S.S. <https://www.mpi-hd.mpg.de/hfm/HESS/>`_.
In general Jolideco is designed to work with data from any instrument affected by Poisson noise.
It has a nice user interface and is designed to be simple to use.

.. admonition:: Where to start?

    |:computer:| The best place to start is with the :doc:`user/installation` and then continue with the :doc:`user/tutorials/index`.
    
    |:book:| If you are interested in the details and ideas behind Jolideco, check out the :doc:`user/overview` section.
    
    |:bug:| If you find a bug or have a feature request, please `open an issue <https://github.com/jolideco/jolideco/issues/new/choose>`_ .
    For general support and questions you can also use the `GitHub Discussions <https://github.com/jolideco/jolideco/discussions>`_.

    |:point_left:| Use the sidebar to navigate through the documentation.

.. [#] "Jolideco" is short for "(Jo)int (Li)kelihood (Deco) onvolution" and means "pretty decoration" in French. 


|

.. toctree::
  :caption: Table of Contents:
  :maxdepth: 2

  user/index.rst
  user/tutorials/index
  api-reference/index.rst
  developer.rst
  references.rst
  changelog.rst
