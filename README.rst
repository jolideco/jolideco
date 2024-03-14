.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

.. image:: https://readthedocs.org/projects/jolideco/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://jolideco.readthedocs.io/en/latest/?badge=latest

.. image:: https://github.com/jolideco/jolideco/actions/workflows/ci_tests.yml/badge.svg?style=flat
    :target: https://github.com/jolideco/jolideco/actions
    :alt: GitHub actions CI


Jolideco: a Python package for Poisson Joint Likelihood Deconvolution
---------------------------------------------------------------------

.. image:: docs/jolideco-illustration.png
    :width: 600
    :alt: Jolideco illustration
    :align: center

Jolideco is a Python package for Joint Likelihood Deconvolution of astronomical images affected by
Poisson noise. It allows you to deblur and denoise images and do a joint image reconstruction of
multiple images from different instruments, while taking their specific instrument response functions,
such as point spread functions, exposure and instrument specific background emission into account.
To ensure a high fidelity of reconstructed features in the images, Jolideco relies on a patch based
image prior, which is based on a Gaussian Mixture Model (GMM). 

Contributing Code, Documentation, or Feedback
---------------------------------------------
Jolideco is an open-source project and we welcome contributions of all kinds: 
new features, bug fixes, documentation improvements, and more. If you are interested
in contributing, please get in contact with the maintainers and make sure to read the
`Code of Conduct <https://github.com/jolideco/jolideco/blob/main/CODE_OF_CONDUCT.md>`_.

Citation
--------

When using Jolideco, please cite the following paper references:

TBD

Further Resources
------------------

Please also take a look at the following associated repositories:

- `Jolideco GMM Library <https://github.com/jolideco/jolideco-gmm-prior-library>`_
- `Jolideco Fermi-LAT Example <https://github.com/jolideco/jolideco-fermi-examples>`_
- `Jolideco Chandra Example <https://github.com/jolideco/jolideco-chandra-examples>`_
- `Webpage with Result Comparisons for Toy Datasets <https://jolideco.github.io/jolideco-comparison/>`_
- `Jolideco Performance Benchmarks <https://github.com/jolideco/jolideco-performance-benchmark>`_


Contributing
------------
While contributions are welcome in general, currently I cannot review PRs, nor help with implementations,
because of a lack of time. So PRs are unlikely to get merged. However any kind of bug report or feature
requests are welcome as well.
