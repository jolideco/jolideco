***********
Development
***********

This is the developer documentation.

Setup
-----
For the development you can rely on one of the pre-defined test environments::

    tox --devenv venv-jolideco-dev -e py39
    source venv-jolieco-dev/bin/activate

This will create a new ``venv-jolideco-env`` environment, that you can activate
using the ``source`` command. To leave the environment again use ``deactivate``.
The command requires that you ahve Python 3.9 installed on your system. In case
you do not have it installed you could change the command to the corresponding
Python version like::

    tox --devenv venv-jolideco-dev -e py38
    tox --devenv venv-jolideco-dev -e py37

However it is recommanded to use a rather new Python version for development.

Running Tests
-------------

Note: running tests is no longer done using ``python setup.py test``. Instead
you will need to run::

    tox -e test

If you don't already have tox installed, you can install it with::

    pip install tox

If you only want to run part of the test suite, you can also use pytest
directly with::

    pip install -e .[test]
    pytest


Building Docs
-------------

Building the documentation is no longer done using
``python setup.py build_docs``. Instead you will need to run::

    tox -e build_docs

If you don't already have tox installed, you can install it with::

    pip install tox

You can also build the documentation with Sphinx directly using::

    pip install -e .[docs]
    cd docs
    make html

