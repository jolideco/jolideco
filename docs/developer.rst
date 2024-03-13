***************
Developer Guide
***************

This is the developer documentation. The following sections will guide you through
the process of setting up a development environment and running the tests.

Setup
-----

Contributions to Jolideco should happen through pull requests on GitHub. 
So if you have not done yet please `fork the repository <https://github.com/jolideco/jolideco/fork>`_ 
first. Then you can clone your fork to your local computer::

    git clone https://github.com/your-github/jolideco


To start on a new feature or bug fix, create a new branch from the main branch
using::

    git checkout -b my-new-feature

Then you can start working on your changes. Once you are done, you can push your
branch to your fork on GitHub and create a pull request. The pull request will
be reviewed and if everything is fine, it will be merged into the main branch.

Development Environment
-----------------------

For handling the development and test
environments Jolideco uses the tool `tox <https://tox.wiki/>`_. You can start
from any Python environment and install tox first for example using pip::

    pip install tox

From there all the necessary dependencies will be installed automatically
and be handled by tox. 


For the development you can rely on one of the pre-defined test environments::

    tox --devenv venv-jolideco-dev -e py310
    source venv-jolieco-dev/bin/activate

This will create a new ``venv-jolideco-env`` environment, that you can activate
using the ``source`` command. To leave the environment again use ``deactivate``.
The command requires that you have Python 3.10 installed on your system. In case
you do not have it installed you could change the command to the corresponding
Python version like::

    tox --devenv venv-jolideco-dev -e py311

However it is recommended to use a rather new Python version for development.

Running Tests
-------------

To run the unit test you can use the following ``tox`` command::

    tox -e test


If you only want to run part of the test suite, you can also use pytest
directly with::

    pip install -e .[test]
    pytest


Building Docs
-------------

Building the documentation is no longer done using
``python setup.py build_docs``. Instead you will need to run::

    tox -e docs


You can also build the documentation with Sphinx directly using::

    pip install -e .[docs]
    cd docs
    make html

