MultiProFit
###########

.. todo image:: https://travis-ci.org/ICRAR/multiprofit.svg?branch=master
   .. todo   :target: https://travis-ci.org/lsst-dm/multiprofit

.. todo image:: https://img.shields.io/pypi/v/multiprofit.svg
   .. todo   :target: https://pypi.python.org/pypi/multiprofit

.. todo image:: https://img.shields.io/pypi/pyversions/multiprofit.svg
   .. todo   :target: https://pypi.python.org/pypi/multiprofit

*multiprofit* is a Python astronomical source modelling code, inspired by `ProFit <https://www.github
.com/ICRAR/ProFit>`_, but made for LSST Data Management. MultiProFit means Multiple Profile Fitting. The
multi- aspect can be multi-object, multi-component, multi-band, multi-instrument, and someday multi-epoch.

*multiprofit* can fit any kind of imaging data while modelling sources as Gaussian mixtures - including
approximations to Sersic profiles - using a Gaussian pixel-convolved point spread function. It can also use
`GalSim <https://github.com/GalSim-developers/GalSim/>`_ or `libprofit <https://github.com/ICRAR/libprofit/>`_
via `pyprofit <https://github.com/ICRAR/pyprofit/>`_ to generate true Sersic and/or other supported
models convolved with arbitrary PSFs images or models.

*multiprofit* has support for multi-object fitting and experimental support for multi-band fitting, albeit
currently limited to pixel-matched images of identical dimensions. Unlike ProFit, Bayesian MCMC is not
available (yet).

*multiprofit* requires `Eigen3 <https://bitbucket.org/eigen/eigen>`_ and `pybind11 <https://github
.com/pybind/pybind11>`_ for its python bindings and can be installed from setup.py like so:

CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$EIGEN_DIR  python setup.py install --user

The CMAKE_PREFIX_PATH may be unnecessary if using CMake 3.0+.

.. todo *multiprofit* is available in `PyPI <https://pypi.python.org/pypi/multiprofit>`_
   .. and thus can be easily installed via::

.. pip install multiprofit
