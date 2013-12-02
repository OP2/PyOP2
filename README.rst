.. image:: https://travis-ci.org/OP2/PyOP2.png?branch=master
  :target: https://travis-ci.org/OP2/PyOP2
  :alt: build status

Installing PyOP2
================

The main testing platform for PyOP2 is Ubuntu 12.04 64-bit with Python
2.7.3. Other UNIX-like systems may or may not work. Microsoft Windows is
not supported.

Quick start
-----------

For the impatient there is a script for the unattended installation of
PyOP2 and its dependencies on a Ubuntu 12.04 or compatible platform.
Only the sequential and OpenMP backends are covered at the moment.

.. note::
  This script is not intended to be used by PyOP2 developers. If you intend
  to contribute to PyOP2 it is recommended to follow the instructions below
  for a manual installation.

Running with superuser privileges will install missing packages and
Python dependencies will be installed system wide::

  wget -O - https://github.com/OP2/PyOP2/raw/master/install.sh | sudo bash

.. warning::
  This will fail if you if you require a password for ``sudo``. Run e.g. the
  following beforehand to assure your password is cached ::

      sudo whoami

Running without superuser privileges will instruct you which packages
need to be installed. Python dependencies will be installed to the user
site ``~/.local``::

  wget -O - https://github.com/OP2/PyOP2/raw/master/install.sh | bash

In each case, OP2-Common and PyOP2 will be cloned to subdirectories of
the current directory.

After installation has completed and a rudimentary functionality check,
the test suite is run. The script indicates whether all these steps have
completed successfully and only in this case will exit with return code
0.

Only high-level progress updates are printed to screen. Most of the
output is redirected to a log file ``pyop2_install.log``. Please consult
this log file in the case of errors. If you can't figure out the cause
of discover a bug in the installation script, please `report
it <https://github.com/OP2/PyOP2/issues>`__.

This completes the quick start installation. More complete
instructions follow for virtual machine and native installations.

Provisioning a virtual machine
------------------------------

A ``Vagrantfile`` is provided for automatic provisioning of a Ubuntu
12.04 64bit virtual machine with PyOP2 preinstalled. It requires
`VirtualBox 4.2 <https://www.virtualbox.org/wiki/Linux_Downloads>`__ and
`Vagrant <http://www.vagrantup.com>`__ to be installed, which are
available for Linux, Mac and Windows.

Creating and launching a virtual machine is a single command: run
``vagrant up`` to automatically download the base VM image, configure it
for use with VirtualBox, boot the VM and install PyOP2 and all
dependencies using the above install script.

Preparing the system
--------------------

PyOP2 require a number of tools to be available: 

* gcc, make, CMake 
* bzr, Git, Mercurial 
* pip and the Python headers 
* SWIG

On a Debian-based system (Ubuntu, Mint, etc.) install them by running::

  sudo apt-get install -y build-essential python-dev bzr git-core mercurial \
  cmake cmake-curses-gui python-pip swig

Dependencies
------------

To install dependencies system-wide use ``sudo -E pip install ...``, to
install to a user site use ``pip install --user ...``. If you don't want
PyOP2 or its dependencies interfering with your existing Python environment,
consider creating a `virtualenv <http://virtualenv.org/>`__.

**Note:** In the following we will use ``pip install ...`` to mean any
of the above options.

**Note:** Installing to the user site does not always give packages
priority over system installed packages on your ``sys.path``.

Common
~~~~~~

Common dependencies: 

* Cython >= 0.17 
* decorator 
* instant revision 7301ecb or newer
* numpy >= 1.6 
* PETSc_ >= 3.4 with Fortran interfaces
* PETSc4py_ >= 3.4

Testing dependencies (optional, required to run the tests):

* pytest >= 2.3
* flake8 >= 2.1.0

With the exception of the PETSc_ dependencies, these can be installed
using the package management system of your OS, or via ``pip``.

Install the dependencies via the package manager (Debian based systems)::

  sudo apt-get install cython python-decorator python-numpy

**Note:** This may not give you recent enough versions of those packages
(in particular the Cython version shipped with 12.04 is too old). You
can selectively upgrade packages via ``pip``, see below.

Install dependencies via ``pip``::

  pip install "Cython>=0.17" decorator "numpy>=1.6"
  pip install git+https://bitbucket.org/fenics-project/instant

Additional Python 2.6 dependencies: 

* argparse 
* ordereddict

Install these via ``pip``::

  pip install argparse ordereddict

PETSc
~~~~~

PyOP2 uses petsc4py_, the Python bindings for the PETSc_ linear algebra
library and requires:

* an MPI implementation built with *shared libraries* 
* PETSc_ 3.4 or later built with *shared libraries*

If you have a suitable PETSc_ installed on your system, ``PETSC_DIR`` and
``PETSC_ARCH`` need to be set for the petsc4py_ installer to find it. On
a Debian/Ubuntu system with PETSc_ 3.4 installed, this can be achieved
via e.g. (adapt for your system) ::

  export PETSC_DIR=/usr/lib/petscdir/3.4
  export PETSC_ARCH=linux-gnu-c-opt

If you are on Ubuntu 12.04 LTS, you can install the binary package for PETSc_
3.4.2 from the PPA_ ``ppa:amcg/petsc3.4``.

If not, make sure all PETSc_ dependencies (BLAS/LAPACK, MPI and a Fortran
compiler) are installed. On a Debian based system, run::

  sudo apt-get install -y libopenmpi-dev openmpi-bin libblas-dev liblapack-dev gfortran

Then install PETSc_ via ``pip`` ::

  PETSC_CONFIGURE_OPTIONS="--with-fortran-interfaces=1 --with-c++-support" \
    pip install "petsc >= 3.4"
  unset PETSC_DIR
  unset PETSC_ARCH

If you built PETSc_ using ``pip``, ``PETSC_DIR`` and ``PETSC_ARCH``
should be left unset when building petsc4py_.

Install petsc4py_ via ``pip``::

  pip install "petsc4py >= 3.4"

CUDA backend:
~~~~~~~~~~~~~

Dependencies: 

* boost-python 
* Cusp 0.3.1 
* codepy >= 2013.1 
* Jinja2 
* mako 
* pycparser >= 2.10
* pycuda >= 2013.1

The `cusp library <http://cusplibrary.github.io>`__ version 0.3.1
headers need to be in your (CUDA) include path.

**Note:** Using the trunk version of Cusp will *not* work, since
revision f525d61 introduces a change that break backwards compatibility
with CUDA 4.x.

Install dependencies via the package manager (Debian based systems)::

  sudo apt-get install libboost-python-dev python-jinja2 python-mako python-pycuda

**Note:** The version of pycparser available in the package repositories
is too old, you will need to install it via ``pip``, see below.

Install dependencies via ``pip``::

  pip install codepy Jinja2 mako pycparser>=2.10

If a pycuda package is not available, it will be necessary to install it
manually. Make sure ``nvcc`` is in your ``$PATH`` and ``libcuda.so`` in
your ``$LIBRARY_PATH`` if in a non-standard location::

  export CUDA_ROOT=/usr/local/cuda # change as appropriate 
  git clone https://github.com/inducer/pycuda.git 
  cd pycuda 
  git submodule init 
  git submodule update 
  # libcuda.so is in a non-standard location on Ubuntu systems 
  ./configure.py --no-use-shipped-boost \
  --cudadrv-lib-dir="/usr/lib/nvidia-current,${CUDA_ROOT}/lib,${CUDA_ROOT}/lib64" 
  python setup.py build 
  sudo python setup.py install 
  sudo cp siteconf.py /etc/aksetup-defaults.py

OpenCL backend:
~~~~~~~~~~~~~~~

Dependencies: 

* Jinja2 
* mako 
* pycparser >= 2.10
* pyopencl >= 2012.1

pyopencl requires the OpenCL header ``CL/cl.h`` in a standard include
path. On a Debian system, install it via the package manager::

  sudo apt-get install opencl-headers

If you want to use OpenCL headers and/or libraries from a non-standard
location you need to configure pyopencl manually::

  export OPENCL_ROOT=/usr/local/opencl # change as appropriate 
  git clone https://github.com/inducer/pyopencl.git 
  cd pyopencl 
  git submodule init 
  git submodule update 
  ./configure.py --no-use-shipped-boost \
  --cl-inc-dir=${OPENCL_ROOT}/include --cl-lib-dir=${OPENCL_ROOT}/lib 
  python setup.py build 
  sudo python setup.py install

Otherwise, install dependencies via ``pip``::

  pip install Jinja2 mako pyopencl>=2012.1 pycparser>=2.10

Installing the Intel OpenCL toolkit (64bit systems only)::

  cd /tmp 
  # install alien to convert the rpm to a deb package 
  sudo apt-get install alien 
  fakeroot wget http://registrationcenter.intel.com/irc_nas/2563/intel_sdk_for_ocl_applications_2012_x64.tgz
  tar xzf intel_sdk_for_ocl_applications_2012_x64.tgz 
  fakeroot alien *.rpm 
  sudo dpkg -i --force-overwrite *.deb

The ``--force-overwrite`` option is necessary in order to resolve
conflicts with the opencl-headers package (if installed).

Installing the `AMD OpenCL
toolkit <http://developer.amd.com/tools/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/>`__
(32bit and 64bit systems)::

  wget http://developer.amd.com/wordpress/media/2012/11/AMD-APP-SDK-v2.8-lnx64.tgz 
  # on a 32bit system, instead 
  wget http://developer.amd.com/wordpress/media/2012/11/AMD-APP-SDK-v2.8-lnx32.tgz 
  tar xzf AMD-APP-SDK-v2.8-lnx*.tgz 
  # Install to /usr/local instead of /opt 
  sed -ie 's:/opt:/usr/local:g' default-install_lnx*.pl
  sudo ./Install-AMD-APP.sh

HDF5
~~~~

PyOP2 allows initializing data structures using data stored in HDF5
files. To use this feature you need the optional dependency
`h5py <http://h5py.org>`__.

On a Debian-based system, run::

  sudo apt-get install libhdf5-mpi-dev python-h5py

Alternatively, if the HDF5 library is available, ``pip install h5py``.

Building PyOP2
--------------

Clone the PyOP2 repository::

  git clone git://github.com/OP2/PyOP2.git
 
PyOP2 uses `Cython <http://cython.org>`__ extension modules, which need to be built
in-place when using PyOP2 from the source tree::

  python setup.py build_ext --inplace

When running PyOP2 from the source tree, make sure it is on your
``$PYTHONPATH``::

  export PYTHONPATH=/path/to/PyOP2:$PYTHONPATH

When installing PyOP2 via ``python setup.py install`` the extension
modules will be built automatically and amending ``$PYTHONPATH`` is not
necessary.

FFC Interface
-------------

Solving UFL_ finite element equations requires a fork of FFC_, UFL_ and FIAT_.

Install FFC_ and all dependencies via pip::

  pip install \
    git+https://bitbucket.org/mapdes/ffc.git#egg=ffc
    git+https://bitbucket.org/mapdes/ufl.git#egg=ufl
    git+https://bitbucket.org/mapdes/fiat.git#egg=fiat
    hg+https://bitbucket.org/khinsen/scientificpython

Setting up the environment
--------------------------

To make sure PyOP2 finds all its dependencies, create a file ``.env``
e.g. in your PyOP2 root directory and source it via ``. .env`` when
using PyOP2. Use the template below, adjusting paths and removing
definitions as necessary::

  #PETSc installation, not necessary when PETSc was installed via pip
  export PETSC_DIR=/path/to/petsc 
  export PETSC_ARCH=linux-gnu-c-opt

  #Add UFL and FFC to PYTHONPATH if in non-standard location
  export UFL_DIR=/path/to/ufl 
  export FFC_DIR=/path/to/ffc 
  export PYTHONPATH=$UFL_DIR:$FFC_DIR:$PYTHONPATH 
  # Add any other Python module in non-standard locations

  #Add PyOP2 to PYTHONPATH
  export PYTHONPATH=/path/to/PyOP2:$PYTHONPATH

Alternatively, package the configuration in an `environment
module <http://modules.sourceforge.net/>`__.

Testing your installation
-------------------------

PyOP2 unit tests use `pytest <http://pytest.org>`__ >= 2.3. Install via package
manager::

  sudo apt-get install python-pytest 

or pip::

  pip install "pytest>=2.3"

The code linting test uses `flake8 <http://flake8.readthedocs.org>`__.
Install via pip::

  pip install "flake8>=2.1.0"

If you install *pytest* and *flake8* using ``pip --user``, you should
include the binary folder of your local site in your path by adding the
following to ``~/.bashrc`` or ``.env``::

  # Add pytest binaries to the path
  export PATH=${PATH}:${HOME}/.local/bin

If all tests in our test suite pass, you should be good to go::

  make test

This will run both unit and regression tests, the latter require UFL_ and FFC_.

This will attempt to run tests for all backends and skip those for not
available backends. If the FFC_ fork is not found, tests for the FFC_ interface
are xfailed.

Troubleshooting
---------------

Start by verifying that PyOP2 picks up the "correct" dependencies, in
particular if you have several versions of a Python package installed in
different places on the system.

Run ``pydoc <module>`` to find out where a module/package is loaded
from. To print the module search path, run::

 python -c 'from pprint import pprint; import sys; pprint(sys.path)'

.. _PPA: https://launchpad.net/~amcg/+archive/petsc3.4/
.. _PETSc: http://www.mcs.anl.gov/petsc/
.. _petsc4py: http://pythonhosted.org/petsc4py/
.. _FFC: https://bitbucket.org/mapdes/ffc
.. _FIAT: https://bitbucket.org/mapdes/fiat
.. _UFL: https://bitbucket.org/mapdes/ufl
