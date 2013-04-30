# Installation

The main testing platform for PyOP2 is Ubuntu 12.04 64-bit with Python 2.7.3.
Other UNIX-like systems may or may not work. Microsoft Windows is not
supported.

## Preparing the system

OP2 and PyOP2 require a number of tools to be available:
  * gcc, make, CMake
  * bzr, Git, Mercurial
  * pip and the Python headers
  * SWIG

On a Debian-based system (Ubuntu, Mint, etc.) install them by running
```
sudo apt-get install -y build-essential python-dev bzr git-core mercurial \
       cmake cmake-curses-gui python-pip swig
```

## OP2-Common

PyOP2 depends on the [OP2-Common](https://github.com/OP2/OP2-Common) library
(only sequential is needed), which is built in-place as follows:
```
git clone git://github.com/OP2/OP2-Common.git
cd OP2-Common/op2/c
./cmake.local -DOP2_WITH_CUDA=0 -DOP2_WITH_HDF5=0 -DOP2_WITH_MPI=0 -DOP2_WITH_OPENMP=0
cd ..
export OP2_DIR=`pwd`
```

For further instructions refer to the [OP2-Common README]
(https://github.com/OP2/OP2-Common/blob/master/op2/c/README).

If you have already built OP2-Common, make sure `OP2_DIR` is exported or the
PyOP2 setup will fail.

## Dependencies

To install dependencies system-wide use `sudo -E pip install ...`, to install
to a user site use `pip install --user ...`. If you don't want PyOP2 or its
dependencies interfering with your exisiting Pyhton environment, consider
creating a [virtualenv](http://virtualenv.org/). In the following we will use
`pip install ...` to mean any of these options.

**Note:** Installing to the user site does not always give packages priority
over system installed packages on your `sys.path`.

### Common
Common dependencies:
  * Cython >= 0.17
  * decorator
  * instant >= 1.0
  * numpy >= 1.6
  * [PETSc][petsc_repo] >= 3.2 with Fortran interface, C++ and OpenMP support
  * [PETSc4py][petsc4py_repo] >= 3.3
  * PyYAML

Additional Python 2.6 dependencies:
  * argparse
  * ordereddict

Install dependencies via `pip`:
```
pip install Cython decorator instant numpy pyyaml
pip install argparse ordereddict # python < 2.7 only
```

### PETSc

PyOP2 uses [petsc4py](http://packages.python.org/petsc4py/), the Python
bindings for the [PETSc](http://www.mcs.anl.gov/petsc/) linear algebra library.

We maintain [a fork of petsc4py][petsc4py_repo] with extensions that are
required by PyOP2 and requires:
  * an MPI implementation built with *shared libraries*
  * PETSc 3.3 built with *shared libraries*

If you have a suitable PETSc installed on your system, `PETSC_DIR` and
`PETSC_ARCH` need to be set for the petsc4py installer to find it. On a
Debian/Ubuntu system with PETSc 3.3 installed, this can be achieved via:
```
export PETSC_DIR=/usr/lib/petscdir/3.3
export PETSC_ARCH=linux-gnu-c-opt
```


If not, make sure all PETSc dependencies (BLAS/LAPACK, MPI and a Fortran
compiler) are installed. On a Debian based system, run:
```
sudo apt-get install -y libopenmpi-dev openmpi-bin libblas-dev liblapack-dev gfortran
```
If you want OpenMP support or don't have a suitable PETSc installed on your
system, build the [PETSc OMP branch][petsc_repo]:
```
PETSC_CONFIGURE_OPTIONS="--with-fortran-interfaces=1 --with-c++-support --with-openmp" \
  pip install hg+https://bitbucket.org/ggorman/petsc-3.3-omp
unset PETSC_DIR
unset PETSC_ARCH
```

If you built PETSc using `pip`, `PETSC_DIR` and `PETSC_ARCH` should be
left unset when building petsc4py.

Install [petsc4py][petsc4py_repo] via `pip`:
```
pip install hg+https://bitbucket.org/fr710/petsc4py#egg=petsc4py
```

**Note:** When using PyOP2 with Fluidity it's crucial that both are built
against the same PETSc, which must be build with Fortran support!

### CUDA backend:
Dependencies:
  * codepy >= 2012.1.2
  * Jinja2
  * mako
  * pycparser >= 2.09.1 (revision a460398 or newer)
  * pycuda revision a6c9b40 or newer

The [cusp library](https://code.google.com/p/cusp-library/) headers need to be
in your (CUDA) include path.

Install via `pip`:
```
pip install codepy Jinja2 mako hg+https://bitbucket.org/eliben/pycparser#egg=pycparser-2.09.1
```

PyCuda can be installed on recent versions of Debian/Ubuntu by executing:
```
sudo apt-get install python-pycuda
```

If a PyCuda package is not available, it will be necessary to install it manually.
Make sure `nvcc` is in your `$PATH` and `libcuda.so` in your
`$LIBRARY_PATH` if in a non-standard location:
```
export CUDA_ROOT=/usr/local/cuda # change as appropriate
cd /tmp
git clone http://git.tiker.net/trees/pycuda.git
cd pycuda
git submodule init
git submodule update
# libcuda.so is in a non-standard location on Ubuntu systems
./configure.py --no-use-shipped-boost \
  --cudadrv-lib-dir='/usr/lib/nvidia-current,${CUDA_ROOT}/lib,${CUDA_ROOT}/lib64'
python setup.py build
sudo python setup.py install
sudo cp siteconf.py /etc/aksetup-defaults.py
```

### OpenCL backend:
Dependencies:
  * Jinja2
  * mako
  * pycparser >= 2.09.1 (revision a460398 or newer)
  * pyopencl >= 2012.1

Install via `pip`:
```
pip install Jinja2 mako pyopencl>=2012.1 hg+https://bitbucket.org/eliben/pycparser#egg=pycparser-2.09.1
```

Installing the Intel OpenCL toolkit (64bit systems only):
```
cd /tmp
# install alien to convert the rpm to a deb package
sudo apt-get install alien fakeroot
wget http://registrationcenter.intel.com/irc_nas/2563/intel_sdk_for_ocl_applications_2012_x64.tgz
tar xzf intel_sdk_for_ocl_applications_2012_x64.tgz
fakeroot alien *.rpm
sudo dpkg -i --force-overwrite *.deb
```

The `--force-overwrite` option is necessary in order to resolve conflicts with
the opencl-headers package (if installed).

Installing the [AMD OpenCL toolkit][AMD_opencl] (32bit and 64bit systems):
```
wget http://developer.amd.com/wordpress/media/2012/11/AMD-APP-SDK-v2.8-lnx64.tgz
# on a 32bit system, instead
# wget http://developer.amd.com/wordpress/media/2012/11/AMD-APP-SDK-v2.8-lnx32.tgz
tar xzf AMD-APP-SDK-v2.8-lnx*.tgz
# Install to /usr/local instead of /opt
sed -ie 's:/opt:/usr/local:g' default-install_lnx*.pl
sudo ./Install-AMD-APP.sh
```

### HDF5

PyOP2 allows initializing data structures using data stored in HDF5 files.
To use this feature you need the optional dependency [h5py](http://h5py.org).

On a Debian-based system, run:
```
sudo apt-get install libhdf5-mpi-dev python-h5py
```

Alternatively, if the HDF5 library is available, `pip install h5py`.

## Building PyOP2

Clone the PyOP2 repository:
```
git clone git://github.com/OP2/PyOP2.git
```

If not set, `OP2_DIR` should be set to the location of the 'op2' folder within
the OP2-Common build. PyOP2 uses [Cython](http://cython.org) extension modules,
which need to be built in-place when using PyOP2 from the source tree:
```
python setup.py build_ext --inplace
```

When running PyOP2 from the source tree, make sure it is on your `$PYTHONPATH`:
```
export PYTHONPATH=/path/to/PyOP2:$PYTHONPATH
```

When installing PyOP2 via `python setup.py install` the extension modules will
be built automatically and amending `$PYTHONPATH` is not necessary.

## FFC Interface

Solving [UFL](https://launchpad.net/ufl) finite element equations requires a
[fork of FFC][ffc_repo] and dependencies:
  * [UFL](https://launchpad.net/ufl)
  * [UFC](https://launchpad.net/ufc)
  * [FIAT](https://launchpad.net/fiat)

### Install via the package manager

On a supported platform, get all the dependencies for FFC by installing the
FEniCS toolchain from [packages](http://fenicsproject.org/download/):
```
sudo apt-get install fenics
```

Our [FFC fork][ffc_repo] is required, and must be added to your `$PYTHONPATH`:
```
bzr branch lp:~mapdes/ffc/pyop2 $FFC_DIR
export PYTHONPATH=$FFC_DIR:$PYTHONPATH
```

This branch of FFC also requires the trunk version of
[UFL](https://launchpad.net/ufl), also added to `$PYTHONPATH`:
```
bzr branch lp:ufl $UFL_DIR
export PYTHONPATH=$UFL_DIR:$PYTHONPATH
```

### Install via pip

Alternatively, install FFC and all dependencies via pip:
```
pip install \
  bzr+http://bazaar.launchpad.net/~mapdes/ffc/pyop2#egg=ffc \
  bzr+http://bazaar.launchpad.net/~florian-rathgeber/ufc/python-setup#egg=ufc_utils \
  bzr+http://bazaar.launchpad.net/~ufl-core/ufl/main#egg=ufl \
  bzr+http://bazaar.launchpad.net/~fiat-core/fiat/main#egg=fiat \
  hg+https://bitbucket.org/khinsen/scientificpython#egg=ScientificPython
```

## Setting up the environment

To make sure PyOP2 finds all its dependencies, create a file `.env` e.g. in
your PyOP2 root directory and source it via `. .env` when using PyOP2. Use the
template below, adjusting paths and removing definitions as necessary:
```
# Root directory of your OP2 installation, always needed
export OP2_DIR=/path/to/OP2-Common/op2
# If you have installed the OP2 library define e.g.
export OP2_PREFIX=/usr/local

# PETSc installation, not necessary when PETSc was installed via pip
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=linux-gnu-c-opt

# Add UFL and FFC to PYTHONPATH if in non-standard location
export UFL_DIR=/path/to/ufl
export FFC_DIR=/path/to/ffc
export PYTHONPATH=$UFL_DIR:$FFC_DIR:$PYTHONPATH
# Add any other Python module in non-standard locations

# Add PyOP2 to PYTHONPATH
export PYTHONPATH=/path/to/PyOP2:$PYTHONPATH
```

Alternatively, package the configuration in an
[environment module](http://modules.sourceforge.net/).

## Testing your installation

PyOP2 unit tests use [pytest](http://pytest.org). Install via package manager
```
sudo apt-get install python-pytest
```
or pip
```
pip install pytest
```

If you install pytest using `pip --user`, you should include the pip binary
folder in you path by adding the following to `.env`.

```
# Add pytest binaries to the path
export PATH=${PATH}:${HOME}/.local/bin
```

If all tests in our test suite pass, you should be good to go:
```
make test
```
This will run both unit and regression tests, the latter require UFL and FFC.

This will attempt to run tests for all backends and skip those for not
available backends. If the [FFC fork][ffc_repo] is not found, tests for the
FFC interface are xfailed.

## Troubleshooting

Start by verifying that PyOP2 picks up the "correct" dependencies, in
particular if you have several versions of a Python package installed in
different places on the system.

Run `pydoc <module>` to find out where a module/package is loaded from. To
print the module search path, run:
```
python -c 'from pprint import pprint; import sys; pprint(sys.path)'
```

[petsc_repo]: https://bitbucket.org/ggorman/petsc-3.3-omp
[petsc4py_repo]: https://bitbucket.org/fr710/petsc4py
[ffc_repo]: https://code.launchpad.net/~mapdes/ffc/pyop2
[AMD_opencl]: http://developer.amd.com/tools/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/
