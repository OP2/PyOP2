"""
PyOP2 is a library for parallel computations on unstructured meshes and
delivers performance-portability across a range of platforms:

* multi-core CPU (sequential, OpenMP, OpenCL and MPI)
* GPU (CUDA and OpenCL)
"""

# CX1 hack
import sys
try:
    sys.path.remove('/apps/python/2.7.3/lib/python2.7/site-packages/petsc4py-3.5-py2.7-linux-x86_64.egg')
except:
    pass
try:
    sys.path.remove('/apps/python/2.7.3/lib/python2.7/site-packages/mpi4py-1.3.1-py2.7-linux-x86_64.egg')
    sys.path.insert(1, '/home/fl1612/Packages/mpi4py-2.0.0/build/lib.linux-x86_64-2.7')
except:
    pass

from op2 import *
from version import __version__ as ver, __version_info__  # noqa: just expose

from ._version import get_versions
__version__ = get_versions(default={"version": ver, "full": ""})['version']
del get_versions
