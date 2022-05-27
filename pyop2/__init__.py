"""
PyOP2 is a library for parallel computations on unstructured meshes.
"""
from pyop2.op2 import *  # noqa
from pyop2.version import __version_info__  # noqa: just expose

from pyop2._version import get_versions
__version__ = get_versions()['version']
del get_versions

from pyop2.configuration import configuration
from pyop2.compilation import max_simd_width
if configuration["vectorization_strategy"]:
    configuration["simd_width"] = max_simd_width()
