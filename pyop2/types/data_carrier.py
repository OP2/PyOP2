import abc

import numpy as np

from pyop2 import (
    datatypes as dtypes,
    mpi,
    utils
)
from pyop2.types.access import Access
from pyop2.array import MirroredArray


class DataCarrier(abc.ABC):

    """Abstract base class for OP2 data.

    Actual objects will be :class:`DataCarrier` objects of rank 0
    (:class:`Global`), rank 1 (:class:`Dat`), or rank 2
    (:class:`Mat`)"""

    DEFAULT_DTYPE = dtypes.ScalarType

    @utils.cached_property
    def dtype(self):
        """The Python type of the data."""
        return self._data.dtype

    @utils.cached_property
    def ctype(self):
        """The c type of the data."""
        return dtypes.as_cstr(self.dtype)

    @utils.cached_property
    def name(self):
        """User-defined label."""
        return self._name

    @utils.cached_property
    def dim(self):
        """The shape tuple of the values for each element of the object."""
        return self._dim

    @utils.cached_property
    def cdim(self):
        """The scalar number of values for each member of the object. This is
        the product of the dim tuple."""
        return self._cdim


class VecAccessMixin(abc.ABC):
    """Mixin class providing access to arrays as PETSc Vecs."""

    @property
    @mpi.collective
    @abc.abstractmethod
    def vec(self):
        """Return a read-write context manager for wrapping data with a PETSc Vec."""
        pass

    @property
    @mpi.collective
    @abc.abstractmethod
    def vec_ro(self):
        """Return a read-only context manager for wrapping data with a PETSc Vec."""
        pass

    @property
    @mpi.collective
    @abc.abstractmethod
    def vec_wo(self):
        """Return a write-only context manager for wrapping data with a PETSc Vec."""
        pass
