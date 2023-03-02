import collections
import contextlib
import ctypes
import operator
import numbers

import numpy as np
from petsc4py import PETSc

from pyop2 import (
    datatypes,
    exceptions as ex,
    mpi,
    utils
)
from pyop2.types.access import Access
from pyop2.types.data_carrier import DataCarrier, VecAccessMixin
from pyop2.array import MirroredArray
from pyop2.offload_utils import _offloading as offloading


class Global(DataCarrier, VecAccessMixin):

    """OP2 global value.

    When a ``Global`` is passed to a :func:`pyop2.op2.par_loop`, the access
    descriptor is passed by `calling` the ``Global``.  For example, if
    a ``Global`` named ``G`` is to be accessed for reading, this is
    accomplished by::

      G(pyop2.READ)

    It is permissible to pass `None` as the `data` argument.  In this
    case, allocation of the data buffer is postponed until it is
    accessed.

    .. note::
        If the data buffer is not passed in, it is implicitly
        initialised to be zero.
    """

    _modes = [Access.READ, Access.INC, Access.MIN, Access.MAX]

    @utils.validate_type(('name', str, ex.NameTypeError))
    def __init__(self, dim, data=None, dtype=None, name=None, comm=None):
        if isinstance(dim, Global):
            # If g is a Global, Global(g) performs a deep copy. This is for compatibility with Dat.
            self.__init__(dim._dim, None, dtype=dim.dtype,
                          name="copy_of_%s" % dim.name, comm=dim.comm)
            dim.copy(self)
            return

        dim = utils.as_tuple(dim, int)

        # TODO make this a function so can be shared by Dat and Global
        # handle the interplay between dataset, data and dtype
        if data is not None:
            data = utils.verify_reshape(data, dtype, dim)
            self._stashed_data = MirroredArray.new((data.shape, data.dtype))
        else:
            dtype = np.dtype(dtype or DataCarrier.DEFAULT_DTYPE)
            data = (dim, dtype)
            self._stashed_data = MirroredArray.new(data)
        self._data = MirroredArray.new(data)

        # TODO shouldn't need to set these
        self._dim = dim
        self._cdim = np.prod(dim)

        self._name = name or "global_#x%x" % id(self)
        self.comm = mpi.internal_comm(comm)

    def __del__(self):
        if hasattr(self, "comm"):
            mpi.decref(self.comm)

    @property
    def kernel_args_rw(self):
        return (self._data.ptr_rw,)

    @property
    def kernel_args_ro(self):
        return (self._data.ptr_ro,)

    @property
    def kernel_args_wo(self):
        return (self._data.ptr_wo,)

    @utils.cached_property
    def _argtypes_(self):
        return (ctypes.c_voidp, )

    @utils.cached_property
    def _wrapper_cache_key_(self):
        return (type(self), self.dtype, self.shape)

    @utils.validate_in(('access', _modes, ex.ModeValueError))
    def __call__(self, access, map_=None):
        from pyop2.parloop import GlobalLegacyArg

        assert map_ is None
        return GlobalLegacyArg(self, access)

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    def __getitem__(self, idx):
        """Return self if ``idx`` is 0, raise an error otherwise."""
        if idx != 0:
            raise ex.IndexValueError("Can only extract component 0 from %r" % self)
        return self

    def __str__(self):
        return "OP2 Global Argument: %s with dim %s and value %s" \
            % (self._name, self._dim, self.data_ro)

    def __repr__(self):
        return "Global(%r, %r, %r, %r)" % (self.shape, self.data_ro,
                                           self.data.dtype, self._name)

    @utils.cached_property
    def dataset(self):
        from pyop2.op2 import compute_backend
        return compute_backend.GlobalDataSet(self)

    @property
    def shape(self):
        return self._data.shape

    @property
    def data(self):
        """Data array."""
        return self._data.data

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def data_ro(self):
        """Data array."""
        return self._data.data_ro

    @property
    def data_with_halos(self):
        return self.data

    @property
    def data_ro_with_halos(self):
        return self.data_ro

    @property
    def split(self):
        return (self,)

    @property
    def nbytes(self):
        """Return an estimate of the size of the data associated with this
        :class:`Global` in bytes. This will be the correct size of the
        data payload, but does not take into account the overhead of
        the object and its metadata. This renders this method of
        little statistical significance, however it is included to
        make the interface consistent.
        """

        return self.dtype.itemsize * self._cdim

    @mpi.collective
    def duplicate(self):
        """Return a deep copy of self."""
        return type(self)(self.dim, data=np.copy(self.data_ro),
                          dtype=self.dtype, name=self.name)

    @mpi.collective
    def copy(self, other, subset=None):
        """Copy the data in this :class:`Global` into another.

        :arg other: The destination :class:`Global`
        :arg subset: A :class:`Subset` of elements to copy (optional)"""

        other.data = np.copy(self.data_ro)

    @mpi.collective
    def zero(self, subset=None):
        assert subset is None
        self._data.data[...] = 0

    @mpi.collective
    def global_to_local_begin(self, access_mode):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        pass

    @mpi.collective
    def global_to_local_end(self, access_mode):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        pass

    @mpi.collective
    def local_to_global_begin(self, insert_mode):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        pass

    @mpi.collective
    def local_to_global_end(self, insert_mode):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        pass

    @mpi.collective
    def frozen_halo(self, access_mode):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        return contextlib.nullcontext()

    @mpi.collective
    def freeze_halo(self, access_mode):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        pass

    @mpi.collective
    def unfreeze_halo(self):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        pass

    def _op(self, other, op):
        ret = type(self)(self.dim, dtype=self.dtype, name=self.name, comm=self.comm)
        if isinstance(other, Global):
            ret.data[:] = op(self.data_ro, other.data_ro)
        else:
            ret.data[:] = op(self.data_ro, other)
        return ret

    def _iop(self, other, op):
        if isinstance(other, Global):
            op(self.data[:], other.data_ro)
        else:
            op(self.data[:], other)
        return self

    def __pos__(self):
        return self.duplicate()

    def __add__(self, other):
        """Pointwise addition of fields."""
        return self._op(other, operator.add)

    def __radd__(self, other):
        """Pointwise addition of fields.

        self.__radd__(other) <==> other + self."""
        return self + other

    def __neg__(self):
        return type(self)(self.dim, data=-np.copy(self.data_ro),
                          dtype=self.dtype, name=self.name)

    def __sub__(self, other):
        """Pointwise subtraction of fields."""
        return self._op(other, operator.sub)

    def __rsub__(self, other):
        """Pointwise subtraction of fields.

        self.__rsub__(other) <==> other - self."""
        ret = -self
        ret += other
        return ret

    def __mul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        return self._op(other, operator.mul)

    def __rmul__(self, other):
        """Pointwise multiplication or scaling of fields.

        self.__rmul__(other) <==> other * self."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Pointwise division or scaling of fields."""
        return self._op(other, operator.truediv)

    def __iadd__(self, other):
        """Pointwise addition of fields."""
        return self._iop(other, operator.iadd)

    def __isub__(self, other):
        """Pointwise subtraction of fields."""
        return self._iop(other, operator.isub)

    def __imul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        return self._iop(other, operator.imul)

    def __itruediv__(self, other):
        """Pointwise division or scaling of fields."""
        return self._iop(other, operator.itruediv)

    def inner(self, other):
        assert isinstance(other, Global)
        return np.dot(self.data_ro, np.conj(other.data_ro))

    @property
    def dat_version(self):
        return self._data.state

    @property
    def petsc_vec(self):
        if self._lazy_petsc_vec is None:
            if self.dtype != datatypes.ScalarType:
                raise ex.DataTypeError(
                    "Only arrays with dtype matching PETSc's scalar type can be "
                    "represented as a PETSc Vec")

            assert self.comm is not None
            assert self._data._lazy_host_data is not None

            self._lazy_petsc_vec = PETSc.Vec().createWithArray(
                self._data._lazy_host_data, size=self._data.size,
                bsize=self.cdim, comm=self.comm)

        return self._lazy_petsc_vec

    @property
    @contextlib.contextmanager
    def vec(self):
        """TODO"""
        if offloading:
            raise NotImplementedError("TODO")
        # if offloading:
        #     if not self._data.is_available_on_device:
        #         self._data.host_to_device_copy()
        #     self.petsc_vec.bindToCPU(False)
        # else:
        #     if not self._data.is_available_on_host:
        #         self._data.device_to_host_copy()
        #     self.petsc_vec.bindToCPU(True)

        self.petsc_vec.stateSet(self._data.state)
        yield self.petsc_vec
        self._data.state = self.petsc_vec.stateGet()

        if offloading:
            self._data.is_available_on_host = False
        else:
            self._data.is_available_on_device = False
        # not sure this is right
        self.comm.Bcast(self._data._lazy_host_data, 0)

    @contextlib.contextmanager
    def vec_context(self, access):
        """A context manager for a :class:`PETSc.Vec` from a :class:`Global`."""

    @property
    @contextlib.contextmanager
    def vec_ro(self):
        """TODO"""
        if offloading:
            raise NotImplementedError("TODO")
        if offloading:
            if not self._data.is_available_on_device:
                self._data.host_to_device_copy()
            self.petsc_vec.bindToCPU(False)
        else:
            if not self._data.is_available_on_host:
                self._data.device_to_host_copy()
            self.petsc_vec.bindToCPU(True)

        self.petsc_vec.stateSet(self._data.state)
        yield self.petsc_vec
        self._data.state = self.petsc_vec.stateGet()

    @property
    @contextlib.contextmanager
    def vec_wo(self):
        """TODO"""
        if offloading:
            raise NotImplementedError("TODO")
        if offloading:
            self.petsc_vec.bindToCPU(False)
        else:
            self.petsc_vec.bindToCPU(True)

        self.petsc_vec.stateSet(self._data.state)
        yield self.petsc_vec
        self._data.state = self.petsc_vec.stateGet()

        self.halo_valid = False
        if offloading:
            self._data.is_available_on_host = False
        else:
            self._data.is_available_on_device = False

        # not sure this is right
        self.comm.Bcast(self._data._lazy_host_data, 0)
