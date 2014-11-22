# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""Base classes for OP2 objects, containing metadata and runtime data
information which is backend independent. Individual runtime backends should
subclass these as required to implement backend-specific features.
"""

import weakref
import numpy as np
import operator
import types
from hashlib import md5

from configuration import configuration
from caching import Cached, ObjectCached
from versioning import Versioned, modifies, modifies_argn, CopyOnWrite, \
    shallow_copy, zeroes
from exceptions import *
from utils import *
from backends import _make_object
from mpi import MPI, _MPI, _check_comm, collective
from profiling import profile, timed_region, timed_function
from sparsity import build_sparsity
from version import __version__ as version

from coffee.ast_base import Node
from coffee import ast_base as ast


# Data API


class Access(object):

    """OP2 access type. In an :py:class:`Arg`, this describes how the
    :py:class:`DataCarrier` will be accessed.

    .. warning ::
        Access should not be instantiated by user code. Instead, use
        the predefined values: :const:`READ`, :const:`WRITE`, :const:`RW`,
        :const:`INC`, :const:`MIN`, :const:`MAX`
    """

    _modes = ["READ", "WRITE", "RW", "INC", "MIN", "MAX"]

    @validate_in(('mode', _modes, ModeValueError))
    def __init__(self, mode):
        self._mode = mode

    def __str__(self):
        return "OP2 Access: %s" % self._mode

    def __repr__(self):
        return "Access(%r)" % self._mode

READ = Access("READ")
"""The :class:`Global`, :class:`Dat`, or :class:`Mat` is accessed read-only."""

WRITE = Access("WRITE")
"""The  :class:`Global`, :class:`Dat`, or :class:`Mat` is accessed write-only,
and OP2 is not required to handle write conflicts."""

RW = Access("RW")
"""The  :class:`Global`, :class:`Dat`, or :class:`Mat` is accessed for reading
and writing, and OP2 is not required to handle write conflicts."""

INC = Access("INC")
"""The kernel computes increments to be summed onto a :class:`Global`,
:class:`Dat`, or :class:`Mat`. OP2 is responsible for managing the write
conflicts caused."""

MIN = Access("MIN")
"""The kernel contributes to a reduction into a :class:`Global` using a ``min``
operation. OP2 is responsible for reducing over the different kernel
invocations."""

MAX = Access("MAX")
"""The kernel contributes to a reduction into a :class:`Global` using a ``max``
operation. OP2 is responsible for reducing over the different kernel
invocations."""

# Data API


class Arg(object):

    """An argument to a :func:`pyop2.op2.par_loop`.

    .. warning ::
        User code should not directly instantiate :class:`Arg`.
        Instead, use the call syntax on the :class:`DataCarrier`.
    """

    def __init__(self, data=None, map=None, idx=None, access=None, flatten=False):
        """
        :param data: A data-carrying object, either :class:`Dat` or class:`Mat`
        :param map:  A :class:`Map` to access this :class:`Arg` or the default
                     if the identity map is to be used.
        :param idx:  An index into the :class:`Map`: an :class:`IterationIndex`
                     when using an iteration space, an :class:`int` to use a
                     given component of the mapping or the default to use all
                     components of the mapping.
        :param access: An access descriptor of type :class:`Access`
        :param flatten: Treat the data dimensions of this :class:`Arg` as flat
                        s.t. the kernel is passed a flat vector of length
                        ``map.arity * data.dataset.cdim``.

        Checks that:

        1. the maps used are initialized i.e. have mapping data associated, and
        2. the to Set of the map used to access it matches the Set it is
           defined on.

        A :class:`MapValueError` is raised if these conditions are not met."""
        self._dat = data
        self._map = map
        self._idx = idx
        self._access = access
        self._flatten = flatten
        self._in_flight = False  # some kind of comms in flight for this arg
        self._position = None
        self._indirect_position = None

        # Check arguments for consistency
        if not (self._is_global or map is None):
            for j, m in enumerate(map):
                if m.iterset.total_size > 0 and len(m.values_with_halo) == 0:
                    raise MapValueError("%s is not initialized." % map)
                if self._is_mat and m.toset != data.sparsity.dsets[j].set:
                    raise MapValueError(
                        "To set of %s doesn't match the set of %s." % (map, data))
            if self._is_dat and map.toset != data.dataset.set:
                raise MapValueError(
                    "To set of %s doesn't match the set of %s." % (map, data))

        # Determine the iteration space extents, if any
        if self._is_mat and flatten:
            rdims = tuple(d.cdim for d in data.sparsity.dsets[0])
            cdims = tuple(d.cdim for d in data.sparsity.dsets[1])
            self._block_shape = tuple(tuple((mr.arity * dr, mc.arity * dc)
                                      for mc, dc in zip(map[1], cdims))
                                      for mr, dr in zip(map[0], rdims))
        elif self._is_mat:
            self._block_shape = tuple(tuple((mr.arity, mc.arity)
                                      for mc in map[1])
                                      for mr in map[0])
        elif self._uses_itspace and flatten:
            self._block_shape = tuple(((m.arity * d.cdim,),) for m, d in zip(map, data))
        elif self._uses_itspace:
            self._block_shape = tuple(((m.arity,),) for m in map)
        else:
            self._block_shape = None

    def __eq__(self, other):
        """:class:`Arg`\s compare equal of they are defined on the same data,
        use the same :class:`Map` with the same index and the same access
        descriptor."""
        return self._dat == other._dat and self._map == other._map and \
            self._idx == other._idx and self._access == other._access

    def __ne__(self, other):
        """:class:`Arg`\s compare equal of they are defined on the same data,
        use the same :class:`Map` with the same index and the same access
        descriptor."""
        return not self == other

    def __str__(self):
        return "OP2 Arg: dat %s, map %s, index %s, access %s" % \
            (self._dat, self._map, self._idx, self._access)

    def __repr__(self):
        return "Arg(%r, %r, %r, %r)" % \
            (self._dat, self._map, self._idx, self._access)

    def __iter__(self):
        for arg in self.split:
            yield arg

    @property
    def split(self):
        """Split a mixed argument into a tuple of constituent arguments."""
        if self._is_mixed_dat:
            return tuple(_make_object('Arg', d, m, self._idx, self._access)
                         for d, m in zip(self._dat, self._map))
        elif self._is_mixed_mat:
            s = self.data.sparsity.shape
            mr, mc = self.map
            return tuple(_make_object('Arg', self.data[i, j], (mr.split[i], mc.split[j]),
                                      self._idx, self._access)
                         for j in range(s[1]) for i in range(s[0]))
        else:
            return (self,)

    @property
    def name(self):
        """The generated argument name."""
        return "arg%d" % self._position

    @property
    def position(self):
        """The position of this :class:`Arg` in the :class:`ParLoop` argument list"""
        return self._position

    @position.setter
    def position(self, val):
        """Set the position of this :class:`Arg` in the :class:`ParLoop` argument list"""
        self._position = val

    @property
    def indirect_position(self):
        """The position of the first unique occurence of this
    indirect :class:`Arg` in the :class:`ParLoop` argument list."""
        return self._indirect_position

    @indirect_position.setter
    def indirect_position(self, val):
        """Set the position of the first unique occurence of this
    indirect :class:`Arg` in the :class:`ParLoop` argument list."""
        self._indirect_position = val

    @property
    def ctype(self):
        """String representing the C type of the data in this ``Arg``."""
        return self.data.ctype

    @property
    def dtype(self):
        """Numpy datatype of this Arg"""
        return self.data.dtype

    @property
    def map(self):
        """The :class:`Map` via which the data is to be accessed."""
        return self._map

    @property
    def idx(self):
        """Index into the mapping."""
        return self._idx

    @property
    def access(self):
        """Access descriptor. One of the constants of type :class:`Access`"""
        return self._access

    @property
    def _is_soa(self):
        return self._is_dat and self._dat.soa

    @property
    def _is_vec_map(self):
        return self._is_indirect and self._idx is None

    @property
    def _is_mat(self):
        return isinstance(self._dat, Mat)

    @property
    def _is_mixed_mat(self):
        return self._is_mat and self._dat.sparsity.shape > (1, 1)

    @property
    def _is_global(self):
        return isinstance(self._dat, Global)

    @property
    def _is_global_reduction(self):
        return self._is_global and self._access in [INC, MIN, MAX]

    @property
    def _is_dat(self):
        return isinstance(self._dat, Dat)

    @property
    def _is_mixed_dat(self):
        return isinstance(self._dat, MixedDat)

    @property
    def _is_mixed(self):
        return self._is_mixed_dat or self._is_mixed_mat

    @property
    def _is_INC(self):
        return self._access == INC

    @property
    def _is_MIN(self):
        return self._access == MIN

    @property
    def _is_MAX(self):
        return self._access == MAX

    @property
    def _is_direct(self):
        return isinstance(self._dat, Dat) and self.map is None

    @property
    def _is_indirect(self):
        return isinstance(self._dat, Dat) and self.map is not None

    @property
    def _is_indirect_and_not_read(self):
        return self._is_indirect and self._access is not READ

    @property
    def _is_indirect_reduction(self):
        return self._is_indirect and self._access is INC

    @property
    def _uses_itspace(self):
        return self._is_mat or isinstance(self.idx, IterationIndex)

    @collective
    def halo_exchange_begin(self):
        """Begin halo exchange for the argument if a halo update is required.
        Doing halo exchanges only makes sense for :class:`Dat` objects."""
        assert not self._in_flight, \
            "Halo exchange already in flight for Arg %s" % self
        if self.access in [READ, RW] and self.data.needs_halo_update:
            self.data.needs_halo_update = False
            self._in_flight = True
            self.data.halo_exchange_begin()

    @collective
    def halo_exchange_end(self):
        """End halo exchange if it is in flight.
        Doing halo exchanges only makes sense for :class:`Dat` objects."""
        if self.access in [READ, RW] and self._in_flight:
            self._in_flight = False
            self.data.halo_exchange_end()

    @collective
    def reduction_begin(self):
        """Begin reduction for the argument if its access is INC, MIN, or MAX.
        Doing a reduction only makes sense for :class:`Global` objects."""
        assert self._is_global, \
            "Doing global reduction only makes sense for Globals"
        assert not self._in_flight, \
            "Reduction already in flight for Arg %s" % self
        if self.access is not READ:
            self._in_flight = True
            if self.access is INC:
                op = _MPI.SUM
            elif self.access is MIN:
                op = _MPI.MIN
            elif self.access is MAX:
                op = _MPI.MAX
            # If the MPI supports MPI-3, this could be MPI_Iallreduce
            # instead, to allow overlapping comp and comms.
            # We must reduce into a temporary buffer so that when
            # executing over the halo region, which occurs after we've
            # called this reduction, we don't subsequently overwrite
            # the result.
            MPI.comm.Allreduce(self.data._data, self.data._buf, op=op)

    @collective
    def reduction_end(self):
        """End reduction for the argument if it is in flight.
        Doing a reduction only makes sense for :class:`Global` objects."""
        assert self._is_global, \
            "Doing global reduction only makes sense for Globals"
        if self.access is not READ and self._in_flight:
            self._in_flight = False
            # Must have a copy here, because otherwise we just grab a
            # pointer.
            self.data._data = np.copy(self.data._buf)

    @property
    def data(self):
        """Data carrier of this argument: :class:`Dat`, :class:`Mat`,
        :class:`Const` or :class:`Global`."""
        return self._dat


class Set(object):

    """OP2 set.

    :param size: The size of the set.
    :type size: integer or list of four integers.
    :param dim: The shape of the data associated with each element of this ``Set``.
    :type dim: integer or tuple of integers
    :param string name: The name of the set (optional).
    :param halo: An exisiting halo to use (optional).

    When the set is employed as an iteration space in a
    :func:`pyop2.op2.par_loop`, the extent of any local iteration space within
    each set entry is indicated in brackets. See the example in
    :func:`pyop2.op2.par_loop` for more details.

    The size of the set can either be an integer, or a list of four
    integers.  The latter case is used for running in parallel where
    we distinguish between:

      - `CORE` (owned and not touching halo)
      - `OWNED` (owned, touching halo)
      - `EXECUTE HALO` (not owned, but executed over redundantly)
      - `NON EXECUTE HALO` (not owned, read when executing in the execute halo)

    If a single integer is passed, we assume that we're running in
    serial and there is no distinction.

    The division of set elements is: ::

        [0, CORE)
        [CORE, OWNED)
        [OWNED, EXECUTE HALO)
        [EXECUTE HALO, NON EXECUTE HALO).

    Halo send/receive data is stored on sets in a :class:`Halo`.
    """

    _globalcount = 0

    _CORE_SIZE = 0
    _OWNED_SIZE = 1
    _IMPORT_EXEC_SIZE = 2
    _IMPORT_NON_EXEC_SIZE = 3

    @validate_type(('size', (int, tuple, list, np.ndarray), SizeTypeError),
                   ('name', str, NameTypeError))
    def __init__(self, size=None, name=None, halo=None):
        if type(size) is int:
            size = [size] * 4
        size = as_tuple(size, int, 4)
        assert size[Set._CORE_SIZE] <= size[Set._OWNED_SIZE] <= \
            size[Set._IMPORT_EXEC_SIZE] <= size[Set._IMPORT_NON_EXEC_SIZE], \
            "Set received invalid sizes: %s" % size
        self._core_size = size[Set._CORE_SIZE]
        self._size = size[Set._OWNED_SIZE]
        self._ieh_size = size[Set._IMPORT_EXEC_SIZE]
        self._inh_size = size[Set._IMPORT_NON_EXEC_SIZE]
        self._name = name or "set_%d" % Set._globalcount
        self._halo = halo
        self._partition_size = 1024
        self._extruded = False
        if self.halo:
            self.halo.verify(self)
        # A cache of objects built on top of this set
        self._cache = {}
        Set._globalcount += 1
        self.core_part = SetPartition(self, 0, self.core_size, name="core")
        self.owned_part = SetPartition(self, self.core_size, self.size - self.core_size, name="owned")
        self.exec_part = SetPartition(self, self.size, self.exec_size - self.size, name="exec")
        self.all_part = SetPartition(self, 0, self.exec_size, name="all")

    @property
    def core_size(self):
        """Core set size.  Owned elements not touching halo elements."""
        return self._core_size

    @property
    def size(self):
        """Set size, owned elements."""
        return self._size

    @property
    def exec_size(self):
        """Set size including execute halo elements.

        If a :class:`ParLoop` is indirect, we do redundant computation
        by executing over these set elements as well as owned ones.
        """
        return self._ieh_size

    @property
    def total_size(self):
        """Total set size, including halo elements."""
        return self._inh_size

    @property
    def sizes(self):
        """Set sizes: core, owned, execute halo, total."""
        return self._core_size, self._size, self._ieh_size, self._inh_size

    @property
    def name(self):
        """User-defined label"""
        return self._name

    @property
    def halo(self):
        """:class:`Halo` associated with this Set"""
        return self._halo

    @property
    def partition_size(self):
        """Default partition size"""
        return self._partition_size

    @partition_size.setter
    def partition_size(self, partition_value):
        """Set the partition size"""
        self._partition_size = partition_value

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    def __str__(self):
        return "OP2 Set: %s with size %s" % (self._name, self._size)

    def __repr__(self):
        return "Set(%r, %r)" % (self._size, self._name)

    def __call__(self, *indices):
        """Build a :class:`Subset` from this :class:`Set`

        :arg indices: The elements of this :class:`Set` from which the
                      :class:`Subset` should be formed.

        """
        if len(indices) == 1:
            indices = indices[0]
            if np.isscalar(indices):
                indices = [indices]
        return _make_object('Subset', self, indices)

    def __contains__(self, dset):
        """Indicate whether a given DataSet is compatible with this Set."""
        return dset.set is self

    def __pow__(self, e):
        """Derive a :class:`DataSet` with dimension ``e``"""
        return DataSet(self, dim=e)

    @property
    def layers(self):
        """Return None (not an :class:`ExtrudedSet`)."""
        return None

    @classmethod
    def fromhdf5(cls, f, name):
        """Construct a :class:`Set` from set named ``name`` in HDF5 data ``f``"""
        slot = f[name]
        if slot.shape != (1,):
            raise SizeTypeError("Shape of %s is incorrect" % name)
        size = slot.value.astype(np.int)
        return cls(size[0], name)


class ExtrudedSet(Set):

    """OP2 ExtrudedSet.

    :param parent: The parent :class:`Set` to build this :class:`ExtrudedSet` on top of
    :type parent: a :class:`Set`.
    :param layers: The number of layers in this :class:`ExtrudedSet`.
    :type layers: an integer.

    The number of layers indicates the number of time the base set is
    extruded in the direction of the :class:`ExtrudedSet`.  As a
    result, there are ``layers-1`` extruded "cells" in an extruded set.
    """

    @validate_type(('parent', Set, TypeError))
    def __init__(self, parent, layers):
        self._parent = parent
        if layers < 2:
            raise SizeTypeError("Number of layers must be > 1 (not %s)" % layers)
        self._layers = layers
        self._extruded = True
        self.core_part = SetPartition(self, 0, self.core_size, name="core")
        self.owned_part = SetPartition(self, self.core_size, self.size - self.core_size, name="owned")
        self.exec_part = SetPartition(self, self.size, self.exec_size - self.size, name="exec")
        self.all_part = SetPartition(self, 0, self.exec_size, name="all")

    def __getattr__(self, name):
        """Returns a :class:`Set` specific attribute."""
        return getattr(self._parent, name)

    def __contains__(self, set):
        return set is self.parent

    def __str__(self):
        return "OP2 ExtrudedSet: %s with size %s (%s layers)" % \
            (self._name, self._size, self._layers)

    def __repr__(self):
        return "ExtrudedSet(%r, %r)" % (self._parent, self._layers)

    @property
    def parent(self):
        return self._parent

    @property
    def layers(self):
        """The number of layers in this extruded set."""
        return self._layers


class Subset(ExtrudedSet):

    """OP2 subset.

    :param superset: The superset of the subset.
    :type superset: a :class:`Set` or a :class:`Subset`.
    :param indices: Elements of the superset that form the
        subset. Duplicate values are removed when constructing the subset.
    :type indices: a list of integers, or a numpy array.
    """
    @validate_type(('superset', Set, TypeError),
                   ('indices', (list, tuple, np.ndarray), TypeError))
    def __init__(self, superset, indices):
        # sort and remove duplicates
        indices = np.unique(indices)
        if isinstance(superset, Subset):
            # Unroll indices to point to those in the parent
            indices = superset.indices[indices]
            superset = superset.superset
        assert type(superset) is Set or type(superset) is ExtrudedSet, \
            'Subset construction failed, should not happen'

        self._superset = superset
        self._indices = verify_reshape(indices, np.int32, (len(indices),))

        if len(self._indices) > 0 and (self._indices[0] < 0 or
                                       self._indices[-1] >= self._superset.total_size):
            raise SubsetIndexOutOfBounds(
                'Out of bounds indices in Subset construction: [%d, %d) not [0, %d)' %
                (self._indices[0], self._indices[-1], self._superset.total_size))

        self._core_size = (self._indices < superset._core_size).sum()
        self._size = (self._indices < superset._size).sum()
        self._ieh_size = (self._indices < superset._ieh_size).sum()
        self._inh_size = len(self._indices)
        self.core_part = SetPartition(self, 0, self.core_size, name="core")
        self.owned_part = SetPartition(self, self.core_size, self.size - self.core_size, name="owned")
        self.exec_part = SetPartition(self, self.size, self.exec_size - self.size, name="exec")
        self.all_part = SetPartition(self, 0, self.exec_size, name="all")

    # Look up any unspecified attributes on the _set.
    def __getattr__(self, name):
        """Returns a :class:`Set` specific attribute."""
        return getattr(self._superset, name)

    def __pow__(self, e):
        """Derive a :class:`DataSet` with dimension ``e``"""
        raise NotImplementedError("Deriving a DataSet from a Subset is unsupported")

    def __str__(self):
        return "OP2 Subset: %s with size %s" % \
            (self._name, self._size)

    def __repr__(self):
        return "Subset(%r, %r)" % (self._superset, self._indices)

    def __call__(self, *indices):
        """Build a :class:`Subset` from this :class:`Subset`

        :arg indices: The elements of this :class:`Subset` from which the
                      :class:`Subset` should be formed.

        """
        if len(indices) == 1:
            indices = indices[0]
            if np.isscalar(indices):
                indices = [indices]
        return _make_object('Subset', self, indices)

    @property
    def superset(self):
        """Returns the superset Set"""
        return self._superset

    @property
    def indices(self):
        """Returns the indices pointing in the superset."""
        return self._indices

    @property
    def _argtype(self):
        """Ctypes argtype for this :class:`Subset`"""
        return np.ctypeslib.ctypes.c_voidp


class SetPartition(object):
    def __init__(self, set, offset, size, name):
        self.set = set
        self.offset = offset
        self.size = size
        self.name = name


class MixedSet(Set, ObjectCached):
    """A container for a bag of :class:`Set`\s."""

    def __init__(self, sets):
        """:param iterable sets: Iterable of :class:`Set`\s or :class:`ExtrudedSet`\s"""
        if self._initialized:
            return
        self._sets = sets
        assert all(s.layers == self._sets[0].layers for s in sets), \
            "All components of a MixedSet must have the same number of layers."
        self._initialized = True

    @classmethod
    def _process_args(cls, sets, **kwargs):
        sets = [s for s in sets]
        try:
            sets = as_tuple(sets, ExtrudedSet)
        except TypeError:
            sets = as_tuple(sets, Set)
        cache = sets[0]
        return (cache, ) + (sets, ), kwargs

    @classmethod
    def _cache_key(cls, sets, **kwargs):
        return sets

    def __getitem__(self, idx):
        """Return :class:`Set` with index ``idx`` or a given slice of sets."""
        return self._sets[idx]

    @property
    def split(self):
        """The underlying tuple of :class:`Set`\s."""
        return self._sets

    @property
    def core_size(self):
        """Core set size. Owned elements not touching halo elements."""
        return sum(s.core_size for s in self._sets)

    @property
    def size(self):
        """Set size, owned elements."""
        return sum(s.size for s in self._sets)

    @property
    def exec_size(self):
        """Set size including execute halo elements."""
        return sum(s.exec_size for s in self._sets)

    @property
    def total_size(self):
        """Total set size, including halo elements."""
        return sum(s.total_size for s in self._sets)

    @property
    def sizes(self):
        """Set sizes: core, owned, execute halo, total."""
        return (self.core_size, self.size, self.exec_size, self.total_size)

    @property
    def name(self):
        """User-defined labels."""
        return tuple(s.name for s in self._sets)

    @property
    def halo(self):
        """:class:`Halo`\s associated with these :class:`Set`\s."""
        halos = tuple(s.halo for s in self._sets)
        return halos if any(halos) else None

    @property
    def _extruded(self):
        return isinstance(self._sets[0], ExtrudedSet)

    @property
    def layers(self):
        """Numbers of layers in the extruded mesh (or None if this MixedSet is not extruded)."""
        return self._sets[0].layers

    def __iter__(self):
        """Yield all :class:`Set`\s when iterated over."""
        for s in self._sets:
            yield s

    def __len__(self):
        """Return number of contained :class:`Set`s."""
        return len(self._sets)

    def __pow__(self, e):
        """Derive a :class:`MixedDataSet` with dimensions ``e``"""
        return MixedDataSet(self._sets, e)

    def __str__(self):
        return "OP2 MixedSet composed of Sets: %s" % (self._sets,)

    def __repr__(self):
        return "MixedSet(%r)" % (self._sets,)


class DataSet(ObjectCached):
    """PyOP2 Data Set

    Set used in the op2.Dat structures to specify the dimension of the data.
    """
    _globalcount = 0

    @validate_type(('iter_set', Set, SetTypeError),
                   ('dim', (int, tuple, list), DimTypeError),
                   ('name', str, NameTypeError))
    def __init__(self, iter_set, dim=1, name=None):
        if self._initialized:
            return
        if isinstance(iter_set, Subset):
            raise NotImplementedError("Deriving a DataSet from a Subset is unsupported")
        self._set = iter_set
        self._dim = as_tuple(dim, int)
        self._cdim = np.asscalar(np.prod(self._dim))
        self._name = name or "dset_%d" % DataSet._globalcount
        DataSet._globalcount += 1
        self._initialized = True

    @classmethod
    def _process_args(cls, *args, **kwargs):
        return (args[0], ) + args, kwargs

    @classmethod
    def _cache_key(cls, iter_set, dim=1, name=None):
        return (iter_set, as_tuple(dim, int))

    def __getstate__(self):
        """Extract state to pickle."""
        return self.__dict__

    def __setstate__(self, d):
        """Restore from pickled state."""
        self.__dict__.update(d)

    # Look up any unspecified attributes on the _set.
    def __getattr__(self, name):
        """Returns a Set specific attribute."""
        return getattr(self.set, name)

    @property
    def dim(self):
        """The shape tuple of the values for each element of the set."""
        return self._dim

    @property
    def cdim(self):
        """The scalar number of values for each member of the set. This is
        the product of the dim tuple."""
        return self._cdim

    @property
    def name(self):
        """Returns the name of the data set."""
        return self._name

    @property
    def set(self):
        """Returns the parent set of the data set."""
        return self._set

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    def __str__(self):
        return "OP2 DataSet: %s on set %s, with dim %s" % \
            (self._name, self._set, self._dim)

    def __repr__(self):
        return "DataSet(%r, %r, %r)" % (self._set, self._dim, self._name)

    def __contains__(self, dat):
        """Indicate whether a given Dat is compatible with this DataSet."""
        return dat.dataset == self


class MixedDataSet(DataSet, ObjectCached):
    """A container for a bag of :class:`DataSet`\s.

    Initialized either from a :class:`MixedSet` and an iterable or iterator of
    ``dims`` of corresponding length ::

        mdset = op2.MixedDataSet(mset, [dim1, ..., dimN])

    or from a tuple of :class:`Set`\s and an iterable of ``dims`` of
    corresponding length ::

        mdset = op2.MixedDataSet([set1, ..., setN], [dim1, ..., dimN])

    If all ``dims`` are to be the same, they can also be given as an
    :class:`int` for either of above invocations ::

        mdset = op2.MixedDataSet(mset, dim)
        mdset = op2.MixedDataSet([set1, ..., setN], dim)

    Initialized from a :class:`MixedSet` without explicitly specifying ``dims``
    they default to 1 ::

        mdset = op2.MixedDataSet(mset)

    Initialized from an iterable or iterator of :class:`DataSet`\s and/or
    :class:`Set`\s, where :class:`Set`\s are implicitly upcast to
    :class:`DataSet`\s of dim 1 ::

        mdset = op2.MixedDataSet([dset1, ..., dsetN])
    """

    def __init__(self, arg, dims=None):
        """
        :param arg:  a :class:`MixedSet` or an iterable or a generator
                     expression of :class:`Set`\s or :class:`DataSet`\s or a
                     mixture of both
        :param dims: `None` (the default) or an :class:`int` or an iterable or
                     generator expression of :class:`int`\s, which **must** be
                     of same length as `arg`

        .. Warning ::
            When using generator expressions for ``arg`` or ``dims``, these
            **must** terminate or else will cause an infinite loop.
        """
        if self._initialized:
            return
        self._dsets = arg
        self._initialized = True

    @classmethod
    def _process_args(cls, arg, dims=None):
        # If the second argument is not None it is expect to be a scalar dim
        # or an iterable of dims and the first is expected to be a MixedSet or
        # an iterable of Sets
        if dims is not None:
            # If arg is a MixedSet, get its Sets tuple
            sets = arg.split if isinstance(arg, MixedSet) else tuple(arg)
            # If dims is a scalar, turn it into a tuple of right length
            dims = (dims,) * len(sets) if isinstance(dims, int) else tuple(dims)
            if len(sets) != len(dims):
                raise ValueError("Got MixedSet of %d Sets but %s dims" %
                                 (len(sets), len(dims)))
            dsets = tuple(s ** d for s, d in zip(sets, dims))
        # Otherwise expect the first argument to be an iterable of Sets and/or
        # DataSets and upcast Sets to DataSets as necessary
        else:
            arg = [s if isinstance(s, DataSet) else s ** 1 for s in arg]
            dsets = as_tuple(arg, type=DataSet)

        return (dsets[0].set, ) + (dsets, ), {}

    @classmethod
    def _cache_key(cls, arg, dims=None):
        return arg

    def __getitem__(self, idx):
        """Return :class:`DataSet` with index ``idx`` or a given slice of datasets."""
        return self._dsets[idx]

    @property
    def split(self):
        """The underlying tuple of :class:`DataSet`\s."""
        return self._dsets

    @property
    def dim(self):
        """The shape tuple of the values for each element of the sets."""
        return tuple(s.dim for s in self._dsets)

    @property
    def cdim(self):
        """The sum of the scalar number of values for each member of the sets.
        This is the sum of products of the dim tuples."""
        return sum(s.cdim for s in self._dsets)

    @property
    def name(self):
        """Returns the name of the data sets."""
        return tuple(s.name for s in self._dsets)

    @property
    def set(self):
        """Returns the :class:`MixedSet` this :class:`MixedDataSet` is
        defined on."""
        return MixedSet(s.set for s in self._dsets)

    def __iter__(self):
        """Yield all :class:`DataSet`\s when iterated over."""
        for ds in self._dsets:
            yield ds

    def __len__(self):
        """Return number of contained :class:`DataSet`s."""
        return len(self._dsets)

    def __str__(self):
        return "OP2 MixedDataSet composed of DataSets: %s" % (self._dsets,)

    def __repr__(self):
        return "MixedDataSet(%r)" % (self._dsets,)


class Halo(object):

    """A description of a halo associated with a :class:`Set`.

    The halo object describes which :class:`Set` elements are sent
    where, and which :class:`Set` elements are received from where.

    The `sends` should be a dict whose key is the process we want to
    send to, similarly the `receives` should be a dict whose key is the
    process we want to receive from.  The value should in each case be
    a numpy array of the set elements to send to/receive from each
    `process`.

    The gnn2unn array is a map from process-local set element
    numbering to cross-process set element numbering.  It must
    correctly number all the set elements in the halo region as well
    as owned elements.  Providing this array is only necessary if you
    will access :class:`Mat` objects on the :class:`Set` this `Halo`
    lives on.  Insertion into :class:`Dat`\s always uses process-local
    numbering, however insertion into :class:`Mat`\s uses cross-process
    numbering under the hood.
    """

    def __init__(self, sends, receives, comm=None, gnn2unn=None):
        # Fix up old style list of sends/receives into dict of sends/receives
        if not isinstance(sends, dict):
            tmp = {}
            for i, s in enumerate(sends):
                if len(s) > 0:
                    tmp[i] = s
            sends = tmp
        if not isinstance(receives, dict):
            tmp = {}
            for i, s in enumerate(receives):
                if len(s) > 0:
                    tmp[i] = s
            receives = tmp
        self._sends = sends
        self._receives = receives
        # The user might have passed lists, not numpy arrays, so fix that here.
        for i, a in self._sends.iteritems():
            self._sends[i] = np.asarray(a)
        for i, a in self._receives.iteritems():
            self._receives[i] = np.asarray(a)
        self._global_to_petsc_numbering = gnn2unn
        self._comm = _check_comm(comm) if comm is not None else MPI.comm
        # FIXME: is this a necessity?
        assert self._comm == MPI.comm, "Halo communicator not COMM"
        rank = self._comm.rank

        assert rank not in self._sends, \
            "Halo was specified with self-sends on rank %d" % rank
        assert rank not in self._receives, \
            "Halo was specified with self-receives on rank %d" % rank

    @property
    def sends(self):
        """Return the sends associated with this :class:`Halo`.

        A dict of numpy arrays, keyed by the rank to send to, with
        each array indicating the :class:`Set` elements to send.

        For example, to send no elements to rank 0, elements 1 and 2 to rank 1
        and no elements to rank 2 (with ``comm.size == 3``) we would have: ::

            {1: np.array([1,2], dtype=np.int32)}.
        """
        return self._sends

    @property
    def receives(self):
        """Return the receives associated with this :class:`Halo`.

        A dict of numpy arrays, keyed by the rank to receive from,
        with each array indicating the :class:`Set` elements to
        receive.

        See :func:`Halo.sends` for an example.
        """
        return self._receives

    @property
    def global_to_petsc_numbering(self):
        """The mapping from global (per-process) dof numbering to
    petsc (cross-process) dof numbering."""
        return self._global_to_petsc_numbering

    @property
    def comm(self):
        """The MPI communicator this :class:`Halo`'s communications
    should take place over"""
        return self._comm

    def verify(self, s):
        """Verify that this :class:`Halo` is valid for a given
:class:`Set`."""
        for dest, sends in self.sends.iteritems():
            assert (sends >= 0).all() and (sends < s.size).all(), \
                "Halo send to %d is invalid (outside owned elements)" % dest

        for source, receives in self.receives.iteritems():
            assert (receives >= s.size).all() and \
                (receives < s.total_size).all(), \
                "Halo receive from %d is invalid (not in halo elements)" % \
                source

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['_comm']
        return odict

    def __setstate__(self, d):
        self.__dict__.update(d)
        # Update old pickle dumps to new Halo format
        sends = self.__dict__['_sends']
        receives = self.__dict__['_receives']
        if not isinstance(sends, dict):
            tmp = {}
            for i, s in enumerate(sends):
                if len(s) > 0:
                    tmp[i] = s
            sends = tmp
        if not isinstance(receives, dict):
            tmp = {}
            for i, s in enumerate(receives):
                if len(s) > 0:
                    tmp[i] = s
            receives = tmp
        self._sends = sends
        self._receives = receives
        # FIXME: This will break for custom halo communicators
        self._comm = MPI.comm


class IterationSpace(object):

    """OP2 iteration space type.

    .. Warning ::
        User code should not directly instantiate :class:`IterationSpace`.
        This class is only for internal use inside a
        :func:`pyop2.op2.par_loop`."""

    @validate_type(('iterset', Set, SetTypeError))
    def __init__(self, iterset, block_shape=None):
        self._iterset = iterset
        if block_shape:
            # Try the Mat case first
            try:
                self._extents = (sum(b[0][0] for b in block_shape),
                                 sum(b[1] for b in block_shape[0]))
            # Otherwise it's a Dat and only has one extent
            except IndexError:
                self._extents = (sum(b[0][0] for b in block_shape),)
        else:
            self._extents = ()
        self._block_shape = block_shape or ((self._extents,),)

    @property
    def iterset(self):
        """The :class:`Set` over which this IterationSpace is defined."""
        return self._iterset

    @property
    def extents(self):
        """Extents of the IterationSpace within each item of ``iterset``"""
        return self._extents

    @property
    def name(self):
        """The name of the :class:`Set` over which this IterationSpace is
        defined."""
        return self._iterset.name

    @property
    def core_size(self):
        """The number of :class:`Set` elements which don't touch halo elements in the set
        over which this IterationSpace is defined"""
        return self._iterset.core_size

    @property
    def size(self):
        """The size of the :class:`Set` over which this IterationSpace is defined."""
        return self._iterset.size

    @property
    def exec_size(self):
        """The size of the :class:`Set` over which this IterationSpace
        is defined, including halo elements to be executed over"""
        return self._iterset.exec_size

    @property
    def layers(self):
        """Number of layers in the extruded set (or None if this is not an
        extruded iteration space)
        """
        return self._iterset.layers

    @property
    def _extruded(self):
        return self._iterset._extruded

    @property
    def partition_size(self):
        """Default partition size"""
        return self.iterset.partition_size

    @property
    def total_size(self):
        """The total size of :class:`Set` over which this IterationSpace is defined.

        This includes all halo set elements."""
        return self._iterset.total_size

    @property
    def _extent_ranges(self):
        return [e for e in self.extents]

    def __iter__(self):
        """Yield all block shapes with their indices as i, j, shape, offsets
        tuples."""
        roffset = 0
        for i, row in enumerate(self._block_shape):
            coffset = 0
            for j, shape in enumerate(row):
                yield i, j, shape, (roffset, coffset)
                if len(shape) > 1:
                    coffset += shape[1]
            if len(shape) > 0:
                roffset += shape[0]

    def __eq__(self, other):
        """:class:`IterationSpace`s compare equal if they are defined on the
        same :class:`Set` and have the same ``extent``."""
        return self._iterset == other._iterset and self._extents == other._extents

    def __ne__(self, other):
        """:class:`IterationSpace`s compare equal if they are defined on the
        same :class:`Set` and have the same ``extent``."""
        return not self == other

    def __str__(self):
        return "OP2 Iteration Space: %s with extents %s" % (self._iterset, self._extents)

    def __repr__(self):
        return "IterationSpace(%r, %r)" % (self._iterset, self._extents)

    @property
    def cache_key(self):
        """Cache key used to uniquely identify the object in the cache."""
        return self._extents, self._block_shape, self.iterset._extruded, \
            isinstance(self._iterset, Subset)


class DataCarrier(Versioned):

    """Abstract base class for OP2 data.

    Actual objects will be :class:`DataCarrier` objects of rank 0
    (:class:`Const` and :class:`Global`), rank 1 (:class:`Dat`), or rank 2
    (:class:`Mat`)"""

    class Snapshot(object):
        """A snapshot of the current state of the DataCarrier object. If
        is_valid() returns True, then the object hasn't changed since this
        snapshot was taken (and still exists)."""
        def __init__(self, obj):
            self._duplicate = obj.duplicate()
            self._original = weakref.ref(obj)

        def is_valid(self):
            objref = self._original()
            if objref is not None:
                return self._duplicate == objref
            return False

    def create_snapshot(self):
        """Returns a snapshot of the current object. If not overriden, this
        method will return a full duplicate object."""
        return type(self).Snapshot(self)

    @property
    def dtype(self):
        """The Python type of the data."""
        return self._data.dtype

    @property
    def ctype(self):
        """The c type of the data."""
        # FIXME: Complex and float16 not supported
        typemap = {"bool": "unsigned char",
                   "int": "int",
                   "int8": "char",
                   "int16": "short",
                   "int32": "int",
                   "int64": "long long",
                   "uint8": "unsigned char",
                   "uint16": "unsigned short",
                   "uint32": "unsigned int",
                   "uint64": "unsigned long",
                   "float": "double",
                   "float32": "float",
                   "float64": "double"}
        return typemap[self.dtype.name]

    @property
    def name(self):
        """User-defined label."""
        return self._name

    @property
    def dim(self):
        """The shape tuple of the values for each element of the object."""
        return self._dim

    @property
    def cdim(self):
        """The scalar number of values for each member of the object. This is
        the product of the dim tuple."""
        return self._cdim


class _EmptyDataMixin(object):
    """A mixin for :class:`Dat` and :class:`Global` objects that takes
    care of allocating data on demand if the user has passed nothing
    in.

    Accessing the :attr:`_data` property allocates a zeroed data array
    if it does not already exist.
    """
    def __init__(self, data, dtype, shape):
        if data is None:
            self._dtype = np.dtype(dtype if dtype is not None else np.float64)
            self._version_set_zero()
        else:
            self._data = verify_reshape(data, dtype, shape, allow_none=True)
            self._dtype = self._data.dtype

    @property
    def _data(self):
        """Return the user-provided data buffer, or a zeroed buffer of
        the correct size if none was provided."""
        if not self._is_allocated:
            self._numpy_data = np.zeros(self.shape, dtype=self._dtype)
        return self._numpy_data

    @_data.setter
    def _data(self, value):
        """Set the data buffer to `value`."""
        self._numpy_data = value

    @property
    def _is_allocated(self):
        """Return True if the data buffer has been allocated."""
        return hasattr(self, '_numpy_data')


class SetAssociated(DataCarrier):
    """Intermediate class between DataCarrier and subtypes associated with a
    Set (vectors and matrices)."""

    class Snapshot(object):
        """A snapshot for SetAssociated objects is valid if the snapshot
        version is the same as the current version of the object"""

        def __init__(self, obj):
            self._original = weakref.ref(obj)
            self._snapshot_version = obj._version

        def is_valid(self):
            objref = self._original()
            if objref is not None:
                return self._snapshot_version == objref._version
            return False


class Dat(SetAssociated, _EmptyDataMixin, CopyOnWrite):
    """OP2 vector data. A :class:`Dat` holds values on every element of a
    :class:`DataSet`.

    If a :class:`Set` is passed as the ``dataset`` argument, rather
    than a :class:`DataSet`, the :class:`Dat` is created with a default
    :class:`DataSet` dimension of 1.

    If a :class:`Dat` is passed as the ``dataset`` argument, a copy is
    returned.

    It is permissible to pass `None` as the `data` argument.  In this
    case, allocation of the data buffer is postponed until it is
    accessed.

    .. note::
        If the data buffer is not passed in, it is implicitly
        initialised to be zero.

    When a :class:`Dat` is passed to :func:`pyop2.op2.par_loop`, the map via
    which indirection occurs and the access descriptor are passed by
    calling the :class:`Dat`. For instance, if a :class:`Dat` named ``D`` is
    to be accessed for reading via a :class:`Map` named ``M``, this is
    accomplished by ::

      D(pyop2.READ, M)

    The :class:`Map` through which indirection occurs can be indexed
    using the index notation described in the documentation for the
    :class:`Map`. Direct access to a Dat is accomplished by
    omitting the path argument.

    :class:`Dat` objects support the pointwise linear algebra operations
    ``+=``, ``*=``, ``-=``, ``/=``, where ``*=`` and ``/=`` also support
    multiplication / division by a scalar.
    """

    _globalcount = 0
    _modes = [READ, WRITE, RW, INC]

    @validate_type(('dataset', (DataCarrier, DataSet, Set), DataSetTypeError),
                   ('name', str, NameTypeError))
    @validate_dtype(('dtype', None, DataTypeError))
    def __init__(self, dataset, data=None, dtype=None, name=None,
                 soa=None, uid=None):

        if isinstance(dataset, Dat):
            self.__init__(dataset.dataset, None, dtype=dataset.dtype,
                          name="copy_of_%s" % dataset.name, soa=dataset.soa)
            dataset.copy(self)
            return
        if type(dataset) is Set or type(dataset) is ExtrudedSet:
            # If a Set, rather than a dataset is passed in, default to
            # a dataset dimension of 1.
            dataset = dataset ** 1
        self._shape = (dataset.total_size,) + (() if dataset.cdim == 1 else dataset.dim)
        _EmptyDataMixin.__init__(self, data, dtype, self._shape)

        self._dataset = dataset
        # Are these data to be treated as SoA on the device?
        self._soa = bool(soa)
        self._needs_halo_update = False
        # If the uid is not passed in from outside, assume that Dats
        # have been declared in the same order everywhere.
        if uid is None:
            self._id = Dat._globalcount
            Dat._globalcount += 1
        else:
            self._id = uid
        self._name = name or "dat_%d" % self._id
        halo = dataset.halo
        if halo is not None:
            self._send_reqs = {}
            self._send_buf = {}
            self._recv_reqs = {}
            self._recv_buf = {}

    @validate_in(('access', _modes, ModeValueError))
    def __call__(self, access, path=None, flatten=False):
        if isinstance(path, Arg):
            return _make_object('Arg', data=self, map=path.map, idx=path.idx,
                                access=access, flatten=flatten)
        if path and path.toset != self.dataset.set:
            raise MapValueError("To Set of Map does not match Set of Dat.")
        return _make_object('Arg', data=self, map=path, access=access, flatten=flatten)

    def __getitem__(self, idx):
        """Return self if ``idx`` is 0, raise an error otherwise."""
        if idx != 0:
            raise IndexValueError("Can only extract component 0 from %r" % self)
        return self

    @property
    def split(self):
        """Tuple containing only this :class:`Dat`."""
        return (self,)

    @property
    def dataset(self):
        """:class:`DataSet` on which the Dat is defined."""
        return self._dataset

    @property
    def dim(self):
        """The shape of the values for each element of the object."""
        return self.dataset.dim

    @property
    def cdim(self):
        """The scalar number of values for each member of the object. This is
        the product of the dim tuple."""
        return self.dataset.cdim

    @property
    def soa(self):
        """Are the data in SoA format?"""
        return self._soa

    @property
    def _argtype(self):
        """Ctypes argtype for this :class:`Dat`"""
        return np.ctypeslib.ctypes.c_voidp

    @property
    @modifies
    @collective
    def data(self):
        """Numpy array containing the data values.

        With this accessor you are claiming that you will modify
        the values you get back.  If you only need to look at the
        values, use :meth:`data_ro` instead.

        This only shows local values, to see the halo values too use
        :meth:`data_with_halos`.

        """
        if self.dataset.total_size > 0 and self._data.size == 0 and self.cdim > 0:
            raise RuntimeError("Illegal access: no data associated with this Dat!")
        maybe_setflags(self._data, write=True)
        v = self._data[:self.dataset.size].view()
        self.needs_halo_update = True
        return v

    @property
    @collective
    def data_with_halos(self):
        """A view of this :class:`Dat`\s data.

        This accessor marks the :class:`Dat` as dirty, see
        :meth:`data` for more details on the semantics.

        With this accessor, you get to see up to date halo values, but
        you should not try and modify them, because they will be
        overwritten by the next halo exchange."""
        self.data               # force evaluation
        self.halo_exchange_begin()
        self.halo_exchange_end()
        self.needs_halo_update = True
        maybe_setflags(self._data, write=True)
        return self._data

    @property
    @collective
    def data_ro(self):
        """Numpy array containing the data values.  Read-only.

        With this accessor you are not allowed to modify the values
        you get back.  If you need to do so, use :meth:`data` instead.

        This only shows local values, to see the halo values too use
        :meth:`data_ro_with_halos`.

        """
        if self.dataset.total_size > 0 and self._data.size == 0 and self.cdim > 0:
            raise RuntimeError("Illegal access: no data associated with this Dat!")
        v = self._data[:self.dataset.size].view()
        v.setflags(write=False)
        return v

    @property
    @collective
    def data_ro_with_halos(self):
        """A view of this :class:`Dat`\s data.

        This accessor does not mark the :class:`Dat` as dirty, and is
        a read only view, see :meth:`data_ro` for more details on the
        semantics.

        With this accessor, you get to see up to date halo values, but
        you should not try and modify them, because they will be
        overwritten by the next halo exchange.

        """
        self.data_ro            # force evaluation
        self.halo_exchange_begin()
        self.halo_exchange_end()
        self.needs_halo_update = False
        v = self._data.view()
        v.setflags(write=False)
        return v

    def save(self, filename):
        """Write the data array to file ``filename`` in NumPy format."""
        np.save(filename, self.data_ro)

    def load(self, filename):
        """Read the data stored in file ``filename`` into a NumPy array
        and store the values in :meth:`_data`.
        """
        # The np.save method appends a .npy extension to the file name
        # if the user has not supplied it. However, np.load does not,
        # so we need to handle this ourselves here.
        if(filename[-4:] != ".npy"):
            filename = filename + ".npy"

        if isinstance(self.data, tuple):
            # MixedDat case
            for d, d_from_file in zip(self.data, np.load(filename)):
                d[:] = d_from_file[:]
        else:
            self.data[:] = np.load(filename)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def nbytes(self):
        """Return an estimate of the size of the data associated with this
        :class:`Dat` in bytes. This will be the correct size of the data
        payload, but does not take into account the (presumably small)
        overhead of the object and its metadata.

        Note that this is the process local memory usage, not the sum
        over all MPI processes.
        """

        return self.dtype.itemsize * self.dataset.total_size * self.dataset.cdim

    @property
    def needs_halo_update(self):
        '''Has this :class:`Dat` been written to since the last halo exchange?'''
        return self._needs_halo_update

    @needs_halo_update.setter
    @collective
    def needs_halo_update(self, val):
        """Indictate whether this Dat requires a halo update"""
        self._needs_halo_update = val

    @zeroes
    @collective
    def zero(self):
        """Zero the data associated with this :class:`Dat`"""
        if not hasattr(self, '_zero_parloop'):
            k = ast.FunDecl("void", "zero",
                            [ast.Decl(self.ctype, ast.Symbol("*self"))],
                            body=ast.c_for("n", self.cdim,
                                           ast.Assign(ast.Symbol("self", ("n", )),
                                                      ast.FlatBlock("(%s)0" % self.ctype)),
                                           pragma=None))
            k = _make_object('Kernel', k, 'zero')
            self._zero_parloop = _make_object('ParLoop', k, self.dataset.set, self(WRITE))
        self._zero_parloop._run()

    @modifies_argn(0)
    @collective
    def copy(self, other, subset=None):
        """Copy the data in this :class:`Dat` into another.

        :arg other: The destination :class:`Dat`
        :arg subset: A :class:`Subset` of elements to copy (optional)"""

        self._copy_parloop(other, subset=subset)._run()

    @collective
    def _copy_parloop(self, other, subset=None):
        """Create the :class:`ParLoop` implementing copy."""
        if not hasattr(self, '_copy_kernel'):
            k = ast.FunDecl("void", "copy",
                            [ast.Decl(self.ctype, ast.Symbol("*self"),
                                      qualifiers=["const"]),
                             ast.Decl(other.ctype, ast.Symbol("*other"))],
                            body=ast.c_for("n", self.cdim,
                                           ast.Assign(ast.Symbol("other", ("n", )),
                                                      ast.Symbol("self", ("n", ))),
                                           pragma=None))
            self._copy_kernel = _make_object('Kernel', k, 'copy')
        return _make_object('ParLoop', self._copy_kernel,
                            subset or self.dataset.set,
                            self(READ), other(WRITE))

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    def __eq__(self, other):
        """:class:`Dat`\s compare equal if defined on the same
        :class:`DataSet` and containing the same data."""
        try:
            if self._is_allocated and other._is_allocated:
                return (self._dataset == other._dataset and
                        self.dtype == other.dtype and
                        np.array_equal(self._data, other._data))
            elif not (self._is_allocated or other._is_allocated):
                return (self._dataset == other._dataset and
                        self.dtype == other.dtype)
            return False
        except AttributeError:
            return False

    def __ne__(self, other):
        """:class:`Dat`\s compare equal if defined on the same
        :class:`DataSet` and containing the same data."""
        return not self == other

    @collective
    def _cow_actual_copy(self, src):
        # Force the execution of the copy parloop

        # We need to ensure that PyOP2 allocates fresh storage for this copy.
        try:
            del self._numpy_data
        except AttributeError:
            pass

        self._cow_parloop._run()

    @collective
    def _cow_shallow_copy(self):

        other = shallow_copy(self)

        # Set up the copy to happen when required.
        other._cow_parloop = self._copy_parloop(other)
        # Remove the write dependency of the copy (in order to prevent
        # premature execution of the loop).
        other._cow_parloop.writes = set()
        return other

    def __str__(self):
        return "OP2 Dat: %s on (%s) with datatype %s" \
               % (self._name, self._dataset, self.dtype.name)

    def __repr__(self):
        return "Dat(%r, None, %r, %r)" \
               % (self._dataset, self.dtype, self._name)

    def _check_shape(self, other):
        if other.dataset != self.dataset:
            raise ValueError('Mismatched shapes in operands %s and %s' %
                             self.dataset.dim, other.dataset.dim)

    def _op(self, other, op):
        ops = {operator.add: '+',
               operator.sub: '-',
               operator.mul: '*',
               operator.div: '/'}
        ret = _make_object('Dat', self.dataset, None, self.dtype)
        name = "binop_%s" % op.__name__
        if np.isscalar(other):
            other = _make_object('Global', 1, data=other)
            k = ast.FunDecl("void", name,
                            [ast.Decl(self.ctype, ast.Symbol("*self"),
                                      qualifiers=["const"]),
                             ast.Decl(other.ctype, ast.Symbol("*other"),
                                      qualifiers=["const"]),
                             ast.Decl(self.ctype, ast.Symbol("*ret"))],
                            ast.c_for("n", self.cdim,
                                      ast.Assign(ast.Symbol("ret", ("n", )),
                                                 ast.BinExpr(ast.Symbol("self", ("n", )),
                                                             ast.Symbol("other", ("0", )),
                                                             op=ops[op])),
                                      pragma=None))

            k = _make_object('Kernel', k, name)
        else:
            self._check_shape(other)
            k = ast.FunDecl("void", name,
                            [ast.Decl(self.ctype, ast.Symbol("*self"),
                                      qualifiers=["const"]),
                             ast.Decl(other.ctype, ast.Symbol("*other"),
                                      qualifiers=["const"]),
                             ast.Decl(self.ctype, ast.Symbol("*ret"))],
                            ast.c_for("n", self.cdim,
                                      ast.Assign(ast.Symbol("ret", ("n", )),
                                                 ast.BinExpr(ast.Symbol("self", ("n", )),
                                                             ast.Symbol("other", ("n", )),
                                                             op=ops[op])),
                                      pragma=None))

            k = _make_object('Kernel', k, name)
        par_loop(k, self.dataset.set, self(READ), other(READ), ret(WRITE))
        return ret

    @modifies
    def _iop(self, other, op):
        ops = {operator.iadd: ast.Incr,
               operator.isub: ast.Decr,
               operator.imul: ast.IMul,
               operator.idiv: ast.IDiv}
        name = "iop_%s" % op.__name__
        if np.isscalar(other):
            other = _make_object('Global', 1, data=other)
            k = ast.FunDecl("void", name,
                            [ast.Decl(self.ctype, ast.Symbol("*self")),
                             ast.Decl(other.ctype, ast.Symbol("*other"),
                                      qualifiers=["const"])],
                            ast.c_for("n", self.cdim,
                                      ops[op](ast.Symbol("self", ("n", )),
                                              ast.Symbol("other", ("0", ))),
                                      pragma=None))
            k = _make_object('Kernel', k, name)
        else:
            self._check_shape(other)
            quals = ["const"] if self is not other else []
            k = ast.FunDecl("void", name,
                            [ast.Decl(self.ctype, ast.Symbol("*self")),
                             ast.Decl(other.ctype, ast.Symbol("*other"),
                                      qualifiers=quals)],
                            ast.c_for("n", self.cdim,
                                      ops[op](ast.Symbol("self", ("n", )),
                                              ast.Symbol("other", ("n", ))),
                                      pragma=None))
            k = _make_object('Kernel', k, name)
        par_loop(k, self.dataset.set, self(INC), other(READ))
        return self

    def _uop(self, op):
        ops = {operator.sub: ast.Neg}
        name = "uop_%s" % op.__name__
        k = ast.FunDecl("void", name,
                        [ast.Decl(self.ctype, ast.Symbol("*self"))],
                        ast.c_for("n", self.cdim,
                                  ast.Assign(ast.Symbol("self", ("n", )),
                                             ops[op](ast.Symbol("self", ("n", )))),
                                  pragma=None))
        k = _make_object('Kernel', k, name)
        par_loop(k, self.dataset.set, self(RW))
        return self

    def inner(self, other):
        """Compute the l2 inner product of the flattened :class:`Dat`

        :arg other: the other :class:`Dat` to compute the inner
             product against.

        """
        self._check_shape(other)
        ret = _make_object('Global', 1, data=0, dtype=self.dtype)

        k = ast.FunDecl("void", "inner",
                        [ast.Decl(self.ctype, ast.Symbol("*self"),
                                  qualifiers=["const"]),
                         ast.Decl(other.ctype, ast.Symbol("*other"),
                                  qualifiers=["const"]),
                         ast.Decl(self.ctype, ast.Symbol("*ret"))],
                        ast.c_for("n", self.cdim,
                                  ast.Incr(ast.Symbol("ret", (0, )),
                                           ast.Prod(ast.Symbol("self", ("n", )),
                                                    ast.Symbol("other", ("n", )))),
                                  pragma=None))
        k = _make_object('Kernel', k, "inner")
        par_loop(k, self.dataset.set, self(READ), other(READ), ret(INC))
        return ret.data_ro[0]

    @property
    def norm(self):
        """Compute the l2 norm of this :class:`Dat`

        .. note::

           This acts on the flattened data (see also :meth:`inner`)."""
        from math import sqrt
        return sqrt(self.inner(self))

    def __pos__(self):
        pos = _make_object('Dat', self)
        return pos

    def __add__(self, other):
        """Pointwise addition of fields."""
        return self._op(other, operator.add)

    def __radd__(self, other):
        """Pointwise addition of fields.

        self.__radd__(other) <==> other + self."""
        return self + other

    def __neg__(self):
        neg = _make_object('Dat', self)
        return neg._uop(operator.sub)

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

    def __div__(self, other):
        """Pointwise division or scaling of fields."""
        return self._op(other, operator.div)

    def __iadd__(self, other):
        """Pointwise addition of fields."""
        return self._iop(other, operator.iadd)

    def __isub__(self, other):
        """Pointwise subtraction of fields."""
        return self._iop(other, operator.isub)

    def __imul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        return self._iop(other, operator.imul)

    def __idiv__(self, other):
        """Pointwise division or scaling of fields."""
        return self._iop(other, operator.idiv)

    @collective
    def halo_exchange_begin(self):
        """Begin halo exchange."""
        halo = self.dataset.halo
        if halo is None:
            return
        for dest, ele in halo.sends.iteritems():
            self._send_buf[dest] = self._data[ele]
            self._send_reqs[dest] = halo.comm.Isend(self._send_buf[dest],
                                                    dest=dest, tag=self._id)
        for source, ele in halo.receives.iteritems():
            self._recv_buf[source] = self._data[ele]
            self._recv_reqs[source] = halo.comm.Irecv(self._recv_buf[source],
                                                      source=source, tag=self._id)

    @collective
    def halo_exchange_end(self):
        """End halo exchange. Waits on MPI recv."""
        halo = self.dataset.halo
        if halo is None:
            return
        with timed_region("Halo exchange receives wait"):
            _MPI.Request.Waitall(self._recv_reqs.values())
        with timed_region("Halo exchange sends wait"):
            _MPI.Request.Waitall(self._send_reqs.values())
        self._recv_reqs.clear()
        self._send_reqs.clear()
        self._send_buf.clear()
        # data is read-only in a ParLoop, make it temporarily writable
        maybe_setflags(self._data, write=True)
        for source, buf in self._recv_buf.iteritems():
            self._data[halo.receives[source]] = buf
        maybe_setflags(self._data, write=False)
        self._recv_buf.clear()

    @classmethod
    def fromhdf5(cls, dataset, f, name):
        """Construct a :class:`Dat` from a Dat named ``name`` in HDF5 data ``f``"""
        slot = f[name]
        data = slot.value
        soa = slot.attrs['type'].find(':soa') > 0
        ret = cls(dataset, data, name=name, soa=soa)
        return ret


class MixedDat(Dat):
    """A container for a bag of :class:`Dat`\s.

    Initialized either from a :class:`MixedDataSet`, a :class:`MixedSet`, or
    an iterable of :class:`DataSet`\s and/or :class:`Set`\s, where all the
    :class:`Set`\s are implcitly upcast to :class:`DataSet`\s ::

        mdat = op2.MixedDat(mdset)
        mdat = op2.MixedDat([dset1, ..., dsetN])

    or from an iterable of :class:`Dat`\s ::

        mdat = op2.MixedDat([dat1, ..., datN])
    """

    def __init__(self, mdset_or_dats):
        if isinstance(mdset_or_dats, MixedDat):
            self._dats = tuple(_make_object('Dat', d) for d in mdset_or_dats)
            return
        self._dats = tuple(d if isinstance(d, Dat) else _make_object('Dat', d)
                           for d in mdset_or_dats)
        if not all(d.dtype == self._dats[0].dtype for d in self._dats):
            raise DataValueError('MixedDat with different dtypes is not supported')

    def __getitem__(self, idx):
        """Return :class:`Dat` with index ``idx`` or a given slice of Dats."""
        return self._dats[idx]

    @property
    def dtype(self):
        """The NumPy dtype of the data."""
        return self._dats[0].dtype

    @property
    def split(self):
        """The underlying tuple of :class:`Dat`\s."""
        return self._dats

    @property
    def dataset(self):
        """:class:`MixedDataSet`\s this :class:`MixedDat` is defined on."""
        return MixedDataSet(tuple(s.dataset for s in self._dats))

    @property
    def soa(self):
        """Are the data in SoA format?"""
        return tuple(s.soa for s in self._dats)

    @property
    def _data(self):
        """Return the user-provided data buffer, or a zeroed buffer of
        the correct size if none was provided."""
        return tuple(d._data for d in self)

    @property
    @collective
    def data(self):
        """Numpy arrays containing the data excluding halos."""
        return tuple(s.data for s in self._dats)

    @property
    @collective
    def data_with_halos(self):
        """Numpy arrays containing the data including halos."""
        return tuple(s.data_with_halos for s in self._dats)

    @property
    @collective
    def data_ro(self):
        """Numpy arrays with read-only data excluding halos."""
        return tuple(s.data_ro for s in self._dats)

    @property
    @collective
    def data_ro_with_halos(self):
        """Numpy arrays with read-only data including halos."""
        return tuple(s.data_ro_with_halos for s in self._dats)

    @property
    def needs_halo_update(self):
        """Has this Dat been written to since the last halo exchange?"""
        return any(s.needs_halo_update for s in self._dats)

    @needs_halo_update.setter
    def needs_halo_update(self, val):
        """Indictate whether this Dat requires a halo update"""
        for d in self._dats:
            d.needs_halo_update = val

    @collective
    def halo_exchange_begin(self):
        for s in self._dats:
            s.halo_exchange_begin()

    @collective
    def halo_exchange_end(self):
        for s in self._dats:
            s.halo_exchange_end()

    @collective
    def zero(self):
        """Zero the data associated with this :class:`MixedDat`."""
        for d in self._dats:
            d.zero()

    @property
    def nbytes(self):
        """Return an estimate of the size of the data associated with this
        :class:`MixedDat` in bytes. This will be the correct size of the data
        payload, but does not take into account the (presumably small)
        overhead of the object and its metadata.

        Note that this is the process local memory usage, not the sum
        over all MPI processes.
        """

        return np.sum([d.nbytes for d in self._dats])

    @collective
    def copy(self, other, subset=None):
        """Copy the data in this :class:`MixedDat` into another.

        :arg other: The destination :class:`MixedDat`
        :arg subset: Subsets are not supported, this must be :class:`None`"""

        if subset is not None:
            raise NotImplementedError("MixedDat.copy with a Subset is not supported")
        for s, o in zip(self, other):
            s._copy_parloop(o)._run()

    @collective
    def _cow_actual_copy(self, src):
        # Force the execution of the copy parloop

        for d, s in zip(self._dats, src._dats):
            d._cow_actual_copy(d, s)

    @collective
    def _cow_shallow_copy(self):

        other = shallow_copy(self)

        other._dats = [d._cow_shallow_copy for d in self._dats]

        return other

    def __iter__(self):
        """Yield all :class:`Dat`\s when iterated over."""
        for d in self._dats:
            yield d

    def __len__(self):
        """Return number of contained :class:`Dats`\s."""
        return len(self._dats)

    def __eq__(self, other):
        """:class:`MixedDat`\s are equal if all their contained :class:`Dat`\s
        are."""
        try:
            return self._dats == other._dats
        # Deal with the case of comparing to a different type
        except AttributeError:
            return False

    def __ne__(self, other):
        """:class:`MixedDat`\s are equal if all their contained :class:`Dat`\s
        are."""
        return not self == other

    def __str__(self):
        return "OP2 MixedDat composed of Dats: %s" % (self._dats,)

    def __repr__(self):
        return "MixedDat(%r)" % (self._dats,)

    def inner(self, other):
        """Compute the l2 inner product.

        :arg other: the other :class:`MixedDat` to compute the inner product against"""
        ret = 0
        for s, o in zip(self, other):
            ret += s.inner(o)
        return ret

    def _op(self, other, op):
        ret = []
        if np.isscalar(other):
            for s in self:
                ret.append(op(s, other))
        else:
            self._check_shape(other)
            for s, o in zip(self, other):
                ret.append(op(s, o))
        return _make_object('MixedDat', ret)

    def _iop(self, other, op):
        if np.isscalar(other):
            for s in self:
                op(s, other)
        else:
            self._check_shape(other)
            for s, o in zip(self, other):
                op(s, o)
        return self

    def __pos__(self):
        ret = []
        for s in self:
            ret.append(s.__pos__())
        return _make_object('MixedDat', ret)

    def __neg__(self):
        ret = []
        for s in self:
            ret.append(s.__neg__())
        return _make_object('MixedDat', ret)

    def __add__(self, other):
        """Pointwise addition of fields."""
        return self._op(other, operator.add)

    def __radd__(self, other):
        """Pointwise addition of fields.

        self.__radd__(other) <==> other + self."""
        return self._op(other, operator.add)

    def __sub__(self, other):
        """Pointwise subtraction of fields."""
        return self._op(other, operator.sub)

    def __rsub__(self, other):
        """Pointwise subtraction of fields.

        self.__rsub__(other) <==> other - self."""
        return self._op(other, operator.sub)

    def __mul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        return self._op(other, operator.mul)

    def __rmul__(self, other):
        """Pointwise multiplication or scaling of fields.

        self.__rmul__(other) <==> other * self."""
        return self._op(other, operator.mul)

    def __div__(self, other):
        """Pointwise division or scaling of fields."""
        return self._op(other, operator.div)

    def __iadd__(self, other):
        """Pointwise addition of fields."""
        return self._iop(other, operator.iadd)

    def __isub__(self, other):
        """Pointwise subtraction of fields."""
        return self._iop(other, operator.isub)

    def __imul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        return self._iop(other, operator.imul)

    def __idiv__(self, other):
        """Pointwise division or scaling of fields."""
        return self._iop(other, operator.idiv)


class Const(DataCarrier):

    """Data that is constant for any element of any set."""

    class Snapshot(object):
        """Overridden from DataCarrier; a snapshot is always valid as long as
        the Const object still exists"""
        def __init__(self, obj):
            self._original = weakref.ref(obj)

        def is_valid(self):
            objref = self._original()
            if objref is not None:
                return True
            return False

    class NonUniqueNameError(ValueError):

        """The Names of const variables are required to be globally unique.
        This exception is raised if the name is already in use."""

    _defs = set()
    _globalcount = 0

    @validate_type(('name', str, NameTypeError))
    def __init__(self, dim, data=None, name=None, dtype=None):
        self._dim = as_tuple(dim, int)
        self._cdim = np.asscalar(np.prod(self._dim))
        self._data = verify_reshape(data, dtype, self._dim, allow_none=True)
        self._name = name or "const_%d" % Const._globalcount
        if any(self._name is const._name for const in Const._defs):
            raise Const.NonUniqueNameError(
                "OP2 Constants are globally scoped, %s is already in use" % self._name)
        Const._defs.add(self)
        Const._globalcount += 1

    def duplicate(self):
        """A Const duplicate can always refer to the same data vector, since
        it's read-only"""
        return type(self)(self.dim, data=self._data, dtype=self.dtype, name=self.name)

    @property
    def _argtype(self):
        """Ctypes argtype for this :class:`Const`"""
        return np.ctypeslib.ctypes.c_voidp

    @property
    def data(self):
        """Data array."""
        if len(self._data) is 0:
            raise RuntimeError("Illegal access: No data associated with this Const!")
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    def __str__(self):
        return "OP2 Const: %s of dim %s and type %s with value %s" \
               % (self._name, self._dim, self._data.dtype.name, self._data)

    def __repr__(self):
        return "Const(%r, %r, %r)" \
               % (self._dim, self._data, self._name)

    @classmethod
    def _definitions(cls):
        return sorted(Const._defs, key=lambda c: c.name)

    def remove_from_namespace(self):
        """Remove this Const object from the namespace

        This allows the same name to be redeclared with a different shape."""
        Const._defs.discard(self)

    def _format_declaration(self):
        d = {'type': self.ctype,
             'name': self.name,
             'dim': self.cdim}

        if self.cdim == 1:
            return "static %(type)s %(name)s;" % d

        return "static %(type)s %(name)s[%(dim)s];" % d

    @classmethod
    def fromhdf5(cls, f, name):
        """Construct a :class:`Const` from const named ``name`` in HDF5 data ``f``"""
        slot = f[name]
        dim = slot.shape
        data = slot.value
        if len(dim) < 1:
            raise DimTypeError("Invalid dimension value %s" % dim)
        return cls(dim, data, name)


class Global(DataCarrier, _EmptyDataMixin):

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

    _globalcount = 0
    _modes = [READ, INC, MIN, MAX]

    @validate_type(('name', str, NameTypeError))
    def __init__(self, dim, data=None, dtype=None, name=None):
        self._dim = as_tuple(dim, int)
        self._cdim = np.asscalar(np.prod(self._dim))
        _EmptyDataMixin.__init__(self, data, dtype, self._dim)
        self._buf = np.empty(self.shape, dtype=self.dtype)
        self._name = name or "global_%d" % Global._globalcount
        Global._globalcount += 1

    @validate_in(('access', _modes, ModeValueError))
    def __call__(self, access, path=None, flatten=False):
        """Note that the flatten argument is only passed in order to
        have the same interface as :class:`Dat`. Its value is
        ignored."""
        return _make_object('Arg', data=self, access=access)

    def __eq__(self, other):
        """:class:`Global`\s compare equal when having the same ``dim`` and
        ``data``."""
        try:
            return (self._dim == other._dim and
                    np.array_equal(self._data, other._data))
        except AttributeError:
            return False

    def __ne__(self, other):
        """:class:`Global`\s compare equal when having the same ``dim`` and
        ``data``."""
        return not self == other

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    def __getitem__(self, idx):
        """Return self if ``idx`` is 0, raise an error otherwise."""
        if idx != 0:
            raise IndexValueError("Can only extract component 0 from %r" % self)
        return self

    def __str__(self):
        return "OP2 Global Argument: %s with dim %s and value %s" \
            % (self._name, self._dim, self._data)

    def __repr__(self):
        return "Global(%r, %r, %r, %r)" % (self._dim, self._data,
                                           self._data.dtype, self._name)

    @property
    def _argtype(self):
        """Ctypes argtype for this :class:`Global`"""
        return np.ctypeslib.ctypes.c_voidp

    @property
    def shape(self):
        return self._dim

    @property
    def data(self):
        """Data array."""
        if len(self._data) is 0:
            raise RuntimeError("Illegal access: No data associated with this Global!")
        return self._data

    @property
    def dtype(self):
        return self._dtype

    @property
    def data_ro(self):
        """Data array."""
        view = self.data.view()
        view.setflags(write=False)
        return view

    @data.setter
    @modifies
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)

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

    @property
    def soa(self):
        """Are the data in SoA format? This is always false for :class:`Global`
        objects."""
        return False

    def duplicate(self):
        """Return a deep copy of self."""
        return type(self)(self.dim, data=np.copy(self.data_ro),
                          dtype=self.dtype, name=self.name)


# FIXME: Part of kernel API, but must be declared before Map for the validation.


class IterationIndex(object):

    """OP2 iteration space index

    Users should not directly instantiate :class:`IterationIndex` objects. Use
    ``op2.i`` instead."""

    def __init__(self, index=None):
        assert index is None or isinstance(index, int), "i must be an int"
        self._index = index

    def __str__(self):
        return "OP2 IterationIndex: %s" % self._index

    def __repr__(self):
        return "IterationIndex(%r)" % self._index

    @property
    def index(self):
        """Return the integer value of this index."""
        return self._index

    def __getitem__(self, idx):
        return IterationIndex(idx)

    # This is necessary so that we can convert an IterationIndex to a
    # tuple.  Because, __getitem__ returns a new IterationIndex
    # we have to explicitly provide an iterable interface
    def __iter__(self):
        """Yield self when iterated over."""
        yield self

i = IterationIndex()
"""Shorthand for constructing :class:`IterationIndex` objects.

``i[idx]`` builds an :class:`IterationIndex` object for which the `index`
property is `idx`.
"""


class Map(object):

    """OP2 map, a relation between two :class:`Set` objects.

    Each entry in the ``iterset`` maps to ``arity`` entries in the
    ``toset``. When a map is used in a :func:`pyop2.op2.par_loop`, it is
    possible to use Python index notation to select an individual entry on the
    right hand side of this map. There are three possibilities:

    * No index. All ``arity`` :class:`Dat` entries will be passed to the
      kernel.
    * An integer: ``some_map[n]``. The ``n`` th entry of the
      map result will be passed to the kernel.
    * An :class:`IterationIndex`, ``some_map[pyop2.i[n]]``. ``n``
      will take each value from ``0`` to ``e-1`` where ``e`` is the
      ``n`` th extent passed to the iteration space for this
      :func:`pyop2.op2.par_loop`. See also :data:`i`.


    For extruded problems (where ``iterset`` is an
    :class:`ExtrudedSet`) with boundary conditions applied at the top
    and bottom of the domain, one needs to provide a list of which of
    the `arity` values in each map entry correspond to values on the
    bottom boundary and which correspond to the top.  This is done by
    supplying two lists of indices in `bt_masks`, the first provides
    indices for the bottom, the second for the top.

    """

    _globalcount = 0

    @validate_type(('iterset', Set, SetTypeError), ('toset', Set, SetTypeError),
                   ('arity', int, ArityTypeError), ('name', str, NameTypeError))
    def __init__(self, iterset, toset, arity, values=None, name=None, offset=None, parent=None, bt_masks=None):
        self._iterset = iterset
        self._toset = toset
        self._arity = arity
        self._values = verify_reshape(values, np.int32, (iterset.total_size, arity),
                                      allow_none=True)
        self._name = name or "map_%d" % Map._globalcount
        self._offset = offset
        # This is intended to be used for modified maps, for example
        # where a boundary condition is imposed by setting some map
        # entries negative.
        self._parent = parent
        # A cache for objects built on top of this map
        self._cache = {}
        # Which indices in the extruded map should be masked out for
        # the application of strong boundary conditions
        self._bottom_mask = np.zeros(len(offset)) if offset is not None else []
        self._top_mask = np.zeros(len(offset)) if offset is not None else []
        if offset is not None and bt_masks is not None:
            self._bottom_mask[bt_masks[0]] = -1
            self._top_mask[bt_masks[1]] = -1
        Map._globalcount += 1

    @validate_type(('index', (int, IterationIndex), IndexTypeError))
    def __getitem__(self, index):
        if isinstance(index, int) and not (0 <= index < self.arity):
            raise IndexValueError("Index must be in interval [0,%d]" % (self._arity - 1))
        if isinstance(index, IterationIndex) and index.index not in [0, 1]:
            raise IndexValueError("IterationIndex must be in interval [0,1]")
        return _make_object('Arg', map=self, idx=index)

    # This is necessary so that we can convert a Map to a tuple
    # (needed in as_tuple).  Because, __getitem__ no longer returns a
    # Map we have to explicitly provide an iterable interface
    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    def __getslice__(self, i, j):
        raise NotImplementedError("Slicing maps is not currently implemented")

    @property
    def _argtype(self):
        """Ctypes argtype for this :class:`Map`"""
        return np.ctypeslib.ctypes.c_voidp

    @property
    def split(self):
        return (self,)

    @property
    def iteration_region(self):
        """Return the iteration region for the current map. For a normal map it
        will always be ALL. For a :class:`DecoratedMap` it will specify over which mesh
        region the iteration will take place."""
        return frozenset([ALL])

    @property
    def implicit_bcs(self):
        """Return any implicit (extruded "top" or "bottom") bcs to
        apply to this :class:`Map`.  Normally empty except in the case of
        some :class:`DecoratedMap`\s."""
        return frozenset([])

    @property
    def iterset(self):
        """:class:`Set` mapped from."""
        return self._iterset

    @property
    def toset(self):
        """:class:`Set` mapped to."""
        return self._toset

    @property
    def arity(self):
        """Arity of the mapping: number of toset elements mapped to per
        iterset element."""
        return self._arity

    @property
    def arities(self):
        """Arity of the mapping: number of toset elements mapped to per
        iterset element.

        :rtype: tuple"""
        return (self._arity,)

    @property
    def arange(self):
        """Tuple of arity offsets for each constituent :class:`Map`."""
        return (0, self._arity)

    @property
    def values(self):
        """Mapping array.

        This only returns the map values for local points, to see the
        halo points too, use :meth:`values_with_halo`."""
        return self._values[:self.iterset.size]

    @property
    def values_with_halo(self):
        """Mapping array.

        This returns all map values (including halo points), see
        :meth:`values` if you only need to look at the local
        points."""
        return self._values

    @property
    def name(self):
        """User-defined label"""
        return self._name

    @property
    def offset(self):
        """The vertical offset."""
        return self._offset

    @property
    def top_mask(self):
        """The top layer mask to be applied on a mesh cell."""
        return self._top_mask

    @property
    def bottom_mask(self):
        """The bottom layer mask to be applied on a mesh cell."""
        return self._bottom_mask

    def __str__(self):
        return "OP2 Map: %s from (%s) to (%s) with arity %s" \
               % (self._name, self._iterset, self._toset, self._arity)

    def __repr__(self):
        return "Map(%r, %r, %r, None, %r)" \
               % (self._iterset, self._toset, self._arity, self._name)

    def __le__(self, o):
        """self<=o if o equals self or self._parent <= o."""
        if isinstance(o, DecoratedMap):
            # The iteration region of self must be a subset of the
            # iteration region of the sparsitymap.
            return len(self.iteration_region - o.iteration_region) == 0 and self <= o._map
        return self == o or (isinstance(self._parent, Map) and self._parent <= o)

    @classmethod
    def fromhdf5(cls, iterset, toset, f, name):
        """Construct a :class:`Map` from set named ``name`` in HDF5 data ``f``"""
        slot = f[name]
        values = slot.value
        arity = slot.shape[1:]
        if len(arity) != 1:
            raise ArityTypeError("Unrecognised arity value %s" % arity)
        return cls(iterset, toset, arity[0], values, name)


class DecoratedMap(Map, ObjectCached):
    """Augmented type for a map used for attaching extra information
    used to inform code generation and/or sparsity building about the
    implicit structure of the extruded :class:`Map`.

    :param map: The original class:`Map`.

    :kwarg iteration_region: The class:`IterationRegion` of the mesh over which
                             the parallel loop will iterate.
    :kwarg implicit_bcs: Any "top" or "bottom" boundary conditions to apply
                         when assembling :class:`Mat`\s.

    The :data:`map` parameter may be an existing :class:`DecoratedMap`
    in which case, if either the :data:`iteration_region` or
    :data:`implicit_bcs` arguments are :data:`None`, they will be
    copied over from the supplied :data:`map`."""

    def __new__(cls, map, iteration_region=None, implicit_bcs=None):
        if isinstance(map, DecoratedMap):
            # Need to add information, rather than replace if we
            # already have a decorated map (but overwrite if we're
            # told to)
            if iteration_region is None:
                iteration_region = [x for x in map.iteration_region]
            if implicit_bcs is None:
                implicit_bcs = [x for x in map.implicit_bcs]
            return DecoratedMap(map.map, iteration_region=iteration_region,
                                implicit_bcs=implicit_bcs)
        if isinstance(map, MixedMap):
            return MixedMap([DecoratedMap(m, iteration_region=iteration_region,
                                          implicit_bcs=implicit_bcs)
                             for m in map])
        return super(DecoratedMap, cls).__new__(cls, map, iteration_region=iteration_region,
                                                implicit_bcs=implicit_bcs)

    def __init__(self, map, iteration_region=None, implicit_bcs=None):
        if self._initialized:
            return
        self._map = map
        if iteration_region is None:
            iteration_region = [ALL]
        iteration_region = as_tuple(iteration_region, IterationRegion)
        self._iteration_region = frozenset(iteration_region)
        if implicit_bcs is None:
            implicit_bcs = []
        implicit_bcs = as_tuple(implicit_bcs)
        self._implicit_bcs = frozenset(implicit_bcs)
        self._initialized = True

    @classmethod
    def _process_args(cls, m, **kwargs):
        return (m, ) + (m, ), kwargs

    @classmethod
    def _cache_key(cls, map, iteration_region=None, implicit_bcs=None):
        ir = as_tuple(iteration_region, IterationRegion) if iteration_region else ()
        bcs = as_tuple(implicit_bcs) if implicit_bcs else ()
        return (map, ir, bcs)

    def __repr__(self):
        return "DecoratedMap(%r, %r, %r)" % (self._map, self._iteration_region, self.implicit_bcs)

    def __str__(self):
        return "OP2 DecoratedMap on %s with region %s, implicit bcs %s" % \
            (self._map, self._iteration_region, self.implicit_bcs)

    def __le__(self, other):
        """self<=other if the iteration regions of self are a subset of the
        iteration regions of other and self._map<=other"""
        if isinstance(other, DecoratedMap):
            return len(self.iteration_region - other.iteration_region) == 0 and self._map <= other._map
        else:
            return len(self.iteration_region - other.iteration_region) == 0 and self._map <= other

    def __getattr__(self, name):
        return getattr(self._map, name)

    @property
    def map(self):
        """The :class:`Map` this :class:`DecoratedMap` is decorating"""
        return self._map

    @property
    def iteration_region(self):
        """Returns the type of the iteration to be performed."""
        return self._iteration_region

    @property
    def implicit_bcs(self):
        """Return the set (if any) of implicit ("top" or "bottom") bcs
    to be applied to the :class:`Map`."""
        return self._implicit_bcs


class MixedMap(Map, ObjectCached):
    """A container for a bag of :class:`Map`\s."""

    def __init__(self, maps):
        """:param iterable maps: Iterable of :class:`Map`\s"""
        if self._initialized:
            return
        self._maps = maps
        # Make sure all itersets are identical
        if not all(m.iterset == self._maps[0].iterset for m in self._maps):
            raise MapTypeError("All maps in a MixedMap need to share the same iterset")
        self._initialized = True

    @classmethod
    def _process_args(cls, *args, **kwargs):
        maps = as_tuple(args[0], type=Map)
        cache = maps[0]
        return (cache, ) + (maps, ), kwargs

    @classmethod
    def _cache_key(cls, maps):
        return maps

    @property
    def split(self):
        """The underlying tuple of :class:`Map`\s."""
        return self._maps

    @property
    def iterset(self):
        """:class:`MixedSet` mapped from."""
        return self._maps[0].iterset

    @property
    def toset(self):
        """:class:`MixedSet` mapped to."""
        return MixedSet(tuple(m.toset for m in self._maps))

    @property
    def arity(self):
        """Arity of the mapping: total number of toset elements mapped to per
        iterset element."""
        return sum(m.arity for m in self._maps)

    @property
    def arities(self):
        """Arity of the mapping: number of toset elements mapped to per
        iterset element.

        :rtype: tuple"""
        return tuple(m.arity for m in self._maps)

    @property
    def arange(self):
        """Tuple of arity offsets for each constituent :class:`Map`."""
        return (0,) + tuple(np.cumsum(self.arities))

    @property
    def values(self):
        """Mapping arrays excluding data for halos.

        This only returns the map values for local points, to see the
        halo points too, use :meth:`values_with_halo`."""
        return tuple(m.values for m in self._maps)

    @property
    def values_with_halo(self):
        """Mapping arrays including data for halos.

        This returns all map values (including halo points), see
        :meth:`values` if you only need to look at the local
        points."""
        return tuple(m.values_with_halo for m in self._maps)

    @property
    def name(self):
        """User-defined labels"""
        return tuple(m.name for m in self._maps)

    @property
    def offset(self):
        """Vertical offsets."""
        return tuple(m.offset for m in self._maps)

    def __iter__(self):
        """Yield all :class:`Map`\s when iterated over."""
        for m in self._maps:
            yield m

    def __len__(self):
        """Number of contained :class:`Map`\s."""
        return len(self._maps)

    def __le__(self, o):
        """self<=o if o equals self or its self._parent==o."""
        return self == o or all(m <= om for m, om in zip(self, o))

    def __str__(self):
        return "OP2 MixedMap composed of Maps: %s" % (self._maps,)

    def __repr__(self):
        return "MixedMap(%r)" % (self._maps,)


class Sparsity(ObjectCached):

    """OP2 Sparsity, the non-zero structure a matrix derived from the union of
    the outer product of pairs of :class:`Map` objects.

    Examples of constructing a Sparsity: ::

        Sparsity(single_dset, single_map, 'mass')
        Sparsity((row_dset, col_dset), (single_rowmap, single_colmap))
        Sparsity((row_dset, col_dset),
                 [(first_rowmap, first_colmap), (second_rowmap, second_colmap)])

    .. _MatMPIAIJSetPreallocation: http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatMPIAIJSetPreallocation.html
    """

    def __init__(self, dsets, maps, name=None):
        """
        :param dsets: :class:`DataSet`\s for the left and right function
            spaces this :class:`Sparsity` maps between
        :param maps: :class:`Map`\s to build the :class:`Sparsity` from
        :type maps: a pair of :class:`Map`\s specifying a row map and a column
            map, or an iterable of pairs of :class:`Map`\s specifying multiple
            row and column maps - if a single :class:`Map` is passed, it is
            used as both a row map and a column map
        :param string name: user-defined label (optional)
        """
        # Protect against re-initialization when retrieved from cache
        if self._initialized:
            return

        # Split into a list of row maps and a list of column maps
        self._rmaps, self._cmaps = zip(*maps)
        self._dsets = dsets

        # All rmaps and cmaps have the same data set - just use the first.
        self._nrows = self._rmaps[0].toset.size
        self._ncols = self._cmaps[0].toset.size
        self._dims = (self._dsets[0].cdim, self._dsets[1].cdim)

        self._name = name or "sparsity_%d" % Sparsity._globalcount
        Sparsity._globalcount += 1

        # If the Sparsity is defined on MixedDataSets, we need to build each
        # block separately
        if isinstance(dsets[0], MixedDataSet) or isinstance(dsets[1], MixedDataSet):
            self._blocks = []
            for i, rds in enumerate(dsets[0]):
                row = []
                for j, cds in enumerate(dsets[1]):
                    row.append(Sparsity((rds, cds), [(rm.split[i], cm.split[j]) for rm, cm in maps]))
                self._blocks.append(row)
            self._rowptr = tuple(s._rowptr for s in self)
            self._colidx = tuple(s._colidx for s in self)
            self._d_nnz = tuple(s._d_nnz for s in self)
            self._o_nnz = tuple(s._o_nnz for s in self)
            self._d_nz = sum(s._d_nz for s in self)
            self._o_nz = sum(s._o_nz for s in self)
        else:
            with timed_region("Build sparsity"):
                build_sparsity(self, parallel=MPI.parallel)
            self._blocks = [[self]]
        self._initialized = True

    _cache = {}
    _globalcount = 0

    @classmethod
    @validate_type(('dsets', (Set, DataSet, tuple, list), DataSetTypeError),
                   ('maps', (Map, tuple, list), MapTypeError),
                   ('name', str, NameTypeError))
    def _process_args(cls, dsets, maps, name=None, *args, **kwargs):
        "Turn maps argument into a canonical tuple of pairs."

        # A single data set becomes a pair of identical data sets
        dsets = [dsets, dsets] if isinstance(dsets, (Set, DataSet)) else list(dsets)
        # Upcast Sets to DataSets
        dsets = [s ** 1 if isinstance(s, Set) else s for s in dsets]

        # Check data sets are valid
        for dset in dsets:
            if not isinstance(dset, DataSet):
                raise DataSetTypeError("All data sets must be of type DataSet, not type %r" % type(dset))

        # A single map becomes a pair of identical maps
        maps = (maps, maps) if isinstance(maps, Map) else maps
        # A single pair becomes a tuple of one pair
        maps = (maps,) if isinstance(maps[0], Map) else maps

        # Check maps are sane
        for pair in maps:
            for m in pair:
                if not isinstance(m, Map):
                    raise MapTypeError(
                        "All maps must be of type map, not type %r" % type(m))
                if len(m.values_with_halo) == 0 and m.iterset.total_size > 0:
                    raise MapValueError(
                        "Unpopulated map values when trying to build sparsity.")
            # Make sure that the "to" Set of each map in a pair is the set of
            # the corresponding DataSet set
            if not (pair[0].toset == dsets[0].set and
                    pair[1].toset == dsets[1].set):
                raise RuntimeError("Map to set must be the same as corresponding DataSet set")

            # Each pair of maps must have the same from-set (iteration set)
            if not pair[0].iterset == pair[1].iterset:
                raise RuntimeError("Iterset of both maps in a pair must be the same")

        rmaps, cmaps = zip(*maps)

        if not len(rmaps) == len(cmaps):
            raise RuntimeError("Must pass equal number of row and column maps")

        # Each row map must have the same to-set (data set)
        if not all(m.toset == rmaps[0].toset for m in rmaps):
            raise RuntimeError("To set of all row maps must be the same")

        # Each column map must have the same to-set (data set)
        if not all(m.toset == cmaps[0].toset for m in cmaps):
            raise RuntimeError("To set of all column maps must be the same")

        # Need to return the caching object, a tuple of the processed
        # arguments and a dict of kwargs (empty in this case)
        if isinstance(dsets[0].set, MixedSet):
            cache = dsets[0].set[0]
        else:
            cache = dsets[0].set
        return (cache, ) + (tuple(dsets), tuple(sorted(uniquify(maps))), name), {}

    @classmethod
    def _cache_key(cls, dsets, maps, *args, **kwargs):
        return (dsets, maps)

    def __getitem__(self, idx):
        """Return :class:`Sparsity` block with row and column given by ``idx``
        or a given row of blocks."""
        try:
            i, j = idx
            return self._blocks[i][j]
        except TypeError:
            return self._blocks[idx]

    @property
    def dsets(self):
        """A pair of :class:`DataSet`\s for the left and right function
        spaces this :class:`Sparsity` maps between."""
        return self._dsets

    @property
    def maps(self):
        """A list of pairs (rmap, cmap) where each pair of
        :class:`Map` objects will later be used to assemble into this
        matrix. The iterset of each of the maps in a pair must be the
        same, while the toset of all the maps which appear first
        must be common, this will form the row :class:`Set` of the
        sparsity. Similarly, the toset of all the maps which appear
        second must be common and will form the column :class:`Set` of
        the ``Sparsity``."""
        return zip(self._rmaps, self._cmaps)

    @property
    def cmaps(self):
        """The list of column maps this sparsity is assembled from."""
        return self._cmaps

    @property
    def rmaps(self):
        """The list of row maps this sparsity is assembled from."""
        return self._rmaps

    @property
    def dims(self):
        """A pair giving the number of rows per entry of the row
        :class:`Set` and the number of columns per entry of the column
        :class:`Set` of the ``Sparsity``."""
        return self._dims

    @property
    def shape(self):
        """Number of block rows and columns."""
        return len(self._dsets[0]), len(self._dsets[1])

    @property
    def nrows(self):
        """The number of rows in the ``Sparsity``."""
        return self._nrows

    @property
    def ncols(self):
        """The number of columns in the ``Sparsity``."""
        return self._ncols

    @property
    def name(self):
        """A user-defined label."""
        return self._name

    def __iter__(self):
        """Iterate over all :class:`Sparsity`\s by row and then by column."""
        for row in self._blocks:
            for s in row:
                yield s

    def __str__(self):
        return "OP2 Sparsity: dsets %s, rmaps %s, cmaps %s, name %s" % \
               (self._dsets, self._rmaps, self._cmaps, self._name)

    def __repr__(self):
        return "Sparsity(%r, %r, %r)" % (self.dsets, self.maps, self.name)

    @property
    def rowptr(self):
        """Row pointer array of CSR data structure."""
        return self._rowptr

    @property
    def colidx(self):
        """Column indices array of CSR data structure."""
        return self._colidx

    @property
    def nnz(self):
        """Array containing the number of non-zeroes in the various rows of the
        diagonal portion of the local submatrix.

        This is the same as the parameter `d_nnz` used for preallocation in
        PETSc's MatMPIAIJSetPreallocation_."""
        return self._d_nnz

    @property
    def onnz(self):
        """Array containing the number of non-zeroes in the various rows of the
        off-diagonal portion of the local submatrix.

        This is the same as the parameter `o_nnz` used for preallocation in
        PETSc's MatMPIAIJSetPreallocation_."""
        return self._o_nnz

    @property
    def nz(self):
        """Number of non-zeroes in the diagonal portion of the local
        submatrix."""
        return int(self._d_nz)

    @property
    def onz(self):
        """Number of non-zeroes in the off-diagonal portion of the local
        submatrix."""
        return int(self._o_nz)

    def __contains__(self, other):
        """Return true if other is a pair of maps in self.maps(). This
        will also return true if the elements of other have parents in
        self.maps()."""

        for maps in self.maps:
            if tuple(other) <= maps:
                return True

        return False


class Mat(SetAssociated):
    """OP2 matrix data. A ``Mat`` is defined on a sparsity pattern and holds a value
    for each element in the :class:`Sparsity`.

    When a ``Mat`` is passed to :func:`pyop2.op2.par_loop`, the maps via which
    indirection occurs for the row and column space, and the access
    descriptor are passed by `calling` the ``Mat``. For instance, if a
    ``Mat`` named ``A`` is to be accessed for reading via a row :class:`Map`
    named ``R`` and a column :class:`Map` named ``C``, this is accomplished by::

     A(pyop2.READ, (R[pyop2.i[0]], C[pyop2.i[1]]))

    Notice that it is `always` necessary to index the indirection maps
    for a ``Mat``. See the :class:`Mat` documentation for more
    details.

    .. note ::

       After executing :func:`par_loop`\s that write to a ``Mat`` and
       before using it (for example to view its values), you must call
       :meth:`assemble` to finalise the writes.
    """

    _globalcount = 0
    _modes = [WRITE, INC]

    @validate_type(('sparsity', Sparsity, SparsityTypeError),
                   ('name', str, NameTypeError))
    def __init__(self, sparsity, dtype=None, name=None):
        self._sparsity = sparsity
        self._datatype = np.dtype(dtype)
        self._name = name or "mat_%d" % Mat._globalcount
        Mat._globalcount += 1

    @validate_in(('access', _modes, ModeValueError))
    def __call__(self, access, path, flatten=False):
        path = as_tuple(path, Arg, 2)
        path_maps = [arg.map for arg in path]
        path_idxs = [arg.idx for arg in path]
        if tuple(path_maps) not in self.sparsity:
            raise MapValueError("Path maps not in sparsity maps")
        return _make_object('Arg', data=self, map=path_maps, access=access,
                            idx=path_idxs, flatten=flatten)

    def assemble(self):
        """Finalise this :class:`Mat` ready for use.

        Call this /after/ executing all the par_loops that write to
        the matrix before you want to look at it.
        """
        self._assemble()

    def _assemble(self):
        raise NotImplementedError("Abstract Mat base class doesn't know how to assemble itself")

    @property
    def _argtype(self):
        """Ctypes argtype for this :class:`Mat`"""
        return np.ctypeslib.ctypes.c_voidp

    @property
    def dims(self):
        """A pair of integers giving the number of matrix rows and columns for
        each member of the row :class:`Set` and column :class:`Set`
        respectively. This corresponds to the ``cdim`` member of a
        :class:`DataSet`."""
        return self._sparsity._dims

    @property
    def sparsity(self):
        """:class:`Sparsity` on which the ``Mat`` is defined."""
        return self._sparsity

    @property
    def _is_scalar_field(self):
        # Sparsity from Dat to MixedDat has a shape like (1, (1, 1))
        # (which you can't take the product of)
        return all(np.prod(d) == 1 for d in self.dims)

    @property
    def _is_vector_field(self):
        return not self._is_scalar_field

    @property
    def values(self):
        """A numpy array of matrix values.

        .. warning ::
            This is a dense array, so will need a lot of memory.  It's
            probably not a good idea to access this property if your
            matrix has more than around 10000 degrees of freedom.
        """
        raise NotImplementedError("Abstract base Mat does not implement values()")

    @property
    def dtype(self):
        """The Python type of the data."""
        return self._datatype

    @property
    def nbytes(self):
        """Return an estimate of the size of the data associated with this
        :class:`Mat` in bytes. This will be the correct size of the
        data payload, but does not take into account the (presumably
        small) overhead of the object and its metadata. The memory
        associated with the sparsity pattern is also not recorded.

        Note that this is the process local memory usage, not the sum
        over all MPI processes.
        """

        return (self._sparsity.nz + self._sparsity.onz) \
            * self.dtype.itemsize * np.sum(np.prod(self._sparsity.dims))

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __mul__(self, other):
        """Multiply this :class:`Mat` with the vector ``other``."""
        raise NotImplementedError("Abstract base Mat does not implement multiplication")

    def __str__(self):
        return "OP2 Mat: %s, sparsity (%s), datatype %s" \
               % (self._name, self._sparsity, self._datatype.name)

    def __repr__(self):
        return "Mat(%r, %r, %r)" \
               % (self._sparsity, self._datatype, self._name)

# Kernel API


class Kernel(Cached):

    """OP2 kernel type.

    :param code: kernel function definition, including signature; either a
        string or an AST :class:`.Node`
    :param name: kernel function name; must match the name of the kernel
        function given in `code`
    :param opts: options dictionary for :doc:`PyOP2 IR optimisations <ir>`
        (optional, ignored if `code` is a string)
    :param include_dirs: list of additional include directories to be searched
        when compiling the kernel (optional, defaults to empty)
    :param headers: list of system headers to include when compiling the kernel
        in the form ``#include <header.h>`` (optional, defaults to empty)
    :param user_code: code snippet to be executed once at the very start of
        the generated kernel wrapper code (optional, defaults to empty)

    Consider the case of initialising a :class:`~pyop2.Dat` with seeded random
    values in the interval 0 to 1. The corresponding :class:`~pyop2.Kernel` is
    constructed as follows: ::

      op2.Kernel("void setrand(double *x) { x[0] = (double)random()/RAND_MAX); }",
                 name="setrand",
                 headers=["#include <stdlib.h>"], user_code="srandom(10001);")

    .. note::
        When running in parallel with MPI the generated code must be the same
        on all ranks.
    """

    _globalcount = 0
    _cache = {}

    @classmethod
    @validate_type(('name', str, NameTypeError))
    def _cache_key(cls, code, name, opts={}, include_dirs=[], headers=[],
                   user_code=""):
        # Both code and name are relevant since there might be multiple kernels
        # extracting different functions from the same code
        # Also include the PyOP2 version, since the Kernel class might change

        # HACK: Temporary fix!
        if isinstance(code, Node):
            code = code.gencode()
        return md5(str(hash(code)) + name + str(opts) + str(include_dirs) +
                   str(headers) + version).hexdigest()

    def _ast_to_c(self, ast, opts={}):
        """Transform an Abstract Syntax Tree representing the kernel into a
        string of C code."""
        if isinstance(ast, Node):
            self._ast = ast
            return ast.gencode()
        return ast

    def __init__(self, code, name, opts={}, include_dirs=[], headers=[],
                 user_code=""):
        # Protect against re-initialization when retrieved from cache
        if self._initialized:
            return
        self._name = name or "kernel_%d" % Kernel._globalcount
        Kernel._globalcount += 1
        # Record used optimisations
        self._opts = opts
        self._applied_blas = False
        self._applied_ap = False
        self._include_dirs = include_dirs
        self._headers = headers
        self._user_code = user_code
        self._code = self._ast_to_c(code, opts)
        self._initialized = True

    @property
    def name(self):
        """Kernel name, must match the kernel function name in the code."""
        return self._name

    @property
    def code(self):
        """String containing the c code for this kernel routine. This
        code must conform to the OP2 user kernel API."""
        return self._code

    def __str__(self):
        return "OP2 Kernel: %s" % self._name

    def __repr__(self):
        return 'Kernel("""%s""", %r)' % (self._code, self._name)


class JITModule(Cached):

    """Cached module encapsulating the generated :class:`ParLoop` stub.

    .. warning::

       Note to implementors.  This object is *cached* and therefore
       should not hold any references to objects you might want to be
       collected (such PyOP2 data objects)."""

    _cache = {}

    @classmethod
    def _cache_key(cls, kernel, itspace, *args, **kwargs):
        key = (kernel.cache_key, itspace.cache_key)
        for arg in args:
            if arg._is_global:
                key += (arg.data.dim, arg.data.dtype, arg.access)
            elif arg._is_dat:
                if isinstance(arg.idx, IterationIndex):
                    idx = (arg.idx.__class__, arg.idx.index)
                else:
                    idx = arg.idx
                map_arity = arg.map.arity if arg.map else None
                key += (arg.data.dim, arg.data.dtype, map_arity, idx, arg.access)
            elif arg._is_mat:
                idxs = (arg.idx[0].__class__, arg.idx[0].index,
                        arg.idx[1].index)
                map_arities = (arg.map[0].arity, arg.map[1].arity)
                # Implicit boundary conditions (extruded "top" or
                # "bottom") affect generated code, and therefore need
                # to be part of cache key
                map_bcs = (arg.map[0].implicit_bcs, arg.map[1].implicit_bcs)
                key += (arg.data.dims, arg.data.dtype, idxs,
                        map_arities, map_bcs, arg.access)

        iterate = kwargs.get("iterate", None)
        if iterate is not None:
            key += ((iterate,))

        # The currently defined Consts need to be part of the cache key, since
        # these need to be uploaded to the device before launching the kernel
        for c in Const._definitions():
            key += (c.name, c.dtype, c.cdim)

        return key

    def _dump_generated_code(self, src, ext=None):
        """Write the generated code to a file for debugging purposes.

        :arg src: The source string to write
        :arg ext: The file extension of the output file (if not `None`)

        Output will only be written if the `dump_gencode`
        configuration parameter is `True`.  The output file will be
        written to the directory specified by the PyOP2 configuration
        parameter `dump_gencode_path`.  See :class:`Configuration` for
        more details.

        """
        if configuration['dump_gencode']:
            import os
            import hashlib
            fname = "%s-%s.%s" % (self._kernel.name,
                                  hashlib.md5(src).hexdigest(),
                                  ext if ext is not None else "c")
            if not os.path.exists(configuration['dump_gencode_path']):
                os.makedirs(configuration['dump_gencode_path'])
            output = os.path.abspath(os.path.join(configuration['dump_gencode_path'],
                                                  fname))
            with open(output, "w") as f:
                f.write(src)


class IterationRegion(object):
    """ Class that specifies the way to iterate over a column of extruded
    mesh elements. A column of elements refers to the elements which are
    in the extrusion direction. The accesses to these elements are direct.
    """

    _iterates = ["ON_BOTTOM", "ON_TOP", "ON_INTERIOR_FACETS", "ALL"]

    @validate_in(('iterate', _iterates, IterateValueError))
    def __init__(self, iterate):
        self._iterate = iterate

    @property
    def where(self):
        return self._iterate

    def __str__(self):
        return "OP2 Iterate: %s" % self._iterate

    def __repr__(self):
        return "%r" % self._iterate

ON_BOTTOM = IterationRegion("ON_BOTTOM")
"""Iterate over the cells at the bottom of the column in an extruded mesh."""

ON_TOP = IterationRegion("ON_TOP")
"""Iterate over the top cells in an extruded mesh."""

ON_INTERIOR_FACETS = IterationRegion("ON_INTERIOR_FACETS")
"""Iterate over the interior facets of an extruded mesh."""

ALL = IterationRegion("ALL")
"""Iterate over all cells of an extruded mesh."""


class ParLoop(object):
    """Represents the kernel, iteration space and arguments of a parallel loop
    invocation.

    .. note ::

        Users should not directly construct :class:`ParLoop` objects, but
        use :func:`pyop2.op2.par_loop` instead.

    An optional keyword argument, ``iterate``, can be used to specify
    which region of an :class:`ExtrudedSet` the parallel loop should
    iterate over.
    """

    @validate_type(('kernel', Kernel, KernelTypeError),
                   ('iterset', Set, SetTypeError))
    def __init__(self, kernel, iterset, *args, **kwargs):
        # INCs into globals need to start with zero and then sum back
        # into the input global at the end.  This has the same number
        # of reductions but means that successive par_loops
        # incrementing into a global get the "right" value in
        # parallel.
        # Don't care about MIN and MAX because they commute with the reduction
        self._reduced_globals = {}
        self._reduction_args = []
        for i, arg in enumerate(args):
            if arg._is_global_reduction:
                if arg.access == INC:
                    glob = arg.data
                    self._reduced_globals[i] = glob
                    args[i]._dat = _make_object('Global', glob.dim, data=np.zeros_like(glob.data_ro), dtype=glob.dtype)
                self._reduction_args.append(arg)

        # Always use the current arguments, also when we hit cache
        self._actual_args = args
        self._kernel = kernel
        self._is_layered = iterset._extruded
        self._halo_exchange_args = []
        if not self.is_direct:
            for arg in self.args:
                if arg._is_dat:
                    self._halo_exchange_args.append(arg)
        self._iteration_region = kwargs.get("iterate", None)
        self._args_to_mark_for_halo_exchange = []
        for arg in self.args:
            if arg._is_dat and arg.access in [INC, WRITE, RW]:
                self._args_to_mark_for_halo_exchange.append(arg)
        self._args_to_mark_read_only = []
        for arg in self.args:
            if arg._is_dat:
                self._args_to_mark_read_only.append(arg)

        for i, arg in enumerate(self._actual_args):
            arg.position = i
            arg.indirect_position = i
        for i, arg1 in enumerate(self._actual_args):
            if arg1._is_dat and arg1._is_indirect:
                for arg2 in self._actual_args[i:]:
                    # We have to check for identity here (we really
                    # want these to be the same thing, not just look
                    # the same)
                    if arg2.data is arg1.data and arg2.map is arg1.map:
                        arg2.indirect_position = arg1.indirect_position

        self._it_space = self.build_itspace(iterset)
        self.needs_exec_halo = any(arg._is_indirect_and_not_read or arg._is_mat
                                   for arg in self.args)

    def replace_arg_data(self, data, idx):
        """Replace the data payload of an argument in this :class:`ParLoop`.

        :arg data: the :class:`DataCarrier` to use as a replacement.
        :arg idx: the index of the argument in the :class:`ParLoop`.

        .. warning::

           No error checking is performed in this routine.  It is up
           to the caller to ensure that they do not accidentally
           replace the payload of (say) a :class:`Global` :class:`Arg`
           with a :class:`Mat`.

           Do not use this function unless you know what you're doing!
        """
        self._actual_args[idx]._dat = data

    def _run(self):
        for a in self.args:
            if a.access != READ:
                a.data._version_bump()
        return self.compute()

    @collective
    @timed_function('ParLoop compute')
    @profile
    def compute(self):
        """Executes the kernel over all members of the iteration space."""
        iterset = self.it_space.iterset
        self.halo_exchange_begin()
        self._compute(iterset.core_part)
        self.halo_exchange_end()
        self._compute(iterset.owned_part)
        self.reduction_begin()
        if self.needs_exec_halo:
            self._compute(iterset.exec_part)
        self.reduction_end()
        self.maybe_set_halo_update_needed()
        self.mark_dats_read_only()

    @collective
    def _compute(self, start, end):
        """Executes the kernel over all members of a MPI-part of the iteration space."""
        raise RuntimeError("Must select a backend")

    def mark_dats_read_only(self):
        for arg in self._args_to_mark_read_only:
            if arg.data._is_allocated:
                for d in arg.data:
                    maybe_setflags(d._data, write=False)

    @collective
    @timed_function('ParLoop halo exchange begin')
    def halo_exchange_begin(self):
        """Start halo exchanges."""
        for arg in self._halo_exchange_args:
            arg.halo_exchange_begin()

    @collective
    @timed_function('ParLoop halo exchange end')
    def halo_exchange_end(self):
        """Finish halo exchanges (wait on irecvs)"""
        for arg in self._halo_exchange_args:
            arg.halo_exchange_end()

    @collective
    @timed_function('ParLoop reduction begin')
    def reduction_begin(self):
        """Start reductions"""
        for arg in self._reduction_args:
            arg.reduction_begin()

    @collective
    @timed_function('ParLoop reduction end')
    def reduction_end(self):
        """End reductions"""
        for arg in self._reduction_args:
            arg.reduction_end()
        # Finalise global increments
        for i, glob in self._reduced_globals.iteritems():
            # These can safely access the _data member directly
            # because the reduction_end on the temporary global pulled
            # data back from the device if necessary.
            glob._data += self.args[i].data._data

    @collective
    def maybe_set_halo_update_needed(self):
        """Set halo update needed for :class:`Dat` arguments that are written to
        in this parallel loop."""
        for arg in self._args_to_mark_for_halo_exchange:
            arg.data.needs_halo_update = True

    def build_itspace(self, iterset):
        """Checks that the iteration set of the :class:`ParLoop` matches the
        iteration set of all its arguments. A :class:`MapValueError` is raised
        if this condition is not met.

        Also determines the size of the local iteration space and checks all
        arguments using an :class:`IterationIndex` for consistency.

        :return: class:`IterationSpace` for this :class:`ParLoop`"""

        _iterset = iterset.superset if isinstance(iterset, Subset) else iterset
        if isinstance(_iterset, MixedSet):
            raise SetTypeError("Cannot iterate over MixedSets")
        block_shape = None
        for i, arg in enumerate(self._actual_args):
            if arg._is_global:
                continue
            if arg._is_direct:
                if arg.data.dataset.set != _iterset:
                    raise MapValueError(
                        "Iterset of direct arg %s doesn't match ParLoop iterset." % i)
                continue
            for j, m in enumerate(arg._map):
                if isinstance(_iterset, ExtrudedSet):
                    if m.iterset != _iterset and m.iterset not in _iterset:
                        raise MapValueError(
                            "Iterset of arg %s map %s doesn't match ParLoop iterset." % (i, j))
                elif m.iterset != _iterset:
                    raise MapValueError(
                        "Iterset of arg %s map %s doesn't match ParLoop iterset." % (i, j))
            if arg._uses_itspace:
                _block_shape = arg._block_shape
                if block_shape and block_shape != _block_shape:
                    raise IndexValueError("Mismatching iteration space size for argument %d" % i)
                block_shape = _block_shape
        return IterationSpace(iterset, block_shape)

    @property
    def offset_args(self):
        """The offset args that need to be added to the argument list."""
        _args = []
        for arg in self.args:
            if arg._is_indirect or arg._is_mat:
                maps = as_tuple(arg.map, Map)
                for map in maps:
                    for m in map:
                        if m.iterset._extruded:
                            _args.append(m.offset)
        return _args

    @property
    def layer_arg(self):
        """The layer arg that needs to be added to the argument list."""
        if self._is_layered:
            return [self._it_space.layers]
        return []

    @property
    def it_space(self):
        """Iteration space of the parallel loop."""
        return self._it_space

    @property
    def is_direct(self):
        """Is this parallel loop direct? I.e. are all the arguments either
        :class:Dats accessed through the identity map, or :class:Global?"""
        return all(a.map is None for a in self.args)

    @property
    def is_indirect(self):
        """Is the parallel loop indirect?"""
        return not self.is_direct

    @property
    def kernel(self):
        """Kernel executed by this parallel loop."""
        return self._kernel

    @property
    def args(self):
        """Arguments to this parallel loop."""
        return self._actual_args

    @property
    def _has_soa(self):
        return any(a._is_soa for a in self._actual_args)

    @property
    def is_layered(self):
        """Flag which triggers extrusion"""
        return self._is_layered

    @property
    def iteration_region(self):
        """Specifies the part of the mesh the parallel loop will
        be iterating over. The effect is the loop only iterates over
        a certain part of an extruded mesh, for example on top cells, bottom cells or
        interior facets."""
        return self._iteration_region

DEFAULT_SOLVER_PARAMETERS = {'ksp_type': 'cg',
                             'pc_type': 'jacobi',
                             'ksp_rtol': 1.0e-7,
                             'ksp_atol': 1.0e-50,
                             'ksp_divtol': 1.0e+4,
                             'ksp_max_it': 10000,
                             'ksp_monitor': False,
                             'plot_convergence': False,
                             'plot_prefix': '',
                             'error_on_nonconvergence': True,
                             'ksp_gmres_restart': 30}

"""All parameters accepted by PETSc KSP and PC objects are permissible
as options to the :class:`op2.Solver`."""


class Solver(object):

    """OP2 Solver object. The :class:`Solver` holds a set of parameters that are
    passed to the underlying linear algebra library when the ``solve`` method
    is called. These can either be passed as a dictionary ``parameters`` *or*
    as individual keyword arguments (combining both will cause an exception).

    Recognized parameters either as dictionary keys or keyword arguments are:

    :arg ksp_type: the solver type ('cg')
    :arg pc_type: the preconditioner type ('jacobi')
    :arg ksp_rtol: relative solver tolerance (1e-7)
    :arg ksp_atol: absolute solver tolerance (1e-50)
    :arg ksp_divtol: factor by which the residual norm may exceed the
        right-hand-side norm before the solve is considered to have diverged:
        ``norm(r) >= dtol*norm(b)`` (1e4)
    :arg ksp_max_it: maximum number of solver iterations (10000)
    :arg error_on_nonconvergence: abort if the solve does not converge in the
      maximum number of iterations (True, if False only a warning is printed)
    :arg ksp_monitor: print the residual norm after each iteration
        (False)
    :arg plot_convergence: plot a graph of the convergence history after the
        solve has finished and save it to file (False, implies *ksp_monitor*)
    :arg plot_prefix: filename prefix for plot files ('')
    :arg ksp_gmres_restart: restart period when using GMRES

    """

    def __init__(self, parameters=None, **kwargs):
        self.parameters = DEFAULT_SOLVER_PARAMETERS.copy()
        if parameters and kwargs:
            raise RuntimeError("Solver options are set either by parameters or kwargs")
        if parameters:
            self.parameters.update(parameters)
        else:
            self.parameters.update(kwargs)

    @collective
    def update_parameters(self, parameters):
        """Update solver parameters

        :arg parameters: Dictionary containing the parameters to update.
        """
        self.parameters.update(parameters)

    @modifies_argn(1)
    @collective
    def solve(self, A, x, b):
        """Solve a matrix equation.

        :arg A: The :class:`Mat` containing the matrix.
        :arg x: The :class:`Dat` to receive the solution.
        :arg b: The :class:`Dat` containing the RHS.
        """
        # Finalise assembly of the matrix, we know we need to this
        # because we're about to look at it.
        A.assemble()
        self._solve(A, x, b)

    def _solve(self, A, x, b):
        raise NotImplementedError("solve must be implemented by backend")


@collective
def par_loop(kernel, it_space, *args, **kwargs):
    if isinstance(kernel, types.FunctionType):
        import pyparloop
        return pyparloop.ParLoop(pyparloop.Kernel(kernel), it_space, *args, **kwargs)._run()
    return _make_object('ParLoop', kernel, it_space, *args, **kwargs)._run()
