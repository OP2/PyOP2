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

import numpy as np
import operator
from hashlib import md5

import configuration as cfg
from caching import Cached
from exceptions import *
from utils import *
from backends import _make_object
from mpi import MPI, _MPI, _check_comm, collective
from sparsity import build_sparsity


class LazyComputation(object):

    """Helper class holding computation to be carried later on.
    """

    def __init__(self, reads, writes):
        self.reads = reads
        self.writes = writes
        self._scheduled = False

    def enqueue(self):
        global _trace
        _trace.append(self)
        return self

    def _run(self):
        assert False, "Not implemented"


class ExecutionTrace(object):

    """Container maintaining delayed computation until they are executed."""

    def __init__(self):
        self._trace = list()

    def append(self, computation):
        if not cfg['lazy_evaluation']:
            assert not self._trace
            computation._run()
        elif cfg['lazy_max_trace_length'] > 0 and cfg['lazy_max_trace_length'] == len(self._trace):
            self.evaluate(computation.reads, computation.writes)
            computation._run()
        else:
            self._trace.append(computation)

    def in_queue(self, computation):
        return computation in self._trace

    def clear(self):
        """Forcefully drops delayed computation. Only use this if you know what you
        are doing.
        """
        self._trace = list()

    def evaluate_all(self):
        """Forces the evaluation of all delayed computations."""
        for comp in self._trace:
            comp._run()
        self._trace = list()

    def evaluate(self, reads, writes):
        """Forces the evaluation of delayed computation on which reads and writes
        depend.
        """
        def _depends_on(reads, writes, cont):
            return reads & cont.writes or writes & cont.reads or writes & cont.writes

        for comp in reversed(self._trace):
            if _depends_on(reads, writes, comp):
                comp._scheduled = True
                reads = reads | comp.reads - comp.writes
                writes = writes | comp.writes
            else:
                comp._scheduled = False

        new_trace = list()
        for comp in self._trace:
            if comp._scheduled:
                comp._run()
            else:
                new_trace.append(comp)
        self._trace = new_trace


_trace = ExecutionTrace()

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
        if self._is_global or map is None:
            return
        for j, m in enumerate(map):
            if m.iterset.total_size > 0 and len(m.values) == 0:
                raise MapValueError("%s is not initialized." % map)
            if self._is_mat and m.toset != data.sparsity.dsets[j].set:
                raise MapValueError(
                    "To set of %s doesn't match the set of %s." % (map, data))
            if self._is_dat and m._toset != data.dataset.set:
                raise MapValueError(
                    "To set of %s doesn't match the set of %s." % (map, data))

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
    def _is_global(self):
        return isinstance(self._dat, Global)

    @property
    def _is_global_reduction(self):
        return self._is_global and self._access in [INC, MIN, MAX]

    @property
    def _is_dat(self):
        return isinstance(self._dat, Dat)

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
        assert self._is_dat, "Doing halo exchanges only makes sense for Dats"
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
        assert self._is_dat, "Doing halo exchanges only makes sense for Dats"
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
    def __init__(self, size=None, name=None, halo=None, layers=None):
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
        self._layers = layers if layers is not None else 1
        self._partition_size = 1024
        if self.halo:
            self.halo.verify(self)
        Set._globalcount += 1

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
    def layers(self):
        """Number of layers in the extruded mesh"""
        return self._layers

    @property
    def partition_size(self):
        """Default partition size"""
        return self._partition_size

    @partition_size.setter
    def partition_size(self, partition_value):
        """Set the partition size"""
        self._partition_size = partition_value

    def __str__(self):
        return "OP2 Set: %s with size %s" % (self._name, self._size)

    def __repr__(self):
        return "Set(%r, %r)" % (self._size, self._name)

    def __contains__(self, dset):
        """Indicate whether a given DataSet is compatible with this Set."""
        return dset.set is self

    def __pow__(self, e):
        """Derive a :class:`DataSet` with dimension ``e``"""
        return DataSet(self, dim=e)

    @classmethod
    def fromhdf5(cls, f, name):
        """Construct a :class:`Set` from set named ``name`` in HDF5 data ``f``"""
        slot = f[name]
        if slot.shape != (1,):
            raise SizeTypeError("Shape of %s is incorrect" % name)
        size = slot.value.astype(np.int)
        return cls(size[0], name)

    @property
    def core_part(self):
        return SetPartition(self, 0, self.core_size)

    @property
    def owned_part(self):
        return SetPartition(self, self.core_size, self.size - self.core_size)

    @property
    def exec_part(self):
        return SetPartition(self, self.size, self.exec_size - self.size)

    @property
    def all_part(self):
        return SetPartition(self, 0, self.exec_size)


class SetPartition(object):
    def __init__(self, set, offset, size):
        self.set = set
        self.offset = offset
        self.size = size


class DataSet(object):
    """PyOP2 Data Set

    Set used in the op2.Dat structures to specify the dimension of the data.
    """
    _globalcount = 0

    @validate_type(('iter_set', Set, SetTypeError),
                   ('dim', (int, tuple, list), DimTypeError),
                   ('name', str, NameTypeError))
    def __init__(self, iter_set, dim=1, name=None):
        self._set = iter_set
        self._dim = as_tuple(dim, int)
        self._cdim = np.asscalar(np.prod(self._dim))
        self._name = name or "dset_%d" % DataSet._globalcount
        DataSet._globalcount += 1

    def __getstate__(self):
        """Extract state to pickle."""
        return self.__dict__

    def __setstate__(self, d):
        """Restore from pickled state."""
        self.__dict__.update(d)

    # Look up any unspecified attributes on the _set.
    def __getattr__(self, name):
        """Returns a Set specific attribute."""
        return getattr(self._set, name)

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

    def __eq__(self, other):
        """:class:`DataSet`\s compare equal if they are defined on the same
        :class:`Set` and have the same ``dim``."""
        return self.set == other.set and self.dim == other.dim

    def __ne__(self, other):
        """:class:`DataSet`\s compare equal if they are defined on the same
        :class:`Set` and have the same ``dim``."""
        return not self == other

    def __str__(self):
        return "OP2 DataSet: %s on set %s, with dim %s" % \
            (self._name, self._set, self._dim)

    def __repr__(self):
        return "DataSet(%r, %r, %r)" % (self._set, self._dim, self._name)

    def __contains__(self, dat):
        """Indicate whether a given Dat is compatible with this DataSet."""
        return dat.dataset == self


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
    def __init__(self, iterset, extents=()):
        self._iterset = iterset
        self._extents = as_tuple(extents, int)

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
        """Number of layers in the extruded mesh"""
        return self._iterset.layers

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
        return self._extents, self.iterset.layers


class DataCarrier(object):

    """Abstract base class for OP2 data.

    Actual objects will be :class:`DataCarrier` objects of rank 0
    (:class:`Const` and :class:`Global`), rank 1 (:class:`Dat`), or rank 2
    (:class:`Mat`)"""

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

    def _force_evaluation(self):
        """Force the evaluation of any outstanding computation to ensure that this DataCarrier is up to date"""
        _trace.evaluate(set([self]), set([self]))


class Dat(DataCarrier):

    """OP2 vector data. A :class:`Dat` holds values on every element of a
    :class:`DataSet`.

    If a :class:`Set` is passed as the ``dataset`` argument, rather
    than a :class:`DataSet`, the :class:`Dat` is created with a default
    :class:`DataSet` dimension of 1.

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

    @validate_type(('dataset', (DataSet, Set), DataSetTypeError), ('name', str, NameTypeError))
    @validate_dtype(('dtype', None, DataTypeError))
    def __init__(self, dataset, data=None, dtype=None, name=None,
                 soa=None, uid=None):
        if type(dataset) is Set:
            # If a Set, rather than a dataset is passed in, default to
            # a dataset dimension of 1.
            dataset = dataset ** 1
        self._shape = (dataset.total_size,) + (() if dataset.cdim == 1 else dataset.dim)
        self._dataset = dataset
        if data is None:
            self._dtype = dtype if dtype is not None else np.float64
        else:
            self._data = verify_reshape(data, dtype, self._shape, allow_none=True)
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
            path._dat = self
            path._access = access
            path._flatten = flatten
            return path
        if path and path.toset != self.dataset.set:
            raise MapValueError("To Set of Map does not match Set of Dat.")
        return _make_object('Arg', data=self, map=path, access=access, flatten=flatten)

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
    @collective
    def data(self):
        """Numpy array containing the data values."""
        _trace.evaluate(set([self]), set([self]))
        if self.dataset.total_size > 0 and self._data.size == 0:
            raise RuntimeError("Illegal access: no data associated with this Dat!")
        maybe_setflags(self._data, write=True)
        self.needs_halo_update = True
        return self._data

    @property
    def data_ro(self):
        """Numpy array containing the data values.  Read-only"""
        _trace.evaluate(set([self]), set())
        if self.dataset.total_size > 0 and self._data.size == 0:
            raise RuntimeError("Illegal access: no data associated with this Dat!")
        maybe_setflags(self._data, write=False)
        return self._data

    def save(self, filename):
        """Write the data array to file ``filename`` in NumPy format."""
        np.save(filename, self.data_ro)

    @property
    def _data(self):
        if not self._is_allocated:
            self._numpy_data = np.zeros(self._shape, dtype=self._dtype)
        return self._numpy_data

    @_data.setter
    def _data(self, value):
        self._numpy_data = value

    @property
    def _is_allocated(self):
        return hasattr(self, '_numpy_data')

    @property
    def dtype(self):
        return self._data.dtype if self._is_allocated else self._dtype

    @property
    def needs_halo_update(self):
        '''Has this Dat been written to since the last halo exchange?'''
        return self._needs_halo_update

    @needs_halo_update.setter
    @collective
    def needs_halo_update(self, val):
        """Indictate whether this Dat requires a halo update"""
        self._needs_halo_update = val

    @collective
    def zero(self):
        """Zero the data associated with this :class:`Dat`"""
        if not hasattr(self, '_zero_kernel'):
            k = """void zero(%(t)s *dat) {
                for (int n = 0; n < %(dim)s; ++n) {
                    dat[n] = (%(t)s)0;
                }
            }""" % {'t': self.ctype, 'dim': self.cdim}
            self._zero_kernel = _make_object('Kernel', k, 'zero')
        _make_object('ParLoop', self._zero_kernel, self.dataset.set,
                     self(WRITE)).enqueue()

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

    def __str__(self):
        return "OP2 Dat: %s on (%s) with datatype %s" \
               % (self._name, self._dataset, self._data.dtype.name)

    def __repr__(self):
        return "Dat(%r, None, %r, %r)" \
               % (self._dataset, self._data.dtype, self._name)

    def _check_shape(self, other):
        pass

    def _op(self, other, op):
        if np.isscalar(other):
            return Dat(self.dataset,
                       op(self._data, as_type(other, self.dtype)), self.dtype)
        self._check_shape(other)
        return Dat(self.dataset,
                   op(self._data, as_type(other.data, self.dtype)), self.dtype)

    def _iop(self, other, op):
        if np.isscalar(other):
            op(self._data, as_type(other, self.dtype))
        else:
            self._check_shape(other)
            op(self._data, as_type(other.data, self.dtype))
        return self

    def __add__(self, other):
        """Pointwise addition of fields."""
        return self._op(other, operator.add)

    def __sub__(self, other):
        """Pointwise subtraction of fields."""
        return self._op(other, operator.sub)

    def __mul__(self, other):
        """Pointwise multiplication or scaling of fields."""
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
        _MPI.Request.Waitall(self._recv_reqs.values())
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

    @property
    def norm(self):
        """The L2-norm on the flattened vector."""
        return np.linalg.norm(self._data)

    @classmethod
    def fromhdf5(cls, dataset, f, name):
        """Construct a :class:`Dat` from a Dat named ``name`` in HDF5 data ``f``"""
        slot = f[name]
        data = slot.value
        soa = slot.attrs['type'].find(':soa') > 0
        ret = cls(dataset, data, name=name, soa=soa)
        return ret


class Const(DataCarrier):

    """Data that is constant for any element of any set."""

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

    @property
    def data(self):
        """Data array."""
        if len(self._data) is 0:
            raise RuntimeError("Illegal access: No data associated with this Const!")
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)

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
        _trace.evaluate(set(), set([self]))
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


class Global(DataCarrier):

    """OP2 global value.

    When a ``Global`` is passed to a :func:`pyop2.op2.par_loop`, the access
    descriptor is passed by `calling` the ``Global``.  For example, if
    a ``Global`` named ``G`` is to be accessed for reading, this is
    accomplished by::

      G(pyop2.READ)
    """

    _globalcount = 0
    _modes = [READ, INC, MIN, MAX]

    @validate_type(('name', str, NameTypeError))
    def __init__(self, dim, data=None, dtype=None, name=None):
        self._dim = as_tuple(dim, int)
        self._cdim = np.asscalar(np.prod(self._dim))
        self._data = verify_reshape(data, dtype, self._dim, allow_none=True)
        self._buf = np.empty_like(self._data)
        self._name = name or "global_%d" % Global._globalcount
        Global._globalcount += 1

    @validate_in(('access', _modes, ModeValueError))
    def __call__(self, access, path=None):
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

    def __str__(self):
        return "OP2 Global Argument: %s with dim %s and value %s" \
            % (self._name, self._dim, self._data)

    def __repr__(self):
        return "Global(%r, %r, %r, %r)" % (self._dim, self._data,
                                           self._data.dtype, self._name)

    @property
    def data(self):
        """Data array."""
        _trace.evaluate(set([self]), set())
        if len(self._data) is 0:
            raise RuntimeError("Illegal access: No data associated with this Global!")
        return self._data

    @data.setter
    def data(self, value):
        _trace.evaluate(set(), set([self]))
        self._data = verify_reshape(value, self.dtype, self.dim)

    @property
    def soa(self):
        """Are the data in SoA format? This is always false for :class:`Global`
        objects."""
        return False

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
    """

    _globalcount = 0

    @validate_type(('iterset', Set, SetTypeError), ('toset', Set, SetTypeError),
                  ('arity', int, ArityTypeError), ('name', str, NameTypeError))
    def __init__(self, iterset, toset, arity, values=None, name=None, offset=None):
        self._iterset = iterset
        self._toset = toset
        self._arity = arity
        self._values = verify_reshape(values, np.int32, (iterset.total_size, arity),
                                      allow_none=True)
        self._name = name or "map_%d" % Map._globalcount
        self._offset = offset
        Map._globalcount += 1

    @validate_type(('index', (int, IterationIndex), IndexTypeError))
    def __getitem__(self, index):
        if isinstance(index, int) and not (0 <= index < self._arity):
            raise IndexValueError("Index must be in interval [0,%d]" % (self._arity - 1))
        if isinstance(index, IterationIndex) and index.index not in [0, 1]:
            raise IndexValueError("IterationIndex must be in interval [0,1]")
        return _make_object('Arg', map=self, idx=index)

    # This is necessary so that we can convert a Map to a tuple
    # (needed in as_tuple).  Because, __getitem__ no longer returns a
    # Map we have to explicitly provide an iterable interface
    def __iter__(self):
        yield self

    def __getslice__(self, i, j):
        raise NotImplementedError("Slicing maps is not currently implemented")

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
    def values(self):
        """Mapping array."""
        return self._values

    @property
    def name(self):
        """User-defined label"""
        return self._name

    @property
    def offset(self):
        """The vertical offset."""
        return self._offset

    def __str__(self):
        return "OP2 Map: %s from (%s) to (%s) with arity %s" \
               % (self._name, self._iterset, self._toset, self._arity)

    def __repr__(self):
        return "Map(%r, %r, %r, None, %r)" \
               % (self._iterset, self._toset, self._arity, self._name)

    def __eq__(self, o):
        """:class:`Map`\s compare equal if defined on the same ``iterset``,
        ``toset`` and have the same ``arity`` and ``data``."""
        try:
            return (self._iterset == o._iterset and self._toset == o._toset and
                    self._arity == o.arity and np.array_equal(self._values, o._values))
        except AttributeError:
            return False

    def __ne__(self, o):
        return not self == o

    @classmethod
    def fromhdf5(cls, iterset, toset, f, name):
        """Construct a :class:`Map` from set named ``name`` in HDF5 data ``f``"""
        slot = f[name]
        values = slot.value
        arity = slot.shape[1:]
        if len(arity) != 1:
            raise ArityTypeError("Unrecognised arity value %s" % arity)
        return cls(iterset, toset, arity[0], values, name)


class Sparsity(Cached):

    """OP2 Sparsity, the non-zero structure a matrix derived from the union of
    the outer product of pairs of :class:`Map` objects.

    Examples of constructing a Sparsity: ::

        Sparsity(single_dset, single_map, 'mass')
        Sparsity((row_dset, col_dset), (single_rowmap, single_colmap))
        Sparsity((row_dset, col_dset),
                 [(first_rowmap, first_colmap), (second_rowmap, second_colmap)])

    .. _MatMPIAIJSetPreallocation: http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatMPIAIJSetPreallocation.html
    """

    _cache = {}
    _globalcount = 0

    @classmethod
    @validate_type(('dsets', (Set, DataSet, tuple), DataSetTypeError),
                   ('maps', (Map, tuple), MapTypeError),
                   ('name', str, NameTypeError))
    def _process_args(cls, dsets, maps, name=None, *args, **kwargs):
        "Turn maps argument into a canonical tuple of pairs."

        # A single data set becomes a pair of identical data sets
        dsets = [dsets, dsets] if isinstance(dsets, (Set, DataSet)) else list(dsets)

        # Check data sets are valid
        for i, _ in enumerate(dsets):
            if type(dsets[i]) is Set:
                dsets[i] = (dsets[i]) ** 1
            dset = dsets[i]
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
                if len(m.values) == 0:
                    raise MapValueError(
                        "Unpopulated map values when trying to build sparsity.")
        # Need to return a list of args and dict of kwargs (empty in this case)
        return [tuple(dsets), tuple(sorted(maps)), name], {}

    @classmethod
    def _cache_key(cls, dsets, maps, *args, **kwargs):
        return (dsets, maps)

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

        # Default to a dataset dimension of 1 if we got a Set instead.
        for i, _ in enumerate(dsets):
            if type(dsets[i]) is Set:
                dsets[i] = (dsets[i]) ** 1

        self._dsets = dsets

        assert len(self._rmaps) == len(self._cmaps), \
            "Must pass equal number of row and column maps"

        # Make sure that the "to" Set of each map in a pair is the set of the
        # corresponding DataSet set
        for pair in maps:
            if not (pair[0].toset == dsets[0].set and
                    pair[1].toset == dsets[1].set):
                raise RuntimeError("Map to set must be the same as corresponding DataSet set")

        # Each pair of maps must have the same from-set (iteration set)
        for pair in maps:
            if not pair[0].iterset == pair[1].iterset:
                raise RuntimeError("Iterset of both maps in a pair must be the same")

        # Each row map must have the same to-set (data set)
        if not all(m.toset == self._rmaps[0].toset for m in self._rmaps):
            raise RuntimeError("To set of all row maps must be the same")

        # Each column map must have the same to-set (data set)
        if not all(m.toset == self._cmaps[0].toset for m in self._cmaps):
            raise RuntimeError("To set of all column maps must be the same")

        # All rmaps and cmaps have the same data set - just use the first.
        self._nrows = self._rmaps[0].toset.size
        self._ncols = self._cmaps[0].toset.size
        self._dims = (self._dsets[0].cdim, self._dsets[1].cdim)

        self._name = name or "sparsity_%d" % Sparsity._globalcount
        Sparsity._globalcount += 1
        build_sparsity(self, parallel=MPI.parallel)
        self._initialized = True

    @property
    def _nmaps(self):
        return len(self._rmaps)

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
        """Number of non-zeroes per row in diagonal portion of the local
        submatrix.

        This is the same as the parameter `d_nz` used for preallocation in
        PETSc's MatMPIAIJSetPreallocation_."""
        return int(self._d_nz)

    @property
    def onz(self):
        """Number of non-zeroes per row in off-diagonal portion of the local
        submatrix.

        This is the same as the parameter o_nz used for preallocation in
        PETSc's MatMPIAIJSetPreallocation_."""
        return int(self._o_nz)


class Mat(DataCarrier):

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
    details."""

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
        if tuple(path_maps) not in self.sparsity.maps:
            raise MapValueError("Path maps not in sparsity maps")
        return _make_object('Arg', data=self, map=path_maps, access=access,
                            idx=path_idxs, flatten=flatten)

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
        return np.prod(self.dims) == 1

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

    def __str__(self):
        return "OP2 Mat: %s, sparsity (%s), datatype %s" \
               % (self._name, self._sparsity, self._datatype.name)

    def __repr__(self):
        return "Mat(%r, %r, %r)" \
               % (self._sparsity, self._datatype, self._name)

# Kernel API


class Kernel(Cached):

    """OP2 kernel type."""

    _globalcount = 0
    _cache = {}

    @classmethod
    @validate_type(('name', str, NameTypeError))
    def _cache_key(cls, code, name):
        # Both code and name are relevant since there might be multiple kernels
        # extracting different functions from the same code
        return md5(code + name).hexdigest()

    def __init__(self, code, name):
        # Protect against re-initialization when retrieved from cache
        if self._initialized:
            return
        self._name = name or "kernel_%d" % Kernel._globalcount
        self._code = preprocess(code)
        Kernel._globalcount += 1
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

    """Cached module encapsulating the generated :class:`ParLoop` stub."""

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
                key += (arg.data.dims, arg.data.dtype, idxs,
                        map_arities, arg.access)

        # The currently defined Consts need to be part of the cache key, since
        # these need to be uploaded to the device before launching the kernel
        for c in Const._definitions():
            key += (c.name, c.dtype, c.cdim)

        return key


class ParLoop(LazyComputation):
    """Represents the kernel, iteration space and arguments of a parallel loop
    invocation.

    .. note ::

        Users should not directly construct :class:`ParLoop` objects, but
        use :func:`pyop2.op2.par_loop` instead.
    """

    @validate_type(('kernel', Kernel, KernelTypeError),
                   ('iterset', Set, SetTypeError))
    def __init__(self, kernel, iterset, *args):
        LazyComputation.__init__(self,
                                 set([a.data for a in args if a.access in [READ, RW]]) | Const._defs,
                                 set([a.data for a in args if a.access in [RW, WRITE, MIN, MAX, INC]]))
        # Always use the current arguments, also when we hit cache
        self._actual_args = args
        self._kernel = kernel
        self._is_layered = iterset.layers > 1

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

        self._it_space = IterationSpace(iterset, self.check_args(iterset))

    def _run(self):
        return self.compute()

    @collective
    def compute(self):
        """Executes the kernel over all members of the iteration space."""
        self.halo_exchange_begin()
        self.maybe_set_dat_dirty()
        self._compute_if_not_empty(self.it_space.iterset.core_part)
        self.halo_exchange_end()
        self._compute_if_not_empty(self.it_space.iterset.owned_part)
        self.reduction_begin()
        if self.needs_exec_halo:
            self._compute_if_not_empty(self.it_space.iterset.exec_part)
        self.reduction_end()
        self.maybe_set_halo_update_needed()
        self.assemble()

    def _compute_if_not_empty(self, part):
        if part.size > 0:
            self._compute(part)

    def _compute(self, part):
        """Executes the kernel over all members of a MPI-part of the iteration space."""
        raise RuntimeError("Must select a backend")

    def maybe_set_dat_dirty(self):
        for arg in self.args:
            if arg._is_dat:
                maybe_setflags(arg.data._data, write=False)

    @collective
    def halo_exchange_begin(self):
        """Start halo exchanges."""
        if self.is_direct:
            # No need for halo exchanges for a direct loop
            return
        for arg in self.args:
            if arg._is_dat:
                arg.halo_exchange_begin()

    @collective
    def halo_exchange_end(self):
        """Finish halo exchanges (wait on irecvs)"""
        if self.is_direct:
            return
        for arg in self.args:
            if arg._is_dat:
                arg.halo_exchange_end()

    @collective
    def reduction_begin(self):
        """Start reductions"""
        for arg in self.args:
            if arg._is_global_reduction:
                arg.reduction_begin()

    @collective
    def reduction_end(self):
        """End reductions"""
        for arg in self.args:
            if arg._is_global_reduction:
                arg.reduction_end()

    @collective
    def maybe_set_halo_update_needed(self):
        """Set halo update needed for :class:`Dat` arguments that are written to
        in this parallel loop."""
        for arg in self.args:
            if arg._is_dat and arg.access in [INC, WRITE, RW]:
                arg.data.needs_halo_update = True

    def assemble(self):
        for arg in self.args:
            if arg._is_mat:
                arg.data._assemble()

    def check_args(self, iterset):
        """Checks that the iteration set of the :class:`ParLoop` matches the
        iteration set of all its arguments. A :class:`MapValueError` is raised
        if this condition is not met.

        Also determines the size of the local iteration space and checks all
        arguments using an :class:`IterationIndex` for consistency.

        :return: size of the local iteration space"""
        itspace = ()
        for i, arg in enumerate(self._actual_args):
            if arg._is_global or arg.map is None:
                continue
            for j, m in enumerate(arg._map):
                if m.iterset != iterset:
                    raise MapValueError(
                        "Iterset of arg %s map %s doesn't match ParLoop iterset." % (i, j))
            if arg._uses_itspace:
                _itspace = tuple(m.arity for m in arg._map)
                if itspace and itspace != _itspace:
                    raise IndexValueError("Mismatching iteration space size for argument %d" % i)
                itspace = _itspace
        return itspace

    def offset_args(self):
        """The offset args that need to be added to the argument list."""
        _args = []
        for arg in self.args:
            if arg._is_indirect or arg._is_mat:
                maps = as_tuple(arg.map, Map)
                for map in maps:
                    if map.iterset.layers is not None and map.iterset.layers > 1:
                        _args.append(map.offset)
        return _args

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
    def needs_exec_halo(self):
        """Does the parallel loop need an exec halo?"""
        return any(arg._is_indirect_and_not_read or arg._is_mat
                   for arg in self.args)

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

DEFAULT_SOLVER_PARAMETERS = {'ksp_type': 'cg',
                             'pc_type': 'jacobi',
                             'ksp_rtol': 1.0e-7,
                             'ksp_atol': 1.0e-50,
                             'ksp_divtol': 1.0e+4,
                             'ksp_max_it': 1000,
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

    :arg linear_solver: the solver type ('cg')
    :arg preconditioner: the preconditioner type ('jacobi')
    :arg relative_tolerance: relative solver tolerance (1e-7)
    :arg absolute_tolerance: absolute solver tolerance (1e-50)
    :arg divergence_tolerance: factor by which the residual norm may exceed
        the right-hand-side norm before the solve is considered to have
        diverged: ``norm(r) >= dtol*norm(b)`` (1e4)
    :arg maximum_iterations: maximum number of solver iterations (1000)
    :arg error_on_nonconvergence: abort if the solve does not converge in the
      maximum number of iterations (True, if False only a warning is printed)
    :arg monitor_convergence: print the residual norm after each iteration
        (False)
    :arg plot_convergence: plot a graph of the convergence history after the
        solve has finished and save it to file (False, implies monitor_convergence)
    :arg plot_prefix: filename prefix for plot files ('')
    :arg gmres_restart: restart period when using GMRES

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

    @collective
    def solve(self, A, x, b):
        """Solve a matrix equation.

        :arg A: The :class:`Mat` containing the matrix.
        :arg x: The :class:`Dat` to receive the solution.
        :arg b: The :class:`Dat` containing the RHS.
        """
        _trace.evaluate(set([A, b]), set([x]))
        self._solve(A, x, b)

    def _solve(self, A, x, b):
        raise NotImplementedError("solve must be implemented by backend")


@collective
def par_loop(kernel, it_space, *args):
    return _make_object('ParLoop', kernel, it_space, *args).enqueue()
