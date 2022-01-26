import ctypes
import itertools
import functools
from functools import cached_property
import numbers

import numpy as np

from pyop2 import (
    caching,
    datatypes as dtypes,
    exceptions as ex,
    utils
)
from pyop2.types.set import GlobalSet, MixedSet, Set


class Map:

    """OP2 map, a relation between two :class:`Set` objects.

    Each entry in the ``iterset`` maps to ``arity`` entries in the
    ``toset``. When a map is used in a :func:`pyop2.op2.par_loop`, it is
    possible to use Python index notation to select an individual entry on the
    right hand side of this map. There are three possibilities:

    * No index. All ``arity`` :class:`Dat` entries will be passed to the
      kernel.
    * An integer: ``some_map[n]``. The ``n`` th entry of the
      map result will be passed to the kernel.
    """

    dtype = dtypes.IntType

    @utils.validate_type(('iterset', Set, ex.SetTypeError), ('toset', Set, ex.SetTypeError),
                         ('arity', numbers.Integral, ex.ArityTypeError), ('name', str, ex.NameTypeError))
    def __init__(self, iterset, toset, arity, values=None, name=None, offset=None):
        self._iterset = iterset
        self._toset = toset
        self.comm = toset.comm
        self._arity = arity
        self._values = utils.verify_reshape(values, dtypes.IntType,
                                            (iterset.total_size, arity), allow_none=True)
        self.shape = (iterset.total_size, arity)
        self._name = name or "map_#x%x" % id(self)
        if offset is None or len(offset) == 0:
            self._offset = None
        else:
            self._offset = utils.verify_reshape(offset, dtypes.IntType, (arity, ))
        # A cache for objects built on top of this map
        self._cache = {}

    @cached_property
    def _kernel_args_(self):
        return (self._values.ctypes.data, )

    @cached_property
    def _argtypes_(self):
        return (ctypes.c_voidp, )

    @cached_property
    def _wrapper_cache_key_(self):
        return (type(self), self.arity, utils.tuplify(self.offset))

    # This is necessary so that we can convert a Map to a tuple
    # (needed in as_tuple).  Because, __getitem__ no longer returns a
    # Map we have to explicitly provide an iterable interface
    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    @cached_property
    def split(self):
        return (self,)

    @cached_property
    def iterset(self):
        """:class:`Set` mapped from."""
        return self._iterset

    @cached_property
    def toset(self):
        """:class:`Set` mapped to."""
        return self._toset

    @cached_property
    def arity(self):
        """Arity of the mapping: number of toset elements mapped to per
        iterset element."""
        return self._arity

    @cached_property
    def arities(self):
        """Arity of the mapping: number of toset elements mapped to per
        iterset element.

        :rtype: tuple"""
        return (self._arity,)

    @cached_property
    def arange(self):
        """Tuple of arity offsets for each constituent :class:`Map`."""
        return (0, self._arity)

    @cached_property
    def values(self):
        """Mapping array.

        This only returns the map values for local points, to see the
        halo points too, use :meth:`values_with_halo`."""
        return self._values[:self.iterset.size]

    @cached_property
    def values_with_halo(self):
        """Mapping array.

        This returns all map values (including halo points), see
        :meth:`values` if you only need to look at the local
        points."""
        return self._values

    @cached_property
    def name(self):
        """User-defined label"""
        return self._name

    @cached_property
    def offset(self):
        """The vertical offset."""
        return self._offset

    def __str__(self):
        return "OP2 Map: %s from (%s) to (%s) with arity %s" \
               % (self._name, self._iterset, self._toset, self._arity)

    def __repr__(self):
        return "Map(%r, %r, %r, None, %r)" \
               % (self._iterset, self._toset, self._arity, self._name)

    def __le__(self, o):
        """self<=o if o equals self or self._parent <= o."""
        return self == o


class PermutedMap(Map):
    """Composition of a standard :class:`Map` with a constant permutation.

    :arg map_: The map to permute.
    :arg permutation: The permutation of the map indices.

    Where normally staging to element data is performed as

    .. code-block::

       local[i] = global[map[i]]

    With a :class:`PermutedMap` we instead get

    .. code-block::

       local[i] = global[map[permutation[i]]]

    This might be useful if your local kernel wants data in a
    different order to the one that the map provides, and you don't
    want two global-sized data structures.
    """
    def __init__(self, map_, permutation):
        self.map_ = map_
        self.permutation = np.asarray(permutation, dtype=Map.dtype)
        assert (np.unique(permutation) == np.arange(map_.arity, dtype=Map.dtype)).all()

    @cached_property
    def _wrapper_cache_key_(self):
        return super()._wrapper_cache_key_ + (tuple(self.permutation),)

    def __getattr__(self, name):
        return getattr(self.map_, name)


class MixedMap(Map, caching.ObjectCached):
    r"""A container for a bag of :class:`Map`\s."""

    def __init__(self, maps):
        r""":param iterable maps: Iterable of :class:`Map`\s"""
        if self._initialized:
            return
        self._maps = maps
        if not all(m is None or m.iterset == self.iterset for m in self._maps):
            raise ex.MapTypeError("All maps in a MixedMap need to share the same iterset")
        # TODO: Think about different communicators on maps (c.f. MixedSet)
        # TODO: What if all maps are None?
        comms = tuple(m.comm for m in self._maps if m is not None)
        if not all(c == comms[0] for c in comms):
            raise ex.MapTypeError("All maps needs to share a communicator")
        if len(comms) == 0:
            raise ex.MapTypeError("Don't know how to make communicator")
        self.comm = comms[0]
        self._initialized = True

    @classmethod
    def _process_args(cls, *args, **kwargs):
        maps = utils.as_tuple(args[0], type=Map, allow_none=True)
        cache = maps[0]
        return (cache, ) + (maps, ), kwargs

    @classmethod
    def _cache_key(cls, maps):
        return maps

    @cached_property
    def _kernel_args_(self):
        return tuple(itertools.chain(*(m._kernel_args_ for m in self if m is not None)))

    @cached_property
    def _argtypes_(self):
        return tuple(itertools.chain(*(m._argtypes_ for m in self if m is not None)))

    @cached_property
    def _wrapper_cache_key_(self):
        return tuple(m._wrapper_cache_key_ for m in self if m is not None)

    @cached_property
    def split(self):
        r"""The underlying tuple of :class:`Map`\s."""
        return self._maps

    @cached_property
    def iterset(self):
        """:class:`MixedSet` mapped from."""
        return functools.reduce(lambda a, b: a or b, map(lambda s: s if s is None else s.iterset, self._maps))

    @cached_property
    def toset(self):
        """:class:`MixedSet` mapped to."""
        return MixedSet(tuple(GlobalSet(comm=self.comm) if m is None else
                              m.toset for m in self._maps))

    @cached_property
    def arity(self):
        """Arity of the mapping: total number of toset elements mapped to per
        iterset element."""
        return sum(m.arity for m in self._maps)

    @cached_property
    def arities(self):
        """Arity of the mapping: number of toset elements mapped to per
        iterset element.

        :rtype: tuple"""
        return tuple(m.arity for m in self._maps)

    @cached_property
    def arange(self):
        """Tuple of arity offsets for each constituent :class:`Map`."""
        return (0,) + tuple(np.cumsum(self.arities))

    @cached_property
    def values(self):
        """Mapping arrays excluding data for halos.

        This only returns the map values for local points, to see the
        halo points too, use :meth:`values_with_halo`."""
        return tuple(m.values for m in self._maps)

    @cached_property
    def values_with_halo(self):
        """Mapping arrays including data for halos.

        This returns all map values (including halo points), see
        :meth:`values` if you only need to look at the local
        points."""
        return tuple(None if m is None else
                     m.values_with_halo for m in self._maps)

    @cached_property
    def name(self):
        """User-defined labels"""
        return tuple(m.name for m in self._maps)

    @cached_property
    def offset(self):
        """Vertical offsets."""
        return tuple(0 if m is None else m.offset for m in self._maps)

    def __iter__(self):
        r"""Yield all :class:`Map`\s when iterated over."""
        for m in self._maps:
            yield m

    def __len__(self):
        r"""Number of contained :class:`Map`\s."""
        return len(self._maps)

    def __le__(self, o):
        """self<=o if o equals self or its self._parent==o."""
        return self == o or all(m <= om for m, om in zip(self, o))

    def __str__(self):
        return "OP2 MixedMap composed of Maps: %s" % (self._maps,)

    def __repr__(self):
        return "MixedMap(%r)" % (self._maps,)
