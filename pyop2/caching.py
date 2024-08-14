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

"""Provides common base classes for cached objects."""

import cachetools
import hashlib
import os
import pickle
from collections.abc import MutableMapping
from pathlib import Path
from warnings import warn  # noqa F401
from functools import wraps, partial

from pyop2.configuration import configuration
from pyop2.logger import debug
from pyop2.mpi import comm_cache_keyval, COMM_WORLD


def report_cache(typ):
    """Report the size of caches of type ``typ``

    :arg typ: A class of cached object.  For example
        :class:`ObjectCached` or :class:`Cached`.
    """
    from collections import defaultdict
    from inspect import getmodule
    from gc import get_objects
    typs = defaultdict(lambda: 0)
    n = 0
    for x in get_objects():
        if isinstance(x, typ):
            typs[type(x)] += 1
            n += 1
    if n == 0:
        print("\nNo %s objects in caches" % typ.__name__)
        return
    print("\n%d %s objects in caches" % (n, typ.__name__))
    print("Object breakdown")
    print("================")
    for k, v in typs.iteritems():
        mod = getmodule(k)
        if mod is not None:
            name = "%s.%s" % (mod.__name__, k.__name__)
        else:
            name = k.__name__
        print('%s: %d' % (name, v))


class ObjectCached(object):
    """Base class for objects that should be cached on another object.

    Derived classes need to implement classmethods
    :meth:`_process_args` and :meth:`_cache_key` (which see for more
    details).  The object on which the cache is stored should contain
    a dict in its ``_cache`` attribute.

    .. warning::

        The derived class' :meth:`__init__` is still called if the
        object is retrieved from cache. If that is not desired,
        derived classes can set a flag indicating whether the
        constructor has already been called and immediately return
        from :meth:`__init__` if the flag is set. Otherwise the object
        will be re-initialized even if it was returned from cache!

    """

    @classmethod
    def _process_args(cls, *args, **kwargs):
        """Process the arguments to ``__init__`` into a form suitable
        for computing a cache key on.

        The first returned argument is popped off the argument list
        passed to ``__init__`` and is used as the object on which to
        cache this instance.  As such, *args* should be returned as a
        two-tuple of ``(cache_object, ) + (original_args, )``.

        *kwargs* must be a (possibly empty) dict.
        """
        raise NotImplementedError("Subclass must implement _process_args")

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        """Compute a cache key from the constructor's preprocessed arguments.
        If ``None`` is returned, the object is not to be cached.

        .. note::

           The return type **must** be hashable.

        """
        raise NotImplementedError("Subclass must implement _cache_key")

    def __new__(cls, *args, **kwargs):
        args, kwargs = cls._process_args(*args, **kwargs)
        # First argument is the object we're going to cache on
        cache_obj = args[0]
        # These are now the arguments to the subclass constructor
        args = args[1:]
        key = cls._cache_key(*args, **kwargs)

        def make_obj():
            obj = super(ObjectCached, cls).__new__(cls)
            obj._initialized = False
            # obj.__init__ will be called twice when constructing
            # something not in the cache.  The first time here, with
            # the canonicalised args, the second time directly in the
            # subclass.  But that one should hit the cache and return
            # straight away.
            obj.__init__(*args, **kwargs)
            return obj

        # Don't bother looking in caches if we're not meant to cache
        # this object.
        if key is None or cache_obj is None:
            return make_obj()

        # Does the caching object know about the caches?
        try:
            cache = cache_obj._cache
        except AttributeError:
            raise RuntimeError("Provided caching object does not have a '_cache' attribute.")

        # OK, we have a cache, let's go ahead and try and find our
        # object in it.
        try:
            return cache[key]
        except KeyError:
            obj = make_obj()
            cache[key] = obj
            return obj


class _CacheMiss:
    pass


CACHE_MISS = _CacheMiss()


class _CacheKey:
    def __init__(self, key_value):
        self.value = key_value


def _as_hexdigest(*args):
    hash_ = hashlib.md5()
    for a in args:
        hash_.update(str(a).encode())
    return hash_.hexdigest()


def clear_memory_cache(comm):
    if comm.Get_attr(comm_cache_keyval) is not None:
        comm.Set_attr(comm_cache_keyval, {})


class DictLikeDiskAccess(MutableMapping):
    def __init__(self, cachedir):
        """

        :arg cachedir: The cache directory.
        """
        self.cachedir = cachedir
        self._keys = set()

    def __getitem__(self, key):
        """Retrieve a value from the disk cache.

        :arg key: The cache key, a 2-tuple of strings.
        :returns: The cached object if found.
        """
        filepath = Path(self.cachedir, key[0][:2], key[0][2:] + key[1])
        try:
            with self.open(filepath, "rb") as fh:
                value = self.read(fh)
        except FileNotFoundError:
            raise KeyError("File not on disk, cache miss")
        return value

    def __setitem__(self, key, value):
        """Store a new value in the disk cache.

        :arg key: The cache key, a 2-tuple of strings.
        :arg value: The new item to store in the cache.
        """
        self._keys.add(key)
        k1, k2 = key[0][:2], key[0][2:] + key[1]
        basedir = Path(self.cachedir, k1)
        basedir.mkdir(parents=True, exist_ok=True)

        tempfile = basedir.joinpath(f"{k2}_p{os.getpid()}.tmp")
        filepath = basedir.joinpath(k2)
        with self.open(tempfile, "wb") as fh:
            self.write(fh, value)
        tempfile.rename(filepath)

    def __delitem__(self, key):
        raise ValueError(f"Cannot remove items from {self.__class__.__name__}")

    def keys(self):
        return self._keys

    def __iter__(self):
        for k in self._keys:
            yield k

    def __len__(self):
        return len(self._keys)

    def __repr__(self):
        return "{" + " ".join(f"{k}: {v}" for k, v in self.items()) + "}"

    def open(self, *args, **kwargs):
        return open(*args, **kwargs)

    def read(self, filehandle):
        return pickle.load(filehandle)

    def write(self, filehandle, value):
        pickle.dump(value, filehandle)


def default_comm_fetcher(*args, **kwargs):
    return kwargs.get("comm")


default_parallel_hashkey = cachetools.keys.hashkey


def parallel_cache(hashkey=default_parallel_hashkey, comm_fetcher=default_comm_fetcher, cache_factory=None, broadcast=True):
    """Memory only cache decorator.

    Decorator for wrapping a function to be called over a communiucator in a
    cache that stores broadcastable values in memory. If the value is found in
    the cache of rank 0 it is broadcast to all other ranks.

    :arg key: Callable returning the cache key for the function inputs. This
        function must return a 2-tuple where the first entry is the
        communicator to be collective over and the second is the key. This is
        required to ensure that deadlocks do not occur when using different
        subcommunicators.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """ Extract the key and then try the memory cache before falling back
            on calling the function and populating the cache.
            """
            comm = comm_fetcher(*args, **kwargs)
            k = hashkey(*args, **kwargs)
            key = _as_hexdigest(k), func.__qualname__

            # Fetch the per-comm cache_collection or set it up if not present
            # A collection is required since different types of cache can be set up on the same comm
            cache_collection = comm.Get_attr(comm_cache_keyval)
            if cache_collection is None:
                cache_collection = {}
                comm.Set_attr(comm_cache_keyval, cache_collection)
            # If this kind of cache is already present on the
            # cache_collection, get it, otherwise create it
            local_cache = cache_collection.setdefault(
                (cf := cache_factory()).__class__.__name__,
                cf
            )

            if broadcast:
                # Grab value from rank 0 memory cache and broadcast result
                if comm.rank == 0:
                    value = local_cache.get(key, CACHE_MISS)
                    if value is None:
                        debug(f'{COMM_WORLD.name} R{COMM_WORLD.rank}, {comm.name} R{comm.rank}: {k} memory cache miss')
                    else:
                        debug(f'{COMM_WORLD.name} R{COMM_WORLD.rank}, {comm.name} R{comm.rank}: {k} memory cache hit')
                    # TODO: Add communication tags to avoid cross-broadcasting
                    comm.bcast(value, root=0)
                else:
                    value = comm.bcast(CACHE_MISS, root=0)
                    if isinstance(value, _CacheMiss):
                        # We might have the CACHE_MISS from rank 0 and
                        # `(value is CACHE_MISS) == False` which is confusing,
                        # so we set it back to the local value
                        value = CACHE_MISS
            else:
                # Grab value from all ranks cache and broadcast cache hit/miss
                value = local_cache.get(key, CACHE_MISS)
                if value is CACHE_MISS:
                    debug(f'{COMM_WORLD.name} R{COMM_WORLD.rank}, {comm.name} R{comm.rank}: {k} memory cache miss')
                    cache_hit = False
                else:
                    debug(f'{COMM_WORLD.name} R{COMM_WORLD.rank}, {comm.name} R{comm.rank}: {k} memory cache hit')
                    cache_hit = True
                all_present = comm.allgather(cache_hit)

                # If not present in the cache of all ranks we need to recompute on all ranks
                if not min(all_present):
                    value = CACHE_MISS

            if value is CACHE_MISS:
                value = func(*args, **kwargs)
            return local_cache.setdefault(key, value)

        return wrapper
    return decorator


# A small collection of default simple caches
class DEFAULT_CACHE(dict):
    pass


memory_cache = partial(parallel_cache, cache_factory=lambda: DEFAULT_CACHE())


def disk_only_cache(*args, cachedir=configuration["cache_dir"], **kwargs):
    return parallel_cache(*args, **kwargs, cache_factory=lambda: DictLikeDiskAccess(cachedir))


def memory_and_disk_cache(*args, cachedir=configuration["cache_dir"], **kwargs):
    def decorator(func):
        return memory_cache(*args, **kwargs)(disk_only_cache(*args, cachedir=cachedir, **kwargs)(func))
    return decorator


# ### Some notes from Connor:
#
# def pcache(comm_seeker, key=None, cache_factory=dict):
#
#     comm = comm_seeker()
#     cache = cache_factory()
#
# @pcache(cachetools.LRUCache)
#
# @pcache(DiskCache)
#
# @pcache(MemDiskCache)
#
# @pcache(MemCache)
#
# mem_cache = pcache(cache_factory=cachetools.LRUCache)
# disk_cache = mem_cache(cache_factory=DiskCache)
#
# @pcache(comm_seeker=lambda obj, *_, **_: obj.comm, cache_factory=lambda: cachetools.LRUCache(maxsize=1000))
#
#
# @pmemcache
#
# @pmemdiskcache
#
# class ParallelObject(ABC):
#     @abc.abstractproperty
#     def _comm(self):
#         pass
#
# class MyObj(ParallelObject):
#
#     @pcached_property  # assumes that obj has a "comm" attr
#     @pcached_property(lambda self: self.comm)
#     def myproperty(self):
#         ...
#
#
# def pcached_property():
#     def wrapper(self):
#         assert isinstance(self, ParallelObject)
#         ...
#
#
# from futils.mpi import ParallelObject
#
# from futils.cache import pcached_property
#
# from footils.cache import *
#
# footils == firedrake utils

# * parallel cached property
# * memcache / cache / cached
# * diskonlycache / disk_only_cached
# * memdiskcache / diskcache / disk_cached
# * memcache_no_bcast / broadcast=False
#
# parallel_cached_property = parallel_cache(lambda self: self._comm, key=lambda self: ())
#
# @time
# @timed
# def myslowfunc():
#
#     ..
#
# my_fast_fun = cache(my_slow_fn)
#
####


# TODO:
# Implement an @parallel_cached_property decorator function
