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
from pathlib import Path
from warnings import warn
from functools import wraps

from pyop2.configuration import configuration
from pyop2.logger import debug
from pyop2.mpi import comm_cache_keyval, COMM_WORLD
from pyop2.utils import cached_property


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


class Cached(object):

    """Base class providing global caching of objects. Derived classes need to
    implement classmethods :meth:`_process_args` and :meth:`_cache_key`
    and define a class attribute :attr:`_cache` of type :class:`dict`.

    .. warning::
        The derived class' :meth:`__init__` is still called if the object is
        retrieved from cache. If that is not desired, derived classes can set
        a flag indicating whether the constructor has already been called and
        immediately return from :meth:`__init__` if the flag is set. Otherwise
        the object will be re-initialized even if it was returned from cache!
    """

    def __new__(cls, *args, **kwargs):
        args, kwargs = cls._process_args(*args, **kwargs)
        key = cls._cache_key(*args, **kwargs)

        def make_obj():
            obj = super(Cached, cls).__new__(cls)
            obj._key = key
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
        if key is None:
            return make_obj()
        try:
            return cls._cache_lookup(key)
        except (KeyError, IOError):
            obj = make_obj()
            cls._cache_store(key, obj)
            return obj

    @classmethod
    def _cache_lookup(cls, key):
        return cls._cache[key]

    @classmethod
    def _cache_store(cls, key, val):
        cls._cache[key] = val

    @classmethod
    def _process_args(cls, *args, **kwargs):
        """Pre-processes the arguments before they are being passed to
        :meth:`_cache_key` and the constructor.

        :rtype: *must* return a :class:`list` of *args* and a
            :class:`dict` of *kwargs*"""
        return args, kwargs

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        """Compute the cache key given the preprocessed constructor arguments.

        :rtype: Cache key to use or ``None`` if the object is not to be cached

        .. note:: The cache key must be hashable."""
        return tuple(args) + tuple([(k, v) for k, v in kwargs.items()])

    @cached_property
    def cache_key(self):
        """Cache key."""
        return self._key


class _CacheMiss:
    pass


CACHE_MISS = _CacheMiss()


class _CacheKey:
    def __init__(self, key_value):
        self.value = key_value


class DiskCachedObject:
    def __new__(cls, *args, **kwargs):
        if isinstance(args[0], _CacheKey):
            return super().__new__(cls)
        comm, disk_key = cls._key(*args, **kwargs)
        k = _as_hexdigest((disk_key, cls.__qualname__))
        if comm.rank == 0:
            value = _disk_cache_get(cls._cachedir, k)
            if value is None:
                value = CACHE_MISS
            id_str = f"{COMM_WORLD.name} R{COMM_WORLD.rank}, {comm.name} R{comm.rank}: "
            if value is CACHE_MISS:
                debug(id_str + f'Disk cache miss for {cls.__qualname__}({args}{kwargs})')
            else:
                debug(id_str + f'Disk cache hit for {cls.__qualname__}({args}{kwargs})')
            # TODO: Add communication tags to avoid cross-broadcasting
            comm.bcast(value, root=0)
        else:
            value = comm.bcast(CACHE_MISS, root=0)
            if isinstance(value, _CacheMiss):
                # We might have the CACHE_MISS from rank 0 and
                # `(value is CACHE_MISS) == False` which is confusing,
                # so we set it back to the local value
                value = CACHE_MISS

        if value is CACHE_MISS:
            # We can't call the constructor as `cls(*args, **kwargs)` since we
            # would call `__new__` and recurse infinitely. The solution is to
            # create a new object and pass that to `__init__`
            value = object.__new__(cls)
            value.__init__(*args, **kwargs)
            value._cache_key = _CacheKey(k)
            if comm.rank == 0:
                # Only write to the disk cache on rank 0
                _disk_cache_set(cls._cachedir, k, value)

        debug("Disk cache modifying init")
        cls.__init__ = cls._skip_init(cls.__init__)
        return value

    @classmethod
    def _skip_init(cls, init):
        """This function allows a class to skip it's init method"""
        def restore_init(*args, **kwargs):
            debug("Disk reset init")
            cls.__init__ = init
        return restore_init

    def __init_subclass__(cls, cachedir=None, key=None, **kwargs):
        if cachedir is None or key is None:
            raise TypeError(
                f"A `cache` and a `key` are required to subclass {__class__.__name__}.\n"
                "Try declaring your subclass as follows:\n"
                f"\tclass {cls.__name__}({cls.__bases__[0].__name__}, cachedir=my_cache, key=my_key)"
            )
        super().__init_subclass__(**kwargs)
        cls._cachedir = cachedir
        cls._key = key

    def __getnewargs__(self):
        return (self._cache_key, )


# TODO: Implement this...
# class MemoryCachedObject:
#     def __new__(cls, *args, **kwargs):
#         k = cls._key(*args, **kwargs), cls.__qualname__
#         value = cls._cache.get(k, CACHE_MISS)
#         if value is CACHE_MISS:
#             print(f'Cache miss for {cls.__qualname__}({args}{kwargs})')
#             # We can't call the constructor as `cls(*args, **kwargs)` since we
#             # would call `__new__` and recurse infinitely. The solution is to
#             # create a new object and pass that to `__init__`
#             value = object.__new__(cls)
#             value.__init__(*args, **kwargs)
#             cls._cache[k] = value
#         else:
#             print(f'Cache hit for {cls.__qualname__}({args}{kwargs})')
#         cls.__init__ = _skip_init(cls, cls.__init__)
#         return value
#
#     def __init_subclass__(cls, cache=None, key=None, **kwargs):
#         if cache is None or key is None:
#             raise TypeError(
#                 f"A `cache` and a `key` are required to subclass {__class__.__name__}.\n"
#                 "Try declaring your subclass as follows:\n"
#                 f"\tclass {cls.__name__}({cls.__bases__[0].__name__}, cache=my_cache, key=my_key)"
#             )
#         super().__init_subclass__(**kwargs)
#         cls._cache = cache
#         cls._key = key


class MemoryAndDiskCachedObject(DiskCachedObject, cachedir="", key=""):
    def __new__(cls, *args, **kwargs):
        if isinstance(args[0], _CacheKey):
            return super().__new__(cls, *args)
        comm, disk_key = cls._key(*args, **kwargs)
        # Throw the qualified name into the key as a string so the memory cache
        # can be debugged (a little bit) by a human. This shouldn't really be
        # necessary, but some classes do not implement a proper repr.
        k = _as_hexdigest((disk_key, cls.__qualname__)), cls.__qualname__

        # Fetch the per-comm cache or set it up if not present
        # from pyop2.mpi import COMM_WORLD, comm_cache_keyval
        local_cache = comm.Get_attr(comm_cache_keyval)
        if local_cache is None:
            local_cache = {}
            comm.Set_attr(comm_cache_keyval, local_cache)

        id_str = f"{COMM_WORLD.name} R{COMM_WORLD.rank}, {comm.name} R{comm.rank}: "
        if comm.rank == 0:
            value = local_cache.get(k, CACHE_MISS)
            if value is CACHE_MISS:
                debug(id_str + f'Memory cache miss for {cls.__qualname__}({args}{kwargs})')
            else:
                debug(id_str + f'Memory cache hit for {cls.__qualname__}({args}{kwargs})')
            # TODO: Add communication tags to avoid cross-broadcasting
            comm.bcast(value, root=0)
        else:
            value = comm.bcast(CACHE_MISS, root=0)
            if isinstance(value, _CacheMiss):
                # We might have the CACHE_MISS from rank 0 and
                # `(value is CACHE_MISS) == False` which is confusing,
                # so we set it back to the local value
                value = CACHE_MISS

        if value is CACHE_MISS:
            # TODO: Fix comment
            # We can call the constructor as `cls(*args, **kwargs)` here since we
            # are subclassing `DiskCachedObject` and _want_ to call __new__ in
            # case the object is in the disk cache.
            value = super().__new__(cls, *args, **kwargs)
            # Regardless whether the object was disk cached, init has already
            # been called here
            local_cache[k] = value

        return value

    def __init_subclass__(cls, cachedir=None, key=None, **kwargs):
        if cachedir is None or key is None:
            raise TypeError(
                f"A `cache` and a `key` are required to subclass {__class__.__name__}.\n"
                "Try declaring your subclass as follows:\n"
                f"\tclass {cls.__name__}({cls.__bases__[0].__name__}, cache=my_cache, key=my_key)"
            )
        super().__init_subclass__(cachedir=cachedir, key=key, **kwargs)

    def __getnewargs__(self):
        return (self._cache_key, )


# TODO: Remove class wrapper, this was a bad idea
def disk_cache(cachedir, key):
    def decorator(orig_obj):
        if isinstance(orig_obj, type(lambda: None)):
            # Cached function wrapper
            @wraps(orig_obj)
            def _wrapper(*args, **kwargs):
                comm, disk_key = key(*args, **kwargs)
                k = _as_hexdigest((disk_key, orig_obj.__qualname__))
                if comm.rank == 0:
                    # Only read from disk on rank 0
                    value = _disk_cache_get(cachedir, k)
                    id_str = f"{COMM_WORLD.name} R{COMM_WORLD.rank}, {comm.name} R{comm.rank}: "
                    if value is CACHE_MISS:
                        debug(id_str + f"Disk cache miss for {orig_obj.__qualname__}({args}{kwargs})")
                    else:
                        debug(id_str + f'Disk cache hit for {orig_obj.__qualname__}({args}{kwargs})')
                    comm.bcast(value, root=0)
                else:
                    value = comm.bcast(CACHE_MISS, root=0)

                if value is CACHE_MISS:
                    value = orig_obj(*args, **kwargs)
                    # Only write to the disk cache on rank 0
                    if comm.rank == 0:
                        _disk_cache_set(cachedir, k, value)
                return value
        elif isinstance(orig_obj, type(object)):
            # Cached object wrapper
            @wraps(orig_obj, updated=())
            class _wrapper(orig_obj):
                def __new__(cls, *args, **kwargs):
                    comm, disk_key = key(*args, **kwargs)
                    k = _as_hexdigest((disk_key, orig_obj.__qualname__))
                    if comm.rank == 0:
                        # Only read from disk on rank 0
                        value = _disk_cache_get(cachedir, k)
                        if value is None:
                            value = CACHE_MISS
                        id_str = f"{COMM_WORLD.name} R{COMM_WORLD.rank}, {comm.name} R{comm.rank}: "
                        if value is CACHE_MISS:
                            debug(id_str + f'Disk cache miss for {orig_obj.__qualname__}({args}{kwargs})')
                        else:
                            debug(id_str + f'Disk cache hit for {orig_obj.__qualname__}({args}{kwargs})')
                        comm.bcast(value, root=0)
                    else:
                        value = comm.bcast(CACHE_MISS, root=0)

                    if value is CACHE_MISS:
                        # We can't call the constructor as `orig_obj(*args, **kwargs)`
                        # since we might be subclassing another cached object. The
                        # solution is to create a new object and pass it to `__init__`
                        value = object.__new__(orig_obj)
                        orig_obj.__init__(value, *args, **kwargs)
                        if comm.rank == 0:
                            # Only write to the disk cache on rank 0
                            _disk_cache_set(cachedir, k, value)
                    return value
        else:
            raise ValueError("Unknown object passed to decorator")
        return _wrapper
    return decorator


cached = cachetools.cached
"""Cache decorator for functions. See the cachetools documentation for more
information.

.. note::
    If you intend to use this decorator to cache things that are collective
    across a communicator then you must include the communicator as part of
    the cache key.

    You should also make sure to use unbounded caches as otherwise some ranks
    may evict results leading to deadlocks.
"""


def default_parallel_hashkey(*args, **kwargs):
    comm = kwargs.get('comm')
    return comm, cachetools.keys.hashkey(*args, **kwargs)

#### connor bits

# ~ def pcache(comm_seeker, key=None, cache_factory=dict):

    # ~ comm = comm_seeker()
    # ~ cache = cache_factory()

# ~ @pcache(cachetools.LRUCache)

@pcache(DiskCache)

@pcache(MemDiskCache)

@pcache(MemCache)

mem_cache = pcache(cache_factory=cachetools.LRUCache)
disk_cache = mem_cache(cache_factory=DiskCache)

# ~ @pcache(comm_seeker=lambda obj, *_, **_: obj.comm, cache_factory=lambda: cachetools.LRUCache(maxsize=1000))


# ~ @pmemcache

# ~ @pmemdiskcache

# ~ class ParallelObject(ABC):
    # ~ @abc.abstractproperty
    # ~ def _comm(self):
        # ~ pass

# ~ class MyObj(ParallelObject):

    # ~ @pcached_property  # assumes that obj has a "comm" attr
    # ~ @pcached_property(lambda self: self.comm)
    # ~ def myproperty(self):
        # ~ ...


# ~ def pcached_property():
    # ~ def wrapper(self):
        # ~ assert isinstance(self, ParallelObject)
        # ~ ...


# ~ from futils.mpi import ParallelObject

# ~ from futils.cache import pcached_property

# ~ from footils.cache import *

# footils == firedrake utils

# * parallel cached property
# * memcache / cache / cached
# * diskonlycache / disk_only_cached
# * memdiskcache / diskcache / disk_cached
# * memcache_no_bcast / broadcast=False

# ~ parallel_cached_property = parallel_cache(lambda self: self._comm, key=lambda self: ())

# ~ @time
# ~ @timed
# ~ def myslowfunc():
    # ~ ..

# ~ my_fast_fun = cache(my_slow_fn)

####


# TODO:
# Implement an @parallel_cached_property decorator function



def parallel_memory_only_cache(key=default_parallel_hashkey):
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
        def wrapper(*args, **kwargs):
            """ Extract the key and then try the memory cache before falling back
            on calling the function and populating the cache.
            """
            comm, mem_key = key(*args, **kwargs)
            k = _as_hexdigest(mem_key), func.__qualname__

            # Fetch the per-comm cache or set it up if not present
            local_cache = comm.Get_attr(comm_cache_keyval)
            if local_cache is None:
                local_cache = {}
                comm.Set_attr(comm_cache_keyval, local_cache)

            # Grab value from rank 0 memory cache and broadcast result
            if comm.rank == 0:
                v = local_cache.get(k)
                if v is None:
                    debug(f'{COMM_WORLD.name} R{COMM_WORLD.rank}, {comm.name} R{comm.rank}: {k} memory cache miss')
                else:
                    debug(f'{COMM_WORLD.name} R{COMM_WORLD.rank}, {comm.name} R{comm.rank}: {k} memory cache hit')
                comm.bcast(v, root=0)
            else:
                v = comm.bcast(None, root=0)

            if v is None:
                v = func(*args, **kwargs)
            return local_cache.setdefault(k, v)

        return wrapper
    return decorator


def parallel_memory_only_cache_no_broadcast(key=default_parallel_hashkey):
    """Memory only cache decorator.

    Decorator for wrapping a function to be called over a communiucator in a
    cache that stores non-broadcastable values in memory, for instance function
    pointers. If the value is not present on all ranks, all ranks repeat the
    work.

    :arg key: Callable returning the cache key for the function inputs. This
        function must return a 2-tuple where the first entry is the
        communicator to be collective over and the second is the key. This is
        required to ensure that deadlocks do not occur when using different
        subcommunicators.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            """ Extract the key and then try the memory cache before falling back
            on calling the function and populating the cache.
            """
            comm, mem_key = key(*args, **kwargs)
            k = _as_hexdigest(mem_key), func.__qualname__

            # Fetch the per-comm cache or set it up if not present
            local_cache = comm.Get_attr(comm_cache_keyval)
            if local_cache is None:
                local_cache = {}
                comm.Set_attr(comm_cache_keyval, local_cache)

            # Grab value from all ranks memory cache and vote
            v = local_cache.get(k)
            if v is None:
                debug(f'{COMM_WORLD.name} R{COMM_WORLD.rank}, {comm.name} R{comm.rank}: {k} memory cache miss')
            else:
                debug(f'{COMM_WORLD.name} R{COMM_WORLD.rank}, {comm.name} R{comm.rank}: {k} memory cache hit')
            all_present = comm.allgather(bool(v))

            # If not present in the cache of all ranks, recompute on all ranks
            if not min(all_present):
                v = func(*args, **kwargs)
            return local_cache.setdefault(k, v)

        return wrapper
    return decorator


# TODO: Change call signature
def disk_cached(cache, cachedir=None, key=cachetools.keys.hashkey, collective=False):
    """Decorator for wrapping a function in a cache that stores values in memory and to disk.

    :arg cache: The in-memory cache, usually a :class:`dict`.
    :arg cachedir: The location of the cache directory. Defaults to ``PYOP2_CACHE_DIR``.
    :arg key: Callable returning the cache key for the function inputs. If ``collective``
        is ``True`` then this function must return a 2-tuple where the first entry is the
        communicator to be collective over and the second is the key. This is required to ensure
        that deadlocks do not occur when using different subcommunicators.
    :arg collective: If ``True`` then cache lookup is done collectively over a communicator.
    """
    if cachedir is None:
        cachedir = configuration["cache_dir"]

    if collective and cache is not None:
        warn(
            "Global cache for collective disk cached call will not be used. "
            "Pass `None` as the first argument"
        )

    def decorator(func):
        if not collective:
            def wrapper(*args, **kwargs):
                """ Extract the key and then try the memory then disk cache
                before falling back on calling the function and populating the
                caches.
                """
                k = _as_hexdigest(key(*args, **kwargs))
                try:
                    v = cache[k]
                    debug(f'Serial: {k} memory cache hit')
                except KeyError:
                    debug(f'Serial: {k} memory cache miss')
                    v = _disk_cache_get(cachedir, k)
                    if v is not None:
                        debug(f'Serial: {k} disk cache hit')

                if v is None:
                    debug(f'Serial: {k} disk cache miss')
                    v = func(*args, **kwargs)
                    _disk_cache_set(cachedir, k, v)
                return cache.setdefault(k, v)

        else:  # Collective
            @parallel_memory_only_cache(key=key)
            def wrapper(*args, **kwargs):
                """ Same as above, but in parallel over `comm`
                """
                comm, disk_key = key(*args, **kwargs)
                k = _as_hexdigest(disk_key)

                # Grab value from rank 0 disk cache and broadcast result
                if comm.rank == 0:
                    v = _disk_cache_get(cachedir, k)
                    if v is not None:
                        debug(f'{COMM_WORLD.name} R{COMM_WORLD.rank}, {comm.name} R{comm.rank}: {k} disk cache hit')
                    else:
                        debug(f'{COMM_WORLD.name} R{COMM_WORLD.rank}, {comm.name} R{comm.rank}: {k} disk cache miss')
                    comm.bcast(v, root=0)
                else:
                    v = comm.bcast(None, root=0)

                if v is None:
                    v = func(*args, **kwargs)
                    # Only write to the disk cache on rank 0
                    if comm.rank == 0:
                        _disk_cache_set(cachedir, k, v)
                return v

        return wrapper
    return decorator


def _as_hexdigest(key):
    return hashlib.md5(str(key).encode()).hexdigest()


def clear_memory_cache(comm):
    if comm.Get_attr(comm_cache_keyval) is not None:
        comm.Set_attr(comm_cache_keyval, {})


def _disk_cache_get(cachedir, key):
    """Retrieve a value from the disk cache.

    :arg cachedir: The cache directory.
    :arg key: The cache key (must be a string).
    :returns: The cached object if found, else ``None``.
    """
    filepath = Path(cachedir, key[:2], key[2:])
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def _disk_cache_set(cachedir, key, value):
    """Store a new value in the disk cache.

    :arg cachedir: The cache directory.
    :arg key: The cache key (must be a string).
    :arg value: The new item to store in the cache.
    """
    k1, k2 = key[:2], key[2:]
    basedir = Path(cachedir, k1)
    basedir.mkdir(parents=True, exist_ok=True)

    tempfile = basedir.joinpath(f"{k2}_p{os.getpid()}.tmp")
    filepath = basedir.joinpath(k2)
    with open(tempfile, "wb") as f:
        pickle.dump(value, f)
    tempfile.rename(filepath)
