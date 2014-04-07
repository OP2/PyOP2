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

import cPickle
import gzip
import os


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
        if isinstance(x, (typ, )):
            typs[type(x)] += 1
            n += 1
    if n == 0:
        print "\nNo %s objects in caches" % typ.__name__
        return
    print "\n%d %s objects in caches" % (n, typ.__name__)
    print "Object breakdown"
    print "================"
    for k, v in typs.iteritems():
        mod = getmodule(k)
        if mod is not None:
            name = "%s.%s" % (mod.__name__, k.__name__)
        else:
            name = k.__name__
        print '%s: %d' % (name, v)


class ObjectCached(object):
    """Base class for objects that should be cached on another object.

    Derived classes need to implement classmethods
    :meth:`_process_args` and :meth:`_cache_key` (which see for more
    details).  The object on which the cache is stored should contain
    a dict in its ``_cache`` attribute.

    .. warning ::

       This kind of cache sets up a circular reference.  If either of
       the objects implements ``__del__``, the Python garbage
       collector will not be able to collect this cycle, and hence
       the cache will never be evicted.

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
            obj = super(ObjectCached, cls).__new__(cls)
            obj._initialized = False
            # obj.__init__ will be called twice when constructing
            # something not in the cache.  The first time here, with
            # the canonicalised args, the second time directly in the
            # subclass.  But that one should hit the cache and return
            # straight away.
            obj.__init__(*args, **kwargs)
            if key is not None:
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
        try:
            return cls._cache_lookup(key)
        except KeyError:
            obj = super(Cached, cls).__new__(cls)
            obj._key = key
            obj._initialized = False
            # obj.__init__ will be called twice when constructing
            # something not in the cache.  The first time here, with
            # the canonicalised args, the second time directly in the
            # subclass.  But that one should hit the cache and return
            # straight away.
            obj.__init__(*args, **kwargs)
            # If key is None we're not supposed to store the object in cache
            if key:
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

    @property
    def cache_key(self):
        """Cache key."""
        return self._key


class DiskCached(Cached):

    """Base class providing global caching of objects on disk. The same notes
    as in :class:`Cached` apply. In addition, derived classes need to
    define a class attribute :attr:`_cachedir` specifying the path where to
    cache objects on disk.

    .. warning ::
        The key returned by :meth:`_cache_key` *must* be a
        :class:`str` safe to use as a filename, such as an md5 hex digest.
    """

    @classmethod
    def _cache_lookup(cls, key):
        return cls._cache.get(key) or cls._read_from_disk(key)

    @classmethod
    def _read_from_disk(cls, key):
        filepath = os.path.join(cls._cachedir, key)
        if os.path.exists(filepath):
            f = gzip.open(filepath, "rb")
            val = cPickle.load(f)
            f.close()
            # Store in memory so we can save ourselves a disk lookup next time
            cls._cache[key] = val
            return val
        raise KeyError("Object with key %s not found in %s" % (key, filepath))

    @classmethod
    def _cache_store(cls, key, val):
        cls._cache[key] = val
        f = gzip.open(os.path.join(cls._cachedir, key), "wb")
        cPickle.dump(val, f)
        f.close()
