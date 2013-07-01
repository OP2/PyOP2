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

"""
Cython implementation of the Plan construction.
"""

import base
from utils import align
import math
import itertools
import numpy
cimport numpy
from libc.stdlib cimport malloc, free
try:
    from collections import OrderedDict
# OrderedDict was added in Python 2.7. Earlier versions can use ordereddict
# from PyPI
except ImportError:
    from ordereddict import OrderedDict

# C type declarations
ctypedef struct map_idx_t:
    # pointer to the raw numpy array containing the map values
    int * map_base
    # dimension of the map
    int dim
    int idx

ctypedef struct flat_race_args_t:
    # Dat size
    int size
    # Temporary array for coloring purpose
    unsigned int* tmp
    # lenght of mip (ie, number of occurences of Dat in the access descriptors)
    int count
    map_idx_t * mip

cdef class Plan:
    """Plan object contains necessary information for data staging and execution scheduling."""

    # NOTE:
    #  - do not rename fields: _nelems, _ind_map, etc in order to get ride of the boilerplate
    # property definitions, these are necessary to allow CUDA and OpenCL to override them without
    # breaking this code

    cdef numpy.ndarray _nelems
    cdef numpy.ndarray _ind_map
    cdef numpy.ndarray _loc_map
    cdef numpy.ndarray _ind_sizes
    cdef numpy.ndarray _nindirect
    cdef numpy.ndarray _ind_offs
    cdef numpy.ndarray _offset
    cdef numpy.ndarray _thrcol
    cdef numpy.ndarray _nthrcol
    cdef numpy.ndarray _ncolblk
    cdef numpy.ndarray _blkmap
    cdef int _nblocks
    cdef int _nargs
    cdef int _ninds
    cdef int _nshared
    cdef int _ncolors

    def __cinit__(self, kernel, iset, *args, **kwargs):
        ps = kwargs.get('partition_size', 1)
        mc = kwargs.get('matrix_coloring', False)
        st = kwargs.get('staging', True)
        tc = kwargs.get('thread_coloring', True)

        assert ps > 0, "partition size must be strictly positive"

        self._need_exec_halo = any(arg._is_indirect_and_not_read or arg._is_mat
                                   for arg in args)

        self._compute_partition_info(iset, ps, mc, args)
        if st:
            self._compute_staging_info(iset, ps, mc, args)

        self._compute_coloring(iset, ps, mc, tc, args)

    def _compute_partition_info(self, iset, ps, mc, args):
        def nelems_iter(len):
            if len:
                acc = 0
                while acc < len:
                    yield min(ps, len - acc)
                    acc += ps

        _gen = (nelems_iter(iset._core_size),
                nelems_iter(iset.size - iset._core_size), )

        if self._need_exec_halo:
            _gen += (nelems_iter(iset._ieh_size - iset._size), )

        self._nelems = numpy.fromiter(itertools.chain(*_gen),
                                      dtype=numpy.int32)

        self._nblocks = self._nelems.size

        def offset_iter():
            _offset = 0
            for nelems in self._nelems:
                yield _offset
                _offset += nelems

        self._offset = numpy.fromiter(offset_iter(), dtype=numpy.int32)

    def _compute_staging_info(self, iset, ps, mc, args):
        """Constructs:
            - nindirect
            - ind_map
            - loc_map
            - ind_sizes
            - ind_offs
            - offset
            - nshared
        """
        # indices referenced for this dat-map pair
        def indices(dat, map):
            return [arg.idx for arg in args if arg.data == dat and arg.map == map]

        self._ninds = 0
        self._nargs = len([arg for arg in args if not arg._is_mat])
        d = OrderedDict()
        for i, arg in enumerate([arg for arg in args if not arg._is_mat]):
            if arg._is_indirect:
                k = (arg.data,arg.map)
                if not d.has_key(k):
                    d[k] = i
                    self._ninds += 1

        inds = dict()
        locs = dict()
        sizes = dict()

        for pi in range(self._nblocks):
            start = self._offset[pi]
            end = start + self._nelems[pi]

            for dat,map in d.iterkeys():
                ii = indices(dat,map)
                l = len(ii)

                inds[(dat,map,pi)], inv = numpy.unique(map.values[start:end,ii], return_inverse=True)
                sizes[(dat,map,pi)] = len(inds[(dat,map,pi)])

                for i, ind in enumerate(sorted(ii)):
                    locs[(dat,map,ind,pi)] = inv[i::l]

        def ind_iter():
            for dat,map in d.iterkeys():
                cumsum = 0
                for pi in range(self._nblocks):
                    cumsum += len(inds[(dat,map,pi)])
                    yield inds[(dat,map,pi)]
                # creates a padding to conform with op2 plan objects
                # fills with -1 for debugging
                # this should be removed and generated code changed
                # once we switch to python plan only
                pad = numpy.empty(len(indices(dat,map)) * iset.size - cumsum, dtype=numpy.int32)
                pad.fill(-1)
                yield pad
        t = tuple(ind_iter())
        self._ind_map = numpy.concatenate(t) if t else numpy.array([], dtype=numpy.int32)

        def size_iter():
            for pi in range(self._nblocks):
                for dat,map in d.iterkeys():
                    yield sizes[(dat,map,pi)]
        self._ind_sizes = numpy.fromiter(size_iter(), dtype=numpy.int32)

        def nindirect_iter():
            for dat,map in d.iterkeys():
                yield sum(sizes[(dat,map,pi)] for pi in range(self._nblocks))
        self._nindirect = numpy.fromiter(nindirect_iter(), dtype=numpy.int32)

        def loc_iter():
            for dat,map in d.iterkeys():
                for i in indices(dat, map):
                    for pi in range(self._nblocks):
                        yield locs[(dat,map,i,pi)].astype(numpy.int16)
        t = tuple(loc_iter())
        self._loc_map = numpy.concatenate(t) if t else numpy.array([], dtype=numpy.int32)

        def off_iter():
            _off = dict()
            for dat,map in d.iterkeys():
                _off[(dat,map)] = 0
            for pi in range(self._nblocks):
                for dat,map in d.iterkeys():
                    yield _off[(dat,map)]
                    _off[(dat,map)] += sizes[(dat,map,pi)]
        self._ind_offs = numpy.fromiter(off_iter(), dtype=numpy.int32)

        # max shared memory required by work groups
        nshareds = [0] * self._nblocks
        for pi in range(self._nblocks):
            for k in d.iterkeys():
                dat, map = k
                nshareds[pi] += align(sizes[(dat,map,pi)] * dat.dtype.itemsize * dat.cdim)
        self._nshared = max(nshareds)

    def _compute_coloring(self, iset, ps, mc, tc, args):
        """Constructs:
            - thrcol
            - nthrcol
            - ncolors
            - blkmap
            - ncolblk
        """
        # args requiring coloring (ie, indirect reduction and matrix args)
        #  key: Dat
        #  value: [(map, idx)] (sorted as they appear in the access descriptors)
        race_args = OrderedDict()
        for arg in args:
            if arg._is_indirect_reduction:
                k = arg.data
                l = race_args.get(k, [])
                l.append((arg.map, arg.idx))
                race_args[k] = l
            elif mc and arg._is_mat:
                k = arg.data
                rowmap = k.sparsity.maps[0][0]
                l = race_args.get(k, [])
                for i in range(rowmap.dim):
                    l.append((rowmap, i))
                race_args[k] = l

        # convert 'OrderedDict race_args' into a flat array for performant access in cython
        cdef int n_race_args = len(race_args)
        cdef flat_race_args_t* flat_race_args = <flat_race_args_t*> malloc(n_race_args * sizeof(flat_race_args_t))
        pcds = [None] * n_race_args
        for i, ra in enumerate(race_args.iterkeys()):
            if isinstance(ra, base.Dat):
                s = ra.dataset.exec_size if self._need_exec_halo else ra.dataset.size
            elif isinstance(ra, base.Mat):
                s = ra.sparsity.maps[0][0].dataset.exec_size if self._need_exec_halo else ra.sparsity.maps[0][0].dataset.size

            pcds[i] = numpy.empty((s,), dtype=numpy.uint32)
            flat_race_args[i].size = s
            flat_race_args[i].tmp = <unsigned int *> numpy.PyArray_DATA(pcds[i])

            flat_race_args[i].count = len(race_args[ra])
            flat_race_args[i].mip = <map_idx_t*> malloc(flat_race_args[i].count * sizeof(map_idx_t))
            for j, mi in enumerate(race_args[ra]):
                map, idx = mi
                flat_race_args[i].mip[j].map_base = <int *> numpy.PyArray_DATA(map.values)
                flat_race_args[i].mip[j].dim = map.dim
                flat_race_args[i].mip[j].idx = idx

        # type constraining a few variables
        cdef int _p
        cdef unsigned int _base_color
        cdef int _t
        cdef unsigned int _mask
        cdef unsigned int _color
        cdef int _rai
        cdef int _mi
        cdef int _i

        # intra partition coloring
        self._thrcol = numpy.empty((iset.exec_size if self._need_exec_halo else iset.size, ), dtype=numpy.int32)
        self._thrcol.fill(-1)

        # create direct reference to numpy array storage
        cdef int * thrcol = <int *> numpy.PyArray_DATA(self._thrcol)
        cdef int * nelems = <int *> numpy.PyArray_DATA(self._nelems)
        cdef int * offset = <int *> numpy.PyArray_DATA(self._offset)

        if tc:
            for _p in range(self._nblocks):
                _base_color = 0
                terminated = False
                while not terminated:
                    terminated = True

                    # zero out working array:
                    for _rai in range(n_race_args):
                        for _i in range(flat_race_args[_rai].size):
                            flat_race_args[_rai].tmp[_i] = 0

                    # color threads
                    for _t in range(offset[_p], offset[_p] + nelems[_p]):
                        if thrcol[_t] == -1:
                            _mask = 0

                            for _rai in range(n_race_args):
                                for _mi in range(flat_race_args[_rai].count):
                                    _mask |= flat_race_args[_rai].tmp[flat_race_args[_rai].mip[_mi].map_base[_t * flat_race_args[_rai].mip[_mi].dim + flat_race_args[_rai].mip[_mi].idx]]

                            if _mask == 0xffffffffu:
                                terminated = False
                            else:
                                _color = 0
                                while _mask & 0x1:
                                    _mask = _mask >> 1
                                    _color += 1
                                thrcol[_t] = _base_color + _color
                                _mask = 1 << _color
                                for _rai in range(n_race_args):
                                    for _mi in range(flat_race_args[_rai].count):
                                        flat_race_args[_rai].tmp[flat_race_args[_rai].mip[_mi].map_base[_t * flat_race_args[_rai].mip[_mi].dim + flat_race_args[_rai].mip[_mi].idx]] |= _mask

                    _base_color += 32

            self._nthrcol = numpy.zeros(self._nblocks,dtype=numpy.int32)
            for _p in range(self._nblocks):
                self._nthrcol[_p] = max(self._thrcol[offset[_p]:(offset[_p] + nelems[_p])]) + 1

        # partition coloring
        pcolors = numpy.empty(self._nblocks, dtype=numpy.int32)
        pcolors.fill(-1)

        cdef int * _pcolors = <int *> numpy.PyArray_DATA(pcolors)

        cdef int _first_block_owned = int(math.ceil(iset._core_size / float(ps)))
        cdef int _first_block_halo = int(math.ceil(iset.size / float(ps)))

        _base_color = 0
        terminated = False
        while not terminated:
            terminated = True

            # zero out working array:
            for _rai in range(n_race_args):
                for _i in range(flat_race_args[_rai].size):
                    flat_race_args[_rai].tmp[_i] = 0

            for _p in range(self._nblocks):
                if _pcolors[_p] == -1:
                    if not terminated:
                        if _p in [_first_block_owned, _first_block_halo]:
                            # break early to finish coloring of the preivous group (core < owned < halo exec)
                            break
                    else:
                        if _p == _first_block_owned:
                            self._ncolors_core = pcolors.max() + 1
                            #self._ncolors_core = max(1, pcolors.max() + 1)
                            _base_color = self._ncolors_core

                        if _p == _first_block_halo:
                            _base_color = pcolors.max() + 1
                            self._ncolors_owned = _base_color + 1

                    _mask = 0
                    for _t in range(offset[_p], offset[_p] + nelems[_p]):
                        for _rai in range(n_race_args):
                            for _mi in range(flat_race_args[_rai].count):
                                _mask |= flat_race_args[_rai].tmp[flat_race_args[_rai].mip[_mi].map_base[_t * flat_race_args[_rai].mip[_mi].dim + flat_race_args[_rai].mip[_mi].idx]]

                    if _mask == 0xffffffffu:
                        terminated = False
                    else:
                        _color = 0
                        while _mask & 0x1:
                            _mask = _mask >> 1
                            _color += 1
                        _pcolors[_p] = _base_color + _color

                        _mask = 1 << _color
                        for _t in range(offset[_p], offset[_p] + nelems[_p]):
                            for _rai in range(n_race_args):
                                for _mi in range(flat_race_args[_rai].count):
                                    flat_race_args[_rai].tmp[flat_race_args[_rai].mip[_mi].map_base[_t * flat_race_args[_rai].mip[_mi].dim + flat_race_args[_rai].mip[_mi].idx]] |= _mask

            _base_color += 32

        # memory free
        for i in range(n_race_args):
            free(flat_race_args[i].mip)
        free(flat_race_args)

        self._ncolors = max(pcolors) + 1
        self._ncolblk = numpy.bincount(pcolors).astype(numpy.int32)
        self._blkmap = numpy.argsort(pcolors, kind='mergesort').astype(numpy.int32)

    @property
    def nargs(self):
        return self._nargs

    @property
    def ninds(self):
        return self._ninds

    @property
    def nshared(self):
        return self._nshared

    @property
    def nblocks(self):
        return self._nblocks

    @property
    def ncolblk(self):
        return self._ncolblk

    @property
    def nindirect(self):
        return self._nindirect

    @property
    def ind_map(self):
        return self._ind_map

    @property
    def ind_sizes(self):
        return self._ind_sizes

    @property
    def ind_offs(self):
        return self._ind_offs

    @property
    def loc_map(self):
        return self._loc_map

    @property
    def blkmap(self):
        return self._blkmap

    @property
    def offset(self):
        return self._offset

    @property
    def nelems(self):
        return self._nelems

    @property
    def nthrcol(self):
        return self._nthrcol

    @property
    def thrcol(self):
        return self._thrcol

    @property
    def ncolors_core(self):
        try:
            return self._ncolors_core
        except AttributeError:
            return self._ncolors

    @property
    def ncolors_owned(self):
        try:
            return self._ncolors_owned
        except AttributeError:
            return self._ncolors

    @property
    def ncolors(self):
        return self._ncolors

    #dummy values for now, to make it run with the cuda backend
    @property
    def nsharedCol(self):
        return numpy.array([self._nshared] * self._ncolors, dtype=numpy.int32)
