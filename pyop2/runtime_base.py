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

""" Base classes for OP2 objects. The versions here extend those from the :mod:`base` module to include runtime data information which is backend independent. Individual runtime backends should subclass these as required to implement backend-specific features."""

import numpy as np

from exceptions import *
from utils import *
import base
from base import READ, WRITE, RW, INC, MIN, MAX, IterationSpace
from base import DataCarrier, IterationIndex, i, IdentityMap, Kernel
from base import _parloop_cache, _empty_parloop_cache, _parloop_cache_size
import op_lib_core as core
from pyop2.utils import OP2_INC, OP2_LIB
from la_petsc import PETSc

# Data API

class Arg(base.Arg):
    """An argument to a :func:`par_loop`.

    .. warning:: User code should not directly instantiate :class:`Arg`. Instead, use the call syntax on the :class:`DataCarrier`.
    """

    @property
    def _c_handle(self):
        if self._lib_handle is None:
            self._lib_handle = core.op_arg(self)
        return self._lib_handle

class Set(base.Set):
    """OP2 set."""

    @validate_type(('size', int, SizeTypeError))
    def __init__(self, size, name=None):
        base.Set.__init__(self, size, name)

    @classmethod
    def fromhdf5(cls, f, name):
        slot = f[name]
        size = slot.value.astype(np.int)
        shape = slot.shape
        if shape != (1,):
            raise SizeTypeError("Shape of %s is incorrect" % name)
        return cls(size[0], name)

    def __call__(self, *dims):
        return IterationSpace(self, dims)

    @property
    def _c_handle(self):
        if self._lib_handle is None:
            self._lib_handle = core.op_set(self)
        return self._lib_handle

class Dat(base.Dat):
    """OP2 vector data. A ``Dat`` holds a value for every member of a :class:`Set`."""

    _arg_type = Arg

    @classmethod
    def fromhdf5(cls, dataset, f, name):
        slot = f[name]
        data = slot.value
        dim = slot.shape[1:]
        soa = slot.attrs['type'].find(':soa') > 0
        if len(dim) < 1:
            raise DimTypeError("Invalid dimension value %s" % dim)
        ret = cls(dataset, dim, data, name=name, soa=soa)
        return ret

    @property
    def _c_handle(self):
        if self._lib_handle is None:
            self._lib_handle = core.op_dat(self)
        return self._lib_handle

class Const(base.Const):
    """Data that is constant for any element of any set."""

    @classmethod
    def fromhdf5(cls, f, name):
        slot = f[name]
        dim = slot.shape
        data = slot.value
        if len(dim) < 1:
            raise DimTypeError("Invalid dimension value %s" % dim)
        return cls(dim, data, name)

class Global(base.Global):
    """OP2 Global object."""
    _arg_type = Arg

class Map(base.Map):
    """OP2 map, a relation between two :class:`Set` objects."""

    _arg_type = Arg

    @property
    def _c_handle(self):
        if self._lib_handle is None:
            self._lib_handle = core.op_map(self)
        return self._lib_handle

    @classmethod
    def fromhdf5(cls, iterset, dataset, f, name):
        slot = f[name]
        values = slot.value
        dim = slot.shape[1:]
        if len(dim) != 1:
            raise DimTypeError("Unrecognised dimension value %s" % dim)
        return cls(iterset, dataset, dim[0], values, name)

_sparsity_cache = dict()
def _empty_sparsity_cache():
    _sparsity_cache.clear()

class Sparsity(base.Sparsity):
    """OP2 Sparsity, a matrix structure derived from the union of the outer product of pairs of :class:`Map` objects."""

    def __init__(self, *args, **kwargs):
        super(Sparsity, self).__init__(*args, **kwargs)
        self._build_sparsity_pattern()

    @property
    def _c_handle(self):
        if self._lib_handle is None:
            key = (self._rmaps, self._cmaps, self._dims)
            self._lib_handle = _sparsity_cache.get(key) or core.op_sparsity(self)
            _sparsity_cache[key] = self._lib_handle
        return self._lib_handle

    @one_time
    def _build_sparsity_pattern(self):
        rmult, cmult = self._dims
        s_diag  = [ set() for i in xrange(self._nrows) ]
        s_odiag = [ set() for i in xrange(self._nrows) ]

        lsize = self._nrows
        for rowmap, colmap in zip(self._rmaps, self._cmaps):
            #FIXME: exec_size will need adding for MPI support
            rsize = rowmap.iterset.size
            for e in xrange(rsize):
                for i in xrange(rowmap.dim):
                    for r in xrange(rmult):
                        row = rmult * rowmap.values[e][i] + r
                        if row < lsize:
                            for c in xrange(cmult):
                                for d in xrange(colmap.dim):
                                    entry = cmult * colmap.values[e][d] + c
                                    if entry < lsize:
                                        s_diag[row].add(entry)
                                    else:
                                        s_odiag[row].add(entry)

        d_nnz = [0]*(cmult * self._nrows)
        o_nnz = [0]*(cmult * self._nrows)
        rowptr = [0]*(self._nrows+1)
        for row in xrange(self._nrows):
            d_nnz[row] = len(s_diag[row])
            o_nnz[row] = len(s_odiag[row])
            rowptr[row+1] = rowptr[row] + d_nnz[row] + o_nnz[row]
        colidx = [0]*rowptr[self._nrows]
        for row in xrange(self._nrows):
            entries = list(s_diag[row]) + list(s_odiag[row])
            colidx[rowptr[row]:rowptr[row+1]] = entries

        self._total_nz = rowptr[self._nrows]
        self._rowptr = rowptr
        self._colidx = colidx
        self._d_nnz = d_nnz

    @property
    def rowptr(self):
        return self._rowptr

    @property
    def colidx(self):
        return self._colidx

    @property
    def d_nnz(self):
        return self._d_nnz

class Mat(base.Mat):
    """OP2 matrix data. A Mat is defined on a sparsity pattern and holds a value
    for each element in the :class:`Sparsity`."""

    _arg_type = Arg

    def __init__(self, *args, **kwargs):
        super(Mat, self).__init__(*args, **kwargs)
        self._handle = None

    def _init(self):
        mat = PETSc.Mat()
        mat.create()
        mat.setType(PETSc.Mat.Type.SEQAIJ)
        mat.setSizes([self.sparsity.nrows, self.sparsity._ncols])
        mat.setPreallocationCSR((self.sparsity.rowptr, self.sparsity.colidx, None))
        self._handle = mat

    def zero(self):
        """Zero the matrix."""
        self._c_handle.zero()

    def zero_rows(self, rows, diag_val):
        """Zeroes the specified rows of the matrix, with the exception of the
        diagonal entry, which is set to diag_val. May be used for applying
        strong boundary conditions."""
        self._c_handle.zero_rows(rows, diag_val)

    def _assemble(self):
        self._c_handle.assemble()

    @property
    def values(self):
        return self._c_handle.values

    @property
    def handle(self):
        if self._handle is None:
            self._init()
        return self._handle

class ParLoop(base.ParLoop):
    def compute(self):
        raise RuntimeError('Must select a backend')

class Solver(base.Solver):

    def __init__(self):
        super(Solver, self).__init__()
