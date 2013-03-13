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
import configuration as cfg
import base
import copy
from base import READ, WRITE, RW, INC, MIN, MAX, IterationSpace
from base import DataCarrier, IterationIndex, i, IdentityMap, Kernel, Global
from base import _parloop_cache, _empty_parloop_cache, _parloop_cache_size
import op_lib_core as core
from mpi4py import MPI
from petsc4py import PETSc

PYOP2_COMM = None

def set_mpi_communicator(comm):
    """Set the MPI communicator for parallel communication."""
    global PYOP2_COMM
    if comm is None:
        PYOP2_COMM = MPI.COMM_WORLD
    elif type(comm) is int:
        # If it's come from Fluidity where an MPI_Comm is just an
        # integer.
        PYOP2_COMM = MPI.Comm.f2py(comm)
    else:
        PYOP2_COMM = comm
    # PETSc objects also need to be built on the same communicator.
    PETSc.Sys.setDefaultComm(PYOP2_COMM)

# Data API

class Arg(base.Arg):
    """An argument to a :func:`par_loop`.

    .. warning:: User code should not directly instantiate :class:`Arg`. Instead, use the call syntax on the :class:`DataCarrier`.
    """

    def halo_exchange_begin(self):
        assert self._is_dat, "Doing halo exchanges only makes sense for Dats"
        assert not self._in_flight, \
            "Halo exchange already in flight for Arg %s" % self
        if self.access in [READ, RW] and self.data.needs_halo_update:
            self.data.needs_halo_update = False
            self._in_flight = True
            self.data.halo_exchange_begin()

    def halo_exchange_end(self):
        assert self._is_dat, "Doing halo exchanges only makes sense for Dats"
        if self.access in [READ, RW] and self._in_flight:
            self._in_flight = False
            self.data.halo_exchange_end()

    def reduction_begin(self):
        assert self._is_global, \
            "Doing global reduction only makes sense for Globals"
        assert not self._in_flight, \
            "Reduction already in flight for Arg %s" % self
        if self.access is not READ:
            self._in_flight = True
            if self.access is INC:
                op = MPI.SUM
            elif self.access is MIN:
                op = MPI.MIN
            elif self.access is MAX:
                op = MPI.MAX
            # If the MPI supports MPI-3, this could be MPI_Iallreduce
            # instead, to allow overlapping comp and comms.
            # We must reduce into a temporary buffer so that when
            # executing over the halo region, which occurs after we've
            # called this reduction, we don't subsequently overwrite
            # the result.
            PYOP2_COMM.Allreduce(self.data._data, self.data._buf, op=op)

    def reduction_end(self):
        assert self._is_global, \
            "Doing global reduction only makes sense for Globals"
        if self.access is not READ and self._in_flight:
            self._in_flight = False
            # Must have a copy here, because otherwise we just grab a
            # pointer.
            self.data._data = np.copy(self.data._buf)

    @property
    def _c_handle(self):
        if self._lib_handle is None:
            self._lib_handle = core.op_arg(self)
        return self._lib_handle

class Set(base.Set):
    """OP2 set."""

    @validate_type(('size', (int, tuple, list), SizeTypeError))
    def __init__(self, size, name=None, halo=None):
        self._set_list = []
        base.Set.__init__(self, size, name, halo)

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

class MultiSet(base.Set):
    """OP2 multi-set."""

    ##TODO: Maybe a nest would need to be used here
    def __init__(self, setlist):
        self._set_list = setlist


class Halo(base.Halo):
    def __init__(self, sends, receives, comm=PYOP2_COMM, gnn2unn=None):
        base.Halo.__init__(self, sends, receives, gnn2unn)
        if type(comm) is int:
            self._comm = MPI.Comm.f2py(comm)
        else:
            self._comm = comm
        # FIXME: is this a necessity?
        assert self._comm == PYOP2_COMM, "Halo communicator not PYOP2_COMM"
        rank = self._comm.rank
        size = self._comm.size

        assert len(self._sends) == size, \
            "Invalid number of sends for Halo, got %d, wanted %d" % \
            (len(self._sends), size)
        assert len(self._receives) == size, \
            "Invalid number of receives for Halo, got %d, wanted %d" % \
            (len(self._receives), size)

        assert self._sends[rank].size == 0, \
            "Halo was specified with self-sends on rank %d" % rank
        assert self._receives[rank].size == 0, \
            "Halo was specified with self-receives on rank %d" % rank

    @property
    def comm(self):
        """The MPI communicator this :class:`Halo`'s communications
    should take place over"""
        return self._comm

    def verify(self, s):
        """Verify that this :class:`Halo` is valid for a given
:class:`Set`."""
        for dest, sends in enumerate(self.sends):
            assert (sends >= 0).all() and (sends < s.size).all(), \
                "Halo send to %d is invalid (outside owned elements)" % dest

        for source, receives in enumerate(self.receives):
            assert (receives >= s.size).all() and \
                (receives < s.total_size).all(), \
                "Halo receive from %d is invalid (not in halo elements)" % \
                source

class Dat(base.Dat):
    """OP2 vector data. A ``Dat`` holds a value for every member of a :class:`Set`."""

    def __init__(self, dataset, dim, data=None, dtype=None, name=None,
                 soa=None, uid=None):

        base.Dat.__init__(self, dataset, dim, data, dtype, name, soa, uid)
        if isinstance(dataset, Set):
          halo = dataset.halo
          if halo is not None:
            self._send_reqs = [None]*halo.comm.size
            self._send_buf = [None]*halo.comm.size
            self._recv_reqs = [None]*halo.comm.size
            self._recv_buf = [None]*halo.comm.size

    def __iadd__(self, other):
        """Pointwise addition of fields."""
        self._data += as_type(other.data, self.dtype)
        return self

    def __isub__(self, other):
        """Pointwise subtraction of fields."""
        self._data -= as_type(other.data, self.dtype)
        return self

    def __imul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        if np.isscalar(other):
            self._data *= as_type(other, self.dtype)
        else:
            self._data *= as_type(other.data, self.dtype)
        return self

    def __idiv__(self, other):
        """Pointwise division or scaling of fields."""
        if np.isscalar(other):
            self._data /= as_type(other, self.dtype)
        else:
            self._data /= as_type(other.data, self.dtype)
        return self

    def halo_exchange_begin(self):
        halo = self.dataset.halo
        if halo is None:
            return
        for dest,ele in enumerate(halo.sends):
            if ele.size == 0:
                # Don't send to self (we've asserted that ele.size ==
                # 0 previously) or if there are no elements to send
                self._send_reqs[dest] = MPI.REQUEST_NULL
                continue
            self._send_buf[dest] = self._data[ele]
            self._send_reqs[dest] = halo.comm.Isend(self._send_buf[dest],
                                                    dest=dest, tag=self._id)
        for source,ele in enumerate(halo.receives):
            if ele.size == 0:
                # Don't receive from self or if there are no elements
                # to receive
                self._recv_reqs[source] = MPI.REQUEST_NULL
                continue
            self._recv_buf[source] = self._data[ele]
            self._recv_reqs[source] = halo.comm.Irecv(self._recv_buf[source],
                                                      source=source, tag=self._id)

    def halo_exchange_end(self):
        halo = self.dataset.halo
        if halo is None:
            return
        MPI.Request.Waitall(self._recv_reqs)
        MPI.Request.Waitall(self._send_reqs)
        self._send_buf = [None]*len(self._send_buf)
        for source, buf in enumerate(self._recv_buf):
            if buf is not None:
                self._data[halo.receives[source]] = buf
        self._recv_buf = [None]*len(self._recv_buf)

    @property
    def norm(self):
        """The L2-norm on the flattened vector."""
        return np.linalg.norm(self._data)

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

class MultiDat(base.Dat):
     def __init__(self, dat_list, name=None):
        self.mixed_name = name
        self._name = "MultiDat"
        self.dats = dat_list
        self._data = []
        self._dataset = []
        self._soa = any(dat.soa for dat in self.dats)
        self._needs_halo_update = False
        for d in self.dats:
            self._data += [d._data]
            self._dataset += [d._dataset]

     @property
     def name(self):
        return self.mixed_name

     @property
     def dim(self):
        '''The number of values at each member of the dataset.'''
        dims = []
        for d in self.dats:
            dims += [d.dim]
        return dims

     @property
     def data(self):
        '''The number of values at each member of the dataset.'''
        datas = []
        for d in self.dats:
            datas += [d.data]
        return datas

     @property
     def soa(self):
        return self._soa



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

class Map(base.Map):
    """OP2 map, a relation between two :class:`Set` objects."""

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

class MultiMap(base.Map):
    def __init__(self, map_list):
        #self._multimap = True
        self._name = "MultiMap"
        #self.maps = []
        #for i in range(len(map_list)):
        #    mps= []
        #    for j in range(len(map_list[i])):
        #        mps += [map_list[i][j]]
        #    self.maps += mps
        self.maps = map_list
        self._lib_handle = None
        Map._globalcount += 1

    @property
    def dim(self):
        """Dimension of the mapping: number of dataset elements mapped to per
        iterset element."""
        dims = []
        for m in self.maps:
            dims += [m.dim]
        return dims


_sparsity_cache = dict()
def _empty_sparsity_cache():
    _sparsity_cache.clear()

class Sparsity(base.Sparsity):
    """OP2 Sparsity, a matrix structure derived from the union of the outer product of pairs of :class:`Map` objects."""

    @validate_type(('maps', (Map, tuple), MapTypeError), \
                   ('dims', (int, tuple), TypeError))
    def __new__(cls, maps, dims, name=None, block=None):
        key = 0
        if isinstance(maps, list):
            #do the stuff for the mixed spaces
            if not isinstance(dims, list):
                raise DimTypeError("The dimension(s) should be given as a list of tuple(s).")
            key = (frozenset(maps), frozenset(dims))
        else:
            #do the stuff for normal spaces
            key = (maps, as_tuple(dims, int, 2))
        cached = _sparsity_cache.get(key)
        if cached is not None:
            return cached
        return super(Sparsity, cls).__new__(cls, maps, dims, name, block)

    #@validate_type(('maps', (Map, tuple), MapTypeError), \
    #               ('dims', (int, tuple), TypeError))
    def __init__(self, maps, dims, name=None, block=None):
        if getattr(self, '_cached', False):
            return

        if isinstance(maps,list):
            for ms in maps:
                for m in ms:
                    for n in as_tuple(m, Map):
                        if len(n.values) == 0:
                            raise MapValueError("Unpopulated map values when trying to build sparsity.")
        else:
            for m in maps:
                    for n in as_tuple(m, Map):
                        if len(n.values) == 0:
                            raise MapValueError("Unpopulated map values when trying to build sparsity.")
        super(Sparsity, self).__init__(maps, dims, name, block)
        key = 0
        self.sparsity_list = []
        if isinstance(maps, list):
            key = (frozenset(maps), frozenset(dims))
            self._cached = True
            if not block:
              for i in range(self._itmaps):
                sparsity = Sparsity(zip(self._rowmaps[i],self._colmaps[i]), zip(self._rowmult[i],self._colmult[i]),"block_"+str(i),True)
                sparsity._nrows = self._nrrows[i]
                sparsity._ncols = self._nrcols[i]
                sparsity._rmaps = self._rowmaps[i]
                sparsity._cmaps = self._colmaps[i]
                sparsity._rmult = self._rowmult[i]
                sparsity._cmult = self._colmult[i]

                core.build_sparsity_mixed(sparsity, parallel=PYOP2_COMM.size > 1)

                self.sparsity_list += [sparsity]

        else:
            key = (maps, as_tuple(dims, int, 2))
            self._cached = True
            core.build_sparsity(self, parallel=PYOP2_COMM.size > 1)
        _sparsity_cache[key] = self

    def __del__(self):
        if hasattr(self, "sparsity_list"):
            for sp in self.sparsity_list:
                core.free_sparsity(sp)
        else:
            core.free_sparsity(self)

    @property
    def rowptr(self):
        return self._rowptr

    @property
    def colidx(self):
        return self._colidx

    @property
    def nnz(self):
        return self._d_nnz

    @property
    def onnz(self):
        return self._o_nnz

    @property
    def nz(self):
        return int(self._d_nz)

    @property
    def onz(self):
        return int(self._o_nz)

    def getRowMaps(self, itset):
        res = []
        for ms in self._rmaps:
            for rowmap in ms:
                if rowmap.iterset is itset:
                    res += [rowmap]
                    break
        return res

    def getColMaps(self, itset):
        res = []
        for ms in self._cmaps:
            for colmap in ms:
                if colmap.iterset is itset:
                    res += [colmap]
                    break
        return res



class Mat(base.Mat):
    """OP2 matrix data. A Mat is defined on a sparsity pattern and holds a value
    for each element in the :class:`Sparsity`."""

    def __init__(self, *args, **kwargs):
        super(Mat, self).__init__(*args, **kwargs)
        self.mat_list = []
        for i in range(len(self._sparsity.sparsity_list)):
            self.mat_list += [Mat(self._sparsity.sparsity_list[i],self._datatype,"mat_"+str(i))]

        if self._sparsity.sparsity_list != []:
            self.createNestedMat()
        else:
            self._handle = None

    def createNestedMat(self):
        mat = PETSc.Mat()
        petsc_matlist = [m.handle for m in self.mat_list]
        #from IPython import embed; embed()
        rows = len(self._sparsity._rmaps)
        cols = len(self._sparsity._cmaps)
        mat.createNest(rows,cols,petsc_matlist)#self.mat_list)
        self._handle = mat

    def _init(self):
        if not self.dtype == PETSc.ScalarType:
            raise RuntimeError("Can only create a matrix of type %s, %s is not supported" \
                    % (PETSc.ScalarType, self.dtype))
        if PYOP2_COMM.size == 1:
            mat = PETSc.Mat()
            row_lg = PETSc.LGMap()
            col_lg = PETSc.LGMap()
            if isinstance(self.sparsity.dims, list):
                #could be wrong in the most general of cases
                rdim, cdim = self.sparsity.dims[0]
                row_lg.create(indices=np.arange(self.sparsity.nrows[0] * rdim, dtype=PETSc.IntType))
                col_lg.create(indices=np.arange(self.sparsity.ncols[0] * cdim, dtype=PETSc.IntType))
                self._array = np.zeros(self.sparsity.nz, dtype=PETSc.RealType)
                mat = mat.createAIJWithArrays((self.sparsity.nrows[0]*rdim, self.sparsity.ncols[0]*cdim),
                                    (self.sparsity._rowptr, self.sparsity._colidx, self._array))

                mat.setLGMap(rmap=row_lg, cmap=col_lg)
            else:
                rdim, cdim = self.sparsity.dims
                row_lg.create(indices=np.arange(self.sparsity.nrows * rdim, dtype=PETSc.IntType))
                col_lg.create(indices=np.arange(self.sparsity.ncols * cdim, dtype=PETSc.IntType))
                self._array = np.zeros(self.sparsity.nz, dtype=PETSc.RealType)
                # We're not currently building a blocked matrix, so need to scale the
                # number of rows and columns by the sparsity dimensions
                # FIXME: This needs to change if we want to do blocked sparse
                # NOTE: using _rowptr and _colidx since we always want the host values
                mat.createAIJWithArrays((self.sparsity.nrows*rdim, self.sparsity.ncols*cdim),
                                    (self.sparsity._rowptr, self.sparsity._colidx, self._array))
                mat.setLGMap(rmap=row_lg, cmap=col_lg)
        else:
            mat = PETSc.Mat()
            row_lg = PETSc.LGMap()
            col_lg = PETSc.LGMap()
            # FIXME: probably not right for vector fields
            row_lg.create(indices=self.sparsity.maps[0][0].dataset.halo.global_to_petsc_numbering)
            col_lg.create(indices=self.sparsity.maps[0][1].dataset.halo.global_to_petsc_numbering)
            rdim, cdim = self.sparsity.dims
            mat.createAIJ(size=((self.sparsity.nrows*rdim, None),
                                (self.sparsity.ncols*cdim, None)),
                          nnz=(self.sparsity.nnz, self.sparsity.onnz))
            mat.setLGMap(rmap=row_lg, cmap=col_lg)
            mat.setOption(mat.Option.IGNORE_OFF_PROC_ENTRIES, True)
            mat.setOption(mat.Option.IGNORE_ZERO_ENTRIES, True)
            mat.setOption(mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        self._handle = mat

    def zero(self):
        """Zero the matrix."""
        self.handle.zeroEntries()

    def zero_rows(self, rows, diag_val):
        """Zeroes the specified rows of the matrix, with the exception of the
        diagonal entry, which is set to diag_val. May be used for applying
        strong boundary conditions."""
        self.handle.zeroRowsLocal(rows, diag_val)

    def _assemble(self):
        self.handle.assemble()

    @property
    def array(self):
        if not hasattr(self, '_array'):
            self._init()
        return self._array

    @property
    def values(self):
        return self.handle[:,:]

    @property
    def handle(self):
        if self._handle is None:
            self._init()
        return self._handle

class ParLoop(base.ParLoop):
    def compute(self):
        raise RuntimeError('Must select a backend')

# FIXME: Eventually (when we have a proper OpenCL solver) this wants to go in
# sequential
class Solver(base.Solver, PETSc.KSP):

    _cnt = 0

    def __init__(self, parameters=None, **kwargs):
        super(Solver, self).__init__(parameters, **kwargs)
        self.create(PETSc.COMM_WORLD)
        converged_reason = self.ConvergedReason()
        self._reasons = dict([(getattr(converged_reason,r), r) \
                              for r in dir(converged_reason) \
                              if not r.startswith('_')])

    def _set_parameters(self):
        self.setType(self.parameters['linear_solver'])
        self.getPC().setType(self.parameters['preconditioner'])
        self.rtol = self.parameters['relative_tolerance']
        self.atol = self.parameters['absolute_tolerance']
        self.divtol = self.parameters['divergence_tolerance']
        self.max_it = self.parameters['maximum_iterations']
        if self.parameters['plot_convergence']:
            self.parameters['monitor_convergence'] = True

    def solve(self, A, x, b):
        self._set_parameters()
        if hasattr(x, "_name") and x._name == "MultiDat":
            x_petsc_vec = []
            for dat in x.dats:
                ppx = PETSc.Vec().createWithArray(dat.data, size=(dat.dataset.size * dat.cdim, None))
                x_petsc_vec += [ppx]
            px = PETSc.Vec().createVecNest(x_petsc_vec)

            b_petsc_vec = []
            for dat in b.dats:
                ppb = PETSc.Vec().createWithArray(dat.data, size=(dat.dataset.size * dat.cdim, None))
                b_petsc_vec += [ppb]
            pb = PETSc.Vec().createVecNest(b_petsc_vec)
        else:
            px = PETSc.Vec().createWithArray(x.data, size=(x.dataset.size * x.cdim, None))
            pb = PETSc.Vec().createWithArray(b.data_ro, size=(b.dataset.size * b.cdim, None))
        self.setOperators(A.handle)
        self.setFromOptions()
        if self.parameters['monitor_convergence']:
            self.reshist = []
            def monitor(ksp, its, norm):
                self.reshist.append(norm)
            self.setMonitor(monitor)
        # Not using super here since the MRO would call base.Solver.solve
        PETSc.KSP.solve(self, pb, px)
        if self.parameters['monitor_convergence']:
            self.cancelMonitor()
            if self.parameters['plot_convergence']:
                try:
                    import pylab
                    pylab.semilogy(self.reshist)
                    pylab.title('Convergence history')
                    pylab.xlabel('Iteration')
                    pylab.ylabel('Residual norm')
                    pylab.savefig('%sreshist_%04d.png' % (self.parameters['plot_prefix'], Solver._cnt))
                    Solver._cnt += 1
                except ImportError:
                    from warnings import warn
                    warn("pylab not available, not plotting convergence history.")
        r = self.getConvergedReason()
        if cfg.debug:
            print "Converged reason: %s" % self._reasons[r]
            print "Iterations: %s" % self.getIterationNumber()
            print "Residual norm: %s" % self.getResidualNorm()
        if r < 0:
            msg = "KSP Solver failed to converge in %d iterations: %s (Residual norm: %e)" \
                    % (self.getIterationNumber(), self._reasons[r], self.getResidualNorm())
            if self.parameters['error_on_nonconvergence']:
                raise RuntimeError(msg)
            else:
                from warnings import warn
                warn(msg)
