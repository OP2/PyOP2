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

"""Base classes for OP2 objects. The versions here extend those from the
:mod:`base` module to include runtime data information which is backend
independent. Individual runtime backends should subclass these as
required to implement backend-specific features.

.. _MatMPIAIJSetPreallocation: http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatMPIAIJSetPreallocation.html
"""

from contextlib import contextmanager
from petsc4py import PETSc, __version__ as petsc4py_version

import base
from base import *
from backends import _make_object
from logger import debug, warning
from versioning import CopyOnWrite, modifies, zeroes
from profiling import timed_region
import mpi
from mpi import collective
import sparsity


if petsc4py_version < '3.4':
    raise RuntimeError("Incompatible petsc4py version %s. At least version 3.4 is required."
                       % petsc4py_version)


class MPIConfig(mpi.MPIConfig):

    def __init__(self):
        super(MPIConfig, self).__init__()
        PETSc.Sys.setDefaultComm(self.comm)

    @mpi.MPIConfig.comm.setter
    @collective
    def comm(self, comm):
        """Set the MPI communicator for parallel communication."""
        self.COMM = mpi._check_comm(comm)
        # PETSc objects also need to be built on the same communicator.
        PETSc.Sys.setDefaultComm(self.comm)

MPI = MPIConfig()
# Override MPI configuration
mpi.MPI = MPI


class Dat(base.Dat):

    @contextmanager
    def vec_context(self, readonly=True):
        """A context manager for a :class:`PETSc.Vec` from a :class:`Dat`.

        :param readonly: Access the data read-only (use :meth:`Dat.data_ro`)
                         or read-write (use :meth:`Dat.data`). Read-write
                         access requires a halo update."""

        acc = (lambda d: d.data_ro) if readonly else (lambda d: d.data)
        # Getting the Vec needs to ensure we've done all current computation.
        self._force_evaluation()
        if not hasattr(self, '_vec'):
            size = (self.dataset.size * self.cdim, None)
            self._vec = PETSc.Vec().createWithArray(acc(self), size=size,
                                                    bsize=self.cdim)
        # PETSc Vecs have a state counter and cache norm computations
        # to return immediately if the state counter is unchanged.
        # Since we've updated the data behind their back, we need to
        # change that state counter.
        self._vec.stateIncrease()
        yield self._vec
        if not readonly:
            self.needs_halo_update = True

    @property
    @modifies
    @collective
    def vec(self):
        """Context manager for a PETSc Vec appropriate for this Dat.

        You're allowed to modify the data you get back from this view."""
        return self.vec_context(readonly=False)

    @property
    @collective
    def vec_ro(self):
        """Context manager for a PETSc Vec appropriate for this Dat.

        You're not allowed to modify the data you get back from this view."""
        return self.vec_context()

    @collective
    def dump(self, filename):
        """Dump the vector to file ``filename`` in PETSc binary format."""
        base._trace.evaluate(set([self]), set())
        vwr = PETSc.Viewer().createBinary(filename, PETSc.Viewer.Mode.WRITE)
        self.vec.view(vwr)


class MixedDat(base.MixedDat):

    @contextmanager
    def vecscatter(self, readonly=True):
        """A context manager scattering the arrays of all components of this
        :class:`MixedDat` into a contiguous :class:`PETSc.Vec` and reverse
        scattering to the original arrays when exiting the context.

        :param readonly: Access the data read-only (use :meth:`Dat.data_ro`)
                         or read-write (use :meth:`Dat.data`). Read-write
                         access requires a halo update.

        .. note::

           The :class:`~PETSc.Vec` obtained from this context is in
           the correct order to be left multiplied by a compatible
           :class:`MixedMat`.  In parallel it is *not* just a
           concatenation of the underlying :class:`Dat`\s."""

        acc = (lambda d: d.vec_ro) if readonly else (lambda d: d.vec)
        # Allocate memory for the contiguous vector, create the scatter
        # contexts and stash them on the object for later reuse
        if not (hasattr(self, '_vec') and hasattr(self, '_sctxs')):
            self._vec = PETSc.Vec().create()
            # Size of flattened vector is product of size and cdim of each dat
            sz = sum(d.dataset.size * d.dataset.cdim for d in self._dats)
            self._vec.setSizes((sz, None))
            self._vec.setUp()
            self._sctxs = []
            # To be compatible with a MatNest (from a MixedMat) the
            # ordering of a MixedDat constructed of Dats (x_0, ..., x_k)
            # on P processes is:
            # (x_0_0, x_1_0, ..., x_k_0, x_0_1, x_1_1, ..., x_k_1, ..., x_k_P)
            # That is, all the Dats from rank 0, followed by those of
            # rank 1, ...
            # Hence the offset into the global Vec is the exclusive
            # prefix sum of the local size of the mixed dat.
            offset = MPI.comm.exscan(sz)
            if offset is None:
                offset = 0

            for d in self._dats:
                sz = d.dataset.size * d.dataset.cdim
                with acc(d) as v:
                    vscat = PETSc.Scatter().create(v, None, self._vec,
                                                   PETSc.IS().createStride(sz, offset, 1))
                offset += sz
                self._sctxs.append(vscat)
        # Do the actual forward scatter to fill the full vector with values
        for d, vscat in zip(self._dats, self._sctxs):
            with acc(d) as v:
                vscat.scatterBegin(v, self._vec, addv=PETSc.InsertMode.INSERT_VALUES)
                vscat.scatterEnd(v, self._vec, addv=PETSc.InsertMode.INSERT_VALUES)
        yield self._vec
        if not readonly:
            # Reverse scatter to get the values back to their original locations
            for d, vscat in zip(self._dats, self._sctxs):
                with acc(d) as v:
                    vscat.scatterBegin(self._vec, v, addv=PETSc.InsertMode.INSERT_VALUES,
                                       mode=PETSc.ScatterMode.REVERSE)
                    vscat.scatterEnd(self._vec, v, addv=PETSc.InsertMode.INSERT_VALUES,
                                     mode=PETSc.ScatterMode.REVERSE)
            self.needs_halo_update = True

    @property
    @modifies
    @collective
    def vec(self):
        """Context manager for a PETSc Vec appropriate for this Dat.

        You're allowed to modify the data you get back from this view."""
        return self.vecscatter(readonly=False)

    @property
    @collective
    def vec_ro(self):
        """Context manager for a PETSc Vec appropriate for this Dat.

        You're not allowed to modify the data you get back from this view."""
        return self.vecscatter()


class ProxySparsity(base.Sparsity):
    def __init__(self, parent, i, j):
        self._dsets = (parent.dsets[0][i], parent.dsets[1][j])
        self._rmaps = tuple(m.split[i] for m in parent.rmaps)
        self._cmaps = tuple(m.split[j] for m in parent.cmaps)
        self._nrows = self._dsets[0].size
        self._ncols = self._dsets[1].size

        self._dims = tuple([tuple([parent.dims[i][j]])])
        self._blocks = [[self]]

    @classmethod
    def _process_args(cls, *args, **kwargs):
        return (None, ) + args, kwargs

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        return None


class ProxyMat(base.Mat):
    def __init__(self, parent, i, j):
        self._parent = parent
        self._i = i
        self._j = j
        self._sparsity = ProxySparsity(parent.sparsity, i, j)
        self._rowis = self._parent.local_ises[0][self._i]
        self._colis = self._parent.local_ises[1][self._j]
        self._diag_dat = _make_object('MixedDat', parent.sparsity.dsets[0])

    def __getitem__(self, idx):
        return self

    def __iter__(self):
        yield self

    def inc_local_diagonal_entries(self, rows, diag_val=1.0):
        rbs, cbs = self.dims[0][0]
        vals = np.repeat(diag_val, len(rows) * rbs * cbs).reshape(-1, rbs*cbs)
        self.handle.setValuesBlockedLocalRCV(rows.reshape(-1, 1), rows.reshape(-1, 1),
                                             vals, addv=PETSc.InsertMode.ADD_VALUES)

    @property
    def handle(self):
        if not hasattr(self, '_handle'):
            self._handle = PETSc.Mat().create()
            self._parent.handle.getLocalSubMatrix(isrow=self._rowis,
                                                  iscol=self._colis,
                                                  submat=self._handle)
        return self._handle

    def assemble(self):
        pass

    @property
    def dims(self):
        return self.sparsity.dims

    @property
    def values(self):
        ris = self._parent.global_ises[0][self._i]
        cis = self._parent.global_ises[1][self._j]
        mat = self._parent.handle.getSubMatrix(isrow=ris,
                                               iscol=cis)
        return mat[:, :]

    @property
    def dtype(self):
        return self._parent.dtype

    @property
    def nbytes(self):
        return self._parent.nbytes / (np.prod(self.sparsity.shape))

    def __repr__(self):
        return "ProxyMat(%r, %r, %r)" % (self._parent, self._i, self._j)


class Mat(base.Mat, CopyOnWrite):
    """OP2 matrix data. A Mat is defined on a sparsity pattern and holds a value
    for each element in the :class:`Sparsity`."""

    @collective
    def _init(self):
        if not self.dtype == PETSc.ScalarType:
            raise RuntimeError("Can only create a matrix of type %s, %s is not supported"
                               % (PETSc.ScalarType, self.dtype))
        # If the Sparsity is defined on MixedDataSets, we need to build a MatNest
        if self.sparsity.shape > (1, 1):
            if configuration["matnest"]:
                self._init_nest()
            else:
                self._init_monolithic()
        else:
            self._init_block()

    def _make_local_ises(self):
        rowises = []
        colises = []
        rows, cols = self.sparsity.shape
        dsets = self.sparsity.dsets
        start = 0
        for i in range(rows):
            bs = self.dims[i][0][0]
            n = dsets[0][i].total_size
            rowis = PETSc.IS().createStride(n, first=start, step=1)
            rowis.setBlockSize(bs)
            start += n
            rowises.append(rowis)
        start = 0
        for i in range(cols):
            bs = self.dims[0][i][1]
            n = dsets[1][i].total_size
            colis = PETSc.IS().createStride(n, first=start, step=1)
            colis.setBlockSize(bs)
            start += n
            colises.append(colis)

        self._local_ises = (tuple(rowises), tuple(colises))

    @property
    def local_ises(self):
        if not hasattr(self, '_local_ises'):
            self._make_local_ises()
        return self._local_ises

    def _make_global_ises(self):
        rowises = []
        colises = []
        rows, cols = self.sparsity.shape
        dsets = self.sparsity.dsets

        nrows = self.nrows
        ncols = self.ncols
        start = MPI.comm.scan(nrows) - nrows
        for i in range(rows):
            bs = self.dims[i][0][0]
            n = dsets[0][i].size
            rowis = PETSc.IS().createStride(n, first=start, step=1)
            rowis.setBlockSize(bs)
            start += n
            rowises.append(rowis)

        start = MPI.comm.scan(ncols) - ncols
        for i in range(cols):
            bs = self.dims[0][i][1]
            n = dsets[1][i].size
            colis = PETSc.IS().createStride(n, first=start, step=1)
            colis.setBlockSize(bs)
            start += n
            colises.append(colis)

        self._global_ises = (tuple(rowises), tuple(colises))

    @property
    def global_ises(self):
        if not hasattr(self, '_global_ises'):
            self._make_global_ises()
        return self._global_ises

    def _make_monolithic_l2g(self, dset):
        # Compute local to global maps for a monolithic mixed system
        # from the individual local to global maps for each field.
        # Exposition:
        #
        # We have N fields and P processes.  The global matrix row
        # ordering is:
        #
        # f_0_p_0, f_1_p_0, ..., f_N_p_0; f_0_p_1, ..., ; f_0_p_P,
        # ..., f_N_p_P.
        # And similarly for the columns.
        #
        # We have per-field local to global numberings, to convert
        # these into multi-field local to global numberings, we note
        # the following:
        #
        # For each entry in the per-field l2g map, we first determine
        # the rank that entry belongs to, call this r.
        #
        # We know that this must be offset by:
        # 1. The sum of all field lengths with rank < r
        # 2. The sum of all lower-numbered field lengths on rank r.
        #
        # Finally, we need to shift the field-local entry by the
        # current field offset.

        idx_size = sum(s.total_size*s.cdim for s in dset)
        indices = -np.ones(idx_size, dtype=PETSc.IntType)

        owned_sz = np.array([sum(s.size * s.cdim for s in dset)], dtype=PETSc.IntType)

        field_offset = np.empty_like(owned_sz)

        MPI.comm.Scan(owned_sz, field_offset)
        field_offset -= owned_sz

        all_field_offsets = np.empty(MPI.comm.size, dtype=PETSc.IntType)
        MPI.comm.Allgather(field_offset, all_field_offsets)

        start = 0
        all_local_offsets = np.zeros(MPI.comm.size, dtype=PETSc.IntType)

        current_offsets = np.zeros(MPI.comm.size + 1, dtype=PETSc.IntType)

        for s in dset:
            l2g = s.halo.global_to_petsc_numbering
            idx = indices[start:start + s.total_size*s.cdim]
            owned_sz[0] = s.size * s.cdim

            MPI.comm.Scan(owned_sz, field_offset)
            MPI.comm.Allgather(field_offset, current_offsets[1:])
            # Find the ranks each entry in the l2g belongs to
            tmp_indices = np.searchsorted(current_offsets, l2g, side="right") - 1
            # Compute the new
            idx[:] = l2g[:] - current_offsets[tmp_indices] + \
                all_field_offsets[tmp_indices] + all_local_offsets[tmp_indices]
            MPI.comm.Allgather(owned_sz, current_offsets[1:])
            all_local_offsets += current_offsets[1:]
            start += s.total_size * s.cdim
        return indices

    def _init_monolithic(self):
        mat = PETSc.Mat()
        mat.createAIJ(size=((self.nrows, None), (self.ncols, None)),
                      nnz=(self.sparsity.nnz, self.sparsity.onnz),
                      bsize=(1, 1))
        row_lg = PETSc.LGMap()

        row_set, col_set = self.sparsity.dsets

        if MPI.comm.size == 1:
            row_lg.create(indices=np.arange(self.nrows, dtype=PETSc.IntType),
                          bsize=1)
            if row_set == col_set:
                col_lg = row_lg
            else:
                col_lg = PETSc.LGMap()
                col_lg.create(indices=np.arange(self.ncols, dtype=PETSc.IntType),
                              bsize=1)
        else:
            rindices = self._make_monolithic_l2g(row_set)
            row_lg.create(indices=rindices, bsize=1)
            if row_set == col_set:
                col_lg = row_lg
            else:
                cindices = self._make_monolithic_l2g(col_set)
                col_lg = PETSc.LGMap()
                col_lg.create(indices=cindices, bsize=1)

        mat.setLGMap(rmap=row_lg, cmap=col_lg)
        self._handle = mat
        self._blocks = []
        rows, cols = self.sparsity.shape
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(ProxyMat(self, i, j))
            self._blocks.append(row)
        mat.setOption(mat.Option.IGNORE_ZERO_ENTRIES, False)
        mat.setOption(mat.Option.KEEP_NONZERO_PATTERN, True)
        # We completely fill the allocated matrix when zeroing the
        # entries, so raise an error if we "missed" one.
        mat.setOption(mat.Option.UNUSED_NONZERO_LOCATION_ERR, True)
        mat.setOption(mat.Option.IGNORE_OFF_PROC_ENTRIES, True)
        mat.setOption(mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        # Put zeros in all the places we might eventually put a value.
        for i in range(rows):
            for j in range(cols):
                sparsity.fill_with_zeros(self[i, j].handle,
                                         self[i, j].sparsity.dims[0][0],
                                         self[i, j].sparsity.maps)

        mat.assemble()
        mat.setOption(mat.Option.IGNORE_ZERO_ENTRIES, True)

    def _init_nest(self):
        mat = PETSc.Mat()
        self._blocks = []
        rows, cols = self.sparsity.shape
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(Mat(self.sparsity[i, j], self.dtype,
                           '_'.join([self.name, str(i), str(j)])))
            self._blocks.append(row)
        # PETSc Mat.createNest wants a flattened list of Mats
        mat.createNest([[m.handle for m in row_] for row_ in self._blocks],
                       isrows=self.global_ises[0], iscols=self.global_ises[1])
        self._handle = mat

    def _init_block(self):
        self._blocks = [[self]]
        mat = PETSc.Mat()
        row_lg = PETSc.LGMap()
        col_lg = PETSc.LGMap()
        rdim, cdim = self.dims[0][0]
        if MPI.comm.size == 1:
            # The PETSc local to global mapping is the identity in the sequential case
            row_lg.create(
                indices=np.arange(self.nblock_rows, dtype=PETSc.IntType),
                bsize=rdim)
            col_lg.create(
                indices=np.arange(self.nblock_cols, dtype=PETSc.IntType),
                bsize=cdim)
        else:
            rindices = self.sparsity.dsets[0].halo.global_to_petsc_numbering
            cindices = self.sparsity.dsets[1].halo.global_to_petsc_numbering
            row_lg.create(indices=rindices, bsize=rdim)
            col_lg.create(indices=cindices, bsize=cdim)

        if rdim == cdim and rdim > 1:
            # Size is total number of rows and columns, but the
            # /sparsity/ is the block sparsity.
            block_sparse = True
            create = mat.createBAIJ
        else:
            # Size is total number of rows and columns, sparsity is
            # the /dof/ sparsity.
            block_sparse = False
            create = mat.createAIJ
        create(size=((self.nrows, None),
                     (self.ncols, None)),
               nnz=(self.sparsity.nnz, self.sparsity.onnz),
               bsize=(rdim, cdim))
        mat.setLGMap(rmap=row_lg, cmap=col_lg)
        # Do not stash entries destined for other processors, just drop them
        # (we take care of those in the halo)
        mat.setOption(mat.Option.IGNORE_OFF_PROC_ENTRIES, True)
        # Any add or insertion that would generate a new entry that has not
        # been preallocated will raise an error
        mat.setOption(mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        # Do not ignore zeros while we fill the initial matrix so that
        # petsc doesn't compress things out.
        if not block_sparse:
            mat.setOption(mat.Option.IGNORE_ZERO_ENTRIES, False)
        # When zeroing rows (e.g. for enforcing Dirichlet bcs), keep those in
        # the nonzero structure of the matrix. Otherwise PETSc would compact
        # the sparsity and render our sparsity caching useless.
        mat.setOption(mat.Option.KEEP_NONZERO_PATTERN, True)
        # We completely fill the allocated matrix when zeroing the
        # entries, so raise an error if we "missed" one.
        mat.setOption(mat.Option.UNUSED_NONZERO_LOCATION_ERR, True)

        # Put zeros in all the places we might eventually put a value.
        sparsity.fill_with_zeros(mat, self.sparsity.dims[0][0], self.sparsity.maps)

        # Now we've filled up our matrix, so the sparsity is
        # "complete", we can ignore subsequent zero entries.
        if not block_sparse:
            mat.setOption(mat.Option.IGNORE_ZERO_ENTRIES, True)
        self._handle = mat
        # Matrices start zeroed.
        self._version_set_zero()

    def __getitem__(self, idx):
        """Return :class:`Mat` block with row and column given by ``idx``
        or a given row of blocks."""
        try:
            i, j = idx
            return self.blocks[i][j]
        except TypeError:
            return self.blocks[idx]

    def __iter__(self):
        """Iterate over all :class:`Mat` blocks by row and then by column."""
        for row in self.blocks:
            for s in row:
                yield s

    @collective
    def dump(self, filename):
        """Dump the matrix to file ``filename`` in PETSc binary format."""
        base._trace.evaluate(set([self]), set())
        vwr = PETSc.Viewer().createBinary(filename, PETSc.Viewer.Mode.WRITE)
        self.handle.view(vwr)

    @zeroes
    @collective
    def zero(self):
        """Zero the matrix."""
        base._trace.evaluate(set(), set([self]))
        self.handle.zeroEntries()

    @modifies
    @collective
    def zero_rows(self, rows, diag_val=1.0):
        """Zeroes the specified rows of the matrix, with the exception of the
        diagonal entry, which is set to diag_val. May be used for applying
        strong boundary conditions.

        :param rows: a :class:`Subset` or an iterable"""
        base._trace.evaluate(set([self]), set([self]))
        rows = rows.indices if isinstance(rows, Subset) else rows
        self.handle.zeroRowsLocal(rows, diag_val)

    @modifies
    @collective
    def set_diagonal(self, vec):
        """Add a vector to the diagonal of the matrix.

        :params vec: vector to add (:class:`Dat` or :class:`PETsc.Vec`)"""
        if self.sparsity.shape != (1, 1):
            if not isinstance(vec, base.MixedDat):
                raise TypeError('Can only set diagonal of blocked Mat from MixedDat')
            if vec.dataset != self.sparsity.dsets[1]:
                raise TypeError('Mismatching datasets for MixedDat and Mat')
            rows, cols = self.sparsity.shape
            for i in range(rows):
                if i < cols:
                    self[i, i].set_diagonal(vec[i])
            return
        r, c = self.handle.getSize()
        if r != c:
            raise MatTypeError('Cannot set diagonal of non-square matrix')
        if not isinstance(vec, (base.Dat, PETSc.Vec)):
            raise TypeError("Can only set diagonal from a Dat or PETSc Vec.")
        if isinstance(vec, PETSc.Vec):
            self.handle.setDiagonal(vec)
        else:
            with vec.vec_ro as v:
                self.handle.setDiagonal(v)

    def _cow_actual_copy(self, src):
        self._handle = src.handle.duplicate(copy=True)
        return self

    @modifies
    @collective
    def inc_local_diagonal_entries(self, rows, diag_val=1.0):
        """Increment the diagonal entry in ``rows`` by a particular value.

        :param rows: a :class:`Subset` or an iterable.
        :param diag_val: the value to add

        The indices in ``rows`` should index the process-local rows of
        the matrix (no mapping to global indexes is applied).

        The diagonal entries corresponding to the complement of rows
        are incremented by zero.
        """
        base._trace.evaluate(set([self]), set([self]))
        vec = self.handle.createVecLeft()
        vec.setOption(vec.Option.IGNORE_OFF_PROC_ENTRIES, True)
        rows = np.asarray(rows)
        rows = rows[rows < self.sparsity.dsets[0].size]
        # If the row DataSet has dimension > 1 we need to treat the given rows
        # as block indices and set all rows in each block
        rdim = self.sparsity.dsets[0].cdim
        if rdim > 1:
            rows = np.dstack([rdim*rows + i for i in range(rdim)]).flatten()
        with vec as array:
            array[rows] = diag_val
        self.handle.setDiagonal(vec, addv=PETSc.InsertMode.ADD_VALUES)

    @collective
    def _assemble(self):
        self.handle.assemble()

    @property
    def blocks(self):
        """2-dimensional array of matrix blocks."""
        if not hasattr(self, '_blocks'):
            self._init()
        return self._blocks

    @property
    @modifies
    def values(self):
        base._trace.evaluate(set([self]), set())
        self._assemble()
        return self.handle[:, :]

    @property
    def handle(self):
        """Petsc4py Mat holding matrix data."""
        if not hasattr(self, '_handle'):
            self._init()
        return self._handle

    def __mul__(self, v):
        """Multiply this :class:`Mat` with the vector ``v``."""
        if not isinstance(v, (base.Dat, PETSc.Vec)):
            raise TypeError("Can only multiply Mat and Dat or PETSc Vec.")
        if isinstance(v, base.Dat):
            with v.vec_ro as vec:
                y = self.handle * vec
        else:
            y = self.handle * v
        if isinstance(v, base.MixedDat):
            dat = _make_object('MixedDat', self.sparsity.dsets[0])
            offset = 0
            for d in dat:
                sz = d.dataset.set.size
                d.data[:] = y.getSubVector(PETSc.IS().createStride(sz, offset, 1)).array[:]
                offset += sz
        else:
            dat = _make_object('Dat', self.sparsity.dsets[0])
            dat.data[:] = y.array[:]
        dat.needs_halo_update = True
        return dat

# FIXME: Eventually (when we have a proper OpenCL solver) this wants to go in
# sequential


class Solver(base.Solver, PETSc.KSP):

    _cnt = 0

    def __init__(self, parameters=None, **kwargs):
        super(Solver, self).__init__(parameters, **kwargs)
        self._count = Solver._cnt
        Solver._cnt += 1
        self.create(PETSc.COMM_WORLD)
        self._opt_prefix = 'pyop2_ksp_%d' % self._count
        self.setOptionsPrefix(self._opt_prefix)
        converged_reason = self.ConvergedReason()
        self._reasons = dict([(getattr(converged_reason, r), r)
                              for r in dir(converged_reason)
                              if not r.startswith('_')])

    @collective
    def _set_parameters(self):
        opts = PETSc.Options(self._opt_prefix)
        for k, v in self.parameters.iteritems():
            if type(v) is bool:
                if v:
                    opts[k] = None
                else:
                    continue
            else:
                opts[k] = v
        self.setFromOptions()

    def __del__(self):
        # Remove stuff from the options database
        # It's fixed size, so if we don't it gets too big.
        if hasattr(self, '_opt_prefix'):
            opts = PETSc.Options()
            for k in self.parameters.iterkeys():
                del opts[self._opt_prefix + k]
            delattr(self, '_opt_prefix')

    @collective
    def _solve(self, A, x, b):
        self._set_parameters()
        # Set up the operator only if it has changed
        if not self.getOperators()[0] == A.handle:
            self.setOperators(A.handle)
            if self.parameters['pc_type'] == 'fieldsplit' and A.sparsity.shape != (1, 1):
                rows, cols = A.sparsity.shape
                ises = []
                nlocal_rows = 0
                for i in range(rows):
                    if i < cols:
                        nlocal_rows += A[i, i].nrows
                offset = 0
                if MPI.comm.rank == 0:
                    MPI.comm.exscan(nlocal_rows)
                else:
                    offset = MPI.comm.exscan(nlocal_rows)
                for i in range(rows):
                    if i < cols:
                        nrows = A[i, i].nrows
                        ises.append((str(i), PETSc.IS().createStride(nrows, first=offset, step=1)))
                        offset += nrows
                self.getPC().setFieldSplitIS(*ises)
        if self.parameters['plot_convergence']:
            self.reshist = []

            def monitor(ksp, its, norm):
                self.reshist.append(norm)
                debug("%3d KSP Residual norm %14.12e" % (its, norm))
            self.setMonitor(monitor)
        # Not using super here since the MRO would call base.Solver.solve
        with timed_region("PETSc Krylov solver"):
            with b.vec_ro as bv:
                with x.vec as xv:
                    PETSc.KSP.solve(self, bv, xv)
        if self.parameters['plot_convergence']:
            self.cancelMonitor()
            try:
                import pylab
                pylab.semilogy(self.reshist)
                pylab.title('Convergence history')
                pylab.xlabel('Iteration')
                pylab.ylabel('Residual norm')
                pylab.savefig('%sreshist_%04d.png' %
                              (self.parameters['plot_prefix'], self._count))
            except ImportError:
                warning("pylab not available, not plotting convergence history.")
        r = self.getConvergedReason()
        debug("Converged reason: %s" % self._reasons[r])
        debug("Iterations: %s" % self.getIterationNumber())
        debug("Residual norm: %s" % self.getResidualNorm())
        if r < 0:
            msg = "KSP Solver failed to converge in %d iterations: %s (Residual norm: %e)" \
                % (self.getIterationNumber(), self._reasons[r], self.getResidualNorm())
            if self.parameters['error_on_nonconvergence']:
                raise RuntimeError(msg)
            else:
                warning(msg)
