from pyop2.types.dat import Dat as BaseDat, MixedDat, DatView
from pyop2.types.set import Set, ExtrudedSet, Subset, MixedSet
from pyop2.types.dataset import DataSet, GlobalDataSet, MixedDataSet
from pyop2.types.map import Map, MixedMap
from pyop2.parloop import AbstractParloop
from pyop2.global_kernel import AbstractGlobalKernel
from pyop2.types.access import READ, INC, MIN, MAX
from pyop2.types.mat import Mat
from pyop2.types.glob import Global as BaseGlobal
from pyop2.backends import AbstractComputeBackend
from petsc4py import PETSc
from pyop2 import (
    compilation,
    mpi,
    utils
)

import ctypes
import os
import loopy as lp
from contextlib import contextmanager
import numpy as np


class Dat(BaseDat):
    @utils.cached_property
    def _vec(self):
        assert self.dtype == PETSc.ScalarType, \
            "Can't create Vec with type %s, must be %s" % (self.dtype,
                                                           PETSc.ScalarType)
        # Can't duplicate layout_vec of dataset, because we then
        # carry around extra unnecessary data.
        # But use getSizes to save an Allreduce in computing the
        # global size.
        size = self.dataset.layout_vec.getSizes()
        data = self._data[:size[0]]
        vec = PETSc.Vec().createWithArray(data, size=size,
                                          bsize=self.cdim, comm=self.comm)
        return vec

    @contextmanager
    def vec_context(self, access):
        # PETSc Vecs have a state counter and cache norm computations
        # to return immediately if the state counter is unchanged.
        # Since we've updated the data behind their back, we need to
        # change that state counter.
        self._vec.stateIncrease()
        yield self._vec
        if access is not READ:
            self.halo_valid = False

    def ensure_availability_on_device(self):
        from pyop2.op2 import compute_backend
        assert compute_backend is cpu_backend
        # data transfer is noop for CPU backend

    def ensure_availability_on_host(self):
        from pyop2.op2 import compute_backend
        assert compute_backend is cpu_backend
        # data transfer is noop for CPU backend


class Global(BaseGlobal):
    @utils.cached_property
    def _vec(self):
        assert self.dtype == PETSc.ScalarType, \
            "Can't create Vec with type %s, must be %s" % (self.dtype,
                                                           PETSc.ScalarType)
        # Can't duplicate layout_vec of dataset, because we then
        # carry around extra unnecessary data.
        # But use getSizes to save an Allreduce in computing the
        # global size.
        data = self._data
        size = self.dataset.layout_vec.getSizes()
        if self.comm.rank == 0:
            return PETSc.Vec().createWithArray(data, size=size,
                                               bsize=self.cdim,
                                               comm=self.comm)
        else:
            return PETSc.Vec().createWithArray(np.empty(0, dtype=self.dtype),
                                               size=size,
                                               bsize=self.cdim,
                                               comm=self.comm)

    @contextmanager
    def vec_context(self, access):
        """A context manager for a :class:`PETSc.Vec` from a :class:`Global`.

        :param access: Access descriptor: READ, WRITE, or RW."""
        yield self._vec
        if access is not READ:
            data = self._data
            self.comm.Bcast(data, 0)

    def ensure_availability_on_device(self):
        from pyop2.op2 import compute_backend
        assert compute_backend is cpu_backend
        # data transfer is noop for CPU backend

    def ensure_availability_on_host(self):
        from pyop2.op2 import compute_backend
        assert compute_backend is cpu_backend
        # data transfer is noop for CPU backend


class GlobalKernel(AbstractGlobalKernel):

    @utils.cached_property
    def code_to_compile(self):
        """Return the C/C++ source code as a string."""
        from pyop2.codegen.rep2loopy import generate

        wrapper = generate(self.builder)
        code = lp.generate_code_v2(wrapper)

        if self.local_kernel.cpp:
            from loopy.codegen.result import process_preambles
            preamble = "".join(
                process_preambles(getattr(code, "device_preambles", [])))
            device_code = "\n\n".join(str(dp.ast) for dp in code.device_programs)
            return preamble + '\nextern "C" {\n' + device_code + "\n}\n"
        return code.device_code()

    @PETSc.Log.EventDecorator()
    @mpi.collective
    def compile(self, comm):
        """Compile the kernel.

        :arg comm: The communicator the compilation is collective over.
        :returns: A ctypes function pointer for the compiled function.
        """
        extension = "cpp" if self.local_kernel.cpp else "c"
        cppargs = (
            tuple("-I%s/include" % d for d in utils.get_petsc_dir())
            + tuple("-I%s" % d for d in self.local_kernel.include_dirs)
            + ("-I%s" % os.path.abspath(os.path.dirname(__file__)),)
        )
        ldargs = (
            tuple("-L%s/lib" % d for d in utils.get_petsc_dir())
            + tuple("-Wl,-rpath,%s/lib" % d for d in utils.get_petsc_dir())
            + ("-lpetsc", "-lm")
            + tuple(self.local_kernel.ldargs)
        )

        return compilation.load(self, extension, self.name,
                                cppargs=cppargs,
                                ldargs=ldargs,
                                restype=ctypes.c_int,
                                comm=comm)


class Parloop(AbstractParloop):
    @PETSc.Log.EventDecorator("ParLoopRednBegin")
    @mpi.collective
    def reduction_begin(self):
        """Begin reductions."""
        requests = []
        for idx in self._reduction_idxs:
            glob = self.arguments[idx].data
            mpi_op = {INC: mpi.MPI.SUM,
                      MIN: mpi.MPI.MIN,
                      MAX: mpi.MPI.MAX}.get(self.accesses[idx])

            if mpi.MPI.VERSION >= 3:
                requests.append(self.comm.Iallreduce(glob._data,
                                                     glob._buf,
                                                     op=mpi_op))
            else:
                self.comm.Allreduce(glob._data, glob._buf, op=mpi_op)
        return tuple(requests)

    @PETSc.Log.EventDecorator("ParLoopRednEnd")
    @mpi.collective
    def reduction_end(self, requests):
        """Finish reductions."""
        if mpi.MPI.VERSION >= 3:
            mpi.MPI.Request.Waitall(requests)
            for idx in self._reduction_idxs:
                glob = self.arguments[idx].data
                glob._data[:] = glob._buf
        else:
            assert len(requests) == 0

            for idx in self._reduction_idxs:
                glob = self.arguments[idx].data
                glob._data[:] = glob._buf


class CPUBackend(AbstractComputeBackend):
    GlobalKernel = GlobalKernel
    Parloop = Parloop
    Set = Set
    ExtrudedSet = ExtrudedSet
    MixedSet = MixedSet
    Subset = Subset
    DataSet = DataSet
    MixedDataSet = MixedDataSet
    Map = Map
    MixedMap = MixedMap
    Dat = Dat
    MixedDat = MixedDat
    DatView = DatView
    Mat = Mat
    Global = Global
    GlobalDataSet = GlobalDataSet
    PETScVecType = PETSc.Vec.Type.STANDARD

    def turn_on_offloading(self):
        pass

    def turn_off_offloading(self):
        pass

    @property
    def cache_key(self):
        return (type(self),)


cpu_backend = CPUBackend()
