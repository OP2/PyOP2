"""OP2 OpenCL backend."""

import os
from hashlib import md5
from contextlib import contextmanager

from pyop2 import compilation, mpi, utils
from pyop2.offload_utils import (AVAILABLE_ON_HOST_ONLY,
                                 AVAILABLE_ON_DEVICE_ONLY,
                                 AVAILABLE_ON_BOTH,
                                 DataAvailability)
from pyop2.configuration import configuration
from pyop2.types.set import (MixedSet, Subset as BaseSubset,
                             ExtrudedSet as BaseExtrudedSet,
                             Set)
from pyop2.types.map import Map as BaseMap, MixedMap
from pyop2.types.dat import Dat as BaseDat, MixedDat, DatView
from pyop2.types.dataset import DataSet, GlobalDataSet, MixedDataSet
from pyop2.types.mat import Mat
from pyop2.types.glob import Global as BaseGlobal
from pyop2.types.access import RW, READ, MIN, MAX, INC
from pyop2.parloop import AbstractParloop
from pyop2.global_kernel import AbstractGlobalKernel
from pyop2.backends import AbstractComputeBackend, cpu as cpu_backend
from petsc4py import PETSc

import numpy
import loopy as lp
from pytools import memoize_method
from dataclasses import dataclass
from typing import Tuple
import ctypes

import pyopencl as cl
import pyopencl.array as cla


def read_only_clarray_setitem(self, *args, **kwargs):
    # emulates np.ndarray.setitem for numpy arrays with read-only flags
    raise ValueError("assignment destination is read-only")


class Map(BaseMap):

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        self._availability_flag = AVAILABLE_ON_HOST_ONLY

    @utils.cached_property
    def _opencl_values(self):
        self._availability_flag = AVAILABLE_ON_BOTH
        return cla.to_device(opencl_backend.queue, self._values)

    def get_availability(self):
        return self._availability_flag

    def ensure_availability_on_device(self):
        self._opencl_values

    def ensure_availability_on_host(self):
        # Map once initialized is not over-written so always available
        # on host.
        pass

    @property
    def _kernel_args_(self):
        if opencl_backend.offloading:
            if not self.is_available_on_device():
                self.ensure_availability_on_device()

            return (self._opencl_values.data,)
        else:
            return super(Map, self)._kernel_args_


class ExtrudedSet(BaseExtrudedSet):
    """
    ExtrudedSet for OpenCL.
    """

    def __init__(self, *args, **kwargs):
        super(ExtrudedSet, self).__init__(*args, **kwargs)
        self._availability_flag = AVAILABLE_ON_HOST_ONLY

    @utils.cached_property
    def opencl_layers_array(self):
        self._availability_flag = AVAILABLE_ON_BOTH
        return cla.to_device(opencl_backend.queue, self.layers_array)

    def get_availability(self):
        return self._availability_flag

    def ensure_availability_on_device(self):
        self.opencl_layers_array

    def ensure_availability_on_host(self):
        # ExtrudedSet once initialized is not over-written so always available
        # on host.
        pass

    @property
    def _kernel_args_(self):
        if opencl_backend.offloading:
            if not self.is_available_on_device():
                self.ensure_availability_on_device()

            return (self.opencl_layers_array.data,)
        else:
            return super(ExtrudedSet, self)._kernel_args_


class Subset(BaseSubset):
    """
    Subset for OpenCL.
    """
    def __init__(self, *args, **kwargs):
        super(Subset, self).__init__(*args, **kwargs)
        self._availability_flag = AVAILABLE_ON_HOST_ONLY

    def get_availability(self):
        return self._availability_flag

    @utils.cached_property
    def _opencl_indices(self):
        self._availability_flag = AVAILABLE_ON_BOTH
        return cla.to_device(opencl_backend.queue, self._indices)

    def ensure_availability_on_device(self):
        self._opencl_indices

    def ensure_availability_on_host(self):
        # Subset once initialized is not over-written so always available
        # on host.
        pass

    @property
    def _kernel_args_(self):
        if opencl_backend.offloading:
            if not self.is_available_on_device():
                self.ensure_availability_on_device()

            return (self._opencl_indices.data,)
        else:
            return super(Subset, self)._kernel_args_


class Dat(BaseDat):
    """
    Dat for OpenCL.
    """
    def __init__(self, *args, **kwargs):
        super(Dat, self).__init__(*args, **kwargs)
        # _availability_flag: only used when Dat cannot be represented as a
        # petscvec; when Dat can be represented as a petscvec the availability
        # flag is directly read from the petsc vec.
        self._availability_flag = AVAILABLE_ON_HOST_ONLY

    @utils.cached_property
    def _opencl_data(self):
        """
        Only used when the Dat's data cannot be represented as a petsc Vec.
        """
        self._availability_flag = AVAILABLE_ON_BOTH
        if self.can_be_represented_as_petscvec:
            with self.vec as petscvec:
                return cla.Array(cq=opencl_backend.queue,
                                 shape=self._data.shape,
                                 dtype=self._data.dtype,
                                 data=cl.Buffer.from_int_ptr(
                                     petscvec.getCLMemHandle("r"),
                                     retain=False),
                                 strides=self._data.strides)
        else:
            return cla.to_device(opencl_backend.queue, self._data)

    def zero(self, subset=None):
        if subset is None:
            self.data[:] = 0
            self.halo_valid = True
        else:
            raise NotImplementedError

    def copy(self, other, subset=None):
        raise NotImplementedError

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
        return PETSc.Vec().createViennaCLWithArrays(data, size=size,
                                                    bsize=self.cdim,
                                                    comm=self.comm)

    def get_availability(self):
        if self.can_be_represented_as_petscvec:
            return DataAvailability(self._vec.getOffloadMask())
        else:
            return self._availability_flag

    def ensure_availability_on_device(self):
        if self.can_be_represented_as_petscvec:
            if not opencl_backend.offloading:
                raise NotImplementedError("PETSc limitation: can ensure availability"
                                          " on GPU only within an offloading"
                                          " context.")
            # perform a host->device transfer if needed
            self._vec.getCLMemHandle("r")
        else:
            if not self.is_available_on_device():
                self._opencl_data.set(self._data)
            self._availability_flag = AVAILABLE_ON_BOTH

    def ensure_availability_on_host(self):
        if self.can_be_represented_as_petscvec:
            # perform a device->host transfer if needed
            self._vec.getArray(readonly=True)
        else:
            if not self.is_available_on_host():
                self._opencl_data.get(opencl_backend.queue, self._data)
            self._availability_flag = AVAILABLE_ON_BOTH

    @contextmanager
    def vec_context(self, access):
        r"""A context manager for a :class:`PETSc.Vec` from a :class:`Dat`.

        :param access: Access descriptor: READ, WRITE, or RW."""
        # PETSc Vecs have a state counter and cache norm computations
        # to return immediately if the state counter is unchanged.
        # Since we've updated the data behind their back, we need to
        # change that state counter.
        self._vec.stateIncrease()

        if opencl_backend.offloading:
            self.ensure_availability_on_device()
            self._vec.bindToCPU(False)
        else:
            self.ensure_availability_on_host()
            self._vec.bindToCPU(True)

        yield self._vec

        if access is not READ:
            self.halo_valid = False

    @property
    def _kernel_args_(self):
        if self.can_be_represented_as_petscvec:
            if opencl_backend.offloading:
                self.ensure_availability_on_device()
                # tell petsc that we have updated the data in the CL buffer
                with self.vec as v:
                    v.getCLMemHandle()
                    v.restoreCLMemHandle()

                return (self._opencl_data.data,)
            else:
                self.ensure_availability_on_host()
                # tell petsc that we have updated the data on the host
                with self.vec as v:
                    v.stateIncrease()
                return (self._data.ctypes.data, )
        else:
            if opencl_backend.offloading:
                self.ensure_availability_on_device()

                self._availability_flag = AVAILABLE_ON_DEVICE_ONLY
                return (self._opencl_data.data, )
            else:
                self.ensure_availability_on_host()

                self._availability_flag = AVAILABLE_ON_HOST_ONLY
                return (self._data.ctypes.data, )

    @mpi.collective
    @property
    def data(self):
        if self.dataset.total_size > 0 and self._data.size == 0 and self.cdim > 0:
            raise RuntimeError("Illegal access: no data associated with this Dat!")

        self.halo_valid = False

        if opencl_backend.offloading:

            self.ensure_availability_on_device()

            # {{{ marking data on the host as invalid

            if self.can_be_represented_as_petscvec:
                # report to petsc that we are updating values on the device
                with self.vec as petscvec:
                    petscvec.getCLMemHandle("w")
                    petscvec.restoreCLMemHandle()
            else:
                self._availability_flag = AVAILABLE_ON_DEVICE_ONLY

            # }}}
            v = self._opencl_data[:self.dataset.size]
        else:
            self.ensure_availability_on_host()
            v = self._data[:self.dataset.size].view()
            v.setflags(write=True)

            # {{{ marking data on the device as invalid

            if self.can_be_represented_as_petscvec:
                # report to petsc that we are altering data on the CPU
                with self.vec as petscvec:
                    petscvec.stateIncrease()
            else:
                self._availability_flag = AVAILABLE_ON_HOST_ONLY

            # }}}

        return v

    @property
    @mpi.collective
    def data_with_halos(self):
        self.global_to_local_begin(RW)
        self.global_to_local_end(RW)
        return self.data

    @property
    @mpi.collective
    def data_ro(self):

        if self.dataset.total_size > 0 and self._data.size == 0 and self.cdim > 0:
            raise RuntimeError("Illegal access: no data associated with this Dat!")

        self.halo_valid = False

        if opencl_backend.offloading:
            self.ensure_availability_on_device()
            v = self._opencl_data[:self.dataset.size].view()
            v.setitem = read_only_clarray_setitem
        else:
            self.ensure_availability_on_host()
            v = self._data[:self.dataset.size].view()
            v.setflags(write=False)

        return v

    @property
    @mpi.collective
    def data_ro_with_halos(self):
        self.global_to_local_begin(READ)
        self.global_to_local_end(READ)

        if opencl_backend.offloading:
            self.ensure_availability_on_device()
            v = self._opencl_data.view()
            v.setitem = read_only_clarray_setitem
        else:
            self.ensure_availability_on_host()
            v = self._data.view()
            v.setflags(write=False)

        return v


class Global(BaseGlobal):
    """
    Global for OpenCL.
    """

    def __init__(self, *args, **kwargs):
        super(Global, self).__init__(*args, **kwargs)
        self._availability_flag = AVAILABLE_ON_HOST_ONLY

    @utils.cached_property
    def _opencl_data(self):
        self._availability_flag = AVAILABLE_ON_BOTH
        return cla.to_device(opencl_backend.queue, self._data)

    def get_availability(self):
        return self._availability_flag

    def ensure_availability_on_device(self):
        if not self.is_available_on_device():
            self._opencl_data.set(ary=self._data,
                                  queue=opencl_backend.queue)
            self._availability_flag = AVAILABLE_ON_BOTH

    def ensure_availability_on_host(self):
        if not self.is_available_on_host():
            self._opencl_data.get(ary=self._data,
                                  queue=opencl_backend.queue)
            self._availability_flag = AVAILABLE_ON_BOTH

    @property
    def _kernel_args_(self):
        if opencl_backend.offloading:
            self.ensure_availability_on_device()
            self._availability_flag = AVAILABLE_ON_DEVICE_ONLY
            return (self._opencl_data.data,)
        else:
            self.ensure_availability_on_host()
            self._availability_flag = AVAILABLE_ON_HOST_ONLY
            return super(Global, self)._kernel_args_

    @mpi.collective
    @property
    def data(self):
        if len(self._data) == 0:
            raise RuntimeError("Illegal access: No data associated with"
                               " this Global!")

        if opencl_backend.offloading:
            self.ensure_availability_on_device()
            self._availability_flag = AVAILABLE_ON_DEVICE_ONLY
            return self._opencl_data
        else:
            self.ensure_availability_on_host()
            self._availability_flag = AVAILABLE_ON_HOST_ONLY
            return self._data

    @property
    def data_ro(self):
        if len(self._data) == 0:
            raise RuntimeError("Illegal access: No data associated with"
                               " this Global!")

        if opencl_backend.offloading:
            self.ensure_availability_on_device()
            view = self._opencl_data.view()
            view.setitem = read_only_clarray_setitem
            return view
        else:
            self.ensure_availability_on_host()
            view = self._data.view()
            view.setflags(write=False)
            return view

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
            return PETSc.Vec().createViennaCLWithArrays(
                data,
                size=size,
                bsize=self.cdim,
                comm=self.comm
            )
        else:
            return PETSc.Vec().createViennaCLWithArrays(
                numpy.empty(0, dtype=self.dtype),
                size=size,
                bsize=self.cdim,
                comm=self.comm
            )

    @contextmanager
    def vec_context(self, access):
        """A context manager for a :class:`PETSc.Vec` from a :class:`Global`.

        :param access: Access descriptor: READ, WRITE, or RW."""
        yield self._vec
        if access is not READ:
            data = self._data
            self.comm.Bcast(data, 0)


@dataclass(frozen=True)
class CLKernelWithExtraArgs:
    cl_kernel: cl.Kernel
    args_to_append: Tuple[cla.Array, ...]

    def __call__(self, grid, block, start, end, *args):
        if all(grid + block):
            self.cl_kernel(opencl_backend.queue,
                           grid, block,
                           numpy.int32(start),
                           numpy.int32(end),
                           *args,
                           *[arg_to_append.data
                             for arg_to_append in self.args_to_append],
                           g_times_l=True)


class GlobalKernel(AbstractGlobalKernel):
    @classmethod
    def _cache_key(cls, *args, **kwargs):
        key = super()._cache_key(*args, **kwargs)
        key = key + (configuration["gpu_strategy"], "opencl")
        return key

    @utils.cached_property
    def encoded_cache_key(self):
        return md5(str(self.cache_key[1:]).encode()).hexdigest()

    @utils.cached_property
    def computational_grid_expr_cache_file_path(self):
        cachedir = configuration["cache_dir"]
        return os.path.join(cachedir, f"{self.encoded_cache_key}_grid_params.py")

    @utils.cached_property
    def extra_args_cache_file_path(self):
        cachedir = configuration["cache_dir"]
        return os.path.join(cachedir, f"{self.encoded_cache_key}_extra_args.npz")

    @memoize_method
    def get_grid_size(self, start, end):
        fpath = self.computational_grid_expr_cache_file_path

        with open(fpath, "r") as f:
            globals_dict = {}
            exec(f.read(), globals_dict)
            get_grid_sizes = globals_dict["get_grid_sizes"]

        return get_grid_sizes(start=start, end=end)

    @memoize_method
    def get_extra_args(self):
        """
        Returns device buffers corresponding to array literals baked into
        :attr:`local_kernel`.
        """
        fpath = self.extra_args_cache_file_path
        npzfile = numpy.load(fpath)
        assert npzfile["ids"].ndim == 1
        assert len(npzfile.files) == len(npzfile["ids"]) + 1
        extra_args_np = [npzfile[arg_id]
                         for arg_id in npzfile["ids"]]
        return tuple(cla.to_device(opencl_backend.queue, arg)
                     for arg in extra_args_np)

    @utils.cached_property
    def argtypes(self):
        result = super().argtypes
        return result + (ctypes.c_voidp,) * len(self.get_extra_args())

    @utils.cached_property
    def code_to_compile(self):
        from pyop2.codegen.rep2loopy import generate
        from pyop2.transforms.gpu_utils import apply_gpu_transforms
        from pymbolic.interop.ast import to_evaluatable_python_function

        t_unit = generate(self.builder,
                          include_math=False,
                          include_petsc=False,
                          include_complex=False)

        # Make temporary variables with initializers kernel's arguments.
        t_unit, extra_args = apply_gpu_transforms(t_unit, "opencl")

        ary_ids = [f"_op2_arg_{i}"
                   for i in range(len(extra_args))]

        numpy.savez(self.extra_args_cache_file_path,
                    ids=numpy.array(ary_ids),
                    **{ary_id: extra_arg
                       for ary_id, extra_arg in zip(ary_ids, extra_args)})

        # {{{ save python code to get grid sizes

        with open(self.computational_grid_expr_cache_file_path, "w") as f:
            glens, llens = (t_unit
                            .default_entrypoint
                            .get_grid_size_upper_bounds_as_exprs(t_unit
                                                                 .callables_table))
            f.write(to_evaluatable_python_function((glens, llens), "get_grid_sizes"))

        code = lp.generate_code_v2(t_unit).device_code()

        # }}}

        return code

    @mpi.collective
    def __call__(self, comm, *args):
        key = id(comm)
        try:
            func = self._func_cache[key]
        except KeyError:
            func = self.compile(comm)
            self._func_cache[key] = func

        grid, block = self.get_grid_size(args[0], args[1])
        func(grid, block, *args)

    @mpi.collective
    def compile(self, comm):
        cl_knl = compilation.get_opencl_kernel(comm,
                                               self)
        return CLKernelWithExtraArgs(cl_knl, self.get_extra_args())


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
                glob.ensure_availability_on_host()
                requests.append(self.comm.Iallreduce(glob._data,
                                                     glob._buf,
                                                     op=mpi_op))
            else:
                glob.ensure_availability_on_host()
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
                glob._availability_flag = AVAILABLE_ON_HOST_ONLY
                glob.ensure_availability_on_device()
        else:
            assert len(requests) == 0

            for idx in self._reduction_idxs:
                glob = self.arguments[idx].data
                glob._data[:] = glob._buf
                glob._availability_flag = AVAILABLE_ON_HOST_ONLY
                glob.ensure_availability_on_device()


class OpenCLBackend(AbstractComputeBackend):
    Parloop_offloading = Parloop
    Parloop_no_offloading = cpu_backend.Parloop
    GlobalKernel_offloading = GlobalKernel
    GlobalKernel_no_offloading = cpu_backend.GlobalKernel

    Parloop = cpu_backend.Parloop
    GlobalKernel = cpu_backend.GlobalKernel
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
    PETScVecType = PETSc.Vec.Type.VIENNACL

    def __init__(self):
        self.offloading = False

    @utils.cached_property
    def context(self):
        # create a dummy vector and extract its underlying context
        x = PETSc.Vec().create(PETSc.COMM_WORLD)
        x.setType("viennacl")
        x.setSizes(size=1)
        ctx_ptr = x.getCLContextHandle()
        return cl.Context.from_int_ptr(ctx_ptr, retain=False)

    @utils.cached_property
    def queue(self):
        # TODO: Instruct the user to pass
        # -viennacl_backend opencl
        # -viennacl_opencl_device_type gpu
        # create a dummy vector and extract its associated command queue
        x = PETSc.Vec().create(PETSc.COMM_WORLD)
        x.setType("viennacl")
        x.setSizes(size=1)
        queue_ptr = x.getCLQueueHandle()
        return cl.CommandQueue.from_int_ptr(queue_ptr, retain=False)

    def turn_on_offloading(self):
        self.offloading = True
        self.Parloop = self.Parloop_offloading
        self.GlobalKernel = self.GlobalKernel_offloading

    def turn_off_offloading(self):
        self.offloading = False
        self.Parloop = self.Parloop_no_offloading
        self.GlobalKernel = self.GlobalKernel_no_offloading

    @property
    def cache_key(self):
        return (type(self), self.offloading)

    def array(self, *args, **kwargs):
        return cla.to_device(self.queue, *args, **kwargs)

    @staticmethod
    def zeros(*args, **kwargs):
        return cla.zeros(*args, **kwargs)


opencl_backend = OpenCLBackend()
