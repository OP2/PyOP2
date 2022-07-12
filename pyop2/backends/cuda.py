"""OP2 CUDA backend."""

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
import pycuda.driver as cuda
import pycuda.gpuarray as cuda_np
import loopy as lp
from pytools import memoize_method
from dataclasses import dataclass
from typing import Tuple
import ctypes


class Map(BaseMap):

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        self._availability_flag = AVAILABLE_ON_HOST_ONLY

    @utils.cached_property
    def _cuda_values(self):
        self._availability_flag = AVAILABLE_ON_BOTH
        return cuda_np.to_gpu(self._values)

    def get_availability(self):
        return self._availability_flag

    def ensure_availability_on_device(self):
        self._cuda_values

    def ensure_availability_on_host(self):
        # Map once initialized is not over-written so always available
        # on host.
        pass

    @property
    def _kernel_args_(self):
        if cuda_backend.offloading:
            if not self.is_available_on_device():
                self.ensure_availability_on_device()

            return (self._cuda_values.gpudata,)
        else:
            return super(Map, self)._kernel_args_


class ExtrudedSet(BaseExtrudedSet):
    """
    ExtrudedSet for CUDA.
    """

    def __init__(self, *args, **kwargs):
        super(ExtrudedSet, self).__init__(*args, **kwargs)
        self._availability_flag = AVAILABLE_ON_HOST_ONLY

    @utils.cached_property
    def cuda_layers_array(self):
        self._availability_flag = AVAILABLE_ON_BOTH
        return cuda_np.to_gpu(self.layers_array)

    def get_availability(self):
        return self._availability_flag

    def ensure_availability_on_device(self):
        self.cuda_layers_array

    def ensure_availability_on_host(self):
        # ExtrudedSet once initialized is not over-written so always available
        # on host.
        pass

    @property
    def _kernel_args_(self):
        if cuda_backend.offloading:
            if not self.is_available_on_device():
                self.ensure_availability_on_device()

            return (self.cuda_layers_array.gpudata,)
        else:
            return super(ExtrudedSet, self)._kernel_args_


class Subset(BaseSubset):
    """
    Subset for CUDA.
    """
    def __init__(self, *args, **kwargs):
        super(Subset, self).__init__(*args, **kwargs)
        self._availability_flag = AVAILABLE_ON_HOST_ONLY

    def get_availability(self):
        return self._availability_flag

    @utils.cached_property
    def _cuda_indices(self):
        self._availability_flag = AVAILABLE_ON_BOTH
        return cuda_np.to_gpu(self._indices)

    def ensure_availability_on_device(self):
        self._cuda_indices

    def ensure_availability_on_host(self):
        # Subset once initialized is not over-written so always available
        # on host.
        pass

    @property
    def _kernel_args_(self):
        if cuda_backend.offloading:
            if not self.is_available_on_device():
                self.ensure_availability_on_device()

            return (self._cuda_indices.gpudata,)
        else:
            return super(Subset, self)._kernel_args_


class Dat(BaseDat):
    """
    Dat for CUDA.
    """
    def __init__(self, *args, **kwargs):
        super(Dat, self).__init__(*args, **kwargs)
        # _availability_flag: only used when Dat cannot be represented as a
        # petscvec; when Dat can be represented as a petscvec the availability
        # flag is directly read from the petsc vec.
        self._availability_flag = AVAILABLE_ON_HOST_ONLY

    @utils.cached_property
    def _cuda_data(self):
        """
        Only used when the Dat's data cannot be represented as a petsc Vec.
        """
        self._availability_flag = AVAILABLE_ON_BOTH
        if self.can_be_represented_as_petscvec:
            with self.vec as petscvec:
                return cuda_np.GPUArray(shape=self._data.shape,
                                        dtype=self._data.dtype,
                                        gpudata=petscvec.getCUDAHandle("r"),
                                        strides=self._data.strides)
        else:
            return cuda_np.to_gpu(self._data)

    def zero(self, subset=None):
        if subset is None:
            self.data[:] = 0*self.data
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
        return PETSc.Vec().createCUDAWithArrays(data, size=size,
                                                bsize=self.cdim,
                                                comm=self.comm)

    def get_availability(self):
        if self.can_be_represented_as_petscvec:
            return DataAvailability(self._vec.getOffloadMask())
        else:
            return self._availability_flag

    def ensure_availability_on_device(self):
        if self.can_be_represented_as_petscvec:
            if not cuda_backend.offloading:
                raise NotImplementedError("PETSc limitation: can ensure availability"
                                          " on GPU only within an offloading"
                                          " context.")
            # perform a host->device transfer if needed
            self._vec.getCUDAHandle("r")
        else:
            if not self.is_available_on_device():
                self._cuda_data.set(self._data)
            self._availability_flag = AVAILABLE_ON_BOTH

    def ensure_availability_on_host(self):
        if self.can_be_represented_as_petscvec:
            # perform a device->host transfer if needed
            self._vec.getArray(readonly=True)
        else:
            if not self.is_available_on_host():
                self._cuda_data.get(self._data)
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

        if cuda_backend.offloading:
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
            if cuda_backend.offloading:
                self.ensure_availability_on_device()
                # tell petsc that we have updated the data in the CL buffer
                with self.vec as v:
                    v.restoreCUDAHandle(v.getCUDAHandle())
                return (self._cuda_data.gpudata,)
            else:
                self.ensure_availability_on_host()
                # tell petsc that we have updated the data on the host
                with self.vec as v:
                    v.stateIncrease()
                return (self._data.ctypes.data, )
        else:
            if cuda_backend.offloading:
                self.ensure_availability_on_device()

                self._availability_flag = AVAILABLE_ON_DEVICE_ONLY
                return (self._cuda_data.gpudata, )
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

        if cuda_backend.offloading:

            self.ensure_availability_on_device()

            # {{{ marking data on the host as invalid

            if self.can_be_represented_as_petscvec:
                # report to petsc that we are updating values on the device
                with self.vec as petscvec:
                    petscvec.restoreCUDAHandle(petscvec.getCUDAHandle("w"))
            else:
                self._availability_flag = AVAILABLE_ON_DEVICE_ONLY

            # }}}
            v = self._cuda_data[:self.dataset.size]
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

        if cuda_backend.offloading:
            self.ensure_availability_on_device()
            v = self._cuda_data[:self.dataset.size].view()
            # FIXME: PyCUDA doesn't support 'read-only' arrays yet
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

        if cuda_backend.offloading:
            self.ensure_availability_on_device()
            v = self._cuda_data.view()
            # FIXME: PyCUDA doesn't support 'read-only' arrays yet
        else:
            self.ensure_availability_on_host()
            v = self._data.view()
            v.setflags(write=False)

        return v


class Global(BaseGlobal):
    """
    Global for CUDA.
    """

    def __init__(self, *args, **kwargs):
        super(Global, self).__init__(*args, **kwargs)
        self._availability_flag = AVAILABLE_ON_HOST_ONLY

    @utils.cached_property
    def _cuda_data(self):
        self._availability_flag = AVAILABLE_ON_BOTH
        return cuda_np.to_gpu(self._data)

    def get_availability(self):
        return self._availability_flag

    def ensure_availability_on_device(self):
        if not self.is_available_on_device():
            self._cuda_data.set(self._data)
            self._availability_flag = AVAILABLE_ON_BOTH

    def ensure_availability_on_host(self):
        if not self.is_available_on_host():
            self._cuda_data.get(ary=self._data)
            self._availability_flag = AVAILABLE_ON_BOTH

    @property
    def _kernel_args_(self):
        if cuda_backend.offloading:
            self.ensure_availability_on_device()
            self._availability_flag = AVAILABLE_ON_DEVICE_ONLY
            return (self._cuda_data.gpudata,)
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

        if cuda_backend.offloading:
            self.ensure_availability_on_device()
            self._availability_flag = AVAILABLE_ON_DEVICE_ONLY
            return self._cuda_data
        else:
            self.ensure_availability_on_host()
            self._availability_flag = AVAILABLE_ON_HOST_ONLY
            return self._data

    @property
    def data_ro(self):
        if len(self._data) == 0:
            raise RuntimeError("Illegal access: No data associated with"
                               " this Global!")

        if cuda_backend.offloading:
            self.ensure_availability_on_device()
            view = self._cuda_data.view()
            # FIXME: PyCUDA doesn't support read-only arrays yet
            return view
        else:
            self.ensure_availability_on_host()
            view = self._data.view()
            view.setflags(write=False)
            return view

    @utils.cached_property
    def _vec(self):
        raise NotImplementedError()

    @contextmanager
    def vec_context(self, access):
        raise NotImplementedError()


@dataclass(frozen=True)
class CUFunctionWithExtraArgs:
    """
    A partial :class:`pycuda.driver.Function` with bound arguments *args_to_append*
    that are appended to the arguments passed in :meth:`__call__`.
    """
    cu_func: cuda.Function
    args_to_append: Tuple[cuda_np.GPUArray, ...]

    def __call__(self, grid, block, *args):
        if all(grid + block):
            self.cu_func.prepared_call(grid, block,
                                       *args,
                                       *[arg_to_append.gpudata
                                         for arg_to_append in self.args_to_append])


class GlobalKernel(AbstractGlobalKernel):
    @classmethod
    def _cache_key(cls, *args, **kwargs):
        key = super()._cache_key(*args, **kwargs)
        key = key + (configuration["gpu_strategy"], "cuda")
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
        """
        Returns a :class:`tuple` of the form ``(grid, block)``, where *grid* is
        the 2-:class:`tuple` corresponding to the CUDA computational grid size
        and ``block`` is the thread block size. Refer to
        `CUDA docs <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model>`_
        for a detailed explanation of these terms.
        """  # noqa: E501
        fpath = self.computational_grid_expr_cache_file_path

        with open(fpath, "r") as f:
            globals_dict = {}
            exec(f.read(), globals_dict)
            get_grid_sizes = globals_dict["get_grid_sizes"]

        grid, block = get_grid_sizes(start=start, end=end)

        # TODO: PyCUDA doesn't allow numpy int's for grid dimensions. Remove
        # these casts after PyCUDA has been patched.
        return (tuple(int(dim) for dim in grid),
                tuple(int(dim) for dim in block))

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

        return tuple([cuda_np.to_gpu(arg_np) for arg_np in extra_args_np])

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
        t_unit, extra_args = apply_gpu_transforms(t_unit, "cuda")

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
            glens = glens + (1,) * (2 - len(glens))  # cuda expects 2d grid shape
            llens = llens + (1,) * (3 - len(llens))  # cuda expects 3d block shape
            f.write(to_evaluatable_python_function((glens, llens), "get_grid_sizes"))

        # }}}

        code = lp.generate_code_v2(t_unit).device_code()

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
        cu_func = compilation.get_prepared_cuda_function(comm,
                                                         self)
        return CUFunctionWithExtraArgs(cu_func, self.get_extra_args())


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
                # FIXME: Get rid of this D2H for CUDA-Aware MPI.
                glob.ensure_availability_on_host()
                requests.append(self.comm.Iallreduce(glob._data,
                                                     glob._buf,
                                                     op=mpi_op))
            else:
                # FIXME: Get rid of this D2H for CUDA-Aware MPI.
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
                # FIXME: Get rid of this H2D for CUDA-Aware MPI.
                glob.ensure_availability_on_device()
        else:
            assert len(requests) == 0

            for idx in self._reduction_idxs:
                glob = self.arguments[idx].data
                glob._data[:] = glob._buf
                glob._availability_flag = AVAILABLE_ON_HOST_ONLY
                # FIXME: Get rid of this H2D for CUDA-Aware MPI.
                glob.ensure_availability_on_device()


class CUDABackend(AbstractComputeBackend):
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
    PETScVecType = PETSc.Vec.Type.CUDA

    def __init__(self):
        self.offloading = False

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


cuda_backend = CUDABackend()
