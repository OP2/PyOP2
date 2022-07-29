import enum

import numpy as np
import pyopencl
from petsc4py import PETSc

from pyop2.configuration import configuration
from pyop2.offload_utils import _offloading as offloading
from pyop2.types.access import READ, RW, WRITE, INC
from pyop2.utils import verify_reshape


class Status(enum.Enum):
    """where is the up-to-date version?"""
    ON_HOST = enum.auto()
    ON_DEVICE = enum.auto()
    ON_BOTH = enum.auto()


ON_HOST = Status.ON_HOST
ON_DEVICE = Status.ON_DEVICE
ON_BOTH = Status.ON_BOTH


class MirroredArray:
    """An array that is available on both host and device, copying values where necessary."""

    def __init__(self, data, dtype, shape):
        # TODO This could be allocated lazily
        self.dtype = np.dtype(dtype)
        if data is None:
            self._host_data = np.zeros(shape, dtype=dtype)
        else:
            self._host_data = verify_reshape(data, dtype, shape)
        self.availability = ON_BOTH

    @classmethod
    def new(cls, *args, **kwargs):
        if configuration["backend"] == "CPU_ONLY":
            return CPUOnlyArray(*args, **kwargs)
        elif configuration["backend"] == "OPENCL":
            return OpenCLArray(*args, **kwargs)
        else:
            raise ValueError

    @property
    def kernel_arg(self):
        # lazy but for now assume that the data is always modified if we access
        # the pointer
        return self.device_ptr if offloading else self.host_ptr

        # if offloading:
        #     # N.B. not considering MAX, MIN here
        #     if access in {RW, WRITE, INC}:
        #         return self.device_ptr
        #     else:
        #         return self.device_ptr_ro
        # else:
        #     # N.B. not considering MAX, MIN here
        #     if access in {RW, WRITE, INC}:
        #         return self.host_ptr
        #     else:
        #         return self.host_ptr_ro

    @property
    def is_available_on_device(self):
        return self.availability in {ON_DEVICE, ON_BOTH}

    def ensure_availability_on_device(self):
        if not self.is_available_on_device:
            self.host_to_device_copy()

    @property
    def is_available_on_host(self):
        return self.availability in {ON_HOST, ON_BOTH}

    def ensure_availability_on_host(self):
        if not self.is_available_on_host:
            self.device_to_host_copy()

    @property
    def vec(self):
        raise NotImplementedError
        # if offload == "cpu":
        #     return PETSc.Vec().createMPI(...)
        # elif offload == "gpu":
        #     if backend == "cuda":
        #         return PETSc.Vec().createCUDA(..)
        #     ...


    @property
    def data(self):
        self.ensure_availability_on_host()
        self.availability = ON_HOST
        v = self._host_data.view()
        v.setflags(write=True)
        return v

    @property
    def data_ro(self):
        self.ensure_availability_on_host()
        v = self._host_data.view()
        v.setflags(write=False)
        return v

    @property
    def host_ptr(self):
        self.ensure_availability_on_host()
        self.availability = ON_HOST
        return self.data.ctypes.data

    @property
    def host_ptr_ro(self):
        self.ensure_availability_on_host()
        return self.data.ctypes.data


class CPUOnlyArray(MirroredArray):

    @property
    def device_ptr(self):
        return self.host_ptr

    @property
    def device_ptr_ro(self):
        return self.host_ptr_ro

    def host_to_device_copy(self):
        pass

    def device_to_host_copy(self):
        pass


if configuration["backend"] == "OPENCL":
    # TODO: Instruct the user to pass
    # -viennacl_backend opencl
    # -viennacl_opencl_device_type gpu
    # create a dummy vector and extract its associated command queue
    x = PETSc.Vec().create(PETSc.COMM_WORLD)
    x.setType("viennacl")
    x.setSizes(size=1)
    queue_ptr = x.getCLQueueHandle()
    cl_queue = pyopencl.CommandQueue.from_int_ptr(queue_ptr, retain=False)


class OpenCLArray(MirroredArray):

    def __init__(self, data, shape, dtype):
        if data is None:
            self._device_data = pyopencl.array.empty(cl_queue, shape, dtype)
        else:
            self._device_data = pyopencl.array.to_device(cl_queue, data)

    @property
    def device_ptr(self):
        self.ensure_availability_on_device()
        self.availability = ON_DEVICE
        return self._device_data.data

    @property
    def device_ptr_ro(self):
        self.ensure_availability_on_device()
        return self._device_data.data

    def host_to_device_copy(self):
        self._device_data.set(self._host_data)

    def device_to_host_copy(self):
        self._device_data.get(self._host_data)
