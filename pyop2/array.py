import abc
import collections
import enum

import numpy as np
from petsc4py import PETSc

from pyop2.configuration import configuration
from pyop2.datatypes import ScalarType
from pyop2.exceptions import DataTypeError
from pyop2.offload_utils import _offloading as offloading, _backend as backend, OffloadingBackend
from pyop2.types.access import READ, RW, WRITE, INC, MIN, MAX
from pyop2.utils import verify_reshape

try:
    import pyopencl
except ImportError:
    pass

try:
    import pycuda
except ImportError:
    pass


class MirroredArray(abc.ABC):
    """An array that is available on both host and device, copying values where necessary."""

    # TODO: Use __new__ instead?
    @classmethod
    def new(cls, *args, **kwargs):
        if backend == OffloadingBackend.CPU:
            return CPUOnlyArray(*args, **kwargs)
        elif backend == OffloadingBackend.OPENCL:
            return OpenCLArray(*args, **kwargs)
        elif backend == OffloadingBackend.CUDA:
            return CUDAArray(*args, **kwargs)
        else:
            raise ValueError

    def __init__(self, data):
        # option 1: passed shape and dtype
        if (len(data) == 2 and isinstance(data[0], collections.abc.Collection)
                and isinstance(data[1], np.dtype)):
            self.shape = data[0]
            self.dtype = data[1]
            self._lazy_host_data = None
        # option 2: passed an array
        elif isinstance(data, np.ndarray):
            self.shape = data.shape
            self.dtype = data.dtype
            self._lazy_host_data = data
        else:
            raise ValueError("Unexpected arguments encountered")

        # for now assume we are only ever passed numpy arrays (not device arrays)
        self._lazy_device_data = None
        self.is_available_on_host = True
        self.is_available_on_device = False

        # counter used to keep track of modifications
        self.state = 0

    @property
    def data(self):
        return self.device_data if offloading else self.host_data

    @property
    def data_ro(self):
        return self.device_data_ro if offloading else self.host_data_ro

    @property
    def host_data(self):
        return self.host_data_rw

    @property
    def host_data_rw(self):
        self.state += 1

        if not self.is_available_on_host:
            self.device_to_host_copy()
            self.is_available_on_host = True

        # modifying on host invalidates the device
        self.is_available_on_device = False

        data = self._host_data.view()
        data.setflags(write=True)
        return data

    @property
    def host_data_ro(self):
        if not self.is_available_on_host:
            self.device_to_host_copy()
            self.is_available_on_host = True

        data = self._host_data.view()
        data.setflags(write=False)
        return data

    @property
    def host_data_wo(self):
        self.state += 1  # TODO make a decorator

        # if we are writing then we don't need to copy from the device
        pass

        # modifying on host invalidates the device
        self.is_available_on_host = True
        self.is_available_on_device = False

        data = self._host_data.view()
        data.setflags(write=True)
        return data

    @property
    def device_data(self):
        return self.device_data_rw

    @property
    @abc.abstractmethod
    def device_data_rw(self):
        pass

    @property
    @abc.abstractmethod
    def device_data_ro(self):
        pass

    @property
    @abc.abstractmethod
    def device_data_wo(self):
        pass

    @property
    def ptr_rw(self):
        return self.device_ptr_rw if offloading else self.host_ptr_rw

    @property
    def ptr_ro(self):
        return self.device_ptr_ro if offloading else self.host_ptr_ro

    @property
    def ptr_wo(self):
        return self.device_ptr_wo if offloading else self.host_ptr_wo

    @property
    def host_ptr_rw(self):
        return self.host_data_rw.ctypes.data

    @property
    def host_ptr_ro(self):
        return self.host_data_ro.ctypes.data

    @property
    def host_ptr_wo(self):
        return self.host_data_wo.ctypes.data

    @property
    @abc.abstractmethod
    def device_ptr_rw(self):
        pass

    @property
    @abc.abstractmethod
    def device_ptr_ro(self):
        pass

    @property
    @abc.abstractmethod
    def device_ptr_wo(self):
        pass

    @abc.abstractmethod
    def host_to_device_copy(self):
        pass

    @abc.abstractmethod
    def device_to_host_copy(self):
        pass

    @abc.abstractmethod
    def alloc_device_data(self, shape, dtype):
        pass

    @property
    def size(self):
        return self.data_ro.size

    @property
    def _host_data(self):
        if self._lazy_host_data is None:
            self._lazy_host_data = np.zeros(self.shape, self.dtype)
        return self._lazy_host_data

    @property
    def _device_data(self):
        if self._lazy_device_data is None:
            self._lazy_device_data = self.alloc_device_data(self.shape, self.dtype)
        return self._lazy_device_data


class CPUOnlyArray(MirroredArray):
    """TODO"""

    @property
    def device_data_rw(self):
        return self.host_data_rw

    @property
    def device_data_ro(self):
        return self.host_data_ro

    @property
    def device_data_wo(self):
        return self.host_data_wo

    @property
    def device_ptr_rw(self):
        return self.host_ptr_rw

    @property
    def device_ptr_ro(self):
        return self.host_ptr_ro

    @property
    def device_ptr_wo(self):
        return self.host_ptr_wo

    def host_to_device_copy(self):
        pass

    def device_to_host_copy(self):
        pass

    def alloc_device_data(self, shape, dtype):
        raise AssertionError("Should not hit this")


class OpenCLArray(MirroredArray):

    @property
    def device_data_rw(self):
        raise NotImplementedError

    @property
    def device_data_ro(self):
        raise NotImplementedError

    @property
    def device_data_wo(self):
        raise NotImplementedError

    @property
    def device_ptr_rw(self):
        raise NotImplementedError

    @property
    def device_ptr_ro(self):
        raise NotImplementedError

    @property
    def device_ptr_wo(self):
        raise NotImplementedError

    def host_to_device_copy(self):
        self._device_data.set(self._host_data)

    def device_to_host_copy(self):
        # FIXME: self.queue not a thing
        self._device_data.get(self.queue, self._host_data)

    def alloc_device_data(self, shape, dtype):
        raise NotImplementedError("TODO")
