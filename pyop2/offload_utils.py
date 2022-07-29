from enum import IntEnum
from contextlib import contextmanager


_offloading = False
"""Global state indicating whether or not we are running on the host or device"""


class DataAvailability(IntEnum):
    """
    Indicates whether the device or host contains valid data.
    """
    AVAILABLE_ON_HOST_ONLY = 1
    AVAILABLE_ON_DEVICE_ONLY = 2
    AVAILABLE_ON_BOTH = 3


class OffloadMixin:
    def get_availability(self):
        raise NotImplementedError

    def ensure_availability_on_host(self):
        raise NotImplementedError

    def ensure_availaibility_on_device(self):
        raise NotImplementedError

    def is_available_on_host(self):
        # bitwise op to detect both AVAILABLE_ON_HOST and AVAILABLE_ON_BOTH
        return bool(self.get_availability() & AVAILABLE_ON_HOST_ONLY)

    def is_available_on_device(self):
        # bitwise op to detect both AVAILABLE_ON_DEVICE and AVAILABLE_ON_BOTH
        return bool(self.get_availability() & AVAILABLE_ON_DEVICE_ONLY)


AVAILABLE_ON_HOST_ONLY = DataAvailability.AVAILABLE_ON_HOST_ONLY
AVAILABLE_ON_DEVICE_ONLY = DataAvailability.AVAILABLE_ON_DEVICE_ONLY
AVAILABLE_ON_BOTH = DataAvailability.AVAILABLE_ON_BOTH


def set_offloading_backend(backend):
    """
    Sets a backend for offloading operations to.

    By default the operations are executed on the host, to mark operations for
    offloading wrap them within a :func:`~pyop2.offload_utils.offloading`
    context.

    :arg backend: An instance of :class:`pyop2.backends.AbstractComputeBackend`.

    .. warning::

        * Must be called before any instance of PyOP2 type is allocated.
        * Calling :func:`set_offloading_bacckend` different values of *backend*
          over the course of the program is an undefined behavior. (i.e.
          preferably avoided)
    """
    from pyop2 import op2
    from pyop2.backends import AbstractComputeBackend
    assert isinstance(backend, AbstractComputeBackend)
    op2.compute_backend = backend


@contextmanager
def offloading():
    """
    Operations (such as manipulating a :class:`~pyop2.types.Dat`, calling a
    :class:`~pyop2.global_kernel.GlobalKernel`, etc) within the offloading
    region will be executed on backend as selected via
    :func:`set_offloading_backend`.
    """
    global _offloading

    _offloading = True
    yield
    _offloading = False
    return
