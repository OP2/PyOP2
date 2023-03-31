from contextlib import contextmanager
import enum


class OffloadingBackend(enum.Enum):
    """TODO"""
    CPU = enum.auto()
    OPENCL = enum.auto()
    CUDA = enum.auto()


_offloading = False
"""Global state indicating whether or not we are running on the host or device"""

_backend = OffloadingBackend.CPU
"""TODO"""


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
    global _backend

    if not isinstance(backend, OffloadingBackend):
        raise TypeError("TODO")
    _backend = backend


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
