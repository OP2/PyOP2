class Status(enum.Enum):
    """where is the up-to-date version?"""
    ON_HOST = enum.auto()
    ON_DEVICE = enum.auto()
    ON_BOTH = enum.auto()


"""
e.g. 1

d = Dat(...)

!# this will create a numpy array of zeros

e.g. 2:

with offloading():
    d = Dat(...)

!# this will create a *backend* array of zeros

e.g. 3:

parloop(..., target=DEVICE)
"""


class Array:
    """An array that is available on both host and device, copying values where necessary."""
    def __init__(self, shape, dtype):
        self.status = ON_HOST # could change with offloading ctxt mngr
        #self._host_data = None
        #self._dev_ptr = None
        self.initdata(offload)

    def initdata(self, offload=False):
        # offloat in {"cpu", "gpu"}
        if offload == "cpu":
            self._host_data = np.zeros(shape, dtype)
            self._dev_ptr = None
        elif offload == "gpu":
            self._host_data = None
            # this if statement is needed because backend is known at the
            # start of the program but offload, i.e. where are we currently, is not.
            if backend == "opencl":
                self._dev_ptr = ...  # some incantation or other
            elif backend == "cuda":
                ...

    def as_petsc_vec(self):
        if offload == "cpu":
            return PETSc.Vec().createMPI(...)
        elif offload == "gpu":
            if backend == "cuda":
                return PETSc.Vec().createCUDA(..)
            ...


    @property
    def data(self):
        """numpy array"""
        if self._host_data is None:
            ...

        if self.status == ON_DEVICE:
            self._device2hostcopy()
        self.status = ON_HOST
        return self._host_data  # (a numpy array), initially unallocated then lazily filled


    @property
    def host_ptr(self):
        return self.data.ctypes.data

    @property
    def dev_ptr_ro(self):
        # depend on cuda/openCL/notimplemented (defined at start)

        #if self.status not in {ON_DEVICE, ON_BOTH}:
        if self.status == ON_HOST:
            self._host2devicecopy()
        self.status = ON_BOTH
        return self._dev_ptr

    @property
    def dev_ptr_w(self):
        # depend on cuda/openCL/notimplemented (defined at start)

        #if self.status not in {ON_DEVICE, ON_BOTH}:
        if self.status == ON_HOST:
            self._host2devicecopy()
        self.status = ON_DEVICE
        return self._dev_ptr


class CudaVec(Vec):
    ...


class OpenCLVec(Vec):
    ...
    ...
