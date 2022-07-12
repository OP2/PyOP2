class _not_implemented:  # noqa
    """Not Implemented"""


class AbstractComputeBackend:
    """
    Abstract class to record all the backend specific implementation of
    :mod:`pyop2`'s data structures.
    """
    GlobalKernel = _not_implemented
    Parloop = _not_implemented
    Set = _not_implemented
    ExtrudedSet = _not_implemented
    MixedSet = _not_implemented
    Subset = _not_implemented
    DataSet = _not_implemented
    MixedDataSet = _not_implemented
    Map = _not_implemented
    MixedMap = _not_implemented
    Dat = _not_implemented
    MixedDat = _not_implemented
    DatView = _not_implemented
    Mat = _not_implemented
    Global = _not_implemented
    GlobalDataSet = _not_implemented
    PETScVecType = _not_implemented

    def __getattribute__(self, key):
        val = super().__getattribute__(key)
        if val is _not_implemented:
            raise NotImplementedError(f"'{val}' is not implemented for backend"
                                      f" '{type(self).__name__}'.")
        return val

    def turn_on_offloading(self):
        raise NotImplementedError()

    def turn_off_offloading(self):
        raise NotImplementedError()

    @property
    def cache_key(self):
        raise NotImplementedError()
