class _not_implemented:  # noqa
    """Not Implemented"""


class AbstractComputeBackend:
    Arg = _not_implemented()
    ParLoop = _not_implemented()
    Kernel = _not_implemented()
    Dat = _not_implemented()
    Set = _not_implemented()
    ExtrudedSet = _not_implemented()
    MixedSet = _not_implemented()
    Subset = _not_implemented()
    DataSet = _not_implemented()
    MixedDataSet = _not_implemented()
    Map = _not_implemented()
    MixedMap = _not_implemented()
    Sparsity = _not_implemented()
    Halo = _not_implemented()
    Dat = _not_implemented()
    MixedDat = _not_implemented()
    DatView = _not_implemented()
    Mat = _not_implemented()
    Global = _not_implemented()
    GlobalDataSet = _not_implemented()

    def _getattr_(self, key):
        val = super(AbstractComputeBackend, self)._getattr_(key)
        if isinstance(val, _not_implemented):
            raise NotImplementedError("'{}' is not implemented for backend"
                    " '{}'.".format(val, self.__name__))
        return val
