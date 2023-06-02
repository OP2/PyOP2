import ctypes

import loopy as lp
import numpy as np
from pyop2.petsc import get_petsc_variables


def _parse_int_type():
    vars = get_petsc_variables()
    int_size = int(vars["PETSC_INDEX_SIZE"])
    if int_size == 32:
        return np.dtype(np.int32)
    else:
        assert int_size == 64
        return np.dtype(np.int64)


def _parse_real_type():
    return np.dtype(np.float64)


def _parse_scalar_type():
    vars = get_petsc_variables()
    scalar_type = vars["PETSC_SCALAR"]
    if scalar_type == "real":
        return np.dtype(np.float64)
    else:
        assert scalar_type == "complex"
        return np.dtype(np.complex128)


IntType = _parse_int_type()
RealType = _parse_real_type()
ScalarType = _parse_scalar_type()


def as_cstr(dtype):
    """Convert a numpy dtype like object to a C type as a string."""
    return {"bool": "unsigned char",
            "int": "int",
            "int8": "int8_t",
            "int16": "int16_t",
            "int32": "int32_t",
            "int64": "int64_t",
            "uint8": "uint8_t",
            "uint16": "uint16_t",
            "uint32": "uint32_t",
            "uint64": "uint64_t",
            "float32": "float",
            "float64": "double",
            "complex128": "double complex"}[np.dtype(dtype).name]


def as_ctypes(dtype):
    """Convert a numpy dtype like object to a ctypes type."""
    return {"bool": ctypes.c_bool,
            "int": ctypes.c_int,
            "int8": ctypes.c_char,
            "int16": ctypes.c_int16,
            "int32": ctypes.c_int32,
            "int64": ctypes.c_int64,
            "uint8": ctypes.c_ubyte,
            "uint16": ctypes.c_uint16,
            "uint32": ctypes.c_uint32,
            "uint64": ctypes.c_uint64,
            "float32": ctypes.c_float,
            "float64": ctypes.c_double}[np.dtype(dtype).name]


def as_numpy_dtype(dtype):
    """Convert a dtype-like object into a numpy dtype."""
    if isinstance(dtype, np.dtype):
        return dtype
    elif isinstance(dtype, lp.types.NumpyType):
        return dtype.numpy_dtype
    else:
        raise ValueError


def dtype_limits(dtype):
    """Attempt to determine the min and max values of a datatype.

    :arg dtype: A numpy datatype.
    :returns: a 2-tuple of min, max
    :raises ValueError: If numeric limits could not be determined.
    """
    try:
        info = np.finfo(dtype)
    except ValueError:
        # maybe an int?
        try:
            info = np.iinfo(dtype)
        except ValueError as e:
            raise ValueError("Unable to determine numeric limits from %s" % dtype) from e
    return info.min, info.max
