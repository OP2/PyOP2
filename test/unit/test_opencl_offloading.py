import pytest
pytest.importorskip("pyopencl")

import sys
import petsc4py
petsc4py.init(sys.argv
              + "-viennacl_backend opencl".split()
              + "-viennacl_opencl_device_type cpu".split())
from pyop2 import op2
import pyopencl.array as cla
import numpy as np
import loopy as lp
lp.set_caching_enabled(False)


def allclose(a, b, rtol=1e-05, atol=1e-08):
    """
    Prefer this routine over np.allclose(...) to allow pycuda/pyopencl arrays
    """
    return bool(abs(a - b) < (atol + rtol * abs(b)))


def pytest_generate_tests(metafunc):
    if "backend" in metafunc.fixturenames:
        from pyop2.backends.opencl import opencl_backend
        metafunc.parametrize("backend", [opencl_backend])


def test_new_backend_raises_not_implemented_error():
    from pyop2.backends import AbstractComputeBackend
    unimplemented_backend = AbstractComputeBackend()

    attrs = ["GlobalKernel", "Parloop", "Set", "ExtrudedSet", "MixedSet",
             "Subset", "DataSet", "MixedDataSet", "Map", "MixedMap", "Dat",
             "MixedDat", "DatView", "Mat", "Global", "GlobalDataSet",
             "PETScVecType"]

    for attr in attrs:
        with pytest.raises(NotImplementedError):
            getattr(unimplemented_backend, attr)


def test_dat_with_petscvec_representation(backend):
    op2.set_offloading_backend(backend)

    nelems = 9
    data = np.random.rand(nelems)
    set_ = op2.compute_backend.Set(nelems)
    dset = op2.compute_backend.DataSet(set_, 1)
    dat = op2.compute_backend.Dat(dset, data.copy())

    assert isinstance(dat.data_ro, np.ndarray)
    dat.data[:] *= 3

    with op2.offloading():
        assert isinstance(dat.data_ro, cla.Array)
        dat.data[:] *= 2

    assert isinstance(dat.data_ro, np.ndarray)
    np.testing.assert_allclose(dat.data_ro, 6*data)


def test_dat_not_as_petscvec(backend):
    op2.set_offloading_backend(backend)

    nelems = 9
    data = np.random.randint(low=-10, high=10,
                             size=nelems,
                             dtype=np.int64)
    set_ = op2.compute_backend.Set(nelems)
    dset = op2.compute_backend.DataSet(set_, 1)
    dat = op2.compute_backend.Dat(dset, data.copy())

    assert isinstance(dat.data_ro, np.ndarray)
    dat.data[:] *= 3

    with op2.offloading():
        assert isinstance(dat.data_ro, cla.Array)
        dat.data[:] *= 2

    assert isinstance(dat.data_ro, np.ndarray)
    np.testing.assert_allclose(dat.data_ro, 6*data)


def test_global_reductions(backend):
    op2.set_offloading_backend(backend)

    sum_knl = lp.make_function(
        "{ : }",
        """
        g[0] = g[0] + x[0]
        """,
        [lp.GlobalArg("g,x",
                      dtype="float64",
                      shape=lp.auto,
                      )],
        name="sum_knl",
        target=lp.CWithGNULibcTarget(),
        lang_version=(2018, 2))

    rng = np.random.default_rng()
    nelems = 4_000
    data_to_sum = rng.random(nelems)

    with op2.offloading():

        set_ = op2.compute_backend.Set(4_000)
        dset = op2.compute_backend.DataSet(set_, 1)

        dat = op2.compute_backend.Dat(dset, data_to_sum.copy())
        glob = op2.compute_backend.Global(1, 0, np.float64, "g")

        op2.parloop(op2.Kernel(sum_knl, "sum_knl"),
                    set_,
                    glob(op2.INC),
                    dat(op2.READ))

        assert isinstance(glob.data_ro, cla.Array)
        assert allclose(glob.data_ro[0], data_to_sum.sum())
