# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

import pytest
import numpy
from numpy.testing import assert_allclose

from pyop2 import op2

# Large enough that there is more than one block and more than one
# thread per element in device backends
nelems = 4096

class TestGlobalReductions:
    """
    Global reduction argument tests
    """

    @pytest.fixture(scope='module')
    def set(cls):
        return op2.Set(nelems, 'set')

    @pytest.fixture
    def d1(cls, set):
        return op2.Dat(set, 1, numpy.arange(nelems)+1, dtype=numpy.uint32)

    @pytest.fixture
    def d2(cls, set):
        return op2.Dat(set, 2, numpy.arange(2*nelems)+1, dtype=numpy.uint32)

    @pytest.fixture(scope='module')
    def k1_write_to_dat(cls):
        k = """
        void k(unsigned int *x, unsigned int *g) { *x = *g; }
        """
        return op2.Kernel(k, "k")

    @pytest.fixture(scope='module')
    def k1_inc_to_global(cls):
        k = """
        void k(unsigned int *x, unsigned int *g) { *g += *x; }
        """
        return op2.Kernel(k, "k")

    @pytest.fixture(scope='module')
    def k1_min_to_global(cls):
        k = """
        void k(unsigned int *x, unsigned int *g) { if (*x < *g) *g = *x; }
        """
        return op2.Kernel(k, "k")

    @pytest.fixture(scope='module')
    def k2_min_to_global(cls):
        k = """
        void k(unsigned int *x, unsigned int *g) {
        if (x[0] < g[0]) g[0] = x[0];
        if (x[1] < g[1]) g[1] = x[1];
        }
        """
        return op2.Kernel(k, "k")

    @pytest.fixture(scope='module')
    def k1_max_to_global(cls):
        k = """
        void k(unsigned int *x, unsigned int *g) {
        if (*x > *g) *g = *x;
        }
        """
        return op2.Kernel(k, "k")

    @pytest.fixture(scope='module')
    def k2_max_to_global(cls):
        k = """
        void k(unsigned int *x, unsigned int *g) {
        if (x[0] > g[0]) g[0] = x[0];
        if (x[1] > g[1]) g[1] = x[1];
        }
        """
        return op2.Kernel(k, "k")

    @pytest.fixture(scope='module')
    def k2_write_to_dat(cls, request):
        k = """
        void k(unsigned int *x, unsigned int *g) { *x = g[0] + g[1]; }
        """
        return op2.Kernel(k, "k")

    @pytest.fixture(scope='module')
    def k2_inc_to_global(cls):
        k = """
        void k(unsigned int *x, unsigned int *g) { g[0] += x[0]; g[1] += x[1]; }
        """
        return op2.Kernel(k, "k")

    @pytest.fixture
    def duint32(cls, set):
        return op2.Dat(set, 1, [12]*nelems, numpy.uint32, "duint32")

    @pytest.fixture
    def dint32(cls, set):
        return op2.Dat(set, 1, [-12]*nelems, numpy.int32, "dint32")

    @pytest.fixture
    def dfloat32(cls, set):
        return op2.Dat(set, 1, [-12.0]*nelems, numpy.float32, "dfloat32")

    @pytest.fixture
    def dfloat64(cls, set):
        return op2.Dat(set, 1, [-12.0]*nelems, numpy.float64, "dfloat64")


    def test_direct_min_uint32(self, backend, set, duint32):
        kernel_min = """
void kernel_min(unsigned int* x, unsigned int* g)
{
  if ( *x < *g ) *g = *x;
}
"""
        g = op2.Global(1, 8, numpy.uint32, "g")

        op2.par_loop(op2.Kernel(kernel_min, "kernel_min"), set,
                     duint32(op2.IdentityMap, op2.READ),
                     g(op2.MIN))
        assert g.data[0] == 8

    def test_direct_min_int32(self, backend, set, dint32):
        kernel_min = """
void kernel_min(int* x, int* g)
{
  if ( *x < *g ) *g = *x;
}
"""
        g = op2.Global(1, 8, numpy.int32, "g")

        op2.par_loop(op2.Kernel(kernel_min, "kernel_min"), set,
                     dint32(op2.IdentityMap, op2.READ),
                     g(op2.MIN))
        assert g.data[0] == -12

    def test_direct_max_int32(self, backend, set, dint32):
        kernel_max = """
void kernel_max(int* x, int* g)
{
  if ( *x > *g ) *g = *x;
}
"""
        g = op2.Global(1, -42, numpy.int32, "g")

        op2.par_loop(op2.Kernel(kernel_max, "kernel_max"), set,
                     dint32(op2.IdentityMap, op2.READ),
                     g(op2.MAX))
        assert g.data[0] == -12


    def test_direct_min_float(self, backend, set, dfloat32):
        kernel_min = """
void kernel_min(float* x, float* g)
{
  if ( *x < *g ) *g = *x;
}
"""
        g = op2.Global(1, -.8, numpy.float32, "g")

        op2.par_loop(op2.Kernel(kernel_min, "kernel_min"), set,
                     dfloat32(op2.IdentityMap, op2.READ),
                     g(op2.MIN))

        assert_allclose(g.data[0], -12.0)

    def test_direct_max_float(self, backend, set, dfloat32):
        kernel_max = """
void kernel_max(float* x, float* g)
{
  if ( *x > *g ) *g = *x;
}
"""
        g = op2.Global(1, -42.8, numpy.float32, "g")

        op2.par_loop(op2.Kernel(kernel_max, "kernel_max"), set,
                     dfloat32(op2.IdentityMap, op2.READ),
                     g(op2.MAX))
        assert_allclose(g.data[0], -12.0)


    def test_direct_min_double(self, backend, set, dfloat64):
        kernel_min = """
void kernel_min(double* x, double* g)
{
  if ( *x < *g ) *g = *x;
}
"""
        g = op2.Global(1, -.8, numpy.float64, "g")

        op2.par_loop(op2.Kernel(kernel_min, "kernel_min"), set,
                     dfloat64(op2.IdentityMap, op2.READ),
                     g(op2.MIN))
        assert_allclose(g.data[0], -12.0)

    def test_direct_max_double(self, backend, set, dfloat64):
        kernel_max = """
void kernel_max(double* x, double* g)
{
  if ( *x > *g ) *g = *x;
}
"""
        g = op2.Global(1, -42.8, numpy.float64, "g")

        op2.par_loop(op2.Kernel(kernel_max, "kernel_max"), set,
                     dfloat64(op2.IdentityMap, op2.READ),
                     g(op2.MAX))
        assert_allclose(g.data[0], -12.0)

    def test_1d_read(self, backend, k1_write_to_dat, set, d1):
        g = op2.Global(1, 1, dtype=numpy.uint32)
        op2.par_loop(k1_write_to_dat, set,
                     d1(op2.IdentityMap, op2.WRITE),
                     g(op2.READ))

        assert all(d1.data == g.data)

    def test_2d_read(self, backend, k2_write_to_dat, set, d1):
        g = op2.Global(2, (1, 2), dtype=numpy.uint32)
        op2.par_loop(k2_write_to_dat, set,
                     d1(op2.IdentityMap, op2.WRITE),
                     g(op2.READ))

        assert all(d1.data == g.data.sum())

    def test_1d_inc(self, backend, k1_inc_to_global, set, d1):
        g = op2.Global(1, 0, dtype=numpy.uint32)
        op2.par_loop(k1_inc_to_global, set,
                     d1(op2.IdentityMap, op2.READ),
                     g(op2.INC))

        assert g.data == d1.data.sum()

    def test_1d_min_dat_is_min(self, backend, k1_min_to_global, set, d1):
        val = d1.data.min() + 1
        g = op2.Global(1, val, dtype=numpy.uint32)
        op2.par_loop(k1_min_to_global, set,
                     d1(op2.IdentityMap, op2.READ),
                     g(op2.MIN))

        assert g.data == d1.data.min()

    def test_1d_min_global_is_min(self, backend, k1_min_to_global, set, d1):
        d1.data[:] += 10
        val = d1.data.min() - 1
        g = op2.Global(1, val, dtype=numpy.uint32)
        op2.par_loop(k1_min_to_global, set,
                     d1(op2.IdentityMap, op2.READ),
                     g(op2.MIN))
        assert g.data == val

    def test_1d_max_dat_is_max(self, backend, k1_max_to_global, set, d1):
        val = d1.data.max() - 1
        g = op2.Global(1, val, dtype=numpy.uint32)
        op2.par_loop(k1_max_to_global, set,
                     d1(op2.IdentityMap, op2.READ),
                     g(op2.MAX))

        assert g.data == d1.data.max()

    def test_1d_max_global_is_max(self, backend, k1_max_to_global, set, d1):
        val = d1.data.max() + 1
        g = op2.Global(1, val, dtype=numpy.uint32)
        op2.par_loop(k1_max_to_global, set,
                     d1(op2.IdentityMap, op2.READ),
                     g(op2.MAX))

        assert g.data == val

    def test_2d_inc(self, backend, k2_inc_to_global, set, d2):
        g = op2.Global(2, (0, 0), dtype=numpy.uint32)
        op2.par_loop(k2_inc_to_global, set,
                     d2(op2.IdentityMap, op2.READ),
                     g(op2.INC))

        assert g.data[0] == d2.data[:,0].sum()
        assert g.data[1] == d2.data[:,1].sum()

    def test_2d_min_dat_is_min(self, backend, k2_min_to_global, set, d2):
        val_0 = d2.data[:,0].min() + 1
        val_1 = d2.data[:,1].min() + 1
        g = op2.Global(2, (val_0, val_1), dtype=numpy.uint32)
        op2.par_loop(k2_min_to_global, set,
                     d2(op2.IdentityMap, op2.READ),
                     g(op2.MIN))

        assert g.data[0] == d2.data[:,0].min()
        assert g.data[1] == d2.data[:,1].min()

    def test_2d_min_global_is_min(self, backend, k2_min_to_global, set, d2):
        d2.data[:,0] += 10
        d2.data[:,1] += 10
        val_0 = d2.data[:,0].min() - 1
        val_1 = d2.data[:,1].min() - 1
        g = op2.Global(2, (val_0, val_1), dtype=numpy.uint32)
        op2.par_loop(k2_min_to_global, set,
                     d2(op2.IdentityMap, op2.READ),
                     g(op2.MIN))
        assert g.data[0] == val_0
        assert g.data[1] == val_1

    def test_2d_max_dat_is_max(self, backend, k2_max_to_global, set, d2):
        val_0 = d2.data[:,0].max() - 1
        val_1 = d2.data[:,1].max() - 1
        g = op2.Global(2, (val_0, val_1), dtype=numpy.uint32)
        op2.par_loop(k2_max_to_global, set,
                     d2(op2.IdentityMap, op2.READ),
                     g(op2.MAX))

        assert g.data[0] == d2.data[:,0].max()
        assert g.data[1] == d2.data[:,1].max()

    def test_2d_max_global_is_max(self, backend, k2_max_to_global, set, d2):
        max_val_0 = d2.data[:,0].max() + 1
        max_val_1 = d2.data[:,1].max() + 1
        g = op2.Global(2, (max_val_0, max_val_1), dtype=numpy.uint32)
        op2.par_loop(k2_max_to_global, set,
                     d2(op2.IdentityMap, op2.READ),
                     g(op2.MAX))

        assert g.data[0] == max_val_0
        assert g.data[1] == max_val_1

    def test_1d_multi_inc_same_global(self, backend, k1_inc_to_global, set, d1):
        g = op2.Global(1, 0, dtype=numpy.uint32)
        op2.par_loop(k1_inc_to_global, set,
                     d1(op2.IdentityMap, op2.READ),
                     g(op2.INC))
        assert g.data == d1.data.sum()

        op2.par_loop(k1_inc_to_global, set,
                     d1(op2.IdentityMap, op2.READ),
                     g(op2.INC))

        assert g.data == d1.data.sum()*2

    def test_1d_multi_inc_same_global_reset(self, backend, k1_inc_to_global, set, d1):
        g = op2.Global(1, 0, dtype=numpy.uint32)
        op2.par_loop(k1_inc_to_global, set,
                     d1(op2.IdentityMap, op2.READ),
                     g(op2.INC))
        assert g.data == d1.data.sum()

        g.data = 10
        op2.par_loop(k1_inc_to_global, set,
                     d1(op2.IdentityMap, op2.READ),
                     g(op2.INC))

        assert g.data == d1.data.sum() + 10

    def test_1d_multi_inc_diff_global(self, backend, k1_inc_to_global, set, d1):
        g = op2.Global(1, 0, dtype=numpy.uint32)
        g2 = op2.Global(1, 10, dtype=numpy.uint32)
        op2.par_loop(k1_inc_to_global, set,
                     d1(op2.IdentityMap, op2.READ),
                     g(op2.INC))
        assert g.data == d1.data.sum()

        op2.par_loop(k1_inc_to_global, set,
                     d1(op2.IdentityMap, op2.READ),
                     g2(op2.INC))
        assert g2.data == d1.data.sum() + 10

