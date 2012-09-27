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

from pyop2 import op2

backends = ['sequential', 'opencl', 'cuda']

# Large enough that there is more than one block and more than one
# thread per element in device backends
nelems = 4096

def elems():
    return op2.Set(nelems, "elems")

def xarray():
    return numpy.array(range(nelems), dtype=numpy.uint32)

class TestDirectLoop:
    """
    Direct Loop Tests
    """

    def pytest_funcarg__x(cls, request):
        return op2.Dat(elems(),  1, xarray(), numpy.uint32, "x")

    def pytest_funcarg__y(cls, request):
        return op2.Dat(elems(),  2, [xarray(), xarray()], numpy.uint32, "x")

    def pytest_funcarg__g(cls, request):
        return op2.Global(1, 0, numpy.uint32, "g")

    def pytest_funcarg__h(cls, request):
        return op2.Global(1, 1, numpy.uint32, "h")

    def pytest_funcarg__soa(cls, request):
        return op2.Dat(elems(), 2, [xarray(), xarray()], numpy.uint32, "x", soa=True)

    def test_wo(self, backend, x):
        kernel_wo = """
void kernel_wo(unsigned int* x) { *x = 42; }
"""
        l = op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"), elems(), x(op2.IdentityMap, op2.WRITE))
        assert all(map(lambda x: x==42, x.data))

    def test_rw(self, backend, x):
        kernel_rw = """
void kernel_rw(unsigned int* x) { (*x) = (*x) + 1; }
"""
        l = op2.par_loop(op2.Kernel(kernel_rw, "kernel_rw"), elems(), x(op2.IdentityMap, op2.RW))
        assert sum(x.data) == nelems * (nelems + 1) / 2

    def test_global_inc(self, backend, x, g):
        kernel_global_inc = """
void kernel_global_inc(unsigned int* x, unsigned int* inc) { (*x) = (*x) + 1; (*inc) += (*x); }
"""
        l = op2.par_loop(op2.Kernel(kernel_global_inc, "kernel_global_inc"), elems(), x(op2.IdentityMap, op2.RW), g(op2.INC))
        assert g.data[0] == nelems * (nelems + 1) / 2

    def test_global_inc_init_not_zero(self, backend, g):
        k = """
void k(unsigned int* inc) { (*inc) += 1; }
"""
        g.data[0] = 10
        op2.par_loop(op2.Kernel(k, 'k'), elems(), g(op2.INC))
        assert g.data[0] == elems().size + 10

    def test_global_max_dat_is_max(self, backend, x, g):
        k_code = """
        void k(unsigned int *x, unsigned int *g) {
        if ( *g < *x ) { *g = *x; }
        }"""
        k = op2.Kernel(k_code, 'k')

        op2.par_loop(k, elems(), x(op2.IdentityMap, op2.READ), g(op2.MAX))
        assert g.data[0] == x.data.max()

    def test_global_max_g_is_max(self, backend, x, g):
        k_code = """
        void k(unsigned int *x, unsigned int *g) {
        if ( *g < *x ) { *g = *x; }
        }"""

        k = op2.Kernel(k_code, 'k')

        g.data[0] = nelems * 2

        op2.par_loop(k, elems(), x(op2.IdentityMap, op2.READ), g(op2.MAX))

        assert g.data[0] == nelems * 2

    def test_global_min_dat_is_min(self, backend, x, g):
        k_code = """
        void k(unsigned int *x, unsigned int *g) {
        if ( *g > *x ) { *g = *x; }
        }"""
        k = op2.Kernel(k_code, 'k')
        g.data[0] = 1000
        op2.par_loop(k, elems(), x(op2.IdentityMap, op2.READ), g(op2.MIN))

        assert g.data[0] == x.data.min()

    def test_global_min_g_is_min(self, backend, x, g):
        k_code = """
        void k(unsigned int *x, unsigned int *g) {
        if ( *g > *x ) { *g = *x; }
        }"""

        k = op2.Kernel(k_code, 'k')
        g.data[0] = 10
        x.data[:] = 11
        op2.par_loop(k, elems(), x(op2.IdentityMap, op2.READ), g(op2.MIN))

        assert g.data[0] == 10

    def test_global_read(self, backend, x, h):
        kernel_global_read = """
void kernel_global_read(unsigned int* x, unsigned int* h) { (*x) += (*h); }
"""
        op2.par_loop(op2.Kernel(kernel_global_read, "kernel_global_read"), elems(), x(op2.IdentityMap, op2.RW), h(op2.READ))
        assert sum(x.data) == nelems * (nelems + 1) / 2

    def test_2d_dat(self, backend, y):
        kernel_2d_wo = """
void kernel_2d_wo(unsigned int* x) { x[0] = 42; x[1] = 43; }
"""
        l = op2.par_loop(op2.Kernel(kernel_2d_wo, "kernel_2d_wo"), elems(), y(op2.IdentityMap, op2.WRITE))
        assert all(map(lambda x: all(x==[42,43]), y.data))

    def test_2d_dat_soa(self, backend, soa):
        kernel_soa = """
void kernel_soa(unsigned int * x) { OP2_STRIDE(x, 0) = 42; OP2_STRIDE(x, 1) = 43; }
"""
        l = op2.par_loop(op2.Kernel(kernel_soa, "kernel_soa"), elems(), soa(op2.IdentityMap, op2.WRITE))
        assert all(soa.data[0] == 42) and all(soa.data[1] == 43)

    def test_parloop_should_set_ro_flag(self, backend, x):
        kernel = """void k(unsigned int *x) { *x = 1; }"""
        x_data = x.data
        op2.par_loop(op2.Kernel(kernel, 'k'), elems(), x(op2.IdentityMap, op2.WRITE))
        with pytest.raises(RuntimeError):
            x_data[0] = 1

    def test_host_write_works(self, backend, x, g):
        kernel = """void k(unsigned int *x, unsigned int *g) { *g += *x; }"""
        x.data[:] = 1
        g.data[:] = 0
        op2.par_loop(op2.Kernel(kernel, 'k'), elems(), x(op2.IdentityMap, op2.READ), g(op2.INC))
        assert g.data[0] == nelems

        x.data[:] = 2
        g.data[:] = 0
        op2.par_loop(op2.Kernel(kernel, 'k'), elems(), x(op2.IdentityMap, op2.READ), g(op2.INC))
        assert g.data[0] == 2*nelems

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
