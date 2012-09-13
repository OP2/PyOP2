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

backends = ['opencl']

#max...
#nelems = 92681
nelems = 10

def elems():
    return op2.Set(nelems, "elems")

def xarray():
    return numpy.array(range(nelems), dtype=numpy.uint32)

class TestDirectLoopFusion:
    """
    Direct Loop Fusion Tests
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
        kernel_wo2 = """
void kernel_wo2(unsigned int* x) { *x = 40; }
"""
        l = op2.par_loop2(op2.Kernel(kernel_wo, "kernel_wo"), op2.Kernel(kernel_wo2, "kernel_wo2"), elems(), [x(op2.IdentityMap, op2.WRITE)], [x(op2.IdentityMap, op2.WRITE)])
        print x.data
        assert all(map(lambda x: x==40, x.data))

    def test_rw(self, backend, x):
        kernel_rw = """
void kernel_rw(unsigned int* x) { (*x) = (*x) + 1; }
"""
        kernel_rw2 = """
void kernel_rw2(unsigned int* x) { (*x) = (*x) - 2; }
"""
        l = op2.par_loop2(op2.Kernel(kernel_rw, "kernel_rw"), op2.Kernel(kernel_rw2, "kernel_rw2"), elems(), [x(op2.IdentityMap, op2.RW)], [x(op2.IdentityMap, op2.RW)])
        print sum(x.data)
        print ((nelems - 2) * (nelems - 1) / 2 - 1)
        assert sum(x.data) == ((nelems - 2) * (nelems - 1) / 2 - 1)

    def test_global_inc(self, backend, x, g):
        kernel_global_inc = """
void kernel_global_inc(unsigned int* x, unsigned int* inc) { (*x) = (*x) + 1; }
"""
        kernel_global_inc2 = """
void kernel_global_inc2(unsigned int* x, unsigned int* inc) { (*inc) += (*x); }
"""
        args = [x(op2.IdentityMap, op2.RW), g(op2.INC)]

        l = op2.par_loop2(op2.Kernel(kernel_global_inc, "kernel_global_inc"), op2.Kernel(kernel_global_inc2, "kernel_global_inc2"), elems(), args, args)
        assert g.data[0] == nelems * (nelems + 1) / 2

    def test_global_read(self, backend, x, h):
        kernel_global_read = """
void kernel_global_read(unsigned int* x, unsigned int* h) { (*x) += (*h) / 2; }
"""
        kernel_global_read2 = """
void kernel_global_read2(unsigned int* x, unsigned int* h) { (*x) += (*h) / 2 + (*h) % 2; }
"""
        args = [x(op2.IdentityMap, op2.RW), h(op2.READ)]

        l = op2.par_loop2(op2.Kernel(kernel_global_read, "kernel_global_read"), op2.Kernel(kernel_global_read2, "kernel_global_read2"), elems(), args, args)
        assert sum(x.data) == nelems * (nelems + 1) / 2

    def test_2d_dat(self, backend, y):
        kernel_2d_wo = """
void kernel_2d_wo(unsigned int* x) { x[0] = 42; x[1] = 43; }
"""
        kernel_2d_wo2 = """
void kernel_2d_wo2(unsigned int* x) { x[0] = 43; x[1] = 42; }
"""
        args = [y(op2.IdentityMap, op2.WRITE)]
        l = op2.par_loop2(op2.Kernel(kernel_2d_wo, "kernel_2d_wo"), op2.Kernel(kernel_2d_wo2, "kernel_2d_wo2"), elems(), args, args)
        assert all(map(lambda x: all(x==[43, 42]), y.data))

    def test_2d_dat_soa(self, backend, soa):
        kernel_soa = """
void kernel_soa(unsigned int * x) { OP2_STRIDE(x, 0) = 42; OP2_STRIDE(x, 1) = 43; }
"""
        kernel_soa2 = """
void kernel_soa2(unsigned int * x) { OP2_STRIDE(x, 0) = 43; OP2_STRIDE(x, 1) = 42; }
"""
        args = [soa(op2.IdentityMap, op2.WRITE)]
        l = op2.par_loop2(op2.Kernel(kernel_soa, "kernel_soa"), op2.Kernel(kernel_soa2, "kernel_soa2"), elems(), args, args)
        assert all(soa.data[0] == 43) and all(soa.data[1] == 42)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
