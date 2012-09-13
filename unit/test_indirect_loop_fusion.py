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
import random

from pyop2 import op2

backends = ['sequential', 'opencl']

def _seed():
    return 0.02041724

#max...
nelems = 92681
#nelems = 7700
#nelems = 12

class TestIndirectLoopFusion:
    """
    Indirect Loop Fusion Tests
    """

    def pytest_funcarg__iterset(cls, request):
        return op2.Set(nelems, "iterset")

    def pytest_funcarg__indset(cls, request):
        return op2.Set(nelems, "indset")

    def pytest_funcarg__x(cls, request):
        return op2.Dat(request.getfuncargvalue('indset'), 1, range(nelems), numpy.uint32, "x")

    def pytest_funcarg__iterset2indset(cls, request):
        u_map = numpy.array(range(nelems), dtype=numpy.uint32)
        random.shuffle(u_map, _seed)
        return op2.Map(request.getfuncargvalue('iterset'), request.getfuncargvalue('indset'), 1, u_map, "iterset2indset")

    def pytest_funcarg__reversediterset2indset(cls, request):
        u_map = numpy.array(range(nelems), dtype=numpy.uint32)
        random.shuffle(u_map, _seed)
        u_map = u_map[::-1].copy()
        return op2.Map(request.getfuncargvalue('iterset'), request.getfuncargvalue('indset'), 1, u_map, "rev_iterset2indset")

    def test_onecolor_wo(self, backend, iterset, x, iterset2indset):
        kernel_wo = "void kernel_wo(unsigned int* x) { *x = 42; }"
        kernel_wo2 = "void kernel_wo2(unsigned int* x) { *x = 40; }"
        args = [x(iterset2indset[0], op2.WRITE)]

        op2.par_loop2(op2.Kernel(kernel_wo, "kernel_wo"), op2.Kernel(kernel_wo2, "kernel_wo2"), iterset, args, args)
        assert all(map(lambda x: x==40, x.data))

    def test_onecolor_rw(self, backend, iterset, x, iterset2indset):
        kernel_rw = "void kernel_rw(unsigned int* x) { (*x) = (*x) - 1; }"
        kernel_rw2 = "void kernel_rw2(unsigned int* x) { (*x) = (*x) + 2; }"
        args = [x(iterset2indset[0], op2.RW)]

        op2.par_loop2(op2.Kernel(kernel_rw, "kernel_rw"), op2.Kernel(kernel_rw2, "kernel_rw2"), iterset, args, args)
        assert sum(x.data) == nelems * (nelems + 1) / 2

    def test_indirect_inc(self, backend, iterset):
        unitset = op2.Set(1, "unitset")

        u = op2.Dat(unitset, 1, numpy.array([0], dtype=numpy.uint32), numpy.uint32, "u")

        u_map = numpy.zeros(nelems, dtype=numpy.uint32)
        iterset2unit = op2.Map(iterset, unitset, 1, u_map, "iterset2unitset")

        kernel_inc = "void kernel_inc(unsigned int* x) { (*x) = (*x) + 1; }"
        kernel_inc2 = "void kernel_inc2(unsigned int* x) { (*x) = (*x) + 2; }"
        args = [u(iterset2unit[0], op2.INC)]

        op2.par_loop2(op2.Kernel(kernel_inc, "kernel_inc"), op2.Kernel(kernel_inc2, "kernel_inc2"), iterset, args, args)
        assert u.data[0] == nelems * 3

    def test_global_read(self, backend, iterset, x, iterset2indset):
        g = op2.Global(1, 2, numpy.uint32, "g")

        kernel_global_read = "void kernel_global_read(unsigned int* x, unsigned int* g) { (*x) /= (*g); }\n"
        kernel_global_read2 = "void kernel_global_read2(unsigned int* x, unsigned int* g) { (*x) += (*g); }\n"
        args = [x(iterset2indset[0], op2.RW), g(op2.READ)]

        op2.par_loop2(op2.Kernel(kernel_global_read, "kernel_global_read"), op2.Kernel(kernel_global_read2, "kernel_global_read2"), iterset, args, args)
        assert sum(x.data) == sum(map(lambda v: (v / 2) + 2, range(nelems)))

    def test_global_inc(self, backend, iterset, x, iterset2indset):
        g = op2.Global(1, 0, numpy.uint32, "g")

        kernel_global_inc = "void kernel_global_inc(unsigned int *x, unsigned int *inc) { (*x) = (*x) - 2 ; (*inc) += (*x); }"
        kernel_global_inc2 = "void kernel_global_inc2(unsigned int *x, unsigned int *inc) { (*x) = (*x) + 3; (*inc) += (*x); }"
        args = [x(iterset2indset[0], op2.RW), g(op2.INC)]

        op2.par_loop2(op2.Kernel(kernel_global_inc, "kernel_global_inc"), op2.Kernel(kernel_global_inc2, "kernel_global_inc2"),
                                   iterset, args, args)
        assert sum(x.data) == nelems * (nelems + 1) / 2
        assert g.data[0] == numpy.uint32(nelems * (nelems - 2) & 0xFFFFFFFF)

    def test_2d_dat(self, backend, iterset, indset, iterset2indset):
        x = op2.Dat(indset, 2, numpy.array([range(nelems), range(nelems)], dtype=numpy.uint32), numpy.uint32, "x")

        kernel_wo = "void kernel_wo(unsigned int* x) { x[0] = 42; x[1] = 43; }"
        kernel_wo2 = "void kernel_wo2(unsigned int* x) { x[0] = 43; x[1] = 42; }"
        args = [x(iterset2indset[0], op2.WRITE)]

        op2.par_loop2(op2.Kernel(kernel_wo, "kernel_wo"), op2.Kernel(kernel_wo2, "kernel_wo2"), iterset, args, args)
        assert all(map(lambda x: all(x==[43, 42]), x.data))

    def test_2d_map(self, backend):
        nedges = nelems - 1
        nodes = op2.Set(nelems, "nodes")
        edges = op2.Set(nedges, "edges")
        node_vals = op2.Dat(nodes, 1, numpy.array(range(nelems), dtype=numpy.uint32), numpy.uint32, "node_vals")
        edge_vals = op2.Dat(edges, 1, numpy.array([0] * nedges, dtype=numpy.uint32), numpy.uint32, "edge_vals")

        e_map = numpy.array([(i, i + 1) for i in range(nedges)], dtype=numpy.uint32)
        edge2node = op2.Map(edges, nodes, 2, e_map, "edge2node")

        kernel_sum = """
        void kernel_sum(unsigned int *nodes1, unsigned int *nodes2, unsigned int *edge)       { *edge = *nodes1 + 2 * *nodes2; }
        """
        kernel_sum2 = """
        void kernel_sum2(unsigned int *nodes2, unsigned int *edge)
        { *edge -= *nodes2; }
        """
        args1 = [node_vals(edge2node[0], op2.READ),
                 node_vals(edge2node[1], op2.READ),
                 edge_vals(op2.IdentityMap, op2.WRITE)]

        args2 = [node_vals(edge2node[1], op2.READ),
                 edge_vals(op2.IdentityMap, op2.RW)]

        op2.par_loop2(op2.Kernel(kernel_sum, "kernel_sum"), op2.Kernel(kernel_sum2, "kernel_sum2"),
                      edges, args1, args2)
        expected = numpy.asarray(range(1, nedges * 2 + 1, 2)).reshape(nedges, 1)
        assert all(expected == edge_vals.data)

    def test_reversed_calculate(self, backend, iterset, x, iterset2indset, reversediterset2indset):
        kernel_global_inc = "void kernel_global_inc(unsigned int *x) { (*x) += 1; }"
        kernel_global_inc2 = "void kernel_global_inc2(unsigned int *x) { (*x) *= 2; }"
        args1 = [x(iterset2indset[0], op2.RW)]
        args2 = [x(reversediterset2indset[0], op2.RW)]
        op2.par_loop2(op2.Kernel(kernel_global_inc, "kernel_global_inc"), op2.Kernel(kernel_global_inc2, "kernel_global_inc2"),
                                   iterset, args1, args2)
        expected = numpy.asarray(range(2, nelems * 2 + 1, 2)).reshape(nelems, 1)
        print expected
        print x.data
        assert all(expected == x.data)
