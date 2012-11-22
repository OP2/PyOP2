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

backends = ['sequential', 'opencl', 'cuda']

def _seed():
    return 0.02041724

# Large enough that there is more than one block and more than one
# thread per element in device backends
nnodes = 4096
nele = nnodes / 2

class TestIterationSpaceDats:
    """
    Test IterationSpace access to Dat objects
    """

    @pytest.fixture(scope='module')
    def node_set(cls):
        return op2.Set(nnodes, 'node_set')

    @pytest.fixture(scope='module')
    def ele_set(cls):
        return op2.Set(nele, 'ele_set')

    @pytest.fixture
    def d1(cls, node_set):
        return op2.Dat(node_set, 1, numpy.zeros(nnodes), dtype=numpy.int32)

    @pytest.fixture
    def d2(cls, node_set):
        return op2.Dat(node_set, 2, numpy.zeros(2 * nnodes), dtype=numpy.int32)

    @pytest.fixture
    def vd1(cls, ele_set):
        return op2.Dat(ele_set, 1, numpy.zeros(nele), dtype=numpy.int32)

    @pytest.fixture
    def vd2(cls, ele_set):
        return op2.Dat(ele_set, 2, numpy.zeros(2 * nele), dtype=numpy.int32)

    @pytest.fixture(scope='module')
    def node2ele(cls, node_set, ele_set):
        vals = numpy.arange(nnodes)/2
        return op2.Map(node_set, ele_set, 1, vals, 'node2ele')

    def test_sum_nodes_to_edges(self, backend):
        """Creates a 1D grid with edge values numbered consecutively.
        Iterates over edges, summing the node values."""

        nedges = nnodes-1
        nodes = op2.Set(nnodes, "nodes")
        edges = op2.Set(nedges, "edges")

        node_vals = op2.Dat(nodes, 1, numpy.arange(nnodes, dtype=numpy.uint32), numpy.uint32, "node_vals")
        edge_vals = op2.Dat(edges, 1, numpy.zeros(nedges, dtype=numpy.uint32), numpy.uint32, "edge_vals")

        e_map = numpy.array([(i, i+1) for i in range(nedges)], dtype=numpy.uint32)
        edge2node = op2.Map(edges, nodes, 2, e_map, "edge2node")

        kernel_sum = """
void kernel_sum(unsigned int* nodes, unsigned int *edge, int i)
{ *edge += nodes[0]; }
"""

        op2.par_loop(op2.Kernel(kernel_sum, "kernel_sum"), edges(edge2node.dim),
                       node_vals(edge2node[op2.i[0]],       op2.READ),
                       edge_vals(op2.IdentityMap, op2.INC))

        expected = numpy.arange(1, nedges*2+1, 2).reshape(nedges, 1)
        assert all(expected == edge_vals.data)

    def test_read_1d_itspace_map(self, backend, node_set, d1, vd1, node2ele):
        vd1.data[:] = numpy.arange(nele).reshape(nele, 1)
        k = """
        void k(int *d, int *vd, int i) {
        d[0] = vd[0];
        }"""
        op2.par_loop(op2.Kernel(k, 'k'), node_set(node2ele.dim),
                     d1(op2.IdentityMap, op2.WRITE),
                     vd1(node2ele[op2.i[0]], op2.READ))
        assert all(d1.data[::2] == vd1.data)
        assert all(d1.data[1::2] == vd1.data)

    def test_write_1d_itspace_map(self, backend, node_set, vd1, node2ele):
        k = """
        void k(int *vd, int i) {
        vd[0] = 2;
        }
        """

        op2.par_loop(op2.Kernel(k, 'k'), node_set(node2ele.dim),
                     vd1(node2ele[op2.i[0]], op2.WRITE))
        assert all(vd1.data == 2)

    def test_inc_1d_itspace_map(self, backend, node_set, d1, vd1, node2ele):
        vd1.data[:] = 3
        d1.data[:] = numpy.arange(nnodes).reshape(d1.data.shape)

        k = """
        void k(int *d, int *vd, int i) {
        vd[0] += *d;
        }"""
        op2.par_loop(op2.Kernel(k, 'k'), node_set(node2ele.dim),
                     d1(op2.IdentityMap, op2.READ),
                     vd1(node2ele[op2.i[0]], op2.INC))
        expected = numpy.zeros_like(vd1.data)
        expected[:] = 3
        expected += numpy.arange(start=0, stop=nnodes, step=2).reshape(expected.shape)
        expected += numpy.arange(start=1, stop=nnodes, step=2).reshape(expected.shape)
        assert all(vd1.data == expected)

    def test_read_2d_itspace_map(self, backend, node_set, d2, vd2, node2ele):
        vd2.data[:] = numpy.arange(nele*2).reshape(nele, 2)
        k = """
        void k(int *d, int *vd, int i) {
        d[0] = vd[0];
        d[1] = vd[1];
        }"""
        op2.par_loop(op2.Kernel(k, 'k'), node_set(node2ele.dim),
                     d2(op2.IdentityMap, op2.WRITE),
                     vd2(node2ele[op2.i[0]], op2.READ))
        assert all(d2.data[::2,0] == vd2.data[:,0])
        assert all(d2.data[::2,1] == vd2.data[:,1])
        assert all(d2.data[1::2,0] == vd2.data[:,0])
        assert all(d2.data[1::2,1] == vd2.data[:,1])

    def test_write_2d_itspace_map(self, backend, node_set, vd2, node2ele):
        k = """
        void k(int *vd, int i) {
        vd[0] = 2;
        vd[1] = 3;
        }
        """

        op2.par_loop(op2.Kernel(k, 'k'), node_set(node2ele.dim),
                     vd2(node2ele[op2.i[0]], op2.WRITE))
        assert all(vd2.data[:,0] == 2)
        assert all(vd2.data[:,1] == 3)

    def test_inc_2d_itspace_map(self, backend, node_set, d2, vd2, node2ele):
        vd2.data[:, 0] = 3
        vd2.data[:, 1] = 4
        d2.data[:] = numpy.arange(2 * nnodes).reshape(d2.data.shape)

        k = """
        void k(int *d, int *vd, int i) {
        vd[0] += d[0];
        vd[1] += d[1];
        }"""
        op2.par_loop(op2.Kernel(k, 'k'), node_set(node2ele.dim),
                     d2(op2.IdentityMap, op2.READ),
                     vd2(node2ele[op2.i[0]], op2.INC))

        expected = numpy.zeros_like(vd2.data)
        expected[:, 0] = 3
        expected[:, 1] = 4
        expected[:, 0] += numpy.arange(start=0, stop=2*nnodes, step=4)
        expected[:, 0] += numpy.arange(start=2, stop=2*nnodes, step=4)
        expected[:, 1] += numpy.arange(start=1, stop=2*nnodes, step=4)
        expected[:, 1] += numpy.arange(start=3, stop=2*nnodes, step=4)
        assert all(vd2.data[:,0] == expected[:,0])
        assert all(vd2.data[:,1] == expected[:,1])

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
