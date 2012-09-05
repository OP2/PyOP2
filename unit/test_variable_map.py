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
nnodes = 92681

class TestVariableMap:
    """
    Variable Map Tests
    """

    def test_sum_nodes_to_edges(self, backend):
        """Creates a 1D grid with edge values numbered consecutively.
        Iterates over edges, summing the node values. This uses a variable arity
        map with all the arities being the same."""

        nedges = nnodes-1
        nodes = op2.Set(nnodes, "nodes")
        edges = op2.Set(nedges, "edges")

        node_vals = op2.Dat(nodes, 1, numpy.array(range(nnodes), dtype=numpy.uint32), numpy.uint32, "node_vals")
        edge_vals = op2.Dat(edges, 1, numpy.array([0]*nedges, dtype=numpy.uint32), numpy.uint32, "edge_vals")

        e_map = numpy.array([(i, i+1) for i in range(nedges)], dtype=numpy.uint32)
        e_dim = numpy.array([(2 * i) for i in range(nedges + 1)], dtype=numpy.uint32)
        edge2node = op2.Map(edges, nodes, e_dim, e_map, "edge2node")

        kernel_sum = """
void kernel_sum(unsigned int* nodes[1], unsigned int *edge)
{ *edge = 0; unsigned int i = -1; while(nodes[++i] != 0) *edge += nodes[i][0]; }
"""

        op2.par_loop(op2.Kernel(kernel_sum, "kernel_sum"), edges, \
                       node_vals(edge2node,       op2.READ),      \
                       edge_vals(op2.IdentityMap, op2.WRITE))

        expected = numpy.asarray(range(1, nedges*2+1, 2)).reshape(nedges, 1)
        assert all(expected == edge_vals.data)

    def test_small_sum(self, backend):
        """Creates a 1D grid with edge values numbered consecutively.
        Iterates over edges, summing the node values. This uses a variable
        arity map where each edge can have more than 2 nodes attached to it."""

        nedges = nnodes - 1
        nodes = op2.Set(nnodes, "nodes")
        edges = op2.Set(nedges, "edges")

        node_vals = op2.Dat(nodes, 1, numpy.array(range(nnodes), dtype=numpy.uint32), numpy.uint32, "node_vals")
        edge_vals = op2.Dat(edges, 1, numpy.array([0]*nedges, dtype=numpy.uint32), numpy.uint32, "edge_vals")

        random.seed(_seed())
        map_vals = []
        map_dims = [0]
        for i in range(nedges):
            #We limit the maximum nodes to the node count // 2048 because of
            #it taking too long to calculate so many random numbers
            count = random.randrange(1, nnodes // 2048)
            map_dims += [map_dims[-1] + count]
            for j in range(count):
                map_vals += [random.randrange(0, nnodes)]

        e_map = numpy.array(map_vals, dtype=numpy.uint32)
        e_dim = numpy.array(map_dims, dtype=numpy.uint32)
        edge2node = op2.Map(edges, nodes, e_dim, e_map, "edge2node")

        kernel_sum = """
void kernel_sum(unsigned int* nodes[1], unsigned int *edge)
{ *edge = 0; unsigned int i = -1; while(nodes[++i] != 0) *edge += nodes[i][0]; }
"""

        op2.par_loop(op2.Kernel(kernel_sum, "kernel_sum"), edges, \
                       node_vals(edge2node,       op2.READ),      \
                       edge_vals(op2.IdentityMap, op2.WRITE))

        for i in range(len(map_dims) - 1):
            assert sum(map_vals[map_dims[i]:map_dims[i+1]]) == edge_vals.data[i]

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
