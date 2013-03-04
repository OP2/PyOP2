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

"""PyOP2 2D mass equation demo (vector field version)

This demo solves the identity equation for a vector variable on a quadrilateral
domain. The initial condition is that all DoFs are [1, 2]^T

This demo requires the pyop2 branch of ffc, which can be obtained with:

bzr branch lp:~mapdes/ffc/pyop2

This may also depend on development trunk versions of other FEniCS programs.
"""

from pyop2 import op2, utils
from ufl import *
from pyop2.ffc_interface import compile_form

import numpy as np

parser = utils.parser(group=True, description=__doc__)
parser.add_argument('-s', '--save-output',
                    action='store_true',
                    help='Save the output of the run (used for testing)')
opt = vars(parser.parse_args())
op2.init(**opt)

# Set up finite element identity problem
E = FiniteElement("Lagrange", "triangle", 1)
V = VectorElement("Lagrange", "triangle", 1)
W = V * E

v, q = TestFunctions(W)
u, r = TrialFunctions(W)
s = TestFunction(W)
t = TrialFunction(W)
f = Coefficient(W)
g = Coefficient(V)
h = Coefficient(E)

a = inner(v,u)*dx + inner(q,r)*dx
#a = inner(s,t)*dx
L = inner(s,f)*dx
#L = inner(v,g)*dx + inner(q,h)*dx

# Generate code for mass and rhs assembly.

mass = compile_form(a, "mass")
rhs = compile_form(L, "rhs")

# Set up simulation data structures

NUM_ELE   = 2
NUM_EDGES = 5
NUM_NODES = 4
valuetype = np.float64

nodes = op2.Set(NUM_NODES, "nodes")
edges = op2.Set(NUM_EDGES, "edges")
elements = op2.Set(NUM_ELE, "elements")

elem_node_map  = np.asarray([ 0, 1, 3, 2, 3, 1 ], dtype=np.uint32)
elem_node_map2 = np.asarray([ 3, 1, 0, 2, 1, 3 ], dtype=np.uint32)
edge_node_map  = np.asarray([ 0, 1, 1, 2, 2, 3, 3, 0, 1, 3 ], dtype=np.uint32)

elem_node  = op2.Map(elements, nodes, 3, elem_node_map,  "elem_node")
elem_node1 = op2.Map(elements, nodes, 3, elem_node_map,  "elem_node1")
elem_node2 = op2.Map(elements, nodes, 3, elem_node_map2, "elem_node2")

edge_node1 = op2.Map(edges, nodes, 2, edge_node_map, "edge_node1")

#sparsity = op2.Sparsity(((elem_node, elem_node),(elem_node, elem_node)), 2, "sparsity")
#print "========"

#sparsity = op2.Sparsity([((elem_node1, elem_node1),(edge_node1, edge_node1)),(elem_node2, elem_node2)], [2,1], "sparsity")
#print "========"

sparsity = op2.Sparsity([(elem_node1, elem_node1),(elem_node1, elem_node1)], [2,1], "sparsity")
print "========"

#from IPython import embed; embed()

mat = op2.Mat(sparsity, valuetype, "mat")

coord_vals = np.asarray([ (0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (0.0, 1.5) ],
                           dtype=valuetype)
coords = op2.Dat(nodes, 2, coord_vals, valuetype, "coords")

velocity_vals = np.asarray([ (1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0) ],
                           dtype=valuetype)
velocity = op2.Dat(nodes, 2, velocity_vals, valuetype, "velocity")

pressure_vals = np.asarray([ 0.1, 2.0, 1.0, 0.2 ],
                           dtype=valuetype)
pressure = op2.Dat(nodes, 1, pressure_vals, valuetype, "pressure")

mixed_sets = op2.MultiSet([nodes, nodes], [2,1], "mixed_sets")
mixed_dats = op2.MultiDat(mixed_sets, [velocity, pressure], "mixed_dats")
mixed_maps = op2.MultiMap([elem_node1, elem_node1], [2,1], "mixed_maps")

#mixed = [velocity, pressure]
#print mixed

# 2 * NUM_NODES is no longer valid here,
# We need to replace that by the size of the matrix which is the lsize of
#  the sparsity object

dofs = op2.Set(sparsity._lsize, "dofs")

b_vals = np.asarray([0.0]*sparsity._lsize, dtype=valuetype)
x_vals = np.asarray([0.0]*sparsity._lsize, dtype=valuetype)

b = op2.Dat(mixed_dats, [2,1], b_vals, valuetype, "b")
x = op2.Dat(mixed_dats, [2,1], x_vals, valuetype, "x")

# Assemble and solve

op2.par_loop(mass, elements(3,3),
             mat(mixed_maps[op2.i[0]], mixed_maps[op2.i[1]], op2.INC), #first the rowmaps then the colmaps
             coords(elem_node1, op2.READ),
             mixed_dats(mixed_maps, op2.READ))

op2.par_loop(rhs, elements(3),
                     b([elem_node1,elem_node1][op2.i[0]], op2.INC),
                     coords(elem_node1, op2.READ),
                     f(elem_node1, op2.READ))

solver = op2.Solver()
solver.solve(mat, x, b)

# Print solution

print "Expected solution: %s" % f.data
print "Computed solution: %s" % x.data

# Save output (if necessary)
if opt['save_output']:
    import pickle
    with open("mass_vector.out","w") as out:
        pickle.dump((f.data, x.data), out)
