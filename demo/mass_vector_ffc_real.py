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
E = FiniteElement("DG", "triangle", 0)
V = VectorElement("Lagrange", "triangle", 1)
W = V * E

v, q = TestFunctions(W)
u, r = TrialFunctions(W)
s = TestFunction(W)
t = TrialFunction(W)
f = Coefficient(W)
g = Coefficient(V)
h = Coefficient(E)

a = inner(s,t)*dx
L = inner(s,f)*dx

# Generate code for mass and rhs assembly.

mass, = compile_form(a, "mass")
rhs, = compile_form(L, "rhs")

# Set up simulation data structures

NUM_ELE   = 2
NUM_EDGES = 5
NUM_NODES = 4
valuetype = np.float64

nodes = op2.Set(NUM_NODES, "nodes")
edges = op2.Set(NUM_EDGES, "edges")
elements = op2.Set(NUM_ELE, "elements")

elem_node_map  = np.asarray([ 0, 1, 3, 2, 3, 1 ], dtype=np.uint32)
elem_elem_map = np.asarray([ 0, 1 ], dtype=np.uint32)
edge_node_map  = np.asarray([ 0, 1, 1, 2, 2, 3, 3, 0, 1, 3 ], dtype=np.uint32)

elem_node1 = op2.Map(elements, nodes, 3, elem_node_map,  "elem_node1")
elem_elem = op2.Map(elements, elements, 1, elem_elem_map, "elem_elem")

edge_node1 = op2.Map(edges, nodes, 2, edge_node_map, "edge_node1")


sparsity = op2.Sparsity([((elem_node1, elem_node1),(edge_node1, edge_node1)),(elem_elem, elem_elem)], [2,1], "sparsity")

##
## THE LIST OF BLOCKS WILL BE IN: sparsity.sparsity_list
##

mat = op2.Mat(sparsity, valuetype, "mat")

coord_vals = np.asarray([ (0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (0.0, 1.5) ],
                           dtype=valuetype)
coords = op2.Dat(nodes, 2, coord_vals, valuetype, "coords")

velocity_vals = np.asarray([ (1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0) ],
                           dtype=valuetype)
velocity = op2.Dat(nodes, 2, velocity_vals, valuetype, "velocity")

pressure_vals = np.asarray([ 0.1, 0.1 ],
                           dtype=valuetype)
pressure = op2.Dat(elements, 1, pressure_vals, valuetype, "pressure")

f = op2.MultiDat([velocity, pressure], "fields")

b_block1 = np.asarray([0.0]*sparsity.b_sizes[0], dtype=valuetype)
b_block2 = np.asarray([0.0]*sparsity.b_sizes[1], dtype=valuetype)

b1 = op2.Dat(nodes,2,b_block1,valuetype,"b1")
b2 = op2.Dat(elements,1,b_block2,valuetype,"b2")

x_block1 = np.asarray([0.0]*sparsity.x_sizes[0], dtype=valuetype)
x_block2 = np.asarray([0.0]*sparsity.x_sizes[1], dtype=valuetype)

x1 = op2.Dat(nodes,2,x_block1,valuetype,"x1")
x2 = op2.Dat(elements,1,x_block2,valuetype,"x2")

b = op2.MultiDat([b1, b2], "b")
x = op2.MultiDat([x1, x2], "x")

row_maps = op2.MultiMap(sparsity.getRowMaps(elements)) #list of row maps [(),()]

col_maps = op2.MultiMap(sparsity.getColMaps(elements)) # list of colmaps *  as above  *

# Assemble and solve
op2.par_loop(mass, elements(7,7),
             mat((row_maps[op2.i[0]], col_maps[op2.i[1]]), op2.INC), #first the rowmaps then the colmaps
             coords(elem_node1, op2.READ))

op2.par_loop(rhs, elements(7),
             b(row_maps[op2.i[0]], op2.INC),
             coords(elem_node1, op2.READ),
             f(row_maps, op2.READ))

solver = op2.Solver(linear_solver='gmres', preconditioner="none")
solver.solve(mat, x, b)

# Print solution
print "Computed solution: %s" % x.data
print "Forcing vector: %s" % f.data

# Save output (if necessary)
if opt['save_output']:
    import pickle
    with open("mass_vector.out","w") as out:
        pickle.dump((f.data, x.data), out)
