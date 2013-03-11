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
#E = FiniteElement("DG", "triangle", 0)
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

mass, = compile_form(a, "mass")
rhs, = compile_form(L, "rhs")

#from IPython import embed
#embed()

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

sparsity = op2.Sparsity([((elem_node1, elem_node1),(edge_node1, edge_node1)),(elem_node2, elem_node2)], [2,1], "sparsity")
print "========"

##
## THE LIST OF BLOCKS WILL BE IN: sparsity.sparsity_list
##

#sparsity = op2.Sparsity([((elem_node1, elem_node1),(elem_node1, elem_node1)),(elem_node1, elem_node1)], [2,1], "sparsity")
#print "========"

#from IPython import embed; embed()

mat = op2.Mat(sparsity, valuetype, "mat")

print "====== done with mat creation"

coord_vals = np.asarray([ (0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (0.0, 1.5) ],
                           dtype=valuetype)
coords = op2.Dat(nodes, 2, coord_vals, valuetype, "coords")

velocity_vals = np.asarray([ (1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0) ],
                           dtype=valuetype)
velocity = op2.Dat(nodes, 2, velocity_vals, valuetype, "velocity")

pressure_vals = np.asarray([ 0.1, 2.0, 1.0, 0.2 ],
                           dtype=valuetype)
pressure = op2.Dat(nodes, 1, pressure_vals, valuetype, "pressure")

#nodes_p1 = op2.Set(4)
#nodes_p2 = op2.Set(9)
#nodes_p3 = op2.Set(16)
#nodes_p4 = op2.Set(25)

#dat_p1 = op2.Dat(nodes_p1,2,np_arr_p1,dtype=valuetype)
#dat_p2 = op2.Dat(nodes_p2,1)
#dat_p3 = op2.Dat(nodes_p3,1)
#dat_p4 = op2.Dat(nodes_p4,2)

#joined_dats = op2.Dat([dat_p1,dat_p2)

#mixed_sets = op2.Multiset([nodes_p1,nodes_p2,nodes_p3,nodes_p4])
#mixed_dats = op2.MultiDat(mixed_sets, [2,1,1,2])

#mixed_sets = op2.MultiSet([nodes, nodes], "mixed_sets")
#mixed_dats = op2.MultiDat(mixed_sets, [velocity, pressure], "mixed_dats")
#mixed_maps = op2.MultiMap([elem_node1, elem_node1], [2,1], "mixed_maps")

#mixed = [velocity, pressure]
#print mixed

# 2 * NUM_NODES is no longer valid here,
# We need to replace that by the size of the matrix which is the lsize of
#  the sparsity object

#dofs = op2.Set(sparsity._lsize, "dofs")


##version 2
#f_vals = np.asarray([(1.0, 2.0)]*sparsity.x_size, dtype=valuetype)
#b_vals = np.asarray([0.0]*sparsity.b_size, dtype=valuetype)
#x_vals = np.asarray([0.0]*sparsity.x_size, dtype=valuetype)

#version 3
fields = op2.MultiDat([velocity, pressure], "fields")

xset = op2.MultiSet(sparsity.x_sets)
bset = op2.MultiSet(sparsity.b_sets)

b_block1 = np.asarray([0.0]*sparsity.b_sizes[0], dtype=valuetype)
b_block2 = np.asarray([0.0]*sparsity.b_sizes[1], dtype=valuetype)

print sparsity.b_sizes[0]
print sparsity.b_sizes[1]

b1 = op2.Dat(nodes,2,b_block1,valuetype,"b1")
b2 = op2.Dat(nodes,1,b_block2,valuetype,"b2")

x_block1 = np.asarray([0.0]*sparsity.x_sizes[0], dtype=valuetype)
x_block2 = np.asarray([0.0]*sparsity.x_sizes[1], dtype=valuetype)

x1 = op2.Dat(nodes,2,x_block1,valuetype,"x1")
x2 = op2.Dat(nodes,1,x_block2,valuetype,"x2")

f_block1 = np.asarray([0.0]*sparsity.x_sizes[0], dtype=valuetype)
f_block2 = np.asarray([0.0]*sparsity.x_sizes[1], dtype=valuetype)

f1 = op2.Dat(nodes,2,f_block1,valuetype,"f1")
f2 = op2.Dat(nodes,1,f_block2,valuetype,"f2")

f = op2.MultiDat([f1, f2], "f")
b = op2.MultiDat([b1, b2], "b")
x = op2.MultiDat([x1, x2], "x")

row_maps = op2.MultiMap(sparsity.getRowMaps(elements)) #list of row maps [(),()]
                                         # THE LIST OF MAPS MUST
                                         # CONTAIN ALL THE MAPS.
                                         # The maps will then be filetered at
                                         # loop level to get from the multiple
                                         # elemnts such as: [... [(),()] ...]
                                         # the maps that have the iterset
                                         # equal to the loop iterset
                                         # so the map list becomes:
                                         # [... [()] ...]

                                         # DOES THE CONDITION THAT IN A TUPLE OF TUPLES
                                         # of maps, only one of the pairs
                                         # holds the maps from the iteration set of the
                                         # loop to other mesh elements

                                         # the filtering at loops level
                                         # has effects over what happens
                                         # when the sizes are computed

col_maps = op2.MultiMap(sparsity.getColMaps(elements)) # list of colmaps *  as above  *

print sparsity.getRowMaps(elements)
print sparsity.getColMaps(elements)

#Alternative for the user if he doesn't want to call the sparsity methods
#row_maps = op2.MultiMap([elem_node1, elem_node2])
#col_maps = op2.MultiMap([elem_node1, elem_node2])

mass._code = """
        void mass_cell_integral_0_otherwise(double A[1][1], double *x[2], double *velocity[2], double* pressure[1], int j, int k)
{
    printf(" This is the Kernel Code that's not generated yet! -> %d %d \\n",j,k);
    A[0][0] = 2*j + k;

}

        """

rhs._code = """
void rhs_cell_integral_0_otherwise(double A[1][1], double *x[2], double *f_vec_0[2], double *f_vec_1[1], int j)
{

    printf("This is the RHS Kernel code that's not generated yet! -> %d \\n", j);
    A[0][0] = 10;

}
"""

# Assemble and solve
print "======= start first loop"
op2.par_loop(mass, elements(3,3),
             mat((row_maps[op2.i[0]], col_maps[op2.i[1]]), op2.INC), #first the rowmaps then the colmaps
             coords(elem_node1, op2.READ),
             fields(row_maps, op2.READ))

print mat.mat_list[0].handle.view()

print "======= start second loop"
op2.par_loop(rhs, elements(3),
                     b(row_maps[op2.i[0]], op2.INC),
                     coords(elem_node1, op2.READ),
                     f(row_maps, op2.READ))

print b.data[0]

solver = op2.Solver(preconditioner="none")
solver.solve(mat, x, b)

# Print solution
print "Computed solution: %s" % x.data

# Save output (if necessary)
#if opt['save_output']:
#    import pickle
#    with open("mass_vector.out","w") as out:
#        pickle.dump((f.data, x.data), out)
