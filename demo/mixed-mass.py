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

"""PyOP2 2D mass equation demo (space version)

This demo solves the identity equation for a vector variable on a quadrilateral
domain. The initial condition is that all DoFs are [(1, 2), 3]^T

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

V = VectorElement("Lagrange", "triangle", 1)
H = FiniteElement("Lagrange", "triangle", 1)

W = V*H

v, q     = TestFunctions(W)
u, h     = TrialFunctions(W)
f_u, f_h = split(Coefficient(W))

a_u = inner(v,u)*dx
a_h = q*h*dx
a   = a_u + a_h
L_u = inner(v,f_u)*dx
L_h = inner(q,f_h)*dx
L   = L_u + L_h

# Generate code for mass and rhs assembly.

# Mass kernels are: [[mass_uu, mass_uh], [mass_hu, mass_hh]]
mass = compile_form(a, "mass")
# RHS kernels are: [rhs_u, rhs_h]
rhs  = compile_form(L, "rhs")

# Set up simulation data structures

NUM_ELE   = 2
NUM_NODES = 4
valuetype = np.float64

nodes = op2.Set(NUM_NODES, "nodes")
elements = op2.Set(NUM_ELE, "elements")

elem_node_map = np.asarray([ 0, 1, 3, 2, 3, 1 ], dtype=np.uint32)
elem_node = op2.Map(elements, nodes, 3, elem_node_map, "elem_node")

sparsity = op2.Sparsity([[((elem_node, elem_node)), ((elem_node, elem_node))],
                         [((elem_node, elem_node)), ((elem_node, elem_node))]],
                        [[(2,2), (2,1)],
                         [(1,2), (1,1)]],
                        "sparsity")
mat = op2.Mat(sparsity, valuetype, "mat")

coord_vals = np.asarray([ (0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (0.0, 1.5) ],
                           dtype=valuetype)
coords = op2.Dat(nodes, 2, coord_vals, valuetype, "coords")

f_u_vals = np.asarray([(1.0, 2.0)]*4,    dtype=valuetype)
f_h_vals = np.asarray([3.0]*4,           dtype=valuetype)
b_u_vals = np.asarray([0.0]*2*NUM_NODES, dtype=valuetype)
b_h_vals = np.asarray([0.0]*NUM_NODES,   dtype=valuetype)
x_u_vals = np.asarray([0.0]*2*NUM_NODES, dtype=valuetype)
x_h_vals = np.asarray([0.0]*NUM_NODES,   dtype=valuetype)
f_u = op2.Dat(nodes, 2, f_u_vals, valuetype, "f_u")
f_h = op2.Dat(nodes, 1, f_h_vals, valuetype, "f_h")
f   = [f_u, f_h]
b_u = op2.Dat(nodes, 2, b_u_vals, valuetype, "b_u")
b_h = op2.Dat(nodes, 1, b_h_vals, valuetype, "b_h")
b   = [b_u, b_h]
x_u = op2.Dat(nodes, 2, x_u_vals, valuetype, "x_u")
x_h = op2.Dat(nodes, 1, x_h_vals, valuetype, "x_h")
x   = [x_u, x_h]

# Assemble and solve

for i in xrange(2):
    op2.par_loop(rhs[i], elements(3),
                 b[i](elem_node[op2.i[0]], op2.INC),
                 coords(elem_node, op2.READ),
                 f[i](elem_node, op2.READ))

    for i in xrange(2):
        op2.par_loop(mass[i][j], elements(3,3),
                     mat[i][j]((elem_node[op2.i[0]], elem_node[op2.i[1]]), op2.INC),
                     coords(elem_node, op2.READ))

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
