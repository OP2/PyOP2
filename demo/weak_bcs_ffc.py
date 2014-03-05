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

"""PyOP2 laplace equation demo (weak BCs)

This demo uses ffc-generated kernels to solve the Laplace equation on a unit
square with boundary conditions:

  u     = 1 on y = 0
  du/dn = 2 on y = 1

The domain is meshed as follows:

  *-*-*
  |/|/|
  *-*-*
  |/|/|
  *-*-*

This demo requires the pyop2 branch of ffc, which can be obtained with:

bzr branch lp:~mapdes/ffc/pyop2

This may also depend on development trunk versions of other FEniCS programs.
"""

import os
import numpy as np

from pyop2 import op2, utils

_kerneldir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernels')


def main(opt):
    if opt['firedrake']:
        # Set up finite element problem
        from firedrake.ffc_interface import compile_form
        from ufl import *

        E = FiniteElement("Lagrange", "triangle", 1)

        v = TestFunction(E)
        u = TrialFunction(E)
        f = Coefficient(E)
        g = Coefficient(E)

        a = dot(grad(v), grad(u)) * dx
        L = v * f * dx + v * g * ds(2)

        # Generate code for Laplacian and rhs assembly.

        laplacian, = compile_form(a, "laplacian")
        rhs, weak = compile_form(L, "rhs")
        if opt['update_kernels']:
            with open(os.path.join(_kerneldir, 'laplacian.c'), 'w') as f:
                f.write(laplacian._code)
            with open(os.path.join(_kerneldir, 'laplacian_weak_rhs.c'), 'w') as f:
                f.write(rhs._code)
            with open(os.path.join(_kerneldir, 'laplacian_weak.c'), 'w') as f:
                f.write(weak._code)
    else:
        with open(os.path.join(_kerneldir, 'laplacian.c')) as f:
            laplacian = op2.Kernel(f.read(), "laplacian_cell_integral_0_otherwise")
        with open(os.path.join(_kerneldir, 'laplacian_weak_rhs.c')) as f:
            rhs = op2.Kernel(f.read(), "rhs_cell_integral_0_otherwise")
        with open(os.path.join(_kerneldir, 'laplacian_weak.c')) as f:
            weak = op2.Kernel(f.read(), "rhs_exterior_facet_integral_0_2")

    # Set up simulation data structures

    NUM_ELE = 8
    NUM_NODES = 9
    NUM_BDRY_ELE = 2
    NUM_BDRY_NODE = 3
    valuetype = np.float64

    nodes = op2.Set(NUM_NODES, "nodes")
    elements = op2.Set(NUM_ELE, "elements")

    # Elements that Weak BC will be assembled over
    top_bdry_elements = op2.Set(NUM_BDRY_ELE, "top_boundary_elements")
    # Nodes that Strong BC will be applied over
    bdry_nodes = op2.Set(NUM_BDRY_NODE, "boundary_nodes")

    elem_node_map = np.array([0, 1, 4, 4, 3, 0, 1, 2, 5, 5, 4, 1, 3, 4, 7, 7,
                              6, 3, 4, 5, 8, 8, 7, 4], dtype=np.uint32)
    elem_node = op2.Map(elements, nodes, 3, elem_node_map, "elem_node")

    top_bdry_elem_node_map = np.array([7, 6, 3, 8, 7, 4], dtype=valuetype)
    top_bdry_elem_node = op2.Map(top_bdry_elements, nodes, 3,
                                 top_bdry_elem_node_map, "top_bdry_elem_node")

    bdry_node_node_map = np.array([0, 1, 2], dtype=valuetype)
    bdry_node_node = op2.Map(
        bdry_nodes, nodes, 1, bdry_node_node_map, "bdry_node_node")

    sparsity = op2.Sparsity((nodes, nodes), (elem_node, elem_node), "sparsity")
    mat = op2.Mat(sparsity, valuetype, "mat")

    coord_vals = np.array([(0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
                           (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),
                           (0.0, 1.0), (0.5, 1.0), (1.0, 1.0)],
                          dtype=valuetype)
    coords = op2.Dat(nodes ** 2, coord_vals, valuetype, "coords")

    u_vals = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    f = op2.Dat(nodes, np.zeros(NUM_NODES, dtype=valuetype), valuetype, "f")
    b = op2.Dat(nodes, np.zeros(NUM_NODES, dtype=valuetype), valuetype, "b")
    x = op2.Dat(nodes, np.zeros(NUM_NODES, dtype=valuetype), valuetype, "x")
    u = op2.Dat(nodes, u_vals, valuetype, "u")

    bdry = op2.Dat(bdry_nodes, np.ones(3, dtype=valuetype), valuetype, "bdry")

    # This isn't perfect, defining the boundary gradient on more nodes than are on
    # the boundary is couter-intuitive
    bdry_grad_vals = np.asarray([2.0] * 9, dtype=valuetype)
    bdry_grad = op2.Dat(nodes, bdry_grad_vals, valuetype, "gradient")
    facet = op2.Global(1, 2, np.uint32, "facet")

    # If a form contains multiple integrals with differing coefficients, FFC
    # generates kernels that take all the coefficients of the entire form (not
    # only the respective integral) as arguments. Arguments that correspond to
    # forms that are not used in that integral are simply not referenced.
    # We therefore need a dummy argument in place of the coefficient that is not
    # used in the par_loop for OP2 to generate the correct kernel call.

    # Assemble matrix and rhs

    op2.par_loop(laplacian, elements,
                 mat(op2.INC, (elem_node[op2.i[0]], elem_node[op2.i[1]])),
                 coords(op2.READ, elem_node, flatten=True))

    op2.par_loop(rhs, elements,
                 b(op2.INC, elem_node[op2.i[0]]),
                 coords(op2.READ, elem_node, flatten=True),
                 f(op2.READ, elem_node),
                 bdry_grad(op2.READ, elem_node))  # argument ignored

    # Apply weak BC

    op2.par_loop(weak, top_bdry_elements,
                 b(op2.INC, top_bdry_elem_node[op2.i[0]]),
                 coords(op2.READ, top_bdry_elem_node, flatten=True),
                 f(op2.READ, top_bdry_elem_node),  # argument ignored
                 bdry_grad(op2.READ, top_bdry_elem_node),
                 facet(op2.READ))

    # Apply strong BC

    mat.zero_rows([0, 1, 2], 1.0)
    strongbc_rhs = op2.Kernel("""
    void strongbc_rhs(double *val, double *target) { *target = *val; }
    """, "strongbc_rhs")
    op2.par_loop(strongbc_rhs, bdry_nodes,
                 bdry(op2.READ),
                 b(op2.WRITE, bdry_node_node[0]))

    solver = op2.Solver(ksp_type='gmres')
    solver.solve(mat, x, b)

    # Print solution
    if opt['return_output']:
        return u.data, x.data
    if opt['print_output']:
        print "Expected solution: %s" % u.data
        print "Computed solution: %s" % x.data

    # Save output (if necessary)
    if opt['save_output']:
        import pickle
        with open("weak_bcs.out", "w") as out:
            pickle.dump((u.data, x.data), out)

parser = utils.parser(group=True, description=__doc__)
parser.add_argument('--print-output', action='store_true', help='Print output')
parser.add_argument('-r', '--return-output', action='store_true',
                    help='Return output for testing')
parser.add_argument('-s', '--save-output', action='store_true',
                    help='Save the output of the run (used for testing)')
parser.add_argument('-p', '--profile', action='store_true',
                    help='Create a cProfile for the run')
parser.add_argument('-f', '--firedrake', action='store_true',
                    help='Obtain kernels via Firedrake')
parser.add_argument('-u', '--update-kernels', action='store_true',
                    help='Update FFC-generated kernels (requires -f)')

if __name__ == '__main__':
    opt = vars(parser.parse_args())
    op2.init(**opt)

    if opt['profile']:
        import cProfile
        cProfile.run('main(opt)', filename='weak_bcs_ffc.cprofile')
    else:
        main(opt)
