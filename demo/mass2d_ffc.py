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

"""PyOP2 2D mass equation demo

This is a demo of the use of ffc to generate kernels. It solves the identity
equation on a quadrilateral domain. It requires the pyop2 branch of ffc,
which can be obtained with:

bzr branch lp:~mapdes/ffc/pyop2

This may also depend on development trunk versions of other FEniCS programs.
"""

import os
import numpy as np

from pyop2 import op2, utils

_kerneldir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernels')


def main(opt):
    if opt['firedrake']:
        # Set up finite element identity problem
        from firedrake.ffc_interface import compile_form
        from ufl import *

        E = FiniteElement("Lagrange", "triangle", 1)

        v = TestFunction(E)
        u = TrialFunction(E)
        f = Coefficient(E)

        a = v * u * dx
        L = v * f * dx

        # Generate code for mass and rhs assembly.

        mass, = compile_form(a, "mass")
        rhs, = compile_form(L, "rhs")
        if opt['update_kernels']:
            with open(os.path.join(_kerneldir, 'mass2d.c'), 'w') as f:
                f.write(mass._code)
            with open(os.path.join(_kerneldir, 'mass2d_rhs.c'), 'w') as f:
                f.write(rhs._code)
    else:
        with open(os.path.join(_kerneldir, 'mass2d.c')) as f:
            mass = op2.Kernel(f.read(), "mass_cell_integral_0_otherwise")
        with open(os.path.join(_kerneldir, 'mass2d_rhs.c')) as f:
            rhs = op2.Kernel(f.read(), "rhs_cell_integral_0_otherwise")

    # Set up simulation data structures

    NUM_ELE = 2
    NUM_NODES = 4
    valuetype = np.float64

    nodes = op2.Set(NUM_NODES, "nodes")
    elements = op2.Set(NUM_ELE, "elements")

    elem_node_map = np.array([0, 1, 3, 2, 3, 1], dtype=np.uint32)
    elem_node = op2.Map(elements, nodes, 3, elem_node_map, "elem_node")

    sparsity = op2.Sparsity((nodes, nodes), (elem_node, elem_node), "sparsity")
    mat = op2.Mat(sparsity, valuetype, "mat")

    coord_vals = np.array([(0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (0.0, 1.5)],
                          dtype=valuetype)
    coords = op2.Dat(nodes ** 2, coord_vals, valuetype, "coords")

    f = op2.Dat(nodes, np.array([1.0, 2.0, 3.0, 4.0]), valuetype, "f")
    b = op2.Dat(nodes, np.zeros(NUM_NODES, dtype=valuetype), valuetype, "b")
    x = op2.Dat(nodes, np.zeros(NUM_NODES, dtype=valuetype), valuetype, "x")

    # Assemble and solve

    op2.par_loop(mass, elements,
                 mat(op2.INC, (elem_node[op2.i[0]], elem_node[op2.i[1]])),
                 coords(op2.READ, elem_node, flatten=True))

    op2.par_loop(rhs, elements,
                 b(op2.INC, elem_node[op2.i[0]]),
                 coords(op2.READ, elem_node, flatten=True),
                 f(op2.READ, elem_node))

    solver = op2.Solver()
    solver.solve(mat, x, b)

    # Print solution
    if opt['print_output']:
        print "Expected solution: %s" % f.data
        print "Computed solution: %s" % x.data

    # Save output (if necessary)
    if opt['return_output']:
        return f.data, x.data
    if opt['save_output']:
        import pickle
        with open("mass2d.out", "w") as out:
            pickle.dump((f.data, x.data), out)

parser = utils.parser(group=True, description=__doc__)
parser.add_argument('--print-output', action='store_true', help='Print output')
parser.add_argument('-r', '--return-output', action='store_true',
                    help='Return output for testing')
parser.add_argument('-s', '--save-output',
                    action='store_true',
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
        cProfile.run('main(opt)', filename='mass2d_ffc.cprofile')
    else:
        main(opt)
