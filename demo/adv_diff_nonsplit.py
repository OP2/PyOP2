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

"""PyOP2 P1 advection-diffusion demo

This demo solves the advection-diffusion equation and is advanced in time using
a theta scheme with theta = 0.5.

The domain read in from a triangle file.

This demo requires the pyop2 branch of ffc, which can be obtained with:

bzr branch lp:~mapdes/ffc/pyop2

This may also depend on development trunk versions of other FEniCS programs.

FEniCS Viper is also required and is used to visualise the solution.
"""

import os
import numpy as np

from pyop2 import op2, utils
from triangle_reader import read_triangle

_kerneldir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernels')


def viper_shape(array):
    """Flatten a numpy array into one dimension to make it suitable for
    passing to Viper."""
    return array.reshape((array.shape[0]))


def main(opt):
    dt = 0.0001

    if opt['firedrake']:
        # Set up finite element problem
        from firedrake.ffc_interface import compile_form
        from ufl import *

        T = FiniteElement("Lagrange", "triangle", 1)
        V = VectorElement("Lagrange", "triangle", 1)

        p = TrialFunction(T)
        q = TestFunction(T)
        t = Coefficient(T)
        u = Coefficient(V)

        diffusivity = 0.1

        M = p * q * dx

        d = dt * (diffusivity * dot(grad(q), grad(p)) - dot(grad(q), u) * p) * dx

        a = M + 0.5 * d
        L = action(M - 0.5 * d, t)

        # Generate code for mass and rhs assembly.

        lhs, = compile_form(a, "lhs")
        rhs, = compile_form(L, "rhs")
        if opt['update_kernels']:
            with open(os.path.join(_kerneldir, 'adv_diff.c'), 'w') as f:
                f.write(lhs._code)
            with open(os.path.join(_kerneldir, 'adv_diff_rhs.c'), 'w') as f:
                f.write(rhs._code)
    else:
        with open(os.path.join(_kerneldir, 'adv_diff.c')) as f:
            lhs = op2.Kernel(f.read(), "lhs_cell_integral_0_otherwise")
        with open(os.path.join(_kerneldir, 'adv_diff_rhs.c')) as f:
            rhs = op2.Kernel(f.read(), "rhs_cell_integral_0_otherwise")

    # Set up simulation data structures

    valuetype = np.float64

    nodes, coords, elements, elem_node = read_triangle(opt['mesh'])

    num_nodes = nodes.size

    sparsity = op2.Sparsity((nodes, nodes), (elem_node, elem_node), "sparsity")
    mat = op2.Mat(sparsity, valuetype, "mat")

    tracer_vals = np.zeros(num_nodes, dtype=valuetype)
    tracer = op2.Dat(nodes, tracer_vals, valuetype, "tracer")

    b_vals = np.zeros(num_nodes, dtype=valuetype)
    b = op2.Dat(nodes, b_vals, valuetype, "b")

    velocity_vals = np.asarray([1.0, 0.0] * num_nodes, dtype=valuetype)
    velocity = op2.Dat(nodes ** 2, velocity_vals, valuetype, "velocity")

    # Set initial condition

    i_cond_code = """void i_cond(double *c, double *t)
{
  double A   = 0.1; // Normalisation
  double D   = 0.1; // Diffusivity
  double pi  = 3.14159265358979;
  double x   = c[0]-(0.45+%(T)f);
  double y   = c[1]-0.5;
  double r2  = x*x+y*y;

  *t = A*(exp(-r2/(4*D*%(T)f))/(4*pi*D*%(T)f));
}
"""

    T = 0.01

    i_cond = op2.Kernel(i_cond_code % {'T': T}, "i_cond")

    op2.par_loop(i_cond, nodes,
                 coords(op2.READ, flatten=True),
                 tracer(op2.WRITE))

    # Assemble and solve

    if opt['visualize']:
        vis_coords = np.asarray([[x, y, 0.0] for x, y in coords.data_ro],
                                dtype=np.float64)
        import viper
        v = viper.Viper(x=viper_shape(tracer.data_ro),
                        coordinates=vis_coords, cells=elem_node.values)

    solver = op2.Solver()

    while T < 0.015:

        mat.zero()
        op2.par_loop(lhs, elements,
                     mat(op2.INC, (elem_node[op2.i[0]], elem_node[op2.i[1]])),
                     coords(op2.READ, elem_node, flatten=True),
                     velocity(op2.READ, elem_node, flatten=True))

        b.zero()
        op2.par_loop(rhs, elements,
                     b(op2.INC, elem_node[op2.i[0]]),
                     coords(op2.READ, elem_node, flatten=True),
                     tracer(op2.READ, elem_node),
                     velocity(op2.READ, elem_node, flatten=True))

        solver.solve(mat, tracer, b)

        if opt['visualize']:
            v.update(viper_shape(tracer.data_ro))

        T = T + dt

    if opt['print_output'] or opt['test_output'] or opt['return_output']:
        analytical_vals = np.zeros(num_nodes, dtype=valuetype)
        analytical = op2.Dat(nodes, analytical_vals, valuetype, "analytical")

        i_cond = op2.Kernel(i_cond_code % {'T': T}, "i_cond")

        op2.par_loop(i_cond, nodes,
                     coords(op2.READ, flatten=True),
                     analytical(op2.WRITE))

    # Print error w.r.t. analytical solution
    if opt['print_output']:
        print "Expected - computed  solution: %s" % (tracer.data - analytical.data)

    if opt['test_output'] or opt['return_output']:
        if opt['firedrake']:
            l2norm = dot(t - a, t - a) * dx
            l2_kernel, = compile_form(l2norm, "error_norm")
            if opt['update_kernels']:
                with open(os.path.join(_kerneldir, 'error_norm.c'), 'w') as f:
                    f.write(l2_kernel._code)
        else:
            with open(os.path.join(_kerneldir, 'error_norm.c')) as f:
                l2_kernel = op2.Kernel(f.read(), "error_norm_cell_integral_0_otherwise")
        result = op2.Global(1, [0.0])
        op2.par_loop(l2_kernel, elements,
                     result(op2.INC),
                     coords(op2.READ, elem_node, flatten=True),
                     tracer(op2.READ, elem_node),
                     analytical(op2.READ, elem_node))
        if opt['test_output']:
            with open("adv_diff.%s.out" % os.path.split(opt['mesh'])[-1], "w") as out:
                out.write(str(result.data[0]) + "\n")
        if opt['return_output']:
            return result.data[0]


parser = utils.parser(group=True, description=__doc__)
parser.add_argument('-m', '--mesh', required=True,
                    help='Base name of triangle mesh \
                          (excluding the .ele or .node extension)')
parser.add_argument('-v', '--visualize', action='store_true',
                    help='Visualize the result using viper')
parser.add_argument('--print-output', action='store_true', help='Print output')
parser.add_argument('-r', '--return-output', action='store_true',
                    help='Return output for testing')
parser.add_argument('-t', '--test-output', action='store_true',
                    help='Save output for testing')
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
        filename = 'adv_diff_nonsplit.%s.cprofile' % os.path.split(opt['mesh'])[-1]
        cProfile.run('main(opt)', filename=filename)
    else:
        main(opt)
