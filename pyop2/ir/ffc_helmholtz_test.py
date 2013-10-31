from ufl import *
from pyop2 import op2
from ffc import default_parameters, compile_form

from kernel_plan import KernelPlan


def main():

    # Set up Helmholtz problem (only left hand side)
    P = FiniteElement("Lagrange", "triangle", 2)

    v = TestFunction(P)
    u = TrialFunction(P)

    a = (dot(grad(v), grad(u)) - v * u) * dx

    # Set up compiler parameters
    ffc_parameters = default_parameters()
    ffc_parameters['write_file'] = False
    ffc_parameters['format'] = 'pyop2'
    ffc_parameters["pyop2-ir"] = True

    kernel = compile_form(a, prefix="helmholtz", parameters=ffc_parameters)

    # Create a plan for executing this kernel
    plan = KernelPlan(kernel)

    opts = {}
    opts["interchange"] = ()  # (1, 2, 0)
    opts["licm"] = True
    opts["pad_and_align"] = False  # True
    opts["outer-product tiling"] = 0

    plan.plan_cpu("AVX", "INTEL", opts)  # FIXME: backend

    embed()


if __name__ == '__main__':
    op2.init()
    main()
