from ufl import *
from pyop2 import op2
from ffc import default_parameters, compile_form

from kernel_plan import KernelPlan


def main():

    # Set up MixedPoisson problem (only left hand side)
    q = 1
    BDM = FiniteElement("Brezzi-Douglas-Marini", triangle, q)
    DG = FiniteElement("Discontinuous Lagrange", triangle, q - 1)
    mixed_element = BDM * DG
    (sigma, u) = TrialFunctions(mixed_element)
    (tau, w) = TestFunctions(mixed_element)
    a = (inner(sigma, tau) - div(tau) * u + div(sigma) * w) * dx

    # Set up compiler parameters
    ffc_parameters = default_parameters()
    ffc_parameters['write_file'] = False
    ffc_parameters['format'] = 'pyop2'
    ffc_parameters["pyop2-ir"] = True

    kernel = compile_form(a, prefix="helmholtz", parameters=ffc_parameters)

    # Create a plan for executing this kernel
    plan = KernelPlan(kernel)
    plan.plan_cpu("AVX", "INTEL")  # FIXME: backend


if __name__ == '__main__':
    op2.init()
    main()
