from ufl import *
from pyop2.ffc_interface import compile_form


def main():

    # Set up Helmholtz problem (only left hand side)
    P = FiniteElement("Lagrange", "triangle", 1)

    v = TestFunction(P)
    u = TrialFunction(P)

    a = (dot(grad(v), grad(u)) - v * u) * dx

    kernel = compile_form(a, "helmholtz")

    print kernel


if __name__ == '__main__':
    main()
