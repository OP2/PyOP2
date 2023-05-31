import atexit
import sys

import petsc4py


class LazyPETSc:
    """Wrapper around petsc4py.PETSc to enable delayed PETSc init

    TODO write more explanation here
    """

    def __init__(self):
        self.PETSc = None

    def __getattr__(self, name):
        if not self.PETSc:
            print("INITIALISING PETSC", flush=True)
            breakpoint()
            petsc4py.init(sys.argv)
            from petsc4py import PETSc

            # start logging
            event = PETSc.Log.Event("pyop2")
            event.begin()
            atexit.register(lambda: event.end())

            self.PETSc = PETSc
        return getattr(self.PETSc, name)


PETSc = LazyPETSc()
