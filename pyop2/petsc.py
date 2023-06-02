import atexit
import functools
import os
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
            petsc4py.init(sys.argv)
            from petsc4py import PETSc

            # start logging
            event = PETSc.Log.Event("PETSc")
            event.begin()
            atexit.register(lambda: event.end())

            self.PETSc = PETSc
        return getattr(self.PETSc, name)


PETSc = LazyPETSc()


@functools.lru_cache()
def get_petsc_variables():
    """Get dict of PETSc environment variables from the file:
    $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/petscvariables

    The result is memoized to avoid constantly reading the file.
    """
    config = petsc4py.get_config()
    path = [config["PETSC_DIR"], config["PETSC_ARCH"], "lib/petsc/conf/petscvariables"]
    variables_path = os.path.join(*path)
    with open(variables_path) as fh:
        # Split lines on first '=' (assignment)
        splitlines = (line.split("=", maxsplit=1) for line in fh.readlines())
    return {k.strip(): v.strip() for k, v in splitlines}
