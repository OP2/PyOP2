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

"""The PyOP2 API specification."""

import atexit

import backends
import configuration as cfg
import base
from base import READ, WRITE, RW, INC, MIN, MAX, IdentityMap, i
from logger import debug, info, warning, error, critical, set_log_level
from mpi import MPI, collective
from utils import validate_type
from exceptions import MatTypeError, DatTypeError

__all__ = ['cfg', 'READ', 'WRITE', 'RW', 'INC', 'MIN', 'MAX', 'IdentityMap',
           'i', 'debug', 'info', 'warning', 'error', 'critical',
           'set_log_level', 'MPI', 'init', 'exit', 'Kernel', 'Set', 'DataSet',
           'Halo', 'Dat', 'Mat', 'Const', 'Global', 'Map', 'Sparsity',
           'Solver', 'par_loop', 'solve']


@collective
def init(**kwargs):
    """Initialise OP2: select the backend and potentially other configuration
    options.

    :arg backend: Set the hardware-specific backend. Current choices
     are ``"sequential"``, ``"openmp"``, ``"opencl"`` and ``"cuda"``.
    :arg debug: The level of debugging output.
    :arg comm: The MPI communicator to use for parallel communication,
               defaults to `MPI_COMM_WORLD`

    .. note::
       Calling ``init`` again with a different backend raises an exception.
       Changing the backend is not possible. Calling ``init`` again with the
       same backend or not specifying a backend will update the configuration.
       Calling ``init`` after ``exit`` has been called is an error and will
       raise an exception.
    """
    backend = backends.get_backend()
    if backend == 'pyop2.finalised':
        raise RuntimeError("Calling init() after exit() is illegal.")
    if 'backend' in kwargs and backend not in ('pyop2.void', 'pyop2.' + kwargs['backend']):
        raise RuntimeError("Changing the backend is not possible once set.")
    cfg.configure(**kwargs)
    set_log_level(cfg['log_level'])
    if backend == 'pyop2.void':
        backends.set_backend(cfg.backend)
        backends._BackendSelector._backend._setup()
        if 'comm' in kwargs:
            backends._BackendSelector._backend.MPI.comm = kwargs['comm']
        global MPI
        MPI = backends._BackendSelector._backend.MPI  # noqa: backend override


@atexit.register
@collective
def exit():
    """Exit OP2 and clean up"""
    cfg.reset()
    if backends.get_backend() != 'pyop2.void':
        backends.unset_backend()


class Kernel(base.Kernel):
    __metaclass__ = backends._BackendSelector


class Set(base.Set):
    __metaclass__ = backends._BackendSelector


class DataSet(base.DataSet):
    __metaclass__ = backends._BackendSelector


@collective
class Halo(base.Halo):
    __metaclass__ = backends._BackendSelector


class Dat(base.Dat):
    __metaclass__ = backends._BackendSelector


class Mat(base.Mat):
    __metaclass__ = backends._BackendSelector


class Const(base.Const):
    __metaclass__ = backends._BackendSelector


class Global(base.Global):
    __metaclass__ = backends._BackendSelector


class Map(base.Map):
    __metaclass__ = backends._BackendSelector


class Sparsity(base.Sparsity):
    __metaclass__ = backends._BackendSelector


class Solver(base.Solver):
    __metaclass__ = backends._BackendSelector


@collective
def par_loop(kernel, iterset, *args):
    """Invocation of an OP2 kernel

    :arg kernel: The :class:`Kernel` to be executed.
    :arg iterset: The iteration :class:`Set` over which the kernel should be
                  executed.
    :arg \*args: One or more :class:`base.Arg`\s constructed from a
                 :class:`Global`, :class:`Dat` or :class:`Mat` using the call
                 syntax and passing in an optionally indexed :class:`Map`
                 through which this :class:`base.Arg` is accessed and the
                 :class:`base.Access` descriptor indicating how the
                 :class:`Kernel` is going to access this data (see the example
                 below). These are the global data structures from and to
                 which the kernel will read and write.

    .. warning ::
        It is the caller's responsibility that the number and type of all
        :class:`base.Arg`\s passed to the :func:`par_loop` match those expected
        by the :class:`Kernel`. No runtime check is performed to ensure this!

    If a :func:`par_loop` argument indexes into a :class:`Map` using an
    :class:`base.IterationIndex`, this implies the use of a local
    :class:`base.IterationSpace` of a size given by the arity of the
    :class:`Map`. It is an error to have several arguments using local
    iteration spaces of different size.

    :func:`par_loop` invocation is illustrated by the following example ::

      pyop2.par_loop(mass, elements,
                     mat((elem_node[pyop2.i[0]]), elem_node[pyop2.i[1]]), pyop2.INC),
                     coords(elem_node, pyop2.READ))

    This example will execute the :class:`Kernel` ``mass`` over the
    :class:`Set` ``elements`` executing 3x3 times for each
    :class:`Set` member, assuming the :class:`Map` ``elem_node`` is of arity 3.
    The :class:`Kernel` takes four arguments, the first is a :class:`Mat` named
    ``mat``, the second is a field named ``coords``. The remaining two arguments
    indicate which local iteration space point the kernel is to execute.

    A :class:`Mat` requires a pair of :class:`Map` objects, one each
    for the row and column spaces. In this case both are the same
    ``elem_node`` map. The row :class:`Map` is indexed by the first
    index in the local iteration space, indicated by the ``0`` index
    to :data:`pyop2.i`, while the column space is indexed by
    the second local index.  The matrix is accessed to increment
    values using the ``pyop2.INC`` access descriptor.

    The ``coords`` :class:`Dat` is also accessed via the ``elem_node``
    :class:`Map`, however no indices are passed so all entries of
    ``elem_node`` for the relevant member of ``elements`` will be
    passed to the kernel as a vector.
    """
    return backends._BackendSelector._backend.par_loop(kernel, iterset, *args)


@collective
@validate_type(('M', base.Mat, MatTypeError),
               ('x', base.Dat, DatTypeError),
               ('b', base.Dat, DatTypeError))
def solve(M, x, b):
    Solver().solve(M, x, b)
