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

import backends
import configuration as cfg
import op_lib_core as core
import base
from base import READ, WRITE, RW, INC, MIN, MAX, IdentityMap, i


def init(**kwargs):
    """Initialise OP2: select the backend."""
    cfg.configure(**kwargs)
    backends.set_backend(cfg.backend)
    core.op_init(args=None, diags=0)

def exit():
    """Exit OP2 and clean up"""
    cfg.reset()
    if backends.get_backend() != 'pyop2.void':
        core.op_exit()
        backends.unset_backend()

class IterationSpace(base.IterationSpace):
    __metaclass__ = backends._BackendSelector

class Kernel(base.Kernel):
    __metaclass__ = backends._BackendSelector

class Set(base.Set):
    __metaclass__ = backends._BackendSelectorWithH5

class Dat(base.Dat):
    __metaclass__ = backends._BackendSelectorWithH5

class Mat(base.Mat):
    __metaclass__ = backends._BackendSelector

class Const(base.Const):
    __metaclass__ = backends._BackendSelectorWithH5

class Global(base.Global):
    __metaclass__ = backends._BackendSelector

class Map(base.Map):
    __metaclass__ = backends._BackendSelectorWithH5

class Sparsity(base.Sparsity):
    __metaclass__ = backends._BackendSelector

def par_loop(kernel, it_space, *args):
    """Invocation of an OP2 kernel

    :arg kernel: The :class:`Kernel` to be executed.
    :arg it_space: The iteration space over which the kernel should be executed. The primary iteration space will be a :class:`Set`. If a local iteration space is required, then this can be provided in brackets. For example, to iterate over a :class:`Set` named ``elements`` assembling a 3x3 local matrix at each entry, the ``it_space`` argument should be ``elements(3,3)``.
    :arg \*args: One or more objects of type :class:`Global`, :class:`Dat` or :class:`Mat` which are the global data structures from and to which the kernel will read and write.

    ``par_loop`` invocation is illustrated by the following example::

      pyop2.par_loop(mass, elements(3,3),
             mat((elem_node[pyop2.i[0]]), elem_node[pyop2.i[1]]), pyop2.INC),
             coords(elem_node, pyop2.READ))

    This example will execute the :class:`Kernel` ``mass`` over the
    :class:`Set` ``elements`` executing 3x3 times for each
    :class:`Set` member. The :class:`Kernel` takes four arguments, the
    first is a :class:`Mat` named ``mat``, the second is a field named
    `coords`. The remaining two arguments indicate which local
    iteration space point the kernel is to execute.

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
    return backends._BackendSelector._backend.par_loop(kernel, it_space, *args)

def solve(M, b, x):
    """Solve a the matrix equation.

    :arg M: The :class:`Mat` containing the matrix.
    :arg b: The :class:`Dat` containing the RHS.
    :arg x: The :class:`Dat` to receive the solution.
    """
    return backends._BackendSelector._backend.solve(M, b, x)
