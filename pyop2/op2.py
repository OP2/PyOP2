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

from pyop2.configuration import configuration
from pyop2.logger import debug, info, warning, error, critical, set_log_level
from pyop2.mpi import MPI, COMM_WORLD, collective

from pyop2.sequential import par_loop, Kernel  # noqa: F401
from pyop2.sequential import READ, WRITE, RW, INC, MIN, MAX  # noqa: F401
from pyop2.base import ON_BOTTOM, ON_TOP, ON_INTERIOR_FACETS, ALL  # noqa: F401
from pyop2.sequential import Set, ExtrudedSet, MixedSet, Subset, DataSet, MixedDataSet  # noqa: F401
from pyop2.sequential import Map, ComposedMap, MixedMap, Sparsity, Halo  # noqa: F401
from pyop2.sequential import Global, GlobalDataSet        # noqa: F401
from pyop2.sequential import Dat, MixedDat, DatView, Mat  # noqa: F401
from pyop2.sequential import ParLoop as SeqParLoop
from pyop2.pyparloop import ParLoop as PyParLoop

import types
import loopy

__all__ = ['configuration', 'READ', 'WRITE', 'RW', 'INC', 'MIN', 'MAX',
           'ON_BOTTOM', 'ON_TOP', 'ON_INTERIOR_FACETS', 'ALL',
           'debug', 'info', 'warning', 'error', 'critical', 'initialised',
           'set_log_level', 'MPI', 'init', 'exit', 'Kernel', 'Set', 'ExtrudedSet',
           'MixedSet', 'Subset', 'DataSet', 'GlobalDataSet', 'MixedDataSet',
           'Halo', 'Dat', 'MixedDat', 'Mat', 'Global', 'Map', 'ComposedMap', 'MixedMap',
           'Sparsity', 'par_loop', 'ParLoop',
           'DatView']


def ParLoop(kernel, *args, **kwargs):
    if isinstance(kernel, types.FunctionType):
        return PyParLoop(kernel, *args, **kwargs)
    else:
        return SeqParLoop(kernel, *args, **kwargs)


_initialised = False

# turn off loopy caching because pyop2 kernels are cached already
loopy.set_caching_enabled(False)


def initialised():
    """Check whether PyOP2 has been yet initialised but not yet finalised."""
    return _initialised


@collective
def init(**kwargs):
    """Initialise PyOP2: select the backend and potentially other configuration
    options.

    :arg debug:     The level of debugging output.
    :arg comm:      The MPI communicator to use for parallel communication,
                    defaults to `MPI_COMM_WORLD`
    :arg log_level: The log level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

    For debugging purposes, `init` accepts all keyword arguments
    accepted by the PyOP2 :class:`Configuration` object, see
    :meth:`Configuration.__init__` for details of further accepted
    options.

    .. note::
       Calling ``init`` again with a different backend raises an exception.
       Changing the backend is not possible. Calling ``init`` again with the
       same backend or not specifying a backend will update the configuration.
       Calling ``init`` after ``exit`` has been called is an error and will
       raise an exception.
    """
    global _initialised
    configuration.reconfigure(**kwargs)

    set_log_level(configuration['log_level'])

    _initialised = True


@atexit.register
@collective
def exit():
    """Exit OP2 and clean up"""
    if configuration['print_cache_size'] and COMM_WORLD.rank == 0:
        from caching import report_cache, Cached, ObjectCached
        print('**** PyOP2 cache sizes at exit ****')
        report_cache(typ=ObjectCached)
        report_cache(typ=Cached)
    configuration.reset()
    global _initialised
    _initialised = False
