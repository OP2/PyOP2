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

"""OP2 sequential backend."""

import ctypes
from numpy.ctypeslib import ndpointer

from base import ON_BOTTOM, ON_TOP, ON_INTERIOR_FACETS
from exceptions import *
import host
from mpi import collective
from petsc_base import *
from profiling import timed_region
from host import Kernel, Arg  # noqa: needed by BackendSelector
from utils import as_tuple, cached_property
from optimizer import optimize_wrapper
from wrapper import compose_openmp4_wrapper


def _detect_openmp_flags():
    p = Popen(['mpicc', '--version'], stdout=PIPE, shell=False)
    _version, _ = p.communicate()
    if _version.find('Free Software Foundation') != -1:
        return '-fopenmp', '-lgomp'
    elif _version.find('Intel Corporation') != -1:
        return '-openmp', '-liomp5'
    elif _version.find('clang') != -1:
        return '-fopenmp=libomp', '-lomp'
    else:
        warning('Unknown mpicc version:\n%s' % _version)
        return '', ''


class JITModule(host.JITModule):
    ompflag, omplib = _detect_openmp_flags()
    _cppargs = [os.environ.get('OMP_CXX_FLAGS') or ompflag]
    _libraries = [ompflag] + [os.environ.get('OMP_LIBS') or omplib]
    _system_headers = ['#include <omp.h>']

    _wrapper = compose_openmp4_wrapper()

    def set_argtypes(self, iterset, *args):
        argtypes = [ctypes.c_int, ctypes.c_int]
        if isinstance(iterset, Subset):
            argtypes.append(iterset._argtype)
        for arg in args:
            if arg._is_mat:
                argtypes.append(arg.data._argtype)
            else:
                for d in arg.data:
                    argtypes.append(d._argtype)
            if arg._is_indirect or arg._is_mat:
                maps = as_tuple(arg.map, Map)
                for map in maps:
                    for m in map:
                        argtypes.append(m._argtype)

        for c in Const._definitions():
            argtypes.append(c._argtype)

        if iterset._extruded:
            argtypes.append(ctypes.c_int)
            argtypes.append(ctypes.c_int)
            argtypes.append(ctypes.c_int)

        if configuration['hpc_profiling']:
            argtypes.append(ndpointer(np.dtype('float64'), shape=(8,)))

        self._argtypes = argtypes

    def generate_code(self):
        code_dict = super(JITModule, self).generate_code()
        print "=> Running OPENMP 4.0 on HOST"
        # Offloading is not required
        code_dict.update({'offload_one': ""})

        # Init pragma placeholders
        code_dict.update({'parallel_pragma_one': ""})
        code_dict.update({'parallel_pragma_two': ""})
        code_dict.update({'parallel_pragma_three': ""})

        optimize_wrapper(self, code_dict, host=True)
        return code_dict

    def backend_flags(self, cppargs, more_args, ldargs):
        super(JITModule, self).backend_flags(cppargs, more_args, ldargs)
        cppargs += ["-mcpu=pwr8", "-mtune=pwr8", "-fopenmp=libomp", "-O3", '-v']
        # Include LIBOMP
        cppargs += ["-I" + os.environ.get('LIBOMP_LIB') or ""]
        ldargs += ["-L" + os.environ.get('LIBOMP_LIB') or ""]
        ldargs += ["-Wl,-rpath," + os.environ.get('LIBOMP_LIB') or ""]

    @classmethod
    def _cache_key(cls, kernel, itspace, *args, **kwargs):
        key = super(JITModule, cls)._cache_key(kernel, itspace, *args, **kwargs)
        halo = kwargs.get("halo", None)
        if halo is not None:
            key += ((halo,))
        return key


class ParLoop(host.ParLoop):

    def prepare_arglist(self, iterset, *args):
        arglist = []
        if isinstance(iterset, Subset):
            arglist.append(iterset._indices.ctypes.data)

        for arg in args:
            if arg._is_mat:
                arglist.append(arg.data.handle.handle)
            else:
                for d in arg.data:
                    # Cannot access a property of the Dat or we will force
                    # evaluation of the trace
                    arglist.append(d._data.ctypes.data)
            if arg._is_indirect or arg._is_mat:
                for map in arg._map:
                    for m in map:
                        arglist.append(m._values.ctypes.data)

        for c in Const._definitions():
            arglist.append(c._data.ctypes.data)

        if iterset._extruded:
            region = self.iteration_region
            # Set up appropriate layer iteration bounds
            if region is ON_BOTTOM:
                arglist.append(0)
                arglist.append(1)
                arglist.append(iterset.layers - 1)
            elif region is ON_TOP:
                arglist.append(iterset.layers - 2)
                arglist.append(iterset.layers - 1)
                arglist.append(iterset.layers - 1)
            elif region is ON_INTERIOR_FACETS:
                arglist.append(0)
                arglist.append(iterset.layers - 2)
                arglist.append(iterset.layers - 2)
            else:
                arglist.append(0)
                arglist.append(iterset.layers - 1)
                arglist.append(iterset.layers - 1)

        if configuration['hpc_profiling']:
            arglist.append(np.zeros(8, dtype=np.float64))

        return arglist

    @cached_property
    def _jitmodule(self):
        return JITModule(self.kernel, self.it_space, *self.args,
                         direct=self.is_direct, iterate=self.iteration_region)

    @cached_property
    def _jitmodule_halo(self):
        return self._jitmodule

    @collective
    def _compute(self, part, fun, *arglist):
        time = 0.0
        with timed_region("ParLoop kernel"):
            # time = fun(*self._jit_args, argtypes=self._argtypes, restype=ctypes.c_double)
            time = fun(part.offset, part.offset + part.size, *arglist)
            if configuration['hpc_profiling']:
                ms = arglist[-1]
                return time, [m for m in ms]
        return time, np.zeros(8)


def generate_cell_wrapper(itspace, args, forward_args=(), kernel_name=None, wrapper_name=None):
    """Generates wrapper for a single cell. No iteration loop, but cellwise data is extracted.
    Cell is expected as an argument to the wrapper. For extruded, the numbering of the cells
    is columnwise continuous, bottom to top.

    :param itspace: :class:`IterationSpace` object. Can be built from
                    iteration :class:`Set` using pyop2.base.build_itspace
    :param args: :class:`Arg`s
    :param forward_args: To forward unprocessed arguments to the kernel via the wrapper,
                         give an iterable of strings describing their C types.
    :param kernel_name: Kernel function name
    :param wrapper_name: Wrapper function name

    :return: string containing the C code for the single-cell wrapper
    """

    direct = all(a.map is None for a in args)
    snippets = host.wrapper_snippets(itspace, args, kernel_name=kernel_name, wrapper_name=wrapper_name)

    if itspace._extruded:
        snippets['index_exprs'] = """int i = cell / nlayers;
    int j = cell % nlayers;"""
        snippets['nlayers_arg'] = ", int nlayers"
        snippets['extr_pos_loop'] = "{" if direct else "for (int j_0 = 0; j_0 < j; ++j_0) {"
    else:
        snippets['index_exprs'] = "int i = cell;"
        snippets['nlayers_arg'] = ""
        snippets['extr_pos_loop'] = ""

    snippets['wrapper_fargs'] = "".join("{1} farg{0}, ".format(i, arg) for i, arg in enumerate(forward_args))
    snippets['kernel_fargs'] = "".join("farg{0}, ".format(i) for i in xrange(len(forward_args)))

    template = """static inline void %(wrapper_name)s(%(wrapper_fargs)s%(wrapper_args)s%(const_args)s%(nlayers_arg)s, int cell)
{
    %(user_code)s
    %(wrapper_decs)s;
    %(const_inits)s;
    %(map_decl)s
    %(vec_decs)s;
    %(index_exprs)s
    %(vec_inits)s;
    %(map_init)s;
    %(extr_pos_loop)s
        %(apply_offset)s;
    %(extr_loop_close)s
    %(map_bcs_m)s;
    %(buffer_decl)s;
    %(buffer_gather)s
    %(kernel_name)s(%(kernel_fargs)s%(kernel_args)s);
    %(itset_loop_body)s
    %(map_bcs_p)s;
}
"""
    return template % snippets


def _setup():
    pass
