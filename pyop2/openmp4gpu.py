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
from base import WRITE, INC
import pyop2.openmp4 as omp4
from exceptions import *
import host
from mpi import collective
from petsc_base import *
from profiling import timed_region
from host import Kernel, Arg  # noqa: needed by BackendSelector
from utils import as_tuple, cached_property
from optimizer import optimize_wrapper, optimize_kernel
from configuration import configuration


def _detect_openmp_flags():
    p = Popen(['mpicc', '--version'], stdout=PIPE, shell=False)
    _version, _ = p.communicate()
    if _version.find('Free Software Foundation') != -1:
        return '-fopenmp', '-lgomp'
    elif _version.find('Intel Corporation') != -1:
        return '-openmp', '-liomp5'
    elif _version.find('clang') != -1:
        return '-fopenmp=libomp', ''
    else:
        warning('Unknown mpicc version:\n%s' % _version)
        return '', ''


class Arg(host.Arg):

    def c_print_dat(self):
        if self._is_mat:
            call_args = ["TODO"]
        else:
            call_args = ["""for(int i=0; i<%(size)s; i++) { printf("---> %%f\\n", %(name)s[i]); %(name)s[i] = 999; }""" % \
                         { "name": self.c_arg_name(i),
                           "size": str(len(self.data[i].data) * self.data[i].cdim)} \
                         for i in range(len(self.data))]
        return call_args

    def c_offload_dat(self):
        if self._is_mat:
            call_args = ["TODO"]
        else:
            call_args = [self.c_arg_name(i) + "[:%(size)s]" % {'size': str(len(self.data[i].data) * self.data[i].cdim)}
                         for i in range(len(self.data))]
        return call_args

    def c_offload_map(self):
        call_args = []
        if self._is_indirect or self._is_mat:
            for i, map in enumerate(as_tuple(self.map, Map)):
                for j, m in enumerate(map):
                    call_args += [self.c_map_name(i, j) + "[:%(size)s]" % {'size': str(len(m.values_with_halo) * m.arity)}]
        return call_args

    def c_offload_offset(self):
        maps = as_tuple(self.map, Map)
        call_args = []
        for i, map in enumerate(maps):
            if not map.iterset._extruded:
                continue
            for j, m in enumerate(map):
                call_args += [self.c_offset_name(i, j)]
        return ', '.join(call_args)


class JITModule(host.JITModule):
    ompflag, omplib = _detect_openmp_flags()
    _cppargs = [os.environ.get('OMP_CXX_FLAGS') or ompflag]
    _libraries = [ompflag] + [os.environ.get('OMP_LIBS') or omplib]
    _system_headers = ['#include <omp.h>']

    _wrapper = """
double %(wrapper_name)s(int start, int end,
                      %(ssinds_arg)s
                      %(wrapper_args)s
                      %(const_args)s
                      %(off_args)s
                      %(layer_arg)s
                      %(other_args)s) {
  %(user_code)s
  %(timer_declare)s
  %(offload_one)s
  {
      %(timer_start)s
      %(times_loop_start)s
      %(parallel_pragma_two)s
      for ( int n = start; n < end; n++ ) {
        %(wrapper_decs)s;
        %(const_inits)s;
        %(map_decl)s
        %(vec_decs)s;
        int i = %(index_expr)s;
        %(vec_inits)s;
        %(map_init)s;
        %(parallel_pragma_three)s
        %(extr_loop)s
        %(map_bcs_m)s;
        %(buffer_decl)s;
        %(buffer_gather)s

        // Kernel invocation
        %(kernel_name)s(%(kernel_args)s);

        %(itset_loop_body)s
        %(map_bcs_p)s;
        %(apply_offset)s;
        %(extr_loop_close)s
      }
      %(times_loop_end)s
      %(timer_stop)s
  }
  %(timer_end)s
}
"""

    def set_argtypes(self, iterset, *args):
        argtypes = [ctypes.c_int, ctypes.c_int]
        offset_args = []
        if isinstance(iterset, Subset):
            argtypes.append(iterset._argtype)
        for arg in args:
            if arg._is_mat:
                argtypes.append(arg.data._argtype)
            else:
                for d in arg.data:
                    # from IPython import embed; embed()
                    # argtypes.append(d._argtype)
                    argtypes.append(d.data.ctypes.data_as(ctypes.c_void_p))
            if arg._is_indirect or arg._is_mat:
                maps = as_tuple(arg.map, Map)
                for map in maps:
                    for m in map:
                        argtypes.append(m._argtype)
                        if m.iterset._extruded:
                            offset_args.append(ctypes.c_void_p)

        for c in Const._definitions():
            argtypes.append(c._argtype)

        argtypes.extend(offset_args)

        if iterset._extruded:
            argtypes.append(ctypes.c_int)
            argtypes.append(ctypes.c_int)

        if configuration['hpc_profiling']:
            argtypes.append(ndpointer(np.dtype('float64'), shape=(8,)))

        self._argtypes = argtypes

    def generate_code(self):

        # Most of the code to generate is the same as that for sequential
        code_dict = super(JITModule, self).generate_code()

        _read_args = []
        _inc_args = []
        _write_args = []
        _print_args = []
        for arg in self._args:
            # Offloading for dats based on access type:
            # READ  'to'
            # INC   'tofrom'
            # WRITE 'from'
            if arg.access in [WRITE]:
                _write_args += arg.c_offload_dat()
                _print_args += arg.c_print_dat()
            elif arg.access in [INC]:
                _inc_args += arg.c_offload_dat()
                _print_args += arg.c_print_dat()
            else:
                _read_args += arg.c_offload_dat()
                _print_args += arg.c_print_dat()
            # Offload all maps as 'to'
            _read_args += arg.c_offload_map()

        _read_args = "map(to: " + ', '.join(_read_args) + ")" if _read_args else ""
        _inc_args = "map(tofrom: " + ', '.join(_inc_args) + ")" if _inc_args else ""
        _write_args = "map(from: " + ', '.join(_write_args) + ")" if _write_args else ""
        _print_args = "\n".join(_print_args)

        _offload_one = "#pragma omp target data %(read_args)s %(inc_args)s %(write_args)s" % \
                       {'read_args': _read_args,
                        'inc_args': _inc_args,
                        'write_args': _write_args}

        # _offload_one = """%(print_args)s
        #                   double *dd = arg0_0;
        #                   printf("Just before offload\\n");
        #                   printf("Address of arg0_0 is: 0x%%llx\\n", *&arg0_0);
        #                   printf("Address of arg0_0 is: 0x%%llx\\n", arg0_0);
        #                   printf("Address of arg0_0 is: 0x%%llx\\n", &arg0_0);
        #                   printf("Value of arg0_0[0] is: %%f\\n", (*&arg0_0)[0]);
        #                   %(offld)s
        #                """ % {"print_args": "", # _print_args,
        #                       "offld": _offload_one}

        # If the work is performed on an extruded mesh then we have to also offload the offset
        # associated with each map
        # if self._itspace._extruded:
        #     _read_args = ", ".join([_read_args] + [arg.c_offload_offset() for arg in self._args
        #                                            if arg._uses_itspace or arg._is_vec_map])
        
        code_dict.update({'offload_one': _offload_one})

        # This is a good place to apply some application level optimizations
        optimize_wrapper(self, code_dict)
        return code_dict

    def get_c_code(self, kernel_code, wrapper_code):
        # This is a good place to apply some application level optimizations
        kernel_code = optimize_kernel(self, kernel_code)
        return super(JITModule, self).get_c_code(kernel_code, wrapper_code)

    def backend_flags(self, cppargs, more_args, ldargs):
        # Most of the code to generate is the same as that for sequential
        super(JITModule, self).backend_flags(cppargs, more_args, ldargs)
        cppargs += ["-fopenmp=libomp", "-O3", "-omptargets=nvptx64sm_35-nvidia-linux", '-save-temps', '-v']
        # Add custom number of teams and threads:
        cppargs += ["-DTEAMS=%(teams)s" % {'teams': configuration["teams"]}]
        cppargs += ["-DTHREADS=%(threads)s" % {'threads': configuration["threads"]}]
        # Include LOMP
        # TODO: Replace hardcoded path with lomp path function in utils
        cppargs += ["-I/localhd/gbercea/lomp/lomp/source/"]
        ldargs += ["-L/localhd/gbercea/lomp/lomp/source/lib64"]
        ldargs += ["-Wl,-rpath,/localhd/gbercea/lomp/lomp/source/lib64"]

    @collective
    def compile(self, argtypes=None, restype=None):
        fun = super(JITModule, self).compile(argtypes, restype)
        if configuration["hpc_profiling"]:
            print configuration["basename"]
            with open(configuration["basename"]+".err") as h:
                spill_info = ""
                for line in h:
                    if "bytes spill stores" in line:
                        spill_info = line
                    if "registers" in line and "nvlink" in line:
                        configuration["nvlink_info"] = " ".join((line.split(":")[1] + spill_info).split())
                        break
        return fun

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
        offset_args = []
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
                        if m.iterset._extruded:
                            offset_args.append(m.offset.ctypes.data)

        for c in Const._definitions():
            arglist.append(c._data.ctypes.data)

        arglist.extend(offset_args)

        if iterset._extruded:
            region = self.iteration_region
            # Set up appropriate layer iteration bounds
            if region is ON_BOTTOM:
                arglist.append(0)
                arglist.append(1)
            elif region is ON_TOP:
                arglist.append(iterset.layers - 2)
                arglist.append(iterset.layers - 1)
            elif region is ON_INTERIOR_FACETS:
                arglist.append(0)
                arglist.append(iterset.layers - 2)
            else:
                arglist.append(0)
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
        # Return the host-side OpenMP 4.0 version of the function.
        return omp4.JITModule(self.kernel, self.it_space, *self.args,
                         direct=self.is_direct, iterate=self.iteration_region, halo=True)

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


def _setup():
    pass
