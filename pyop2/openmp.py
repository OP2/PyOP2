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

"""OP2 OpenMP backend."""

import os
import numpy as np
import math

from exceptions import *
from utils import *
import op_lib_core as core
import petsc_base
from petsc_base import *
import host
import device
from subprocess import Popen, PIPE

# hard coded value to max openmp threads
_max_threads = 32

def _detect_openmp_flags():
    p = Popen(['mpicc', '--version'], stdout=PIPE, shell=False)
    _version, _ = p.communicate()
    if _version.find('Free Software Foundation') != -1:
        return '-fopenmp', 'gomp'
    elif _version.find('Intel Corporation') != -1:
        return '-openmp', 'iomp5'
    else:
        from warnings import warn
        warn('Unknown mpicc version:\n%s' % _version)
        return '', ''

class Arg(host.Arg):

    def c_vec_name(self, idx=None):
        return self.c_arg_name() + "_vec[%s]" % (idx or 'tid')

    def c_kernel_arg_name(self, idx=None):
        return "p_%s[%s]" % (self.c_arg_name(), idx or 'tid')

    def c_global_reduction_name(self):
        return "%s_l[tid]" % self.c_arg_name()

    def c_local_tensor_name(self):
        return self.c_kernel_arg_name(str(_max_threads))

    def c_vec_dec(self):
        return ";\n%(type)s *%(vec_name)s[%(dim)s]" % \
                   {'type' : self.ctype,
                    'vec_name' : self.c_vec_name(str(_max_threads)),
                    'dim' : self.map.dim}

    def c_reduction_dec(self):
        return "%(type)s %(name)s_l[%(max_threads)s][%(dim)s]" % \
          {'type' : self.ctype,
           'name' : self.c_arg_name(),
           'dim' : self.data.cdim,
           # Ensure different threads are on different cache lines
           'max_threads' : _max_threads}

    def c_reduction_init(self):
        if self.access == INC:
            init = "(%(type)s)0" % {'type' : self.ctype}
        else:
            init = "%(name)s[i]" % {'name' : self.c_arg_name()}
        return "for ( int i = 0; i < %(dim)s; i++ ) %(name)s_l[tid][i] = %(init)s" % \
          {'dim' : self.data.cdim,
           'name' : self.c_arg_name(),
           'init' : init}

    def c_reduction_finalisation(self):
        d = {'gbl': self.c_arg_name(),
             'local': "%s_l[thread][i]" % self.c_arg_name()}
        if self.access == INC:
            combine = "%(gbl)s[i] += %(local)s" % d
        elif self.access == MIN:
            combine = "%(gbl)s[i] = %(gbl)s[i] < %(local)s ? %(gbl)s[i] : %(local)s" % d
        elif self.access == MAX:
            combine = "%(gbl)s[i] = %(gbl)s[i] > %(local)s ? %(gbl)s[i] : %(local)s" % d
        return """
        for ( int thread = 0; thread < nthread; thread++ ) {
            for ( int i = 0; i < %(dim)s; i++ ) %(combine)s;
        }""" % {'combine' : combine,
                'dim' : self.data.cdim}

# Parallel loop API

def par_loop(kernel, it_space, *args):
    """Invocation of an OP2 kernel with an access descriptor"""
    ParLoop(kernel, it_space, *args).compute()

class JITModule(host.JITModule):

    ompflag, omplib = _detect_openmp_flags()
    _cppargs = [os.environ.get('OMP_CXX_FLAGS') or ompflag]
    _libraries = [os.environ.get('OMP_LIBS') or omplib]
    _system_headers = ['omp.h']

    wrapper = """
void wrap_%(kernel_name)s__(
  PyObject* _boffset,
  PyObject* _nblocks,
  %(wrapper_args)s %(const_args)s,
  PyObject* _blkmap,
  PyObject* _offset,
  PyObject* _nelems) {

  int boffset = (int)PyInt_AsLong(_boffset);
  int nblocks = (int)PyInt_AsLong(_nblocks);
  int* blkmap = (int *)(((PyArrayObject *)_blkmap)->data);
  int* offset = (int *)(((PyArrayObject *)_offset)->data);
  int* nelems = (int *)(((PyArrayObject *)_nelems)->data);

  %(wrapper_decs)s;
  %(const_inits)s;
  %(local_tensor_decs)s;

  #ifdef _OPENMP
  int nthread = omp_get_max_threads();
  #else
  int nthread = 1;
  #endif

  %(reduction_decs)s;

  #pragma omp parallel default(shared)
  {
    int tid = omp_get_thread_num();
    %(reduction_inits)s;

    #pragma omp for schedule(static)
    for ( int __b = boffset; __b < (boffset + nblocks); __b++ ) {
      int bid = blkmap[__b];
      int nelem = nelems[bid];
      int efirst = offset[bid];
      for (int i = efirst; i < (efirst + nelem); i++ ) {
        %(vec_inits)s;
        %(itspace_loops)s
        %(zero_tmps)s;
        %(kernel_name)s(%(kernel_args)s);
        %(addtos_vector_field)s;
        %(itspace_loop_close)s
        %(addtos_scalar_field)s;
      }
    }
  }
  %(reduction_finalisations)s
  boffset += nblocks;
  }
"""

    def generate_code(self):

        # Most of the code to generate is the same as that for sequential
        code_dict = super(JITModule, self).generate_code()

        _reduction_decs = ';\n'.join([arg.c_reduction_dec() for arg in self._args if arg._is_global_reduction])
        _reduction_inits = ';\n'.join([arg.c_reduction_init() for arg in self._args if arg._is_global_reduction])
        _reduction_finalisations = '\n'.join([arg.c_reduction_finalisation() for arg in self._args if arg._is_global_reduction])

        code_dict.update({'reduction_decs' : _reduction_decs,
                          'reduction_inits' : _reduction_inits,
                          'reduction_finalisations' : _reduction_finalisations})
        return code_dict

class ParLoop(device.ParLoop, host.ParLoop):

    def compute(self):
        fun = JITModule(self.kernel, self.it_space.extents, *self.args)
        _args = [None, None]
        for arg in self.args:
            if arg._is_mat:
                _args.append(arg.data.handle.handle)
            else:
                _args.append(arg.data._data)

            if arg._is_dat:
                maybe_setflags(arg.data._data, write=False)

            if arg._is_indirect or arg._is_mat:
                maps = as_tuple(arg.map, Map)
                for map in maps:
                    _args.append(map.values)

        for c in Const._definitions():
            _args.append(c.data)

        #TODO: compute partition size
        plan = self._get_plan(1024)
        _args.append(plan.blkmap)
        _args.append(plan.offset)
        _args.append(plan.nelems)

        self.halo_exchange_begin()

        boffset = 0
        for c in range(plan.ncolors):
            if c == plan.ncolors_core:
                self.halo_exchange_end()

            nblocks = plan.ncolblk[c]
            if nblocks > 0:
                _args[0] = boffset
                _args[1] = nblocks
                fun(*_args)
                boffset += nblocks

            if c == plan.ncolors_owned - 1:
                self.reduction_begin()

        self.reduction_end()
        self.maybe_set_halo_update_needed()
        for arg in self.args:
            if arg._is_mat:
                arg.data._assemble()


    def _get_plan(self, part_size):
        if [arg for arg in self.args if arg._is_indirect or arg._is_mat]:
            plan = device.Plan(self._kernel, self._it_space.iterset,
                               *self._unwound_args,
                               partition_size=part_size,
                               matrix_coloring=True,
                               staging=False,
                               thread_coloring=False)

        else:
            # Create a fake plan for direct loops.
            # TODO:
            # Make the fake plan according to the number of cores available
            # to OpenMP instead of the partition size
            class FakePlan:
                def __init__(self, iset, part_size):
                    nblocks = int(math.ceil(iset.size / float(part_size)))
                    self.ncolors = 1
                    self.ncolors_core = 0
                    self.ncolors_owned = 1
                    self.ncolblk = np.array([nblocks], dtype=np.int32)
                    self.blkmap = np.arange(nblocks, dtype=np.int32)
                    self.nelems = np.array([min(part_size, iset.size - i * part_size) for i in range(nblocks)],
                                           dtype=np.int32)
                    self.offset = np.arange(0, iset.size, part_size, dtype=np.int32)

            plan = FakePlan(self._it_space.iterset, part_size)
        return plan



    @property
    def _requires_matrix_coloring(self):
        """Direct code generation to follow colored execution for global matrix insertion."""
        return True

def _setup():
    pass
