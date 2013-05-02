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

import os
import numpy as np

from exceptions import *
from utils import as_tuple
import op_lib_core as core
import petsc_base
from petsc_base import *
import host
from host import Arg

# Parallel loop API

def par_loop(kernel, it_space, *args):
    """Invocation of an OP2 kernel with an access descriptor"""
    ParLoop(kernel, it_space, *args).compute()

class ParLoop(host.ParLoop):

    wrapper = """
              void wrap_%(kernel_name)s__(PyObject *_start, PyObject *_end, %(wrapper_args)s %(const_args)s) {
                int start = (int)PyInt_AsLong(_start);
                int end = (int)PyInt_AsLong(_end);
                %(wrapper_decs)s;
                %(local_tensor_decs)s;
                %(local_tensor_blocksizes)s
                %(const_inits)s;
                for ( int i = start; i < end; i++ ) {
                  %(vec_inits)s;
                  %(mixed_block_loops)s
                  %(itspace_loops)s
                  %(ind)s%(zero_tmps)s;
                  %(ind)s%(kernel_name)s(%(kernel_args)s);
                  %(ind)s%(addto_mixed_vec)s
                  %(ind)s%(addto_mixed_mat)s
                  %(ind)s%(addtos_vector_field)s;
                  %(itspace_loop_close)s
                  %(mixed_block_loops_close)s
                  %(addtos_scalar_field)s;
                }
              }
              """

    def compute(self):
        _fun = self.build()
        _args = [0, 0]          # start, stop
        for arg in self.args:
            if arg._is_mat:
                if arg._rowcol_map:
                    for i in range(len(arg.data.mat_list)):
                        _args.append(arg.data.mat_list[i].handle.handle)
                else:
                    _args.append(arg.data.handle.handle)
            else:
                if arg._multimap:
                    for dat in arg.data.dats:
                        _args.append(dat._data)
                else:
                    _args.append(arg.data._data)

            if arg._is_dat:
                if arg._multimap:
                    for i in range(len(arg.data.dats)):
                        maybe_setflags(arg.data.dats[i]._data, write=False)
                else:
                    maybe_setflags(arg.data._data, write=False)

            if arg._is_indirect or arg._is_mat:
                if arg._rowcol_map:
                    for i in range(len(arg.map)):
                        for map in arg.map[i]:
                            _args.append(map.values)
                elif arg._multimap:
                    for map in arg.map.maps:
                        _args.append(map.values)
                else:
                    maps = as_tuple(arg.map, Map)
                    for map in maps:
                        _args.append(map.values)

        for c in Const._definitions():
            _args.append(c.data)

        # kick off halo exchanges
        self.halo_exchange_begin()
        # compute over core set elements
        _args[0] = 0
        _args[1] = self.it_space.core_size
        _fun(*_args)
        # wait for halo exchanges to complete
        self.halo_exchange_end()
        # compute over remaining owned set elements
        _args[0] = self.it_space.core_size
        _args[1] = self.it_space.size
        _fun(*_args)
        # By splitting the reduction here we get two advantages:
        # - we don't double count contributions in halo elements
        # - once our MPI supports the asynchronous collectives in
        #   MPI-3, we can do more comp/comms overlap
        self.reduction_begin()
        if self.needs_exec_halo:
            _args[0] = self.it_space.size
            _args[1] = self.it_space.exec_size
            _fun(*_args)
        self.reduction_end()
        self.maybe_set_halo_update_needed()
        for arg in self.args:
            if arg._is_mat:
                arg.data._assemble()

def _setup():
    pass
