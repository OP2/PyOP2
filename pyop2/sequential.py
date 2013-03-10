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
from find_op2 import *
from utils import *
import op_lib_core as core
import runtime_base as rt
from runtime_base import *

# Parallel loop API

def par_loop(kernel, it_space, *args):
    """Invocation of an OP2 kernel with an access descriptor"""
    ParLoop(kernel, it_space, *args).compute()

class ParLoop(rt.ParLoop):
    def compute(self):
        _fun = self.generate_code()
        _args = [0, 0]          # start, stop
        for arg in self.args:
            print " ==== pass args ===="
            if arg._is_mat:
                print "pass 1"
                _args.append(arg.data.handle.handle)
            else:
                print "pass 1.1"
                if arg._multimap:
                    for dat in arg.data.dats:
                        print " -*- " 
                        _args.append(dat._data)
                else:
                    _args.append(arg.data._data)

            if arg._is_dat:
                print "pass 2"
                if arg._multimap:
                    for i in range(len(arg.data.dats)):
                        maybe_setflags(arg.data.dats[i]._data, write=False)
                else:
                    maybe_setflags(arg.data._data, write=False)

            if arg._is_indirect or arg._is_mat:
                print "pass 3"
                if arg._rowcol_map:
                    print "pass 3.1"
                    for i in range(len(arg.map)):
                        for map in arg.map[i]:
                            print " -*- "
                            _args.append(map.values)
                elif arg._multimap:
                    print "pass 3.2"
                    for map in arg.map.maps:
                        print " -*- "
                        _args.append(map.values)
                else:
                    print "pass 3.3"
                    maps = as_tuple(arg.map, Map)
                    for map in maps:
                        print " -*- "
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

    def generate_code(self):
        key = self._cache_key
        _fun = rt._parloop_cache.get(key)

        if _fun is not None:
            return _fun

        from instant import inline_with_numpy

        def c_arg_name(arg):
            name = arg.data.name
            if arg._is_indirect and not (arg._is_vec_map or arg._uses_itspace):
                name += str(arg.idx)
            return name

        def c_vec_name(arg):
            return c_arg_name(arg) + "_vec"
            
        def c_vec_name_multi(arg,i):
            return arg.data.dats[i].name + "_vec"

        def c_map_name(arg, i=None, j=None):
            if i != None and j != None:
                if i == 0:
                    #this is a row map
                    return c_arg_name(arg) + "_map_r_" + str(j)
                else:
                    return c_arg_name(arg) + "_map_c_" + str(j)
            else:
                if i != None:
                    return c_arg_name(arg) + "_map_" + str(i)
            return c_arg_name(arg) + "_map"

        def c_wrapper_arg(arg):
            # do the dats within the args
            val = ""
            if arg.data._name == "MultiDat":
                if not isinstance(arg.data.dats, list):
                    raise RuntimeError("The data of the MultiDat arg must be a list of OP2 Dats")
                for i in range(len(arg.data.dats)):
                    val += "PyObject *_%(name)s" % {'name' : c_arg_name(arg) + "_" + arg.data.dats[i].name }
                    if i != len(arg.data.dats) - 1:
                        val += ", "
            else:
                val = "PyObject *_%(name)s" % {'name' : c_arg_name(arg) }
            # now handle the maps within the arg
            if arg._is_indirect or arg._is_mat:
                print arg.map
                if arg._rowcol_map:
                    # if the arg is a mat arg and has a list of lists of maps
                    for i in range(len(arg.map)):
                        if not isinstance(arg.map[i], list):
                            raise RuntimeError("The arg requires a list of lists of maps as it's a mixed mat arg")
                        for j in range(len(arg.map[i])):
                            val += ", PyObject *_%(name)s" % {'name' : c_map_name(arg, i, j)}
                else:
                    if hasattr(arg.map, "_name") and arg.map._name == "MultiMap":
                        # if the arg is MultiDat which has a MultiMap
                        if not isinstance(arg.map.maps, list):
                            raise RuntimeError("The MultiMap must contain a list of maps")
                        for i in range(len(arg.map.maps)):
                            val += ", PyObject *_%(name)s" % {'name' : c_map_name(arg, i)}
                    else:
                        # old version of the code for regular arg
                        val += ", PyObject *_%(name)s" % {'name' : c_map_name(arg)}
                        maps = as_tuple(arg.map, Map)
                        if len(maps) is 2:
                            val += ", PyObject *_%(name)s" % {'name' : c_map_name(arg)+'2'}
            print "----> val is:"
            print val
            return val

        def c_wrapper_dec(arg):
            print "declare wrapper args"
            val = ""
            if arg._is_mat:
                print "branch 1"
                val = "Mat %(name)s = (Mat)((uintptr_t)PyLong_AsUnsignedLong(_%(name)s))" % \
                     { "name": c_arg_name(arg) }
            else:
                print "branch 2"
                if arg._multimap:
                    for i in range(len(arg.data.dats)):
                        if i > 0:
                            val += ";\n"
                        val += "%(type)s *%(name)s = (%(type)s *)(((PyArrayObject *)_%(name)s)->data)" % \
                            {'name' : c_arg_name(arg) + '_' + arg.data.dats[i].name, 'type' : arg.ctype}
                else:
                    val = "%(type)s *%(name)s = (%(type)s *)(((PyArrayObject *)_%(name)s)->data)" % \
                        {'name' : c_arg_name(arg), 'type' : arg.ctype}
            if arg._is_indirect or arg._is_mat:
                print "branch 3"
                if arg._rowcol_map:
                    for i in range(len(arg.map)):
                        for j in range(len(arg.map[i])):
                            val += ";\nint *%(name)s = (int *)(((PyArrayObject *)_%(name)s)->data)" % \
                                                {'name' : c_map_name(arg,i,j)}
                    return val
                else:
                    if arg._multimap:
                        for i in range(len(arg.data.dats)):
                            val += ";\nint *%(name)s = (int *)(((PyArrayObject *)_%(name)s)->data)" % \
                                                {'name' : c_map_name(arg,i)}
                    else:
                        val += ";\nint *%(name)s = (int *)(((PyArrayObject *)_%(name)s)->data)" % \
                            {'name' : c_map_name(arg)}
            if arg._is_mat:
                print "branch 4"
                val += ";\nint *%(name)s2 = (int *)(((PyArrayObject *)_%(name)s2)->data)" % \
                           {'name' : c_map_name(arg)}
            if arg._is_vec_map:
                print "branch 5"
                if arg._multimap:
                    print arg.map
                    for i in range(len(arg.data.dats)):
                        val += ";\n%(type)s *%(vec_name)s[%(dim)s]" % \
                            {'type' : arg.ctype,
                            'vec_name' : c_vec_name(arg) + "_" + str(i),
                            'dim' : arg.map.maps[i].dim}
                else:
                    val += ";\n%(type)s *%(vec_name)s[%(dim)s]" % \
                       {'type' : arg.ctype,
                        'vec_name' : c_vec_name(arg),
                        'dim' : arg.map.dim}
            return val

        def c_ind_data(arg, idx):
            return "%(name)s + %(map_name)s[i * %(map_dim)s + %(idx)s] * %(dim)s" % \
                    {'name' : c_arg_name(arg),
                     'map_name' : c_map_name(arg),
                     'map_dim' : arg.map.dim,
                     'idx' : idx,
                     'dim' : arg.data.cdim}
                  
        def c_ind_data_multi(arg, idx, j):
            print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
            print arg.data
            return "%(name)s + %(map_name)s[i * %(map_dim)s + %(idx)s] * %(dim)s" % \
                    {'name' : c_arg_name(arg) + '_' + arg.data.dats[j].name,
                     'map_name' : c_map_name(arg, j),
                     'map_dim' : arg.map.dim[0],
                     'idx' : idx,
                     'dim' : arg.data.dats[j].cdim}

        def c_kernel_arg(arg):
            if arg._uses_itspace:
                if arg._is_mat:
                    print "ker 1"
                    name = "p_%s" % c_arg_name(arg)
                    if arg.data._is_vector_field:
                        print "ker 1.1"
                        print name
                        return name
                    elif arg.data._is_scalar_field:
                        print "ker 1.2"
                        idx = ''.join(["[i_%d]" % i for i, _ in enumerate(arg.data.dims)])
                        return "(%(t)s (*)[1])&%(name)s%(idx)s" % \
                            {'t' : arg.ctype,
                             'name' : name,
                             'idx' : idx}
                    else:
                        raise RuntimeError("Don't know how to pass kernel arg %s" % arg)
                elif arg._row_map:
                    name = "p_%s" % c_arg_name(arg)
                    return name
                else:
                    print "ker 2"
                    return c_ind_data(arg, "i_%d" % arg.idx.index)
            elif arg._is_indirect:
                print "ker 3"
                if arg._multimap:
                    print "3.0"
                    ker_args = ""
                    for i in range(len(arg.data.dats)):
                        ker_args += " " + c_vec_name(arg) + "_" + str(i)
                        if i < len(arg.data.dats)-1:
                            ker_args += ","
                    return ker_args 
                elif arg._is_vec_map:
                    print "3.1"
                    print c_vec_name(arg)
                    return c_vec_name(arg)
                print "3.2"
                print c_ind_data(arg,arg.idx)
                return c_ind_data(arg, arg.idx)
            elif isinstance(arg.data, Global):
                print "ker 4"
                return c_arg_name(arg)
            else:
                print "ker 5"
                return "%(name)s + i * %(dim)s" % \
                    {'name' : c_arg_name(arg),
                     'dim' : arg.data.cdim}

        def c_vec_init(arg):
            val = []
            if arg._multimap:
                for j in range(len(arg.map.maps)):
                    for i in range(arg.map.maps[j]._dim):
                        print arg.map.maps[j], arg.map.maps[j]._dim
                        val.append("%(vec_name)s_%(index)s[%(idx)s] = %(data)s" %
                           {'vec_name' : c_vec_name(arg),
                            'idx' : i,
                            'data' : c_ind_data_multi(arg, i, j),
                            'index' : j})
            else:
              for i in range(arg.map._dim):
                val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                           {'vec_name' : c_vec_name(arg),
                            'idx' : i,
                            'data' : c_ind_data(arg, i)} )
            print "vec init"
            print val
            return ";\n".join(val)

        def c_addto_scalar_field(arg):
            name = c_arg_name(arg)
            p_data = 'p_%s' % name
            maps = as_tuple(arg.map, Map)
            nrows = maps[0].dim
            ncols = maps[1].dim

            return 'addto_vector(%(mat)s, %(vals)s, %(nrows)s, %(rows)s, %(ncols)s, %(cols)s, %(insert)d)' % \
                {'mat' : name,
                 'vals' : p_data,
                 'nrows' : nrows,
                 'ncols' : ncols,
                 'rows' : "%s + i * %s" % (c_map_name(arg), nrows),
                 'cols' : "%s2 + i * %s" % (c_map_name(arg), ncols),
                 'insert' : arg.access == rt.WRITE }
                 
        def c_addto_mixed_mat(arg):
            return "//THIS IS THE ADD TO MIXED MAT CODE\n"
            
        def c_addto_mixed_vec(args):
            for arg in args:
                if arg._row_map:
                    
                    return "//THIS IS THE ADD TO MIXED VEC CODE\n"
            return ""

        def c_addto_vector_field(arg):
            name = c_arg_name(arg)
            print name
            p_data = 'p_%s' % name
            print p_data
            if arg._rowcol_map:
                return c_addto_mixed_mat(arg)
            maps = as_tuple(arg.map, Map)
            nrows = maps[0].dim
            ncols = maps[1].dim
            dims = arg.data.sparsity.dims
            rmult = dims[0]
            cmult = dims[1]
            s = []
            for i in xrange(rmult):
                for j in xrange(cmult):
                    idx = '[%d][%d]' % (i, j)
                    val = "&%s%s" % (p_data, idx)
                    row = "%(m)s * %(map)s[i * %(dim)s + i_0] + %(i)s" % \
                          {'m' : rmult,
                           'map' : c_map_name(arg),
                           'dim' : nrows,
                           'i' : i }
                    col = "%(m)s * %(map)s2[i * %(dim)s + i_1] + %(j)s" % \
                          {'m' : cmult,
                           'map' : c_map_name(arg),
                           'dim' : ncols,
                           'j' : j }

                    s.append('addto_scalar(%s, %s, %s, %s, %d)' \
                            % (name, val, row, col, arg.access == rt.WRITE))
            return ';\n'.join(s)

        def itspace_loop(i, d):
            return "for (int i_%d=0; i_%d<%d; ++i_%d){" % (i, i, d, i)

        def tmp_decl(arg, extents):
            t = arg.data.ctype
            print t
            print arg.data._is_scalar_field
            print arg.data._is_vector_field
            print arg.data._is_mixed_field
            if arg.data._is_mixed_field:
                #dims = ''.join(["[%d]" % d for d in extents])
                return "" #"%s p_%s%s" % (t, c_arg_name(arg), dims)
            if arg.data._is_scalar_field:
                dims = ''.join(["[%d]" % d for d in extents])
            elif arg.data._is_vector_field:
                dims = ''.join(["[%d]" % d for d in arg.data.dims])
            else:
                raise RuntimeError("Don't know how to declare temp array for %s" % arg)
            return "%s p_%s%s" % (t, c_arg_name(arg), dims)

        def c_zero_tmp(arg):
            name = "p_" + c_arg_name(arg)
            t = arg.ctype
            if arg.data._is_scalar_field:
                idx = ''.join(["[i_%d]" % i for i,_ in enumerate(arg.data.dims)])
                return "%(name)s%(idx)s = (%(t)s)0" % \
                    {'name' : name, 't' : t, 'idx' : idx}
            elif arg.data._is_vector_field:
                size = np.prod(arg.data.dims)
                return "memset(%(name)s, 0, sizeof(%(t)s) * %(size)s)" % \
                    {'name' : name, 't' : t, 'size' : size}
            else:
                raise RuntimeError("Don't know how to zero temp array for %s" % arg)

        def c_const_arg(c):
            return 'PyObject *_%s' % c.name

        def c_const_init(c):
            d = {'name' : c.name,
                 'type' : c.ctype}
            if c.cdim == 1:
                return '%(name)s = ((%(type)s *)(((PyArrayObject *)_%(name)s)->data))[0]' % d
            tmp = '%(name)s[%%(i)s] = ((%(type)s *)(((PyArrayObject *)_%(name)s)->data))[%%(i)s]' % d
            return ';\n'.join([tmp % {'i' : i} for i in range(c.cdim)])
            
        def c_mixed_block_loops(args):
            for arg in args:
                if arg._rowcol_map:
                    print "This is is where we do the block loops"
                    print arg
                    val = "for(int b_1 = 0; b_1 < %(row_blocks)s; b_1++){ \n \
                                for(int b_2 = 0; b_2 < %(col_blocks)s; b_2++){ " % \
                                    {'row_blocks' : len(arg._map[0]),
                                        'col_blocks' : len(arg._map[1])}
                                    
                    return val
                if arg._row_map:
                    print "This is is where we do the block loops"
                    print arg
                    val = "for(int b_1 = 0; b_1 < %(row_blocks)s; b_1++){ " % \
                                    {'row_blocks' : len(arg._map.maps)}
                                    
                    return val
            return ""       
        def c_mixed_block_loops_close(args):
            for arg in args:
                if arg._rowcol_map:
                    val = "}} //end of the block loops"
                    return val
                if arg._row_map:
                    val = "} //end of the block loop"
                    return val
            return ""
            
        def c_itspace_loops(args):
            
            for arg in args:
                print arg._row_map
                if arg._rowcol_map:
                    _itspace_loops = "for(int i_0 = 0; i_0 < row_blk_size[b_1]; i_0++){\n \
                            for(int i_1 = 0; i_1 < col_blk_size[b_2]; i_1++) {\n"
                    return _itspace_loops
                if arg._row_map:
                    _itspace_loops = "for(int i_0 = 0; i_0 < row_blk_size[b_1]; i_0++){\n"
                    return _itspace_loops
            return '\n'.join([itspace_loop(i,e) for i, e in zip(range(len(self._it_space.extents)), self._it_space.extents)])

        def c_tmp_blocksizes(args):
            for arg in args:
                if arg._rowcol_map:
                    rows = "int row_blk_size[%d] = {" % len(arg._map[0])
                    for i in range(len(arg._map[0])):
                        rows += " %d" % (arg.data.sparsity.dims[i][0] * arg._map[0][i].dim)
                        if i < len(arg._map[0])-1:
                            rows += ","
                    rows += " };\n"
                    cols = "int col_blk_size[%d] = {" % len(arg._map[1])
                    for i in range(len(arg._map[1])):
                        cols += " %d" % (arg.data.sparsity.dims[i][1] * arg._map[1][i].dim)
                        if i < len(arg._map[1])-1:
                            cols += ","
                    cols += " };\n"
                    return rows+cols
                if arg._row_map:
                    rows = "int row_blk_size[%d] = {" % len(arg.data.dats)
                    for i in range(len(arg.data.dats)):
                        rows += " %d" % arg.data.dats[i].dim
                        if i < len(arg.data.dats)-1:
                            rows += ","
                    rows += " };\n"
                    return rows
            return ""
            
        def c_zero_tmps(args):
            for arg in args:
                if arg._rowcol_map:
                    name = "p_" + c_arg_name(arg)
                    t = arg.ctype
                    return "%(type)s %(name)s[1][1];\n" % { 'type': t, 'name':name }
                if arg._row_map:
                    name = "p_" + c_arg_name(arg)
                    t = arg.ctype
                    return "%(type)s %(name)s[1][1];\n" % { 'type': t, 'name':name }
            return ';\n'.join([c_zero_tmp(arg) for arg in args if arg._is_mat])

        args = self.args
        _wrapper_args = ', '.join([c_wrapper_arg(arg) for arg in args])

        print "====== start tmp decs"
        _tmp_decs = ';\n'.join([tmp_decl(arg, self._it_space.extents) for arg in args if arg._is_mat])
        print _tmp_decs
        print "====== start wrapper decs"
        _wrapper_decs = ';\n'.join([c_wrapper_dec(arg) for arg in args])

        print "====== start const decs"
        _const_decs = '\n'.join([const._format_declaration() for const in Const._definitions()]) + '\n'

        print "====== start kernel user args"
        _kernel_user_args = [c_kernel_arg(arg) for arg in args]
        print "====== start kernel it args"
        _kernel_it_args   = ["i_%d" % d for d in range(len(self._it_space.extents))]
        print "====== start kernel args"
        _kernel_args = ', '.join(_kernel_user_args + _kernel_it_args)

        print "====== start vec inits"
        _vec_inits = ';\n'.join([c_vec_init(arg) for arg in args \
                                 if not arg._is_mat and arg._is_vec_map])
        print "====== start itspace loops"
        _itspace_loops =  c_itspace_loops(args) #'\n'.join([itspace_loop(i,e) for i, e in zip(range(len(self._it_space.extents)), self._it_space.extents)])
        print _itspace_loops
        _itspace_loop_close = '}'*len(self._it_space.extents)
        print "====== start vec field"
        _addtos_vector_field = ';\n'.join([c_addto_vector_field(arg) for arg in args \
                                           if arg._is_mat and arg.data._is_vector_field])
        print "====== start scalar field"
        _addtos_scalar_field = ';\n'.join([c_addto_scalar_field(arg) for arg in args \
                                           if arg._is_mat and arg.data._is_scalar_field])
                                           
        _addto_mixed_vec = c_addto_mixed_vec(args)
                                           
        print "====== start mixed space loops"
        _mixed_block_loops = c_mixed_block_loops(args)
        _mixed_block_loops_close = c_mixed_block_loops_close(args)
        
        _tmp_blocksizes = c_tmp_blocksizes(args)

        _zero_tmps = c_zero_tmps(args) #';\n'.join([c_zero_tmp(arg) for arg in args if arg._is_mat])

        if len(Const._defs) > 0:
            _const_args = ', '
            _const_args += ', '.join([c_const_arg(c) for c in Const._definitions()])
        else:
            _const_args = ''
        _const_inits = ';\n'.join([c_const_init(c) for c in Const._definitions()])
        wrapper = """
            void wrap_%(kernel_name)s__(PyObject *_start, PyObject *_end, %(wrapper_args)s %(const_args)s) {
            int start = (int)PyInt_AsLong(_start);
            int end = (int)PyInt_AsLong(_end);
            %(wrapper_decs)s;
            %(tmp_decs)s;
            %(tmp_blocksizes)s
            %(const_inits)s;
            for ( int i = start; i < end; i++ ) {
            %(vec_inits)s;
            %(mixed_block_loops)s
            %(itspace_loops)s
            %(zero_tmps)s;
            %(kernel_name)s(%(kernel_args)s);
            %(addto_mixed_vec)s
            %(addtos_vector_field)s;
            %(itspace_loop_close)s
            %(mixed_block_loops_close)s
            %(addtos_scalar_field)s;
            }
            }"""

        print "====== start stride"
        
        if any(arg._is_soa for arg in args):
            kernel_code = """
            #define OP2_STRIDE(a, idx) a[idx]
            inline %(code)s
            #undef OP2_STRIDE
            """ % {'code' : self._kernel.code}
        else:
            kernel_code = """
            inline %(code)s
            """ % {'code' : self._kernel.code }
            
        print "====== start code to compile"
        code_to_compile =  wrapper % { 'kernel_name' : self._kernel.name,
                                       'wrapper_args' : _wrapper_args,
                                       'wrapper_decs' : _wrapper_decs,
                                       'const_args' : _const_args,
                                       'const_inits' : _const_inits,
                                       'tmp_decs' : _tmp_decs,
                                       'itspace_loops' : _itspace_loops,
                                       'itspace_loop_close' : _itspace_loop_close,
                                       'vec_inits' : _vec_inits,
                                       'zero_tmps' : _zero_tmps,
                                       'kernel_args' : _kernel_args,
                                       'addtos_vector_field' : _addtos_vector_field,
                                       'addtos_scalar_field' : _addtos_scalar_field,
                                       'mixed_block_loops' : _mixed_block_loops,
                                       'mixed_block_loops_close' : _mixed_block_loops_close,
                                       'tmp_blocksizes': _tmp_blocksizes,
                                       'addto_mixed_vec' : _addto_mixed_vec }

        print code_to_compile
        # We need to build with mpicc since that's required by PETSc
        cc = os.environ.get('CC')
        os.environ['CC'] = 'mpicc'
        _fun = inline_with_numpy(code_to_compile, additional_declarations = kernel_code,
                                 additional_definitions = _const_decs + kernel_code,
                                 include_dirs=[OP2_INC, get_petsc_dir()+'/include'],
                                 source_directory=os.path.dirname(os.path.abspath(__file__)),
                                 wrap_headers=["mat_utils.h"],
                                 library_dirs=[OP2_LIB, get_petsc_dir()+'/lib'],
                                 libraries=['op2_seq', 'petsc'],
                                 sources=["mat_utils.cxx"])
        if cc:
            os.environ['CC'] = cc
        else:
            os.environ.pop('CC')

        rt._parloop_cache[key] = _fun
        return _fun

def _setup():
    pass
