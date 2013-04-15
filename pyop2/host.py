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

"""Base classes extending those from the :mod:`base` module with functionality
common to backends executing on the host."""

from textwrap import dedent

import base
from base import *
from base import _parloop_cache
from utils import as_tuple
import configuration as cfg
from find_op2 import *

class Arg(base.Arg):

    def c_arg_name(self):
        name = self.data.name
        if self._is_indirect and not (self._is_vec_map or self._uses_itspace):
            name += str(self.idx)
        return name

    def c_vec_name(self):
        return self.c_arg_name() + "_vec"

    def c_map_name(self, i=None, j=None):
        suffix = ""
        if j is not None:
            suffix += "_%s" % j
        if i is not None:
            suffix += "_%s" % i
        return self.c_arg_name() + "_map" + suffix

    def c_wrapper_arg(self):
        # do the dats within the args
        if isinstance(self.data, MultiDat):
            if not isinstance(self.data.dats, list):
                raise RuntimeError("The data of the MultiDat arg must be a list of OP2 Dats")
            val = ", ".join(["PyObject *_%s_%s" % (self.c_arg_name(), dat.name) for dat in self.data.dats])
        else:
            if self._rowcol_map:
                val = ', '.join(["PyObject *_%s" % mat.name for mat in self.data.mat_list])
            else:
                val = "PyObject *_%(name)s" % {'name' : self.c_arg_name() }
        # now handle the maps within the arg
        if self._is_indirect or self._is_mat:
            if self._rowcol_map:
                # if the arg is a mat arg and has a list of lists of maps
                for i in range(len(self.map)):
                    if not isinstance(self.map[i], list):
                        raise RuntimeError("The arg requires a list of lists of maps as it's a mixed mat arg")
                    for j in range(len(self.map[i])):
                        val += ", PyObject *_%s" % self.c_map_name(i, j)
            else:
                if isinstance(self.map, MultiMap):
                    # if the arg is MultiDat which has a MultiMap
                    if not isinstance(self.map.maps, list):
                        raise RuntimeError("The MultiMap must contain a list of maps")
                    for i in range(len(self.map.maps)):
                        val += ", PyObject *_%s" % self.c_map_name(i)
                else:
                    # old version of the code for regular arg
                    val += ", PyObject *_%s" % self.c_map_name()
                    maps = as_tuple(self.map, Map)
                    if len(maps) is 2:
                        val += ", PyObject *_%s2" % self.c_map_name()
        return val

    def c_vec_dec(self, dim):
        return ";\n%(type)s *%(vec_name)s[%(dim)s]" % \
               {'type' : self.ctype,
                'vec_name' : self.c_vec_name(),
                'dim' : dim}

    def c_wrapper_dec(self):
        if self._is_mat:
            if self._rowcol_map:
                val = ';\n'.join(["Mat %(name)s = (Mat)((uintptr_t)PyLong_AsUnsignedLong(_%(name)s));\n" % \
                        {'name': mat.name} for mat in self.data.mat_list])
            else:
                val = "Mat %(name)s = (Mat)((uintptr_t)PyLong_AsUnsignedLong(_%(name)s))" % \
                        { "name": self.c_arg_name() }
        else:
            if self._multimap:
                val = ';\n'.join(["%(type)s *%(name)s = (%(type)s *)(((PyArrayObject *)_%(name)s)->data)" % \
                    {'name' : self.c_arg_name() + '_' + dat.name, 'type' : self.ctype} for dat in self.data.dats])
            else:
                val = "%(type)s *%(name)s = (%(type)s *)(((PyArrayObject *)_%(name)s)->data)" % \
                    {'name' : self.c_arg_name(), 'type' : self.ctype}
        if self._is_indirect or self._is_mat:
            if self._rowcol_map:
                for i in range(len(self.map)):
                    for j in range(len(self.map[i])):
                        val += ";\nint *%(name)s = (int *)(((PyArrayObject *)_%(name)s)->data)" % \
                                            {'name' : self.c_map_name(i, j)}
                return val
            else:
                if self._multimap:
                    for i in range(len(self.data.dats)):
                        val += ";\nint *%(name)s = (int *)(((PyArrayObject *)_%(name)s)->data)" % \
                                            {'name' : self.c_map_name(i)}
                else:
                    val += ";\nint *%(name)s = (int *)(((PyArrayObject *)_%(name)s)->data)" % \
                        {'name' : self.c_map_name()}
        if self._is_mat:
            val += ";\nint *%(name)s2 = (int *)(((PyArrayObject *)_%(name)s2)->data)" % \
                       {'name' : self.c_map_name()}
        if self._is_vec_map:
            if self._multimap:
                total_dim = sum([self.map.maps[i].dim * self.data.dats[i].cdim
                                 for i in range(len(self.data.dats))])
                val += self.c_vec_dec(total_dim)
            else:
                val += self.c_vec_dec(self.map.dim * self.data.cdim)
        return val

    def c_ind_data(self, idx):
        return "%(name)s + %(map_name)s[i * %(map_dim)s + %(idx)s] * %(dim)s" % \
                {'name' : self.c_arg_name(),
                 'map_name' : self.c_map_name(),
                 'map_dim' : self.map.dim,
                 'idx' : idx,
                 'dim' : self.data.cdim}

    def c_ind_data_new(self, idx, j):
        return "%(name)s + %(map_name)s[i * %(map_dim)s + %(idx)s] * %(dim)s + %(j)s" % \
                {'name' : self.c_arg_name(),
                 'map_name' : self.c_map_name(),
                 'map_dim' : self.map.dim,
                 'idx' : idx,
                 'dim' : self.data.cdim,
                 'j' : j}

    def c_ind_data_multi(self, idx, j, l):
        return "%(name)s + %(map_name)s[i * %(map_dim)s + %(idx)s] * %(dim)s + %(l)s" % \
                {'name' : self.c_arg_name() + '_' + self.data.dats[j].name,
                 'map_name' : self.c_map_name(j),
                 'map_dim' : self.map.dim[j],
                 'idx' : idx,
                 'dim' : self.data.dats[j].cdim,
                 'l' : l}

    def c_kernel_arg_name(self):
        return "p_%s" % self.c_arg_name()

    def c_global_reduction_name(self):
        return self.c_arg_name()

    def c_local_tensor_name(self):
        return self.c_kernel_arg_name()

    def c_kernel_arg(self):
        if self._uses_itspace:
            if self._is_mat:
                if self.data._is_vector_field:
                    return self.c_kernel_arg_name()
                elif self.data._is_scalar_field:
                    idx = ''.join(["[i_%d]" % i for i, _ in enumerate(self.data.dims)])
                    return "(%(t)s (*)[1])&%(name)s%(idx)s" % \
                        {'t' : self.ctype,
                         'name' : self.c_kernel_arg_name(),
                         'idx' : idx}
                else:
                    raise RuntimeError("Don't know how to pass kernel arg %s" % self)
            elif self._row_map:
                name = "p_%s" % self.c_arg_name()
                return name
            else:
                return self.c_ind_data("i_%d" % self.idx.index)
        elif self._is_indirect:
            if self._is_vec_map:
                return self.c_vec_name()
            return self.c_ind_data(self.idx)
        elif self._is_global_reduction:
            return self.c_global_reduction_name()
        elif isinstance(self.data, Global):
            return self.c_arg_name()
        else:
            return "%(name)s + i * %(dim)s" % \
                {'name' : self.c_arg_name(),
                 'dim' : self.data.cdim}

    def c_vec_init(self):
        val = []
        k = 0
        if self._multimap:
            for j in range(len(self.map.maps)):
                for l in range(self.data.dats[j].cdim):
                    for i in range(self.map.maps[j]._dim):
                        val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                           {'vec_name' : self.c_vec_name(),
                            'idx' : k,
                            'data' : self.c_ind_data_multi(i, j, l)})
                        k += 1
        else:
            for j in range(self.data.cdim):
                for i in range(self.map._dim):
                    val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                            {'vec_name' : self.c_vec_name(),
                                'idx' : k,
                                'data' : self.c_ind_data_new(i, j)} )
                    k += 1
        return ";\n".join(val)

    def c_addto_scalar_field(self):
        maps = as_tuple(self.map, Map)
        nrows = maps[0].dim
        ncols = maps[1].dim

        return 'addto_vector(%(mat)s, %(vals)s, %(nrows)s, %(rows)s, %(ncols)s, %(cols)s, %(insert)d)' % \
            {'mat' : self.c_arg_name(),
             'vals' : self.c_kernel_arg_name(),
             'nrows' : nrows,
             'ncols' : ncols,
             'rows' : "%s + i * %s" % (self.c_map_name(), nrows),
             'cols' : "%s2 + i * %s" % (self.c_map_name(), ncols),
             'insert' : self.access == WRITE }

    def c_addto_vector_field(self):
        maps = as_tuple(self.map, Map)
        nrows = maps[0].dim
        ncols = maps[1].dim
        dims = self.data.sparsity.dims
        rmult = dims[0]
        cmult = dims[1]
        s = []
        for i in xrange(rmult):
            for j in xrange(cmult):
                idx = '[%d][%d]' % (i, j)
                val = "&%s%s" % (self.c_kernel_arg_name(), idx)
                row = "%(m)s * %(map)s[i * %(dim)s + i_0] + %(i)s" % \
                      {'m' : rmult,
                       'map' : self.c_map_name(),
                       'dim' : nrows,
                       'i' : i }
                col = "%(m)s * %(map)s2[i * %(dim)s + i_1] + %(j)s" % \
                      {'m' : cmult,
                       'map' : self.c_map_name(),
                       'dim' : ncols,
                       'j' : j }

                s.append('addto_scalar(%s, %s, %s, %s, %d)' \
                        % (self.c_arg_name(), val, row, col, self.access == WRITE))
        return ';\n'.join(s)

    def c_local_tensor_dec(self, extents):
        t = self.data.ctype
        if self.data._is_scalar_field:
            dims = ''.join(["[%d]" % d for d in extents])
        elif self.data._is_vector_field:
            dims = ''.join(["[%d]" % d for d in self.data.dims])
        else:
            raise RuntimeError("Don't know how to declare temp array for %s" % self)
        return "%s %s%s" % (t, self.c_local_tensor_name(), dims)

    def c_zero_tmp(self):
        t = self.ctype
        if self.data._is_scalar_field:
            idx = ''.join(["[i_%d]" % i for i,_ in enumerate(self.data.dims)])
            return "%(name)s%(idx)s = (%(t)s)0" % \
                {'name' : self.c_kernel_arg_name(), 't' : t, 'idx' : idx}
        elif self.data._is_vector_field:
            size = np.prod(self.data.dims)
            return "memset(%(name)s, 0, sizeof(%(t)s) * %(size)s)" % \
                {'name' : self.c_kernel_arg_name(), 't' : t, 'size' : size}
        else:
            raise RuntimeError("Don't know how to zero temp array for %s" % self)

class ParLoop(base.ParLoop):

    _cppargs = []
    _system_headers = []

    def build(self):

        key = self._cache_key
        _fun = _parloop_cache.get(key)

        if _fun is not None:
            return _fun

        from instant import inline_with_numpy

        if any(arg._is_soa for arg in self.args):
            kernel_code = """
            #define OP2_STRIDE(a, idx) a[idx]
            inline %(code)s
            #undef OP2_STRIDE
            """ % {'code' : self._kernel.code}
        else:
            kernel_code = """
            inline %(code)s
            """ % {'code' : self._kernel.code }
        code_to_compile = dedent(self.wrapper) % self.generate_code()

        _const_decs = '\n'.join([const._format_declaration() for const in Const._definitions()]) + '\n'

        # We need to build with mpicc since that's required by PETSc
        cc = os.environ.get('CC')
        os.environ['CC'] = 'mpicc'
        _fun = inline_with_numpy(code_to_compile, additional_declarations = kernel_code,
                                 additional_definitions = _const_decs + kernel_code,
                                 cppargs=self._cppargs + ['-O0', '-g'] if cfg.debug else [],
                                 include_dirs=[OP2_INC, get_petsc_dir()+'/include'],
                                 source_directory=os.path.dirname(os.path.abspath(__file__)),
                                 wrap_headers=["mat_utils.h"],
                                 system_headers=self._system_headers,
                                 library_dirs=[OP2_LIB, get_petsc_dir()+'/lib'],
                                 libraries=['op2_seq', 'petsc'],
                                 sources=["mat_utils.cxx"])
        if cc:
            os.environ['CC'] = cc
        else:
            os.environ.pop('CC')

        _parloop_cache[key] = _fun
        return _fun

    def generate_code(self):

        def itspace_loop(i, d):
            return "for (int i_%d=0; i_%d<%d; ++i_%d) {" % (i, i, d, i)

        def c_const_arg(c):
            return 'PyObject *_%s' % c.name

        def c_const_init(c):
            d = {'name' : c.name,
                 'type' : c.ctype}
            if c.cdim == 1:
                return '%(name)s = ((%(type)s *)(((PyArrayObject *)_%(name)s)->data))[0]' % d
            tmp = '%(name)s[%%(i)s] = ((%(type)s *)(((PyArrayObject *)_%(name)s)->data))[%%(i)s]' % d
            return ';\n'.join([tmp % {'i' : i} for i in range(c.cdim)])

        def c_addto_mixed_mat(args):
            for arg in args:
                if arg._rowcol_map:
                    s = ""
                    name = arg.c_arg_name()
                    p_data = 'p_%s' % name
                    rsize = len(arg._map[0])
                    csize = len(arg._map[1])
                    for i in range(rsize):
                        for j in range(csize):
                            s += "if (b_1 == %d && b_2 == %d) {" % (i, j)
                            dims = arg.data.sparsity.sparsity_list[rsize * i + j].dims
                            nrows = arg._map[0][i].dim
                            ncols = arg._map[1][j].dim
                            rmult = dims[0][0]
                            cmult = dims[0][1]
                            idx = '[0][0]'
                            val = "&%s%s" % (p_data, idx)
                            row = "(j_0 / %(dim)s) + %(rmult)s * %(map)s[i * %(dim)s + (j_0 %% %(dim)s)]" % \
                                {'map' : arg.c_map_name(i, j),
                                'dim' : nrows,
                                'rmult' : rmult }
                            col = "(j_1 / %(dim)s) + %(cmult)s * %(map)s[i * %(dim)s + (j_1 %% %(dim)s)]" % \
                                {'map' : arg.c_map_name(i, j),
                                'dim' : ncols,
                                'cmult' : cmult }
                            pos = i * rsize + j
                            mat_name = name + "_" + str(pos)
                            s += "addto_scalar(%s, %s, %s, %s, %d); }\n"  % (mat_name, val, row, col, arg.access == WRITE)
                    return s
            return ""

        def c_addto_mixed_vec(args):
            for arg in args:
                if arg._row_map:
                    s = ""
                    name = arg.c_arg_name()
                    p_data = 'p_%s' % name
                    vsize = len(arg.data.dats)
                    for i in range(vsize):
                        s += "if (b_1 == %d) { " % i
                        dim = arg.data.dats[i].dim[0]
                        mapdim = arg.map.maps[i].dim
                        dname = arg.data.dats[i].name
                        pos = "(j_0 / %(mdim)s) + %(dim)s * %(n)s_map_%(i)s[%(mdim)s*i + (j_0 %% %(mdim)s)]" % {
                                'n':name,
                                'i':str(i),
                                'mdim':mapdim,
                                'dim': dim }
                        s += "%(n)s_%(dn)s[%(pos)s] += p_%(n)s[0];" % { 'n':name, 'dn':dname, 'pos': pos }
                        s+="}\n"
                    return s
            return ""

        def c_mixed_block_loops(args):
            for arg in args:
                if arg._rowcol_map:
                    val = "for(int b_1 = 0; b_1 < %(row_blocks)s; b_1++) { \n \
                             for(int b_2 = 0; b_2 < %(col_blocks)s; b_2++) { " % \
                             {'row_blocks' : len(arg._map[0]),
                              'col_blocks' : len(arg._map[1])}
                    return val
                if arg._row_map:
                    val = "for(int b_1 = 0; b_1 < %(row_blocks)s; b_1++) { " % \
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
                if arg._rowcol_map:
                    _itspace_loops = "for(int j_0 = 0; j_0 < row_blk_size[b_1]; j_0++){\n"
                    _itspace_loops += "for(int j_1 = 0; j_1 < col_blk_size[b_2]; j_1++) {\n"
                    _itspace_loops += "const int i_0 = j_0 + row_blk_start[b_1];\n"
                    _itspace_loops += "const int i_1 = j_1 + col_blk_start[b_2];\n"
                    return _itspace_loops
                if arg._row_map:
                    _itspace_loops = "for(int j_0 = 0; j_0 < row_blk_size[b_1]; j_0++){\n"
                    _itspace_loops += "const int i_0 = j_0 + row_blk_start[b_1];\n"
                    return _itspace_loops
            return '\n'.join(['  ' * i + itspace_loop(i,e) for i, e in enumerate(self._it_space.extents)])

        def c_local_tensor_blocksizes(args):
            for arg in args:
                if arg._rowcol_map:
                    rows = "int row_blk_size[%d] = {" % len(arg._map[0])
                    rowstart = "int row_blk_start[%d] = {" % len(arg._map[0])
                    cnt = 0
                    for i in range(len(arg._map[0])):
                        sz = arg.data.sparsity.dims[i][0] * arg._map[0][i].dim
                        rows += " %d" % sz
                        rowstart += " %d" % cnt
                        if i < len(arg._map[0])-1:
                            rows += ","
                            rowstart += ","
                        cnt += sz
                    rows += " };\n"
                    rowstart += " };\n"
                    cols = "int col_blk_size[%d] = {" % len(arg._map[1])
                    colstart = "int col_blk_start[%d] = {" % len(arg._map[1])
                    cnt = 0
                    for i in range(len(arg._map[1])):
                        sz = arg.data.sparsity.dims[i][1] * arg._map[1][i].dim
                        cols += " %d" % sz
                        colstart += " %d" % cnt
                        if i < len(arg._map[1])-1:
                            cols += ","
                            colstart += ","
                        cnt += sz
                    cols += " };\n"
                    colstart += " };\n"
                    return '\n'.join([rows, rowstart, cols, colstart])
                if arg._row_map:
                    rowstart = "int row_blk_start[%d] = {" % len(arg.data.dats)
                    cnt = 0
                    rows = "int row_blk_size[%d] = {" % len(arg.data.dats)
                    for i in range(len(arg.data.dats)):
                        ssize = arg.data.dats[i].dim[0] * arg.map.maps[i].dim
                        rows += " %d" % ssize
                        rowstart += " %d" % cnt
                        if i < len(arg.data.dats)-1:
                            rows += ","
                            rowstart += ","
                        cnt += ssize
                    rows += " };\n"
                    rowstart += " };\n"
                    return '\n'.join([rows, rowstart])
            return ""

        def c_zero_tmps(args):
            for arg in args:
                if arg._rowcol_map:
                    name = "p_" + arg.c_arg_name()
                    t = arg.ctype
                    return "%(type)s %(name)s[1][1] = {{0}};\n" % { 'type': t, 'name':name }
                if arg._row_map:
                    name = "p_" + arg.c_arg_name()
                    t = arg.ctype
                    return "%(type)s %(name)s[1] = {0};\n" % { 'type': t, 'name':name }
            return ';\n'.join([arg.c_zero_tmp() for arg in args if arg._is_mat])

        def c_addto_mixed_mat(self):
            return ''

        def c_addto_mixed_vec(self):
            return ''

        _wrapper_args = ', '.join([arg.c_wrapper_arg() for arg in self.args])

        _local_tensor_decs = ';\n'.join([arg.c_local_tensor_dec(self._it_space.extents) for arg in self.args if arg._is_mat])
        _wrapper_decs = ';\n'.join([arg.c_wrapper_dec() for arg in self.args])

        _kernel_user_args = [arg.c_kernel_arg() for arg in self.args]
        _kernel_it_args   = ["i_%d" % d for d in range(len(self._it_space.extents))]
        _kernel_args = ', '.join(_kernel_user_args + _kernel_it_args)
        _vec_inits = ';\n'.join([arg.c_vec_init() for arg in self.args \
                                 if not arg._is_mat and arg._is_vec_map])

        nloops = len(self._it_space.extents)
        _itspace_loops = c_itspace_loops(self.args)
        _itspace_loop_close = '\n'.join('  ' * i + '}' for i in range(nloops - 1, -1, -1))

        _addtos_vector_field = ';\n'.join([arg.c_addto_vector_field() for arg in self.args \
                                           if arg._is_mat and arg.data._is_vector_field and not arg._rowcol_map])
        _addtos_scalar_field = ';\n'.join([arg.c_addto_scalar_field() for arg in self.args \
                                           if arg._is_mat and arg.data._is_scalar_field])
        _addto_mixed_mat = c_addto_mixed_mat(self.args)

        _addto_mixed_vec = c_addto_mixed_vec(self.args)

        _mixed_block_loops = c_mixed_block_loops(self.args)
        _mixed_block_loops_close = c_mixed_block_loops_close(self.args)

        _local_tensor_blocksizes = c_local_tensor_blocksizes(self.args)

        _zero_tmps = c_zero_tmps(self.args)

        if len(Const._defs) > 0:
            _const_args = ', '
            _const_args += ', '.join([c_const_arg(c) for c in Const._definitions()])
        else:
            _const_args = ''
        _const_inits = ';\n'.join([c_const_init(c) for c in Const._definitions()])

        indent = lambda t, i: ('\n' + '  ' * i).join(t.split('\n'))
        return {'ind': '  ' * nloops,
                'kernel_name': self._kernel.name,
                'wrapper_args': _wrapper_args,
                'wrapper_decs': indent(_wrapper_decs, 1),
                'const_args': _const_args,
                'const_inits': indent(_const_inits, 1),
                'local_tensor_decs': indent(_local_tensor_decs, 1),
                'itspace_loops': indent(_itspace_loops, 2),
                'itspace_loop_close': indent(_itspace_loop_close, 2),
                'vec_inits': indent(_vec_inits, 2),
                'zero_tmps': indent(_zero_tmps, 2 + nloops),
                'kernel_args': _kernel_args,
                'addtos_vector_field': indent(_addtos_vector_field, 2 + nloops),
                'addtos_scalar_field': indent(_addtos_scalar_field, 2),
                'mixed_block_loops' : indent(_mixed_block_loops, 2),
                'mixed_block_loops_close' : indent(_mixed_block_loops_close, 2),
                'local_tensor_blocksizes': indent(_local_tensor_blocksizes, 1),
                'addto_mixed_vec' : indent(_addto_mixed_vec, 2 + nloops),
                'addto_mixed_mat' : indent(_addto_mixed_mat, 2 + nloops)}
