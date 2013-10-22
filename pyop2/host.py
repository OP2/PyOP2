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
from utils import as_tuple, flatten
import configuration as cfg


class Arg(base.Arg):

    def c_arg_name(self):
        name = self.name
        if self._is_indirect and not (self._is_vec_map or self._uses_itspace):
            name = "%s_%d" % (name, self.idx)
        return name

    def c_vec_name(self):
        return self.c_arg_name() + "_vec"

    def c_map_name(self, idx=0):
        return self.c_arg_name() + "_map%d" % idx

    def c_wrapper_arg(self):
        val = "PyObject *_%(name)s" % {'name': self.c_arg_name()}
        if self._is_indirect or self._is_mat:
            for idx, _ in enumerate(as_tuple(self.map, Map)):
                val += ", PyObject *_%(name)s" % {'name': self.c_map_name(idx)}
        return val

    def c_vec_dec(self):
        cdim = self.data.dataset.cdim if self._flatten else 1
        return ";\n%(type)s *%(vec_name)s[%(arity)s]" % \
            {'type': self.ctype,
             'vec_name': self.c_vec_name(),
             'arity': self.map.arity * cdim}

    def c_wrapper_dec(self):
        if self._is_mat:
            val = "Mat %(name)s = (Mat)((uintptr_t)PyLong_AsUnsignedLong(_%(name)s))" % \
                {"name": self.c_arg_name()}
        else:
            val = "%(type)s *%(name)s = (%(type)s *)(((PyArrayObject *)_%(name)s)->data)" % \
                {'name': self.c_arg_name(), 'type': self.ctype}
        if self._is_indirect or self._is_mat:
            for idx, _ in enumerate(as_tuple(self.map, Map)):
                val += ";\nint *%(name)s = (int *)(((PyArrayObject *)_%(name)s)->data)" % \
                       {'name': self.c_map_name(idx)}
        if self._is_vec_map:
            val += self.c_vec_dec()
        return val

    def c_ind_data(self, idx, j=0):
        return "%(name)s + %(map_name)s[i * %(arity)s + %(idx)s] * %(dim)s%(off)s" % \
            {'name': self.c_arg_name(),
             'map_name': self.c_map_name(),
             'arity': self.map.arity,
             'idx': idx,
             'dim': self.data.cdim,
             'off': ' + %d' % j if j else ''}

    def c_ind_data_xtr(self, idx):
        return "%(name)s + xtr_%(map_name)s[%(idx)s] * %(dim)s" % \
            {'name': self.c_arg_name(),
             'map_name': self.c_map_name(),
             'idx': idx,
             'dim': self.data.cdim}

    def c_kernel_arg_name(self):
        return "p_%s" % self.c_arg_name()

    def c_global_reduction_name(self, count=None):
        return self.c_arg_name()

    def c_local_tensor_name(self):
        return self.c_kernel_arg_name()

    def c_kernel_arg(self, count):
        if self._uses_itspace:
            if self._is_mat:
                if self.data._is_vector_field:
                    return self.c_kernel_arg_name()
                elif self.data._is_scalar_field:
                    idx = ''.join(["[i_%d]" % i for i, _ in enumerate(self.data.dims)])
                    return "(%(t)s (*)[1])&%(name)s%(idx)s" % \
                        {'t': self.ctype,
                         'name': self.c_kernel_arg_name(),
                         'idx': idx}
                else:
                    raise RuntimeError("Don't know how to pass kernel arg %s" % self)
            else:
                if self.data is not None and self.data.dataset.set.layers > 1:
                    return self.c_ind_data_xtr("i_%d" % self.idx.index)
                elif self._flatten:
                    return "%(name)s + %(map_name)s[i * %(arity)s + i_0 %% %(arity)d] * %(dim)s + (i_0 / %(arity)d)" % \
                        {'name': self.c_arg_name(),
                         'map_name': self.c_map_name(),
                         'arity': self.map.arity,
                         'dim': self.data.cdim}
                else:
                    return self.c_ind_data("i_%d" % self.idx.index)
        elif self._is_indirect:
            if self._is_vec_map:
                return self.c_vec_name()
            return self.c_ind_data(self.idx)
        elif self._is_global_reduction:
            return self.c_global_reduction_name(count)
        elif isinstance(self.data, Global):
            return self.c_arg_name()
        else:
            return "%(name)s + i * %(dim)s" % \
                {'name': self.c_arg_name(),
                 'dim': self.data.cdim}

    def c_vec_init(self):
        val = []
        if self._flatten:
            for j in range(self.data.dataset.cdim):
                for idx in range(self.map.arity):
                    val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                               {'vec_name': self.c_vec_name(),
                                'idx': j * self.map.arity + idx,
                                'data': self.c_ind_data(idx, j)})
        else:
            for idx in range(self.map.arity):
                val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                           {'vec_name': self.c_vec_name(),
                            'idx': idx,
                            'data': self.c_ind_data(idx)})
        return ";\n".join(val)

    def c_addto_scalar_field(self, extruded):
        maps = as_tuple(self.map, Map)
        nrows = maps[0].arity
        ncols = maps[1].arity
        rows_str = "%s + i * %s" % (self.c_map_name(0), nrows)
        cols_str = "%s + i * %s" % (self.c_map_name(1), ncols)

        if extruded is not None:
            rows_str = extruded + self.c_map_name(0)
            cols_str = extruded + self.c_map_name(1)

        return 'addto_vector(%(mat)s, %(vals)s, %(nrows)s, %(rows)s, %(ncols)s, %(cols)s, %(insert)d)' % \
            {'mat': self.c_arg_name(),
             'vals': self.c_kernel_arg_name(),
             'nrows': nrows,
             'ncols': ncols,
             'rows': rows_str,
             'cols': cols_str,
             'insert': self.access == WRITE}

    def c_addto_vector_field(self):
        maps = as_tuple(self.map, Map)
        nrows = maps[0].arity
        ncols = maps[1].arity
        dims = self.data.sparsity.dims
        rmult = dims[0]
        cmult = dims[1]
        s = []
        if self._flatten:
            idx = '[0][0]'
            val = "&%s%s" % (self.c_kernel_arg_name(), idx)
            row = "%(m)s * %(map)s[i * %(dim)s + i_0 %% %(dim)s] + (i_0 / %(dim)s)" % \
                  {'m': rmult,
                   'map': self.c_map_name(idx=0),
                   'dim': nrows}
            col = "%(m)s * %(map)s[i * %(dim)s + i_1 %% %(dim)s] + (i_1 / %(dim)s)" % \
                  {'m': cmult,
                   'map': self.c_map_name(idx=1),
                   'dim': ncols}
            return 'addto_scalar(%s, %s, %s, %s, %d)' \
                % (self.c_arg_name(), val, row, col, self.access == WRITE)
        for i in xrange(rmult):
            for j in xrange(cmult):
                idx = '[%d][%d]' % (i, j)
                val = "&%s%s" % (self.c_kernel_arg_name(), idx)
                row = "%(m)s * %(map)s[i * %(dim)s + i_0] + %(i)s" % \
                      {'m': rmult,
                       'map': self.c_map_name(idx=0),
                       'dim': nrows,
                       'i': i}
                col = "%(m)s * %(map)s[i * %(dim)s + i_1] + %(j)s" % \
                      {'m': cmult,
                       'map': self.c_map_name(idx=1),
                       'dim': ncols,
                       'j': j}

                s.append('addto_scalar(%s, %s, %s, %s, %d)'
                         % (self.c_arg_name(), val, row, col, self.access == WRITE))
        return ';\n'.join(s)

    def c_local_tensor_dec(self, extents):
        t = self.data.ctype
        if self.data._is_scalar_field:
            dims = ''.join(["[%d]" % d for d in extents])
        elif self.data._is_vector_field:
            dims = ''.join(["[%d]" % d for d in self.data.dims])
            if self._flatten:
                dims = '[1][1]'
        else:
            raise RuntimeError("Don't know how to declare temp array for %s" % self)
        return "%s %s%s" % (t, self.c_local_tensor_name(), dims)

    def c_zero_tmp(self):
        t = self.ctype
        if self.data._is_scalar_field:
            idx = ''.join(["[i_%d]" % i for i, _ in enumerate(self.data.dims)])
            return "%(name)s%(idx)s = (%(t)s)0" % \
                {'name': self.c_kernel_arg_name(), 't': t, 'idx': idx}
        elif self.data._is_vector_field:
            if self._flatten:
                return "%(name)s[0][0] = (%(t)s)0" % \
                    {'name': self.c_kernel_arg_name(), 't': t}
            size = np.prod(self.data.dims)
            return "memset(%(name)s, 0, sizeof(%(t)s) * %(size)s)" % \
                {'name': self.c_kernel_arg_name(), 't': t, 'size': size}
        else:
            raise RuntimeError("Don't know how to zero temp array for %s" % self)

    def c_add_offset(self):
        return '\n'.join(["%(name)s[%(j)d] += _off%(num)s[%(j)d] * %(dim)s;" %
                          {'name': self.c_vec_name(),
                           'j': j,
                           'num': self.c_offset(),
                           'dim': self.data.cdim} for j in range(self.map.arity)])

    # New globals generation which avoids false sharing.
    def c_intermediate_globals_decl(self, count):
        return "%(type)s %(name)s_l%(count)s[1][%(dim)s]" % \
            {'type': self.ctype,
             'name': self.c_arg_name(),
             'count': str(count),
             'dim': self.data.cdim}

    def c_intermediate_globals_init(self, count):
        if self.access == INC:
            init = "(%(type)s)0" % {'type': self.ctype}
        else:
            init = "%(name)s[i]" % {'name': self.c_arg_name()}
        return "for ( int i = 0; i < %(dim)s; i++ ) %(name)s_l%(count)s[0][i] = %(init)s" % \
            {'dim': self.data.cdim,
             'name': self.c_arg_name(),
             'count': str(count),
             'init': init}

    def c_intermediate_globals_writeback(self, count):
        d = {'gbl': self.c_arg_name(),
             'local': "%(name)s_l%(count)s[0][i]" %
             {'name': self.c_arg_name(), 'count': str(count)}}
        if self.access == INC:
            combine = "%(gbl)s[i] += %(local)s" % d
        elif self.access == MIN:
            combine = "%(gbl)s[i] = %(gbl)s[i] < %(local)s ? %(gbl)s[i] : %(local)s" % d
        elif self.access == MAX:
            combine = "%(gbl)s[i] = %(gbl)s[i] > %(local)s ? %(gbl)s[i] : %(local)s" % d
        return """
#pragma omp critical
for ( int i = 0; i < %(dim)s; i++ ) %(combine)s;
""" % {'combine': combine, 'dim': self.data.cdim}

    def c_map_decl(self):
        maps = as_tuple(self.map, Map)
        nrows = maps[0].arity
        ncols = maps[1].arity
        return '\n'.join(["int xtr_%(name)s[%(dim_row)s];" %
                          {'name': self.c_map_name(idx),
                           'dim_row': nrows,
                           'dim_col': ncols} for idx in range(2)])

    def c_map_decl_itspace(self):
        map = self.map
        nrows = map.arity
        return "int xtr_%(name)s[%(dim_row)s];\n" % \
            {'name': self.c_map_name(),
             'dim_row': str(nrows)}

    def c_map_init(self):
        return '\n'.join(flatten([["xtr_%(name)s[%(ind)s] = *(%(name)s + i * %(dim)s + %(ind)s);"
                                   % {'name': self.c_map_name(i),
                                      'dim': map.arity,
                                      'ind': idx}
                                   for idx in range(map.arity)]
                                  for i, map in enumerate(as_tuple(self.map, Map))]))

    def c_offset(self, idx=0):
        return "%s%s" % (self.position, idx)

    def c_add_offset_map(self):
        return '\n'.join(flatten([["xtr_%(name)s[%(ind)s] += _off%(off)s[%(ind)s];"
                                   % {'name': self.c_map_name(i),
                                      'off': self.c_offset(i),
                                      'ind': idx}
                                   for idx in range(map.arity)]
                                  for i, map in enumerate(as_tuple(self.map, Map))]))

    def c_offset_init(self):
        return ''.join([", PyObject *off%s" % self.c_offset(i)
                        for i in range(len(as_tuple(self.map, Map)))])

    def c_offset_decl(self):
        return ';\n'.join(['int * _off%(cnt)s = (int *)(((PyArrayObject *)off%(cnt)s)->data)'
                           % {'cnt': self.c_offset(i)}
                           for i in range(len(as_tuple(self.map, Map)))])


class JITModule(base.JITModule):

    _cppargs = []
    _system_headers = []
    _libraries = []

    def __init__(self, kernel, itspace, *args):
        # No need to protect against re-initialization since these attributes
        # are not expensive to set and won't be used if we hit cache
        self._kernel = kernel
        self._itspace = itspace
        self._extents = itspace.extents
        self._layers = itspace.layers
        self._args = args

    def __call__(self, *args):
        self.compile()(*args)

    def compile(self):
        if hasattr(self, '_fun'):
            return self._fun
        from instant import inline_with_numpy
        strip = lambda code: '\n'.join([l for l in code.splitlines()
                                        if l.strip() and l.strip() != ';'])

        if any(arg._is_soa for arg in self._args):
            kernel_code = """
            #define OP2_STRIDE(a, idx) a[idx]
            inline %(code)s
            #undef OP2_STRIDE
            """ % {'code': self._kernel.code}
        else:
            kernel_code = """
            inline %(code)s
            """ % {'code': self._kernel.code}
        code_to_compile = strip(dedent(self._wrapper) % self.generate_code())

        _const_decs = '\n'.join([const._format_declaration()
                                for const in Const._definitions()]) + '\n'

        # We need to build with mpicc since that's required by PETSc
        cc = os.environ.get('CC')
        os.environ['CC'] = 'mpicc'
        self._fun = inline_with_numpy(
            code_to_compile, additional_declarations=kernel_code,
            additional_definitions=_const_decs + kernel_code,
            cppargs=self._cppargs + (['-O0', '-g'] if cfg.debug else []),
            include_dirs=[d + '/include' for d in get_petsc_dir()],
            source_directory=os.path.dirname(os.path.abspath(__file__)),
            wrap_headers=["mat_utils.h"],
            system_headers=self._system_headers,
            library_dirs=[d + '/lib' for d in get_petsc_dir()],
            libraries=['petsc'] + self._libraries,
            sources=["mat_utils.cxx"],
            modulename=self._kernel.name if cfg.debug else None)
        if cc:
            os.environ['CC'] = cc
        else:
            os.environ.pop('CC')
        return self._fun

    def generate_code(self):

        def itspace_loop(i, d):
            return "for (int i_%d=0; i_%d<%d; ++i_%d) {" % (i, i, d, i)

        def c_const_arg(c):
            return 'PyObject *_%s' % c.name

        def c_const_init(c):
            d = {'name': c.name,
                 'type': c.ctype}
            if c.cdim == 1:
                return '%(name)s = ((%(type)s *)(((PyArrayObject *)_%(name)s)->data))[0]' % d
            tmp = '%(name)s[%%(i)s] = ((%(type)s *)(((PyArrayObject *)_%(name)s)->data))[%%(i)s]' % d
            return ';\n'.join([tmp % {'i': i} for i in range(c.cdim)])

        def extrusion_loop(d):
            return "for (int j_0=0; j_0<%d; ++j_0){" % d

        _ssinds_arg = ""
        _ssinds_dec = ""
        _index_expr = "n"
        if isinstance(self._itspace._iterset, Subset):
            _ssinds_arg = "PyObject* _ssinds,"
            _ssinds_dec = "int* ssinds = (int*) (((PyArrayObject*) _ssinds)->data);"
            _index_expr = "ssinds[n]"

        _wrapper_args = ', '.join([arg.c_wrapper_arg() for arg in self._args])

        _local_tensor_decs = ';\n'.join(
            [arg.c_local_tensor_dec(self._extents) for arg in self._args if arg._is_mat])
        _wrapper_decs = ';\n'.join([arg.c_wrapper_dec() for arg in self._args])

        _kernel_user_args = [arg.c_kernel_arg(count)
                             for count, arg in enumerate(self._args)]
        _kernel_it_args = ["i_%d" % d for d in range(len(self._extents))]
        _kernel_args = ', '.join(_kernel_user_args + _kernel_it_args)
        _vec_inits = ';\n'.join([arg.c_vec_init() for arg in self._args
                                 if not arg._is_mat and arg._is_vec_map])

        nloops = len(self._extents)
        extents = list(self._extents)
        for arg in self._args:
            if arg._flatten:
                if arg._is_mat:
                    dims = arg.data.sparsity.dims
                    extents[0] *= dims[0]
                    extents[1] *= dims[1]
                    break
                if arg._is_dat and arg._uses_itspace:
                    extents[0] *= arg.data.cdim
                    break
        _itspace_loops = '\n'.join(['  ' * i + itspace_loop(i, e)
                                    for i, e in enumerate(extents)])
        _itspace_loop_close = '\n'.join('  ' * i + '}' for i in range(nloops - 1, -1, -1))

        _addtos_vector_field = ';\n'.join([arg.c_addto_vector_field() for arg in self._args
                                           if arg._is_mat and arg.data._is_vector_field])
        _addtos_scalar_field = ';\n'.join([arg.c_addto_scalar_field(None) for arg in self._args
                                           if arg._is_mat and arg.data._is_scalar_field])

        _zero_tmps = ';\n'.join([arg.c_zero_tmp() for arg in self._args if arg._is_mat])

        if len(Const._defs) > 0:
            _const_args = ', '
            _const_args += ', '.join([c_const_arg(c) for c in Const._definitions()])
        else:
            _const_args = ''
        _const_inits = ';\n'.join([c_const_init(c) for c in Const._definitions()])

        _intermediate_globals_decl = ';\n'.join(
            [arg.c_intermediate_globals_decl(count)
             for count, arg in enumerate(self._args)
             if arg._is_global_reduction])
        _intermediate_globals_init = ';\n'.join(
            [arg.c_intermediate_globals_init(count)
             for count, arg in enumerate(self._args)
             if arg._is_global_reduction])
        _intermediate_globals_writeback = ';\n'.join(
            [arg.c_intermediate_globals_writeback(count)
             for count, arg in enumerate(self._args)
             if arg._is_global_reduction])

        _apply_offset = ""
        if self._layers > 1:
            _off_args = ''.join([arg.c_offset_init() for arg in self._args
                                 if arg._uses_itspace or arg._is_vec_map])
            _off_inits = ';\n'.join([arg.c_offset_decl() for arg in self._args
                                     if arg._uses_itspace or arg._is_vec_map])
            _apply_offset += ';\n'.join([arg.c_add_offset_map() for arg in self._args
                                        if arg._uses_itspace])
            _apply_offset += ';\n'.join([arg.c_add_offset() for arg in self._args
                                         if arg._is_vec_map])
            _map_init = ';\n'.join([arg.c_map_init() for arg in self._args
                                    if arg._uses_itspace])
            _map_decl = ''

            _map_decl += ';\n'.join([arg.c_map_decl() for arg in self._args
                                     if arg._is_mat and arg.data._is_scalar_field])
            _map_decl += ';\n'.join([arg.c_map_decl_itspace() for arg in self._args
                                     if arg._uses_itspace and not arg._is_mat])

            _addtos_scalar_field_extruded = ';\n'.join([arg.c_addto_scalar_field("xtr_") for arg in self._args
                                                        if arg._is_mat and arg.data._is_scalar_field])
            _addtos_scalar_field = ""

            _extr_loop = '\n' + extrusion_loop(self._layers - 1)
            _extr_loop_close = '}\n'
        else:
            _off_args = ""
            _off_inits = ""
            _extr_loop = ""
            _extr_loop_close = ""
            _addtos_scalar_field_extruded = ""
            _map_decl = ""
            _map_init = ""

        indent = lambda t, i: ('\n' + '  ' * i).join(t.split('\n'))

        return {'ind': '  ' * nloops,
                'kernel_name': self._kernel.name,
                'ssinds_arg': _ssinds_arg,
                'ssinds_dec': _ssinds_dec,
                'index_expr': _index_expr,
                'wrapper_args': _wrapper_args,
                'wrapper_decs': indent(_wrapper_decs, 1),
                'const_args': _const_args,
                'const_inits': indent(_const_inits, 1),
                'local_tensor_decs': indent(_local_tensor_decs, 1),
                'itspace_loops': indent(_itspace_loops, 2),
                'itspace_loop_close': indent(_itspace_loop_close, 2),
                'vec_inits': indent(_vec_inits, 5),
                'zero_tmps': indent(_zero_tmps, 2 + nloops),
                'kernel_args': _kernel_args,
                'addtos_vector_field': indent(_addtos_vector_field, 2 + nloops),
                'addtos_scalar_field': indent(_addtos_scalar_field, 2),
                'apply_offset': indent(_apply_offset, 3),
                'off_args': _off_args,
                'off_inits': indent(_off_inits, 1),
                'extr_loop': indent(_extr_loop, 5),
                'extr_loop_close': indent(_extr_loop_close, 2),
                'interm_globals_decl': indent(_intermediate_globals_decl, 3),
                'interm_globals_init': indent(_intermediate_globals_init, 3),
                'interm_globals_writeback': indent(_intermediate_globals_writeback, 3),
                'addtos_scalar_field_extruded': indent(_addtos_scalar_field_extruded, 2 + nloops),
                'map_init': indent(_map_init, 5),
                'map_decl': indent(_map_decl, 1)}
