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
from copy import deepcopy as dcopy

import base
import compilation
from base import *
from mpi import collective
from configuration import configuration
from utils import as_tuple

import coffee.plan
from coffee import base as ast
from coffee.plan import ASTKernel
from coffee.utils import visit, ast_update_id


class Kernel(base.Kernel):

    def _ast_to_c(self, ast, opts={}):
        """Transform an Abstract Syntax Tree representing the kernel into a
        string of code (C syntax) suitable to CPU execution."""
        self._original_ast = dcopy(ast)
        ast_handler = ASTKernel(ast, self._include_dirs)
        ast_handler.plan_cpu(self._opts)
        self._applied_blas = ast_handler.blas
        return ast_handler.gencode()


class Arg(base.Arg):

    def c_arg_name(self, i=0, j=None):
        name = self.name
        if self._is_indirect and not (self._is_vec_map or self._uses_itspace):
            name = "%s_%d" % (name, self.idx)
        if i is not None:
            # For a mixed ParLoop we can't necessarily assume all arguments are
            # also mixed. If that's not the case we want index 0.
            if not self._is_mat and len(self.data) == 1:
                i = 0
            name += "_%d" % i
        if j is not None:
            name += "_%d" % j
        return name

    def c_vec_name(self):
        return self.c_arg_name() + "_vec"

    def c_map_name(self, i, j):
        return self.c_arg_name() + "_map%d_%d" % (i, j)

    def c_offset_name(self, i, j):
        return self.c_arg_name() + "_off%d_%d" % (i, j)

    def c_wrapper_arg(self):
        if self._is_mat:
            val = "Mat %s_" % self.c_arg_name()
        else:
            val = ', '.join(["%s *%s" % (self.ctype, self.c_arg_name(i))
                             for i in range(len(self.data))])
        if self._is_indirect or self._is_mat:
            for i, map in enumerate(as_tuple(self.map, Map)):
                for j, m in enumerate(map):
                    val += ", int *%s" % self.c_map_name(i, j)
        return val

    def c_vec_dec(self, is_facet=False):
        facet_mult = 2 if is_facet else 1
        cdim = self.data.dataset.cdim if self._flatten else 1
        return "%(type)s *%(vec_name)s[%(arity)s];\n" % \
            {'type': self.ctype,
             'vec_name': self.c_vec_name(),
             'arity': self.map.arity * cdim * facet_mult}

    def c_wrapper_dec(self):
        val = ""
        if self._is_mixed_mat:
            rows, cols = self.data.sparsity.shape
            for i in range(rows):
                for j in range(cols):
                    val += "Mat %(iname)s; MatNestGetSubMat(%(name)s_, %(i)d, %(j)d, &%(iname)s);\n" \
                        % {'name': self.c_arg_name(),
                           'iname': self.c_arg_name(i, j),
                           'i': i,
                           'j': j}
        elif self._is_mat:
            val += "Mat %(iname)s = %(name)s_;\n" % {'name': self.c_arg_name(),
                                                     'iname': self.c_arg_name(0, 0)}
        return val

    def c_ind_data(self, idx, i, j=0, is_top=False, layers=1, offset=None):
        return "%(name)s + (%(map_name)s[i * %(arity)s + %(idx)s]%(top)s%(off_mul)s%(off_add)s)* %(dim)s%(off)s" % \
            {'name': self.c_arg_name(i),
             'map_name': self.c_map_name(i, 0),
             'arity': self.map.split[i].arity,
             'idx': idx,
             'top': ' + start_layer' if is_top else '',
             'dim': self.data[i].cdim,
             'off': ' + %d' % j if j else '',
             'off_mul': ' * %d' % offset if is_top and offset is not None else '',
             'off_add': ' + %d' % offset if not is_top and offset is not None else ''}

    def ast_ind_data(self, idx, i, j=0, is_top=False, layers=1, offset=None):
        index = ast.Prod(ast.Symbol("i"), ast.Symbol(self.map.split[i].arity))
        index = ast.Sum(index, ast.Symbol(idx))
        tail = ast.Symbol(self.c_map_name(i, 0), rank=(index,))
        if is_top:
            tail = ast.Sum(tail, ast.Symbol("start_layer"))
            if offset:
                tail = ast.Prod(tail, ast.Symbol(offset))
        if not is_top and offset:
            tail = ast.Sum(tail, offset)
        tail = ast.Prod(tail, ast.Symbol(self.data[i].cdim))
        if j:
            tail = ast.Sum(tail, ast.Symbol(j))
        return ast.Sum(ast.Symbol(self.c_arg_name(i)), tail)
        # return "%(name)s + (%(map_name)s[i * %(arity)s + %(idx)s]%(top)s%(off_mul)s%(off_add)s)* %(dim)s%(off)s" % \
        #     {'name': self.c_arg_name(i),
        #      'map_name': self.c_map_name(i, 0),
        #      'arity': self.map.split[i].arity,
        #      'idx': idx,
        #      'top': ' + start_layer' if is_top else '',
        #      'dim': self.data[i].cdim,
        #      'off': ' + %d' % j if j else '',
        #      'off_mul': ' * %d' % offset if is_top and offset is not None else '',
        #      'off_add': ' + %d' % offset if not is_top and offset is not None else ''}

    def ast_ind_data_xtr(self, idx, i, j=0, is_top=False, layers=1):
        tail = ast.Symbol("xtr_" + self.c_map_name(i, 0), rank=(idx,))
        if is_top:
            tail = ast.Sum(tail, ast.Symbol(start_layer))
            tail = ast.Prod(tail, ast.Symbol('_'+self.c_offset_name(i, 0), rank=(idx,)))
        tail = ast.Prod(tail, ast.Symbol(1 if self._flatten else self.data[i].cdim))
        if j:
            tail = ast.Sum(tail, ast.Symbol(j))
        return ast.Sum(ast.Symbol(self.c_arg_name(i)), tail)
        # return "%(name)s + (xtr_%(map_name)s[%(idx)s]%(top)s%(offset)s)*%(dim)s%(off)s" % \
        #     {'name': self.c_arg_name(i),
        #      'map_name': self.c_map_name(i, 0),
        #      'idx': idx,
        #      'top': ' + start_layer' if is_top else '',
        #      'dim': 1 if self._flatten else str(self.data[i].cdim),
        #      'off': ' + %d' % j if j else '',
        #      'offset': ' * _'+self.c_offset_name(i, 0)+'['+idx+']' if is_top else ''}

    def c_kernel_arg_name(self, i, j):
        return "p_%s" % self.c_arg_name(i, j)

    def c_global_reduction_name(self, count=None):
        return self.c_arg_name()

    def c_local_tensor_name(self, i, j):
        return self.c_kernel_arg_name(i, j)

    def c_kernel_arg(self, count, i=0, j=0, shape=(0,), is_top=False, layers=1):
        if self._uses_itspace:
            if self._is_mat:
                if self.data[i, j]._is_vector_field:
                    return self.c_kernel_arg_name(i, j)
                elif self.data[i, j]._is_scalar_field:
                    return "(%(t)s (*)[%(dim)d])&%(name)s" % \
                        {'t': self.ctype,
                         'dim': shape[0],
                         'name': self.c_kernel_arg_name(i, j)}
                else:
                    raise RuntimeError("Don't know how to pass kernel arg %s" % self)
            else:
                if self.data is not None and self.data.dataset._extruded:
                    return self.ast_ind_data_xtr("i_%d" % self.idx.index, i, is_top=is_top, layers=layers)
                elif self._flatten:
                    return "%(name)s + %(map_name)s[i * %(arity)s + i_0 %% %(arity)d] * %(dim)s + (i_0 / %(arity)d)" % \
                        {'name': self.c_arg_name(),
                         'map_name': self.c_map_name(0, i),
                         'arity': self.map.arity,
                         'dim': self.data[i].cdim}
                else:
                    return self.c_ind_data("i_%d" % self.idx.index, i)
        elif self._is_indirect:
            if self._is_vec_map:
                return self.c_vec_name()
            return self.c_ind_data(self.idx, i)
        elif self._is_global_reduction:
            return self.c_global_reduction_name(count)
        elif isinstance(self.data, Global):
            return self.c_arg_name(i)
        else:
            return "%(name)s + i * %(dim)s" % {'name': self.c_arg_name(i),
                                               'dim': self.data[i].cdim}

    def ast_kernel_arg(self, count, i=0, j=0, shape=(0,), is_top=False, layers=1):
        if self._uses_itspace:
            if self._is_mat:
                if self.data[i, j]._is_vector_field:
                    return ast.Symbol(self.c_kernel_arg_name(i, j))
                elif self.data[i, j]._is_scalar_field:
                    return ast.Cast(self.ctype + "(*)[" + str(shape[0]) + "]", ast.Addr(ast.Symbol(self.c_kernel_arg_name(i, j))))
                    # flat_block = "(%(t)s (*)[%(dim)d])&%(name)s" % \
                    #     {'t': self.ctype,
                    #      'dim': shape[0],
                    #      'name': self.c_kernel_arg_name(i, j)}
                    # return ast.FlatBlock(flat_block)
                else:
                    raise RuntimeError("Don't know how to pass kernel arg %s" % self)
            else:
                if self.data is not None and self.data.dataset._extruded:
                    return self.ast_ind_data_xtr("i_%d" % self.idx.index, i, is_top=is_top, layers=layers)
                elif self._flatten:
                    index = ast.Prod(ast.Symbol("i"), ast.Symbol(self.map.arity))
                    index = ast.Sum(index, ast.Mod(ast.Symbol("i_0"), ast.Symbol(self.map.arity)))
                    tail = ast.Symbol(self.c_map_name(i, 0), rank=(index,))
                    tail = ast.Prod(tail, ast.Symbol(self.data[i].cdim))
                    tail = ast.Sum(tail, ast.Div(ast.Symbol("i_0"), ast.Symbol(self.map.arity)))
                    return ast.Sum(ast.Symbol(self.c_arg_name()), tail)
                    # return "%(name)s + %(map_name)s[i * %(arity)s + i_0 %% %(arity)d] * %(dim)s + (i_0 / %(arity)d)" % \
                    #     {'name': self.c_arg_name(),
                    #      'map_name': self.c_map_name(0, i),
                    #      'arity': self.map.arity,
                    #      'dim': self.data[i].cdim}
                else:
                    return self.ast_ind_data("i_%d" % self.idx.index, i)
        elif self._is_indirect:
            if self._is_vec_map:
                return ast.Symbol(self.c_vec_name())
            return self.ast_ind_data(self.idx, i)
        elif self._is_global_reduction:
            return ast.Symbol(self.c_global_reduction_name(count))
        elif isinstance(self.data, Global):
            return ast.Symbol(self.c_arg_name(i))
        else:
            return ast.Sum(ast.Symbol(self.c_arg_name(i)), ast.Prod(ast.Symbol("i"), ast.Symbol(self.data[i].cdim)))

    def c_vec_init(self, is_top, layers, is_facet=False):
        is_top_init = is_top
        val = []
        vec_idx = 0
        for i, (m, d) in enumerate(zip(self.map, self.data)):
            is_top = is_top_init and m.iterset._extruded
            if self._flatten:
                for k in range(d.dataset.cdim):
                    for idx in range(m.arity):
                        val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                                   {'vec_name': self.c_vec_name(),
                                    'idx': vec_idx,
                                    'data': self.c_ind_data(idx, i, k, is_top=is_top, layers=layers,
                                                            offset=m.offset[idx] if is_top else None)})
                        vec_idx += 1
                    # In the case of interior horizontal facets the map for the
                    # vertical does not exist so it has to be dynamically
                    # created by adding the offset to the map of the current
                    # cell. In this way the only map required is the one for
                    # the bottom layer of cells and the wrapper will make sure
                    # to stage in the data for the entire map spanning the facet.
                    if is_facet:
                        for idx in range(m.arity):
                            val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                                       {'vec_name': self.c_vec_name(),
                                        'idx': vec_idx,
                                        'data': self.c_ind_data(idx, i, k, is_top=is_top, layers=layers,
                                                                offset=m.offset[idx])})
                            vec_idx += 1
            else:
                for idx in range(m.arity):
                    val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                               {'vec_name': self.c_vec_name(),
                                'idx': vec_idx,
                                'data': self.c_ind_data(idx, i, is_top=is_top, layers=layers,
                                                        offset=m.offset[idx] if is_top else None)})
                    vec_idx += 1
                if is_facet:
                    for idx in range(m.arity):
                        val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                                   {'vec_name': self.c_vec_name(),
                                    'idx': vec_idx,
                                    'data': self.c_ind_data(idx, i, is_top=is_top, layers=layers,
                                                            offset=m.offset[idx])})
                        vec_idx += 1
        return ";\n".join(val)

    def ast_addto(self, i, j, buf_name, tmp_name, tmp_decl,
                  extruded=None, is_facet=False, applied_blas=False):
        maps = as_tuple(self.map, Map)
        nrows = maps[0].split[i].arity
        ncols = maps[1].split[j].arity
        rows_ast = ast.Sum(ast.Symbol(self.c_map_name(0, i)), ast.Prod(ast.Symbol("i"), ast.Symbol(nrows)))
        cols_ast = ast.Sum(ast.Symbol(self.c_map_name(1, j)), ast.Prod(ast.Symbol("i"), ast.Symbol(ncols)))

        if extruded is not None:
            rows_ast = ast.Symbol(extruded + self.c_map_name(0, i))
            cols_ast = ast.Symbol(extruded + self.c_map_name(1, j))

        if is_facet:
            nrows *= 2
            ncols *= 2

        ret = []
        rbs, cbs = self.data.sparsity[i, j].dims[0][0]
        rdim = rbs * nrows
        addto_name = buf_name
        addto = 'MatSetValuesLocal'
        if self.data._is_vector_field:
            addto = 'MatSetValuesBlockedLocal'
            if self._flatten:
                ridx_l = ast.Sum(ast.Prod(ast.Symbol(rbs), ast.Symbol("j")), ast.Symbol("k"))
                cidx_l = ast.Sum(ast.Prod(ast.Symbol(cbs), ast.Symbol("l")), ast.Symbol("m"))
                ridx_r = ast.Sum(ast.Prod(ast.Symbol(nrows), ast.Symbol("k")), ast.Symbol("j"))
                cidx_r = ast.Sum(ast.Prod(ast.Symbol(ncols), ast.Symbol("m")), ast.Symbol("l"))
                if applied_blas:
                    # idx = "[(%%(ridx)s)*%d + (%%(cidx)s)]" % rdim
                    # We only have one rank
                    rank_l = (ast.Sum(ast.Prod(ridx_l, ast.Symbol(rdim)), cidx_l),)
                    rank_r = (ast.Sum(ast.Prod(ridx_r, ast.Symbol(rdim)), cidx_r),)
                else:
                    # idx = "[%(ridx)s][%(cidx)s]"
                    # This has two ranks
                    rank_l = (ridx_l, cidx_l)
                    rank_r = (ridx_r, cidx_r)
                ret = []
                # Shuffle xxx yyy zzz into xyz xyz xyz
                # %(tmp_name)s%(idx_l)s = %(buf_name)s%(idx_r)s;
                nest = ast.Assign(ast.Symbol(tmp_name, rank=rank_l),
                                  ast.Symbol(buf_name, rank=rank_r))
                nest = ast.For(ast.Decl("int", ast.Symbol("m"), ast.Symbol(0)),
                               ast.Less(ast.Symbol("m"), ast.Symbol(cbs)),
                               ast.Incr(ast.Symbol("m"), ast.Symbol(1)), nest)
                nest = ast.For(ast.Decl("int", ast.Symbol("l"), ast.Symbol(0)),
                               ast.Less(ast.Symbol("l"), ast.Symbol(ncols)),
                               ast.Incr(ast.Symbol("l"), ast.Symbol(1)), nest)
                nest = ast.For(ast.Decl("int", ast.Symbol("k"), ast.Symbol(0)),
                               ast.Less(ast.Symbol("k"), ast.Symbol(rbs)),
                               ast.Incr(ast.Symbol("k"), ast.Symbol(1)), nest)
                nest = ast.For(ast.Decl("int", ast.Symbol("j"), ast.Symbol(0)),
                               ast.Less(ast.Symbol("j"), ast.Symbol(nrows)),
                               ast.Incr(ast.Symbol("j"), ast.Symbol(1)), nest)
                ret = [tmp_decl, nest]
                addto_name = tmp_name

        ret.append(ast.FunCall(addto,
                               ast.Symbol(self.c_arg_name(i, j)),
                               ast.Symbol(nrows), rows_ast,
                               ast.Symbol(ncols), cols_ast,
                               ast.Cast("const PetscScalar *", ast.Symbol(addto_name)),
                               ast.Symbol("INSERT_VALUES") if self.access == WRITE else ast.Symbol("ADD_VALUES")))
        return ret

    def c_local_tensor_dec(self, extents, i, j):
        if self._is_mat:
            size = 1
        else:
            size = self.data.split[i].cdim
        return tuple([d * size for d in extents])

    def c_zero_tmp(self, i, j):
        t = self.ctype
        if self.data[i, j]._is_scalar_field:
            idx = ''.join(["[i_%d]" % ix for ix in range(len(self.data.dims))])
            return "%(name)s%(idx)s = (%(t)s)0" % \
                {'name': self.c_kernel_arg_name(i, j), 't': t, 'idx': idx}
        elif self.data[i, j]._is_vector_field:
            if self._flatten:
                return "%(name)s[0][0] = (%(t)s)0" % \
                    {'name': self.c_kernel_arg_name(i, j), 't': t}
            size = np.prod(self.data[i, j].dims)
            return "memset(%(name)s, 0, sizeof(%(t)s) * %(size)s)" % \
                {'name': self.c_kernel_arg_name(i, j), 't': t, 'size': size}
        else:
            raise RuntimeError("Don't know how to zero temp array for %s" % self)

    def ast_add_offset(self, is_facet=False):
        if not self.map.iterset._extruded:
            return ""
        val_ast = []
        vec_idx = 0
        for i, (m, d) in enumerate(zip(self.map, self.data)):
            for k in range(d.dataset.cdim if self._flatten else 1):
                for idx in range(m.arity):
                    lhs = ast.Symbol(self.c_vec_name(), rank=(vec_idx,))
                    rhs = ast.Symbol(m.offset[idx] * d.dataset.cdim)
                    # rhs = ast.Symbol(self.c_offset_name(i, 0), rank=(idx,))
                    # rhs = ast.Prod(rhs, ast.Symbol(d.dataset.cdim))
                    val_ast.append(ast.Incr(lhs, rhs))
                    # val.append("%(name)s[%(j)d] += %(off)d * %(dim)s;" %
                    #            {'name': self.c_vec_name(),
                    #             'i': idx,
                    #             'j': vec_idx,
                    #             'offset': self.c_offset_name(i, 0),
                    #             'dim': d.dataset.cdim,
                    #             'off': m.offset[idx]})
                    vec_idx += 1
                if is_facet:
                    for idx in range(m.arity):
                        lhs = ast.Symbol(self.c_vec_name(), rank=(vec_idx,))
                        rhs = ast.Symbol(m.offset[idx] * d.dataset.cdim)
                        # rhs = ast.Symbol(self.c_offset_name(i, 0), rank=(idx,))
                        # rhs = ast.Prod(rhs, ast.Symbol(d.dataset.cdim))
                        val_ast.append(ast.Incr(lhs, rhs))
                        # val.append("%(name)s[%(j)d] += %(off)d * %(dim)s;" %
                        #            {'name': self.c_vec_name(),
                        #             'i': idx,
                        #             'j': vec_idx,
                        #             'offset': self.c_offset_name(i, 0),
                        #             'dim': d.dataset.cdim,
                        #             'off': m.offset[idx]})
                        vec_idx += 1
        return val_ast

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

    def c_map_decl(self, is_facet=False):
        if self._is_mat:
            dsets = self.data.sparsity.dsets
        else:
            dsets = (self.data.dataset,)
        val = []
        for i, (map, dset) in enumerate(zip(as_tuple(self.map, Map), dsets)):
            for j, (m, d) in enumerate(zip(map, dset)):
                dim = m.arity
                if self._is_dat and self._flatten:
                    dim *= d.cdim
                if is_facet:
                    dim *= 2
                val.append("int xtr_%(name)s[%(dim)s];" %
                           {'name': self.c_map_name(i, j), 'dim': dim})
        return '\n'.join(val)+'\n'

    def ast_map_init(self, is_top=False, layers=1, is_facet=False):
        if self._is_mat:
            dsets = self.data.sparsity.dsets
        else:
            dsets = (self.data.dataset,)
        val_ast = []
        for i, (map, dset) in enumerate(zip(as_tuple(self.map, Map), dsets)):
            for j, (m, d) in enumerate(zip(map, dset)):
                for idx in range(m.arity):
                    if self._is_dat and self._flatten and d.cdim > 1:
                        for k in range(d.cdim):
                            lhs = ast.Symbol("xtr_" + self.c_map_name(i, j), rank=(m.arity * k + idx,))
                            rhs = ast.Symbol(self.c_map_name(i, j), rank=(ast.Sum(ast.Prod(ast.Symbol("i"), ast.Symbol(m.arity)), ast.Symbol(idx)),))
                            if is_top:
                                rhs = ast.Sum(rhs, ast.Prod(ast.Symbol("start_layer"), ast.Symbol(m.offset[idx])))
                            rhs = ast.Prod(ast.Symbol(d.cdim), rhs)
                            if k > 0:
                                rhs = ast.Sum(rhs, ast.Symbol(k))
                            val_ast.append(ast.Assign(lhs, rhs))
                            # val.append("xtr_%(name)s[%(ind_flat)s] = %(dat_dim)s * (*(%(name)s + i * %(dim)s + %(ind)s)%(off_top)s)%(offset)s;" %
                            #            {'name': self.c_map_name(i, j),
                            #             'dim': m.arity,
                            #             'ind': idx,
                            #             'dat_dim': d.cdim,
                            #             'ind_flat': m.arity * k + idx,
                            #             'offset': ' + '+str(k) if k > 0 else '',
                            #             'off_top': ' + start_layer * '+str(m.offset[idx]) if is_top else ''})
                    else:
                        lhs = ast.Symbol("xtr_" + self.c_map_name(i, j), rank=(idx,))
                        rhs = ast.Symbol(self.c_map_name(i, j), rank=(ast.Sum(ast.Prod(ast.Symbol("i"), ast.Symbol(m.arity)), ast.Symbol(idx)),))
                        if is_top:
                            rhs = ast.Sum(rhs, ast.Prod(ast.Symbol("start_layer"), ast.Symbol(m.offset[idx])))
                        val_ast.append(ast.Assign(lhs, rhs))
                        # val.append("xtr_%(name)s[%(ind)s] = *(%(name)s + i * %(dim)s + %(ind)s)%(off_top)s;" %
                        #            {'name': self.c_map_name(i, j),
                        #             'dim': m.arity,
                        #             'ind': idx,
                        #             'off_top': ' + start_layer * '+str(m.offset[idx]) if is_top else ''})
                if is_facet:
                    for idx in range(m.arity):
                        if self._is_dat and self._flatten and d.cdim > 1:
                            for k in range(d.cdim):
                                lhs = ast.Symbol("xtr_" + self.c_map_name(i, j), rank=(m.arity * (k + d.cdim) + idx,))
                                rhs = ast.Symbol(self.c_map_name(i, j), rank=(ast.Sum(ast.Prod(ast.Symbol("i"), ast.Symbol(m.arity)), ast.Symbol(idx)),))
                                rhs = ast.Sum(rhs, ast.Symbol(m.offset[idx]))
                                rhs = ast.Prod(rhs, ast.Symbol(d.cdim))
                                if k > 0:
                                    rhs = ast.Sum(rhs, ast.Symbol(k))
                                val_ast.append(ast.Assign(lhs, rhs))
                                # val.append("xtr_%(name)s[%(ind_flat)s] = %(dat_dim)s * (*(%(name)s + i * %(dim)s + %(ind)s)%(off)s)%(offset)s;" %
                                #            {'name': self.c_map_name(i, j),
                                #             'dim': m.arity,
                                #             'ind': idx,
                                #             'dat_dim': d.cdim,
                                #             'ind_flat': m.arity * (k + d.cdim) + idx,
                                #             'offset': ' + '+str(k) if k > 0 else '',
                                #             'off': ' + ' + str(m.offset[idx])})
                        else:
                            lhs = ast.Symbol("xtr_" + self.c_map_name(i, j), rank=(idx + m.arity,))
                            rhs = ast.Symbol(self.c_map_name(i, j), rank=(ast.Sum(ast.Prod(ast.Symbol("i"), ast.Symbol(m.arity)), ast.Symbol(idx)),))
                            if is_top:
                                rhs = ast.Sum(rhs, ast.Symbol("start_layer"))
                            rhs = ast.Sum(rhs, ast.Symbol(m.offset[idx]))
                            val_ast.append(ast.Assign(lhs, rhs))
                            # val.append("xtr_%(name)s[%(ind)s] = *(%(name)s + i * %(dim)s + %(ind_zero)s)%(off_top)s%(off)s;" %
                            #            {'name': self.c_map_name(i, j),
                            #             'dim': m.arity,
                            #             'ind': idx + m.arity,
                            #             'ind_zero': idx,
                            #             'off_top': ' + start_layer' if is_top else '',
                            #             'off': ' + ' + str(m.offset[idx])})
        return '\n'.join([a.gencode() for a in val_ast])+'\n'

    def ast_map_bcs(self, sign):
        maps = as_tuple(self.map, Map)
        val_ast = []
        val_ast_bottom = []
        val_ast_top = []
        # To throw away boundary condition values, we subtract a large
        # value from the map to make it negative then add it on later to
        # get back to the original
        max_int = 10000000

        need_bottom = False
        # Apply any bcs on the first (bottom) layer
        for i, map in enumerate(maps):
            if not map.iterset._extruded:
                continue
            for j, m in enumerate(map):
                if 'bottom' not in m.implicit_bcs:
                    continue
                need_bottom = True
                for idx in range(m.arity):
                    if m.bottom_mask[idx] < 0:
                        lhs = ast.Symbol("xtr_" + self.c_map_name(i, j), rank=(idx,))
                        rhs = ast.Symbol(max_int)
                        if sign == '-':
                            rhs = ast.Neg(rhs)
                        val_ast.append(ast.Incr(lhs, rhs))
                        # val.append("xtr_%(name)s[%(ind)s] %(sign)s= %(val)s;" %
                        #            {'name': self.c_map_name(i, j),
                        #             'val': max_int,
                        #             'ind': idx,
                        #             'sign': sign})
        if need_bottom:
            cond = ast.Eq(ast.Symbol("j_0"), ast.Symbol(0))
            val_ast_bottom = [ast.If(cond, [val_ast])]
            val_ast = []

        need_top = False
        # Apply any bcs on last (top) layer
        for i, map in enumerate(maps):
            if not map.iterset._extruded:
                continue
            for j, m in enumerate(map):
                if 'top' not in m.implicit_bcs:
                    continue
                need_top = True
                for idx in range(m.arity):
                    if m.top_mask[idx] < 0:
                        lhs = ast.Symbol("xtr_" + self.c_map_name(i, j), rank=(idx,))
                        rhs = ast.Symbol(max_int)
                        if sign == '-':
                            rhs = ast.Neg(rhs)
                        val_ast.append(ast.Incr(lhs, rhs))
                        # val.append("xtr_%(name)s[%(ind)s] %(sign)s= %(val)s;" %
                        #            {'name': self.c_map_name(i, j),
                        #             'val': max_int,
                        #             'ind': idx,
                        #             'sign': sign})
        if need_top:
            cond = ast.Eq(ast.Symbol("j_0"), ast.Sub(ast.Symbol("end_layer"), ast.Symbol(1)))
            val_ast_top = [ast.If(cond, [val_ast])]
        return val_ast_bottom + val_ast_top

    def ast_add_offset_map(self, is_facet=False):
        if self._is_mat:
            dsets = self.data.sparsity.dsets
        else:
            dsets = (self.data.dataset,)
        val_ast = []
        for i, (map, dset) in enumerate(zip(as_tuple(self.map, Map), dsets)):
            if not map.iterset._extruded:
                continue
            for j, (m, d) in enumerate(zip(map, dset)):
                for idx in range(m.arity):
                    if self._is_dat and self._flatten and d.cdim > 1:
                        for k in range(d.cdim):
                            lhs = ast.Symbol("xtr_" + self.c_map_name(i, j), rank=(m.arity * k + idx,))
                            rhs = ast.Symbol(m.offset[idx] * d.cdim)
                            val_ast.append(ast.Incr(lhs, rhs))
                            # val.append("xtr_%(name)s[%(ind_flat)s] += %(off)s[%(ind)s] * %(dim)s;" %
                            #            {'name': self.c_map_name(i, j),
                            #             'off': self.c_offset_name(i, j),
                            #             'ind': idx,
                            #             'ind_flat': m.arity * k + idx,
                            #             'dim': d.cdim})
                    else:
                        lhs = ast.Symbol("xtr_" + self.c_map_name(i, j), rank=(idx,))
                        rhs = ast.Symbol(m.offset[idx])
                        val_ast.append(ast.Incr(lhs, rhs))
                        # val.append("xtr_%(name)s[%(ind)s] += %(off)s[%(ind)s];" %
                        #            {'name': self.c_map_name(i, j),
                        #             'off': self.c_offset_name(i, j),
                        #             'ind': idx})
                if is_facet:
                    for idx in range(m.arity):
                        if self._is_dat and self._flatten and d.cdim > 1:
                            for k in range(d.cdim):
                                lhs = ast.Symbol("xtr_" + self.c_map_name(i, j), rank=(m.arity * (k + d.cdim) + idx,))
                                rhs = ast.Symbol(m.offset[idx] * d.cdim)
                                val_ast.append(ast.Incr(lhs, rhs))
                                # val.append("xtr_%(name)s[%(ind_flat)s] += %(off)s[%(ind)s] * %(dim)s;" %
                                #            {'name': self.c_map_name(i, j),
                                #             'off': self.c_offset_name(i, j),
                                #             'ind': idx,
                                #             'ind_flat': m.arity * (k + d.cdim) + idx,
                                #             'dim': d.cdim})
                        else:
                            lhs = ast.Symbol("xtr_" + self.c_map_name(i, j), rank=(m.arity + idx,))
                            rhs = ast.Symbol(m.offset[idx])
                            val_ast.append(ast.Incr(lhs, rhs))
                            # val.append("xtr_%(name)s[%(ind)s] += %(off)s[%(ind_zero)s];" %
                            #            {'name': self.c_map_name(i, j),
                            #             'off': self.c_offset_name(i, j),
                            #             'ind': m.arity + idx,
                            #             'ind_zero': idx})
        return val_ast

    def c_offset_init(self):
        maps = as_tuple(self.map, Map)
        val = []
        for i, map in enumerate(maps):
            if not map.iterset._extruded:
                continue
            for j, m in enumerate(map):
                val.append("int *%s" % self.c_offset_name(i, j))
        if len(val) == 0:
            return ""
        return ", " + ", ".join(val)

    def ast_buffer_decl(self, size, idx, buf_name, is_facet=False, init=True):
        buf_type = self.data.ctype
        dim = len(size)
        compiler = coffee.plan.compiler
        isa = coffee.plan.isa
        align = compiler['align'](isa["alignment"]) if compiler and size[-1] % isa["dp_reg"] == 0 else ""
        sizes = tuple([d * (2 if is_facet else 1) for d in size])
        buf = ast.Symbol(buf_name, rank=sizes)
        init = ast.ArrayInit("%(init)s" % {"init": "{" * dim + "0.0" + "}" * dim})
        if init:
            return ast.Decl(buf_type, buf, init, attributes=[align])
        return ast.Decl(buf_type, buf, attributes=[align])

    def ast_buffer_gather(self, size, idx, buf_name):
        dim = 1 if self._flatten else self.data.cdim
        val_ast = []
        for j in range(dim):
            index = ast.Prod(ast.Symbol("i_0"), ast.Symbol(dim))
            if j:
                index = ast.Sum(index, ast.Symbol(j))
            lhs = ast.Symbol(buf_name, rank=(index,))
            # rhs = ast.Symbol(self.c_kernel_arg(idx), rank=(ast.Symbol(j),))
            rhs = self.c_kernel_arg(idx)
            if j:
                rhs = ast.Sum(rhs, ast.Symbol(j))
            rhs = ast.Deref(rhs)
            val_ast.append(ast.Assign(lhs, rhs))
        return val_ast
        # return ";\n".join(["%(name)s[i_0*%(dim)d%(ofs)s] = *(%(ind)s%(ofs)s);\n" %
        #                    {"name": buf_name,
        #                     "dim": dim,
        #                     "ind": self.c_kernel_arg(idx),
        #                     "ofs": " + %s" % j if j else ""} for j in range(dim)])

    def ast_buffer_scatter_vec(self, count, i, j, mxofs, buf_name):
        dim = 1 if self._flatten else self.data.split[i].cdim
        val_ast = []
        for o in range(dim):
            lhs = self.c_kernel_arg(count, i, j)
            index = ast.Prod(ast.Symbol("i_0"), ast.Symbol(dim))
            if o:
                index = ast.Sum(index, ast.Symbol(o))
                lhs = ast.Sum(lhs, ast.Symbol(o))
            lhs = ast.Deref(lhs)
            if mxofs:
                index = ast.Sum(index, ast.Symbol(mxofs[0] * dim))
            rhs = ast.Symbol(buf_name, rank=(index,))
            if self.access == WRITE:
                val_ast.append(ast.Assign(lhs, rhs))
            else:
                val_ast.append(ast.Incr(lhs, rhs))
        return val_ast
        # return ";\n".join(["*(%(ind)s%(nfofs)s) %(op)s %(name)s[i_0*%(dim)d%(nfofs)s%(mxofs)s]" %
        #                    {"ind": self.c_kernel_arg(count, i, j),
        #                     "op": "=" if self.access == WRITE else "+=",
        #                     "name": buf_name,
        #                     "dim": dim,
        #                     "nfofs": " + %d" % o if o else "",
        #                     "mxofs": " + %d" % (mxofs[0] * dim) if mxofs else ""}
        #                    for o in range(dim)])


class JITModule(base.JITModule):

    _cppargs = []
    _libraries = []
    _system_headers = []
    _extension = 'c'

    def __init__(self, kernel, itspace, *args, **kwargs):
        """
        A cached compiled function to execute for a specified par_loop.

        See :func:`~.par_loop` for the description of arguments.

        .. warning ::

           Note to implementors.  This object is *cached*, and therefore
           should not hold any long term references to objects that
           you want to be collected.  In particular, after the
           ``args`` have been inspected to produce the compiled code,
           they **must not** remain part of the object's slots,
           otherwise they (and the :class:`~.Dat`\s, :class:`~.Map`\s
           and :class:`~.Mat`\s they reference) will never be collected.
        """
        # Return early if we were in the cache.
        if self._initialized:
            return
        self._kernel = kernel
        self._fun = None
        self._itspace = itspace
        self._args = args
        self._direct = kwargs.get('direct', False)
        self._iteration_region = kwargs.get('iterate', ALL)
        self._initialized = True
        # Copy the class variables, so we don't overwrite them
        self._cppargs = dcopy(type(self)._cppargs)
        self._libraries = dcopy(type(self)._libraries)
        self._system_headers = dcopy(type(self)._system_headers)
        self.set_argtypes(itspace.iterset, *args)
        self.compile()

    @collective
    def __call__(self, *args):
        return self._fun(*args)

    @property
    def _wrapper_name(self):
        return 'wrap_%s' % self._kernel.name

    @collective
    def compile(self):
        # If we weren't in the cache we /must/ have arguments
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule has no args associated with it, should never happen")

        compiler = coffee.plan.compiler
        blas = coffee.plan.blas
        blas_header, blas_namespace, externc_open, externc_close = ("", "", "", "")
        if self._kernel._applied_blas:
            blas_header = blas.get('header')
            blas_namespace = blas.get('namespace', '')
            if blas['name'] == 'eigen':
                externc_open = 'extern "C" {'
                externc_close = '}'
        headers = "\n".join([compiler.get('vect_header', ""), blas_header])
        if any(arg._is_soa for arg in self._args):
            kernel_code = """
            #define OP2_STRIDE(a, idx) a[idx]
            %(header)s
            %(namespace)s
            %(externc_open)s
            %(code)s
            #undef OP2_STRIDE
            """ % {'code': self._kernel.code(),
                   'externc_open': externc_open,
                   'namespace': blas_namespace,
                   'header': headers}
        else:
            kernel_code = """
            %(header)s
            %(namespace)s
            %(externc_open)s
            %(code)s
            """ % {'code': self._kernel.code(),
                   'externc_open': externc_open,
                   'namespace': blas_namespace,
                   'header': headers}
        code_to_compile = strip(dedent(self._wrapper) % self.generate_code())

        _const_decs = '\n'.join([const._format_declaration()
                                for const in Const._definitions()]) + '\n'

        code_to_compile = """
        #include <petsc.h>
        #include <stdbool.h>
        #include <math.h>
        %(sys_headers)s
        %(consts)s

        %(kernel)s

        %(wrapper)s
        %(externc_close)s
        """ % {'consts': _const_decs, 'kernel': kernel_code,
               'wrapper': code_to_compile,
               'externc_close': externc_close,
               'sys_headers': '\n'.join(self._kernel._headers + self._system_headers)}

        self._dump_generated_code(code_to_compile)
        if configuration["debug"]:
            self._wrapper_code = code_to_compile

        extension = self._extension
        cppargs = self._cppargs
        cppargs += ["-I%s/include" % d for d in get_petsc_dir()] + \
                   ["-I%s" % d for d in self._kernel._include_dirs] + \
                   ["-I%s" % os.path.abspath(os.path.dirname(__file__))]
        if compiler:
            cppargs += [compiler[coffee.plan.isa['inst_set']]]
        ldargs = ["-L%s/lib" % d for d in get_petsc_dir()] + \
                 ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()] + \
                 ["-lpetsc", "-lm"] + self._libraries
        if self._kernel._applied_blas:
            blas_dir = blas['dir']
            if blas_dir:
                cppargs += ["-I%s/include" % blas_dir]
                ldargs += ["-L%s/lib" % blas_dir]
            ldargs += blas['link']
            if blas['name'] == 'eigen':
                extension = "cpp"
        self._fun = compilation.load(code_to_compile,
                                     extension,
                                     self._wrapper_name,
                                     cppargs=cppargs,
                                     ldargs=ldargs,
                                     argtypes=self._argtypes,
                                     restype=None,
                                     compiler=compiler.get('name'))
        # Blow away everything we don't need any more
        del self._args
        del self._kernel
        del self._itspace
        del self._direct
        return self._fun

    def generate_code(self):

        def ast_itspace_loop(i, d, body):
            ind = ast.Symbol("i_" + str(i))
            return ast.For(ast.Decl("int", ind, ast.Symbol(0)),
                           ast.Less(ind, ast.Symbol(d)),
                           ast.Incr(ind, ast.Symbol(1)), body)

        def c_const_arg(c):
            return '%s *%s_' % (c.ctype, c.name)

        def c_const_init(c):
            d = {'name': c.name,
                 'type': c.ctype}
            if c.cdim == 1:
                return '%(name)s = *%(name)s_' % d
            tmp = '%(name)s[%%(i)s] = %(name)s_[%%(i)s]' % d
            return ';\n'.join([tmp % {'i': i} for i in range(c.cdim)])

        def ast_extrusion_loop(body):
            if self._direct:
                return ast.Block(body, open_scope=True)
            i = ast.Symbol("j_0")
            return ast.For(ast.Decl("int", i, ast.Symbol("start_layer")),
                           ast.Less(i, ast.Symbol("end_layer")),
                           ast.Incr(i, ast.Symbol(1)), body)

        def get_kernel_decl_node(kernel_ast):
            # Find the reference to node continaing the kernel header
            if isinstance(kernel_ast, ast.FunDecl):
                return kernel_ast
            if kernel_ast and not isinstance(kernel_ast, str):
                for child in kernel_ast.children:
                    c = get_kernel_decl_node(child)
                    if isinstance(c, ast.FunDecl):
                        return c
            return None

        def change_kernel_loop_index_ids(visitor):
            fors = visitor['search'][ast.For]
            for f in fors:
                old_name = str(f.dim)
                for_visitor = visit(f, search=ast.Symbol)
                for symbol in for_visitor['search'][ast.Symbol]:
                    ast_update_id(symbol, old_name, 0)

        def rename_var(visitor, old_name, new_name):
            symbols = visitor['search'][ast.Symbol]
            for s in symbols:
                ast_update_id(s, old_name, new_name, replace=True)

        _ssinds_arg = ""
        _index_expr = "n"
        is_top = (self._iteration_region == ON_TOP)
        is_facet = (self._iteration_region == ON_INTERIOR_FACETS)

        if isinstance(self._itspace._iterset, Subset):
            _ssinds_arg = "int* ssinds,"
            _index_expr = "ssinds[n]"

        _wrapper_args = ', '.join([arg.c_wrapper_arg() for arg in self._args])

        # Pass in the is_facet flag to mark the case when it's an interior horizontal facet in
        # an extruded mesh.
        _wrapper_decs = ';\n'.join([arg.c_wrapper_dec() for arg in self._args])

        _vec_decs = ';\n'.join([arg.c_vec_dec(is_facet=is_facet) for arg in self._args if arg._is_vec_map])

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

        _vec_inits = ';\n'.join([arg.c_vec_init(is_top, self._itspace.layers, is_facet=is_facet) for arg in self._args
                                 if not arg._is_mat and arg._is_vec_map])

        indent = lambda t, i: ('\n' + '  ' * i).join(t.split('\n'))

        _map_decl = ""
        _apply_offset = []
        _map_init = ""
        _map_bcs_m = []
        _map_bcs_p = []
        _layer_arg = ""
        if self._itspace._extruded:
            _layer_arg = ", int start_layer, int end_layer"
            _off_args = ''.join([arg.c_offset_init() for arg in self._args
                                 if arg._uses_itspace or arg._is_vec_map])
            _map_decl += ';\n'.join([arg.c_map_decl(is_facet=is_facet)
                                     for arg in self._args if arg._uses_itspace])
            _map_init += ';\n'.join([arg.ast_map_init(is_top=is_top, layers=self._itspace.layers, is_facet=is_facet)
                                     for arg in self._args if arg._uses_itspace])
            for arg in self._args:
                if arg._is_mat:
                    _map_bcs_m += arg.ast_map_bcs("-")
            for arg in self._args:
                if arg._is_mat:
                    _map_bcs_p += arg.ast_map_bcs("+")
            for arg in self._args:
                if arg._uses_itspace:
                    _apply_offset += arg.ast_add_offset_map(is_facet=is_facet)
            for arg in self._args:
                if arg._is_vec_map:
                    _apply_offset += arg.ast_add_offset(is_facet=is_facet)
        else:
            _off_args = ""

        # Build kernel invocation. Let X be a parameter of the kernel representing a
        # tensor accessed in an iteration space. Let BUFFER be an array of the same
        # size as X.  BUFFER is declared and intialized in the wrapper function.
        # In particular, if:
        # - X is written or incremented, then BUFFER is initialized to 0
        # - X is read, then BUFFER gathers data expected by X
        _buf_name, _buf_decl, _buf_gather, _tmp_decl, _tmp_name = {}, {}, {}, {}, {}
        for count, arg in enumerate(self._args):
            if not arg._uses_itspace:
                continue
            _buf_name[arg] = "buffer_%s" % arg.c_arg_name(count)
            _tmp_name[arg] = "tmp_%s" % _buf_name[arg]
            _buf_size = list(self._itspace._extents)
            if not arg._is_mat:
                # Readjust size to take into account the size of a vector space
                _dat_size = (arg.data.cdim, )
                # Only adjust size if not flattening (in which case the buffer is extents*dat.dim)
                if not arg._flatten:
                    _buf_size = [sum([e*d for e, d in zip(_buf_size, _dat_size)])]
                    _loop_size = [_buf_size[i]/_dat_size[i] for i in range(len(_buf_size))]
                else:
                    _buf_size = [sum(_buf_size)]
                    _loop_size = _buf_size
            else:
                if self._kernel._applied_blas:
                    _buf_size = [reduce(lambda x, y: x*y, _buf_size)]
            # Lists of ASTs
            _buf_decl[arg] = arg.ast_buffer_decl(_buf_size, count, _buf_name[arg], is_facet=is_facet)
            _tmp_decl[arg] = arg.ast_buffer_decl(_buf_size, count, _tmp_name[arg], is_facet=is_facet, init=False)
            if arg.access not in [WRITE, INC]:
                # List of buffer gather ASTs
                ast_node = arg.ast_buffer_gather(_buf_size, count, _buf_name[arg])
                for n, e in list(reversed(list(enumerate(_loop_size)))):
                    ast_node = ast_itspace_loop(n, e, ast_node)
                _buf_gather[arg] = ast_node
        _kernel_args_str = [arg.ast_kernel_arg(count) if not arg._uses_itspace else ast.Symbol(_buf_name[arg])
                            for count, arg in enumerate(self._args)]
        _kernel_args = [arg.ast_kernel_arg(count) if not arg._uses_itspace else ast.Symbol(_buf_name[arg])
                        for count, arg in enumerate(self._args)]
        # List of AST nodes
        _buf_gather = _buf_gather.values()
        # List of AST nodes
        _buf_decl = _buf_decl.values()

        def ast_itset_loop_body(i, j, shape, offsets, is_facet=False):
            mult = 2 if is_facet else 1

            _buf_decl_scatter, _buf_scatter = {}, {}
            for count, arg in enumerate(self._args):
                if not (arg._uses_itspace and arg.access in [WRITE, INC]):
                    continue
                _buf_scatter_name = ""
                if arg._is_mat and arg._is_mixed:
                    _buf_scatter_name = "scatter_buffer_%s" % arg.c_arg_name(i, j)
                    _buf_decl_scatter[arg] = ast.Decl(arg.data.ctype, ast.Symbol(_buf_scatter_name, rank=shape))
                    _buf_scatter[arg] = arg.ast_buffer_scatter_mm(i, j, offsets, _buf_name[arg], _buf_scatter_name)
                elif not arg._is_mat:
                    _buf_scatter[arg] = arg.ast_buffer_scatter_vec(count, i, j, offsets, _buf_name[arg])
            _buf_decl_scatter = [a for a in _buf_decl_scatter.values()]
            _buf_scatter_list = []
            for a in _buf_scatter.values():
                if isinstance(a, list):
                    _buf_scatter_list += a
                else:
                    _buf_scatter_list += [a]
            _buf_scatter = _buf_scatter_list
            _addtos_extruded = []
            _addtos = []
            if self._itspace._extruded:
                for arg in self._args:
                    if arg._is_mat:
                        _addtos_extruded += arg.ast_addto(i, j, _buf_name[arg],
                                                          _tmp_name[arg],
                                                          _tmp_decl[arg],
                                                          "xtr_", is_facet=is_facet,
                                                          applied_blas=self._kernel._applied_blas)
            else:
                for count, arg in enumerate(self._args):
                    if arg._is_mat:
                        _addtos += arg.ast_addto(i, j, _buf_name[arg],
                                                 _tmp_name[arg],
                                                 _tmp_decl[arg],
                                                 applied_blas=self._kernel._applied_blas)
            # Concatenate the two lists
            inner_ast = _buf_scatter
            # If the list is not empty
            if inner_ast:
                for n, e in list(reversed(list(enumerate(shape)))):
                    inner_ast = ast_itspace_loop(n, e*mult, inner_ast)
                inner_ast = [inner_ast]
            else:
                inner_ast = []

            inner_ast = _buf_decl_scatter + inner_ast
            inner_ast += _addtos_extruded
            inner_ast += _addtos
            return inner_ast

        extr_loop_body = _map_bcs_m
        extr_loop_body += _buf_decl
        extr_loop_body += _buf_gather
        kernel_decl = None
        visitor = visit(self._kernel._ast, search=(ast.FlatBlock, ast.Symbol, ast.For))
        if self._kernel._ast and not visitor['search'][ast.FlatBlock]:
            # We only inline kernels which are represented using ASTs and do not contain FlatBlock nodes.
            # The FlatBlock nodes contain a string of C Code which we cannot parse.
            kernel_decl = get_kernel_decl_node(self._kernel._ast)
        if kernel_decl:
            change_kernel_loop_index_ids(visitor)
            for i, arg in enumerate(kernel_decl.args):
                if not isinstance(_kernel_args[i], ast.Symbol):
                    print self._kernel._ast
                    init_visitor = visit(_kernel_args[i], search=ast.Symbol)
                    if str(arg.sym.symbol) in [str(s.symbol) for s in init_visitor['search'][ast.Symbol]]:
                        # If the declared variable name appears in the initilization code then
                        # we need to give it a different name.
                        rename_var(visitor, str(arg.sym.symbol), str(arg.sym.symbol) + "_0")
                    arg.init = _kernel_args[i]
                    extr_loop_body += [arg]
                else:
                    rename_var(visitor, str(arg.sym.symbol), str(_kernel_args[i].symbol))
            extr_loop_body += kernel_decl.children
        else:
            extr_loop_body += [ast.FunCall(self._kernel.name, *_kernel_args_str)]
        for i, j, shape, offsets in self._itspace:
            extr_loop_body += ast_itset_loop_body(i, j, shape, offsets, is_facet=(self._iteration_region == ON_INTERIOR_FACETS))
        extr_loop_body += _map_bcs_p
        extr_loop_body += _apply_offset

        extr_loop = ast.Block(extr_loop_body)
        if self._itspace._extruded:
            extr_loop = ast_extrusion_loop(extr_loop_body)
        return {'wrapper_name': self._wrapper_name,
                'ssinds_arg': _ssinds_arg,
                'index_expr': _index_expr,
                'wrapper_args': _wrapper_args,
                'user_code': self._kernel._user_code,
                'wrapper_decs': indent(_wrapper_decs, 1),
                'const_args': _const_args,
                'const_inits': indent(_const_inits, 1),
                'vec_inits': indent(_vec_inits, 2),
                'off_args': _off_args,
                'layer_arg': _layer_arg,
                'map_decl': indent(_map_decl, 2),
                'vec_decs': indent(_vec_decs, 2),
                'map_init': indent(_map_init, 5),
                'ast_extr_loop': extr_loop.gencode(),
                'interm_globals_decl': indent(_intermediate_globals_decl, 3),
                'interm_globals_init': indent(_intermediate_globals_init, 3),
                'interm_globals_writeback': indent(_intermediate_globals_writeback, 3)}
