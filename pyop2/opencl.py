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

"""OP2 OpenCL backend."""

import runtime_base as op2
from utils import verify_reshape, uniquify
from runtime_base import IdentityMap, READ, WRITE, RW, INC, MIN, MAX, Set
import configuration as cfg
import op_lib_core as core
import pyopencl as cl
import pkg_resources
import pycparser
import numpy as np
import collections
import warnings
import math
from jinja2 import Environment, PackageLoader
from pycparser import c_parser, c_ast, c_generator
import re
import time
import md5

class Kernel(op2.Kernel):
    """OP2 OpenCL kernel type."""

    def __init__(self, code, name):
        op2.Kernel.__init__(self, code, name)

    class Instrument(c_ast.NodeVisitor):
        """C AST visitor for instrumenting user kernels.
             - adds memory space attribute to user kernel declaration
             - appends constant declaration to user kernel param list
             - adds a separate function declaration for user kernel
        """
        def instrument(self, ast, kernel_name, instrument, constants):
            self._kernel_name = kernel_name
            self._instrument = instrument
            self._ast = ast
            self._constants = constants
            self.generic_visit(ast)
            idx = ast.ext.index(self._func_node)
            ast.ext.insert(0, self._func_node.decl)

        def visit_FuncDef(self, node):
            if node.decl.name == self._kernel_name:
                self._func_node = node
                self.visit(node.decl)

        def visit_ParamList(self, node):
            for i, p in enumerate(node.params):
                if self._instrument[i][0]:
                    p.storage.append(self._instrument[i][0])
                if self._instrument[i][1]:
                    p.type.quals.append(self._instrument[i][1])

            for cst in self._constants:
                if cst._is_scalar:
                    t = c_ast.TypeDecl(cst._name, [], c_ast.IdentifierType([cst._cl_type]))
                else:
                    t = c_ast.PtrDecl([], c_ast.TypeDecl(cst._name, ["__constant"], c_ast.IdentifierType([cst._cl_type])))
                decl = c_ast.Decl(cst._name, [], [], [], t, None, 0)
                node.params.append(decl)

    def instrument(self, instrument, constants):
        def comment_remover(text):
            """Remove all C- and C++-style comments from a string."""
            # Reference: http://stackoverflow.com/questions/241327/python-snippet-to-remove-c-and-c-comments
            def replacer(match):
                s = match.group(0)
                if s.startswith('/'):
                    return ""
                else:
                    return s
            pattern = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
                                 re.DOTALL | re.MULTILINE)
            return re.sub(pattern, replacer, text)

        ast = c_parser.CParser().parse(comment_remover(self._code).replace("\\\n", "\n"))
        Kernel.Instrument().instrument(ast, self._name, instrument, constants)
        return c_generator.CGenerator().visit(ast)

    @property
    def md5(self):
        return md5.new(self._name + self._code).digest()


class Arg(op2.Arg):
    """OP2 OpenCL argument type."""

    # Codegen specific
    @property
    def _d_is_staged(self):
        return self._is_direct and not self.data._is_scalar

    @property
    def _i_gen_vec(self):
        assert self._is_vec_map or self._uses_itspace
        return map(lambda i: Arg(self.data, self.map, i, self.access), range(self.map.dim))

class DeviceDataMixin(object):
    """Codegen mixin for datatype and literal translation."""

    ClTypeInfo = collections.namedtuple('ClTypeInfo', ['clstring', 'zero'])
    CL_TYPES = {np.dtype('uint8'): ClTypeInfo('uchar', '0'),
                np.dtype('int8'): ClTypeInfo('char', '0'),
                np.dtype('uint16'): ClTypeInfo('ushort', '0'),
                np.dtype('int16'): ClTypeInfo('short', '0'),
                np.dtype('uint32'): ClTypeInfo('uint', '0u'),
                np.dtype('int32'): ClTypeInfo('int', '0'),
                np.dtype('uint64'): ClTypeInfo('ulong', '0ul'),
                np.dtype('int64'): ClTypeInfo('long', '0l'),
                np.dtype('float32'): ClTypeInfo('float', '0.0f'),
                np.dtype('float64'): ClTypeInfo('double', '0.0')}

    @property
    def bytes_per_elem(self):
        return self.dtype.itemsize * self.cdim

    @property
    def _is_scalar(self):
        return self.cdim == 1

    @property
    def _cl_type(self):
        return DeviceDataMixin.CL_TYPES[self.dtype].clstring

    @property
    def _cl_type_zero(self):
        return DeviceDataMixin.CL_TYPES[self.dtype].zero

    @property
    def _dirty(self):
        if not hasattr(self, '_ddm_dirty'):
            self._ddm_dirty = False
        return self._ddm_dirty

    @_dirty.setter
    def _dirty(self, value):
        self._ddm_dirty = value


def one_time(func):
    # decorator, memoize and return method first call result
    def wrap(self):
        try:
            value = self._memoize[func.__name__]
        except (KeyError, AttributeError):
            value = func(self)
            try:
                cache = self._memoize
            except AttributeError:
                cache = self._memoize = dict()
            cache[func.__name__] = value
        return value

    wrap.__name__ = func.__name__
    wrap.__doc__ = func.__doc__
    return wrap

class Dat(op2.Dat, DeviceDataMixin):
    """OP2 OpenCL vector data type."""

    _arg_type = Arg

    @property
    @one_time
    def _buffer(self):
        _buf = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._data.nbytes)
        if len(self._data) is not 0:
            cl.enqueue_copy(_queue, _buf, self._data, is_blocking=True).wait()
        return _buf

    @property
    def data(self):
        if len(self._data) is 0:
            raise RuntimeError("Temporary dat has no data on the host")

        if self._dirty:
            cl.enqueue_copy(_queue, self._data, self._buffer, is_blocking=True).wait()
            if self.soa:
                np.transpose(self._data)
            self._dirty = False
        return self._data

    def _upload_from_c_layer(self):
        cl.enqueue_copy(_queue, self._buffer, self._data, is_blocking=True).wait()

def solve(M, b, x):
    x.data
    b.data
    core.solve(M, b, x)
    x._upload_from_c_layer()
    b._upload_from_c_layer()

class Mat(op2.Mat, DeviceDataMixin):
    """OP2 OpenCL matrix data type."""

    _arg_type = Arg

    @property
    @one_time
    def _dev_array(self):
        s = self.dtype.itemsize * self._sparsity._c_handle.total_nz
        _buf = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=s)
        return _buf

    @property
    @one_time
    def _dev_colidx(self):
        _buf = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._sparsity._c_handle.colidx.nbytes)
        cl.enqueue_copy(_queue, _buf, self._sparsity._c_handle.colidx, is_blocking=True).wait()
        return _buf

    @property
    @one_time
    def _dev_rowptr(self):
        _buf = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._sparsity._c_handle.rowptr.nbytes)
        cl.enqueue_copy(_queue, _buf, self._sparsity._c_handle.rowptr, is_blocking=True).wait()
        return _buf

    def _upload_array(self):
        cl.enqueue_copy(_queue, self._dev_array, self._c_handle.array, is_blocking=True).wait()
        self._dirty = False

    def assemble(self):
        if self._dirty:
            cl.enqueue_copy(_queue, self._c_handle.array, self._dev_array, is_blocking=True).wait()
            self._c_handle.restore_array()
            self._dirty = False
        self._c_handle.assemble()

    @property
    def cdim(self):
        return np.prod(self.dims)


class Const(op2.Const, DeviceDataMixin):
    """OP2 OpenCL data that is constant for any element of any set."""

    @property
    @one_time
    def _buffer(self):
        _buf = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._data.nbytes)
        cl.enqueue_copy(_queue, _buf, self._data, is_blocking=True).wait()
        return _buf

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        cl.enqueue_copy(_queue, self._buffer, self._data, is_blocking=True).wait()


class Global(op2.Global, DeviceDataMixin):
    """OP2 OpenCL global value."""

    _arg_type = Arg

    @property
    @one_time
    def _buffer(self):
        _buf = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._data.nbytes)
        cl.enqueue_copy(_queue, _buf, self._data, is_blocking=True).wait()
        return _buf

    def _allocate_reduction_array(self, nelems):
        self._h_reduc_array = np.zeros (nelems * self.cdim, dtype=self.dtype)
        self._d_reduc_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._h_reduc_array.nbytes)
        cl.enqueue_copy(_queue, self._d_reduc_buffer, self._h_reduc_array, is_blocking=True).wait()

    @property
    def data(self):
        if self._dirty:
            cl.enqueue_copy(_queue, self._data, self._buffer, is_blocking=True).wait()
            self._dirty = False
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        cl.enqueue_copy(_queue, self._buffer, self._data, is_blocking=True).wait()
        self._dirty = False

    def _post_kernel_reduction_task(self, nelems, reduction_operator):
        assert reduction_operator in [INC, MIN, MAX]

        def generate_code():
            def headers():
                if self.dtype == np.dtype('float64'):
                    return """
#if defined(cl_khr_fp64)
#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

"""
                else:
                    return ""

            def op():
                if reduction_operator is INC:
                    return "INC"
                elif reduction_operator is MIN:
                    return "min"
                elif reduction_operator is MAX:
                        return "max"
                assert False

            return """
%(headers)s
#define INC(a,b) ((a)+(b))
__kernel
void %(name)s_reduction (
  __global %(type)s* dat,
  __global %(type)s* tmp,
  __private int count
)
{
  __private %(type)s accumulator[%(dim)d];
  for (int j = 0; j < %(dim)d; ++j)
  {
    accumulator[j] = dat[j];
  }
  for (int i = 0; i < count; ++i)
  {
    for (int j = 0; j < %(dim)d; ++j)
    {
      accumulator[j] = %(op)s(accumulator[j], *(tmp + i * %(dim)d + j));
    }
  }
  for (int j = 0; j < %(dim)d; ++j)
  {
    dat[j] = accumulator[j];
  }
}
""" % {'headers': headers(), 'name': self._name, 'dim': self.cdim, 'type': self._cl_type, 'op': op()}


        if not _reduction_task_cache.has_key((self.dtype, self.cdim, reduction_operator)):
            _reduction_task_cache[(self.dtype, self.cdim, reduction_operator)] = generate_code()

        src = _reduction_task_cache[(self.dtype, self.cdim, reduction_operator)]
        prg = cl.Program(_ctx, src).build(options="-Werror")
        kernel = prg.__getattr__(self._name + '_reduction')
        kernel.append_arg(self._buffer)
        kernel.append_arg(self._d_reduc_buffer)
        kernel.append_arg(np.int32(nelems))
        cl.enqueue_task(_queue, kernel).wait()

        del self._d_reduc_buffer

class Map(op2.Map):
    """OP2 OpenCL map, a relation between two Sets."""

    _arg_type = Arg

    @property
    @one_time
    def _buffer(self):
        assert self._iterset.size != 0, 'cannot upload IdentityMap'
        _buf = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._values.nbytes)
        cl.enqueue_copy(_queue, _buf, self._values, is_blocking=True).wait()
        return _buf

    @property
    @one_time
    def md5(self):
        return md5.new(self._values).digest()

class OpPlanCache():
    """Cache for OpPlan."""

    def __init__(self):
        self._cache = dict()

    def get_plan(self, parloop, **kargs):
        try:
            plan = self._cache[parloop._plan_key]
        except KeyError:
            cp = core.op_plan(parloop._kernel1, parloop._it_set, *parloop._args, **kargs)
            plan = OpPlan(parloop, cp)
            self._cache[parloop._plan_key] = plan

        return plan

    @property
    def nentries(self):
        return len(self._cache)

class OpPlan():
    """ Helper proxy for core.op_plan."""

    def __init__(self, parloop, core_plan):
        self._parloop = parloop
        self._core_plan = core_plan

        self.load()

    def load(self):
        self.nuinds = sum(map(lambda a: a._is_indirect, self._parloop._unique_args))
        _ind_desc = [-1] * len(self._parloop._unique_args)
        _d = {}
        _c = 0
        for i, arg in enumerate(self._parloop._unique_args):
            if arg._is_indirect:
                if _d.has_key((arg.data, arg.map)):
                    _ind_desc[i] = _d[(arg.data, arg.map)]
                else:
                    _ind_desc[i] = _c
                    _d[(arg.data, arg.map)] = _c
                    _c += 1
        del _c
        del _d

        _off = [0] * (self._core_plan.ninds + 1)
        for i in range(self._core_plan.ninds):
            _c = 0
            for idesc in _ind_desc:
                if idesc == i:
                    _c += 1
            _off[i+1] = _off[i] + _c

        self._ind_map_buffers = [None] * self._core_plan.ninds
        for i in range(self._core_plan.ninds):
            self._ind_map_buffers[i] = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=int(np.int32(0).itemsize * (_off[i+1] - _off[i]) * self._parloop._it_set.size))
            s = self._parloop._it_set.size * _off[i]
            e = s + (_off[i+1] - _off[i]) * self._parloop._it_set.size
            cl.enqueue_copy(_queue, self._ind_map_buffers[i], self._core_plan.ind_map[s:e], is_blocking=True).wait()

        self._loc_map_buffers = [None] * self.nuinds
        for i in range(self.nuinds):
            self._loc_map_buffers[i] = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=int(np.int16(0).itemsize * self._parloop._it_set.size))
            s = i * self._parloop._it_set.size
            e = s + self._parloop._it_set.size
            cl.enqueue_copy(_queue, self._loc_map_buffers[i], self._core_plan.loc_map[s:e], is_blocking=True).wait()

        self._ind_sizes_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.ind_sizes.nbytes)
        cl.enqueue_copy(_queue, self._ind_sizes_buffer, self._core_plan.ind_sizes, is_blocking=True).wait()

        self._ind_offs_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.ind_offs.nbytes)
        cl.enqueue_copy(_queue, self._ind_offs_buffer, self._core_plan.ind_offs, is_blocking=True).wait()

        self._blkmap_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.blkmap.nbytes)
        cl.enqueue_copy(_queue, self._blkmap_buffer, self._core_plan.blkmap, is_blocking=True).wait()

        self._offset_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.offset.nbytes)
        cl.enqueue_copy(_queue, self._offset_buffer, self._core_plan.offset, is_blocking=True).wait()

        self._nelems_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.nelems.nbytes)
        cl.enqueue_copy(_queue, self._nelems_buffer, self._core_plan.nelems, is_blocking=True).wait()

        self._nthrcol_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.nthrcol.nbytes)
        cl.enqueue_copy(_queue, self._nthrcol_buffer, self._core_plan.nthrcol, is_blocking=True).wait()

        self._thrcol_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.thrcol.nbytes)
        cl.enqueue_copy(_queue, self._thrcol_buffer, self._core_plan.thrcol, is_blocking=True).wait()

        if self._parloop._kernel2:
            if _debug:
                start = time.clock()
            self._loop2, flags = self._parloop.generate_flags(self)
            self._flags_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=flags.nbytes)
            cl.enqueue_copy(_queue, self._flags_buffer, flags, is_blocking=True).wait()
            if _debug:
                end = time.clock()
                print("Took %0.03f" % (end - start))
        else:
            self._loop2 = None
            self._flags_buffer = None

        if _debug:
            print 'plan ind_map ' + str(self._core_plan.ind_map)
            print 'plan loc_map ' + str(self._core_plan.loc_map)
            print '_ind_desc ' + str(_ind_desc)
            print 'nuinds %d' % self.nuinds
            print 'ninds %d' % self.ninds
            print '_off ' + str(_off)
            for i in range(self.ninds):
                print 'ind_map[' + str(i) + '] = ' + str(self.ind_map[s:e])
            for i in range(self.nuinds):
                print 'loc_map[' + str(i) + '] = ' + str(self.loc_map[s:e])
            print 'ind_sizes :' + str(self.ind_sizes)
            print 'ind_offs :' + str(self.ind_offs)
            print 'blk_map :' + str(self.blkmap)
            print 'offset :' + str(self.offset)
            print 'nelems :' + str(self.nelems)
            print 'nthrcol :' + str(self.nthrcol)
            print 'thrcol :' + str(self.thrcol)

    @property
    def nshared(self):
        return self._core_plan.nshared

    @property
    def ninds(self):
        return self._core_plan.ninds

    @property
    def ncolors(self):
        return self._core_plan.ncolors

    @property
    def ncolblk(self):
        return self._core_plan.ncolblk

    @property
    def nblocks(self):
        return self._core_plan.nblocks

class DatMapPair(object):
    """ Dummy class needed for codegen
        (could do without but would obfuscate codegen templates)
    """
    def __init__(self, data, map):
        self.data = data
        self.map = map

    def __hash__(self):
        return hash(self.data) ^ hash(self.map)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class ParLoopCall(object):
    """Invocation of an OP2 OpenCL kernel with an access descriptor"""

    def __init__(self, kernel, it_space, *args):
        self._kernel1 = kernel
        self._kernel2 = None

        if isinstance(it_space, op2.IterationSpace):
            self._it_set = it_space._iterset
            self._it_space = it_space
        else:
            self._it_set = it_space
            self._it_space = False

        self._actual_args1 = list(args)
        self._actual_args2 = []

        self._args1 = self.gen_args(self._actual_args1)
        self._args2 = []

        self._actual_args = list(self._actual_args1)
        self._args = list(self._args1)
        self.sort_args()

    def add_kernel(self, kernel, *args):
        self._kernel2 = kernel

        self._actual_args2 = list(args)
        self._args2 = self.gen_args(self._actual_args2)

        self._actual_args = self._actual_args1 + self._actual_args2
        self._args = self._args1 + self._args2
        self.sort_args()

    def gen_args(self, actual_args):
        args = list()
        for a in actual_args:
            if a._is_vec_map:
                for i in range(a.map._dim):
                    args.append(Arg(a.data, a.map, i, a.access))
            elif a._is_mat:
                pass
            elif a._uses_itspace:
                for i in range(self._it_space.extents[a.idx.index]):
                    args.append(Arg(a.data, a.map, i, a.access))
            else:
                args.append(a)
        return args

    def sort_args(self):
        # sort args - keep actual args unchanged
        # order globals r, globals reduc, direct, indirect
        gbls = self._global_non_reduction_args +\
               sorted(self._global_reduction_args,
                      key=lambda arg: (arg.data.dtype.itemsize,arg.data.cdim))
        directs = self._direct_args
        indirects = sorted(self._indirect_args,
                           key=lambda arg: (arg.map.md5, id(arg.data), arg.idx))

        self._args = gbls + directs + indirects

    @property
    def _plan_key(self):
        """Canonical representation of a parloop wrt plan caching."""

        # Globals: irrelevant, they only possibly effect the partition
        # size for reductions.
        # Direct Dats: irrelevant, no staging
        # iteration size: effect ind/loc maps sizes
        # partition size: effect interpretation of ind/loc maps

        # ind: for each dat map pair, the ind and loc map depend on the dim of
        #   the map, and the actual indices referenced
        inds = list()
        for dm in self._dat_map_pairs:
            d = dm.data
            m = dm.map
            indices = tuple(a.idx for a in self._args if a.data == d and a.map == m)

            inds.append((m.md5, m._dim, indices))

        # coloring part of the key,
        # for each dat, includes (map, (idx, ...)) involved (INC)
        # dats do not matter here, but conflicts should be sorted
        cols = list()
        for i, d in enumerate(sorted((dm.data for dm in self._dat_map_pairs),
                                     key=id)):
            conflicts = list()
            has_conflict = False
            for m in uniquify(a.map for a in self._args if a.data == d and a._is_indirect):
                idx = sorted(arg.idx for arg in self._indirect_reduc_args \
                             if arg.data == d and arg.map == m)
                if len(idx) > 0:
                    has_conflict = True
                    conflicts.append((m.md5, tuple(idx)))
            if has_conflict:
                cols.append(tuple(conflicts))

        return (self._it_set.size,
                self._i_partition_size(),
                tuple(inds),
                tuple(cols))

    @property
    def _gencode_key(self):
        """Canonical representation of a parloop wrt generated code caching."""

        # user kernel: md5 of kernel name and code (same code can contain
        #   multiple user kernels)
        # iteration space description
        # for each actual arg:
        #   its type (dat | gbl | mat)
        #   dtype (required for casts and opencl extensions)
        #   dat.dim (dloops: if staged or reduc; indloops; if not direct dat)
        #   access  (dloops: if staged or reduc; indloops; if not direct dat)
        #   the ind map index: gbl = -1, direct = -1, indirect = X (first occurence
        #     of the dat/map pair) (will tell which arg use which ind/loc maps)
        #     vecmap = -X (size of the map)
        # for vec map arg we need the dimension of the map
        # consts in alphabetial order: name, dtype (used in user kernel,
        #   is_scalar (passed as pointed or value)

        def argdimacc(arg):
            if self.is_direct():
                if arg._is_global or (arg._is_dat and not arg.data._is_scalar):
                    return (arg.data.cdim, arg.access)
                else:
                    return ()
            else:
                if (arg._is_global and arg.access is READ) or arg._is_direct:
                    return ()
                else:
                    return (arg.data.cdim, arg.access)

        argdesc = []
        seen = dict()
        c = 0
        for arg in self._actual_args:
            if arg._is_indirect:
                if not seen.has_key((arg.data,arg.map)):
                    seen[(arg.data,arg.map)] = c
                    idesc = (c, (- arg.map.dim) if arg._is_vec_map else arg.idx)
                    c += 1
                else:
                    idesc = (seen[(arg.data,arg.map)], (- arg.map.dim) if arg._is_vec_map else arg.idx)
            else:
                idesc = ()

            d = (arg.data.__class__,
                 arg.data.dtype) + argdimacc(arg) + idesc

            argdesc.append(d)

        consts = map(lambda c: (c.name, c.dtype, c.cdim == 1),
                     sorted(list(Const._defs), key=lambda c: c.name))

        itspace = (self._it_space.extents,) if self._it_space else ((None,))

        dig2 = self._kernel2.md5 if self._kernel2 else None
        return (self._kernel1.md5, dig2) + itspace + tuple(argdesc) + tuple(consts)

    # generic
    @property
    def _global_reduction_args(self):
        return uniquify(a for a in self._args if a._is_global_reduction)

    @property
    def _global_non_reduction_args(self):
        return uniquify(a for a in self._args if a._is_global and not a._is_global_reduction)

    @property
    def _unique_args(self):
        return uniquify(self._args)

    @property
    def _unique_dats(self):
        return uniquify(a.data for a in self._args if a._is_dat)

    @property
    def _indirect_reduc_args(self):
        return uniquify(a for a in self._args if a._is_indirect_reduction)

    @property
    def _indirect_reduc_args1(self):
        return uniquify(a for a in self._args1 if a._is_indirect_reduction)

    @property
    def _indirect_reduc_args2(self):
        return uniquify(a for a in self._args2 if a._is_indirect_reduction)

    @property
    def _direct_args(self):
        return uniquify(a for a in self._args if a._is_direct)

    @property
    def _direct_non_scalar_args(self):
        return [a for a in self._direct_args if not a.data._is_scalar]

    @property
    def _direct_non_scalar_read_args(self):
        return [a for a in self._direct_non_scalar_args if a.access in [READ, RW]]

    @property
    def _direct_non_scalar_written_args(self):
        return [a for a in self._direct_non_scalar_args if a.access in [WRITE, RW]]

    @property
    def _matrix_args(self):
        return [a for a in self._actual_args if a._is_mat]

    @property
    def _matrix_args1(self):
        return [a for a in self._actual_args1 if a._is_mat]

    @property
    def _matrix_args2(self):
        return [a for a in self._actual_args2 if a._is_mat]

    @property
    def _itspace_args(self):
        return [a for a in self._actual_args if a._uses_itspace and not a._is_mat]

    @property
    def _unique_matrix(self):
        return uniquify(a.data for a in self._matrix_args)

    @property
    def _matrix_entry_maps(self):
        """Set of all mappings used in matrix arguments."""
        return uniquify(m for arg in self._actual_args  if arg._is_mat for m in arg.map)

    @property
    def _indirect_args(self):
        return [a for a in self._args if a._is_indirect]

    @property
    def _indirect_args1(self):
        return [a for a in self._args1 if a._is_indirect]

    @property
    def _indirect_args2(self):
        return [a for a in self._args2 if a._is_indirect]

    @property
    def _vec_map_args(self):
        return [a for a in self._actual_args if a._is_vec_map]

    @property
    def _dat_map_pairs(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._indirect_args)

    @property
    def _nonreduc_vec_dat_map_pairs(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._vec_map_args if a.access is not INC)

    @property
    def _reduc_vec_dat_map_pairs(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._vec_map_args if a.access is INC)

    @property
    def _nonreduc_itspace_dat_map_pairs(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._itspace_args if a.access is not INC)

    @property
    def _reduc_itspace_dat_map_pairs(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._itspace_args if a.access is INC)

    @property
    def _read_dat_map_pairs(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._indirect_args if a.access in [READ, RW])

    @property
    def _read_dat_map_pairs1(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._indirect_args1 if a.access in [READ, RW])

    @property
    def _read_dat_map_pairs2(self):
        if self._kernel2:
            return uniquify(DatMapPair(a.data, a.map) for a in self._indirect_args2 if a not in self._indirect_args1 and a.access in [READ, RW])
        return []

    @property
    def _written_dat_map_pairs(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._indirect_args if a.access in [WRITE, RW])

    @property
    def _written_dat_map_pairs1(self):
        pairs = uniquify(a for a in self._indirect_args1 if a.access in [WRITE, RW])

        newpairs = []
        for pair in pairs:
            write = True
            for aux in self._indirect_args2:
                if pair == aux and aux.access == READ:
                    write = False
                    break
            if write:
                newpairs.append(DatMapPair(pair.data, pair.map))
        return newpairs

    @property
    def _written_dat_map_pairs2(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._indirect_args2 if a.access in [WRITE, RW])

    @property
    def _indirect_reduc_dat_map_pairs(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._args if a._is_indirect_reduction)

    @property
    def _indirect_reduc_dat_map_pairs1(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._args1 if a._is_indirect_reduction)

    @property
    def _indirect_reduc_dat_map_pairs2(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._args2 if a._is_indirect_reduction)

    def dump_gen_code(self, src):
        if cfg['dump-gencode']:
            path = cfg['dump-gencode-path'] % {"kernel": self._kernel1._name,
                                               "time": time.strftime('%Y-%m-%d@%H:%M:%S')}

            if not os.path.exists(path):
                with open(path, "w") as f:
                    f.write(src)

    def _d_max_local_memory_required_per_elem(self):
        """Computes the maximum shared memory requirement per iteration set elements."""
        def max_0(iterable):
            return max(iterable) if iterable else 0
        staging = max_0([a.data.bytes_per_elem for a in self._direct_non_scalar_args])
        reduction = max_0([a.data.dtype.itemsize for a in self._global_reduction_args])
        return max(staging, reduction)

    def _i_partition_size(self):
        #TODO FIX: something weird here
        #available_local_memory
        warnings.warn('temporary fix to available local memory computation (-512)')
        available_local_memory = _max_local_memory - 512
        # 16bytes local mem used for global / local indices and sizes
        available_local_memory -= 16
        # (4/8)ptr size per dat passed as argument (dat)
        available_local_memory -= (_address_bits / 8) * (len(self._unique_dats) + len(self._global_non_reduction_args))
        # (4/8)ptr size per dat/map pair passed as argument (ind_map)
        available_local_memory -= (_address_bits / 8) * len(self._dat_map_pairs)
        # (4/8)ptr size per global reduction temp array
        available_local_memory -= (_address_bits / 8) * len(self._global_reduction_args)
        # (4/8)ptr size per indirect arg (loc_map)
        available_local_memory -= (_address_bits / 8) * len(filter(lambda a: not a._is_indirect, self._args))
        # (4/8)ptr size * 7: for plan objects
        available_local_memory -= (_address_bits / 8) * 7
        # 1 uint value for block offset
        available_local_memory -= 4
        # 7: 7bytes potentialy lost for aligning the shared memory buffer to 'long'
        available_local_memory -= 7
        # 12: shared_memory_offset, active_thread_count, active_thread_count_ceiling variables (could be 8 or 12 depending)
        #     and 3 for potential padding after shared mem buffer
        available_local_memory -= 12 + 3
        # 2 * (4/8)ptr size + 1uint32: DAT_via_MAP_indirection(./_size/_map) per dat map pairs
        available_local_memory -= 4 + (_address_bits / 8) * 2 * len(self._dat_map_pairs)
        # inside shared memory padding
        available_local_memory -= 2 * (len(self._dat_map_pairs) - 1)

        max_bytes = sum(map(lambda a: a.data.bytes_per_elem, self._indirect_args))
        return available_local_memory / (2 * _warpsize * max_bytes) * (2 * _warpsize)

    def launch_configuration(self):
        if self.is_direct():
            per_elem_max_local_mem_req = self._d_max_local_memory_required_per_elem()
            shared_memory_offset = per_elem_max_local_mem_req * _warpsize
            if per_elem_max_local_mem_req == 0:
                wgs = _max_work_group_size
            else:
                # 16bytes local mem used for global / local indices and sizes
                # (4/8)ptr bytes for each dat buffer passed to the kernel
                # (4/8)ptr bytes for each temporary global reduction buffer passed to the kernel
                # 7: 7bytes potentialy lost for aligning the shared memory buffer to 'long'
                warnings.warn('temporary fix to available local memory computation (-512)')
                available_local_memory = _max_local_memory - 512
                available_local_memory -= 16
                available_local_memory -= (len(self._unique_dats) + len(self._global_non_reduction_args))\
                                          * (_address_bits / 8)
                available_local_memory -= len(self._global_reduction_args) * (_address_bits / 8)
                available_local_memory -= 7
                ps = available_local_memory / per_elem_max_local_mem_req
                wgs = min(_max_work_group_size, (ps / _warpsize) * _warpsize)
            nwg = min(_pref_work_group_count, int(math.ceil(self._it_set.size / float(wgs))))
            ttc = wgs * nwg

            local_memory_req = per_elem_max_local_mem_req * wgs
            return {'thread_count': ttc,
                    'work_group_size': wgs,
                    'work_group_count': nwg,
                    'local_memory_size': local_memory_req,
                    'local_memory_offset': shared_memory_offset}
        else:
            return {'partition_size': self._i_partition_size()}

    def codegen(self, conf):
        def instrument_user_kernel(kernel, actual_args):
            if not kernel:
                return ""

            inst = []

            for arg in actual_args:
                i = None
                if self.is_direct():
                    if (arg._is_direct and arg.data._is_scalar) or\
                       (arg._is_global and not arg._is_global_reduction):
                        i = ("__global", None)
                    else:
                        i = ("__private", None)
                else: # indirect loop
                    if arg._is_direct or (arg._is_global and not arg._is_global_reduction):
                        i = ("__global", None)
                    elif (arg._is_indirect or arg._is_vec_map) and not arg._is_indirect_reduction:
                        i = ("__local", None)
                    else:
                        i = ("__private", None)

                inst.append(i)

            if self._it_space:
                for i in self._it_space.extents:
                    inst.append(("__private", None))

            return kernel.instrument(inst, sorted(list(Const._defs), key=lambda c: c._name))

        # check cache
        if _kernel_stub_cache.has_key(self._gencode_key):
            return _kernel_stub_cache[self._gencode_key]

        #do codegen
        user_kernel1 = instrument_user_kernel(self._kernel1, self._actual_args1)
        user_kernel2 = instrument_user_kernel(self._kernel2, self._actual_args2)
        template = _jinja2_direct_loop if self.is_direct()\
                                       else _jinja2_indirect_loop

        src = template.render({'parloop': self,
                               'user_kernel1': user_kernel1,
                               'user_kernel2': user_kernel2,
                               'launch': conf,
                               'codegen': {'amd': _AMD_fixes},
                               'op2const': sorted(list(Const._defs),
                                                  key=lambda c: c._name)
                              }).encode("ascii")
        _kernel_stub_cache[self._gencode_key] = src
        return src

    def generate_flags(self, plan):
        def getpartno(n):
            return n // plan._core_plan.part_size

        dats = {}
        flags = np.ones(self._it_set.size, dtype=np.uint32)
        newset = []

        indirect_args1 = self._indirect_args1
        indirect_args2 = self._indirect_args2

        for i, arg in enumerate(self._args):
            if arg._is_indirect:
                for arg in self._indirect_args:
                    dats[arg._dat] = [[[0, None, []] for i in range(2)] for i in range(arg._dat.dataset.size)]

                for i in range(self._it_set.size):
                    for arg in indirect_args1:
                        elem = list(dats[arg._dat][arg._map._values[i][arg._idx]][0])
                        elem[0] = max(elem[0], i)

                        if elem[1] == None:
                            elem[1] = arg._access
                        elif elem[1] == READ and arg._access != READ:
                            elem[1] = arg._access

                        dats[arg._dat][arg._map._values[i][arg._idx]][0] = elem

                    for arg in indirect_args2:
                        elem = list(dats[arg._dat][arg._map._values[i][arg._idx]][1])
                        elem[0] = max(elem[0], i)
                        elem[2].append(i)

                        if elem[1] == None:
                            elem[1] = arg._access
                        elif elem[1] == READ and arg._access != READ:
                            elem[1] = arg._access

                        dats[arg._dat][arg._map._values[i][arg._idx]][1] = elem

                for key, value in dats.iteritems():
                    for v in value:
                        l0 = v[0]
                        l1 = v[1]

                        if(l0[1] is None or l1[1] is None):
                            continue

                        if(l0[1] != READ):
                            if(getpartno(l0[0]) > getpartno(l1[0])):
                                newset += l1[2]
                                for i in l1[2]:
                                    flags[i] = False
                        elif(l1[1] != READ):
                            if(getpartno(l0[0]) > getpartno(l1[0])):
                                newset += l1[2]
                                for i in l1[2]:
                                    flags[i] = False

        if len(newset) == 0:
            return None, flags

        newset.sort()
        newset = uniquify(newset)
        op2set = op2.Set(len(newset))

        map_data = [np.empty(len(newset), dtype=i.map.values.dtype) for i in self._args2]
        for n, i in enumerate(newset):
            for c, (arg, dat) in enumerate(zip(self._args2, dats)):
                val = arg.map.values[i]
                map_data[c][n] = val

        args = [None for i in self._args2]
        for i in range(len(args)):
            args[i] = self._args2[i].data(Map(op2set, self._args2[i].map.dataset, self._args2[i].map.dim, values=map_data[i])[self._args2[i].idx], self._args2[i].access)

        loop = ParLoopCall(self._kernel2, op2set, *args)
        return loop, flags

    def compute(self):
        def compile_kernel(src, name):
            prg = cl.Program(_ctx, source).build(options="-Werror")
            return prg.__getattr__(name + '_stub')

        conf = self.launch_configuration()

        if not self.is_direct():
            plan = _plan_cache.get_plan(self, partition_size=conf['partition_size'])
            conf['local_memory_size'] = plan.nshared
            conf['ninds'] = plan.ninds
            conf['work_group_size'] = min(_max_work_group_size, conf['partition_size'])
            conf['work_group_count'] = plan.nblocks
        conf['warpsize'] = _warpsize

        source = self.codegen(conf)
        kernel = compile_kernel(source, self._kernel1._name)

        for a in self._unique_dats:
            kernel.append_arg(a._buffer)

        for a in self._global_non_reduction_args:
            kernel.append_arg(a.data._buffer)

        for a in self._global_reduction_args:
            a.data._allocate_reduction_array(conf['work_group_count'])
            kernel.append_arg(a.data._d_reduc_buffer)

        for cst in sorted(list(Const._defs), key=lambda c: c._name):
            kernel.append_arg(cst._buffer)

        if self.is_direct():
            kernel.append_arg(np.int32(self._it_set.size))

            cl.enqueue_nd_range_kernel(_queue, kernel, (int(conf['thread_count']),), (int(conf['work_group_size']),), g_times_l=False).wait()
        else:
            for i in range(plan.ninds):
                kernel.append_arg(plan._ind_map_buffers[i])

            for i in range(plan.nuinds):
                kernel.append_arg(plan._loc_map_buffers[i])

            for m in self._unique_matrix:
                kernel.append_arg(m._dev_array)
                m._upload_array()
                kernel.append_arg(m._dev_rowptr)
                kernel.append_arg(m._dev_colidx)

            for m in self._matrix_entry_maps:
                kernel.append_arg(m._buffer)

            if plan._flags_buffer:
                kernel.append_arg(plan._flags_buffer)

            kernel.append_arg(plan._ind_sizes_buffer)
            kernel.append_arg(plan._ind_offs_buffer)
            kernel.append_arg(plan._blkmap_buffer)
            kernel.append_arg(plan._offset_buffer)
            kernel.append_arg(plan._nelems_buffer)
            kernel.append_arg(plan._nthrcol_buffer)
            kernel.append_arg(plan._thrcol_buffer)

            block_offset = 0
            for i in range(plan.ncolors):
                blocks_per_grid = int(plan.ncolblk[i])
                threads_per_block = min(_max_work_group_size, conf['partition_size'])
                thread_count = threads_per_block * blocks_per_grid

                kernel.set_last_arg(np.int32(block_offset))
                cl.enqueue_nd_range_kernel(_queue, kernel, (int(thread_count),), (int(threads_per_block),), g_times_l=False).wait()
                block_offset += blocks_per_grid

        # mark !READ data as dirty
        for arg in self._actual_args:
            if arg.access not in [READ]:
                arg.data._dirty = True

        for mat in [arg.data for arg in self._matrix_args]:
            mat.assemble()

        for i, a in enumerate(self._global_reduction_args):
            a.data._post_kernel_reduction_task(conf['work_group_count'], a.access)

        if not self.is_direct() and plan._loop2:
            plan._loop2.compute()

    def is_direct(self):
        return all(map(lambda a: a._is_direct or isinstance(a.data, Global), self._args))

#Monkey patch pyopencl.Kernel for convenience
_original_clKernel = cl.Kernel

class CLKernel (_original_clKernel):
    def __init__(self, *args, **kargs):
        super(CLKernel, self).__init__(*args, **kargs)
        self._karg = 0

    def reset_args(self):
        self._karg = 0;

    def append_arg(self, arg):
        self.set_arg(self._karg, arg)
        self._karg += 1

    def set_last_arg(self, arg):
        self.set_arg(self._karg, arg)

cl.Kernel = CLKernel

def par_loop(kernel, it_space, *args):
    ParLoopCall(kernel, it_space, *args).compute()

def par_loop2(kernel1, kernel2, it_space, args1, args2):
    loop = ParLoopCall(kernel1, it_space, *args1)
    loop.add_kernel(kernel2, *args2)
    loop.compute()

# backend interface:
def empty_plan_cache():
    global _plan_cache
    _plan_cache = OpPlanCache()

def ncached_plans():
    global _plan_cache
    return _plan_cache.nentries

def empty_gencode_cache():
    global _kernel_stub_cache
    _kernel_stub_cache = dict()

def ncached_gencode():
    global _kernel_stub_cache
    return len(_kernel_stub_cache)

def _setup():
    global _ctx
    global _queue
    global _pref_work_group_count
    global _max_local_memory
    global _address_bits
    global _max_work_group_size
    global _has_dpfloat
    global _warpsize
    global _AMD_fixes
    global _plan_cache
    global _kernel_stub_cache
    global _reduction_task_cache

    _ctx = cl.create_some_context()
    _queue = cl.CommandQueue(_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    _pref_work_group_count = _queue.device.max_compute_units
    _max_local_memory = _queue.device.local_mem_size
    _address_bits = _queue.device.address_bits
    _max_work_group_size = _queue.device.max_work_group_size
    _has_dpfloat = 'cl_khr_fp64' in _queue.device.extensions or 'cl_amd_fp64' in _queue.device.extensions
    if not _has_dpfloat:
        warnings.warn('device does not support double precision floating point computation, expect undefined behavior for double')

    if _queue.device.type == cl.device_type.CPU:
        _warpsize = 1
    elif _queue.device.type == cl.device_type.GPU:
        # assumes nvidia, will probably fail with AMD gpus
        _warpsize = 32

    _AMD_fixes = _queue.device.platform.vendor in ['Advanced Micro Devices, Inc.']
    _plan_cache = OpPlanCache()
    _kernel_stub_cache = dict()
    _reduction_task_cache = dict()

_debug = False
_ctx = None
_queue = None
_pref_work_group_count = 0
_max_local_memory = 0
_address_bits = 32
_max_work_group_size = 0
_has_dpfloat = False
_warpsize = 0
_AMD_fixes = False
_plan_cache = None
_kernel_stub_cache = None
_reduction_task_cache = None

_jinja2_env = Environment(loader=PackageLoader("pyop2", "assets"))
_jinja2_direct_loop = _jinja2_env.get_template("opencl_direct_loop.jinja2")
_jinja2_indirect_loop = _jinja2_env.get_template("opencl_indirect_loop.jinja2")
