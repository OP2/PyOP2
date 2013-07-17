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

from device import *
import device
import petsc_base
from utils import verify_reshape, uniquify, maybe_setflags
import configuration as cfg
import pyopencl as cl
from pyopencl import array
import numpy as np
import collections
import warnings
import math
from jinja2 import Environment, PackageLoader
from pycparser import c_parser, c_ast, c_generator
import os
import time

class Kernel(device.Kernel):
    """OP2 OpenCL kernel type."""

    def __init__(self, code, name):
        device.Kernel.__init__(self, code, name)

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
                    t = c_ast.PtrDecl([], c_ast.TypeDecl(cst._name, ["__constant"],
                                      c_ast.IdentifierType([cst._cl_type])))
                decl = c_ast.Decl(cst._name, [], [], [], t, None, 0)
                node.params.append(decl)

    def instrument(self, instrument, constants):
        ast = c_parser.CParser().parse(self._code)
        Kernel.Instrument().instrument(ast, self._name, instrument, constants)
        return c_generator.CGenerator().visit(ast)

class Arg(device.Arg):
    """OP2 OpenCL argument type."""

    # FIXME actually use this in the template
    def _indirect_kernel_arg_name(self, idx):
        if self._is_global:
            if self._is_global_reduction:
                return self._reduction_local_name
            else:
                return self._name
        if self._is_direct:
            if self.data.soa:
                return "%s + (%s + offset_b)" % (self._name, idx)
            return "%s + (%s + offset_b) * %s" % (self._name, idx,
                                                  self.data.cdim)
        if self._is_indirect:
            if self._is_vec_map:
                return self._vec_name
            if self.access is device.INC:
                return self._local_name()
            else:
                return "%s + loc_map[%s * set_size + %s + offset_b]*%s" \
                    % (self._shared_name, self._which_indirect, idx,
                       self.data.cdim)

    def _direct_kernel_arg_name(self, idx=None):
        if self._is_mat:
            return self._mat_entry_name
        if self._is_staged_direct:
            return self._local_name()
        elif self._is_global_reduction:
            return self._reduction_local_name
        elif self._is_global:
            return self._name
        else:
            return "%s + %s" % (self._name, idx)

class DeviceDataMixin(device.DeviceDataMixin):
    """Codegen mixin for datatype and literal translation."""

    ClTypeInfo = collections.namedtuple('ClTypeInfo', ['clstring', 'zero', 'min', 'max'])
    CL_TYPES = {np.dtype('uint8'): ClTypeInfo('uchar', '0', '0', '255'),
                np.dtype('int8'): ClTypeInfo('char', '0', '-127', '127'),
                np.dtype('uint16'): ClTypeInfo('ushort', '0', '0', '65535'),
                np.dtype('int16'): ClTypeInfo('short', '0', '-32767', '32767'),
                np.dtype('uint32'): ClTypeInfo('uint', '0u', '0u', '4294967295u'),
                np.dtype('int32'): ClTypeInfo('int', '0', '-2147483647', '2147483647'),
                np.dtype('uint64'): ClTypeInfo('ulong', '0ul', '0ul', '18446744073709551615ul'),
                np.dtype('int64'): ClTypeInfo('long', '0l', '-9223372036854775807l', '9223372036854775807l'),
                np.dtype('float32'): ClTypeInfo('float', '0.0f', '-3.4028235e+38f', '3.4028235e+38f'),
                np.dtype('float64'): ClTypeInfo('double', '0.0', '-1.7976931348623157e+308', '1.7976931348623157e+308')}

    def _allocate_device(self):
        if self.state is DeviceDataMixin.DEVICE_UNALLOCATED:
            if self.soa:
                shape = self._data.T.shape
            else:
                shape = self._data.shape
            self._device_data = array.empty(_queue, shape=shape,
                                            dtype=self.dtype)
            self.state = DeviceDataMixin.HOST

    def _to_device(self):
        self._allocate_device()
        if self.state is DeviceDataMixin.HOST:
            self._device_data.set(self._maybe_to_soa(self._data),
                                  queue=_queue)
            self.state = DeviceDataMixin.BOTH

    def _from_device(self):
        flag = self._data.flags['WRITEABLE']
        maybe_setflags(self._data, write=True)
        if self.state is DeviceDataMixin.DEVICE:
            self._device_data.get(_queue, self._data)
            self._data = self._maybe_to_aos(self._data)
            self.state = DeviceDataMixin.BOTH
        maybe_setflags(self._data, write=flag)

    @property
    def _cl_type(self):
        return DeviceDataMixin.CL_TYPES[self.dtype].clstring

    @property
    def _cl_type_zero(self):
        return DeviceDataMixin.CL_TYPES[self.dtype].zero

    @property
    def _cl_type_min(self):
        return DeviceDataMixin.CL_TYPES[self.dtype].min

    @property
    def _cl_type_max(self):
        return DeviceDataMixin.CL_TYPES[self.dtype].max

class Dat(device.Dat, petsc_base.Dat, DeviceDataMixin):
    """OP2 OpenCL vector data type."""

    @property
    def norm(self):
        """The L2-norm on the flattened vector."""
        return np.sqrt(array.dot(self.array, self.array).get())

class Sparsity(device.Sparsity):
    @property
    def colidx(self):
        if not hasattr(self, '__dev_colidx'):
            setattr(self, '__dev_colidx',
                    array.to_device(_queue,
                                    self._colidx))
        return getattr(self, '__dev_colidx')

    @property
    def rowptr(self):
        if not hasattr(self, '__dev_rowptr'):
            setattr(self, '__dev_rowptr',
                    array.to_device(_queue,
                                    self._rowptr))
        return getattr(self, '__dev_rowptr')

class Mat(device.Mat, petsc_base.Mat, DeviceDataMixin):
    """OP2 OpenCL matrix data type."""

    def _allocate_device(self):
        pass

    def _to_device(self):
        pass

    def _from_device(self):
        pass

    @property
    def _dev_array(self):
        if not hasattr(self, '__dev_array'):
            setattr(self, '__dev_array',
                    array.empty(_queue,
                                self.sparsity.nz,
                                self.dtype))
        return getattr(self, '__dev_array')

    @property
    def _colidx(self):
        return self._sparsity.colidx

    @property
    def _rowptr(self):
        return self._sparsity.rowptr

    def _upload_array(self):
        self._dev_array.set(self.array, queue=_queue)
        self.state = DeviceDataMixin.BOTH

    def assemble(self):
        if self.state is DeviceDataMixin.DEVICE:
            self._dev_array.get(queue=_queue, ary=self.array)
            self.state = DeviceDataMixin.BOTH
        self.handle.assemble()

    @property
    def cdim(self):
        return np.prod(self.dims)

class Const(device.Const, DeviceDataMixin):
    """OP2 OpenCL data that is constant for any element of any set."""

    @property
    def _array(self):
        if not hasattr(self, '__array'):
            setattr(self, '__array', array.to_device(_queue, self._data))
        return getattr(self, '__array')

class Global(device.Global, DeviceDataMixin):
    """OP2 OpenCL global value."""

    @property
    def _array(self):
        if not hasattr(self, '_device_data'):
            self._device_data = array.to_device(_queue, self._data)
        return self._device_data

    def _allocate_reduction_array(self, nelems):
        self._d_reduc_array = array.zeros (_queue, nelems * self.cdim, dtype=self.dtype)

    @property
    def data(self):
        if self.state is DeviceDataMixin.DEVICE:
            self._array.get(_queue, ary=self._data)
        if self.state is not DeviceDataMixin.DEVICE_UNALLOCATED:
            self.state = DeviceDataMixin.HOST
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        if self.state is not DeviceDataMixin.DEVICE_UNALLOCATED:
            self.state = DeviceDataMixin.HOST

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

            op = {INC: 'INC', MIN: 'min', MAX: 'max'}

            return """
%(headers)s
#define INC(a,b) ((a)+(b))
__kernel
void global_%(type)s_%(dim)s_post_reduction (
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
""" % {'headers': headers(), 'dim': self.cdim, 'type': self._cl_type, 'op': op[reduction_operator]}


        src, kernel = _reduction_task_cache.get((self.dtype, self.cdim, reduction_operator), (None, None))
        if src is None :
            src = generate_code()
            prg = cl.Program(_ctx, src).build(options="-Werror")
            name = "global_%s_%s_post_reduction" % (self._cl_type, self.cdim)
            kernel = prg.__getattr__(name)
            _reduction_task_cache[(self.dtype, self.cdim, reduction_operator)] = (src, kernel)

        kernel.set_arg(0, self._array.data)
        kernel.set_arg(1, self._d_reduc_array.data)
        kernel.set_arg(2, np.int32(nelems))
        cl.enqueue_task(_queue, kernel).wait()

        del self._d_reduc_array

class Map(device.Map):
    """OP2 OpenCL map, a relation between two Sets."""

    def _to_device(self):
        if not hasattr(self, '_device_values'):
            self._device_values = array.to_device(_queue, self._values)
        else:
            warnings.warn("Copying Map data for %s again, do you really want to do this?" % self)
            self._device_values.set(self._values, _queue)

class Plan(device.Plan):
    @property
    def ind_map(self):
        if not hasattr(self, '_ind_map_array'):
            self._ind_map_array = array.to_device(_queue, super(Plan,self).ind_map)
        return self._ind_map_array

    @property
    def ind_sizes(self):
        if not hasattr(self, '_ind_sizes_array'):
            self._ind_sizes_array = array.to_device(_queue, super(Plan,self).ind_sizes)
        return self._ind_sizes_array

    @property
    def ind_offs(self):
        if not hasattr(self, '_ind_offs_array'):
            self._ind_offs_array = array.to_device(_queue, super(Plan,self).ind_offs)
        return self._ind_offs_array

    @property
    def loc_map(self):
        if not hasattr(self, '_loc_map_array'):
            self._loc_map_array = array.to_device(_queue, super(Plan,self).loc_map)
        return self._loc_map_array

    @property
    def blkmap(self):
        if not hasattr(self, '_blkmap_array'):
            self._blkmap_array = array.to_device(_queue, super(Plan,self).blkmap)
        return self._blkmap_array

    @property
    def offset(self):
        if not hasattr(self, '_offset_array'):
            self._offset_array = array.to_device(_queue, super(Plan,self).offset)
        return self._offset_array

    @property
    def nelems(self):
        if not hasattr(self, '_nelems_array'):
            self._nelems_array = array.to_device(_queue, super(Plan,self).nelems)
        return self._nelems_array

    @property
    def nthrcol(self):
        if not hasattr(self, '_nthrcol_array'):
            self._nthrcol_array = array.to_device(_queue, super(Plan,self).nthrcol)
        return self._nthrcol_array

    @property
    def thrcol(self):
        if not hasattr(self, '_thrcol_array'):
            self._thrcol_array = array.to_device(_queue, super(Plan,self).thrcol)
        return self._thrcol_array


class Solver(petsc_base.Solver):

    def solve(self, A, x, b):
        x._from_device()
        b._from_device()
        super(Solver, self).solve(A, x, b)
        # Explicitly mark solution as dirty so a copy back to device occurs
        if x.state is not DeviceDataMixin.DEVICE_UNALLOCATED:
            x.state = DeviceDataMixin.HOST
        x._to_device()

class JITModule(base.JITModule):

    def __init__(self, kernel, itspace_extents, *args, **kwargs):
        # No need to protect against re-initialization since these attributes
        # are not expensive to set and won't be used if we hit cache
        self._parloop = kwargs.get('parloop')
        self._conf = kwargs.get('conf')

    def compile(self):
        if hasattr(self, '_fun'):
            return self._fun
        def instrument_user_kernel():
            inst = []

            for arg in self._parloop.args:
                i = None
                if self._parloop._is_direct:
                    if (arg._is_direct and (arg.data._is_scalar or arg.data.soa)) or\
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

            for i in self._parloop._it_space.extents:
                inst.append(("__private", None))

            return self._parloop._kernel.instrument(inst, Const._definitions())

        #do codegen
        user_kernel = instrument_user_kernel()
        template = _jinja2_direct_loop if self._parloop._is_direct \
                                       else _jinja2_indirect_loop

        src = template.render({'parloop': self._parloop,
                               'user_kernel': user_kernel,
                               'launch': self._conf,
                               'codegen': {'amd': _AMD_fixes},
                               'op2const': Const._definitions()
                              }).encode("ascii")
        self.dump_gen_code(src)
        prg = cl.Program(_ctx, src).build(options="-Werror")
        self._fun = prg.__getattr__(self._parloop._stub_name)
        return self._fun

    def dump_gen_code(self, src):
        if cfg['dump-gencode']:
            path = cfg['dump-gencode-path'] % {"kernel": self.kernel.name,
                                               "time": time.strftime('%Y-%m-%d@%H:%M:%S')}

            if not os.path.exists(path):
                with open(path, "w") as f:
                    f.write(src)

    def __call__(self, thread_count, work_group_size, *args):
        fun = self.compile()
        for i, arg in enumerate(args):
            fun.set_arg(i, arg)
        cl.enqueue_nd_range_kernel(_queue, fun, (thread_count,),
                                   (work_group_size,), g_times_l=False).wait()

class ParLoop(device.ParLoop):
    @property
    def _matrix_args(self):
        return [a for a in self.args if a._is_mat]

    @property
    def _unique_matrix(self):
        return uniquify(a.data for a in self._matrix_args)

    @property
    def _matrix_entry_maps(self):
        """Set of all mappings used in matrix arguments."""
        return uniquify(m for arg in self.args  if arg._is_mat for m in arg.map)

    @property
    def _requires_matrix_coloring(self):
        """Direct code generation to follow colored execution for global matrix insertion."""
        return not _supports_64b_atomics and not not self._matrix_args

    def _i_partition_size(self):
        #TODO FIX: something weird here
        #available_local_memory
        warnings.warn('temporary fix to available local memory computation (-512)')
        available_local_memory = _max_local_memory - 512
        # 16bytes local mem used for global / local indices and sizes
        available_local_memory -= 16
        # (4/8)ptr size per dat passed as argument (dat)
        available_local_memory -= (_address_bits / 8) * (len(self._unique_dat_args) + len(self._all_global_non_reduction_args))
        # (4/8)ptr size per dat/map pair passed as argument (ind_map)
        available_local_memory -= (_address_bits / 8) * len(self._unique_indirect_dat_args)
        # (4/8)ptr size per global reduction temp array
        available_local_memory -= (_address_bits / 8) * len(self._all_global_reduction_args)
        # (4/8)ptr size per indirect arg (loc_map)
        available_local_memory -= (_address_bits / 8) * len(self._all_indirect_args)
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
        available_local_memory -= 4 + (_address_bits / 8) * 2 * len(self._unique_indirect_dat_args)
        # inside shared memory padding
        available_local_memory -= 2 * (len(self._unique_indirect_dat_args) - 1)

        max_bytes = sum(map(lambda a: a.data._bytes_per_elem, self._all_indirect_args))
        return available_local_memory / (2 * _warpsize * max_bytes) * (2 * _warpsize)

    def launch_configuration(self):
        if self._is_direct:
            per_elem_max_local_mem_req = self._max_shared_memory_needed_per_set_element
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
                available_local_memory -= (len(self._unique_dat_args) + len(self._all_global_non_reduction_args))\
                                          * (_address_bits / 8)
                available_local_memory -= len(self._all_global_reduction_args) * (_address_bits / 8)
                available_local_memory -= 7
                ps = available_local_memory / per_elem_max_local_mem_req
                wgs = min(_max_work_group_size, (ps / _warpsize) * _warpsize)
            nwg = min(_pref_work_group_count, int(math.ceil(self._it_space.size / float(wgs))))
            ttc = wgs * nwg

            local_memory_req = per_elem_max_local_mem_req * wgs
            return {'thread_count': ttc,
                    'work_group_size': wgs,
                    'work_group_count': nwg,
                    'local_memory_size': local_memory_req,
                    'local_memory_offset': shared_memory_offset}
        else:
            return {'partition_size': self._i_partition_size()}

    def compute(self):
        if self._has_soa:
            op2stride = Const(1, self._it_space.size, name='op2stride',
                              dtype='int32')

        conf = self.launch_configuration()

        if self._is_indirect:
            self._plan = Plan(self.kernel, self._it_space.iterset,
                              *self._unwound_args,
                              partition_size=conf['partition_size'],
                              matrix_coloring=self._requires_matrix_coloring)
            conf['local_memory_size'] = self._plan.nshared
            conf['ninds'] = self._plan.ninds
            conf['work_group_size'] = min(_max_work_group_size,
                                          conf['partition_size'])
            conf['work_group_count'] = self._plan.nblocks
        conf['warpsize'] = _warpsize

        fun = JITModule(self.kernel, self.it_space, *self.args, parloop=self, conf=conf)

        args = []
        for arg in self._unique_args:
            arg.data._allocate_device()
            if arg.access is not device.WRITE:
                arg.data._to_device()

        for a in self._unique_dat_args:
            args.append(a.data.array.data)

        for a in self._all_global_non_reduction_args:
            args.append(a.data._array.data)

        for a in self._all_global_reduction_args:
            a.data._allocate_reduction_array(conf['work_group_count'])
            args.append(a.data._d_reduc_array.data)

        for cst in Const._definitions():
            args.append(cst._array.data)

        for m in self._unique_matrix:
            args.append(m._dev_array.data)
            m._upload_array()
            args.append(m._rowptr.data)
            args.append(m._colidx.data)

        for m in self._matrix_entry_maps:
            m._to_device()
            args.append(m._device_values.data)

        if self._is_direct:
            args.append(np.int32(self._it_space.size))
            fun(conf['thread_count'], conf['work_group_size'], *args)
        else:
            args.append(np.int32(self._it_space.size))
            args.append(self._plan.ind_map.data)
            args.append(self._plan.loc_map.data)
            args.append(self._plan.ind_sizes.data)
            args.append(self._plan.ind_offs.data)
            args.append(self._plan.blkmap.data)
            args.append(self._plan.offset.data)
            args.append(self._plan.nelems.data)
            args.append(self._plan.nthrcol.data)
            args.append(self._plan.thrcol.data)

            block_offset = 0
            args.append(0)
            for i in range(self._plan.ncolors):
                blocks_per_grid = int(self._plan.ncolblk[i])
                threads_per_block = min(_max_work_group_size, conf['partition_size'])
                thread_count = threads_per_block * blocks_per_grid

                args[-1] = np.int32(block_offset)
                fun(int(thread_count), int(threads_per_block), *args)
                block_offset += blocks_per_grid

        # mark !READ data as dirty
        for arg in self.args:
            if arg.access is not READ:
                arg.data.state = DeviceDataMixin.DEVICE
            if arg._is_dat:
                maybe_setflags(arg.data._data, write=False)

        for mat in [arg.data for arg in self._matrix_args]:
            mat.assemble()

        for a in self._all_global_reduction_args:
            a.data._post_kernel_reduction_task(conf['work_group_count'], a.access)

        if self._has_soa:
            op2stride.remove_from_namespace()

def par_loop(kernel, it_space, *args):
    ParLoop(kernel, it_space, *args).compute()

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
    global _reduction_task_cache
    global _supports_64b_atomics

    _ctx = cl.create_some_context()
    _queue = cl.CommandQueue(_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    _pref_work_group_count = _queue.device.max_compute_units
    _max_local_memory = _queue.device.local_mem_size
    _address_bits = _queue.device.address_bits
    _max_work_group_size = _queue.device.max_work_group_size
    _has_dpfloat = 'cl_khr_fp64' in _queue.device.extensions or 'cl_amd_fp64' in _queue.device.extensions
    if not _has_dpfloat:
        warnings.warn('device does not support double precision floating point computation, expect undefined behavior for double')

    if 'cl_khr_int64_base_atomics' in _queue.device.extensions:
        _supports_64b_atomics = True

    if _queue.device.type == cl.device_type.CPU:
        _warpsize = 1
    elif _queue.device.type == cl.device_type.GPU:
        # assumes nvidia, will probably fail with AMD gpus
        _warpsize = 32

    _AMD_fixes = _queue.device.platform.vendor in ['Advanced Micro Devices, Inc.']
    _reduction_task_cache = dict()

_supports_64b_atomics = False
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
_reduction_task_cache = None

_jinja2_env = Environment(loader=PackageLoader("pyop2", "assets"))
_jinja2_direct_loop = _jinja2_env.get_template("opencl_direct_loop.jinja2")
_jinja2_indirect_loop = _jinja2_env.get_template("opencl_indirect_loop.jinja2")
