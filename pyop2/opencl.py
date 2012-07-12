import sequential as op2
from sequential import IdentityMap, READ, WRITE, RW, INC, MIN, MAX
import op_lib_core as core
import pyopencl as cl
import pkg_resources
import stringtemplate3
import pycparser
import numpy as np
import collections
import itertools

def round_up(bytes):
    return (bytes + 15) & ~15

class Kernel(op2.Kernel):

    _cparser = pycparser.CParser()

    def __init__(self, code, name):
        op2.Kernel.__init__(self, code, name)
        # deactivate until we have the memory attribute generator
        # in order to allow passing "opencl" C kernels
        # self._ast = Kernel._cparser.parse(self._code)

class Arg(op2.Arg):
    def __init__(self, data=None, map=None, idx=None, access=None):
        op2.Arg.__init__(self, data, map, idx, access)

    @property
    def _d_is_INC(self):
        return self._access == INC

    @property
    def _d_is_staged(self):
        # FIX; stagged only if dim > 1
        return isinstance(self._dat, Dat) and self._access in [READ, WRITE, RW]

    @property
    def _i_is_direct(self):
        return isinstance(self._dat, Dat) and self._map == IdentityMap

    @property
    def _i_is_reduction(self):
        return isinstance(self._dat, Dat) and self._access in [INC, MIN, MAX]

class DeviceDataMixin:

    ClTypeInfo = collections.namedtuple('ClTypeInfo', ['clstring', 'zero'])
    CL_TYPES = {np.dtype('uint32'): ClTypeInfo('unsigned int', '0u')}

    @property
    def _cl_type(self):
        return DeviceDataMixin.CL_TYPES[self._data.dtype].clstring

    @property
    def _cl_type_zero(self):
        return DeviceDataMixin.CL_TYPES[self._data.dtype].zero

class Dat(op2.Dat, DeviceDataMixin):

    _arg_type = Arg

    def __init__(self, dataset, dim, data=None, dtype=None, name=None):
        op2.Dat.__init__(self, dataset, dim, data, dtype, name)
        self._buffer = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._data.nbytes)
        cl.enqueue_write_buffer(_queue, self._buffer, self._data).wait()

    @property
    def bytes_per_elem(self):
        #FIX: probably not the best way to do... (pad, alg ?)
        return self._data.nbytes / self._dataset.size

    @property
    def data(self):
        cl.enqueue_read_buffer(_queue, self._buffer, self._data).wait()
        return self._data

class Mat(op2.Mat, DeviceDataMixin):

    _arg_type = Arg

    def __init__(self, datasets, dim, dtype=None, name=None):
        op2.Mat.__init__(self, datasets, dim, dtype, data, name)
        raise NotImplementedError('Matrix data is unsupported yet')

class Const(op2.Const, DeviceDataMixin):

    def __init__(self, dim, data=None, dtype=None, name=None):
        op2.Const.__init__(self, dim, data, dtype, name)
        raise NotImplementedError('Matrix data is unsupported yet')

class Global(op2.Global, DeviceDataMixin):

    _arg_type = Arg

    def __init__(self, dim, data=None, dtype=None, name=None):
        op2.Global.__init__(self, dim, data, dtype, name)
        self._buffer = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._data.nbytes)
        cl.enqueue_write_buffer(_queue, self._buffer, self._data).wait()

    def _allocate_reduction_array(self, nelems):
        self._h_reduc_array = np.zeros ((round_up(nelems * self._data.itemsize),), dtype=self._data.dtype)
        self._d_reduc_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._h_reduc_array.nbytes)
        #NOTE: the zeroing of the buffer could be made with an opencl kernel call
        cl.enqueue_write_buffer(_queue, self._d_reduc_buffer, self._h_reduc_array).wait()

    def _host_reduction(self, nelems):
        cl.enqueue_read_buffer(_queue, self._d_reduc_buffer, self._h_reduc_array).wait()
        for j in range(self._dim[0]):
            self._data[j] = 0

        for i in range(nelems):
            for j in range(self._dim[0]):
                self._data[j] += self._h_reduc_array[j + i * self._dim[0]]

        # get rid of the buffer and host temporary arrays
        del self._h_reduc_array
        del self._d_reduc_buffer

class Map(op2.Map):

    _arg_type = Arg

    def __init__(self, iterset, dataset, dim, values, name=None):
        op2.Map.__init__(self, iterset, dataset, dim, values, name)
        if self._iterset._size != 0:
            self._buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, self._values.nbytes)
            cl.enqueue_write_buffer(_queue, self._buffer, self._values).wait()

class DatMapPair(object):
    """ Dummy class needed for codegen
        could do without but would obfuscate codegen templates
    """
    def __init__(self, dat, map):
        self._dat = dat
        self._map = map

    @property
    def _i_direct(self):
        return isinstance(self._dat, Dat) and self._map != IdentityMap

#FIXME: some of this can probably be factorised up in common
class ParLoopCall(object):

    def __init__(self, kernel, it_space, *args):
        self._it_space = it_space
        self._kernel = kernel
        self._args = list(args)

    """ code generation specific """
    """ a lot of this can rewriten properly """
    @property
    def _d_staged_args(self):
        assert self.is_direct(), "Should only be called on direct loops"
        return list(set(self._d_staged_in_args + self._d_staged_out_args))

    @property
    def _d_nonreduction_args(self):
        assert self.is_direct(), "Should only be called on direct loops"
        return list(set(filter(lambda a: not isinstance(a._dat, Global), self._args)))

    @property
    def _d_staged_in_args(self):
        assert self.is_direct(), "Should only be called on direct loops"
        return list(set(filter(lambda a: isinstance(a._dat, Dat) and a._access in [READ, RW], self._args)))

    @property
    def _d_staged_out_args(self):
        assert self.is_direct(), "Should only be called on direct loops"
        return list(set(filter(lambda a: isinstance(a._dat, Dat) and a._access in [WRITE, RW], self._args)))

    @property
    def _d_reduction_args(self):
        assert self.is_direct(), "Should only be called on direct loops"
        return list(set(filter(lambda a: isinstance(a._dat, Global) and a._access in [INC, MIN, MAX], self._args)))

    """ maximum shared memory required for staging an op_arg """
    def _d_max_dynamic_shared_memory(self):
        assert self.is_direct(), "Should only be called on direct loops"
        return max(map(lambda a: a._dat.bytes_per_elem, self._d_staged_args))

    @property
    def _unique_dats(self):
        return list(set(map(lambda arg: arg._dat, self._args)))

    @property
    def _i_staged_dat_map_pairs(self):
        assert not self.is_direct(), "Should only be called on indirect loops"
        return set(map(lambda arg: DatMapPair(arg._dat, arg._map), filter(lambda a: a._map != IdentityMap and a._access in [READ, WRITE, RW], self._args)))

    @property
    def _i_staged_in_dat_map_pairs(self):
        assert not self.is_direct(), "Should only be called on indirect loops"
        return set(map(lambda arg: DatMapPair(arg._dat, arg._map), filter(lambda a: a._map != IdentityMap and a._access in [READ, RW], self._args)))

    @property
    def _i_staged_out_dat_map_pairs(self):
        assert not self.is_direct(), "Should only be called on indirect loops"
        return set(map(lambda arg: DatMapPair(arg._dat, arg._map), filter(lambda a: a._map != IdentityMap and a._access in [WRITE, RW], self._args)))

    @property
    def _i_reduc_args(self):
        assert not self.is_direct(), "Should only be called on indirect loops"
        return list(set(filter(lambda a: a._access in [INC, MIN, MAX] and a._map != IdentityMap, self._args)))

    def compute(self):
        if self.is_direct():
            thread_count = _threads_per_block * _blocks_per_grid
            dynamic_shared_memory_size = self._d_max_dynamic_shared_memory()
            shared_memory_offset = dynamic_shared_memory_size * _warpsize
            dynamic_shared_memory_size = dynamic_shared_memory_size * _threads_per_block
            dloop = _stg_direct_loop.getInstanceOf("direct_loop")
            dloop['parloop'] = self
            dloop['const'] = {"warpsize": _warpsize,\
                              "shared_memory_offset": shared_memory_offset,\
                              "dynamic_shared_memory_size": dynamic_shared_memory_size,\
                              "threads_per_block": _threads_per_block}
            source = str(dloop)
            prg = cl.Program (_ctx, source).build(options="-Werror")
            kernel = prg.__getattr__(self._kernel._name + '_stub')
            self._karg = 0
            for a in self._d_nonreduction_args:
                self._kernel_arg_append(kernel, a._dat._buffer)

            for a in self._d_reduction_args:
                a._dat._allocate_reduction_array(_blocks_per_grid)
                self._kernel_arg_append(kernel, a._dat._d_reduc_buffer)

            cl.enqueue_nd_range_kernel(_queue, kernel, (thread_count,), (_threads_per_block,), g_times_l=False).wait()
            for i, a in enumerate(self._d_reduction_args):
                a._dat._host_reduction(_blocks_per_grid)
        else:
            # call the plan function
            # loads plan into device memory
            for a in self._args:
                a.build_core_arg()

            plan = core.op_plan(self._kernel, self._it_space, *self._args)

            # codegen
            iloop = _stg_indirect_loop.getInstanceOf("indirect_loop")
            iloop['parloop'] = self
            source = str(iloop)

            prg = cl.Program(_ctx, source).build(options="-Werror")
            kernel = prg.__getattr__(self._kernel._name + '_stub')

            self._karg = 0
            for a in self._unique_dats:
                self._kernel_arg_append(kernel, a._buffer)

            ind_map = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=plan.ind_map.nbytes)
            cl.enqueue_write_buffer(_queue, ind_map, plan.ind_map).wait()
            for i in range(plan.nind_ele):
                self._kernel_arg_append(kernel, ind_map.get_sub_region(origin=i * self._it_space.size, size=self._it_space.size))

            loc_map = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=plan.loc_map.nbytes)
            cl.enqueue_write_buffer(_queue, loc_map, plan.loc_map).wait()
            for i in range(plan.nind_ele):
                self._kernel_arg_append(kernel, loc_map.get_sub_region(origin=i * self._it_space.size, size=self._it_space.size))

            ind_sizes = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=plan.ind_sizes.nbytes)
            cl.enqueue_write_buffer(_queue, ind_sizes, plan.ind_sizes).wait()
            self._kernel_arg_append(kernel, ind_sizes)

            ind_offs = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=plan.ind_offs.nbytes)
            cl.enqueue_write_buffer(_queue, ind_offs, plan.ind_offs).wait()
            self._kernel_arg_append(kernel, ind_offs)

            blkmap = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=plan.blkmap.nbytes)
            cl.enqueue_write_buffer(_queue, blkmap, plan.blkmap).wait()
            self._kernel_arg_append(kernel, blkmap)

            offset = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=plan.offset.nbytes)
            cl.enqueue_write_buffer(_queue, offset, plan.offset).wait()
            self._kernel_arg_append(kernel, offset)

            nelems = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=plan.nelems.nbytes)
            cl.enqueue_write_buffer(_queue, nelems, plan.nelems).wait()
            self._kernel_arg_append(kernel, nelems)

            nthrcol = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=plan.nthrcol.nbytes)
            cl.enqueue_write_buffer(_queue, nthrcol, plan.nthrcol).wait()
            self._kernel_arg_append(kernel, nthrcol)

            thrcol = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=plan.thrcol.nbytes)
            cl.enqueue_write_buffer(_queue, thrcol, plan.thrcol).wait()
            self._kernel_arg_append(kernel, thrcol)

            thrcol = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=plan.thrcol.nbytes)
            cl.enqueue_write_buffer(_queue, thrcol, plan.thrcol).wait()
            self._kernel_arg_append(kernel, thrcol)

            print 'kernel launch'
            block_offset = 0
            for i in range(plan.ncolors):
                blocks_per_grid = plan.ncolblk[i]
                dynamic_shared_memory_size = plan.nshared
                threads_per_block = _threads_per_block

                self._kernel.set_arg(self._karg, np.int32(block_offset))
                # call the kernel
                block_offset += blocks_per_grid

            raise NotImplementedError()

    def _kernel_arg_append(self, kernel, arg):
        kernel.set_arg(self._karg, arg)
        self._karg += 1

    def is_direct(self):
        return all(map(lambda a: isinstance(a._dat, Global) or (isinstance(a._dat, Dat) and a._map == IdentityMap), self._args))

def par_loop(kernel, it_space, *args):
    ParLoopCall(kernel, it_space, *args).compute()

_ctx = cl.create_some_context()
_queue = cl.CommandQueue(_ctx)
# FIXME: compute from ctx/device
_blocks_per_grid = 2
_threads_per_block = _ctx.get_info(cl.context_info.DEVICES)[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
_warpsize = 1

#preload string template groups
_stg_direct_loop = stringtemplate3.StringTemplateGroup(file=stringtemplate3.StringIO(pkg_resources.resource_string(__name__, "assets/opencl_direct_loop.stg")), lexer="default")
_stg_indirect_loop = stringtemplate3.StringTemplateGroup(file=stringtemplate3.StringIO(pkg_resources.resource_string(__name__, "assets/opencl_indirect_loop.stg")), lexer="default")
