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

"""OP2 CUDA backend."""

import os
import ctypes
from copy import deepcopy as dcopy

from contextlib import contextmanager
from hashlib import md5

from pyop2.datatypes import IntType, as_ctypes
from pyop2 import base
from pyop2 import compilation
from pyop2 import petsc_base
from pyop2.exceptions import *  # noqa: F401
from pyop2.mpi import collective
from pyop2.profiling import timed_region, timed_function
from pyop2.utils import *
from pyop2.configuration import configuration

import loopy
import numpy
import pycuda.driver as cuda
from pytools import memoize_method
from pyop2.petsc_base import PETSc, AbstractPETScBackend
from pyop2.logger import ExecTimeNoter


class Map(base.Map):
    """Map for CuDA"""

    def __init__(self, base_map):
        assert type(base_map) == base.Map
        self._iterset = base_map._iterset
        self._toset = base_map._toset
        self.comm = base_map.comm
        self._arity = base_map._arity
        # maps indexed as `map[icell, idof]`
        self._values = base_map._values
        self._values_cuda = cuda.mem_alloc(int(self._values.nbytes))
        cuda.memcpy_htod(self._values_cuda, self._values)
        self.shape = base_map.shape
        self._name = 'cuda_copy_%d_of_%s' % (base.Map._globalcount, base_map._name)
        self._offset = base_map._offset
        # A cache for objects built on top of this map
        self._cache = {}
        base.Map._globalcount += 1

    @cached_property
    def _kernel_args_(self):
        return (self._values_cuda, )


class ExtrudedSet(base.ExtrudedSet):
    """
    ExtrudedSet for CUDA.
    """
    @cached_property
    def _kernel_args_(self):
        m_gpu = cuda.mem_alloc(int(self.layers_array.nbytes))
        cuda.memcpy_htod(m_gpu, self.layers_array)
        return (m_gpu,)


class Subset(base.Subset):
    """
    Subset for CUDA.
    """
    @cached_property
    def _kernel_args_(self):
        m_gpu = cuda.mem_alloc(int(self._indices.nbytes))
        cuda.memcpy_htod(m_gpu, self._indices)
        return self._superset._kernel_args_ + (m_gpu, )


class DataSet(petsc_base.DataSet):
    @cached_property
    def layout_vec(self):
        """A PETSc Vec compatible with the dof layout of this DataSet."""
        1/0
        size = (self.size * self.cdim, None)
        vec = PETSc.Vec().create(comm=self.comm)
        vec.setSizes(size, bsize=self.cdim)
        vec.setType('cuda')
        vec.setUp()
        return vec

class Dat(petsc_base.Dat):
    """
    Dat for GPU.
    """
    @cached_property
    def _vec(self):
        assert self.dtype == PETSc.ScalarType, \
            "Can't create Vec with type %s, must be %s" % (self.dtype, PETSc.ScalarType)
        # Can't duplicate layout_vec of dataset, because we then
        # carry around extra unnecessary data.
        # But use getSizes to save an Allreduce in computing the
        # global size.
        size = self.dataset.layout_vec.getSizes()
        data = self._data[:size[0]]
        cuda_vec = PETSc.Vec().create(self.comm)
        cuda_vec.setSizes(size=size, bsize=self.cdim)
        cuda_vec.setType('cuda')
        cuda_vec.setArray(data)

        return cuda_vec


class Global(petsc_base.Global):

    @cached_property
    def device_handle(self):
        dev_data = cuda.mem_alloc(self._data.nbytes)
        cuda.memcpy_htod(dev_data, self._data)
        return dev_data

    @cached_property
    def _kernel_args_(self):
        return (self.device_handle, )


class JITModule(base.JITModule):

    _cppargs = []
    _libraries = []
    _system_headers = []

    def __init__(self, kernel, iterset, *args, **kwargs):
        r"""
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
        self.comm = iterset.comm
        self._kernel = kernel
        self._fun = None
        self._iterset = iterset
        self._args = args
        self._iteration_region = kwargs.get('iterate', base.ALL)
        self._pass_layer_arg = kwargs.get('pass_layer_arg', False)
        # Copy the class variables, so we don't overwrite them
        self._cppargs = dcopy(type(self)._cppargs)
        self._libraries = dcopy(type(self)._libraries)
        self._system_headers = dcopy(type(self)._system_headers)
        if not kwargs.get('delay', False):
            self.compile()
            self._initialized = True

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        key = super(JITModule, cls)._cache_key(*args, **kwargs)
        key += (configuration["gpu_strategy"],)
        if configuration["gpu_strategy"] == "scpt":
            pass
        elif configuration["gpu_strategy"] == "user_specified_tile":
            key += (
                    configuration["gpu_cells_per_block"],
                    configuration["gpu_threads_per_cell"],
                    configuration["gpu_op_tile_descriptions"],
                    configuration["gpu_quad_rowtile_lengths"],
                    configuration["gpu_input_to_shared"],
                    configuration["gpu_quad_weights_to_shared"],
                    configuration["gpu_mats_to_shared"],
                    configuration["gpu_tiled_prefetch_of_input"],
                    configuration["gpu_tiled_prefetch_of_quad_weights"],)
        elif configuration["gpu_strategy"] == "auto_tile":
            key += (
                    configuration["gpu_planner_kernel_evals"],)
            assert isinstance(args[1], Set)
            problem_size = args[1].size
            # FIXME: is this a good heuristic?
            # perform experiments to verify it.
            # Also this number should not exceed certain number i.e. when the
            # device would be saturated.
            key += (min(int(numpy.log2(problem_size)), 18),)
        else:
            raise NotImplementedError('For strategy: {}'.format(
                configuration["gpu_strategy"]))
        return key

    @memoize_method
    def grid_size(self, start, end):
        with open(self.config_file_path, 'r') as f:
            glens_llens = f.read()

        _, glens, llens = glens_llens.split('\n')
        from pymbolic import parse, evaluate
        glens = parse(glens)
        llens = parse(llens)

        parameters = {'start': start, 'end': end}

        grid = tuple(int(evaluate(glens[i], parameters)) if i < len(glens) else 1
                for i in range(2))
        block = tuple(int(evaluate(llens[i], parameters)) if i < len(llens) else 1
                for i in range(3))

        return grid, block

    @cached_property
    def get_args_marked_for_globals(self):
        args_to_make_global = []
        for i in range(len(self._fun.arg_format)-len(self.argtypes)):
            args_to_make_global.append(
                    numpy.load(self.ith_added_global_arg_i(i)))

        const_args_as_globals = tuple(cuda.mem_alloc(arg.nbytes) for arg in
            args_to_make_global)
        for arg_gpu, arg in zip(const_args_as_globals,
                args_to_make_global):
            cuda.memcpy_htod(arg_gpu, arg)

        evt = cuda.Event()
        evt.record()
        evt.synchronize()

        return const_args_as_globals

    @cached_property
    def config_file_path(self):
        cachedir = configuration['cache_dir']
        return os.path.join(cachedir, '{}_num_args_to_load_glens_llens'.format(self.get_encoded_cache_key))

    @memoize_method
    def ith_added_global_arg_i(self, i):
        cachedir = configuration['cache_dir']
        return os.path.join(cachedir, '{}_dat_{}.npy'.format(self.get_encoded_cache_key, i))

    @collective
    def __call__(self, *args):
        if self._initialized:
            grid, block = self.grid_size(args[0], args[1])
            extra_global_args = self.get_args_marked_for_globals
        else:
            self.args = args[:]
            self.compile()
            self._initialized = True
            return self.__call__(*args)

        return self._fun.prepared_call(grid, block, *(args+extra_global_args))

    @cached_property
    def _wrapper_name(self):
        return 'wrap_%s' % self._kernel.name

    @cached_property
    def num_args_to_make_global(self):
        with open(self.config_file_path, 'r') as f:
            return int(f.readline().strip())

    @cached_property
    def get_encoded_cache_key(self):
        a = md5(str(self.cache_key).encode()).hexdigest()
        return a

    @cached_property
    def code_to_compile(self):
        assert self.args is not None
        from pyop2.codegen.builder import WrapperBuilder
        from pyop2.codegen.rep2loopy import generate

        builder = WrapperBuilder(iterset=self._iterset, iteration_region=self._iteration_region, pass_layer_to_kernel=self._pass_layer_arg)
        for arg in self._args:
            builder.add_argument(arg)
        builder.set_kernel(self._kernel)

        wrapper = generate(builder)

        code, processed_program, args_to_make_global = generate_gpu_kernel(wrapper, self.args, self.argshapes)
        for i, arg_to_make_global in enumerate(args_to_make_global):
            numpy.save(self.ith_added_global_arg_i(i),
                    arg_to_make_global)

        with open(self.config_file_path, 'w') as f:
            glens, llens = processed_program.get_grid_size_upper_bounds_as_exprs()
            f.write(str(len(args_to_make_global)))
            f.write('\n')
            f.write('('+','.join(str(glen) for glen in glens)+',)')
            f.write('\n')
            f.write('('+','.join(str(llen) for llen in llens)+',)')

        return code

    @collective
    def compile(self):

        # If we weren't in the cache we /must/ have arguments
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule has no args associated with it, should never happen")

        compiler = "nvcc"
        extension = "cu"
        self._fun = compilation.load(self,
                                     extension,
                                     self._wrapper_name,
                                     cppargs=[],
                                     ldargs=[],
                                     compiler=compiler,
                                     comm=self.comm)

        type_map = dict([(ctypes.c_void_p, "P"), (ctypes.c_int, "i")])
        argtypes = "".join(type_map[t] for t in self.argtypes)

        self._fun.prepare(argtypes+"P"*self.num_args_to_make_global)

        # Blow away everything we don't need any more
        del self.args
        del self._args
        del self._kernel
        del self._iterset

    @cached_property
    def argtypes(self):
        index_type = as_ctypes(IntType)
        argtypes = (index_type, index_type)
        argtypes += self._iterset._argtypes_
        for arg in self._args:
            argtypes += arg._argtypes_
        seen = set()
        for arg in self._args:
            maps = arg.map_tuple
            for map_ in maps:
                for k, t in zip(map_._kernel_args_, map_._argtypes_):
                    if k in seen:
                        continue
                    argtypes += (ctypes.c_void_p,)
                    seen.add(k)

        return argtypes

    @cached_property
    def argshapes(self):
        argshapes = ((), ())
        if self._iterset._argtypes_:
            # FIXME: Do not put in a bogus value
            argshapes += ((), )

        for arg in self._args:
            argshapes += (arg.data.shape, )
        seen = set()
        for arg in self._args:
            maps = arg.map_tuple
            for map_ in maps:
                for k, t in zip(map_._kernel_args_, map_._argtypes_):
                    if k in seen:
                        continue
                    argshapes += (map_.shape, )
                    seen.add(k)

        return argshapes


class ParLoop(petsc_base.ParLoop):

    printed = set()

    def __init__(self, *args, **kwargs):
        super(ParLoop, self).__init__(*args, **kwargs)
        self.kernel.cpp = True

    def prepare_arglist(self, iterset, *args):
        nbytes = 0

        arglist = iterset._kernel_args_
        for arg in args:
            arglist += arg._kernel_args_
            if arg.access is base.INC:
                nbytes += arg.data.nbytes * 2
            else:
                nbytes += arg.data.nbytes
        seen = set()
        for arg in args:
            maps = arg.map_tuple
            for map_ in maps:
                for k in map_._kernel_args_:
                    if k in seen:
                        continue
                    arglist += map_._kernel_args_
                    seen.add(k)
                    nbytes += map_.values.nbytes

        self.nbytes = nbytes

        return arglist

    @collective
    def reduction_end(self):
        """End reductions"""
        if not self._has_reduction:
            return
        with self._reduction_event_end:
            for arg in self.global_reduction_args:
                arg.reduction_end(self.comm)
            # Finalise global increments
            for tmp, glob in self._reduced_globals.items():
                # copy results to the host
                cuda.memcpy_dtoh(tmp._data, tmp.device_handle)
                glob._data += tmp._data

    @cached_property
    def _jitmodule(self):
        return JITModule(self.kernel, self.iterset, *self.args,
                         iterate=self.iteration_region,
                         pass_layer_arg=self._pass_layer_arg,
                         delay=True)

    @collective
    def _compute(self, part, fun, *arglist):
        if part.size == 0:
            return

        # how about over here we decide what should the strategy be..

        if configuration["gpu_timer"]:
            start = cuda.Event()
            end = cuda.Event()
            start.record()
            start.synchronize()
            fun(part.offset, part.offset + part.size, *arglist)
            end.record()
            end.synchronize()
            ExecTimeNoter.note(start.time_till(end)/1000)
            # print("{0}_TIME= {1}".format(self._jitmodule._wrapper_name, start.time_till(end)/1000))
            return

        with timed_region("ParLoop_{0}_{1}".format(self.iterset.name, self._jitmodule._wrapper_name)):
            fun(part.offset, part.offset + part.size, *arglist)


def generate_single_cell_wrapper(iterset, args, forward_args=(), kernel_name=None, wrapper_name=None, restart_counter=True):
    """Generates wrapper for a single cell. No iteration loop, but cellwise data is extracted.
    Cell is expected as an argument to the wrapper. For extruded, the numbering of the cells
    is columnwise continuous, bottom to top.

    :param iterset: The iteration set
    :param args: :class:`Arg`s
    :param forward_args: To forward unprocessed arguments to the kernel via the wrapper,
                         give an iterable of strings describing their C types.
    :param kernel_name: Kernel function name
    :param wrapper_name: Wrapper function name
    :param restart_counter: Whether to restart counter in naming variables and indices
                            in code generation.

    :return: string containing the C code for the single-cell wrapper
    """
    from pyop2.codegen.builder import WrapperBuilder
    from pyop2.codegen.rep2loopy import generate
    from loopy.types import OpaqueType

    forward_arg_types = [OpaqueType(fa) for fa in forward_args]
    builder = WrapperBuilder(iterset=iterset, single_cell=True, forward_arg_types=forward_arg_types)
    for arg in args:
        builder.add_argument(arg)
    builder.set_kernel(Kernel("", kernel_name))
    wrapper = generate(builder, wrapper_name, restart_counter)
    code = loopy.generate_code_v2(wrapper)

    return code.device_code()


def transpose_maps(kernel):
    1/0
    from loopy.kernel.array import FixedStrideArrayDimTag
    from pymbolic import parse

    new_dim_tags = (FixedStrideArrayDimTag(1),
            FixedStrideArrayDimTag(parse('end-start')))
    new_args = [arg.copy(dim_tags=new_dim_tags) if arg.name[:3] == 'map' else arg for arg in kernel.args]
    kernel = kernel.copy(args=new_args)
    return kernel


def generate_gpu_kernel(program, args=None, argshapes=None):
    # Kernel transformations
    program = program.copy(target=loopy.CudaTarget())
    kernel = program.root_kernel

    def insn_needs_atomic(insn):
        # updates to global variables are atomic
        import pymbolic
        if isinstance(insn, loopy.Assignment):
            if isinstance(insn.assignee, pymbolic.primitives.Subscript):
                assignee_name = insn.assignee.aggregate.name
            else:
                assert isinstance(insn.assignee, pymbolic.primitives.Variable)
                assignee_name = insn.assignee.name

            if assignee_name in kernel.arg_dict:
                return assignee_name in insn.read_dependency_names()
        return False

    new_insns = []
    args_marked_for_atomic = set()
    for insn in kernel.instructions:
        if insn_needs_atomic(insn):
            atomicity = (loopy.AtomicUpdate(insn.assignee.aggregate.name), )
            insn = insn.copy(atomicity=atomicity)
            args_marked_for_atomic |= set([insn.assignee.aggregate.name])

        new_insns.append(insn)

    # label args as atomic
    new_args = []
    for arg in kernel.args:
        if arg.name in args_marked_for_atomic:
            new_args.append(arg.copy(for_atomic=True))
        else:
            new_args.append(arg)

    kernel = kernel.copy(instructions=new_insns, args=new_args)
    #FIXME: These might not always be true
    # Might need to be removed before going full production
    kernel = loopy.assume(kernel, "start=0")
    kernel = loopy.assume(kernel, "end>0")

    # choose the preferred algorithm here
    # TODO: Not sure if this is the right way to select different
    # transformation strategies based on kernels
    if program.name in [
            "wrap_form0_cell_integral_otherwise",
            "wrap_form0_exterior_facet_integral_otherwise",
            "wrap_form0_interior_facet_integral_otherwise",
            "wrap_form1_cell_integral_otherwise"]:
        if configuration["gpu_strategy"] == "scpt":
            from pyop2.gpu.snpt import snpt_transform
            kernel, args_to_make_global = snpt_transform(kernel,
                        configuration["gpu_cells_per_block"])
        elif configuration["gpu_strategy"] == "user_specified_tile":
            from pyop2.gpu.tile import tiled_transform
            from pyop2.gpu.tile import TilingConfiguration
            kernel, args_to_make_global = tiled_transform(kernel,
                    program.callables_table,
                    TilingConfiguration(configuration["gpu_cells_per_block"],
                        configuration["gpu_threads_per_cell"],
                        configuration["gpu_op_tile_descriptions"],
                        configuration["gpu_quad_rowtile_lengths"],
                        configuration["gpu_coords_to_shared"],
                        configuration["gpu_input_to_shared"],
                        configuration["gpu_mats_to_shared"],
                        configuration["gpu_quad_weights_to_shared"],
                        configuration["gpu_tiled_prefetch_of_input"],
                        configuration["gpu_tiled_prefetch_of_quad_weights"])
                    )
        elif configuration["gpu_strategy"] == "auto_tile":
            assert args is not None
            assert argshapes is not None
            from pyop2.gpu.tile import AutoTiler
            kernel, args_to_make_global = AutoTiler(
                    program.with_root_kernel(kernel),
                    configuration["gpu_planner_kernel_evals"])(args, argshapes)
        else:
            raise ValueError("gpu_strategy can be 'scpt',"
                    " 'user_specified_tile' or 'auto_tile'.")
    elif program.name in ["wrap_zero", "wrap_expression_kernel",
            "wrap_expression", "wrap_pyop2_kernel_uniform_extrusion",
            "wrap_form_cell_integral_otherwise",
            "wrap_loopy_kernel_prolong",
            "wrap_loopy_kernel_restrict",
            "wrap_loopy_kernel_inject", "wrap_copy"
            ]:
        from pyop2.gpu.snpt import snpt_transform
        kernel, args_to_make_global = snpt_transform(kernel,
                    configuration["gpu_cells_per_block"])
    else:
        raise NotImplementedError("Transformation for '%s'." % program.name)

    if False:
        # FIXME
        # optimization for lower order but needs some help from
        # ~firedrake.mesh~ in setting the data layout
        kernel = transpose_maps(kernel)

    program = program.with_root_kernel(kernel)

    code = loopy.generate_code_v2(program).device_code()

    if program.name == "wrap_pyop2_kernel_uniform_extrusion":
        code = code.replace("inline void pyop2_kernel_uniform_extrusion", "__device__ inline void pyop2_kernel_uniform_extrusion")

    return code, program, args_to_make_global


class CUDABackend(AbstractPETScBackend):
    ParLoop = ParLoop
    Set = base.Set
    ExtrudedSet = ExtrudedSet
    MixedSet = base.MixedSet
    Subset = Subset
    DataSet = DataSet
    MixedDataSet = petsc_base.MixedDataSet
    Map = Map
    MixedMap = base.MixedMap
    Dat = Dat
    MixedDat = petsc_base.MixedDat
    DatView = base.DatView
    Mat = petsc_base.Mat
    Global = Global
    GlobalDataSet = petsc_base.GlobalDataSet
    PETScVecType = 'seqcuda'


cuda_backend = CUDABackend()
