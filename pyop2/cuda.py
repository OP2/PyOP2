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

import ctypes
from copy import deepcopy as dcopy

from contextlib import contextmanager

from pyop2.datatypes import IntType, as_ctypes
from pyop2 import base
from pyop2 import petsc_base
from pyop2.base import par_loop                          # noqa: F401
from pyop2.base import READ, WRITE, RW, INC, MIN, MAX    # noqa: F401
from pyop2.base import ALL
from pyop2.base import Map, MixedMap, Sparsity, Halo  # noqa: F401
from pyop2.base import Set, ExtrudedSet, MixedSet, Subset  # noqa: F401
from pyop2.base import DatView                           # noqa: F401
from pyop2.base import Kernel                            # noqa: F401
from pyop2.base import Arg                               # noqa: F401
from pyop2.petsc_base import DataSet, MixedDataSet       # noqa: F401
from pyop2.petsc_base import GlobalDataSet       # noqa: F401
from pyop2.petsc_base import Dat as petsc_Dat
from pyop2.petsc_base import Global as petsc_Global
from pyop2.petsc_base import PETSc, MixedDat, Mat          # noqa: F401
from pyop2.exceptions import *  # noqa: F401
from pyop2.mpi import collective
from pyop2.profiling import timed_region, timed_function
from pyop2.utils import cached_property
from pyop2.configuration import configuration

import loopy
import re
import pycuda
import pycuda.autoinit
import pycuda.driver as cuda_driver
from pytools import memoize_method


class Map(Map):

    @cached_property
    def device_handle(self):
        m_gpu = cuda_driver.mem_alloc(int(self.values.nbytes))
        cuda_driver.memcpy_htod(m_gpu, self.values.flatten())
        return m_gpu

    @cached_property
    def _kernel_args_(self):
        return (self.device_handle, )


class Arg(Arg):
    """
    Arg for GPU
    """


class Dat(petsc_Dat):
    """
    Dat for GPU.
    """

    @contextmanager
    def vec_context(self, access):
        """A context manager for a :class:`PETSc.Vec` from a :class:`Dat`.

        :param access: Access descriptor: READ, WRITE, or RW."""

        assert self.dtype == PETSc.ScalarType, \
            "Can't create Vec with type %s, must be %s" % (self.dtype, PETSc.ScalarType)
        # Getting the Vec needs to ensure we've done all current
        # necessary computation.
        self._force_evaluation(read=access is not base.WRITE,
                               write=access is not base.READ)
        if not hasattr(self, '_vec'):
            # Can't duplicate layout_vec of dataset, because we then
            # carry around extra unnecessary data.
            # But use getSizes to save an Allreduce in computing the
            # global size.
            size = self.dataset.layout_vec.getSizes()
            data = self._data[:size[0]]
            self._vec = PETSc.Vec().create(self.comm)
            self._vec.setSizes(size=size, bsize=self.cdim)
            self._vec.setType('cuda')
            self._vec.setArray(data)
        # PETSc Vecs have a state counter and cache norm computations
        # to return immediately if the state counter is unchanged.
        # Since we've updated the data behind their back, we need to
        # change that state counter.
        self._vec.stateIncrease()
        yield self._vec
        if access is not base.READ:
            self.halo_valid = False

    @cached_property
    def device_handle(self):
        with self.vec as v:
            return v.getCUDAHandle()

    @cached_property
    def _kernel_args_(self):
        return (self.device_handle, )

    @collective
    @property
    def data(self):

        from pyop2.base import _trace

        _trace.evaluate(set([self]), set([self]))

        with self.vec as v:
            v.restoreCUDAHandle(self.device_handle)
            return v.array

        # cuda_driver.memcpy_dtoh(self.data, self.device_handle)
        # return self.data


class Global(petsc_Global):
    @contextmanager
    def vec_context(self, access):
        """A context manager for a :class:`PETSc.Vec` from a :class:`Global`.

        :param access: Access descriptor: READ, WRITE, or RW."""

        assert self.dtype == PETSc.ScalarType, \
            "Can't create Vec with type %s, must be %s" % (self.dtype, PETSc.ScalarType)
        # Getting the Vec needs to ensure we've done all current
        # necessary computation.
        self._force_evaluation(read=access is not base.WRITE,
                               write=access is not base.READ)
        data = self._data
        if not hasattr(self, '_vec'):
            # Can't duplicate layout_vec of dataset, because we then
            # carry around extra unnecessary data.
            # But use getSizes to save an Allreduce in computing the
            # global size.
            size = self.dataset.layout_vec.getSizes()
            self._vec = PETSc.Vec().create(self.comm)
            self._vec.setSizes(size=size, bsize=self.cdim)
            self._vec.setType('cuda')
            self._vec.setArray(data)
        # PETSc Vecs have a state counter and cache norm computations
        # to return immediately if the state counter is unchanged.
        # Since we've updated the data behind their back, we need to
        # change that state counter.
        self._vec.stateIncrease()
        yield self._vec

    @cached_property
    def device_handle(self):
        with self.vec as v:
            return v.getCUDAHandle()

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
        self._iteration_region = kwargs.get('iterate', ALL)
        self._pass_layer_arg = kwargs.get('pass_layer_arg', False)
        # Copy the class variables, so we don't overwrite them
        self._cppargs = dcopy(type(self)._cppargs)
        self._libraries = dcopy(type(self)._libraries)
        self._system_headers = dcopy(type(self)._system_headers)
        self.processed_program = None
        self.args_to_make_global = []
        self.extruded = self._iterset._extruded

        if not kwargs.get('delay', False):
            self.compile()
            self._initialized = True

    @memoize_method
    def grid_size(self, start, end):
        from pymbolic import evaluate

        parameters = {'start': start, 'end': end}
        glens, llens = self.processed_program.get_grid_size_upper_bounds_as_exprs()
        grid_y = 1
        if self.extruded:
            grid_y = glens[1]
        grid = (int(evaluate(glens, parameters)[0]), grid_y)
        block = (int(evaluate(llens, parameters)[0]), 1, 1)

        return grid, block

    @cached_property
    def get_args_marked_for_globals(self):
        const_args_as_globals = tuple(cuda_driver.mem_alloc(arg.nbytes) for arg in
            self.args_to_make_global)
        for arg_gpu, arg in zip(const_args_as_globals,
                self.args_to_make_global):
            cuda_driver.memcpy_htod(arg_gpu, arg)

        evt = cuda_driver.Event()
        evt.record()
        evt.synchronize()

        return const_args_as_globals

    @collective
    def __call__(self, *args):
        grid, block = self.grid_size(args[0], args[1])
        extra_global_args = self.get_args_marked_for_globals
        return self._fun.prepared_call(grid, block, *(args+extra_global_args))

    @cached_property
    def _wrapper_name(self):
        return 'wrap_%s' % self._kernel.name

    @cached_property
    def code_to_compile(self):

        from pyop2.codegen.builder import WrapperBuilder
        from pyop2.codegen.rep2loopy import generate

        builder = WrapperBuilder(iterset=self._iterset, iteration_region=self._iteration_region, pass_layer_to_kernel=self._pass_layer_arg)
        for arg in self._args:
            builder.add_argument(arg)
        builder.set_kernel(self._kernel)

        wrapper = generate(builder)
        code, self.processed_program, self.args_to_make_global = generate_cuda_kernel(wrapper, self.extruded)

        if self._wrapper_name == configuration["cuda_jitmodule_name"]:
            if configuration["load_cuda_kernel"]:
                f = open(configuration["cuda_kernel_name"], "r")
                code = f.read()
                f.close()
            if configuration["dump_cuda_kernel"]:
                f = open(configuration["cuda_kernel_name"], "w")
                f.write(code)
                f.close()

        return code

    @collective
    def compile(self):
        # If we weren't in the cache we /must/ have arguments
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule has no args associated with it, should never happen")

        from pycuda.compiler import SourceModule

        options = ["-use_fast_math"]
        if configuration["cuda_timer_profile"]:
            options.append("-lineinfo")
        func = SourceModule(self.code_to_compile, options=options)
        self._fun = func.get_function(self._wrapper_name)
        self._fun.prepare(self.argtypes+"P"*len(self.args_to_make_global))

        # Blow away everything we don't need any more
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

        type_map = dict([(ctypes.c_void_p, "P"), (ctypes.c_int, "i")])
        argtypes = "".join(type_map[t] for t in argtypes)

        return argtypes


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
            if arg.access is INC:
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
        wrapper_name = "wrap_" + self._kernel.name
        if wrapper_name not in ParLoop.printed:
            print("{0}_BYTES= {1}".format("wrap_" + self._kernel.name, self.nbytes))
            ParLoop.printed.add(wrapper_name)

        return arglist

    @collective
    @timed_function("ParLoopRednEnd")
    def reduction_end(self):
        """End reductions"""
        for arg in self.global_reduction_args:
            arg.reduction_end(self.comm)
        # Finalise global increments
        for tmp, glob in self._reduced_globals.items():
            # These can safely access the _data member directly
            # because lazy evaluation has ensured that any pending
            # updates to glob happened before this par_loop started
            # and the reduction_end on the temporary global pulled
            # data back from the device if necessary.
            # In fact we can't access the properties directly because
            # that forces an infinite loop.
            with tmp.vec as v:
                glob._data += v.array_r

    @cached_property
    def _jitmodule(self):
        return JITModule(self.kernel, self.iterset, *self.args,
                         iterate=self.iteration_region,
                         pass_layer_arg=self._pass_layer_arg)

    @collective
    def _compute(self, part, fun, *arglist):
        if part.size == 0:
            return

        if configuration["cuda_timer"]:
            fun(part.offset, part.offset + part.size, *arglist)  # warm up
            start = cuda_driver.Event()
            end = cuda_driver.Event()
            if configuration["cuda_timer_profile"]:
                cuda_driver.start_profiler()
            start.record()
            for _ in range(configuration["cuda_timer_repeat"]):
                fun(part.offset, part.offset + part.size, *arglist)
            end.record()
            end.synchronize()
            print("{0}_TIME= {1}".format(self._jitmodule._wrapper_name, start.time_till(end)/1000))
            if configuration["cuda_timer_profile"]:
                cuda_driver.stop_profiler()
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


def sept(kernel, extruded=False):
    """
    SEPT := Single Element Per Thread transformation.
    """
    # we don't use shared memory for sept
    cuda_driver.Context.set_cache_config(cuda_driver.func_cache.PREFER_L1)
    pack_consts_to_globals = configuration["cuda_const_as_global"]
    batch_size = configuration["cuda_block_size"]

    if extruded:
        nlayers = configuration["cuda_num_layer"]
        domains = []
        import islpy as isl
        for d in kernel.domains:
            if d.get_dim_name(isl.dim_type.set, 0) == "layer":
                vars = isl.make_zero_and_vars(["layer"])
                nd = vars["layer"].ge_set(vars[0]) & vars["layer"].lt_set(vars[0] + nlayers)
                nd, = nd.get_basic_sets()
                domains.append(nd)
            else:
                domains.append(d)

        from loopy.symbolic import SubstitutionMapper
        from pymbolic.mapper.substitutor import make_subst_func
        from pymbolic.primitives import Variable

        subst_mapper = SubstitutionMapper(
            make_subst_func(dict([(Variable("t0"), 0),
                                  (Variable("t1"), nlayers)])))

        insts = []
        for inst in kernel.instructions:
            if isinstance(inst, loopy.Assignment):
                rhs = subst_mapper(inst.expression)
                inst = inst.copy(expression=rhs)
            insts.append(inst)
        kernel = kernel.copy(domains=domains)
        kernel = kernel.copy(instructions=insts)
        kernel = loopy.assume(kernel, "start < end")

        kernel = loopy.tag_inames(kernel, {"n": "g.0"})
        kernel = loopy.split_iname(kernel, "layer", batch_size, outer_tag="g.1", inner_tag="l.0")

    else:
        kernel = loopy.split_iname(kernel, "n", batch_size, outer_tag="g.0", inner_tag="l.0")
        kernel = loopy.assume(kernel, "{0} mod {1} = 0".format("end", batch_size))
        kernel = loopy.assume(kernel, "exists zz: zz > 0 and {0} = {1}*zz + {2}".format("end", batch_size, "start"))

    # {{{ making consts as globals

    args_to_make_global = []

    if pack_consts_to_globals:
        args_to_make_global = [tv.initializer.flatten()
                for tv in kernel.temporary_variables.values()
                if (tv.initializer is not None
                    and tv.address_space == loopy.AddressSpace.GLOBAL)]

        new_temps = dict((tv.name, tv.copy(initializer=None))
                if (tv.initializer is not None
                    and tv.address_space == loopy.AddressSpace.GLOBAL)
                else (tv.name, tv) for tv in
                kernel.temporary_variables.values())

        kernel = kernel.copy(temporary_variables=new_temps)

    # }}}

    return kernel, args_to_make_global


def _make_tv_array_arg(tv):
    assert tv.address_space != loopy.AddressSpace.PRIVATE
    arg = loopy.ArrayArg(
            name=tv.name,
            dtype=tv.dtype,
            shape=tv.shape,
            dim_tags=tv.dim_tags,
            offset=tv.offset,
            dim_names=tv.dim_names,
            order=tv.order,
            alignment=tv.alignment,
            address_space=tv.address_space,
            is_output_only=not tv.read_only)
    return arg


def transform(kernel, callables_table, ncells_per_block=32,
        nthreads_per_cell=1,
        matvec1_parallelize_across='row', matvec2_parallelize_across='row',
        matvec1_rowtiles=1, matvec1_coltiles=1,
        matvec2_rowtiles=1, matvec2_coltiles=1,
        load_coordinates_to_shared=False,
        load_input_to_shared=False,
        prefetch_tiles=True):
    """
    Matvec1 is the function evaluation part at the quad points.
    Matvec2 is the basis coefficients computation part.
    """

    # {{{ FIXME: Setting names which should be set by TSFC

    quad_iname = 'form_ip'
    output_basis_coeff_temp = 't2'
    input_basis_coeff_temp = 't0'
    scatter_iname = 'i4'
    basis_iname_in_basis_redn = 'form_j'
    quad_iname_in_basis_redn = 'form_ip_basis'
    quad_iname_in_quad_redn = 'form_ip_quad'
    basis_iname_in_quad_redn = 'form_i'
    basis_iname_basis_redn = 'form_j'

    # }}}

    # {{{ sanity checks

    #TODO: Need more sanity checks on the other variables

    assert matvec1_parallelize_across in ['row', 'column']
    assert matvec2_parallelize_across in ['row', 'column']

    # }}}

    # {{{ reading info about the finite element

    nquad = int(loopy.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds('form_ip', constants_only=True).size))
    nbasis = int(loopy.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds('form_j', constants_only=True).size))

    # }}}

    # {{{ tagging the stages of the kernel

    #TODO: Should be interpreted in TSFC 

    new_insns = []

    done_with_jacobi_eval = False
    done_with_quad_init = False
    done_with_quad_reduction = False
    done_with_quad_wrap_up = False
    done_with_basis_reduction = False

    for insn in kernel.instructions:
        if not done_with_jacobi_eval:
            if 'form_ip' in insn.within_inames:
                done_with_jacobi_eval = True

            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["jacobi_eval"])))
                continue
        if not done_with_quad_init:
            if 'form_i' in insn.within_inames:
                done_with_quad_init = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["quad_init"])))
                continue
        if not done_with_quad_reduction:
            if 'form_i' not in insn.within_inames:
                done_with_quad_reduction = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["quad_redn"])))
                continue
        if not done_with_quad_wrap_up:
            if 'basis' in insn.tags:
                done_with_quad_wrap_up = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["quad_wrap_up"])))
                continue
        if not done_with_basis_reduction:
            if 'form_ip' not in insn.within_inames:
                done_with_basis_reduction = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["basis_redn"])))
                continue
        new_insns.append(insn)

    assert done_with_basis_reduction

    kernel = kernel.copy(instructions=new_insns)

    # }}}

    # {{{ privatize temps for function evals and make them LOCAL

    #FIXME: Need these variables from TSFC's metadata
    # This helps to apply transformations separately to the basis part and the
    # quadrature part

    evaluation_variables = (set().union(*[insn.write_dependency_names() for insn in kernel.instructions if 'quad_wrap_up' in insn.tags])
            & set().union(*[insn.read_dependency_names() for insn in kernel.instructions if 'basis' in insn.tags]))

    kernel = loopy.privatize_temporaries_with_inames(kernel, 'form_ip',
            evaluation_variables)
    new_temps = kernel.temporary_variables.copy()
    for eval_var in evaluation_variables:
        new_temps[eval_var] = new_temps[eval_var].copy(
                address_space=loopy.AddressSpace.LOCAL)
    kernel = kernel.copy(temporary_variables=new_temps)

    # }}}

    # {{{ change address space of constants to '__global'

    old_temps = kernel.temporary_variables.copy()
    args_to_make_global = [tv.initializer.flatten() for tv in old_temps.values() if tv.initializer is not None]

    new_temps = dict((tv.name, tv) for tv in old_temps.values() if tv.initializer is None)
    kernel = kernel.copy(
            args=kernel.args+[_make_tv_array_arg(tv) for tv in old_temps.values() if tv.initializer is not None],
            temporary_variables=new_temps)

    # }}}

    #FIXME: Assumes the variable associated with output is 't0'. GENERALIZE THIS!
    kernel = loopy.remove_instructions(kernel, "writes:{} and tag:gather".format(output_basis_coeff_temp))
    kernel = loopy.remove_instructions(kernel, "tag:quad_init")

    from loopy.transform.convert_to_reduction import convert_to_reduction
    kernel = convert_to_reduction(kernel, 'tag:quad_redn', ('form_i', ))
    kernel = convert_to_reduction(kernel, 'tag:basis_redn', ('form_ip', ))

    from loopy.loop import fuse_loop_domains
    kernel = fuse_loop_domains(kernel)

    from loopy.transform.data import remove_unused_axes_in_temporaries
    kernel = remove_unused_axes_in_temporaries(kernel)

    # {{{ remove noops

    noop_insns = set([insn.id for insn in kernel.instructions if
            isinstance(insn, loopy.NoOpInstruction)])
    kernel = loopy.remove_instructions(kernel, noop_insns)

    # }}}

    # Realize CUDA blocks
    kernel = loopy.split_iname(kernel, "n", ncells_per_block,
            outer_iname="iblock", inner_iname="icell")
    #FIXME: Do not use hard-coded inames, this change should also be in TSFC.
    kernel = loopy.rename_iname(kernel, scatter_iname,
            basis_iname_in_basis_redn, True)

    # Duplicate inames to separate transformation logic for quadrature and basis part
    kernel = loopy.duplicate_inames(kernel, quad_iname, "tag:quadrature",
            quad_iname_in_quad_redn)
    kernel = loopy.duplicate_inames(kernel, quad_iname, "tag:basis",
            quad_iname_in_basis_redn)

    if load_coordinates_to_shared:
        #FIXME: Assumes uses the name 't1' for coordinates
        kernel = loopy.privatize_temporaries_with_inames(kernel, 'icell',
                [coords_temp])
        kernel = loopy.assignment_to_subst(kernel, coords_temp)
        raise NotImplementedError()

    if load_input_to_shared:
        #FIXME: Assumes uses the name 't2' for the input basis coeffs
        kernel = loopy.privatize_temporaries_with_inames(kernel, 'icell',
                [input_basis_coeff_temp])
        kernel = loopy.assignment_to_subst(kernel, input_basis_coeff_temp)
        raise NotImplementedError()

    # compute tile lengths
    matvec1_row_tile_length = math.ceil(nquad // matvec1_rowtiles)
    matvec1_col_tile_length = math.ceil(nbasis // matvec1_coltiles)
    matvec2_row_tile_length = math.ceil(nbasis // matvec2_rowtiles)
    matvec2_col_tile_length = math.ceil(nquad // matvec2_coltiles)

    # Splitting for tiles in matvec1
    kernel = loopy.split_iname(kernel, quad_iname_in_quad_redn, matvec1_row_tile_length, outer_iname='irowtile_matvec1')
    kernel = loopy.split_iname(kernel, basis_iname_in_quad_redn, matvec1_col_tile_length, outer_iname='icoltile_matvec1')

    # Splitting for tiles in matvec2
    kernel = loopy.split_iname(kernel, basis_iname_in_basis_redn, matvec2_row_tile_length, outer_iname='irowtile_matvec2')
    kernel = loopy.split_iname(kernel, quad_iname_in_basis_redn, matvec2_col_tile_length, outer_iname='icoltile_matvec2')

    # {{{ Prefetch wizardry

    if prefetch_tiles:
        from loopy.transform.data import add_prefetch_for_single_kernel
        #FIXME: Assuming that in all the constants the one with single axis is
        # the one corresponding to quadrature weights. fix it by passing some
        # metadata from TSFC.
        # FIXME: Sweep inames depends on the parallelization strategies for
        # both the matvecs, that needs to be taken care of.
        const_matrices_names = set([tv.name for tv in old_temps.values() if tv.initializer is not None and len(tv.shape)>1])
        quad_weights, = [tv.name for tv in old_temps.values() if tv.initializer is not None and len(tv.shape) == 1]

        # {{{ Prefetching: QUAD PART

        quad_const_matrices = const_matrices_names & frozenset().union(*[insn.read_dependency_names() for insn in
            kernel.instructions if 'quad_redn' in insn.tags])
        sweep_inames = (quad_iname_in_quad_redn+'_inner',
                basis_iname_in_quad_redn+'_inner')
        fetch_outer_inames = 'iblock,icoltile_matvec1,irowtile_matvec1'

        quad_prefetch_insns = []

        vng = kernel.get_var_name_generator()
        ing = kernel.get_instruction_id_generator()
        quad_temp_names = [vng('quad_cnst_mtrix_prftch') for _ in quad_const_matrices]
        prefetch_inames = [vng("iprftch") for _ in range(2)]
        for temp_name, var_name in zip(quad_temp_names, quad_const_matrices):
            quad_prefetch_insns.append(ing("quad_prftch_insn"))

            kernel = add_prefetch_for_single_kernel(kernel, callables_table,
                    var_name=var_name,
                    sweep_inames=sweep_inames,
                    temporary_address_space=loopy.AddressSpace.LOCAL,
                    dim_arg_names=prefetch_inames,
                    temporary_name=temp_name,
                    compute_insn_id=quad_prefetch_insns[-1],
                    fetch_outer_inames=fetch_outer_inames,
                    default_tag=None,
                    within="tag:quad_redn")

        #FIXME: In order to save on compilation time we are not sticking to
        # coalesced accesses Otherwise we should join the following inames and
        # then split into nthreads_per_cell

        kernel = loopy.split_iname(kernel, prefetch_inames[1],
                nthreads_per_cell, inner_tag="l.0")
        kernel = loopy.tag_inames(kernel, {prefetch_inames[0]: "l.1"})

        # }}}

        # {{{ Prefetching: BASIS PART

        basis_const_matrices = const_matrices_names & frozenset().union(*[insn.read_dependency_names() for insn in
            kernel.instructions if 'basis_redn' in insn.tags])
        basis_temp_names = [vng('basis_cnst_mtrix_prftch') for _ in basis_const_matrices]

        sweep_inames = (basis_iname_in_basis_redn+'_inner',
                quad_iname_in_basis_redn+'_inner')
        fetch_outer_inames = 'iblock,icoltile_matvec2,irowtile_matvec2'

        basis_prefetch_insns = []
        prefetch_inames = [vng("iprftch") for _ in range(2)]
        for temp_name, var_name in zip(basis_temp_names, basis_const_matrices):
            basis_prefetch_insns.append(ing("basis_prftch_insn"))

            kernel = add_prefetch_for_single_kernel(kernel, callables_table,
                    var_name=var_name,
                    sweep_inames=sweep_inames,
                    temporary_address_space=loopy.AddressSpace.LOCAL,
                    dim_arg_names=prefetch_inames,
                    temporary_name=temp_name,
                    compute_insn_id=basis_prefetch_insns[-1],
                    fetch_outer_inames=fetch_outer_inames,
                    default_tag=None,
                    within="tag:basis_redn")

        # See FIXME for the quad part at this point
        kernel = loopy.split_iname(kernel, prefetch_inames[1],
                nthreads_per_cell, inner_tag="l.0")
        kernel = loopy.tag_inames(kernel, {prefetch_inames[0]: "l.1"})

        # }}}

        # {{{ Prefetch: Quad Weights(Set to false now)

        # Unless we load this into the shared memory and do a collective read
        # in a block, this is no good. As the quad weights are accessed only
        # once. So the only way prefetching would help is through a
        # parallelized read.

        prefetch_quad_weights = False

        if prefetch_quad_weights:
            quad_weight_prefetch_insns = []

            if matvec1_parallelize_across == 'row':
                sweep_inames = (quad_iname_in_quad_redn+'_inner_outer', quad_iname_in_quad_redn+'_inner_inner',)
                fetch_outer_inames = 'irowtile_matvec1, icell, iblock'
            else:
                raise NotImplementedError()
            quad_weight_prefetch_insns.append(ing("basis_prftch_insn"))

            kernel = add_prefetch_for_single_kernel(kernel, callables_table,
                    var_name=quad_weights,
                    sweep_inames=sweep_inames,
                    temporary_address_space=loopy.AddressSpace.PRIVATE,
                    temporary_name='cnst_quad_weight_prftch',
                    compute_insn_id=quad_weight_prefetch_insns[-1],
                    fetch_outer_inames=fetch_outer_inames,
                    within="tag:quad_wrap_up")
        # }}}

        # {{{ Adding dependency between the prefetch instructions

        kernel = loopy.add_dependency(kernel,
                " or ".join("id:{}".format(insn_id) for insn_id in
                    basis_prefetch_insns), "tag:quadrature")

        # }}}

        from loopy.transform.data import flatten_variable, absorb_temporary_into
        for var_name in quad_temp_names+basis_temp_names:
            kernel = flatten_variable(kernel, var_name)
        for quad_temp_name, basis_temp_name in zip(quad_temp_names,
                basis_temp_names):
            if (matvec2_row_tile_length*matvec2_col_tile_length >= matvec1_row_tile_length*matvec1_col_tile_length):
                kernel = absorb_temporary_into(kernel, basis_temp_name, quad_temp_name)
            else:
                kernel = absorb_temporary_into(kernel, quad_temp_name, basis_temp_name)

        kernel = loopy.add_dependency(kernel, 'tag:quad_redn', 'id:quad_prftch_insn*')
        kernel = loopy.add_dependency(kernel, 'tag:basis_redn', 'id:basis_prftch_insn*')

        # do not enforce any dependency between the basis reductions and the
        # quadrature reductions.

        kernel = loopy.remove_dependency(kernel, 'tag:quad_redn', 'tag:quad_redn')
        kernel = loopy.remove_dependency(kernel, 'tag:basis_redn', 'tag:basis_redn')
        kernel = loopy.add_dependency(kernel, 'tag:quad_wrap_up', 'tag:quad_redn')

    # }}}

    # {{{ divide matvec1-tile's work across threads

    if matvec1_parallelize_across == 'row':
        kernel = loopy.split_iname(kernel, quad_iname_in_quad_redn+'_inner', nthreads_per_cell, inner_tag="l.0")
    else:
        kernel = loopy.split_iname(kernel, basis_iname_in_quad_redn+'_inner', nthreads_per_cell, inner_tag="l.0")
        kernel = loopy.split_reduction_inward(kernel, basis_iname_in_quad_redn+'_inner_outer')
        kernel = loopy.split_reduction_inward(kernel, basis_iname_in_quad_redn+'_inner_inner')

    # }}}

    # {{{ diving matvec2-tile's work across threads

    if matvec2_parallelize_across == 'row':
        kernel = loopy.split_iname(kernel, basis_iname_in_basis_redn+'_inner', nthreads_per_cell, inner_tag="l.0")
    else:
        kernel = loopy.split_iname(kernel, quad_iname_in_basis_redn+'_inner', nthreads_per_cell, inner_tag="l.0")
        kernel = loopy.split_reduction_inward(kernel, quad_iname_in_basis_redn+'_inner_outer')
        kernel = loopy.split_reduction_inward(kernel, quad_iname_in_basis_redn+'_inner_inner')

    # }}}

    # {{{ mico-optimizations(None implemented yet)

    #FIXME: Need to set the variables 'remove_func_eval_arrays' depending on
    # the input parameters to 'transform'
    # So, currently we don't support this
    # If 'remove_func_eval_arrays' is set True then the following transformations must be performed
    # 1. Use scalars instead of arrays for the variables produced in
    #    quad_wrap_up.
    # 2. Use the same iname for 'form_ip_basis', 'form_ip_quad'
    #
    # These would be the micro-optimization to use less register space for
    # SCPT.

    remove_func_eval_arrays = False
    if remove_func_eval_arrays:
        raise NotImplementedError()

    # Should trigger when matvec1_parallelize_across = 'col'
    # Then no need to put the LHS into shared memory.
    do_not_prefetch_lhs = False
    if do_not_prefetch_lhs:
        raise NotImplementedError()

    # Again for SCPT we need the mico-optimization that we put the constant
    # matrices into the constant memory for broadcasting purposes.

    # }}}

    #FIXME: Need to fix the shape of t0 to whatever portion we are editing.
    # the address space of t0 depends on the parallelization strategy.
    from loopy.preprocess import realize_reduction_for_single_kernel
    kernel = realize_reduction_for_single_kernel(kernel, callables_table)

    # Just translate all the dependencies of form_insn_14 to form_insn_15
    for insn in kernel.instructions:
        if re.match(".*form_insn_14.*", insn.id):
            insn_15_eq = re.sub("(.*)form_insn_14(.*)",
                    "\g<1>form_insn_15\g<2>",
                    insn.id)
            for depends in insn.depends_on:
                if re.match(".*form_insn_14.*", insn.id):
                    kernel = loopy.add_dependency(kernel,
                            "id:{}".format(insn_15_eq),
                            "id:{}".format(depends))
                    kernel = loopy.add_dependency(kernel,
                            "id:{}".format(insn.id),
                            "id:{}".format(re.sub(
                                "(.*)form_insn_14(.*)",
                                "\g<1>form_insn_15\g<2>",
                                depends)))

    if matvec1_parallelize_across == 'row':

        kernel = loopy.privatize_temporaries_with_inames(kernel,
                'form_ip_quad_inner_outer',
                only_var_names=['acc_icoltile_matvec1_form_i_inner',
                    'acc_icoltile_matvec1_form_i_inner_0', 'form_t16', 'form_t17'])
        kernel = loopy.duplicate_inames(kernel, ['form_ip_quad_inner_outer', ],
                within='tag:quad_wrap_up or'
                ' id:red_assign_form_insn_14 or id:red_assign_form_insn_15')

        kernel = loopy.duplicate_inames(kernel,
                ['form_ip_quad_inner_outer'],
                'id:form_insn_14_icoltile_matvec1_form_i_inner_init or id:form_insn_15_icoltile_matvec1_form_i_inner_init')
    else:

        # {{{ make acc_icoltile => local vars

        new_temps = dict((name, tv.copy(address_space=loopy.AddressSpace.LOCAL))
                if name in ['acc_icoltile_matvec1', 'acc_icoltile_matvec1_0']
                else (name, tv) for name, tv in
                kernel.temporary_variables.items())
        kernel = kernel.copy(temporary_variables=new_temps)

        # }}}

        from loopy.transform.batch import save_temporaries_in_loop

        kernel = save_temporaries_in_loop(kernel, 'form_ip_quad_inner', [
            'acc_icoltile_matvec1', 'acc_icoltile_matvec1_0',
            'acc_form_i_inner_inner_0', 'acc_form_i_inner_inner',],
            within="iname:form_ip_quad_inner")

        reduction_assignees = tuple(insn.assignee for insn in kernel.instructions
                if 'quad_redn' in insn.tags)

        # FIXME: These variables should be named using transform addresssing.
        kernel = loopy.assignment_to_subst(kernel, 'neutral_form_i_inner_inner')
        kernel = loopy.assignment_to_subst(kernel, 'neutral_form_i_inner_inner_0')

        kernel = loopy.assignment_to_subst(kernel, "form_t18")
        kernel = loopy.assignment_to_subst(kernel, "form_t19")
        kernel = loopy.assignment_to_subst(kernel, "form_t20")

        for assignee in reduction_assignees:
            kernel = loopy.assignment_to_subst(kernel, assignee.name)

        # {{{ duplicate form_ip_quad_inner in a bunch of equations

        for i in range(int(math.ceil(math.log2(nthreads_per_cell)))):
            kernel = loopy.duplicate_inames(kernel, "form_ip_quad_inner",
                    within="id:red_stage_{0}_form_i_inner_inner_form_insn_14_icoltile_matvec1_update or "
                    "id:red_stage_{0}_form_i_inner_inner_form_insn_15_icoltile_matvec1_update".format(i),)

        kernel = loopy.duplicate_inames(kernel, "form_ip_quad_inner",
                within="id:red_assign_form_insn_14_icoltile_matvec1_update or id:red_assign_form_insn_15_icoltile_matvec1_update")

        kernel = loopy.duplicate_inames(kernel, ["form_ip_quad_inner"],
                new_inames=["form_ip_quad_inner_icoltile_matvec1"],
                within="id:form_insn_14_icoltile_matvec1_init or id:form_insn_15_icoltile_matvec1_init")
        kernel = loopy.split_iname(kernel, "form_ip_quad_inner_icoltile_matvec1",
                nthreads_per_cell, inner_tag="l.0",
                within="id:form_insn_14_icoltile_matvec1_init or id:form_insn_15_icoltile_matvec1_init")

        kernel = loopy.duplicate_inames(kernel, "form_ip_quad_inner",
                new_inames=["form_ip_quad_inner_quad_wrap_up"],
                within="tag:quad_wrap_up")
        kernel = loopy.split_iname(kernel, "form_ip_quad_inner_quad_wrap_up",
                nthreads_per_cell, inner_tag="l.0",
                within="tag:quad_wrap_up")

        # }}}

    if matvec2_parallelize_across == 'row':
        kernel = loopy.privatize_temporaries_with_inames(kernel, 'form_j_inner_outer',
                only_var_names=['acc_icoltile_matvec2_form_ip_basis_inner'])
        kernel = loopy.duplicate_inames(kernel, ['form_j_inner_outer'], within='tag:scatter or'
                ' id:red_assign_form_insn_21')
        kernel = loopy.duplicate_inames(kernel,
                ['form_j_inner_outer'],
                'id:form_insn_21_icoltile_matvec2_form_ip_basis_inner_init')
    else:

        # {{{ make acc_icoltile => local vars

        new_temps = dict((name, tv.copy(address_space=loopy.AddressSpace.LOCAL))
                if name in ['acc_icoltile_matvec2', 't2']
                else (name, tv) for name, tv in
                kernel.temporary_variables.items())
        kernel = kernel.copy(temporary_variables=new_temps)

        # }}}

        kernel = loopy.assignment_to_subst(kernel, 'neutral_form_ip_basis_inner_inner')
        from loopy.transform.batch import save_temporaries_in_loop
        kernel = loopy.save_temporaries_in_loop(kernel,
                'form_j_inner',
                [
                    "acc_icoltile_matvec2",
                    "acc_form_ip_basis_inner_inner"
                    ],
                within="iname:form_j_inner")

        for i in range(int(math.ceil(math.log2(nthreads_per_cell)))):
            kernel = loopy.duplicate_inames(kernel,
                    "form_j_inner",
                    "id:red_stage_{0}_form_ip_basis_inner_inner_form_insn_21_icoltile_matvec2_update".format(i))

        kernel = loopy.duplicate_inames(kernel,
                "form_j_inner",
                within="id:red_assign_form_insn_21_icoltile_matvec2_update")

        kernel = loopy.duplicate_inames(kernel,
                "form_j_inner",
                new_inames=["form_j_inner_icoltile_matvec2_init"],
                within="id:form_insn_21_icoltile_matvec2_init")
        kernel = loopy.split_iname(kernel,
                "form_j_inner_icoltile_matvec2_init", nthreads_per_cell,
                inner_tag="l.0",
                within="id:form_insn_21_icoltile_matvec2_init")

        kernel = loopy.duplicate_inames(kernel,
                "form_j_inner",
                new_inames="form_j_inner_scatter",
                within="tag:scatter or id:red_assign_form_insn_21")
        kernel = loopy.split_iname(kernel,
                "form_j_inner_scatter",
                nthreads_per_cell,
                inner_tag="l.0",
                within="tag:scatter or id:red_assign_form_insn_21")

    kernel = loopy.tag_inames(kernel, "icell:l.1, iblock:g.0")

    return kernel, args_to_make_global


def transpose_maps(kernel):
    print("Caution: The map representation in the kernel is transposed")
    from loopy.kernel.array import FixedStrideArrayDimTag
    from pymbolic import parse

    new_dim_tags = (FixedStrideArrayDimTag(1), FixedStrideArrayDimTag(parse('end')))
    new_args = [arg.copy(dim_tags=new_dim_tags) if arg.name[:3] == 'map' else arg for arg in kernel.args]
    kernel = kernel.copy(args=new_args)
    return kernel


def generate_cuda_kernel(program, extruded=False):
    # Kernel transformations
    args_to_make_global = []
    program = program.copy(target=loopy.CudaTarget())
    kernel = program.root_kernel

    def insn_needs_atomic(insn):
        # updates to global variables are atomic
        assignee_name = insn.assignee.aggregate.name
        return assignee_name in insn.read_dependency_names() and assignee_name not in kernel.temporary_variables

    new_insns = []
    args_marked_for_atomic = set()
    for insn in kernel.instructions:
        if ('scatter' in insn.tags):
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

    # choose the preferred algorithm here
    if program.name == configuration["cuda_jitmodule_name"]:
        if configuration["cuda_strategy"] == "sept":
            kernel, args_to_make_global = sept(kernel, extruded)
            if False:
                # transposing maps
                kernel = transpose_maps(kernel)
        elif configuration["cuda_strategy"] == "general":
            raise NotImplementedError(
                "The general transformation scheme is not fully feature complete.")
            kernel, args_to_make_global = transform(kernel,
                    program.callables_table)
        else:
            raise ValueError("cuda strategy can be 'sept' or 'general'.")
    else:
        kernel, args_to_make_global = sept(kernel, extruded)
        if False:
            # transposing maps
            kernel = transpose_maps(kernel)

    program = program.with_root_kernel(kernel)
    code = loopy.generate_code_v2(program).device_code()

    if program.name == "wrap_pyop2_kernel_uniform_extrusion":
        code = code.replace("inline void pyop2_kernel_uniform_extrusion", "__device__ inline void pyop2_kernel_uniform_extrusion")

    return code, program, args_to_make_global
