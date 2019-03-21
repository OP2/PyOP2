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

from pyop2.datatypes import IntType, as_ctypes
from pyop2 import base
from pyop2 import compilation
from pyop2 import petsc_base
from pyop2.base import par_loop                          # noqa: F401
from pyop2.base import READ, WRITE, RW, INC, MIN, MAX    # noqa: F401
from pyop2.base import ALL
from pyop2.base import Map, MixedMap, DecoratedMap, Sparsity, Halo  # noqa: F401
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
from pyop2.utils import cached_property, get_petsc_dir
from pyop2.configuration import configuration

import loopy
import pycuda
import pycuda.autoinit
import pycuda.driver as cuda_driver
import numpy as np
import islpy
from collections import OrderedDict
from pytools import memoize_method


class Map(Map):

    @cached_property
    def device_handle(self):
        m_gpu = cuda_driver.mem_alloc(int(self.values.nbytes))
        cuda_driver.memcpy_htod(m_gpu, self.values)
        return m_gpu

    @cached_property
    def _kernel_args_(self):
        return (self.device_handle, )


class Arg(Arg):

    pass


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


def thread_transposition(kernel):
    # This might need more looking into.
    nquad = int(loopy.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds('form_ip', constants_only=True).size))
    nbasis = int(loopy.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds('form_j', constants_only=True).size))
    nbatches = 1

    ncells_per_threadblock = int(np.lcm(nquad, nbasis))
    nthreadblocks_per_chunk = 2
    load_within = "tag:gather"
    quad_within = "tag:quadrature"
    basis_within = "tag:basis"

    # {{{ realizing threadblocks

    kernel = loopy.split_iname(kernel, "n",
            nbatches*nthreadblocks_per_chunk * ncells_per_threadblock,
            outer_iname="ichunk")
    kernel = loopy.split_iname(kernel, "n_inner",
            nthreadblocks_per_chunk * ncells_per_threadblock,
            outer_iname="ibatch")
    kernel = loopy.split_iname(kernel, "n_inner_inner",
            ncells_per_threadblock, outer_iname="ithreadblock",
            inner_iname="icell")

    # }}}

    # {{{ remove noops

    noop_insns = set([insn.id for insn in kernel.instructions if
            isinstance(insn, loopy.NoOpInstruction)])
    kernel = loopy.remove_instructions(kernel, noop_insns)

    # }}}

    # {{{ extracting variables that are need to be stored between stages.

    temp_vars = frozenset(kernel.temporary_variables.keys())

    written_in_load = frozenset().union(*[insn.write_dependency_names() for
        insn in kernel.instructions if 'gather' in insn.tags]) & temp_vars

    written_in_quad = frozenset().union(*[insn.write_dependency_names() for
        insn in kernel.instructions if 'quadrature' in insn.tags]) & temp_vars

    read_in_quad = frozenset().union(*[insn.read_dependency_names() for
        insn in kernel.instructions if 'quadrature' in insn.tags]) & temp_vars

    read_in_basis = frozenset().union(*[insn.read_dependency_names() for
        insn in kernel.instructions if 'basis' in insn.tags]) & temp_vars

    # }}}

    # {{{ storing values between the stages

    batch_vars = (written_in_quad & read_in_basis)
    kernel = loopy.save_temporaries_in_loop(kernel, 'form_ip', batch_vars, within='iname:form_ip')

    batch_vars = (written_in_quad & read_in_basis) | (written_in_load & (read_in_basis | read_in_quad))
    kernel = loopy.save_temporaries_in_loop(kernel, 'icell', batch_vars)
    kernel = loopy.save_temporaries_in_loop(kernel, 'ithreadblock', batch_vars)

    # }}}

    # {{{ duplicating inames

    kernel = loopy.duplicate_inames(kernel, ["ichunk", "ithreadblock", "icell"],
            new_inames=["ichunk_load", "ithreadblock_load", "icell_load"],
            within="tag:gather")

    kernel = loopy.duplicate_inames(kernel, ["ichunk", "ithreadblock", "icell"],
            new_inames=["ichunk_quad", "ithreadblock_quad", "icell_quad"],
            within="tag:quadrature")

    kernel = loopy.duplicate_inames(kernel, ["ichunk", "ithreadblock", "icell"],
            new_inames=["ichunk_basis", "ithreadblock_basis", "icell_basis"],
            within="tag:basis or tag:scatter")

    kernel = loopy.duplicate_inames(kernel, ["form_ip"],
            new_inames=["form_ip_quad"], within="tag:quadrature")
    kernel = loopy.duplicate_inames(kernel, ["form_ip"],
            new_inames=["form_ip_basis"], within="tag:basis")

    kernel = loopy.remove_unused_inames(kernel)

    assert not (frozenset(["ithreadblock", "icell", "n", "ichunk"]) & kernel.all_inames())

    # }}}

    # {{{ interpreting the first domain as cuboid

    new_space = kernel.domains[0].get_space()
    new_dom = islpy.BasicSet.universe(new_space)
    for stage in ['load', 'quad', 'basis']:
        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'ithreadblock_%s' % stage: -1,
                    1: nthreadblocks_per_chunk-1}))
        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'ithreadblock_%s' % stage: 1}))

        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'icell_%s' % stage: -1,
                    1: ncells_per_threadblock-1}))
        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'icell_%s' % stage: 1}))

        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'ichunk_%s' % stage:
                    -(nbatches*ncells_per_threadblock*nthreadblocks_per_chunk),
                    'start': -1, 'end': 1, 1: -1}))
        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'ichunk_%s' % stage: 1}))

    new_dom = new_dom.add_constraint(
            islpy.Constraint.ineq_from_names(new_space, {
                'ibatch': -1,
                1: nbatches-1}))
    new_dom = new_dom.add_constraint(
            islpy.Constraint.ineq_from_names(new_space, {
                'ibatch': 1}))

    kernel = kernel.copy(domains=[new_dom]+kernel.domains[1:])

    # }}}

    # {{{ coalescing the entire domain forest

    new_space = kernel.domains[0].get_space()
    pos = kernel.domains[0].n_dim()
    for dom in kernel.domains[1:]:
        # product of all the spaces
        for dim_name, (dim_type, _) in dom.get_space().get_var_dict().items():
            assert dim_type == 3
            new_space = new_space.add_dims(dim_type, 1)
            new_space = new_space.set_dim_name(dim_type, pos, dim_name)
            pos += 1

    new_domain = islpy.BasicSet.universe(new_space)
    for dom in kernel.domains[:]:
        for constraint in dom.get_constraints():
            if constraint.is_equality():
                new_domain = (
                        new_domain.add_constraint(
                            islpy.Constraint.eq_from_names(new_space,
                                constraint.get_coefficients_by_name())))
            else:
                new_domain = (
                        new_domain.add_constraint(
                            islpy.Constraint.ineq_from_names(new_space,
                                constraint.get_coefficients_by_name())))

    kernel = kernel.copy(domains=[new_domain])

    # }}}

    # {{{ adding barriers

    kernel = loopy.add_barrier(kernel, "tag:gather",
            "tag:quadrature", synchronization_kind='local')
    kernel = loopy.add_barrier(kernel, "tag:quadrature",
            "tag:basis", synchronization_kind='local')
    if nbatches > 1:
        kernel = kernel.copy(
                instructions=kernel.instructions+[
                    loopy.BarrierInstruction('closing_barr',
                        synchronization_kind="local", mem_kind="local",
                        depends_on=frozenset([insn.id for insn in
                            kernel.instructions]),
                        depends_on_is_final=True,
                        within_inames=frozenset(["ibatch", "ichunk_basis"]))])

    # }}}

    # {{{ re-distributing the gather work

    kernel = loopy.join_inames(kernel, ["ithreadblock_load", "icell_load"],
            "local_id1", within=load_within)

    # }}}

    # {{{ re-distributing the quadrature evaluation work

    kernel = loopy.split_iname(kernel, "icell_quad", nquad,
            inner_iname="inner_quad_cell", outer_iname="outer_quad_cell",
            within=quad_within)

    kernel = loopy.join_inames(kernel, ["ithreadblock_quad", "outer_quad_cell",
        "form_ip_quad"], "local_id2", within=quad_within)

    # }}}

    # {{{ re-distributing the basis coeffs evaluation work

    # this is the one which we need to take care about.
    basis_inames = (set(kernel.all_inames()).intersection(*[insn.within_inames
        for insn in kernel.instructions if ('basis' in insn.tags)])
        - set(["ithreadblock_basis", "icell_basis", "ichunk_basis",
          "form_ip_basis", "ibatch"]))

    assert len(basis_inames) == 1
    basis_iname = basis_inames.pop()

    scatter_inames = (set(kernel.all_inames()).intersection(*[insn.within_inames
        for insn in kernel.instructions if 'scatter' in insn.tags])
        - set(["ithreadblock_basis", "icell_basis", "ichunk_basis",
          "ibatch"]))
    assert len(scatter_inames) == 1
    scatter_iname = scatter_inames.pop()

    kernel = loopy.split_iname(kernel, "icell_basis", nbasis,
            inner_iname="inner_basis_cell", outer_iname="outer_basis_cell",
            within=basis_within+" or tag:scatter")
    kernel = loopy.join_inames(kernel, ["ithreadblock_basis", "outer_basis_cell",
        basis_iname], "local_id3", within=basis_within)
    kernel = loopy.join_inames(kernel, ["ithreadblock_basis", "outer_basis_cell",
        scatter_iname], "local_id4", within="tag:scatter")

    # }}}

    # tagging inames
    kernel = loopy.tag_inames(kernel, {
        "ichunk_load":      "g.0",
        "ichunk_quad":      "g.0",
        "ichunk_basis":     "g.0",
        "local_id1":        "l.0",
        "local_id2":        "l.0",
        "local_id3":        "l.0",
        "local_id4":        "l.0",
        })

    new_temps = dict((tv.name,
        tv.copy(address_space=loopy.AddressSpace.LOCAL)) if tv.name in batch_vars
        else (tv.name, tv) for tv in kernel.temporary_variables.values())
    kernel = kernel.copy(temporary_variables=new_temps)
    no_sync_with = frozenset([(insn.id, 'local') for insn in
        kernel.instructions])
    new_insns = [insn.copy(no_sync_with=no_sync_with) for
        insn in kernel.instructions]
    new_insns = [insn.copy(within_inames=frozenset(['ibatch'])) if
            isinstance(insn, loopy.BarrierInstruction) else insn for insn in
            new_insns]
    kernel = kernel.copy(instructions=new_insns)
    return kernel


def sept(kernel, extruded=False):
    # we don't use shared memory for sept
    cuda_driver.Context.set_cache_config(cuda_driver.func_cache.PREFER_L1)
    args_to_make_global = []
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


def gcd_tt(kernel):

    # Experiment with these numbers to get speedup
    copy_consts_to_shared = False
    pack_consts_to_globals = configuration["cuda_const_as_global"]
    ncells_per_batch = 32
    nbatches_per_chunk = 1
    args_to_make_global = []

    nquad = int(loopy.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds('form_ip', constants_only=True).size))
    nbasis = int(loopy.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds('form_j', constants_only=True).size))

    nthreads_per_cell = int(np.gcd(nquad, nbasis))
    # should be the minimum number needed to make `nthreads_per_cell` multiple of 32
    load_within = "tag:gather"
    quad_within = "tag:quadrature"
    basis_within = "tag:basis"
    n_lids = 0


    # {{{ feeding the constants into shared memory

    if copy_consts_to_shared:
        from pymbolic.primitives import Variable, Subscript

        new_temps = {}
        var_name_generator = kernel.get_var_name_generator()
        insn_id_generator = kernel.get_instruction_id_generator()
        new_insns = []
        new_domains = []
        priorities = []

        for tv in kernel.temporary_variables.values():
            if tv.address_space == loopy.AddressSpace.GLOBAL:
                old_tv = tv.copy()

                old_name = old_tv.name
                new_name = var_name_generator(based_on="const_"+tv.name)

                inames = tuple(var_name_generator(based_on="icopy") for _
                        in tv.shape)
                priorities.append(inames)
                var_inames = tuple(Variable(iname) for iname in inames)
                new_temps[new_name] = old_tv.copy(name=new_name)
                new_insns.append(loopy.Assignment(
                    id=insn_id_generator(based_on="insn_copy"),
                    assignee=Subscript(Variable(old_name),
                    var_inames), expression=Subscript(Variable(new_name), var_inames),
                    within_inames=frozenset(inames),
                    tags=frozenset(["init_shared"])))
                space = islpy.Space.create_from_names(kernel.isl_context, set=inames)
                domain = islpy.BasicSet.universe(space)
                from loopy.isl_helpers import make_slab
                for iname, axis_len in zip(inames, tv.shape):
                    domain &= make_slab(space, iname, 0, axis_len)
                new_domains.append(domain)
                new_temps[old_name] = old_tv.copy(
                        read_only=False,
                        initializer=None,
                        address_space=loopy.AddressSpace.LOCAL)
            else:
                new_temps[tv.name] = tv

        kernel = kernel.copy(temporary_variables=new_temps,
                instructions=kernel.instructions+new_insns,
                domains=kernel.domains+new_domains)
        kernel = loopy.add_dependency(kernel, "tag:gather", "tag:init_shared")
        for priority in priorities:
            kernel = loopy.prioritize_loops(kernel, ",".join(priority))

        for insn in kernel.instructions:
            if "init_shared" in insn.tags:
                inames_to_merge = insn.within_inames
                # maybe need to split to be valid for all cases?
                for priority in kernel.loop_priority:
                    if frozenset(priority) == inames_to_merge:
                        inames_to_merge = priority
                        break

                inames_to_merge = list(inames_to_merge)

                kernel = loopy.join_inames(kernel, inames_to_merge,
                        "aux_local_id%d" % n_lids, within="id:%s" % insn.id)
                kernel = loopy.split_iname(kernel, "aux_local_id%d" % n_lids,
                        ncells_per_batch*nthreads_per_cell, within="id:%s" % insn.id,
                        inner_iname=("local_id%d" % n_lids))
                n_lids += 1

    # }}}

    # {{{ making consts as globals

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

    # {{{ realizing batches

    kernel = loopy.split_iname(kernel, "n",
            nbatches_per_chunk * ncells_per_batch,
            outer_iname="ichunk",)
    kernel = loopy.split_iname(kernel, "n_inner", ncells_per_batch,
            inner_iname="icell", outer_iname="ibatch")

    # }}}

    # {{{ remove noops

    noop_insns = set([insn.id for insn in kernel.instructions if
            isinstance(insn, loopy.NoOpInstruction)])
    kernel = loopy.remove_instructions(kernel, noop_insns)

    # }}}

    # {{{ extracting variables that are need to be stored between stages.

    temp_vars = frozenset(kernel.temporary_variables.keys())

    written_in_load = frozenset().union(*[insn.write_dependency_names() for
        insn in kernel.instructions if 'gather' in insn.tags]) & temp_vars

    written_in_quad = frozenset().union(*[insn.write_dependency_names() for
        insn in kernel.instructions if 'quadrature' in insn.tags]) & temp_vars

    read_in_quad = frozenset().union(*[insn.read_dependency_names() for
        insn in kernel.instructions if 'quadrature' in insn.tags]) & temp_vars

    read_in_basis = frozenset().union(*[insn.read_dependency_names() for
        insn in kernel.instructions if 'basis' in insn.tags]) & temp_vars

    # }}}

    # {{{ remove unnecessary dependencies on quadrature instructions

    vars_not_neeeded_in_quad = written_in_load - read_in_quad

    # so lets just write in the basis part
    written_in_load = written_in_load - vars_not_neeeded_in_quad

    insns_to_be_added_in_basis = frozenset([insn.id for insn in
        kernel.instructions if insn.write_dependency_names()
        & vars_not_neeeded_in_quad and 'gather' in insn.tags])

    def _remove_unnecessary_deps_on_load(insn):
        return insn.copy(depends_on=insn.depends_on - insns_to_be_added_in_basis)

    kernel = loopy.map_instructions(kernel, quad_within,
            _remove_unnecessary_deps_on_load)

    def _add_unnecessary_instructions_to_basis(insn):
        if insn.id in insns_to_be_added_in_basis:
            return insn.copy(tags=insn.tags-frozenset(["gather"])
                | frozenset(["basis", "basis_init"]))
        return insn
    kernel = loopy.map_instructions(kernel, "id:*",
            _add_unnecessary_instructions_to_basis)

    # }}}

    # {{{ storing values between the stages

    batch_vars = (written_in_quad & read_in_basis)
    kernel = loopy.save_temporaries_in_loop(kernel, 'form_ip', batch_vars, within='iname:form_ip')

    batch_vars = (written_in_quad & read_in_basis) | (written_in_load & (read_in_basis | read_in_quad))
    kernel = loopy.save_temporaries_in_loop(kernel, 'icell', batch_vars,
            within="not tag:init_shared")

    # }}}

    # {{{ duplicating inames

    kernel = loopy.duplicate_inames(kernel, ["ichunk", "icell"],
            new_inames=["ichunk_load", "icell_load"],
            within="tag:gather")

    kernel = loopy.duplicate_inames(kernel, ["ichunk", "icell"],
            new_inames=["ichunk_quad", "icell_quad"],
            within="tag:quadrature")

    kernel = loopy.duplicate_inames(kernel, ["ichunk", "icell"],
            new_inames=["ichunk_basis", "icell_basis"],
            within="tag:basis or tag:scatter")

    kernel = loopy.duplicate_inames(kernel, ["form_ip"],
            new_inames=["form_ip_quad"], within="tag:quadrature")
    kernel = loopy.duplicate_inames(kernel, ["form_ip"],
            new_inames=["form_ip_basis"], within="tag:basis")

    kernel = loopy.remove_unused_inames(kernel)

    # assert not (frozenset(["ithreadblock", "icell", "n", "ichunk"]) & kernel.all_inames())

    # }}}

    # {{{ interpreting the first domain as cuboid

    new_space = kernel.domains[0].get_space()
    new_dom = islpy.BasicSet.universe(new_space)
    for stage in ['load', 'quad', 'basis']:
        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'icell_%s' % stage: -1,
                    1: ncells_per_batch-1}))
        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'icell_%s' % stage: 1}))

        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'ichunk_%s' % stage:
                    -(nbatches_per_chunk*ncells_per_batch),
                    'ibatch':
                    -(ncells_per_batch),
                    'icell_%s' % stage:
                    -1,
                    'start': -1, 'end': 1, 1: -1}))
        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'ichunk_%s' % stage: 1}))

    new_dom = new_dom.add_constraint(
            islpy.Constraint.ineq_from_names(new_space, {
                'ibatch': -1,
                1: nbatches_per_chunk-1}))
    new_dom = new_dom.add_constraint(
            islpy.Constraint.ineq_from_names(new_space, {
                'ibatch': 1}))

    kernel = kernel.copy(domains=[new_dom]+kernel.domains[1:])

    # }}}

    # {{{ coalescing the entire domain forest

    new_space = kernel.domains[0].get_space()
    pos = kernel.domains[0].n_dim()
    for dom in kernel.domains[1:]:
        # product of all the spaces
        for dim_name, (dim_type, _) in dom.get_space().get_var_dict().items():
            assert dim_type == 3
            new_space = new_space.add_dims(dim_type, 1)
            new_space = new_space.set_dim_name(dim_type, pos, dim_name)
            pos += 1

    new_domain = islpy.BasicSet.universe(new_space)
    for dom in kernel.domains[:]:
        for constraint in dom.get_constraints():
            if constraint.is_equality():
                new_domain = (
                        new_domain.add_constraint(
                            islpy.Constraint.eq_from_names(new_space,
                                constraint.get_coefficients_by_name())))
            else:
                new_domain = (
                        new_domain.add_constraint(
                            islpy.Constraint.ineq_from_names(new_space,
                                constraint.get_coefficients_by_name())))

    kernel = kernel.copy(domains=[new_domain])

    # }}}

    kernel = loopy.add_barrier(kernel, "tag:gather",
            "tag:quadrature", synchronization_kind='local')
    kernel = loopy.add_barrier(kernel, "tag:quadrature",
            "tag:basis", synchronization_kind='local')

    # {{{ re-distributing the gather work

    for insn in kernel.instructions:
        if "gather" in insn.tags:
            inames_to_merge = (insn.within_inames
                    - frozenset(["ichunk_load", "icell_load", "ibatch"]))
            # maybe need to split to be valid for all cases?
            for priority in kernel.loop_priority:
                if frozenset(priority) == inames_to_merge:
                    inames_to_merge = priority
                    break

            inames_to_merge = list(inames_to_merge)

            kernel = loopy.join_inames(kernel, inames_to_merge, "aux_local_id%d" %
                    n_lids, within="id:%s" % insn.id)
            kernel = loopy.split_iname(kernel, "aux_local_id%d" % n_lids,
                    nthreads_per_cell, within="id:%s" % insn.id)
            kernel = loopy.join_inames(kernel, ["icell_load",
                "aux_local_id%d_inner"
                % n_lids], "local_id%d" % n_lids, within="id:%s" % insn.id)
            kernel = loopy.tag_inames(kernel, (("aux_local_id%d_outer", "unr"),),
                    ignore_nonexistent=True)
            n_lids += 1

    # }}}

    # {{{ re-distributing the quadrature evaluation work

    kernel = loopy.split_iname(kernel, "form_ip_quad", nthreads_per_cell)

    kernel = loopy.join_inames(kernel, ["icell_quad",
        "form_ip_quad_inner"], "local_id%d" % n_lids, within=quad_within)
    n_lids += 1

    # }}}

    # {{{ re-distributing the basis coeffs evaluation work

    # this is the one which we need to take care about.
    basis_inames = (set(kernel.all_inames()).intersection(*[insn.within_inames
        for insn in kernel.instructions if 'basis' in insn.tags
        and 'basis_init' not in insn.tags])
        - set(["ithreadblock_basis", "icell_basis", "ichunk_basis",
          "form_ip_basis", "ibatch"]))

    assert len(basis_inames) == 1
    basis_iname = basis_inames.pop()

    scatter_inames = (set(kernel.all_inames()).intersection(*[insn.within_inames
        for insn in kernel.instructions if 'scatter' in insn.tags])
        - set(["ithreadblock_basis", "icell_basis", "ichunk_basis",
          "ibatch"]))
    assert len(scatter_inames) == 1
    scatter_iname = scatter_inames.pop()

    kernel = loopy.split_iname(kernel, basis_iname, nthreads_per_cell,
            inner_iname="basis_aux_lid0", within=basis_within)
    kernel = loopy.split_iname(kernel, scatter_iname, nthreads_per_cell,
            inner_iname="basis_aux_lid1", within="tag:scatter")

    kernel = loopy.join_inames(kernel, ["icell_basis", "basis_aux_lid0"],
            "local_id%d" % n_lids, within=basis_within)
    kernel = loopy.join_inames(kernel, ["icell_basis", "basis_aux_lid1"],
            "local_id%d" % (n_lids+1), within="tag:scatter")
    n_lids += 2

    kernel = loopy.rename_iname(kernel, scatter_iname+"_outer",
            basis_iname+"_outer", existing_ok=True, within="tag:scatter")

    from loopy.transform.make_scalar import (
            make_scalar, remove_invariant_inames)
    # FIXME: generalize this
    kernel = make_scalar(kernel, 't0')

    # }}}

    new_insns = [insn.copy(within_inames=frozenset(['ibatch'])) if
            isinstance(insn, loopy.BarrierInstruction) else insn for insn in
            kernel.instructions]
    kernel = kernel.copy(instructions=new_insns)

    iname_tags = {
        "ichunk_load":      "g.0",
        "ichunk_quad":      "g.0",
        "ichunk_basis":     "g.0",
        }
    for i in range(n_lids):
        iname_tags["local_id%d" % i] = "l.0"
        iname_tags["aux_local_id%d_outer" % i] = "ilp"

    kernel = loopy.tag_inames(kernel, iname_tags, ignore_nonexistent=True)
    kernel = remove_invariant_inames(kernel)
    kernel = loopy.add_nosync(kernel, 'local', 'tag:basis', 'tag:scatter')
    return (loopy.remove_unused_inames(kernel).copy(loop_priority=frozenset()),
            args_to_make_global)


def gcd_tt_simple(kernel):
    
    cuda_driver.Context.set_cache_config(cuda_driver.func_cache.PREFER_SHARED)
    
    # Experiment with these numbers to get speedup
    pack_consts_to_globals = configuration["cuda_const_as_global"]
    ncells_per_batch = configuration["cuda_batch_per_block"]
    nbatches_per_chunk = 1
    args_to_make_global = []

    nquad = int(loopy.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds('form_ip', constants_only=True).size))
    nbasis = int(loopy.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds('form_j', constants_only=True).size))

    nthreads_per_cell = int(np.gcd(nquad, nbasis))
    # should be the minimum number needed to make `nthreads_per_cell` multiple of 32
    load_within = "tag:gather"
    quad_within = "tag:quadrature"
    basis_within = "tag:basis"
    n_lids = 0


    # {{{ making consts as globals

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

    # {{{ realizing batches

    kernel = loopy.split_iname(kernel, "n",
            nbatches_per_chunk * ncells_per_batch,
            outer_iname="ichunk",)
    kernel = loopy.split_iname(kernel, "n_inner", ncells_per_batch,
            inner_iname="icell", outer_iname="ibatch")

    # }}}

    # {{{ remove noops

    noop_insns = set([insn.id for insn in kernel.instructions if
            isinstance(insn, loopy.NoOpInstruction)])
    kernel = loopy.remove_instructions(kernel, noop_insns)

    # }}}

    # {{{ extracting variables that are need to be stored between stages.

    temp_vars = frozenset(kernel.temporary_variables.keys())

    written_in_load = frozenset().union(*[insn.write_dependency_names() for
        insn in kernel.instructions if 'gather' in insn.tags]) & temp_vars

    written_in_quad = frozenset().union(*[insn.write_dependency_names() for
        insn in kernel.instructions if 'quadrature' in insn.tags]) & temp_vars

    read_in_quad = frozenset().union(*[insn.read_dependency_names() for
        insn in kernel.instructions if 'quadrature' in insn.tags]) & temp_vars

    read_in_basis = frozenset().union(*[insn.read_dependency_names() for
        insn in kernel.instructions if 'basis' in insn.tags]) & temp_vars

    # }}}

    # {{{ remove unnecessary dependencies on quadrature instructions

    vars_not_neeeded_in_quad = written_in_load - read_in_quad

    # so lets just write in the basis part
    written_in_load = written_in_load - vars_not_neeeded_in_quad

    insns_to_be_added_in_basis = frozenset([insn.id for insn in
        kernel.instructions if insn.write_dependency_names()
        & vars_not_neeeded_in_quad and 'gather' in insn.tags])

    def _remove_unnecessary_deps_on_load(insn):
        return insn.copy(depends_on=insn.depends_on - insns_to_be_added_in_basis)

    kernel = loopy.map_instructions(kernel, quad_within,
            _remove_unnecessary_deps_on_load)

    def _add_unnecessary_instructions_to_basis(insn):
        if insn.id in insns_to_be_added_in_basis:
            return insn.copy(tags=insn.tags-frozenset(["gather"])
                | frozenset(["basis", "basis_init"]))
        return insn
    kernel = loopy.map_instructions(kernel, "id:*",
            _add_unnecessary_instructions_to_basis)

    # }}}

    # {{{ storing values between the stages

    batch_vars = (written_in_quad & read_in_basis)
    kernel = loopy.save_temporaries_in_loop(kernel, 'form_ip', batch_vars, within='iname:form_ip')

    batch_vars = (written_in_quad & read_in_basis) | (written_in_load & (read_in_basis | read_in_quad))
    kernel = loopy.save_temporaries_in_loop(kernel, 'icell', batch_vars,
            within="not tag:init_shared")

    # }}}

    # {{{ duplicating inames

    kernel = loopy.duplicate_inames(kernel, ["ichunk", "icell"],
            new_inames=["ichunk_load", "icell_load"],
            within="tag:gather")

    kernel = loopy.duplicate_inames(kernel, ["ichunk", "icell"],
            new_inames=["ichunk_quad", "icell_quad"],
            within="tag:quadrature")

    kernel = loopy.duplicate_inames(kernel, ["ichunk", "icell"],
            new_inames=["ichunk_basis", "icell_basis"],
            within="tag:basis or tag:scatter")

    kernel = loopy.duplicate_inames(kernel, ["form_ip"],
            new_inames=["form_ip_quad"], within="tag:quadrature")
    kernel = loopy.duplicate_inames(kernel, ["form_ip"],
            new_inames=["form_ip_basis"], within="tag:basis")

    kernel = loopy.remove_unused_inames(kernel)

    # assert not (frozenset(["ithreadblock", "icell", "n", "ichunk"]) & kernel.all_inames())

    # }}}

    # {{{ interpreting the first domain as cuboid

    new_space = kernel.domains[0].get_space()
    new_dom = islpy.BasicSet.universe(new_space)
    for stage in ['load', 'quad', 'basis']:
        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'icell_%s' % stage: -1,
                    1: ncells_per_batch-1}))
        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'icell_%s' % stage: 1}))

        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'ichunk_%s' % stage:
                    -(nbatches_per_chunk*ncells_per_batch),
                    'ibatch':
                    -(ncells_per_batch),
                    'icell_%s' % stage:
                    -1,
                    'start': -1, 'end': 1, 1: -1}))
        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'ichunk_%s' % stage: 1}))

    new_dom = new_dom.add_constraint(
            islpy.Constraint.ineq_from_names(new_space, {
                'ibatch': -1,
                1: nbatches_per_chunk-1}))
    new_dom = new_dom.add_constraint(
            islpy.Constraint.ineq_from_names(new_space, {
                'ibatch': 1}))

    kernel = kernel.copy(domains=[new_dom]+kernel.domains[1:])

    # }}}

    # {{{ coalescing the entire domain forest

    new_space = kernel.domains[0].get_space()
    pos = kernel.domains[0].n_dim()
    for dom in kernel.domains[1:]:
        # product of all the spaces
        for dim_name, (dim_type, _) in dom.get_space().get_var_dict().items():
            assert dim_type == 3
            new_space = new_space.add_dims(dim_type, 1)
            new_space = new_space.set_dim_name(dim_type, pos, dim_name)
            pos += 1

    new_domain = islpy.BasicSet.universe(new_space)
    for dom in kernel.domains[:]:
        for constraint in dom.get_constraints():
            if constraint.is_equality():
                new_domain = (
                        new_domain.add_constraint(
                            islpy.Constraint.eq_from_names(new_space,
                                constraint.get_coefficients_by_name())))
            else:
                new_domain = (
                        new_domain.add_constraint(
                            islpy.Constraint.ineq_from_names(new_space,
                                constraint.get_coefficients_by_name())))

    kernel = kernel.copy(domains=[new_domain])

    # }}}

    kernel = loopy.add_barrier(kernel, "tag:gather",
            "tag:quadrature", synchronization_kind='local')
    kernel = loopy.add_barrier(kernel, "tag:quadrature",
            "tag:basis", synchronization_kind='local')

    # {{{ re-distributing the gather work

    for insn in kernel.instructions:
        if "gather" in insn.tags:
            inames_to_merge = (insn.within_inames
                    - frozenset(["ichunk_load", "icell_load", "ibatch"]))
            # maybe need to split to be valid for all cases?
            for priority in kernel.loop_priority:
                if frozenset(priority) == inames_to_merge:
                    inames_to_merge = priority
                    break

            inames_to_merge = list(inames_to_merge)

            kernel = loopy.join_inames(kernel, inames_to_merge, "aux_local_id%d" %
                    n_lids, within="id:%s" % insn.id)
            kernel = loopy.split_iname(kernel, "aux_local_id%d" % n_lids,
                    nthreads_per_cell, within="id:%s" % insn.id)
            kernel = loopy.join_inames(kernel, ["icell_load",
                "aux_local_id%d_inner"
                % n_lids], "local_id%d" % n_lids, within="id:%s" % insn.id)
            kernel = loopy.tag_inames(kernel, (("aux_local_id%d_outer", "unr"),),
                    ignore_nonexistent=True)
            n_lids += 1

    # }}}

    # {{{ re-distributing the quadrature evaluation work

    kernel = loopy.split_iname(kernel, "form_ip_quad", nthreads_per_cell)

    kernel = loopy.join_inames(kernel, ["icell_quad",
        "form_ip_quad_inner"], "local_id%d" % n_lids, within=quad_within)
    n_lids += 1

    # }}}

    # {{{ re-distributing the basis coeffs evaluation work

    # this is the one which we need to take care about.
    basis_inames = (set(kernel.all_inames()).intersection(*[insn.within_inames
        for insn in kernel.instructions if 'basis' in insn.tags
        and 'basis_init' not in insn.tags])
        - set(["ithreadblock_basis", "icell_basis", "ichunk_basis",
          "form_ip_basis", "ibatch"]))

    assert len(basis_inames) == 1
    basis_iname = basis_inames.pop()

    scatter_inames = (set(kernel.all_inames()).intersection(*[insn.within_inames
        for insn in kernel.instructions if 'scatter' in insn.tags])
        - set(["ithreadblock_basis", "icell_basis", "ichunk_basis",
          "ibatch"]))
    assert len(scatter_inames) == 1
    scatter_iname = scatter_inames.pop()

    kernel = loopy.split_iname(kernel, basis_iname, nthreads_per_cell,
            inner_iname="basis_aux_lid0", within=basis_within)
    kernel = loopy.split_iname(kernel, scatter_iname, nthreads_per_cell,
            inner_iname="basis_aux_lid1", within="tag:scatter")

    kernel = loopy.join_inames(kernel, ["icell_basis", "basis_aux_lid0"],
            "local_id%d" % n_lids, within=basis_within)
    kernel = loopy.join_inames(kernel, ["icell_basis", "basis_aux_lid1"],
            "local_id%d" % (n_lids+1), within="tag:scatter")
    n_lids += 2

    kernel = loopy.rename_iname(kernel, scatter_iname+"_outer",
            basis_iname+"_outer", existing_ok=True, within="tag:scatter")

    from loopy.transform.make_scalar import (
            make_scalar, remove_invariant_inames)
    # FIXME: generalize this
    kernel = make_scalar(kernel, 't0')

    # }}}

    new_insns = [insn.copy(within_inames=frozenset(['ibatch'])) if
            isinstance(insn, loopy.BarrierInstruction) else insn for insn in
            kernel.instructions]
    kernel = kernel.copy(instructions=new_insns)

    iname_tags = {
        "ichunk_load":      "g.0",
        "ichunk_quad":      "g.0",
        "ichunk_basis":     "g.0",
        }
    for i in range(n_lids):
        iname_tags["local_id%d" % i] = "l.0"
        iname_tags["aux_local_id%d_outer" % i] = "ilp"

    kernel = loopy.tag_inames(kernel, iname_tags, ignore_nonexistent=True)
    kernel = remove_invariant_inames(kernel)
    kernel = loopy.add_nosync(kernel, 'local', 'tag:basis', 'tag:scatter')
    return (loopy.remove_unused_inames(kernel).copy(loop_priority=frozenset()),
            args_to_make_global)


def tiled_gcd_tt(kernel, callables_table):

    # {{{ reading info about the finite element

    nquad = int(loopy.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds('form_ip', constants_only=True).size))
    nbasis = int(loopy.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds('form_j', constants_only=True).size))

    nthreads_per_cell = int(np.gcd(nquad, nbasis))

    # }}}

    # {{{ performance params

    copy_consts_to_shared = True
    pack_consts_to_globals = configuration["cuda_const_as_global"]
    tiled_access_to_the_vars = True
    # we can tile only if variables are copied to shared memory
    assert not tiled_access_to_the_vars or copy_consts_to_shared
    ncells_per_chunk = 32
    tile_quad = nthreads_per_cell
    tile_basis = nthreads_per_cell

    # }}}

    args_to_make_global = []  # by default not imposing extra global args
    n_lids = 0  # number of local ids, acts as a counter for the var_name_generation

    # {{{ remove noops

    noop_insns = set([insn.id for insn in kernel.instructions if
            isinstance(insn, loopy.NoOpInstruction)])
    kernel = loopy.remove_instructions(kernel, noop_insns)

    # }}}

    # {{{ identifying the inames used for loop over basis indices

    basis_inames = (set(kernel.all_inames()).intersection(*[insn.within_inames
        for insn in kernel.instructions if 'basis' in insn.tags])
        - set(["n", "form_ip"]))

    assert len(basis_inames) == 1
    basis_iname = basis_inames.pop()

    scatter_inames = (set(kernel.all_inames()).intersection(*[insn.within_inames
        for insn in kernel.instructions if 'scatter' in insn.tags])
        - set(["n"]))
    assert len(scatter_inames) == 1
    scatter_iname = scatter_inames.pop()

    # }}}

    # {{{ feeding the constants into shared memory

    consts_precomputed = set()

    if copy_consts_to_shared:
        # Add temporaries, instructions and domains for copying the constant variables
        from pymbolic.primitives import Variable, Subscript

        new_temps = {}
        var_name_generator = kernel.get_var_name_generator()
        insn_id_generator = kernel.get_instruction_id_generator()
        new_insns = []
        new_domains = []
        priorities = []

        for tv in kernel.temporary_variables.values():
            if tv.address_space == loopy.AddressSpace.GLOBAL:
                # if address space of temporary is GLOBAL, copy to a variables
                old_tv = tv.copy()

                old_name = old_tv.name
                consts_precomputed.add(old_name)
                new_name = var_name_generator(based_on="const_"+tv.name)

                inames = tuple(var_name_generator(based_on="icopy") for _
                        in tv.shape)
                priorities.append(inames)
                var_inames = tuple(Variable(iname) for iname in inames)
                new_temps[new_name] = old_tv.copy(name=new_name)
                new_insns.append(loopy.Assignment(
                    id=insn_id_generator(based_on="insn_copy"),
                    assignee=Subscript(Variable(old_name),
                    var_inames), expression=Subscript(Variable(new_name), var_inames),
                    within_inames=frozenset(inames),
                    tags=frozenset(["init_shared"])))
                space = islpy.Space.create_from_names(kernel.isl_context, set=inames)
                domain = islpy.BasicSet.universe(space)
                from loopy.isl_helpers import make_slab
                for iname, axis_len in zip(inames, tv.shape):
                    domain &= make_slab(space, iname, 0, axis_len)
                new_domains.append(domain)
                new_temps[old_name] = old_tv.copy(
                        read_only=False,
                        initializer=None,
                        address_space=loopy.AddressSpace.LOCAL)
            else:
                new_temps[tv.name] = tv

        kernel = kernel.copy(temporary_variables=new_temps,
                instructions=kernel.instructions+new_insns,
                domains=kernel.domains+new_domains)
        kernel = loopy.add_dependency(kernel, "tag:gather", "tag:init_shared")
        for priority in priorities:
            kernel = loopy.prioritize_loops(kernel, ",".join(priority))

    # }}}

    # {{{ organize for precomputes

    written_count = dict((written_var, 0) for written_var in
            kernel.get_written_variables())
    for insn in kernel.instructions:
        if isinstance(insn.assignee, Variable):
            written_count[insn.assignee.name] += 1
        elif isinstance(insn.assignee, Subscript):
            written_count[insn.assignee.aggregate.name] += 1

    if tiled_access_to_the_vars:
        from loopy.transform.data import remove_unused_axes_in_temporaries
        kernel = remove_unused_axes_in_temporaries(kernel)

        args_to_be_interpreted_as_substs = set()

        for insn in kernel.instructions:
            if frozenset(['gather', 'init_shared']) & insn.tags:
                if isinstance(insn.assignee, Subscript) and (
                        written_count[insn.assignee.aggregate.name] == 1):
                    args_to_be_interpreted_as_substs.add(
                            insn.assignee.aggregate.name)

        substs_to_insns = dict((var_name, []) for var_name in args_to_be_interpreted_as_substs)

        for insn in kernel.instructions:
            precompted_args_referred_in_insn = (insn.read_dependency_names() & args_to_be_interpreted_as_substs)
            for arg_name in precompted_args_referred_in_insn:
                substs_to_insns[arg_name].append(insn.id)

        for arg_name in args_to_be_interpreted_as_substs:
            kernel = loopy.assignment_to_subst(kernel, arg_name)

    # }}}

    # {{{ making consts as globals

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

    # {{{ realizing CUDA blocks(i.e. chunk)

    kernel = loopy.split_iname(kernel, "n", ncells_per_chunk, outer_iname="ichunk", inner_iname="icell")

    # }}}

    # {{{ extracting variables that are need to be stored between stages.

    temp_vars = frozenset(kernel.temporary_variables.keys())

    written_in_load = frozenset().union(*[insn.write_dependency_names() for
        insn in kernel.instructions if 'gather' in insn.tags]) & temp_vars

    written_in_quad = frozenset().union(*[insn.write_dependency_names() for
        insn in kernel.instructions if 'quadrature' in insn.tags]) & temp_vars

    read_in_quad = frozenset().union(*[insn.read_dependency_names() for
        insn in kernel.instructions if 'quadrature' in insn.tags]) & temp_vars

    read_in_basis = frozenset().union(*[insn.read_dependency_names() for
        insn in kernel.instructions if 'basis' in insn.tags]) & temp_vars

    # }}}

    # {{{ remove unnecessary dependencies on quadrature instructions

    # Main aim: The variable in which the result of the basis coefficient is
    # written should be initialized in the basis part itself

    vars_not_neeeded_in_quad = written_in_load - read_in_quad

    # so lets just write in the basis part
    written_in_load = written_in_load - vars_not_neeeded_in_quad

    insns_to_be_added_in_basis = frozenset([insn.id for insn in
        kernel.instructions if insn.write_dependency_names()
        & vars_not_neeeded_in_quad and 'gather' in insn.tags])

    def _remove_unnecessary_deps_on_load(insn):
        return insn.copy(depends_on=insn.depends_on - insns_to_be_added_in_basis)

    kernel = loopy.map_instructions(kernel, 'tag:quadrature',
            _remove_unnecessary_deps_on_load)

    def _add_unnecessary_instructions_to_basis(insn):
        if insn.id in insns_to_be_added_in_basis:
            return insn.copy(tags=insn.tags-frozenset(["gather"])
                | frozenset(["basis", "basis_init"]))
        return insn
    kernel = loopy.map_instructions(kernel, "id:*",
            _add_unnecessary_instructions_to_basis)

    # }}}

    # {{{ storing values between the stages

    batch_vars = (written_in_quad & read_in_basis)  # function evaluation at quadrature
    kernel = loopy.save_temporaries_in_loop(kernel, 'form_ip', batch_vars, within='iname:form_ip')
    kernel = loopy.save_temporaries_in_loop(kernel, 'icell', batch_vars, within="not tag:init_shared")

    # }}}

    # {{{ duplicating inames

    kernel = loopy.duplicate_inames(kernel, ["ichunk", "icell"],
            new_inames=["ichunk_quad", "icell_quad"],
            within="not (tag:basis or tag:scatter)", tags={"ichunk": "g.0"})

    kernel = loopy.duplicate_inames(kernel, ["ichunk", "icell"],
            new_inames=["ichunk_basis", "icell_basis"],
            within="tag:basis or tag:scatter", tags={"ichunk": "g.0"})

    kernel = loopy.duplicate_inames(kernel, ["form_ip"],
            new_inames=["form_ip_quad"], within="tag:quadrature")
    kernel = loopy.duplicate_inames(kernel, ["form_ip"],
            new_inames=["form_ip_basis"], within="tag:basis")

    kernel = loopy.remove_unused_inames(kernel)

    # All these inames are split in some way and should not be used in instructions
    assert not (frozenset(["icell", "n", "ichunk"]) & kernel.all_inames())

    # }}}

    # {{{ realizing which instructions belongs to which part

    # Yes, this should be here. Should be realized from TSFC. But works for
    # now. Is this the worst humanity has ever seen, no(obviously). Is this the
    # worst use of logic in a Scientific Computing library, probably yes.

    new_insns = []

    done_with_jacobi_eval = False
    done_with_quad_init = False
    done_with_quad_reduction = False
    done_with_quad_wrap_up = False
    done_with_basis_init = False
    done_with_basis_reduction = False

    for insn in kernel.instructions:
        if not done_with_jacobi_eval:
            if 'form_ip_quad' in insn.within_inames:
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
            if 'form_ip_quad' not in insn.within_inames:
                done_with_quad_wrap_up = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["quad_wrap_up"])))
                continue
        if not done_with_basis_init:
            if 'form_ip_basis' in insn.within_inames:
                done_with_basis_init = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["basis_init"])))
                continue
        if not done_with_basis_reduction:
            if 'form_ip_basis' not in insn.within_inames:
                done_with_basis_reduction = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["basis_redn"])))
                continue
        new_insns.append(insn)

    kernel = kernel.copy(instructions=new_insns)

    # }}}

    # {{{ dividing substitutions into logical parts

    logical_units = set(['jacobi_eval', 'quad_init', 'quad_redn', 'quad_wrap_up', 'basis_init', 'basis_redn'])

    subst_to_logical_part = {}

    for subst, insn_ids in substs_to_insns.items():
        logical_unit = frozenset().union(*(kernel.id_to_insn[insn_id].tags for insn_id in insn_ids))
        logical_unit &= logical_units

        subst_to_logical_part[subst] = logical_unit

    # }}}

    # {{{ interpreting the domain as a cuboid

    new_space = kernel.domains[0].get_space()
    new_dom = islpy.BasicSet.universe(new_space)
    for stage in ['quad', 'basis']:
        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'icell_%s' % stage: -1,
                    1: ncells_per_chunk-1}))
        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'icell_%s' % stage: 1}))

        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'ichunk_%s' % stage:
                    -(ncells_per_chunk),
                    'icell_%s' % stage:
                    -1,
                    'start': -1, 'end': 1, 1: -1}))
        new_dom = new_dom.add_constraint(
                islpy.Constraint.ineq_from_names(new_space, {
                    'ichunk_%s' % stage: 1}))

    kernel = kernel.copy(domains=[new_dom]+kernel.domains[1:])

    # }}}

    # {{{ coalescing the entire domain forest

    # Why coalsce? In order join inames we need them to be in the same iname forest

    new_space = kernel.domains[0].get_space()
    pos = kernel.domains[0].n_dim()
    for dom in kernel.domains[1:]:
        # product of all the spaces
        for dim_name, (dim_type, _) in dom.get_space().get_var_dict().items():
            assert dim_type == 3
            new_space = new_space.add_dims(dim_type, 1)
            new_space = new_space.set_dim_name(dim_type, pos, dim_name)
            pos += 1

    new_domain = islpy.BasicSet.universe(new_space)
    for dom in kernel.domains[:]:
        for constraint in dom.get_constraints():
            if constraint.is_equality():
                new_domain = (
                        new_domain.add_constraint(
                            islpy.Constraint.eq_from_names(new_space,
                                constraint.get_coefficients_by_name())))
            else:
                new_domain = (
                        new_domain.add_constraint(
                            islpy.Constraint.ineq_from_names(new_space,
                                constraint.get_coefficients_by_name())))

    kernel = kernel.copy(domains=[new_domain])

    # }}}

    # {{{ re-distributing the quadrature evaluation work

    kernel = loopy.split_iname(kernel, "form_ip_quad", nthreads_per_cell)
    kernel = loopy.join_inames(kernel, ["icell_quad", "form_ip_quad_inner"], "local_id%d" % n_lids, within="tag:quadrature")
    n_lids += 1

    # }}}

    # {{{ re-distributing the basis coeffs evaluation work

    kernel = loopy.split_iname(kernel, basis_iname, nthreads_per_cell, within='tag:basis')
    kernel = loopy.join_inames(kernel, ["icell_basis", basis_iname+"_inner"], "local_id%d" % n_lids, within='tag:basis')

    kernel = loopy.split_iname(kernel, scatter_iname, nthreads_per_cell, within="tag:scatter")
    kernel = loopy.join_inames(kernel, ["icell_basis", scatter_iname+"_inner"], "local_id%d" % (n_lids+1), within="tag:scatter")

    n_lids += 2

    kernel = loopy.rename_iname(kernel, scatter_iname+'_outer', basis_iname+'_outer',
            within='tag:scatter', existing_ok=True)

    from loopy.transform.make_scalar import (
            make_scalar, remove_invariant_inames)
    # FIXME: generalize this
    kernel = make_scalar(kernel, 't0')
    kernel = loopy.save_temporaries_in_loop(kernel, basis_iname+"_outer",
            ['t0'], within="tag:basis or tag:scatter")
    kernel = remove_invariant_inames(kernel)

    # }}}

    # {{{ setting tile lengths

    #FIXME: Generalize the iname over basis i.e. 'form_i'
    kernel = loopy.split_iname(kernel, 'form_i', tile_quad)
    kernel = loopy.split_iname(kernel, 'form_ip_basis', tile_basis)

    # }}}

    # {{{ privatizing temporaries

    from loopy.transform.precompute import precompute_for_single_kernel
    vars_to_duplicate_in_quad = (
            frozenset().union(*(insn.write_dependency_names() for insn in kernel.instructions if 'quad_init' in insn.tags)) & kernel.temporary_variables.keys())
    kernel = loopy.privatize_temporaries_with_inames(kernel,
            'form_ip_quad_outer', only_var_names=vars_to_duplicate_in_quad)
    kernel = loopy.privatize_temporaries_with_inames(kernel, 'form_j_outer')

    # }}}

    kernel = loopy.duplicate_inames(kernel, ['form_ip_quad_outer'], new_inames=['form_ip_quad_outer_init'], within='tag:quad_init')
    kernel = loopy.duplicate_inames(kernel, ['form_ip_quad_outer'], new_inames=['form_ip_quad_outer_wrap_up'], within='tag:quad_wrap_up')
    kernel = loopy.duplicate_inames(kernel, ['form_j_outer'], new_inames=['form_j_outer_basis_init'], within='tag:basis_init')
    kernel = loopy.duplicate_inames(kernel, ['form_j_outer'], new_inames=['form_j_outer_scatter'], within='tag:scatter')

    # Changing the order of the loops to facilitate tiling
    kernel = loopy.prioritize_loops(kernel, ("form_i_outer", "form_ip_quad_outer", "form_i_inner"))
    kernel = loopy.prioritize_loops(kernel, ('form_ip_basis_outer', 'form_j_outer', 'form_ip_basis_inner'))

    for subst, logical_units in subst_to_logical_part.items():
        rule = kernel.substitutions[subst+'_subst']
        vng = kernel.get_var_name_generator()

        variables_precomputed_to = []

        for logical_unit in logical_units:
            if logical_unit == 'jacobi_eval':
                sweep_inames = []
                outer_inames = None
                address_space = loopy.AddressSpace.PRIVATE
            elif logical_unit == 'quad_wrap_up':
                address_space = loopy.AddressSpace.LOCAL
                sweep_inames = ['form_ip_quad_outer_wrap_up', 'local_id0']
                outer_inames = frozenset(["ichunk_quad"])
            elif logical_unit == 'quad_redn':
                if subst in consts_precomputed:
                    sweep_inames = ['form_ip_quad_outer', 'form_i_inner', 'local_id0']
                    outer_inames = frozenset(['ichunk_quad', 'form_i_outer'])
                    address_space = loopy.AddressSpace.LOCAL
                else:
                    sweep_inames = ['form_i_inner']
                    outer_inames = frozenset(['ichunk_quad', 'form_i_outer', 'local_id0'])
                    address_space = loopy.AddressSpace.PRIVATE
            elif logical_unit == 'basis_redn':
                if subst in consts_precomputed:
                    sweep_inames = ['form_j_outer', 'form_ip_basis_inner', 'local_id1']
                    outer_inames = frozenset(['ichunk_basis', 'form_ip_basis_outer'])
                    address_space = loopy.AddressSpace.LOCAL
                else:
                    raise NotImplementedError('No known case of private var in basis reduction phase.')
            else:
                raise NotImplementedError('Unknown logical unit %s.' % logical_unit)

            precompute_inames = tuple(vng(based_on='icopy') for _ in rule.arguments)
            temporary_name = vng(based_on=subst+'_temp')
            variables_precomputed_to.append(temporary_name)

            new_insn_id = kernel.get_instruction_id_generator()(based_on='precompute')

            kernel = precompute_for_single_kernel(kernel, callables_table,
                    subst_use=subst+'_subst',
                    sweep_inames=sweep_inames,
                    precompute_outer_inames=outer_inames,
                    temporary_address_space=address_space,
                    precompute_inames=precompute_inames,
                    temporary_name=temporary_name,
                    compute_insn_id=new_insn_id,
                    default_tag=None,
                    within='tag:{0}'.format(logical_unit))

            if address_space == loopy.AddressSpace.LOCAL:
                if len(precompute_inames) > 1:
                    iname_to_split = "aux_local_id%d" % n_lids
                    kernel = loopy.join_inames(kernel, precompute_inames, iname_to_split)
                else:
                    iname_to_split = precompute_inames[0]

                kernel = loopy.split_iname(kernel, iname_to_split,
                        nthreads_per_cell * ncells_per_chunk, inner_tag="l.0",
                        outer_tag="ilp")
                n_lids += 1

            def tag_precompute_instruction(precompute_insn):
                return precompute_insn.copy(tags=frozenset([logical_unit]))

            kernel = loopy.map_instructions(kernel,
                    'id:{0}'.format(new_insn_id), tag_precompute_instruction)

        if len(variables_precomputed_to) > 1:
            assert len(variables_precomputed_to) == 2
            temp_in_quad, temp_in_basis = variables_precomputed_to

            from loopy.transform.data import flatten_variable, absorb_temporary_into
            kernel = flatten_variable(kernel, temp_in_quad)
            kernel = flatten_variable(kernel, temp_in_basis)
            if np.prod(kernel.temporary_variables[temp_in_quad].shape) >= (
                    np.prod(kernel.temporary_variables[temp_in_basis].shape)):
                kernel = absorb_temporary_into(kernel, temp_in_quad, temp_in_basis)
            else:
                kernel = absorb_temporary_into(kernel, temp_in_basis, temp_in_quad)

    kernel = loopy.add_dependency(kernel, 'tag:quad_init', 'tag:jacobi_eval')
    kernel = loopy.add_dependency(kernel, 'tag:quad_redn', 'tag:quad_init')
    kernel = loopy.add_dependency(kernel, 'tag:quad_wrap_up', 'tag:quad_redn')
    kernel = loopy.add_dependency(kernel, 'tag:basis_init', 'tag:quad_wrap_up')
    kernel = loopy.add_dependency(kernel, 'tag:basis_redn', 'tag:basis_init')

    iname_tags = {
        "ichunk_quad":      "g.0",
        "ichunk_basis":     "g.0",
        }
    for i in range(n_lids):
        iname_tags["local_id%d" % i] = "l.0"
        iname_tags["aux_local_id%d_outer" % i] = "ilp"

    kernel = loopy.tag_inames(kernel, iname_tags, ignore_nonexistent=True)

    return (loopy.remove_unused_inames(kernel).copy(loop_priority=frozenset()),
            args_to_make_global)


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
        elif configuration["cuda_strategy"] == "gcdtt":
            kernel, args_to_make_global = gcd_tt_simple(kernel)
        elif configuration["cuda_strategy"] == "tilett":
            kernel, args_to_make_global = tiled_gcd_tt(kernel, program.callables_table)
        else:
            raise NotImplementedError
    else:
        kernel, args_to_make_global = sept(kernel, extruded)
    # kernel = thread_transposition(kernel)
    # kernel, args_to_make_global = tiled_gcd_tt(kernel, program.callables_table)

    program = program.with_root_kernel(kernel)

    code = loopy.generate_code_v2(program).device_code()
    if program.name == "wrap_pyop2_kernel_uniform_extrusion":
        code = code.replace("inline void pyop2_kernel_uniform_extrusion", "__device__ inline void pyop2_kernel_uniform_extrusion")

    return code, program, args_to_make_global
