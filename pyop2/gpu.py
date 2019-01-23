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

"""OP2 GPU backend."""

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
import pyopencl as cl
import numpy as np
from collections import OrderedDict


class Map(Map):

    map_buffer_func = None
    # move ocl_buffer here potentially

    @staticmethod
    def get_map_buffer_func():
        # function handle to create buffer for Map, this is the same function for all Maps
        if Map.map_buffer_func is not None:
            return Map.map_buffer_func

        # arg is needed to get the context
        # better way to do this?
        c_code_str = r"""
#include "petsc.h"
#include "petscviennacl.h"
#include <CL/cl.h>
#include <iostream>

using namespace std;

extern "C" cl_mem get_map_buffer(int * __restrict__ map_array, const int map_size, Vec arg)
{
    const viennacl::vector<PetscScalar> *arg_viennacl;
    VecViennaCLGetArrayRead(arg, &arg_viennacl);

    viennacl::ocl::context ctx = arg_viennacl->handle().opencl_handle().context();
    cl_mem map_buffer = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, map_size*sizeof(cl_int), map_array, NULL);

    VecViennaCLRestoreArrayRead(arg, &arg_viennacl);
    return map_buffer;
}
"""
        from pyop2.sequential import JITModule

        compiler = configuration["compiler"]
        extension = "cpp"
        cppargs = JITModule._cppargs
        cppargs += ["-I%s/include" % d for d in get_petsc_dir()] + \
                   ["-I%s" % os.path.abspath(os.path.dirname(__file__))]

        cppargs += ["-DVIENNACL_WITH_OPENCL"]
        ldargs = ["-L%s/lib" % d for d in get_petsc_dir()] + \
                 ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()] + \
                 ["-lpetsc", "-lm"]
        fun = compilation.load(c_code_str,
                               extension,
                               'get_map_buffer',
                               cppargs=cppargs,
                               ldargs=ldargs,
                               restype=ctypes.c_void_p,
                               compiler=compiler,
                               comm=None)  # does comm matter?
        fun.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
        Map.map_buffer_func = fun
        return fun

    def map_buffer(self, arg):
        if self.ocl_buffer is None:
            # if not in the cache, we /must/ have arguments
            func = Map.get_map_buffer_func()
            self.ocl_buffer = func(self._kernel_args_[0], np.int32(np.product(self.shape)), arg._kernel_args_[0])
        return self.ocl_buffer


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
            self._vec.setType('viennacl')
            self._vec.setArray(data)
        # PETSc Vecs have a state counter and cache norm computations
        # to return immediately if the state counter is unchanged.
        # Since we've updated the data behind their back, we need to
        # change that state counter.
        self._vec.stateIncrease()
        yield self._vec
        if access is not base.READ:
            self.halo_valid = False


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
            self._vec.setType('viennacl')
            self._vec.setArray(data)
        # PETSc Vecs have a state counter and cache norm computations
        # to return immediately if the state counter is unchanged.
        # Since we've updated the data behind their back, we need to
        # change that state counter.
        self._vec.stateIncrease()
        yield self._vec


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
        self.cl_kernel = None

        if not kwargs.get('delay', False):
            self.compile()
            self._initialized = True

    @collective
    def __call__(self, *args):

        if self.cl_kernel is None:
            # compile the CL kernel only once.
            self.cl_kernel = cl.Kernel.from_int_ptr(self.cl_kernel_getter_func(*args))

        return self._fun(self.cl_kernel.int_ptr, *args)

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
        code = generate_cl_kernel_compiler_executor(wrapper)
        return code

    @collective
    def compile(self):
        # If we weren't in the cache we /must/ have arguments
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule has no args associated with it, should never happen")

        from pyop2.configuration import configuration

        compiler = configuration["compiler"]
        extension = "cpp" if self._kernel.cpp else "c"
        cppargs = self._cppargs
        cppargs += ["-I%s/include" % d for d in get_petsc_dir()] + \
                   ["-I%s" % d for d in self._kernel._include_dirs] + \
                   ["-I%s" % os.path.abspath(os.path.dirname(__file__))]
        ldargs = ["-L%s/lib" % d for d in get_petsc_dir()] + \
                 ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()] + \
                 ["-lpetsc", "-lm"] + self._libraries
        ldargs += self._kernel._ldargs

        self.cl_kernel_getter_func = compilation.load(
            self,
            extension,
            self._wrapper_name+'_cl_knl_extractor',
            cppargs=cppargs,
            ldargs=ldargs,
            restype=ctypes.c_void_p,
            compiler=compiler,
            comm=self.comm)
        self.cl_kernel_getter_func.argtypes = self.argtypes[1:]

        self._fun = compilation.load(
            self,
            extension,
            self._wrapper_name+'_executor',
            cppargs=cppargs,
            ldargs=ldargs,
            restype=ctypes.c_int,
            compiler=compiler,
            comm=self.comm)

        # Blow away everything we don't need any more
        del self._args
        # del self._kernel
        del self._iterset

    @cached_property
    def argtypes(self):
        index_type = as_ctypes(IntType)
        argtypes = (ctypes.c_void_p, index_type, index_type)
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
                    argtypes += (ctypes.c_void_p, ctypes.c_int)
                    seen.add(k)

        return argtypes


class ParLoop(petsc_base.ParLoop):

    def __init__(self, *args, **kwargs):
        super(ParLoop, self).__init__(*args, **kwargs)
        self.kernel.cpp = True

    def prepare_arglist(self, iterset, *args):

        arglist = iterset._kernel_args_
        for arg in args:
            arglist += arg._kernel_args_
        seen = set()
        for arg in args:
            maps = arg.map_tuple
            for map_ in maps:
                for k in map_._kernel_args_:
                    if k in seen:
                        continue
                    arglist += (map_.map_buffer(arg), np.int32(np.product(map_.shape)))
                    seen.add(k)
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


def get_grid_sizes(kernel):
    parameters = {}
    for arg in kernel.args:
        if isinstance(arg, loopy.ValueArg) and arg.approximately is not None:
            parameters[arg.name] = arg.approximately

    glens, llens = kernel.get_grid_size_upper_bounds_as_exprs()

    from pymbolic import evaluate
    from pymbolic.mapper.evaluator import UnknownVariableError
    try:
        glens = evaluate(glens, parameters)
        llens = evaluate(llens, parameters)
    except UnknownVariableError as name:
        from warnings import warn
        warn("could not check axis bounds because no value for variable '%s' was passed to check_kernels()" % name)

    return llens, glens


def generate_cl_kernel_compiler_executor(program):

    import pyopencl as cl
    from mako.template import Template

    # Kernel transformations

    kernel = program.root_kernel

    def insn_needs_atomic(insn):
        # updates to global variables are atomic
        assignee_name = insn.assignee.aggregate.name
        return assignee_name in insn.read_dependency_names() and assignee_name not in kernel.temporary_variables

    new_insns = []
    args_marked_for_atomic = set()
    for insn in kernel.instructions:
        if ('pyop2_assign' in insn.tags) or ('tsfc_return_accumulate' in insn.tags):
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

    # batch cells into groups
    assert "start" in kernel.arg_dict and "end" in kernel.arg_dict
    
    batch_size = 128
    g_size = ("(end - start)",)  # global size
    l_size = (batch_size,)  # local size

    kernel = loopy.assume(kernel, "{0} mod {1} = 0".format("end", batch_size))
    kernel = loopy.assume(kernel, "exists zz: zz > 0 and {0} = {1}*zz + {2}".format("end", batch_size, "start"))
    kernel = loopy.split_iname(kernel, "n", batch_size, inner_tag="l.0", outer_tag="g.0")

    # create boiler plate code to compile OpenCL kernel

    program = program.with_root_kernel(kernel)
    ctx = cl.create_some_context(0)
    program = program.copy(target=loopy.PyOpenCLTarget(ctx.devices[0]))

    ast_builder = program.target.get_device_ast_builder()
    arg_dict = OrderedDict()  # arg name -> (idx, type, declariation, size)
    for idx, arg in enumerate(kernel.args):
        name = arg.name
        if isinstance(arg, loopy.ArrayArg):
            if arg.dtype.is_integral():
                # map
                arg_dict[name] = (idx, "map", "cl_mem {0}".format(name), "sizeof({0})".format(name))
                arg_dict[name + "_size"] = (None, "map_size", "const int {0}_size".format(name), "")
            else:
                # vec
                arg_dict[name] = (idx, "vec", "Vec {0}".format(name), "sizeof(PetscScalar)")
        else:
            # start, end
            arg_dict[name] = (idx, "other", str(arg.get_arg_decl(ast_builder))[:-1], arg.dtype.itemsize)

    # code generation template
    c_code_str = r'''
<%
import loopy as lp
vecs = [arg for arg in arg_dict if arg_dict[arg][1] == "vec"]
v0 = vecs[0]
%>
#include <CL/cl.h>
#include "petsc.h"
#include "petscvec.h"
#include "petscviennacl.h"
#include <iostream>
#include <sys/time.h>
#include <cstring>
#include <cstdlib>

// ViennaCL Headers
#include "viennacl/ocl/backend.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/backend/memory.hpp"
#include "viennacl/ocl/error.hpp"

using namespace std;

char kernel_source[] = "${kernel_src}";

extern "C" cl_kernel ${kernel_name}_cl_knl_extractor(${', '.join(v[2] for v in arg_dict.values())})
{
    // viennacl vector declarations
    const viennacl::vector<PetscScalar> *${v0}_viennacl;

    // getting the array from the petsc vecs
    VecViennaCLGetArrayRead(${v0}, &${v0}_viennacl);

    // defining the context
    viennacl::ocl::context ctx = ${v0}_viennacl->handle().opencl_handle().context();

    viennacl::ocl::program & my_prog = ctx.add_program(kernel_source, "kernel_program_${kernel_name}");
    viennacl::ocl::kernel & viennacl_kernel = my_prog.get_kernel("${kernel_name}");

    VecViennaCLRestoreArrayRead(${v0}, &${v0}_viennacl);

    cl_kernel cl_knl = viennacl_kernel.handle().get();
    clRetainKernel(cl_knl);
    return cl_knl;
}

extern "C" void ${kernel_name}_executor(cl_kernel cl_knl, ${', '.join(v[2] for v in arg_dict.values())})
{
    if(end == start)
    {
        // no need to go any further
        return;
    }

    cl_int ocl_err;

    // viennacl vector declarations
    % for vec in vecs:
    viennacl::vector<PetscScalar> *${vec}_viennacl;
    % endfor

    // getting the array from the petsc vecs
    % for vec in vecs:
    VecViennaCLGetArrayReadWrite(${vec}, &${vec}_viennacl);
    % endfor

    // defining the context
    viennacl::ocl::context ctx = ${v0}_viennacl->handle().opencl_handle().context();

    // set the kernel args
    % for arg, (idx, t, _, size) in arg_dict.items():
    % if t == "other":
    ocl_err = clSetKernelArg(cl_knl, ${idx}, ${size}, &${arg});
    % elif t == "map":
    ocl_err = clSetKernelArg(cl_knl, ${idx}, ${size}, &${arg});
    % elif t == "vec":
    ocl_err = clSetKernelArg(cl_knl, ${idx}, ${size}, &(${arg}_viennacl->handle().opencl_handle().get()));
    % endif
    VIENNACL_ERR_CHECK(ocl_err);

    % endfor

    // getting the queue
    cl_command_queue queue= ctx.get_queue().handle().get();

    // set work group sizes
    size_t lwg_size[${len(l_size)}] = {${", ".join(map(str, l_size))}};
    size_t gwg_size[${len(g_size)}] = {${", ".join(map(str, g_size))}};
    clFinish(queue);

    // enqueueing the kernel
    ocl_err = clEnqueueNDRangeKernel(queue, cl_knl, ${len(l_size)}, NULL, gwg_size, lwg_size, 0, NULL, NULL);
    VIENNACL_ERR_CHECK(ocl_err);

    clFinish(queue);

    // restoring the arrays to the petsc vecs
    % for v in vecs:
    VecViennaCLRestoreArrayReadWrite(${v}, &${v}_viennacl);
    % endfor
}
'''
    c_code = Template(c_code_str)

    kernel_code = loopy.generate_code_v2(program).device_code().replace('\n', '\\n"\n"')

    return c_code.render(
        kernel_src=kernel_code,
        kernel_name=program.name,
        arg_dict=arg_dict,
        ast_builder=program.target.get_device_ast_builder(),
        args=program.args,
        l_size=l_size,
        g_size=g_size)
