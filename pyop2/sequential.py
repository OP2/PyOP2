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

"""OP2 sequential backend."""

import os
from copy import deepcopy as dcopy

import ctypes

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
from pyop2.petsc_base import Global, GlobalDataSet       # noqa: F401
from pyop2.petsc_base import Dat, MixedDat, Mat          # noqa: F401
from pyop2.exceptions import *  # noqa: F401
from pyop2.mpi import collective
from pyop2.profiling import timed_region
from pyop2.utils import cached_property, get_petsc_dir

import loopy


# def vectorize_loop(wrapper, iname, batch_size, start, end):
#     # split iname and vectorize the inner loop
#     if builder.batch > 1:
#         if builder.extruded:
#             outer = "layer"
#             inner = "layer_inner"
#             wrapper = loopy.assume(wrapper, "t0 mod {0} = 0".format(builder.batch))
#             wrapper = loopy.assume(wrapper, "exists zz: zz > 0 and t1 = {0}*zz + t0".format(builder.batch))
#         else:
#             outer = "n"
#             inner = "n_inner"
#             wrapper = loopy.assume(wrapper, "start mod {0} = 0".format(builder.batch))
#             wrapper = loopy.assume(wrapper, "exists zz: zz > 0 and end = {0}*zz + start".format(builder.batch))
#
#         wrapper = loopy.split_iname(wrapper, outer, builder.batch, inner_tag="ilp.seq", inner_iname=inner)
#
#     alignment = 64
#     for name in wrapper.temporary_variables:
#         tv = wrapper.temporary_variables[name]
#         wrapper.temporary_variables[name] = tv.copy(alignment=alignment)
#
#     return wrapper


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
        if not kwargs.get('delay', False):
            self.compile()
            self._initialized = True

    @collective
    def __call__(self, *args):
        return self._fun(*args)

    @property
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
        code = loopy.generate_code_v2(wrapper)

        # nbytes = 0
        # for arg in self._args:
        #     nbytes += arg.data.nbytes
        # for m in builder.maps.keys():
        #     nbytes += len(m.values) * 4
        # print("BYTES= {0}".format(nbytes))

        if self._kernel._cpp:
            from loopy.codegen.result import process_preambles
            preamble = "".join(process_preambles(getattr(code, "device_preambles", [])))
            device_code = "\n\n".join(str(dp.ast) for dp in code.device_programs)
            return preamble + "\nextern \"C\" {\n" + device_code + "\n}\n"

        return code.device_code()

    @collective
    def compile(self):
        # If we weren't in the cache we /must/ have arguments
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule has no args associated with it, should never happen")

        from pyop2.configuration import configuration

        compiler = configuration["compiler"]
        extension = "cpp" if self._kernel._cpp else "c"
        cppargs = self._cppargs
        cppargs += ["-I%s/include" % d for d in get_petsc_dir()] + \
                   ["-I%s" % d for d in self._kernel._include_dirs] + \
                   ["-I%s" % os.path.abspath(os.path.dirname(__file__))]
        ldargs = ["-L%s/lib" % d for d in get_petsc_dir()] + \
                 ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()] + \
                 ["-lpetsc", "-lm"] + self._libraries
        ldargs += self._kernel._ldargs

        self._fun = compilation.load(self,
                                     extension,
                                     self._wrapper_name,
                                     cppargs=cppargs,
                                     ldargs=ldargs,
                                     restype=ctypes.c_int,
                                     compiler=compiler,
                                     comm=self.comm)
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
                    argtypes += (t,)
                    seen.add(k)
        return argtypes


class ParLoop(petsc_base.ParLoop):

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
                    arglist += (k,)
                    seen.add(k)
        return arglist

    @cached_property
    def _jitmodule(self):
        return JITModule(self.kernel, self.iterset, *self.args,
                         iterate=self.iteration_region,
                         pass_layer_arg=self._pass_layer_arg)

    @collective
    def _compute(self, part, fun, *arglist):
        with timed_region("ParLoop{0}".format(self.iterset.name)):
            fun(part.offset, part.offset + part.size, *arglist)
            self.log_flops(self.num_flops * part.size)


def generate_single_cell_wrapper(iterset, args, forward_args=(), kernel_name=None, wrapper_name=None):
    """Generates wrapper for a single cell. No iteration loop, but cellwise data is extracted.
    Cell is expected as an argument to the wrapper. For extruded, the numbering of the cells
    is columnwise continuous, bottom to top.

    :param iterset: The iteration set
    :param args: :class:`Arg`s
    :param forward_args: To forward unprocessed arguments to the kernel via the wrapper,
                         give an iterable of strings describing their C types.
    :param kernel_name: Kernel function name
    :param wrapper_name: Wrapper function name

    :return: string containing the C code for the single-cell wrapper
    """
    from pyop2.codegen.builder import WrapperBuilder
    from pyop2.codegen.rep2loopy import generate
    from loopy.types import OpaqueType

    forward_arg_types = [OpaqueType(fa) for fa in forward_args]
    builder = WrapperBuilder(iterset=iterset, single_cell=True, forward_arg_types=forward_arg_types, restart=False)
    for arg in args:
        builder.add_argument(arg)
    builder.set_kernel(Kernel("", kernel_name))
    wrapper = generate(builder, wrapper_name)
    code = loopy.generate_code_v2(wrapper)

    return code.device_code()
