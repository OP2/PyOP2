import ctypes
import numpy

import loopy
import islpy as isl
import pymbolic.primitives as pym

from collections import OrderedDict, defaultdict
from functools import singledispatch, reduce
import itertools
import operator

from gem.node import traversal, Node, Memoizer, reuse_if_untouched

from pyop2.base import READ
from pyop2.datatypes import as_ctypes

from pyop2.codegen.optimise import index_merger

from pyop2.codegen.representation import (Index, FixedIndex, RuntimeIndex,
                                          MultiIndex, Extent, Indexed,
                                          BitShift, BitwiseNot, BitwiseAnd, BitwiseOr,
                                          Conditional, Comparison, DummyInstruction,
                                          LogicalNot, LogicalAnd, LogicalOr,
                                          Materialise, Accumulate, FunctionCall, When,
                                          Argument, Variable, Literal, NamedLiteral,
                                          Symbol, Zero, Sum, Product)
from pyop2.codegen.representation import (PackInst, UnpackInst, KernelInst)


class Bag(object):
    pass


# {{{ manglers

def symbol_mangler(kernel, name):
    if name in {"ADD_VALUES", "INSERT_VALUES"}:
        return loopy.types.to_loopy_type(numpy.int32), name
    return None


class PetscCallable(loopy.ScalarCallable):

    def with_types(self, arg_id_to_dtype, kernel):
        new_arg_id_to_dtype = arg_id_to_dtype.copy()
        return self.copy(
            name_in_target=self.name,
            arg_id_to_dtype=new_arg_id_to_dtype
        )

    def with_descrs(self, arg_id_to_descr):
        from loopy.kernel.function_interface import ArrayArgDescriptor
        from loopy.kernel.array import FixedStrideArrayDimTag
        new_arg_id_to_descr = arg_id_to_descr.copy()
        for i, des in arg_id_to_descr.items():
            # petsc takes 1D arrays as arguments
            if isinstance(des, ArrayArgDescriptor):
                dim_tags = tuple(
                    FixedStrideArrayDimTag(
                        stride=int(numpy.prod(des.shape[i+1:])),
                        layout_nesting_level=len(des.shape)-i-1
                    )
                    for i in range(len(des.shape))
                )
                new_arg_id_to_descr[i] = ArrayArgDescriptor(
                    shape=des.shape,
                    mem_scope=des.mem_scope,
                    dim_tags=dim_tags
                )


        return self.copy(arg_id_to_descr=new_arg_id_to_descr)


def petsc_function_lookup(target, identifier):
    if identifier == 'MatSetValuesBlockedLocal':
        return PetscCallable(name=identifier)
    elif identifier == 'MatSetValuesLocal':
        return PetscCallable(name=identifier)
    return None


# PyOP2 Kernel passed in as string
class PyOP2KernelCallable(loopy.ScalarCallable):

    fields = set(["name", "access", "arg_id_to_dtype", "arg_id_to_descr", "name_in_target"])
    init_arg_names = ("name", "access", "arg_id_to_dtype", "arg_id_to_descr", "name_in_target")

    def __init__(self, name, access, arg_id_to_dtype=None, arg_id_to_descr=None, name_in_target=None):
        super(PyOP2KernelCallable, self).__init__(name, arg_id_to_dtype, arg_id_to_descr, name_in_target)
        self.access = access

    def with_types(self, arg_id_to_dtype, kernel):
        new_arg_id_to_dtype = arg_id_to_dtype.copy()
        return self.copy(
            name_in_target=self.name,
            arg_id_to_dtype=new_arg_id_to_dtype
        )

    def with_descrs(self, arg_id_to_descr):
        from loopy.kernel.function_interface import ArrayArgDescriptor
        from loopy.kernel.array import FixedStrideArrayDimTag
        new_arg_id_to_descr = arg_id_to_descr.copy()
        for i, des in arg_id_to_descr.items():
            # 1D arrays
            if isinstance(des, ArrayArgDescriptor):
                dim_tags = tuple(
                    FixedStrideArrayDimTag(
                        stride=int(numpy.prod(des.shape[i+1:])),
                        layout_nesting_level=len(des.shape)-i-1
                    )
                    for i in range(len(des.shape))
                )
                new_arg_id_to_descr[i] = ArrayArgDescriptor(
                    shape=des.shape,
                    mem_scope=des.mem_scope,
                    dim_tags=dim_tags
                )

        return self.copy(arg_id_to_descr=new_arg_id_to_descr)

    def emit_call_insn(self, insn, target, expression_to_code_mapper):

        # reorder arguments, e.g. a,c = f(b,d) to f(a,b,c,d)
        parameters = []
        reads = iter(insn.expression.parameters)
        writes = iter(insn.assignees)
        for ac in self.access:
            if ac is READ:
                parameters.append(next(reads))
            else:
                parameters.append(next(writes))

        # pass layer argument if needed
        for layer in reads:
            parameters.append(layer)

        par_dtypes = tuple(expression_to_code_mapper.infer_type(p) for p in parameters)

        from loopy.expression import dtype_to_type_context
        from pymbolic.mapper.stringifier import PREC_NONE
        from pymbolic import var

        c_parameters = [
            expression_to_code_mapper(
                par, PREC_NONE, dtype_to_type_context(target, par_dtype),
                par_dtype).expr
            for par, par_dtype in zip(parameters, par_dtypes)]

        assignee_is_returned = False
        return var(self.name_in_target)(*c_parameters), assignee_is_returned


class PyOP2_Kernel_Lookup(object):

    def __init__(self, name, code, access):
        self.name = name
        self.code = code
        self.access = access

    def __hash__(self):
        return hash(self.name + self.code)

    def __eq__(self, other):
        if isinstance(other, PyOP2_Kernel_Lookup):
            return self.name == other.name and self.code == other.code
        return False

    def __call__(self, target, identifier):
        if identifier == self.name:
            return PyOP2KernelCallable(name=identifier, access=self.access)
        return None


@singledispatch
def replace_materialise(node, self):
    raise AssertionError("Unhandled node type %r" % type(node))


replace_materialise.register(Node)(reuse_if_untouched)


@replace_materialise.register(Materialise)
def replace_materialise_materialise(node, self):
    v = Variable(node.name, node.shape, node.dtype)
    inits = list(map(self, node.children))
    label = node.label
    accs = []
    for rvalue, indices in zip(*(inits[0::2], inits[1::2])):
        lvalue = Indexed(v, indices)
        if isinstance(rvalue, When):
            when, rvalue = rvalue.children
            acc = When(when, Accumulate(label, lvalue, rvalue))
        else:
            acc = Accumulate(label, lvalue, rvalue)
        accs.append(acc)
    self.initialisers.append(tuple(accs))
    return v


def runtime_indices(expressions):
    indices = []
    for node in traversal(expressions):
        if isinstance(node, RuntimeIndex):
            indices.append(node.name)

    return frozenset(indices)


def imperatives(exprs):
    for op in traversal(exprs):
        if isinstance(op, (Accumulate, FunctionCall)):
            yield op


def loop_nesting(instructions, deps, outer_inames, kernel_name):

    nesting = {}

    for insn in imperatives(instructions):
        if isinstance(insn, Accumulate):
            if isinstance(insn.children[1], (Zero, Literal)):
                nesting[insn] = outer_inames
            else:
                nesting[insn] = runtime_indices([insn])
        else:
            assert isinstance(insn, FunctionCall)
            if insn.name in [kernel_name, "MatSetValuesBlockedLocal", "MatSetValuesLocal"]:
                nesting[insn] = outer_inames
            else:
                nesting[insn] = runtime_indices([insn])

    # Take care of dependencies. e.g. t1[i] = A[i], t2[j] = B[t1[j]], then t2 should depends on {i, j}
    name_to_insn = dict((n, i) for i, (n, _) in deps.items())
    for insn, (name, _deps) in deps.items():
        s = set(_deps)
        while s:
            d = s.pop()
            nesting[insn] = nesting[insn] | nesting[name_to_insn[d]]
            s = s | set(deps[name_to_insn[d]][1]) - set([name])

    return nesting


def instruction_dependencies(instructions, initialisers):

    deps = {}
    names = {}
    instructions_by_type = defaultdict(list)
    c = itertools.count()
    for op in imperatives(instructions):
        name = "statement%d" % next(c)
        names[op] = name
        instructions_by_type[type(op.label)].append(op)
        deps[op] = frozenset()

    # read-write dependencies in packing instructions
    def variables(exprs):
        for op in traversal(exprs):
            if isinstance(op, (Argument, Variable)):
                yield op

    writers = defaultdict(list)

    for op in instructions_by_type[PackInst]:
        assert isinstance(op, Accumulate)
        lvalue, _ = op.children
        # Only writes to the outer-most variable
        writes = next(variables([lvalue]))
        if isinstance(writes, Variable):
            writers[writes].append(names[op])

    for op in instructions_by_type[PackInst]:
        _, rvalue = op.children
        deps[op] |= frozenset(x for x in itertools.chain(*(writers[r]for r in variables([rvalue]))))
        deps[op] -= frozenset(names[op])

    # kernel instructions depends on packing instructions
    for op in instructions_by_type[KernelInst]:
        deps[op] |= frozenset(names[o] for o in instructions_by_type[PackInst])

    # unpacking instructions depends on kernel instructions
    for op in instructions_by_type[UnpackInst]:
        deps[op] |= frozenset(names[o] for o in instructions_by_type[KernelInst])

    # add sequential instructions
    for inits in initialisers:
        for i, parent in enumerate(inits[1:], 1):
            for p in imperatives([parent]):
                deps[p] |= frozenset(names[c] for c in imperatives(inits[:i])) - frozenset([name])

    # add name to deps
    deps = dict((op, (names[op], dep)) for op, dep in deps.items())
    return deps


def instruction_names(instructions):
    c = itertools.count()
    names = {}
    for insn in traversal(instructions):
        names[insn] = "statement%d" % next(c)
    return names


def generate(builder, wrapper_name=None):
    parameters = Bag()
    parameters.domains = OrderedDict()
    parameters.assumptions = OrderedDict()
    parameters.wrapper_arguments = builder.wrapper_args
    parameters.conditions = []
    parameters.kernel_data = list(None for _ in parameters.wrapper_arguments)
    parameters.temporaries = OrderedDict()
    parameters.kernel_name = builder.kernel.name

    if builder.layer_index is not None:
        outer_inames = frozenset([builder._loop_index.name,
                                  builder.layer_index.name])
    else:
        outer_inames = frozenset([builder._loop_index.name])

    instructions = list(builder.emit_instructions())

    # replace Materialise
    mapper = Memoizer(replace_materialise)
    mapper.initialisers = []
    instructions = list(mapper(i) for i in instructions)

    # merge indices
    merger = index_merger(instructions)
    instructions = list(merger(i) for i in instructions)
    initialiser = list(itertools.chain(*mapper.initialisers))
    merger = index_merger(initialiser)
    initialiser = list(merger(i) for i in initialiser)
    instructions = instructions + initialiser
    mapper.initialisers = [tuple(merger(i) for i in inits) for inits in mapper.initialisers]

    deps = instruction_dependencies(instructions, mapper.initialisers)
    within_inames = loop_nesting(instructions, deps, outer_inames, parameters.kernel_name)

    # generate loopy
    context = Bag()
    context.parameters = parameters
    context.within_inames = within_inames
    context.conditions = []
    context.index_ordering = []
    context.instruction_dependencies = deps

    statements = list(statement(insn, context) for insn in instructions)
    statements = list(s for s in statements if not isinstance(s, DummyInstruction))

    domains = list(parameters.domains.values())
    if builder.single_cell:
        new_domains = []
        for d in domains:
            if d.get_dim_name(isl.dim_type.set, 0) == "n":
                # n = start
                new_domains.append(d.add_constraint(isl.Constraint.eq_from_names(d.space, {"n": 1, "start": -1})))
            else:
                new_domains.append(d)
        domains = new_domains
        if builder.extruded:
            new_domains = []
            for d in domains:
                if d.get_dim_name(isl.dim_type.set, 0) == "layer":
                    # layer = t1 - 1
                    t1 = builder.layer_extents[1].name
                    new_domains.append(d.add_constraint(isl.Constraint.eq_from_names(d.space, {"layer": 1, t1: -1, 1: 1})))
                else:
                    new_domains.append(d)
        domains = new_domains

    assumptions, = reduce(operator.and_,
                          parameters.assumptions.values()).params().get_basic_sets()
    options = loopy.Options(check_dep_resolution=True)

    # TODO: sometimes masks are not used
    for i, arg in enumerate(parameters.wrapper_arguments):
        if parameters.kernel_data[i] is None:
            arg = loopy.GlobalArg(arg.name, dtype=arg.dtype, shape=arg.shape)
            parameters.kernel_data[i] = arg

    if wrapper_name is None:
        wrapper_name = "wrap_%s" % builder.kernel.name
    wrapper = loopy.make_kernel(domains,
                                statements,
                                kernel_data=parameters.kernel_data,
                                target=loopy.CTarget(),
                                temporary_variables=parameters.temporaries,
                                symbol_manglers=[symbol_mangler],
                                options=options,
                                assumptions=assumptions,
                                lang_version=(2018, 1),
                                name=wrapper_name)
    wrapper = loopy.assume(wrapper, "start < end")
    if builder.extruded:
        t0, t1 = builder.layer_extents
        wrapper = loopy.assume(wrapper, "{0} < {1}".format(t0.name, t1.name))

    for indices in context.index_ordering:
        wrapper = loopy.prioritize_loops(wrapper, indices)

    # {{{ vectorization

    kernel = builder.kernel
    alignment = 64

    headers = set(kernel._headers)
    headers = headers | set(["#include <petsc.h>", "#include <math.h>"])
    preamble = "\n".join(sorted(headers))
    # , "#include <Eigen/Dense>"

    if isinstance(kernel._code, loopy.LoopKernel):
        # register kernel
        knl = kernel._code
        wrapper = loopy.register_callable_kernel(wrapper, knl.name, knl)
        # from loopy.transform.register_callable import (
        #         _match_caller_callee_argument_dimension)
        # wrapper = _match_caller_callee_argument_dimension(wrapper, kernel.name)
        wrapper = loopy.inline_callable_kernel(wrapper, knl.name)
        scoped_functions = wrapper.scoped_functions.copy()
        scoped_functions.update(knl.scoped_functions)
        wrapper = wrapper.copy(scoped_functions=scoped_functions)

        atomic_arg_names = set()
        for insn in wrapper.instructions:
            if isinstance(insn, loopy.Assignment):
                atomic_arg_names.update(atm.var_name for atm in insn.atomicity)

        new_args = wrapper.args.copy()
        new_args = []
        for arg in wrapper.args:
            if isinstance(arg, loopy.ArrayArg):
                new_args.append(arg.copy(
                    for_atomic=arg.name in atomic_arg_names))
            else:
                new_args.append(arg)

        wrapper = wrapper.copy(args=new_args)

    else:
        # kernel is a string
        from coffee.base import Node
        if isinstance(kernel._code, Node):
            code = kernel._code.gencode()
        else:
            code = kernel._code
        wrapper = loopy.register_function_lookup(wrapper, PyOP2_Kernel_Lookup(kernel.name, code, tuple(builder.argument_accesses)))
        preamble = preamble + "\n" + code

    # register petsc functions
    wrapper = loopy.register_function_lookup(wrapper, petsc_function_lookup)

    if builder.batch > 1:
        if builder.extruded:
            outer = "layer"
            inner = "layer_inner"
            wrapper = loopy.assume(wrapper, "t0 mod {0} = 0".format(builder.batch))
            wrapper = loopy.assume(wrapper, "exists zz: zz > 0 and t1 = {0}*zz + t0".format(builder.batch))
        else:
            outer = "n"
            inner = "n_inner"
            wrapper = loopy.assume(wrapper, "start mod {0} = 0".format(builder.batch))
            wrapper = loopy.assume(wrapper, "exists zz: zz > 0 and end = {0}*zz + start".format(builder.batch))

        wrapper = loopy.split_iname(wrapper, outer, builder.batch, inner_tag="ilp.seq", inner_iname=inner)

        # Transpose arguments and temporaries
        # def transpose(tags):
        #     new_tags = ["N0"]
        #     for tag in tags[1:]:
        #         new_tags.append("N{0}".format(tag.layout_nesting_level + 1))
        #     return tuple(new_tags)
        #
        # for arg_name in args_to_batch:
        #     arg = kernel.arg_dict[arg_name]
        #     tags = ["N0"]
        #     kernel = loopy.tag_array_axes(kernel, arg_name, transpose(arg.dim_tags))
        #
        # for tv in kernel.temporary_variables.values():
        #     if tv.initializer is None:
        #         kernel = loopy.tag_array_axes(kernel, tv.name, transpose(tv.dim_tags))
        #
        # kernel = loopy.tag_inames(kernel, {"elem": "ilp.seq"})
    for name in wrapper.temporary_variables:
        tv = wrapper.temporary_variables[name]
        use_opencl = 1
        if not use_opencl:
            wrapper.temporary_variables[name] = tv.copy(alignment=alignment)

    # mark inner most loop as omp simd
    # import os
    # try:
    #     innermost = os.environ["INNERMOST"] == "1"
    # except:
    #     innermost = False

    # if innermost:
    #     kernel = kernel.copy(target=loopy.OpenMPTarget())
    #     innermost_iname = set(kernel.all_inames())
    #     priority, = kernel.loop_priority
    #     priority = priority + ('elem',)
    #     for inst in kernel.instructions:
    #         for iname in list(sorted(inst.within_inames, key=lambda iname: priority.index(iname)))[:-1]:
    #             innermost_iname.discard(iname)
    #
    #     for iname in innermost_iname:
    #         kernel = loopy.tag_inames(kernel, {iname: "l.0"})

    use_opencl = 1

    if not use_opencl:
        preamble = "#include <petsc.h>\n"
        preamble = "#include <math.h>\n" + preamble
        wrapper = wrapper.copy(preambles=[("0", preamble)])

    # vectorization }}}

    # refcount = collect_refcount(instructions)
    # for map_, *_ in builder.maps.values():
    #     if refcount[map_] > 1 or builder.extruded:
    #         knl = loopy.add_prefetch(knl, map_.name,
    #                                  footprint_subscripts=[(pym.Variable(builder._loop_index.name),
    #                                                         pym.Slice((None, )))])
    # print(wrapper)
    # from IPython import embed; embed()
    return wrapper


def argtypes(kernel):
    args = []
    for arg in kernel.args:
        if isinstance(arg, loopy.ValueArg):
            args.append(as_ctypes(arg.dtype))
        elif isinstance(arg, loopy.GlobalArg):
            args.append(ctypes.c_voidp)
        else:
            raise ValueError("Unhandled arg type '%s'" % type(arg))
    return args


def map_to_viennacl_vector(arg_handle, map_to_transfer):
    import re

    c_code_str = r"""#include "petsc.h"
    #include "petscviennacl.h"
    #include <CL/cl.h>

    using namespace std;

    extern "C" cl_mem int_array_to_viennacl_vector(int * __restrict__ map_array, const int map_size, Vec arg)
    {
        viennacl::vector<PetscScalar> *arg_viennacl;
        VecViennaCLGetArrayReadWrite(arg, &arg_viennacl);

        viennacl::ocl::context ctx = arg_viennacl->handle().opencl_handle().context();

        cl_mem map_opencl = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, map_size*sizeof(cl_int), map_array, NULL);

        VecViennaCLRestoreArrayReadWrite(arg, &arg_viennacl);
        return map_opencl;
    }
    """

    # remove the whitespaces for pretty printing
    code_to_compile = re.sub("\\n    ", "\n", c_code_str)

    # If we weren't in the cache we /must/ have arguments
    from pyop2.utils import get_petsc_dir
    import coffee.system
    from pyop2.sequential import JITModule
    from pyop2 import compilation
    import os
    import ctypes
    import numpy as np

    compiler = coffee.system.compiler
    extension = "cpp"
    cppargs = JITModule._cppargs
    cppargs += ["-I%s/include" % d for d in get_petsc_dir()] + \
               ["-I%s" % os.path.abspath(os.path.dirname(__file__))]
    if compiler:
        cppargs += [compiler[coffee.system.isa['inst_set']]]
    ldargs = ["-L%s/lib" % d for d in get_petsc_dir()] + \
             ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()] + \
             ["-lpetsc", "-lm"]
    fun = compilation.load(code_to_compile,
                                 extension,
                                 'int_array_to_viennacl_vector',
                                 cppargs=cppargs,
                                 ldargs=ldargs,
                                 argtypes=(ctypes.c_void_p, ctypes.c_int,
                                     ctypes.c_void_p),
                                 restype=ctypes.c_void_p,
                                 compiler=compiler.get('name'),
                                 comm=None)
    map_ocl = fun(map_to_transfer._kernel_args_[0],
            np.int32(np.product(map_to_transfer.shape)), arg_handle)

    return map_ocl


def get_cl_context(kernel, argtypes, comm):
    c_code_str = (
            r"""
            #include <CL/cl.h>

            #include "petsc.h"
            #include "petscvec.h"
            #include "petscviennacl.h"

            extern "C" cl_context get_cl_context(Vec arg) {

                const viennacl::vector<PetscScalar> *arg_viennacl;
                VecViennaCLGetArrayRead(arg, &arg_viennacl);

                cl_context ctx =
                        arg_viennacl->handle().opencl_handle().context().handle().get();

                VecViennaCLRestoreArrayRead(arg, &arg_viennacl);

                return ctx;
            }
            """
            )

    for i, arg in enumerate(kernel.args):
        if isinstance(arg, loopy.ArrayArg) and not (
                arg.dtype.is_integral()):
            arg_pos = i
            break

    # remove the whitespaces for pretty printing
    import re
    code_to_compile = re.sub("\\n        ", "\n", c_code_str)

    # If we weren't in the cache we /must/ have arguments
    from pyop2.utils import get_petsc_dir
    import coffee.system
    from pyop2.sequential import JITModule
    from pyop2 import compilation
    import os
    import ctypes

    compiler = coffee.system.compiler
    extension = "cpp"
    cppargs = JITModule._cppargs
    cppargs += ["-I%s/include" % d for d in get_petsc_dir()] + \
               ["-I%s" % os.path.abspath(os.path.dirname(__file__))]
    if compiler:
        cppargs += [compiler[coffee.system.isa['inst_set']]]
    ldargs = ["-L%s/lib" % d for d in get_petsc_dir()] + \
             ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()] + \
             ["-lpetsc", "-lm"]
    fun = compilation.load(code_to_compile,
            extension,
            'get_cl_context',
            cppargs=cppargs,
            ldargs=ldargs,
            argtypes=(argtypes[arg_pos], ),
            restype=ctypes.c_void_p,
            compiler=compiler.get('name'),
            comm=comm)

    return fun


def get_viennacl_kernel(kernel, argtypes, comm):
    kernel = transform_for_opencl(kernel)
    from mako.template import Template
    import pyopencl as cl
    ctx = cl.create_some_context()
    # TODO: somehow pull the device from the petsc vec for the preprocess
    # checks.
    kernel = kernel.copy(
            target=loopy.PyOpenCLTarget(ctx.devices[0]))

    lsize, gsize = get_grid_sizes(kernel)
    if not lsize:
        # lsize has not been set => run serially on a GPU
        # both gsize and lsize should be 1
        lsize = (1, )
        assert not gsize
        gsize = (1, )
    else:
        # if lsize if set then currently assuming that the parallelization is
        # always over the number of elements.
        gsize = ('(end-start)', )

    c_code_str = (
        r'''
        <%! import loopy as lp %>
        #include <CL/cl.h>
        #include "petsc.h"
        #include "petscvec.h"
        #include "petscviennacl.h"
        #include "viennacl/ocl/backend.hpp"
        #include "viennacl/vector.hpp"
        #include "viennacl/backend/memory.hpp"
        #include <cstdlib>
        // #include <iostream>

        using namespace std;

        char kernel_source[] = "${kernel_src}";
        extern "C" cl_kernel ${kernel_name}_vcl_knl_extractor(${", ".join(
                            [('Vec ' + arg.name)  if isinstance(arg,
                            lp.ArrayArg) and not arg.dtype.is_integral() else
                            ('cl_mem ' + arg.name + ', const int ' +
                            arg.name + '_size') if isinstance(arg,
                            lp.ArrayArg) and arg.dtype.is_integral() else
                            str(arg.get_arg_decl(ast_builder))[:-1]
                            for arg in args])})
        {


            // viennacl vector declarations
            % for arg in args:
            % if isinstance(arg, lp.ArrayArg) and not arg.dtype.is_integral():
            viennacl::vector<PetscScalar> *${arg.name}_viennacl;
            % endif
            % endfor

            // getting the array from the petsc vecs
            % for arg in args:
            % if isinstance(arg, lp.ArrayArg) and not arg.dtype.is_integral():
            VecViennaCLGetArrayReadWrite(${arg.name}, &${arg.name}_viennacl);
            % endif
            % endfor


            // defining the context
            viennacl::ocl::context ctx =
                    ${[arg for arg in args if isinstance(arg,
                    lp.ArrayArg) and not
                    arg.dtype.is_integral()][0].name}_viennacl->handle().opencl_handle().context();

            // declaring the int arrays(if any..)
            % for arg in args:
            % if isinstance(arg, lp.ArrayArg) and arg.dtype.is_integral():
            viennacl::vector<cl_int> ${arg.name}_viennacl(${arg.name}, ${
                    arg.name}_size);
            % endif
            % endfor

            viennacl::ocl::program & my_prog =
                        ctx.add_program(kernel_source,
                        "kernel_program_${kernel_name}");
            viennacl::ocl::kernel & viennacl_kernel =
                    my_prog.get_kernel("${kernel_name}");

            % for arg in args:
            % if isinstance(arg, lp.ArrayArg) and not arg.dtype.is_integral():
            VecViennaCLRestoreArrayReadWrite(${arg.name}, &${arg.name}_viennacl);
            % endif
            % endfor

            cl_kernel cl_knl = viennacl_kernel.handle().get();
            clRetainKernel(cl_knl);
            return cl_knl;
        }''')

    # remove the whitespaces for pretty printing
    import re
    c_code = Template(re.sub("\\n        ", "\n", c_code_str))
    kernel_src = loopy.generate_code_v2(kernel).device_code().replace('\n', '\\n"\n"')
    code_to_compile = c_code.render(
            kernel_src=kernel_src,
            kernel_name=kernel.name,
            ast_builder=kernel.target.get_device_ast_builder(),
            args=kernel.args)

    # If we weren't in the cache we /must/ have arguments
    from pyop2.utils import get_petsc_dir
    import coffee.system
    from pyop2.sequential import JITModule
    from pyop2 import compilation
    import os
    import ctypes

    compiler = coffee.system.compiler
    extension = "cpp"
    cppargs = JITModule._cppargs
    cppargs += ["-I%s/include" % d for d in get_petsc_dir()] + \
               ["-I%s" % os.path.abspath(os.path.dirname(__file__))]
    if compiler:
        cppargs += [compiler[coffee.system.isa['inst_set']]]
    ldargs = ["-L%s/lib" % d for d in get_petsc_dir()] + \
             ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()] + \
             ["-lpetsc", "-lm"]
    fun = compilation.load(code_to_compile,
            extension,
            kernel.name+'_vcl_knl_extractor',
            cppargs=cppargs,
            ldargs=ldargs,
            argtypes=argtypes,
            restype=ctypes.c_void_p,
            compiler=compiler.get('name'),
            comm=comm)
    return fun


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
        warn("could not check axis bounds because no value "
                "for variable '%s' was passed to check_kernels()"
                % name)

    return llens, glens


def transform_for_opencl(kernel):
    """
    Performs transformations on kernels.
    """

    def insn_needs_atomic(insn):
        # TODO: assumes that the assignee is always a subscript
        assignee_name = insn.assignee.aggregate.name
        return (
                assignee_name in insn.read_dependency_names() and
                assignee_name not in kernel.temporary_variables)

    new_insns = []
    args_marked_for_atomic = set()
    for insn in kernel.instructions:
        if ('pyop2_assign' in insn.tags) or (
                'tsfc_return_accumulate' in insn.tags):

            if insn_needs_atomic(insn):
                atomicity = (loopy.AtomicUpdate(insn.assignee.aggregate.name), )
                insn = insn.copy(atomicity=atomicity)
                args_marked_for_atomic |= set([insn.assignee.aggregate.name])

        new_insns.append(insn)

    new_args = []
    for arg in kernel.args:
        if arg.name in args_marked_for_atomic:
            new_args.append(arg.copy(for_atomic=True))
        else:
            new_args.append(arg)

    kernel = kernel.copy(instructions=new_insns,
            args=new_args)

    LOCAL_SIZE = 64

    if kernel.name in ['wrap_zero', 'wrap_copy']:
        kernel = loopy.split_iname(kernel, "n", 1, inner_tag="l.0",
                outer_tag="g.0")
    else:
        kernel = loopy.split_iname(kernel, "n", LOCAL_SIZE, inner_tag="l.0",
                outer_tag="g.0")

    return kernel


def generate_viennacl_code(kernel):
    kernel = transform_for_opencl(kernel)
    import pyopencl as cl
    import re
    from mako.template import Template
    lsize, gsize = get_grid_sizes(kernel)
    if not lsize:
        # lsize has not been set => run serially on a GPU
        # both gsize and lsize should be 1
        lsize = (1, )
        assert not gsize
        gsize = (1, )
    else:
        # if lsize if set then currently assuming that the parallelization is
        # always over the number of elements.
        gsize = ('(end-start)', )

    ctx = cl.create_some_context()
    # TODO: somehow pull the device from the petsc vec for the preprocess
    # checks.
    kernel = kernel.copy(
            target=loopy.PyOpenCLTarget(ctx.devices[0]))

    c_code_str = r'''#include <CL/cl.hpp>
            #include "petsc.h"
            #include "petscvec.h"
            #include "petscviennacl.h"
            #include <iostream>
            #include <sys/time.h>
            #include <cstring>

            <%! import loopy as lp %>

            // ViennaCL Headers
            #include "viennacl/ocl/backend.hpp"
            #include "viennacl/vector.hpp"
            #include "viennacl/backend/memory.hpp"
            #define TIME_DIFF(t2, t1) ((t2).tv_sec - (t1).tv_sec + ((t2).tv_usec - (t1).tv_usec)*1e-6)

            using namespace std;

            extern "C" void ${kernel_name}(cl_kernel cl_knl,
                    ${", ".join(
                        [('Vec ' + arg.name)  if isinstance(arg,
                        lp.ArrayArg) and not arg.dtype.is_integral() else
                        ('cl_mem ' + arg.name + ', const int ' +
                        arg.name + '_size') if isinstance(arg,
                        lp.ArrayArg) and arg.dtype.is_integral() else
                        str(arg.get_arg_decl(ast_builder))[:-1]
                        for arg in args])})
            {
                if(end == start)
                {
                    // no need to go any further
                    return;
                }


                // viennacl vector declarations
                % for arg in args:
                % if isinstance(arg, lp.ArrayArg) and not arg.dtype.is_integral():
                viennacl::vector<PetscScalar> *${arg.name}_viennacl;
                % endif
                % endfor

                // getting the array from the petsc vecs
                % for arg in args:
                % if isinstance(arg, lp.ArrayArg) and not arg.dtype.is_integral():
                VecViennaCLGetArrayReadWrite(${arg.name}, &${arg.name}_viennacl);
                % endif
                % endfor

                // defining the context
                viennacl::ocl::context ctx =
                        ${[arg for arg in args if isinstance(arg,
                        lp.ArrayArg) and not
                        arg.dtype.is_integral()][0].name}_viennacl->handle().opencl_handle().context();



                // set the kernel args which are changing
                % for i, arg in enumerate(args):
                % if isinstance(arg, lp.ValueArg):

                clSetKernelArg(cl_knl, ${i}, ${arg.dtype.itemsize}, &${arg.name});

                % elif isinstance(arg, lp.ArrayArg):
                % if arg.dtype.is_integral():
                clSetKernelArg(cl_knl, ${i}, sizeof(${arg.name}), &${arg.name});

                % else:
                clSetKernelArg(cl_knl, ${i}, sizeof(PetscScalar),
                        &(${arg.name}_viennacl->handle().opencl_handle()));
                % endif
                % else:
                    <% raise RuntimeError("Unknown type of arg.") %>
                % endif
                % endfor


                // getting the queue
                cl_command_queue queue= ctx.get_queue().handle().get();

                // set work group sizes
                size_t lwg_size[${len(lsize)}];
                size_t gwg_size[${len(lsize)}];
                % for i, ls in enumerate(lsize):
                lwg_size[${i}] = ${lsize[i]};
                % endfor

                % for i, ls in enumerate(gsize):
                gwg_size[${i}] = ${gsize[i]};
                % endfor

                clFinish(queue);

                // enqueueing the kernel
                clEnqueueNDRangeKernel(queue, cl_knl, ${len(lsize)}, NULL,
                        gwg_size, lwg_size, 0, NULL, NULL);

                clFinish(queue);

                // restoring the arrays to the petsc vecs
                % for arg in args:
                % if isinstance(arg, lp.ArrayArg) and not arg.dtype.is_integral():
                VecViennaCLRestoreArrayReadWrite(${arg.name}, &${arg.name}_viennacl);
                % endif
                % endfor
            }
            '''

    # remove the whitespaces for pretty printing
    c_code_str = re.sub("\\n            ", "\n", c_code_str)

    c_code = Template(c_code_str)

    kernel_src = loopy.generate_code_v2(kernel).device_code().replace('\n',
            '\\n"\n"')

    krr =  c_code.render(
        kernel_src=kernel_src,
        kernel_name=kernel.name,
        ast_builder=kernel.target.get_device_ast_builder(),
        args=kernel.args,
        lsize=lsize,
        gsize=gsize)

    return krr


@singledispatch
def statement(expr, context):
    raise AssertionError("Unhandled statement type '%s'" % type(expr))


@statement.register(DummyInstruction)
def statement_dummy(expr, context):
    new_children = tuple(expression(c, context.parameters) for c in expr.children)
    return DummyInstruction(expr.label, new_children)


@statement.register(When)
def statement_when(expr, context):
    condition, stmt = expr.children
    context.conditions.append(expression(condition, context.parameters))
    stmt = statement(stmt, context)
    context.conditions.pop()
    return stmt


@statement.register(Accumulate)
def statement_assign(expr, context):
    lvalue, _ = expr.children
    if isinstance(lvalue, Indexed):
        context.index_ordering.append(tuple(i.name for i in lvalue.index_ordering()))
    lvalue, rvalue = tuple(expression(c, context.parameters) for c in expr.children)
    within_inames = context.within_inames[expr]

    id, depends_on = context.instruction_dependencies[expr]
    predicates = frozenset(context.conditions)
    insn = loopy.Assignment(lvalue, rvalue, within_inames=within_inames,
                            predicates=predicates,
                            id=id,
                            depends_on=depends_on, depends_on_is_final=True,
                            tags=frozenset(['pyop2_assign']))

    return insn


@statement.register(FunctionCall)
def statement_functioncall(expr, context):
    # FIXME: function manglers
    parameters = context.parameters

    from loopy.symbolic import SubArrayRef
    from loopy.types import OpaqueType

    # children = list(expression(c, parameters) for c in expr.children)
    # call = pym.Call(pym.Variable(name), children)
    free_indices = set(i.name for i in expr.free_indices)
    writes = []
    reads = []
    for access, child in zip(expr.access, expr.children):
        var = expression(child, parameters)
        if isinstance(var, pym.Subscript):
            # tensor argument
            indices = []
            sweeping_indices = []
            for index in var.index_tuple:
                indices.append(index)
                if isinstance(index, pym.Variable) and index.name in free_indices:
                    sweeping_indices.append(index)
            arg = SubArrayRef(tuple(sweeping_indices), var)
        else:
            # scalar argument or constant
            arg = var
        if access is READ or (isinstance(child, Argument) and isinstance(child.dtype, OpaqueType)):
            # opaque data type treated as read
            reads.append(arg)
        else:
            writes.append(arg)

    within_inames = context.within_inames[expr]
    predicates = frozenset(context.conditions)
    id, depends_on = context.instruction_dependencies[expr]

    call = pym.Call(pym.Variable(expr.name), tuple(reads))

    return loopy.CallInstruction(tuple(writes), call,
                                 within_inames=within_inames,
                                 predicates=predicates,
                                 id=id,
                                 depends_on=depends_on,
                                 depends_on_is_final=True)


@singledispatch
def expression(expr, parameters):
    raise AssertionError("Unhandled expression type '%s'" % type(expr))


@expression.register(Index)
def expression_index(expr, parameters):
    name = expr.name
    if name not in parameters.domains:
        vars = isl.make_zero_and_vars([name])
        zero = vars[0]
        domain = (vars[name].ge_set(zero) & vars[name].lt_set(zero + expr.extent))
        parameters.domains[name] = domain
    return pym.Variable(name)


@expression.register(FixedIndex)
def expression_fixedindex(expr, parameters):
    return expr.value


@expression.register(RuntimeIndex)
def expression_runtimeindex(expr, parameters):
    @singledispatch
    def translate(expr, vars):
        raise AssertionError("Unhandled type '%s' in domain translation" % type(expr))

    @translate.register(Sum)
    def translate_sum(expr, vars):
        return operator.add(*(translate(c, vars) for c in expr.children))

    @translate.register(Argument)
    def translate_argument(expr, vars):
        expr = expression(expr, parameters)
        return vars[expr.name]

    @translate.register(Variable)
    def translate_variable(expr, vars):
        return vars[expr.name]

    @translate.register(Zero)
    def translate_zero(expr, vars):
        assert expr.shape == ()
        return vars[0]

    @translate.register(LogicalAnd)
    def translate_logicaland(expr, vars):
        a, b = (translate(c, vars) for c in expr.children)
        return a & b

    @translate.register(Comparison)
    def translate_comparison(expr, vars):
        a, b = (translate(c, vars) for c in expr.children)
        fn = {">": "gt_set",
              ">=": "ge_set",
              "==": "eq_set",
              "!=": "ne_set",
              "<": "lt_set",
              "<=": "le_set"}[expr.operator]
        return getattr(a, fn)(b)

    name = expr.name
    if name not in parameters.domains:
        lo, hi, constraint = expr.children
        params = list(v.name for v in traversal([lo, hi]) if isinstance(v, (Argument, Variable)))
        vars = isl.make_zero_and_vars([name], params)
        domain = (vars[name].ge_set(translate(lo, vars)) &
                  vars[name].lt_set(translate(hi, vars)))
        parameters.domains[name] = domain
        if constraint is not None:
            parameters.assumptions[name] = translate(constraint, vars)
    return pym.Variable(name)


@expression.register(MultiIndex)
def expression_multiindex(expr, parameters):
    return tuple(expression(c, parameters) for c in expr.children)


@expression.register(Extent)
def expression_extent(expr, parameters):
    # FIXME: There will be a symbolic representation of this.
    multiindex, = expr.children
    return int(numpy.prod(tuple(i.extent for i in multiindex)))


@expression.register(Symbol)
def expression_symbol(expr, parameters):
    # FIXME: symbol manglers!
    return pym.Variable(expr.name)


@expression.register(Argument)
def expression_argument(expr, parameters):
    name = expr.name
    shape = expr.shape
    dtype = expr.dtype
    if shape == ():
        arg = loopy.ValueArg(name, dtype=dtype)
    else:
        arg = loopy.GlobalArg(name,
                              dtype=dtype,
                              shape=shape)
    idx = parameters.wrapper_arguments.index(expr)
    parameters.kernel_data[idx] = arg
    return pym.Variable(name)


@expression.register(Variable)
def expression_variable(expr, parameters):
    name = expr.name
    shape = expr.shape
    dtype = expr.dtype
    if name not in parameters.temporaries:
        parameters.temporaries[name] = loopy.TemporaryVariable(name,
                                                               dtype=dtype,
                                                               shape=shape,
                                                               scope=loopy.auto)
    return pym.Variable(name)


@expression.register(Zero)
def expression_zero(expr, parameters):
    assert expr.shape == ()
    return 0
    # return loopy.symbolic.TypeCast(expr.dtype, expr.dtype.type(0))


@expression.register(Literal)
def expression_literal(expr, parameters):
    assert expr.shape == ()
    if expr.casting:
        return loopy.symbolic.TypeCast(expr.dtype, expr.value)
    return expr.value


@expression.register(NamedLiteral)
def expression_namedliteral(expr, parameters):
    name = expr.name
    val = loopy.TemporaryVariable(name,
                                  dtype=expr.dtype,
                                  shape=expr.shape,
                                  scope=loopy.temp_var_scope.GLOBAL,
                                  read_only=True,
                                  initializer=expr.value)
    parameters.temporaries[name] = val

    return pym.Variable(name)


@expression.register(Conditional)
def expression_conditional(expr, parameters):
    return pym.If(*(expression(c, parameters) for c in expr.children))


@expression.register(Comparison)
def expression_comparison(expr, parameters):
    l, r = (expression(c, parameters) for c in expr.children)
    return pym.Comparison(l, expr.operator, r)


@expression.register(LogicalNot)
@expression.register(BitwiseNot)
def expression_uop(expr, parameters):
    child, = (expression(c, parameters) for c in expr.children)
    return {LogicalNot: pym.LogicalNot,
            BitwiseNot: pym.BitwiseNot}[type(expr)](child)


@expression.register(Sum)
@expression.register(Product)
@expression.register(LogicalAnd)
@expression.register(LogicalOr)
@expression.register(BitwiseAnd)
@expression.register(BitwiseOr)
def expression_binop(expr, parameters):
    children = tuple(expression(c, parameters) for c in expr.children)
    return {Sum: pym.Sum,
            Product: pym.Product,
            LogicalOr: pym.LogicalOr,
            LogicalAnd: pym.LogicalAnd,
            BitwiseOr: pym.BitwiseOr,
            BitwiseAnd: pym.BitwiseAnd}[type(expr)](children)


@expression.register(BitShift)
def expression_bitshift(expr, parameters):
    children = (expression(c, parameters) for c in expr.children)
    return {"<<": pym.LeftShift,
            ">>": pym.RightShift}[expr.direction](*children)


@expression.register(Indexed)
def expression_indexed(expr, parameters):
    aggregate, multiindex = (expression(c, parameters) for c in expr.children)
    return pym.Subscript(aggregate, multiindex)
    extents = [int(numpy.prod(expr.aggregate.shape[i+1:])) for i in range(len(multiindex))]
    make_sum = lambda x, y: pym.Sum((x, y))
    index = reduce(make_sum, [pym.Product((e, m)) for e, m in zip(extents, multiindex)])
    return pym.Subscript(aggregate, (index,))
