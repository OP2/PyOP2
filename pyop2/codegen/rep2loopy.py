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
                                          Conditional, Comparison,
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
            # FIXME: assume 1D arrays as arguments for now
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
        # f(a,b,c,d) instead of a = f(b,c,d)

        from loopy.kernel.instruction import CallInstruction

        # assert self.is_ready_for_codegen()
        assert isinstance(insn, CallInstruction)

        # FIXME: this is not totally correct
        parameters = insn.assignees + insn.expression.parameters
        par_dtypes = tuple(expression_to_code_mapper.infer_type(p) for p in parameters)
        # par_dtype.extend([self.arg_id_to_dtype[i] for i, _ in enumerate(insn.expression.p)])

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


def pyop2_kernel_lookup(target, identifier):
    if identifier[:13] == "pyop2_kernel_":
        return PyOP2KernelCallable(name=identifier)

    return None

# def petsc_function_mangler(kernel, name, arg_dtypes):
#     if name == "CHKERRQ":
#         return loopy.CallMangleInfo(name, (), arg_dtypes)
#     if name in {"MatSetValuesBlockedLocal", "MatSetValuesLocal"}:
#         return loopy.CallMangleInfo(name, (), arg_dtypes)


# }}}


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

    # Take care of dependencies. e.g. t1[i] = A[i], t2[j] = B[t1[j]]
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
        deps[op] = (name, frozenset())

    # kernel instructions depends on packing instructions
    for op in instructions_by_type[KernelInst]:
        name, depends_on = deps[op]
        depends_on = depends_on | frozenset(names[o] for o in instructions_by_type[PackInst])
        deps[op] = (name, depends_on)

    # unpacking instructions depends on kernel instructions
    for op in instructions_by_type[UnpackInst]:
        name, depends_on = deps[op]
        depends_on = depends_on | frozenset(names[o] for o in instructions_by_type[KernelInst])
        deps[op] = (name, depends_on)

    # add sequential instructions
    for inits in initialisers:
        for i, parent in enumerate(inits[1:], 1):
            for p in imperatives([parent]):
                name, depends_on = deps[p]
                depends_on = depends_on | frozenset(names[c] for c in imperatives(inits[:i])) - frozenset([name])
                deps[p] = (name, depends_on)

    return deps


def instruction_names(instructions):
    c = itertools.count()
    names = {}
    for insn in traversal(instructions):
        names[insn] = "statement%d" % next(c)
    return names


def generate(builder):
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
    # instructions = list(builder.emit_instructions())

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
    if not parameters.domains:
        # No nodes use runtime index, e.g. when kernel is only on globals
        expression(builder._loop_index, context.parameters)
        if builder.layer_index:
            expression(builder.layer_index, context.parameters)
    domains = list(parameters.domains.values())
    assumptions, = reduce(operator.and_,
                          parameters.assumptions.values()).params().get_basic_sets()
    options = loopy.Options(check_dep_resolution=True)

    # TODO: sometimes masks are not used
    for i, arg in enumerate(parameters.wrapper_arguments):
        if parameters.kernel_data[i] is None:
            arg = loopy.GlobalArg(arg.name, dtype=arg.dtype, shape=arg.shape)
            parameters.kernel_data[i] = arg

    wrapper = loopy.make_kernel(domains,
                                statements,
                                kernel_data=parameters.kernel_data,
                                target=loopy.CTarget(),
                                temporary_variables=parameters.temporaries,
                                # function_manglers=[petsc_function_mangler, kernel_mangler],
                                symbol_manglers=[symbol_mangler],
                                options=options,
                                assumptions=assumptions,
                                lang_version=(2018, 1),
                                name="wrap_%s" % builder.kernel.name)

    # from IPython import embed; embed()
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
        # from loopy.transform.register_callable import _match_caller_callee_argument_dimension
        # wrapper = _match_caller_callee_argument_dimension(wrapper, kernel.name)
        wrapper = loopy.inline_callable_kernel(wrapper, knl.name)
        scoped_functions = wrapper.scoped_functions.copy()
        scoped_functions.update(knl.scoped_functions)
        wrapper = wrapper.copy(scoped_functions=scoped_functions)
    else:
        # kernel is a string
        wrapper = loopy.register_function_lookup(wrapper, pyop2_kernel_lookup)
        preamble = preamble + "\n" + kernel._code

    # register petsc functions
    wrapper = loopy.register_function_lookup(wrapper, petsc_function_lookup)

    # wrapper = loopy.inline_kernel(wrapper, kernel.name)

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


def prepare_arglist(iterset, *args):
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
                arglist += (k, )
                seen.add(k)
    return arglist


def set_argtypes(iterset, *args):
    from pyop2.datatypes import IntType, as_ctypes
    index_type = as_ctypes(IntType)
    argtypes = (index_type, index_type)
    argtypes += iterset._argtypes_
    for arg in args:
        argtypes += arg._argtypes_
    seen = set()
    for arg in args:
        maps = arg.map_tuple
        for map_ in maps:
            for k, t in zip(map_._kernel_args_, map_._argtypes_):
                if k in seen:
                    continue
                argtypes += (t, )
                seen.add(k)
    return argtypes


def prepare_cache_key(kernel, iterset, *args):
    from pyop2 import Subset
    counter = itertools.count()
    seen = defaultdict(lambda: next(counter))
    key = (kernel._wrapper_cache_key_ + iterset._wrapper_cache_key_ +
           (iterset._extruded, (iterset._extruded and iterset.constant_layers), isinstance(iterset, Subset)))

    for arg in args:
        key += arg._wrapper_cache_key_
        maps = arg.map_tuple
        for map_ in maps:
            key += (seen[map_], )
    return key


@singledispatch
def statement(expr, context):
    raise AssertionError("Unhandled statement type '%s'" % type(expr))


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
    return loopy.Assignment(lvalue, rvalue, within_inames=within_inames,
                            predicates=predicates,
                            id=id,
                            depends_on=depends_on, depends_on_is_final=True)


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
                                 depends_on=depends_on, depends_on_is_final=True)


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
    return loopy.symbolic.TypeCast(expr.dtype, expr.value)


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
