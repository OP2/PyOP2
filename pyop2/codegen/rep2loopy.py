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

from pyop2.base import READ, WRITE
from pyop2.datatypes import as_ctypes

from pyop2.codegen.optimise import merge_indices

from pyop2.codegen.representation import (Index, FixedIndex, RuntimeIndex,
                                          MultiIndex, Extent, Indexed,
                                          BitShift, BitwiseNot, BitwiseAnd, BitwiseOr,
                                          Conditional, Comparison,
                                          LogicalNot, LogicalAnd, LogicalOr,
                                          Materialise, Accumulate, FunctionCall, When,
                                          Argument, Variable, Literal, NamedLiteral,
                                          Symbol, Zero,
                                          Sum, Product)


class Bag(object):
    pass


def symbol_mangler(kernel, name):
    if name in {"ADD_VALUES", "INSERT_VALUES"}:
        return loopy.types.to_loopy_type(numpy.int32), name
    return None


def petsc_function_mangler(kernel, name, arg_dtypes):
    if name == "CHKERRQ":
        return loopy.CallMangleInfo(name, (), arg_dtypes)
    if name in {"MatSetValuesBlockedLocal", "MatSetValuesLocal"}:
        return loopy.CallMangleInfo(name, (), arg_dtypes)


@singledispatch
def replace_materialise(node, self):
    raise AssertionError("Unhandled node type %r" % type(node))


replace_materialise.register(Node)(reuse_if_untouched)


@replace_materialise.register(Materialise)
def replace_materialise_materialise(node, self):
    v = Variable(node.name, node.shape, node.dtype)
    inits = list(map(self, node.children))
    accs = []
    for rvalue, indices in zip(*(inits[0::2], inits[1::2])):
        lvalue = Indexed(v, indices)
        if isinstance(rvalue, When):
            when, rvalue = rvalue.children
            acc = When(when, Accumulate(lvalue, rvalue))
        else:
            acc = Accumulate(lvalue, rvalue)
        accs.append(acc)
    self.initialisers.append(tuple(accs))
    return v


def preprocess(instructions):
    mapper = Memoizer(replace_materialise)
    mapper.initialisers = []
    instructions = list(merge_indices(mapper(i) for i in instructions))
    prefix_cache = {}
    initialisers = list(itertools.chain(*(merge_indices(i, cache=prefix_cache)
                                          for i in mapper.initialisers)))
    return initialisers + instructions


def runtime_indices(expressions):
    indices = []
    for node in traversal(expressions):
        if isinstance(node, RuntimeIndex):
            indices.append(node.name)

    indices = frozenset(indices)


def outer_loop_nesting(instructions, outer_inames, kernel_name):
    nesting = {}
    for insn in traversal(instructions):
        indices = runtime_indices([insn])
        if isinstance(insn, Accumulate):
            if isinstance(insn.children[1], (Zero, Literal)):
                nesting[insn] = outer_inames
            else:
                nesting[insn] = indices
        elif isinstance(insn, FunctionCall):
            if insn.name == kernel_name:
                nesting[insn] = outer_inames
            else:
                nesting[insn] = indices
        else:
            continue
    return nesting


def instruction_dependencies(instructions):
    def variables(exprs):
        for op in traversal(exprs):
            if isinstance(op, (Argument, Variable)):
                yield op

    writers = defaultdict(list)
    deps = {}
    names = {}
    c = itertools.count()
    for op in traversal(instructions):
        if isinstance(op, Accumulate):
            name = "statement%d" % next(c)
            lvalue, _ = op.children
            # Only writes to the outer-most variable
            writes = next(variables([lvalue]))
            writers[writes].append(name)
            names[op] = name
        elif isinstance(op, FunctionCall):
            name = "statement%d" % next(c)
            names[op] = name
            for access, arg in zip(op.access, op.children):
                if access is not READ:
                    writes = next(variables([arg]))
                    writers[writes].append(name)

    for op, name in names.items():
        if isinstance(op, Accumulate):
            _, rvalue = op.children
            depends_on = frozenset(itertools.chain(*(writers[r]
                                                     for r in variables([rvalue]))))
        else:
            depends_on = []
            for access, arg in zip(op.access, op.children):
                if access is not WRITE:
                    depends_on.extend(x for x in
                                      itertools.chain(*(writers[r]
                                                        for r in variables([arg])))
                                      if x is not name)
            depends_on = frozenset(depends_on)
        deps[op] = (name, depends_on)
    return deps


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

    instructions = preprocess(builder.emit_instructions())

    context = Bag()
    context.parameters = parameters
    context.within_inames = outer_loop_nesting(instructions, outer_inames,
                                               parameters.kernel_name)
    context.conditions = []
    context.index_ordering = []

    context.instruction_dependencies = instruction_dependencies(instructions)
    statements = list(statement(insn, context) for insn in instructions)

    def kernel_mangler(kernel, name, arg_dtypes):
        rettypes = []
        if name == builder.kernel.name:
            for arg, access in zip(builder.arguments, builder.argument_accesses):
                if access is not READ:
                    rettypes.append(loopy.types.to_loopy_type(arg.dtype))
            return loopy.CallMangleInfo(name, tuple(rettypes), arg_dtypes)

    domains = list(parameters.domains.values())
    assumptions, = reduce(operator.and_,
                          parameters.assumptions.values()).params().get_basic_sets()
    options = loopy.Options(check_dep_resolution=True)

    knl = loopy.make_kernel(domains,
                            statements,
                            kernel_data=parameters.kernel_data,
                            target=loopy.CTarget(),
                            temporary_variables=parameters.temporaries,
                            function_manglers=[petsc_function_mangler, kernel_mangler],
                            symbol_manglers=[symbol_mangler],
                            options=options,
                            assumptions=assumptions,
                            name="wrap_%s" % builder.kernel.name,
                            preambles=[(0, builder.kernel.code())])

    for indices in context.index_ordering:
        knl = loopy.prioritize_loops(knl, indices)
    # refcount = collect_refcount(instructions)
    # for map_, *_ in builder.maps.values():
    #     if refcount[map_] > 1 or builder.extruded:
    #         knl = loopy.add_prefetch(knl, map_.name,
    #                                  footprint_subscripts=[(pym.Variable(builder._loop_index.name),
    #                                                         pym.Slice((None, )))])
    return knl


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
        maps = arg.map
        if maps is None:
            continue
        for map_ in maps:
            karg = map_._kernel_args_
            if karg in seen:
                continue
            arglist += karg
            seen.add(karg)
    return arglist


@singledispatch
def statement(expr, parameters):
    raise AssertionError("Unhandled type")


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

    statement_id, depends_on = context.instruction_dependencies[expr]
    predicates = frozenset(context.conditions)
    return loopy.Assignment(lvalue, rvalue, within_inames=within_inames,
                            predicates=predicates,
                            id=statement_id,
                            depends_on=depends_on)


@statement.register(FunctionCall)
def statement_functioncall(expr, context):
    # FIXME: function manglers
    parameters = context.parameters
    name = expr.name
    children = list(expression(c, parameters) for c in expr.children)

    call = pym.Call(pym.Variable(name), children)

    written = []
    for access, child in zip(expr.access, children):
        if access is not READ:
            written.append(child)

    within_inames = context.within_inames[expr]
    predicates = frozenset(context.conditions)

    statement_id, depends_on = context.instruction_dependencies[expr]
    return loopy.CallInstruction(tuple(written), call,
                                 within_inames=within_inames,
                                 predicates=predicates,
                                 id=statement_id,
                                 depends_on=depends_on)


@singledispatch
def expression(expr, parameters):
    raise AssertionError("Unhandled type")


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
        raise AssertionError

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
        params = list(v.name for v in traversal([lo, hi])
                      if isinstance(v, (Argument, Variable)))
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
    return loopy.symbolic.TypeCast(expr.dtype, expr.dtype.type(0))


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
    condition, then, else_ = expr.children
    condition = expression(condition, parameters)
    then = expression(then, parameters)
    else_ = expression(else_, parameters)
    return pym.If(condition, then, else_)


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
