from functools import singledispatch
from itertools import chain

import loopy as lp

from pyop2.codegen.ir import (Expr, Loop, Terminal)


def make_kernel(expr, *args, **kwargs):
    """Return a loopy kernel corresponding to ``expr``."""
    domains = parse_domains(expr)
    insns = parse_insns(expr)
    kernel_data = parse_kernel_data(expr)

    return lp.make_kernel(domains, insns, kernel_data)


def parse_domains(expr: Expr):
    return frozenset(expr.domain) | chain(map(parse_domains, expr.children))


def parse_insns(expr: Expr):
    return frozenset(expr.insns) | chain(map(parse_insns, expr.children))

def parse_insns_pack(expr: PackInstruction, domains: List[str]=None):
    return lp.Assignment("tmp = dat[i]")

@parse_insns.register
def parse_insns_assignment(expr: UnpackInstruction, domains: List[str]=None):
    return lp.Assignment("dat[j] = tmp[i]")

def parse_kernel_data(expr: Expr, domains=None):
    ...


@parse_kernel_data.register(PackInstruction)
@parse_kernel_data.register(UnpackInstruction)
def parse_kernel_data_assignment(expr, domains):
    return lp.ArrayArg(...)


if __name__ == "__main__":
    # This is an example of the sort of codegen AST we want
    from pyop2.codegen.ir import (DirectLoop, IndirectLoop, Assignment, FunctionCall)
    from pyop2.codegen.relation import Closure

    expr = DirectLoop(mesh.cells, [
        IndirectLoop(Closure, [Assignment()]),
        FunctionCall(kernel),
        IndirectLoop(Closure, [Assignment()])
    ])
