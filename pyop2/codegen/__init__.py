from functools import singledispatch

import loopy as lp

from pyop2.codegen.ir import (Expr, Loop, Terminal)


def make_kernel(expr, *args, **kwargs):
    """Return a loopy kernel corresponding to ``expr``."""
    domains = ...
    insns = ...
    kernel_data = ...

    return lp.make_kernel(domains, insns, kernel_data)


@singledispatch
def parse_domains(expr: Expr):
    raise NotImplementedError


@parse_domains.register
def parse_domains_loop(expr: Loop):
    subdomains = frozenset(filter(None, chain(map(parse_domains, expr.children))))
    return frozenset(expr.domain) | subdomains


@parse_domains.register
def parse_domains_terminal(expr: Terminal):
    pass


if __name__ == "__main__":
    # This is an example of the sort of codegen AST we want
    from pyop2.codegen.ir import (DirectLoop, IndirectLoop, Assignment, FunctionCall)
    from pyop2.codegen.relation import Closure

    expr = DirectLoop(mesh.cells, [
        IndirectLoop(Closure, [Assignment()]),
        FunctionCall(kernel),
        IndirectLoop(Closure, [Assignment()])
    ])
