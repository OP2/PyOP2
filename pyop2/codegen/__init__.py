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

    # NO. We do not need our own intermediate representation! Loopy plus
    # tags is completely adequate.
    # We should aim for something like:

    from pytools.tag import Tag
    class ClosureTag(Tag):
        def __str__(self):
            return "closure"


    knl = lp.make_kernel(
        [
            "{ [i]: 0 <=i<ni }",
            "{ [j]: 0<=j<nj }"
        ],
        """
        <>t0[j] = dat0[map0[j]] {id=stmt1}
        kernel(t0) {id=stmt2, dep=stmt1}
        dat0[map0[j]] = t0[j] {dep=stmt2}
        """
    )

    knl = lp.tag_inames(knl, ("j", ClosureTag))
    knl = lp.register_callable(knl, "kernel", ...)

    # need to then do a replacement with the closure tags
    # instruction replacement not currently supported by loopy (but easy to add)

    # Wait. This is not the right thing to be doing (quite). This is because
    # the loopy kernel should technically be complete upon instantiation and my
    # transformations modify that. I need to do the processing elsewhere.

    # PyOP2 does "execution of some function over an iteration set". With that design
    # it makes some sense to use our existing API (even though I don't like the DSL design).

    # Perhaps we want something like:
    # kernel_data = pyop2.DatArg(dat, closure*star)
    # or
    # arguments = dat*closure*star (== pyop2.DatArg) (N.B. This should be fine for extruded
    # since dat knows its set)
    # pyop2.parloop(local_kernel, iterset, arguments)

    # I should be able to use this information to create a GlobalKernelBuilder that
    # directly constructs loopy code.

    # def make_global_kernel(...):
    #     ...
