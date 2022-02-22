import collections.abc
from functools import singledispatch
from itertools import chain

import loopy as lp


class DatArg:

    def __init__(self, data, relations):
        if not isinstance(relations, collections.abc.Iterable):
            relations = relations,

        self.data = data  # TODO Maybe this should just be a section or something?
        self.relations = relations

    def __mul__(self, other):
        if isinstance(other, Relation):
            return type(self)(self.data, (*self.relations, other))
        else:
            return NotImplemented


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
