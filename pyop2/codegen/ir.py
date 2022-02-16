from abc import ABC

import loopy as lp


class Expr(ABC):
    """Base class for nodes in the AST."""


class Loop(Expr, ABC):
    
    @property
    @abstractmethod
    def domain(self):
        ...


class DirectLoop(Loop):

    def __init__(self, domain):
        self.domain = domain

    @property
    def insns(self):
        return frozenset()


class IndirectLoop(Loop):
    """Indirect loop class.

    Equivalent to:

    (for cell in cells)
        map, len = get_indices(cell)  # temp variable in kernel
        for i in range(len):  # domain
            j = map[i]  # temp variable

    An indirect loops consists of:
    - insns
    - a loop

    Note:
        This can be lifted if we memoize the maps (done later on).
    """

    def __init__(self, relation):
        self.relation = relation
        self.name = something_unique

        self.map = lp.TemporaryVariable(...)
        self.len = lp.TemporaryVariable()
        self.j = lp.TemporaryVariable(...)

    @property
    def domain(self):
        return "0 <= j <= len"

    @property
    def insns(self):
        yield lp.FunctionCall(...)
        yield lp.Assignment("j = map[i]")

    @property
    def temporary_variables(self):
        return self.map, self.len, self.j


class Terminal(Expr):

    children = frozenset()


class Instruction(Terminal):
    ...


class GetInstruction(Instruction):
    def __init__(self):
        self.temp_var = lp.TemporaryVariable(...)
        self.array_var = lp.ArrayArg(...)


class FunctionCall(Instruction):
    ...
