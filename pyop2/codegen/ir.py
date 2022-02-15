from abc import ABC


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


class IndirectLoop(Loop):

    def __init__(self, relation):
        self.relation = relation


class Terminal(Expr):
    ...


class Instruction(Terminal):
    ...


class Assignment(Instruction):
    ...


class FunctionCall(Instruction):
    ...
