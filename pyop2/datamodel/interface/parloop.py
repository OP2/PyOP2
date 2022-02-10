import operator


# TODO Need to include the notion that maps map between sets (potentially multiple)


# DSL should look something like:
# mapexpr = section(closure(star(iterset)))


# I can refer to entities using their number.
# E.g. the closure of a point is all points greater than or equal to the current one
# E.g. the star is one less
ENTITIES_2D = CELL, EDGE, VERTEX
ENTITIES_3D = CELLS, FACET, EDGE, VERTEX


def parloop(domains, kernels, args):
    if isinstance(kernel, LocalKernel):
        kernel = compile_kernel(domains, kernel, args)

    kernel(domains, args)


def compile_kernel(domains, kernels, args):
    if len(domains) != 1 or len(kernels) != 1:
        raise NotImplementedError

    builder.generate(...)


# map routines

class Expr:

    @property
    @abstractmethod
    def mesh(self):
        ...

class Map(Expr):
    """leaf node will always be a map"""

    @property
    @abstractmethod
    def operand(self):
        ...

    @property
    def mesh(self):
        return self.operand.mesh

    @property
    def section(self):
        ...
    
    @property
    def indices(self):
        """From a PETSc IS."""

    @property
    def pack_insns(self):
        # This does not require PETSc section lookup (fast)
        ...

    @property
    def unpack_insns(self):
        ...

class Closure(Map):

    def __init__(self, o):
        from_set = CELL
        to_sets = ENTITIES_2D[CELL.stratum:]


class Star(PlexOp):

    def __init__(self, operand):
        self.operand = operand

        from_set = operand.set
        to_sets = ENTITIES_2D[operand.set.stratum-1]


class Parent(PlexOp):
    """Map connecting a child mesh to its parent."""
    def __init__(self, operand):
        if not operand.mesh.parent:
            raise ValueError

        self.operand = operand


class Child(PlexOp):
    """Map connecting a parent mesh to its child.

    It maps from an entity in the parent mesh to all of the corresponding entities
    in the child mesh. E.g. cell in coarse mesh to cells in fine mesh.
    """

    def __init__(self, operand):
        if not operand.mesh.child:
            raise ValueError

        self.operand = operand


class IdentityMap(PlexOp):
    """Map that does nothing."""

    def __mul__(self, other):
        return other


class Terminal(Expr):
    """Set-type thing."""


# data objects


class Set:

    def __init__(self):
        self.mesh = mesh


class Section(abc.ABC):
    """Data layout thing."""


class ConstSection(Section):

    def __init__(self, dof):
        self._dof = dof

    def get_dof(self, pt):
        return self._dof

    def get_offset(self, pt):
        return pt * self._dof

class VariableSection(Section):
    """This stores a PETSc Section."""

    def get_dof(self, pt):
        return self.petsc_section.getDof(pt)

    def get_offset(self, pt):
        return self.petsc_section.getOffset(pt)
        

def compose_maps(a2b, b2c):
    """
    Example:

        {1: (3, 4), 2: (4, 5)} * {3: (6, 7), 4: (6, 8), 5: (7, 8)}

        = {1: ((6, 7), (6, 8)), 2: ((6, 8), (7, 8))}
    """
    is_const = all(isinstance(s, ConstSection) for s in {a2b.section, b2c.section}):

    if is_const:
        a2c_section = ConstSection(a2b.section._dof*a2c.section._dof)
    else:
        a2c_section = VariableSection()

    a2c = Map(a2c_section)
    start, stop = a2b.section.getChart():

    if not is_const: 
        for pt in range(start, stop):
            ndofs = 0
            a2b_dof = a2b.section.getDof(pt)
            a2b_offset = a2b.section.getOffset(pt)
            for i in range(a2b_dof):
                pt2 = a2b.indices[a2b_offset+i]
                ndofs += b2c.section.getDof(pt2)
            a2c.section.setDof(pt, ndofs)
        PetscSectionSetUp(a2c_section)

    a2c_indices = ...
    for pt in range(start, stop):
        a2b_dof = a2b.section.getDof(pt)
        a2b_offset = a2b.section.getOffset(pt)
        a2c_offset = a2c_section.getOffset(pt)
        for i in range(a2b_dof):
            pt2 = a2b.indices[a2b_offset+i]
            b2c_offset = b2c.section.getOffset(pt2)
            for j in range(b2c.section.getDof(pt)):
                a2c_indices[a2c_offset+i*a2b_dof+j] = b2c.indices[b2c_offset+j])

    return a2c
def generate_pack_insns(expr):
    ...

@generate_pack_insns.register
def generate_pack_insns_parent(expr: Parent):
    # emit an additional for loop


@singledispatch
def compute_arity(expr):
    ...

@compute_arity.register
def compute_arity_closure(expr: Closure):
    ...


@compute_arity.register
def compute_arity_terminal(expr: Terminal):
    # need to compose map with DoF/entity


def compress(expr):
    """Compress a map expression into a single map."""


@compress.register
def compress(expr: Closure):
    if isinstance(expr.operand, Terminal):
        m = IdentityMap(expr.operand)
    else:
        m = compress(expr.operand)
    return compose_map(m, m.mesh.to_closure)


# Do not do any compression for these mapping operations as they correspond to additional for loops
@compress.register
def compress(expr: Parent|Child):
    return type(expr)(compress(expr.operand))


def closure(expr):
    return Closure(expr, section, index_set)


def star(arg):
    return Star(...)


def child(expr):
    return Child(expr)


def mywrapper(func):
    def wrapper(arg):
        if isinstance(arg, Dat):
            arg = Arg(arg)
        return func(arg)
    return wrapper

# ---


class Mesh:
    
    def __init__(self, dm):
        self.dm = dm

    @lru_cache
    def get_closure(self):
        return Map(self.dm.getClosureIndices())

    @lru_cache
    def get_support(self, stratum):
        ...

    @lru_cache
    def get_cone(self, stratum):
        ...


class SimplexMesh2D:

    # there are 3 edges per cell, 3 vertices per cell, and 2 vertices per edge
    arities = {CELL: {EDGE: 3, VERTEX: 3},
               EDGE: {VERTEX: 2}}

    section = ConstSection(arities)

    @lru_cache
    def get_closure(self):
        _, indices = self.dm.getClosureIndices()
        return Map(self.section, indices)
    

class GlobalKernel:

    def __call__(*args):
        data = [arg.data for arg in args]
        maps = set(compose_maps(arg) for arg in args)
        ...


if __name__ == "__main__":
    cells = Set(...)
    dat1 = Dat(...)

    domains = [cells]
    args = [closure(dat1)]
    parloop(domains, kernels, args)

    # for extruded could be:
    args = [child(closure(parent(dat1)))]
