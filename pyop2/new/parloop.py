import operator


def parloop(domains, kernels, args):
    if isinstance(kernel, LocalKernel):
        kernel = compile_kernel(domains, kernel, args)

    kernel(domains, args)


def compose_maps(arg):
    return reduce(operator.mul, arg.maps) or IdentityMap()


def compile_kernel(domains, kernels, args):
    if len(domains) != 1 or len(kernels) != 1:
        raise NotImplementedError

    builder.generate(...)


# map routines

class PlexRelation(IntEnum):

    CLOSURE = auto()
    STAR = auto()


@mywrapper
def closure(arg):
    return arg.maps + (CLOSURE,)


@mywrapper
def star(arg):
    return arg.maps + (STAR,)


def mywrapper(func):
    def wrapper(arg):
        if isinstance(arg, Dat):
            arg = Arg(arg)
        return func(arg)
    return wrapper

# ---


class DatArg:

    @property
    @abstractmethod
    def dim(self):
        ...

    @property
    @abstractmethod
    def map(self):
        ...

    @property
    def arity(self):
        return self.map.arity

    @property
    def is_direct(self):
        return isinstance(self.map, IdentityMap)



class PetscDatArg:

    def __init__(self, data, maps=None):
        self.data = data
        self.maps = maps or IdentityMap()



class Mesh:
    
    def __init__(self, dm):
        self.dm = dm

    @lru_cache
    def get_closure(self, stratum):
        # return Map(arity=somethingsensible) if no data involved.
        self.dm.getClosureIndices()

    @lru_cache
    def get_support(self, stratum):
        ...

    @lru_cache
    def get_cone(self, stratum):
        ...
    

class GlobalKernel:

    def __call__(*args):
        data = [arg.data for arg in args]
        maps = set(compose_maps(arg) for arg in args)
        ...


class Map:

    def __init__(self, arity, data=None):
        self.arity = arity

    def __mul__(self, other):
        sec1 = self.section
        is1 = self.index_set

        sec2 = other.section
        is2 = other.index_set

        # now the fun algorithm I worked out before
        ...


class IdentityMap:
    """Map that does nothing."""

    def __mul__(self, other):
        return other


class DirectMap:
    """Map that is used when the data may be directly addressed using an offset.

    This is useful for extruded meshes.
    """


if __name__ == "__main__":
    cells = Set(...)
    dat1 = Dat(...)

    domains = [cells]
    args = [closure(dat1)]
    parloop(domains, kernels, args)
