import loopy as lp
from pytools import UniqueNameGenerator


def isread(access):
    return access in {READ, RW}

def iswritten(access):
    return access in {RW, WRITE}


class GlobalKernelBuilder:

    def __init__(self, local_kernel, iterset):
        self._local_kernel = local_kernel
        self._iterset = iterset

        self._generate_domain_param = UniqueNameGenerator(...)

        self._domains = set()
        self._read_insns = set()
        self._local_knl_insn = ...
        self._write_insns = set()
        self._temp_vars = set()

    def build():
        return lp.make_kernel(self._domains, self._insns, ...)

    @singledispatchmethod
    def add_argument(self, arg):
        raise NotImplementedError

    @add_argument.register(DatArg)
    def add_argument_dat(self, arg):
        if not arg.relation:
            # no map required
            raise NotImplementedError

        # Each relation corresponds to a new domain (w. param), indirection, temp vars...
        # TODO How do I ensure uniqueness? Can loopy do that? Probably. Something to do with
        # fusing unused inames.
        within_inames, idx = self._add_relation(arg.relation, self._iterset_iname)

        # TODO arg.access not reasonable as belongs to local kernel
        min_iname = within_inames[-1]
        dat_var = ...
        temp_var = ...
        if isread(arg.access):
            self._read_insns.add(f"{temp_var}[{min_iname}] = {dat_var}[{idx}]")
        if iswritten(arg.access):
            self._write_insns.add(f"{dat_var}[{idx}] = {temp_var}[{min_iname}]")

    def _add_relation(self, relation, within_inames):
        # start search from the bottom (nearest the iterset)
        if relation.subrelation:
            within_inames = self._add_relation(relation.subrelation, within_inames)

        new_iname = ...
        new_domain_param = self._generate_domain_param()
        new_fn_call = ...
        new_idx = ...  # i.e. j = map[i]
        new_map = ...

        self._domains.add(f"[{new_iname}]: 0<={new_iname}<{new_domain_param}")
        self._insns.add(
            lp.FunctionCall(..., within_inames=within_inames),
            f"{new_idx} = {new_map}[{new_iname}]"
        )

        self._temp_vars.add(new_idx, new_map)
        return within_inames

    def _set_rw_deps(self):
        """Set read/write instruction dependencies."""
        # These will not work as written
        self._local_knl_insn.depends_on += self._read_insns
        self._write_insns.depends_on += self._local_knl_insn
