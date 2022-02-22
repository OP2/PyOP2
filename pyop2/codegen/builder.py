from functools import singledispatchmethod

import loopy as lp
from pytools import UniqueNameGenerator


def _is_read(access):
    return access in {READ, RW}


def _is_write(access):
    return access in {RW, WRITE}


class WrapperKernelBuilder:

    def __init__(self, kernel, domains, kernel_data, *, mesh=None):
        if len(kernel.arguments) != len(kernel_data):
            raise ValueError

        if not mesh and any(d.relation is not IDENTITY for d in kernel_data):
            raise ValueError

        self._kernel = kernel
        self._domains = domains
        self._kernel_data = kernel_data

        self._unique_indices = itertools.Count()

    def build(self):
        
        packing_knls = tuple(_packing_kernel(d)
                            for d, access in zip(self._kernel_data, self._kernel.accesses)
                            if _is_read(access))
        eval_knl = ...
        unpacking_knls = ...

        return lp.fuse_kernels(...)

    @singledispatchmethod
    def _gather_kernel(self, arg, access):
        raise NotImplementedError

    @_gather_kernel.register(DatArg)
    def add_argument_dat(self, arg, access):
        insn_id = next(self.insn_counter)
        packed_data = f"p{insn_id}"
        sparse_data = f"dat{insn_id}"

        lp.ArrayArg(sparse_data)
        lp.TemporaryArg(packed_data, within_inames=self._domain_inames)

        # I should recursively add relations here since that is easier than parsing tags
        domains = []
        indirections = []
        inames = []
        kernel_data = []
        tags = {}
        pt = self._domain_inames
        for rel in arg.relations:  # this starts with the 'outer' one (e.g. closure before dofs)
            # Step 1: Define my variables
            idx = next(self.unique_indices)
            iname = f"i{idx}"
            map = f"map{idx}"
            new_pt = ...

            # Step 2: Get the map and its size
            lp.TemporaryVariable(map)
            lp.TemporaryVariable(extent)
            lp.CInstruction(..., assignees={map, extent}, tags={rel.name})
            insns.append(f"ierr = PetscSectionGetClosureIndices(dm,section,&closureSection,&closureIs);CHKERRQ(ierr);")

            # Step 3: This gives us a new domain
            domains.append(f"[{iname}]: 0<={iname}<{extent}")

            # Step 4: Now determine the new placeholder variable
            insns.append((f"{new_pt} = {map}[{pt},{iname}]"))
            pt = new_pt

            tags += {iname: rel.name}

        # Lastly emit the actual pack instruction
        pack_insn = f"p[{inames}] = dat[{tmp_var}]"

        knl = lp.make_kernel([self._domains, f"[{iname}]: 0<={iname}<{extent}"],
                              (f"{temp_var} = map[{iname}]"
                               f"p{all_subinames} = dat[{temp_var}]"))
        knl = lp.tag_inames(knl, tags)

    def _set_rw_deps(self):
        """Set read/write instruction dependencies."""
        # These will not work as written
        self._local_knl_insn.depends_on += self._read_insns
        self._write_insns.depends_on += self._local_knl_insn
