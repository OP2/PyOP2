"""Attempt 2."""


class Set:
    """E.g. an iteration domain defined as an inequality."""


class Relation:
    """A relation between sets (e.g. closure).

    Note: A relation may be 'mixed' where it consists of sub-relations.
    """


def generate_code():
    """Return a loopy kernel given these inputs.

    Required  inputs:
    - local kernel (as loopy)
    - mesh arity information (optional)
    - iteration set (this is not needed since we have nested domains from below)
    - relation between data structure set and the iteration set (i.e. nested domains and functions) (these really belong to the insns not data)
    - data structure section shapes
    """


def precompute_maps(kernel):
    """Precompute any maps so they can be passed in at runtime."""
    maps = set()

    for domain in kernel.domains:
        # Skip constant loop bounds
        if not isinstance(domain, Relation):
            continue

        # This is only valid if we have constant loop bounds (inspect mesh to get arities)
        if (domain.intype, domain.outtype) not in kernel.mesh.arities:
            continue

        # Step 1: Replace the relation with a assignment and modified loop bound
        # I.e. go from:
        #   for j in closure(i):
        #       dat[j]
        # to:
        #   for n in range(N):
        #       j = map[n]
        #       dat[j]
        new_iname = next(iname)
        new_domain = Set(kernel.mesh.arities[domain.intype, domain.outtype])
        pack_insn = Assignment(within_inames=inames+new_iname)
        new_domains.replace(domain, new_domain)

        # Step 2: Add an extra map argument to the kernel
        maps.add(domain)
    new_insns.replace(insn, insn.copy(domains=new_domains))

    return kernel.copy(insns=new_insns, maps=maps)
