class Mesh:
    ...


class ExtrudedMesh(Mesh):

    @property
    def cells(self):
        """Return an ISL domain."""
        # See codegen.old/rep2loopy.py
        # For extruded we should return two domains instead of one.
        return vars[name].ge_set(zero) & vars[name].lt_set(zero + expr.extent)


# this belongs in dmutils.pyx
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
