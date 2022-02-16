class Mesh:
    ...


class ExtrudedMesh(Mesh):

    @property
    def cells(self):
        """Return an ISL domain."""
        # See codegen.old/rep2loopy.py
        # For extruded we should return two domains instead of one.
        return vars[name].ge_set(zero) & vars[name].lt_set(zero + expr.extent)
