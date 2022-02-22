import abc


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
