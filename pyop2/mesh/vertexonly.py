from collections import defaultdict

import numpy as np
from petsc4py import PETSc
import ufl  # FIXME This should not be a dependency

from pyop2.mesh import dmutils as dmcommon
from pyop2.mesh.base import Mesh
from pyop2.mpi import dup_comm
from pyop2.utils import cached_property


# TODO: Could this be merged with MeshTopology given that dmcommon.pyx
# now covers DMSwarms and DMPlexes?
class VertexOnlyMesh(Mesh):
    """
    Representation of a vertex-only mesh topology immersed within
    another mesh.
    """

    @PETSc.Log.EventDecorator()
    def __init__(self, swarm, parentmesh, name, reorder):
        """
        Half-initialise a mesh topology.

        :arg swarm: Particle In Cell (PIC) :class:`DMSwarm` representing
            vertices immersed within a :class:`DMPlex` stored in the
            `parentmesh`
        :arg parentmesh: the mesh within which the vertex-only mesh
            topology is immersed.
        :arg name: name of the mesh
        :arg reorder: whether to reorder the mesh (bool)
        """

        super().__init__(name)

        # TODO: As a performance optimisation, we should renumber the
        # swarm to in parent-cell order so that we traverse efficiently.
        if reorder:
            raise NotImplementedError("Mesh reordering not implemented for vertex only meshes yet.")

        dmcommon.validate_mesh(swarm)
        swarm.setFromOptions()

        self._parent_mesh = parentmesh
        self.topology_dm = swarm
        r"The PETSc DM representation of the mesh topology."
        self._comm = dup_comm(swarm.comm.tompi4py())

        # A cache of shared function space data on this mesh
        self._shared_data_cache = defaultdict(dict)

        # Cell subsets for integration over subregions
        self._subsets = {}

        tdim = 0

        cell = ufl.Cell("vertex")
        self._ufl_mesh = ufl.Mesh(ufl.VectorElement("DG", cell, 0, dim=cell.topological_dimension()))

        # Mark OP2 entities and derive the resulting Swarm numbering
        with PETSc.Log.Event("Mesh: numbering"):
            dmcommon.mark_entity_classes(self.topology_dm)
            self._entity_classes = dmcommon.get_entity_classes(self.topology_dm).astype(int)

            # Derive a cell numbering from the Swarm numbering
            entity_dofs = np.zeros(tdim+1, dtype=IntType)
            entity_dofs[-1] = 1

            self._cell_numbering = self.create_section(entity_dofs)
            entity_dofs[:] = 0
            entity_dofs[0] = 1
            self._vertex_numbering = self.create_section(entity_dofs)

    @property
    def comm(self):
        return self._comm

    @cached_property
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        swarm = self.topology_dm
        tdim = 0

        # Cell numbering and global vertex numbering
        cell_numbering = self._cell_numbering
        vertex_numbering = self._vertex_numbering.createGlobalSection(swarm.getPointSF())

        cell = self.ufl_cell()
        assert tdim == cell.topological_dimension()
        assert cell.is_simplex()

        import FIAT
        topology = FIAT.ufc_cell(cell).get_topology()
        entity_per_cell = np.zeros(len(topology), dtype=IntType)
        for d, ents in topology.items():
            entity_per_cell[d] = len(ents)

        return dmcommon.closure_ordering(swarm, vertex_numbering,
                                         cell_numbering, entity_per_cell)

    entity_orientations = None

    def _facets(self, kind):
        """Raises an AttributeError since cells in a
        `VertexOnlyMeshTopology` have no facets.
        """
        if kind not in ["interior", "exterior"]:
            raise ValueError("Unknown facet type '%s'" % kind)
        raise AttributeError("Cells in a VertexOnlyMeshTopology have no facets.")

    @cached_property
    def exterior_facets(self):
        return self._facets("exterior")

    @cached_property
    def interior_facets(self):
        return self._facets("interior")

    @cached_property
    def cell_to_facets(self):
        """Raises an AttributeError since cells in a
        `VertexOnlyMeshTopology` have no facets.
        """
        raise AttributeError("Cells in a VertexOnlyMeshTopology have no facets.")

    def num_cells(self):
        return self.num_vertices()

    def num_facets(self):
        return 0

    def num_faces(self):
        return 0

    def num_edges(self):
        return 0

    def num_vertices(self):
        return self.topology_dm.getLocalSize()

    def num_entities(self, d):
        if d > 0:
            return 0
        else:
            return self.num_vertices()

    @cached_property
    def cell_set(self):
        size = list(self._entity_classes[self.cell_dimension(), :])
        return op2.Set(size, "Cells", comm=self.comm)

    @property
    def cell_parent_cell_list(self):
        """Return a list of parent mesh cells numbers in vertex only
        mesh cell order.
        """
        cell_parent_cell_list = np.copy(self.topology_dm.getField("parentcellnum"))
        self.topology_dm.restoreField("parentcellnum")
        return cell_parent_cell_list

    @property
    def cell_parent_cell_map(self):
        """Return the :class:`pyop2.Map` from vertex only mesh cells to
        parent mesh cells.
        """
        return op2.Map(self.cell_set, self._parent_mesh.cell_set, 1,
                       self.cell_parent_cell_list, "cell_parent_cell")
