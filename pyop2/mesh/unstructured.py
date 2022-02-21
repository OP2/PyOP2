import os

from pyop2.mesh import dmutils as dmcommon
from pyop2.mesh.impls.base import Mesh


class UnstructuredMesh(Mesh):
    """A representation of mesh topology implemented on a PETSc DMPlex."""

    @PETSc.Log.EventDecorator("CreateMesh")
    def __init__(self, plex, name, reorder, distribution_parameters):
        """Half-initialise a mesh topology.

        :arg plex: :class:`DMPlex` representing the mesh topology
        :arg name: name of the mesh
        :arg reorder: whether to reorder the mesh (bool)
        :arg distribution_parameters: options controlling mesh
            distribution, see :func:`Mesh` for details.
        """

        super().__init__(name)

        # Do some validation of the input mesh
        distribute = distribution_parameters.get("partition")
        self._distribution_parameters = distribution_parameters.copy()
        if distribute is None:
            distribute = True
        partitioner_type = distribution_parameters.get("partitioner_type")
        overlap_type, overlap = distribution_parameters.get("overlap_type",
                                                            (DistributedMeshOverlapType.FACET, 1))

        if overlap < 0:
            raise ValueError("Overlap depth must be >= 0")
        if overlap_type == DistributedMeshOverlapType.NONE:
            def add_overlap():
                pass
            if overlap > 0:
                raise ValueError("Can't have NONE overlap with overlap > 0")
        elif overlap_type == DistributedMeshOverlapType.FACET:
            def add_overlap():
                dmcommon.set_adjacency_callback(self.topology_dm)
                self.topology_dm.distributeOverlap(overlap)
                dmcommon.clear_adjacency_callback(self.topology_dm)
                self._grown_halos = True
        elif overlap_type == DistributedMeshOverlapType.VERTEX:
            def add_overlap():
                # Default is FEM (vertex star) adjacency.
                self.topology_dm.distributeOverlap(overlap)
                self._grown_halos = True
        else:
            raise ValueError("Unknown overlap type %r" % overlap_type)

        dmcommon.validate_mesh(plex)
        plex.setFromOptions()

        self.topology_dm = plex
        r"The PETSc DM representation of the mesh topology."
        self._comm = dup_comm(plex.comm.tompi4py())

        # Mark exterior and interior facets
        # Note.  This must come before distribution, because otherwise
        # DMPlex will consider facets on the domain boundary to be
        # exterior, which is wrong.
        label_boundary = (self.comm.size == 1) or distribute
        dmcommon.label_facets(plex, label_boundary=label_boundary)

        # Distribute/redistribute the dm to all ranks
        if self.comm.size > 1 and distribute:
            # We distribute with overlap zero, in case we're going to
            # refine this mesh in parallel.  Later, when we actually use
            # it, we grow the halo.
            self.set_partitioner(distribute, partitioner_type)
            plex.distribute(overlap=0)
            # plex carries a new dm after distribute, which
            # does not inherit partitioner from the old dm.
            # It probably makes sense as chaco does not work
            # once distributed.

        tdim = plex.getDimension()

        # Allow empty local meshes on a process
        cStart, cEnd = plex.getHeightStratum(0)  # cells
        if cStart == cEnd:
            nfacets = -1
        else:
            nfacets = plex.getConeSize(cStart)

        # TODO: this needs to be updated for mixed-cell meshes.
        nfacets = self.comm.allreduce(nfacets, op=MPI.MAX)

        # Note that the geometric dimension of the cell is not set here
        # despite it being a property of a UFL cell. It will default to
        # equal the topological dimension.
        # Firedrake mesh topologies, by convention, which specifically
        # represent a mesh topology (as here) have geometric dimension
        # equal their topological dimension. This is reflected in the
        # corresponding UFL mesh.
        cell = ufl.Cell(_cells[tdim][nfacets])
        self._ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1, dim=cell.topological_dimension()))

        def callback(self):
            """Finish initialisation."""
            del self._callback
            if self.comm.size > 1:
                add_overlap()
            dmcommon.complete_facet_labels(self.topology_dm)

            if reorder:
                with PETSc.Log.Event("Mesh: reorder"):
                    old_to_new = self.topology_dm.getOrdering(PETSc.Mat.OrderingType.RCM).indices
                    reordering = np.empty_like(old_to_new)
                    reordering[old_to_new] = np.arange(old_to_new.size, dtype=old_to_new.dtype)
            else:
                # No reordering
                reordering = None
            self._did_reordering = bool(reorder)

            # Mark OP2 entities and derive the resulting Plex renumbering
            with PETSc.Log.Event("Mesh: numbering"):
                dmcommon.mark_entity_classes(self.topology_dm)
                self._entity_classes = dmcommon.get_entity_classes(self.topology_dm).astype(int)
                self._plex_renumbering = dmcommon.plex_renumbering(self.topology_dm,
                                                                   self._entity_classes,
                                                                   reordering)

                # Derive a cell numbering from the Plex renumbering
                entity_dofs = np.zeros(tdim+1, dtype=IntType)
                entity_dofs[-1] = 1

                self._cell_numbering = self.create_section(entity_dofs)
                entity_dofs[:] = 0
                entity_dofs[0] = 1
                self._vertex_numbering = self.create_section(entity_dofs)

                entity_dofs[:] = 0
                entity_dofs[-2] = 1
                facet_numbering = self.create_section(entity_dofs)
                self._facet_ordering = dmcommon.get_facet_ordering(self.topology_dm, facet_numbering)
        self._callback = callback

    @classmethod
    def from_file(cls, meshfile):
        comm = kwargs.get("comm", COMM_WORLD)
        name = meshfile
        basename, ext = os.path.splitext(meshfile)

        if ext.lower() in ['.e', '.exo']:
            plex = dmutils._from_exodus(meshfile, comm)
        elif ext.lower() == '.cgns':
            plex = dmutils._from_cgns(meshfile, comm)
        elif ext.lower() == '.msh':
            if geometric_dim is not None:
                opts = {"dm_plex_gmsh_spacedim": geometric_dim}
            else:
                opts = {}
            opts = OptionsManager(opts, "")
            with opts.inserted_options():
                plex = dmutils._from_gmsh(meshfile, comm)
        elif ext.lower() == '.node':
            plex = dmutils._from_triangle(meshfile, geometric_dim, comm)
        else:
            raise RuntimeError("Mesh file %s has unknown format '%s'."
                               % (meshfile, ext[1:]))

        return cls(plex)


    @property
    def comm(self):
        return self._comm

    @utils.cached_property
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        plex = self.topology_dm
        tdim = plex.getDimension()

        # Cell numbering and global vertex numbering
        cell_numbering = self._cell_numbering
        vertex_numbering = self._vertex_numbering.createGlobalSection(plex.getPointSF())

        cell = self.ufl_cell()
        assert tdim == cell.topological_dimension()
        if cell.is_simplex():
            import FIAT
            topology = FIAT.ufc_cell(cell).get_topology()
            entity_per_cell = np.zeros(len(topology), dtype=IntType)
            for d, ents in topology.items():
                entity_per_cell[d] = len(ents)

            return dmcommon.closure_ordering(plex, vertex_numbering,
                                             cell_numbering, entity_per_cell)

        elif cell.cellname() == "quadrilateral":
            from firedrake_citations import Citations
            Citations().register("Homolya2016")
            Citations().register("McRae2016")
            # Quadrilateral mesh
            cell_ranks = dmcommon.get_cell_remote_ranks(plex)

            facet_orientations = dmcommon.quadrilateral_facet_orientations(
                plex, vertex_numbering, cell_ranks)

            cell_orientations = dmcommon.orientations_facet2cell(
                plex, vertex_numbering, cell_ranks,
                facet_orientations, cell_numbering)

            dmcommon.exchange_cell_orientations(plex,
                                                cell_numbering,
                                                cell_orientations)

            return dmcommon.quadrilateral_closure_ordering(
                plex, vertex_numbering, cell_numbering, cell_orientations)

        else:
            raise NotImplementedError("Cell type '%s' not supported." % cell)

    @utils.cached_property
    def entity_orientations(self):
        return dmcommon.entity_orientations(self, self.cell_closure)

    @PETSc.Log.EventDecorator()
    def _facets(self, kind):
        if kind not in ["interior", "exterior"]:
            raise ValueError("Unknown facet type '%s'" % kind)

        dm = self.topology_dm
        facets, classes = dmcommon.get_facets_by_class(dm, (kind + "_facets"),
                                                       self._facet_ordering)
        label = dmcommon.FACE_SETS_LABEL
        if dm.hasLabel(label):
            from mpi4py import MPI
            markers = dmcommon.get_facet_markers(dm, facets)
            local_markers = set(dm.getLabelIdIS(label).indices)

            def merge_ids(x, y, datatype):
                return x.union(y)

            op = MPI.Op.Create(merge_ids, commute=True)

            unique_markers = np.asarray(sorted(self.comm.allreduce(local_markers, op=op)),
                                        dtype=IntType)
            op.Free()
        else:
            markers = None
            unique_markers = None

        local_facet_number, facet_cell = \
            dmcommon.facet_numbering(dm, kind, facets,
                                     self._cell_numbering,
                                     self.cell_closure)

        point2facetnumber = np.full(facets.max(initial=0)+1, -1, dtype=IntType)
        point2facetnumber[facets] = np.arange(len(facets), dtype=IntType)
        obj = _Facets(self, classes, kind,
                      facet_cell, local_facet_number,
                      markers, unique_markers=unique_markers)
        obj.point2facetnumber = point2facetnumber
        return obj

    @utils.cached_property
    def exterior_facets(self):
        return self._facets("exterior")

    @utils.cached_property
    def interior_facets(self):
        return self._facets("interior")

    @utils.cached_property
    def cell_to_facets(self):
        """Returns a :class:`op2.Dat` that maps from a cell index to the local
        facet types on each cell, including the relevant subdomain markers.

        The `i`-th local facet on a cell with index `c` has data
        `cell_facet[c][i]`. The local facet is exterior if
        `cell_facet[c][i][0] == 0`, and interior if the value is `1`.
        The value `cell_facet[c][i][1]` returns the subdomain marker of the
        facet.
        """
        cell_facets = dmcommon.cell_facet_labeling(self.topology_dm,
                                                   self._cell_numbering,
                                                   self.cell_closure)
        if isinstance(self.cell_set, op2.ExtrudedSet):
            dataset = op2.DataSet(self.cell_set.parent, dim=cell_facets.shape[1:])
        else:
            dataset = op2.DataSet(self.cell_set, dim=cell_facets.shape[1:])
        return op2.Dat(dataset, cell_facets, dtype=cell_facets.dtype,
                       name="cell-to-local-facet-dat")

    def num_cells(self):
        cStart, cEnd = self.topology_dm.getHeightStratum(0)
        return cEnd - cStart

    def num_facets(self):
        fStart, fEnd = self.topology_dm.getHeightStratum(1)
        return fEnd - fStart

    def num_faces(self):
        fStart, fEnd = self.topology_dm.getDepthStratum(2)
        return fEnd - fStart

    def num_edges(self):
        eStart, eEnd = self.topology_dm.getDepthStratum(1)
        return eEnd - eStart

    def num_vertices(self):
        vStart, vEnd = self.topology_dm.getDepthStratum(0)
        return vEnd - vStart

    def num_entities(self, d):
        eStart, eEnd = self.topology_dm.getDepthStratum(d)
        return eEnd - eStart

    @utils.cached_property
    def cell_set(self):
        size = list(self._entity_classes[self.cell_dimension(), :])
        return op2.Set(size, "Cells", comm=self.comm)

    @PETSc.Log.EventDecorator()
    def set_partitioner(self, distribute, partitioner_type=None):
        """Set partitioner for (re)distributing underlying plex over comm.

        :arg distribute: Boolean or (sizes, points)-tuple.  If (sizes, point)-
            tuple is given, it is used to set shell partition. If Boolean, no-op.
        :kwarg partitioner_type: Partitioner to be used: "chaco", "ptscotch", "parmetis",
            "shell", or `None` (unspecified). Ignored if the distribute parameter
            specifies the distribution.
        """
        from firedrake_configuration import get_config
        plex = self.topology_dm
        partitioner = plex.getPartitioner()
        if type(distribute) is bool:
            if partitioner_type:
                if partitioner_type not in ["chaco", "ptscotch", "parmetis"]:
                    raise ValueError("Unexpected partitioner_type %s" % partitioner_type)
                if partitioner_type == "chaco":
                    if IntType.itemsize == 8:
                        raise ValueError("Unable to use 'chaco': 'chaco' is 32 bit only, "
                                         "but your Integer is %d bit." % IntType.itemsize * 8)
                if partitioner_type == "parmetis":
                    if not get_config().get("options", {}).get("with_parmetis", False):
                        raise ValueError("Unable to use 'parmetis': Firedrake is not "
                                         "installed with 'parmetis'.")
            else:
                if IntType.itemsize == 8:
                    # Default to PTSCOTCH on 64bit ints (Chaco is 32 bit int only).
                    # Chaco does not work on distributed meshes.
                    if get_config().get("options", {}).get("with_parmetis", False):
                        partitioner_type = "parmetis"
                    else:
                        partitioner_type = "ptscotch"
                else:
                    partitioner_type = "chaco"
            partitioner.setType({"chaco": partitioner.Type.CHACO,
                                 "ptscotch": partitioner.Type.PTSCOTCH,
                                 "parmetis": partitioner.Type.PARMETIS}[partitioner_type])
        else:
            sizes, points = distribute
            partitioner.setType(partitioner.Type.SHELL)
            partitioner.setShellPartition(self.comm.size, sizes, points)
        # Command line option `-petscpartitioner_type <type>` overrides.
        partitioner.setFromOptions()


