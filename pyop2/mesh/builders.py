import numpy as np

import pyop2.mesh.dmutils as dmcommon
from pyop2.mesh.unstructured import UnstructuredMesh
from pyop2.mpi import COMM_WORLD


def make_box_mesh(nx, ny, nz, Lx, Ly, Lz, reorder=None, distribution_parameters=None,
                  diagonal="default", comm=COMM_WORLD):
    """Generate a mesh of a 3D box.

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :arg Lz: The extent in the z direction
    :arg diagonal: Two ways of cutting hexadra, should be cut into 6
        tetrahedra (``"default"``), or 5 tetrahedra thus less biased
        (``"crossed"``)
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).

    The boundary surfaces are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == Lx
    * 3: plane y == 0
    * 4: plane y == Ly
    * 5: plane z == 0
    * 6: plane z == Lz
    """
    for n in (nx, ny, nz):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a postive integer")

    xcoords = np.linspace(0, Lx, nx + 1, dtype=np.double)
    ycoords = np.linspace(0, Ly, ny + 1, dtype=np.double)
    zcoords = np.linspace(0, Lz, nz + 1, dtype=np.double)
    # X moves fastest, then Y, then Z
    coords = np.asarray(np.meshgrid(xcoords, ycoords, zcoords)).swapaxes(0, 3).reshape(-1, 3)
    i, j, k = np.meshgrid(np.arange(nx, dtype=np.int32),
                          np.arange(ny, dtype=np.int32),
                          np.arange(nz, dtype=np.int32))
    if diagonal == "default":
        v0 = k*(nx + 1)*(ny + 1) + j*(nx + 1) + i
        v1 = v0 + 1
        v2 = v0 + (nx + 1)
        v3 = v1 + (nx + 1)
        v4 = v0 + (nx + 1)*(ny + 1)
        v5 = v1 + (nx + 1)*(ny + 1)
        v6 = v2 + (nx + 1)*(ny + 1)
        v7 = v3 + (nx + 1)*(ny + 1)

        cells = [v0, v1, v3, v7,
                 v0, v1, v7, v5,
                 v0, v5, v7, v4,
                 v0, v3, v2, v7,
                 v0, v6, v4, v7,
                 v0, v2, v6, v7]
        cells = np.asarray(cells).swapaxes(0, 3).reshape(-1, 4)
    elif diagonal == "crossed":
        v0 = k*(nx + 1)*(ny + 1) + j*(nx + 1) + i
        v1 = v0 + 1
        v2 = v0 + (nx + 1)
        v3 = v1 + (nx + 1)
        v4 = v0 + (nx + 1)*(ny + 1)
        v5 = v1 + (nx + 1)*(ny + 1)
        v6 = v2 + (nx + 1)*(ny + 1)
        v7 = v3 + (nx + 1)*(ny + 1)

        # There are only five tetrahedra in this cutting of hexahedra
        cells = [v0, v1, v2, v4,
                 v1, v7, v5, v4,
                 v1, v2, v3, v7,
                 v2, v4, v6, v7,
                 v1, v2, v7, v4]
        cells = np.asarray(cells).swapaxes(0, 3).reshape(-1, 4)
        raise NotImplementedError("The crossed cutting of hexahedra has a broken connectivity issue for Pk (k>1) elements")
    else:
        raise ValueError("Unrecognised value for diagonal '%r'", diagonal)

    plex = dmcommon.from_cell_list(3, cells, coords, comm)

    # Apply boundary IDs
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        xtol = Lx/(2*nx)
        ytol = Ly/(2*ny)
        ztol = Lz/(2*nz)
        for face in boundary_faces:
            face_coords = plex.vecGetClosure(coord_sec, coords, face)
            if abs(face_coords[0]) < xtol and abs(face_coords[3]) < xtol and abs(face_coords[6]) < xtol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
            if abs(face_coords[0] - Lx) < xtol and abs(face_coords[3] - Lx) < xtol and abs(face_coords[6] - Lx) < xtol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
            if abs(face_coords[1]) < ytol and abs(face_coords[4]) < ytol and abs(face_coords[7]) < ytol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 3)
            if abs(face_coords[1] - Ly) < ytol and abs(face_coords[4] - Ly) < ytol and abs(face_coords[7] - Ly) < ytol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 4)
            if abs(face_coords[2]) < ztol and abs(face_coords[5]) < ztol and abs(face_coords[8]) < ztol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 5)
            if abs(face_coords[2] - Lz) < ztol and abs(face_coords[5] - Lz) < ztol and abs(face_coords[8] - Lz) < ztol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 6)
    plex.removeLabel("boundary_faces")

    name = "plexmesh"
    return UnstructuredMesh(plex, name, reorder=reorder,
                            distribution_parameters=distribution_parameters)
