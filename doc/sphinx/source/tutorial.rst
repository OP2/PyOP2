PyOP2 Tutorial
==============

In this tutorial we will define a two-element mesh and execute a kernel that
computes the mass and centre of each of the cells in the mesh. Our mesh looks
like:

  .. image:: images/mesh.png

First, we need to initialise PyOP2 by selecting a backend: ::

  from pyop2 import op2
  op2.init(backend='sequential')

We're using the sequential backend in this tutorial as it should work
everywhere, but you can also use `cuda`, `openmp`, or `opencl` if your setup
supports them.

Now we can define the data structures representing the mesh: ::

  # Two cells and four vertices
  cells       = op2.Set(2)
  vertices    = op2.Set(4)
  # Three vertices per cell
  cell_vertex = op2.Map(cells, vertices, 3, [ (0, 1, 2), (2, 3, 0) ] )

Sets are declared by passing their sizes. Mappings are specified by declaring an
`iteration set` and a `data set`. The iteration set is the set of entities that
the map entries are from, and the data set is the set of entities that the
entries are mapped to. The arity of the map is also required, as is a list
containing the map values. Our example mapping is from cells to vertices.

We also need to define some data on our sets: ::

  vertex_coords = op2.Dat(vertices, 2, [ (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0) ] )
  cell_centre = op2.Dat(cells, 2, [ (0.0, 0.0), (0.0, 0.0) ] )
  cell_mass = op2.Dat(cells, 1, [ 0.0, 0.0 ] )

Data (`Dat` s) is declared

A kernel that computes the centre and mass of the cells is defined by embedding
a C function in a string. ::

  mass_centre = op2.Kernel("""
  void mass_centre(double *x[2], double centre[2], double mass[1])
  {
    centre[0] = (x[0][0] + x[1][0] + x[2][0]) / 3.0;
    centre[1] = (x[0][1] + x[1][1] + x[2][1]) / 3.0;
    mass[0] = abs(x[0][0]*(x[1][1]-x[2][1]) + x[1][0]*(x[2][1]-x[0][1]) + x[2][0]*(x[0][1]-x[1][1]) ) / 2.0;
  }""", "mass_centre")

The kernel defines the computation for a single set element, in this case a
single cell. This kernel is invoked with the parallel loop syntax: ::

  op2.par_loop(mass_centre, cells,
               vertex_coords(cell_vertex, op2.READ),
               cell_centre(op2.IdentityMap, op2.WRITE),
               cell_mass(op2.IdentityMap, op2.WRITE))

The parallel loop invocation causes PyOP2 to execute the `mass_centre` kernel for every
element of the iteration set, `cells`. The subsequent arguments to the parallel
loop are the data sets that are to be passed to the kernel, and the order of
these aligns with the order that they are declared in the kernel. The first
argument to each `Dat` is the map through which the data is accessed. If the
data set is defined on the iteration set of the parallel loop, then the
identity map is used. In the case where the `Dat` is defined on another set, a
map from the iteration set to the data set must be passed. The second argument
is the access descriptor, which expresses how the argument is used in the
kernel. Possible values are `READ`, `WRITE`, `RW`, `INC`, `MAX`, and `MIN`. The
latter three will be explained later. The `READ`, `WRITE` and `RW` descriptors imply
that there will be no writes to a member of the dataset from more than one
element of the iteration set.
