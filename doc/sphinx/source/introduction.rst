An introduction to PyOP2
========================

What is PyOP2?
--------------

In a nutshell, PyOP2 is a is a runtime framework that is designed to make it
easy to implement performance-portable finite element solvers.

The central philosophy of the PyOP2 design is that parallel
programming can be made easy when:

* We express algorithms as independent operations on independent items of data,
* where each item of data is a member of a collection,
* and the same operation is performed for every member of the collection.

An implementation of finite element assembly in this way would take the form of
the following pseudocode::

  forall elements:
    forall dof in element.dofs:
      forall quadpt in element.quadraturepoints:
        ele.localtensor[dof] += assemble(ele,dof,quadpt)

This is, in essence, the representation that the user specifies when writing
PyOP2 code. in the subsequent tutorial introduction, we will examine how this
representation is used in practice.

What does PyOP2 do?
-------------------

PyOP2 takes algorithms written in the representation shown above and generates
efficient parallel implementations for different targets. Presently supported
target backends include:

* Sequential C
* C with OpenMP
* OpenCL
* CUDA

In addition, the backends that execute code on the CPU (Sequential C and
OpenMP) can also use MPI for distributed-memory parallel execution.

The public interface of PyOP2 is provided entirely in Python.

The PyOP2 representation
------------------------

The PyOP2 representation consists of three different components:

* Kernels, which represent operations that are applied to every member of a
  given set of mesh entities.
* Data structures, which are used to represent the mesh topology and data
  defined on it.
* Access descriptors, which tell PyOP2 how to schedule kernels in parallel and
  how to get data to kernel invocations.

We will look at how to define these in PyOP2 in the next few sections. But
first, to get started, we must import and initialise PyOP2: ::

  from pyop2 import op2
  op2.init()

Kernels
-------

PyOP2 kernels are functions written in C, with the following special properties:

* Kernels define the computation that is performed for a single member of a set
  of mesh entities. This means that the user does not control the scheduling and
  invocation of a kernel; this is the responsibility of PyOP2.
* Pointers to each item of data that the kernel reads and/or writes are passed
  into kernels. This means that PyOP2 has to arrange for pointers to the correct
  data to be passed to the kernel.
* No pointer arithmetic in kernels is allowed.

To provide a concrete example, consider the following C function: ::

  void mass_centre(double *x[2], *double centre, double *mass)
  {
    centre[0] = (x[0][0] + x[1][0] + x[2][0]) / 3.0;
    centre[1] = (x[0][1] + x[1][1] + x[2][1]) / 3.0;

    *mass = abs(x[0][0]*(x[1][1] - x[2][1])
              + x[1][0]*(x[2][1] - x[0][1])
              + x[2][0]*(x[0][1] - x[1][1]) ) / 2.0;
  }

This kernel computes the mass and centre of a triangular cell. The parameters
are as follows:

* x holds the coordinates of the vertices. There are three vertices, each of
  which has two coordinate values. This is passed in as a list of pointers to an
  array of the coordinate values for each vertex. These values are read-only.
* centre holds the centre coordinate for the cell. Since
  there is only one centre, a single pointer to the centre values is passed. The
  centre values are write-only.
* mass holds the mass value for the cell, which is also passed in as a pointer
  and is also write-only.

Because mass_centre is a C function, we need to wrap it in a string to embed it
in our Python code. We then construct a Kernel object from this string: ::

  mass_centre_code = """void mass_centre(double *x[2], *double centre, double *mass)
  {
    centre[0] = (x[0][0] + x[1][0] + x[2][0]) / 3.0;
    centre[1] = (x[0][1] + x[1][1] + x[2][1]) / 3.0;

    *mass = abs(x[0][0]*(x[1][1] - x[2][1])
              + x[1][0]*(x[2][1] - x[0][1])
              + x[2][0]*(x[0][1] - x[1][1]) ) / 2.0;
  }"""
  mass_centre = op2.Kernel(mass_centre_code, "mass_centre")

The Kernel declaration also requires the name of the kernel function to be
passed. This is for PyOP2 to be able to invoke the function correctly, since
during compilation other functions will be included (such as math library
functions).

When the time comes to execute the kernel, PyOP2 needs to know how
to schedule the kernel for parallel execution and how to get the correct data to
each invocation of the kernel. The information that PyOP2 requires to do this is
outlined in the following two sections.

Data structures
---------------

Data structures are used to define a representation of the mesh topology and the
dat associated with the mesh. There are two parts to the mesh topology:

* Sets are used to represent collections of mesh entities, such as cells,
  vertices, edges, etc.
* Maps are used to represent the connectivity between different sets, which
  provides a representation of the mesh topology. A map from one set to another
  holds a list of integers that represent the values of the mapping.

In order to exemplify the mesh data structures, we consider a two-element mesh
with four nodes:

  .. image:: images/mesh.png

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
  cell_centre = op2.Dat(cells, 2)
  cell_mass = op2.Dat(cells, 1)

Data (`Dat` objects) are declared on a set, and can be of any dimension.

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
