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
efficient parallel implementations for different targets. Targets that are
presently supported include:

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

  void mass_centre(double *x[2], double *centre, double *mass)
  {
    centre[0] = (x[0][0] + x[1][0] + x[2][0]) / 3.0;
    centre[1] = (x[0][1] + x[1][1] + x[2][1]) / 3.0;

    *mass = abs(x[0][0]*(x[1][1] - x[2][1])
              + x[1][0]*(x[2][1] - x[0][1])
              + x[2][0]*(x[0][1] - x[1][1]) ) / 2.0;
  }

This kernel computes the mass and centre of a triangular cell. The parameters
are as follows:

* `x` holds the coordinates of the vertices. There are three vertices, each of
  which has two coordinate values. This is passed in as a list of pointers to an
  array of the coordinate values for each vertex. These values are read-only.
* `centre` holds the centre coordinate for the cell. Since
  there is only one centre, a single pointer to the centre values is passed. The
  centre values are write-only.
* `mass` holds the mass value for the cell, which is also passed in as a pointer
  and is also write-only.

Because `mass_centre` is a C function, we need to wrap it in a string to embed it
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

The kernel declaration also requires the name of the kernel function to be
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
data associated with the mesh. There are two parts to the mesh topology:

* Sets are used to represent collections of mesh entities, such as cells,
  vertices, edges, etc.
* Maps are used to represent the connectivity between different sets, which
  provides a representation of the mesh topology. A Map from one Set to another
  holds a list of integers that represent the values of the mapping.

Once mesh data structures are created, the data on the mesh can then be defined.
Data is defined using Dats, which are associated with a set and hold an item of
data for every member of the set it is associated with. Multiple Dats can be
associated with a single Set - this enables many field variables to be defined.

In order to exemplify the mesh data structures, we consider a two-element mesh
with four nodes:

  .. image:: images/mesh.png

The sets and maps representing the mesh are defined as follows: ::

  # Two cells and four vertices
  cells       = op2.Set(2)
  vertices    = op2.Set(4)
  # Three vertices per cell
  cell_vertex = op2.Map(cells, vertices, 3, [ (0, 1, 2), (2, 3, 0) ] )

The Sets are declared by passing in their sizes. Maps are declared by specifying
the Sets that they map from and to; these are referred to as the Iteration Set
and the Data Set respectively (the reason for these names will become clear in
the following section). Next, the arity of the map is required, which specifies
how many values in the data set a member of the iteration set maps to. Finally,
a list of the map values is required. In our example, we map from from cells to
vertices, because this particular mapping will be required later in the
tutorial.

Note that we do not create any explicit representation of edges of the mesh here
- they are not required for our example.

Now that we have created the mesh topology, we can define data on our sets. For
the purpose of this example, we define data for the vertex coordinates, and also
for storing the centre and mass of the cells: ::

  vertex_coords = op2.Dat(vertices, 2, [ (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0) ] )
  cell_centre = op2.Dat(cells, 2)
  cell_mass = op2.Dat(cells, 1)

The values of the vertex coordinates are provided in this case. The centre and
mass values are not provided. Their data is undefined, but will be written to
later on.

Access descriptors
------------------

Access descriptors tell PyOP2 how to get the right data to each invocation of a
kernel, and are passed when a parallel loop is invoked.

We will consider an example of a parallel loop invocation in order to explore
access descriptors: ::

  op2.par_loop(mass_centre, cells,
               vertex_coords(cell_vertex, op2.READ),
               cell_centre(op2.IdentityMap, op2.WRITE),
               cell_mass(op2.IdentityMap, op2.WRITE))

This is an invocation of the `mass_centre` kernel that we defined earlier. The
first parameter tells PyOP2 which kernel to invoke; the second parameter is the
set that it should iterate over.

Subsequently, an access descriptor must be passed for each argument that the
kernel takes. Access descriptors tell PyOP2:

* Which Dat the data for the argument should come from.
* Which Map the data should be accessed through.
* The access mode, which tells PyOP2 how the data is used by the kernel.

Access descriptors are created by calling Dat objects. The arguments to the call
are the Map for the descriptor, and the access mode. There are two different
cases for the Map that is passed:

* If the Dat is defined on the same set as the iteration set, then the identity
  map can be used. In this case, data items can be located for the current Set
  element.
* If the Dat is defined on a different Set to the one that is being iterated
  over, then an appropriate Map must be passed. An appropriate Map is one which
  maps from the Set that is being iterated over (the Iteration Set) to the Set
  that the Dat is defined on (the Data Set).

In the example above, the Identity Map is used for `cell_centre` and
`cell_mass`, since they are both defined on the set of cells. `vertex_coords` is
defined on the set of vertices, so the mapping from cells to vertices is used.

The access mode is:

* `op2.READ` when the data is only read,
* `op2.WRITE` when the data is only written to, and
* `op2.RW` when the data is read from and written to.

There are other access modes (`op2.INC`, `op2.MAX`, and `op2.MIN`), which are
not covered in this introduction.

When using the `op2.READ` and `op2.RW` access modes, PyOP2 does nothing to avoid
write conflicts. In the `op2.READ` case, it is not necessary. In the case of
`op2.RW`, it is only safe to read from and write to data that no other iteration
will touch.

What happened?
--------------

When `op2.par_loop` was called, PyOP2 called the currently-selected backend
(which is the sequential backend by default), which generated code that iterates
over the set of colles and invokes the user's kernel for every iteration of the
loop. Code that constructs the arguments to the kernel was also generated from
the access descriptors.

The backend then launched the target compiler to compile the generated code, and
linked it back into the running Python interpreter. Finally, the backend calls
the compiled code.

If you pasted all the Python code above into the interpreter, then the effect
of the kernel should now be visible. In order to see this, the data accessors
can be used to read the values of the Dats. We will use the read-only accessors
to inspect the values of the `cell_centre` and `cell_mass` Dats: ::

  cell_centre.data_ro
  cell_mass.data_ro

You should see the output: ::

  array([[ 0.66666667,  0.33333333],
         [ 0.33333333,  0.66666667]])
  array([[0.5],
         [0.5]])

Data structures for finite element assembly
-------------------------------------------

Whilst we have already implemented a minimal PyOP2 program, we have yet to
implement a finite element method using PyOP2. There are several extra
components to this process, but we will begin by looking at how to construct the
data structures for finite element assembly.

Matrices in PyOP2 consists of Sparsities, which represent the non-zero
structure of a matrix, and Matrices, which define data on Sparsities. Because it
is assumed that matrices are only constructed through global assembly, there is
no way to specify the structure of a Sparsity manually.

Instead, the Sparsity structure of matrices is composed from pairs of Maps.
The product of these Maps, the Row Map and the Col Map, is used to generate the
Sparsity pattern. This produces an appropriate pattern for
