import collections.abc
import ctypes
from dataclasses import dataclass
import itertools
import os
from typing import Optional, Tuple

import loopy as lp
from petsc4py import PETSc
import numpy as np

from pyop2 import compilation, mpi
from pyop2.caching import Cached
from pyop2.configuration import configuration
from pyop2.datatypes import IntType, as_ctypes
from pyop2.types import IterationRegion
from pyop2.utils import cached_property, get_petsc_dir
from pyop2 import op2


# We set eq=False to force identity-based hashing. This is required for when
# we check whether or not we have duplicate maps getting passed to the kernel.
@dataclass(eq=False, frozen=True)
class MapKernelArg:
    """Class representing a map argument to the kernel.

    :param arity: The arity of the map (how many indirect accesses are needed
        for each item of the iterset).
    :param offset: Tuple of integers describing the offset for each DoF in the
        base mesh needed to move up the column of an extruded mesh.
    """

    arity: int
    offset: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if not isinstance(self.offset, collections.abc.Hashable):
            raise ValueError("The provided offset must be hashable")

    @property
    def cache_key(self):
        return type(self), self.arity, self.offset


@dataclass(eq=False, frozen=True)
class PermutedMapKernelArg:
    """Class representing a permuted map input to the kernel.

    :param base_map: The underlying :class:`MapKernelArg`.
    :param permutation: Tuple of integers describing the applied permutation.
    """

    base_map: MapKernelArg
    permutation: Tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.permutation, collections.abc.Hashable):
            raise ValueError("The provided permutation must be hashable")

    @property
    def cache_key(self):
        return type(self), self.base_map.cache_key, tuple(self.permutation)


@dataclass(frozen=True)
class GlobalKernelArg:
    """Class representing a :class:`pyop2.types.Global` being passed to the kernel.

    :param dim: The shape of the data.
    """

    dim: Tuple[int, ...]

    @property
    def cache_key(self):
        return type(self), self.dim

    @property
    def maps(self):
        return ()


@dataclass(frozen=True)
class DatKernelArg:
    """Class representing a :class:`pyop2.types.Dat` being passed to the kernel.

    :param dim: The shape at each node of the dataset.
    :param map_: The map used for indirect data access. May be ``None``.
    :param index: The index if the :class:`pyop2.types.Dat` is
        a :class:`pyop2.types.DatView`.
    """

    dim: Tuple[int, ...]
    map_: MapKernelArg = None
    index: Optional[Tuple[int, ...]] = None

    @property
    def pack(self):
        from pyop2.codegen.builder import DatPack
        return DatPack

    @property
    def is_direct(self):
        """Is the data getting accessed directly?"""
        return self.map_ is None

    @property
    def is_indirect(self):
        """Is the data getting accessed indirectly?"""
        return not self.is_direct

    @property
    def cache_key(self):
        map_key = self.map_.cache_key if self.map_ is not None else None
        return type(self), self.dim, map_key, self.index

    @property
    def maps(self):
        if self.map_ is not None:
            return self.map_,
        else:
            return ()


@dataclass(frozen=True)
class MatKernelArg:
    """Class representing a :class:`pyop2.types.Mat` being passed to the kernel.

    :param dims: The shape at each node of each of the datasets.
    :param maps: The indirection maps.
    :param unroll: Is it impossible to set matrix values in 'blocks'?
    """
    dims: Tuple[Tuple[int, ...], Tuple[int, ...]]
    maps: Tuple[MapKernelArg, MapKernelArg]
    unroll: bool = False

    @property
    def pack(self):
        from pyop2.codegen.builder import MatPack
        return MatPack

    @property
    def cache_key(self):
        return type(self), self.dims, tuple(m.cache_key for m in self.maps), self.unroll


@dataclass(frozen=True)
class MixedDatKernelArg:
    """Class representing a :class:`pyop2.types.MixedDat` being passed to the kernel.

    :param arguments: Iterable of :class:`DatKernelArg` instances.
    """

    arguments: Tuple[DatKernelArg, ...]

    def __iter__(self):
        return iter(self.arguments)

    def __len__(self):
        return len(self.arguments)

    @property
    def cache_key(self):
        return tuple(a.cache_key for a in self.arguments)

    @property
    def maps(self):
        return tuple(m for a in self.arguments for m in a.maps)

    @property
    def pack(self):
        from pyop2.codegen.builder import DatPack
        return DatPack


@dataclass(frozen=True)
class MixedMatKernelArg:
    """Class representing a :class:`pyop2.types.MixedDat` being passed to the kernel.

    :param arguments: Iterable of :class:`MatKernelArg` instances.
    :param shape: The shape of the arguments array.
    """

    arguments: Tuple[MatKernelArg, ...]
    shape: Tuple[int, ...]

    def __iter__(self):
        return iter(self.arguments)

    def __len__(self):
        return len(self.arguments)

    @property
    def cache_key(self):
        return tuple(a.cache_key for a in self.arguments)

    @property
    def maps(self):
        return tuple(m for a in self.arguments for m in a.maps)

    @property
    def pack(self):
        from pyop2.codegen.builder import MatPack
        return MatPack


class GlobalKernel(Cached):
    """Class representing the generated code for the global computation.

    :param local_kernel: :class:`pyop2.LocalKernel` instance representing the
        local computation.
    :param arguments: An iterable of :class:`KernelArg` instances describing
        the arguments to the global kernel.
    :param extruded: Are we looping over an extruded mesh?
    :param constant_layers: If looping over an extruded mesh, are the layers the
        same for each base entity?
    :param subset: Are we iterating over a subset?
    :param iteration_region: :class:`IterationRegion` representing the set of
        entities being iterated over. Only valid if looping over an extruded mesh.
        Valid values are:
          - ``ON_BOTTOM``: iterate over the bottom layer of cells.
          - ``ON_TOP`` iterate over the top layer of cells.
          - ``ALL`` iterate over all cells (the default if unspecified)
          - ``ON_INTERIOR_FACETS`` iterate over all the layers
             except the top layer, accessing data two adjacent (in
             the extruded direction) cells at a time.
    :param pass_layer_arg: Should the wrapper pass the current layer into the
        kernel (as an `int`). Only makes sense for indirect extruded iteration.
    """

    _cppargs = []
    _libraries = []
    _system_headers = []

    _cache = {}

    @classmethod
    def _cache_key(cls, local_knl, arguments, **kwargs):
        key = [cls, local_knl.cache_key,
               *kwargs.items(), configuration["simd_width"]]

        key.extend([a.cache_key for a in arguments])

        counter = itertools.count()
        seen_maps = collections.defaultdict(lambda: next(counter))
        key.extend([seen_maps[m] for a in arguments for m in a.maps])

        return tuple(key)

    def __init__(self, local_kernel, arguments, *,
                 extruded=False,
                 constant_layers=False,
                 subset=False,
                 iteration_region=None,
                 pass_layer_arg=False):
        if self._initialized:
            return

        if not len(local_kernel.accesses) == len(arguments):
            raise ValueError("Number of arguments passed to the local "
                             "and global kernels do not match")

        if pass_layer_arg and not extruded:
            raise ValueError("Cannot request layer argument for non-extruded iteration")
        if constant_layers and not extruded:
            raise ValueError("Cannot request constant_layers argument for non-extruded iteration")

        self.local_kernel = local_kernel
        self.arguments = arguments
        self._extruded = extruded
        self._constant_layers = constant_layers
        self._subset = subset
        self._iteration_region = iteration_region
        self._pass_layer_arg = pass_layer_arg

        # Cache for stashing the compiled code
        self._func_cache = {}

        self._initialized = True

    @mpi.collective
    def __call__(self, comm, *args):
        """Execute the compiled kernel.

        :arg comm: Communicator the execution is collective over.
        :*args: Arguments to pass to the compiled kernel.
        """
        # If the communicator changes then we cannot safely use the in-memory
        # function cache. Note here that we are not using dup_comm to get a
        # stable communicator id because we will already be using the internal one.
        key = id(comm)
        try:
            func = self._func_cache[key]
        except KeyError:
            func = self.compile(comm)
            self._func_cache[key] = func
        func(*args)

    @property
    def _wrapper_name(self):
        import warnings
        warnings.warn("GlobalKernel._wrapper_name is a deprecated alias for GlobalKernel.name",
                      DeprecationWarning)
        return self.name

    @cached_property
    def name(self):
        return f"wrap_{self.local_kernel.name}"

    @cached_property
    def zipped_arguments(self):
        """Iterate through arguments for the local kernel and global kernel together."""
        return tuple(zip(self.local_kernel.arguments, self.arguments))

    @cached_property
    def builder(self):
        from pyop2.codegen.builder import WrapperBuilder

        builder = WrapperBuilder(kernel=self.local_kernel,
                                 subset=self._subset,
                                 extruded=self._extruded,
                                 constant_layers=self._constant_layers,
                                 iteration_region=self._iteration_region,
                                 pass_layer_to_kernel=self._pass_layer_arg)
        for arg in self.arguments:
            builder.add_argument(arg)
        return builder

    @cached_property
    def code_to_compile(self):
        """Return the C/C++ source code as a string."""
        from pyop2.codegen.rep2loopy import generate

        wrapper = generate(self.builder)
        if self._extruded:
            iname = "layer"
        else:
            iname = "n"

        has_matrix = any(isinstance(arg, MatKernelArg) for arg in self._args)
        has_rw = any(arg.access == op2.RW for arg in self._args)
        is_cplx = any(arg.dtype.name == 'complex128' for arg in self._args)
        vectorisable = not (has_matrix or has_rw) and (configuration["vectorization_strategy"])

        if (isinstance(self._kernel.code, lp.LoopKernel) and vectorisable):
            wrapper = lp.inline_callable_kernel(wrapper, self._kernel.name)
            if not is_cplx:
                wrapper = self.vectorise(wrapper, iname, configuration["simd_width"])
        code = lp.generate_code_v2(wrapper)

        if self.local_kernel.cpp:
            from lp.codegen.result import process_preambles
            preamble = "".join(process_preambles(getattr(code, "device_preambles", [])))
            device_code = "\n\n".join(str(dp.ast) for dp in code.device_programs)
            return preamble + "\nextern \"C\" {\n" + device_code + "\n}\n"
        return code.device_code()

    def vectorise(wrapper, iname, batch_size):
        """Return a vectorised version of wrapper, vectorising over iname.

        :arg wrapper: A loopy kernel to vectorise.
        :arg iname: The iteration index to vectorise over.
        :arg batch_size: The vector width."""
        if batch_size == 1:
            return wrapper

        # create constant zero vectors
        wrapper = wrapper.copy(target=lp.CVecTarget(batch_size))
        kernel = wrapper.root_kernel

        # split iname and vectorize the inner loop
        slabs = (1, 1)
        inner_iname = iname + "_batch"

        if configuration["vectorization_strategy"] == "ve":
            kernel = lp.split_iname(kernel, iname, batch_size, slabs=slabs, inner_tag="vec", inner_iname=inner_iname)

        alignment = configuration["alignment"]
        tmps = dict((name, tv.copy(alignment=alignment)) for name, tv in kernel.temporary_variables.items())
        kernel = kernel.copy(temporary_variables=tmps)

        from lp.preprocess import check_cvec_vectorizability, cvec_retag_and_privatize, realize_ilp

        kernel = realize_ilp(kernel)  # FIXME: do we also need to realize the reductions first?

        # try to vectorise with vector extensionn
        vector_inst, pragma_inst_to_tag, unr_inst_to_tag = check_cvec_vectorizability(kernel)

        # if not possible fall back to OpenMP SIMD pragmas or unrolling by retagging, then privatize
        kernel = cvec_retag_and_privatize(kernel, vector_inst, pragma_inst_to_tag, unr_inst_to_tag)

        wrapper = wrapper.with_root_kernel(kernel)

        return wrapper

    @PETSc.Log.EventDecorator()
    @mpi.collective
    def compile(self, comm):
        """Compile the kernel.

        :arg comm: The communicator the compilation is collective over.
        :returns: A ctypes function pointer for the compiled function.
        """
        compiler = configuration["compiler"]
        extension = "cpp" if self.local_kernel.cpp else "c"
        cppargs = (self._cppargs
                   + ["-I%s/include" % d for d in get_petsc_dir()]
                   + ["-I%s" % d for d in self.local_kernel.include_dirs]
                   + ["-I%s" % os.path.abspath(os.path.dirname(__file__))])
        ldargs = ["-L%s/lib" % d for d in get_petsc_dir()] + \
                 ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()] + \
                 ["-lpetsc", "-lm"] + self._libraries
        ldargs += self.local_kernel.ldargs

        return compilation.load(self, extension, self.name,
                                cppargs=cppargs,
                                ldargs=ldargs,
                                restype=ctypes.c_int,
                                compiler=compiler,
                                comm=comm)

    @cached_property
    def argtypes(self):
        """Return the ctypes datatypes of the compiled function."""
        # The first two arguments to the global kernel are the 'start' and 'stop'
        # indices. All other arguments are declared to be void pointers.
        dtypes = [as_ctypes(IntType)] * 2
        dtypes.extend([ctypes.c_voidp for _ in self.builder.wrapper_args[2:]])
        return tuple(dtypes)

    def num_flops(self, iterset):
        """Compute the number of FLOPs done by the kernel."""
        size = 1
        if iterset._extruded:
            region = self._iteration_region
            layers = np.mean(iterset.layers_array[:, 1] - iterset.layers_array[:, 0])
            if region is IterationRegion.INTERIOR_FACETS:
                size = layers - 2
            elif region not in {IterationRegion.TOP, IterationRegion.BOTTOM}:
                size = layers - 1
        return size * self.local_kernel.num_flops
