import itertools
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import reduce

import numpy
import loopy as lp
from loopy.symbolic import SubArrayRef
from loopy.types import OpaqueType
from pymbolic import var
import pymbolic.primitives as pym
from pyop2.global_kernel import (GlobalKernelArg, DatKernelArg, MixedDatKernelArg,
                                 MatKernelArg, MixedMatKernelArg, PermutedMapKernelArg)
from pyop2.codegen.representation import (Accumulate, Argument, Comparison,
                                          DummyInstruction, Extent, FixedIndex,
                                          FunctionCall, Index, Indexed,
                                          KernelInst, Literal, LogicalAnd,
                                          Materialise, Max, Min, MultiIndex,
                                          NamedLiteral, PackInst,
                                          PreUnpackInst, Product, RuntimeIndex,
                                          Sum, Symbol, UnpackInst, Variable,
                                          When, Zero)
from pyop2.datatypes import IntType
from pyop2.op2 import (ALL, INC, MAX, MIN, ON_BOTTOM, ON_INTERIOR_FACETS,
                       ON_TOP, READ, RW, WRITE)
from pyop2.utils import cached_property


class PetscMat(OpaqueType):

    def __init__(self):
        super().__init__(name="Mat")


class Map(object):

    __slots__ = ("values", "offset", "interior_horizontal",
                 "variable", "unroll", "layer_bounds",
                 "prefetch", "_pmap_count")

    def __init__(self, interior_horizontal, layer_bounds,
                 arity, dtype,
                 offset=None, unroll=False,
                 extruded=False, constant_layers=False):
        self.variable = extruded and not constant_layers
        self.unroll = unroll
        self.layer_bounds = layer_bounds
        self.interior_horizontal = interior_horizontal
        self.prefetch = {}

        shape = (None, arity)
        values = Argument(shape, dtype=dtype, pfx="map")
        if offset is not None:
            assert type(offset) == tuple
            offset = numpy.array(offset, dtype=numpy.int32)
            if len(set(offset)) == 1:
                offset = Literal(offset[0], casting=True)
            else:
                offset = NamedLiteral(offset, parent=values, suffix="offset")

        self.values = values
        self.offset = offset
        self._pmap_count = itertools.count()

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype

    def indexed(self, multiindex, layer=None, permute=lambda x: x):
        n, i, f = multiindex
        if layer is not None and self.offset is not None:
            # For extruded mesh, prefetch the indirections for each map, so that they don't
            # need to be recomputed.
            # First prefetch the base map (not dependent on layers)
            base_key = None
            if base_key not in self.prefetch:
                j = Index()
                base = Indexed(self.values, (n, permute(j)))
                self.prefetch[base_key] = Materialise(PackInst(), base, MultiIndex(j))

            base = self.prefetch[base_key]

            # Now prefetch the extruded part of the map (inside the layer loop).
            # This is necessary so loopy DTRT for MatSetValues
            # Different f values need to be treated separately.
            key = f.extent
            if key is None:
                key = 1
            if key not in self.prefetch:
                bottom_layer, _ = self.layer_bounds
                k = Index(f.extent if f.extent is not None else 1)
                offset = Sum(Sum(layer, Product(Literal(numpy.int32(-1)), bottom_layer)), k)
                j = Index()
                # Inline map offsets where all entries are identical.
                if self.offset.shape == ():
                    offset = Product(offset, self.offset)
                else:
                    offset = Product(offset, Indexed(self.offset, (j,)))
                base = Indexed(base, (j, ))
                self.prefetch[key] = Materialise(PackInst(), Sum(base, offset), MultiIndex(k, j))

            return Indexed(self.prefetch[key], (f, i)), (f, i)
        else:
            assert f.extent == 1 or f.extent is None
            base = Indexed(self.values, (n, permute(i)))
            return base, (f, i)

    def indexed_vector(self, n, shape, layer=None, permute=lambda x: x):
        shape = self.shape[1:] + shape
        if self.interior_horizontal:
            shape = (2, ) + shape
        else:
            shape = (1, ) + shape
        f, i, j = (Index(e) for e in shape)
        base, (f, i) = self.indexed((n, i, f), layer=layer, permute=permute)
        init = Sum(Product(base, Literal(numpy.int32(j.extent))), j)
        pack = Materialise(PackInst(), init, MultiIndex(f, i, j))
        multiindex = tuple(Index(e) for e in pack.shape)
        return Indexed(pack, multiindex), multiindex


class PMap(Map):
    __slots__ = ("permutation",)

    def __init__(self, map_, permutation):
        # Copy over properties
        self.variable = map_.variable
        self.unroll = map_.unroll
        self.layer_bounds = map_.layer_bounds
        self.interior_horizontal = map_.interior_horizontal
        self.prefetch = {}
        self.values = map_.values
        self.offset = map_.offset
        offset = map_.offset
        # TODO: this is a hack, rep2loopy should be in charge of
        # generating all names!
        count = next(map_._pmap_count)
        if offset is not None:
            if offset.shape:
                # Have a named literal
                offset = offset.value[permutation]
                offset = NamedLiteral(offset, parent=self.values, suffix=f"permutation{count}_offset")
            else:
                offset = map_.offset
        self.offset = offset
        self.permutation = NamedLiteral(permutation, parent=self.values, suffix=f"permutation{count}")

    def indexed(self, multiindex, layer=None):
        permute = lambda x: Indexed(self.permutation, (x,))
        return super().indexed(multiindex, layer=layer, permute=permute)

    def indexed_vector(self, n, shape, layer=None):
        permute = lambda x: Indexed(self.permutation, (x,))
        return super().indexed_vector(n, shape, layer=layer, permute=permute)


class Pack(metaclass=ABCMeta):

    def pick_loop_indices(self, loop_index, layer_index=None, entity_index=None):
        """Override this to select the loop indices used by a pack for indexing."""
        return (loop_index, layer_index)

    @abstractmethod
    def kernel_arg(self, loop_indices=None):
        pass

    @abstractmethod
    def emit_pack_instruction(self, *, loop_indices=None):
        """Either yield an instruction, or else return an empty tuple (to indicate no instruction)"""

    @abstractmethod
    def emit_unpack_instruction(self, *, loop_indices=None):
        """Either yield an instruction, or else return an empty tuple (to indicate no instruction)"""


class GlobalPack(Pack):

    def __init__(self, outer, access, init_with_zero=False):
        self.outer = outer
        self.access = access
        self.init_with_zero = init_with_zero

    def kernel_arg(self, loop_indices=None):
        pack = self.pack(loop_indices)
        return Indexed(pack, (Index(e) for e in pack.shape))

    def emit_pack_instruction(self, loop_indices=None):
        if hasattr(self, "_pack"):
            return self._pack

        shape = self.outer.shape
        if self.access is READ:
            # No packing required
            return self.outer
        # We don't need to pack for memory layout, however packing
        # globals that are written is required such that subsequent
        # vectorisation loop transformations privatise these reduction
        # variables. The extra memory movement cost is minimal.
        loop_indices = self.pick_loop_indices(*loop_indices)
        if self.init_with_zero:
            also_zero = {MIN, MAX}
        else:
            also_zero = set()
        if self.access in {INC, WRITE} | also_zero:
            val = Zero((), self.outer.dtype)
            multiindex = MultiIndex(*(Index(e) for e in shape))
            self._pack = Materialise(PackInst(loop_indices), val, multiindex)
        elif self.access in {READ, RW, MIN, MAX} - also_zero:
            multiindex = MultiIndex(*(Index(e) for e in shape))
            expr = Indexed(self.outer, multiindex)
            self._pack = Materialise(PackInst(loop_indices), expr, multiindex)
        else:
            raise ValueError("Don't know how to initialise pack for '%s' access" % self.access)
        return self._pack

    def emit_unpack_instruction(self, *, loop_indices=None):
        pack = self.pack(loop_indices)
        loop_indices = self.pick_loop_indices(*loop_indices)
        if pack is None:
            return ()
        elif self.access is READ:
            return ()
        elif self.access in {INC, MIN, MAX}:
            op = {INC: Sum,
                  MIN: Min,
                  MAX: Max}[self.access]
            multiindex = tuple(Index(e) for e in pack.shape)
            rvalue = Indexed(self.outer, multiindex)
            yield Accumulate(UnpackInst(loop_indices), rvalue, op(rvalue, Indexed(pack, multiindex)))
        else:
            multiindex = tuple(Index(e) for e in pack.shape)
            rvalue = Indexed(self.outer, multiindex)
            yield Accumulate(UnpackInst(loop_indices), rvalue, Indexed(pack, multiindex))


class DatPack(Pack):
    name_generator = itertools.count()

    def __init__(self, outer, access, map_=None, interior_horizontal=False,
                 view_index=None, layer_bounds=None,
                 init_with_zero=False, within_inames=None):
        self.outer = outer
        self.map_ = map_
        self.access = access
        self.interior_horizontal = interior_horizontal
        self.view_index = view_index
        self.layer_bounds = layer_bounds
        self.init_with_zero = init_with_zero

        self.within_inames = within_inames
        idx = next(self.name_generator)
        self.id = idx
        self.iname = f"i{idx}"
        self.diminame=f"dimi{idx}"
        self.temp_var_name = f"t{idx}"
        self.map_name = "MYMAP"
        self.extent = f"n{idx}"
        self.domain = f"{{[{self.iname}]: 0<={self.iname}<{self.map_.shape[1]}}}"
        self.temp_var = lp.TemporaryVariable(self.temp_var_name, self.outer.dtype, shape=tuple(self.map_.shape[1:]))
        self.array_arg = lp.GlobalArg(self.outer.name, self.outer.dtype, shape=(None,), strides=lp.auto)
        self.map_arg = lp.GlobalArg("MYMAP", IntType, shape=(None, 1), strides=lp.auto)
        self.kernel_data = [self.temp_var, self.array_arg, self.map_arg]


        self.pack_instructions = self.gen_pack_instructions()

    def _mask(self, map_):
        """Override this if the map_ needs a masking condition."""
        return None

    def _rvalue(self, multiindex, loop_indices=None):
        """Returns indexed Dat and masking condition to apply to reads/writes.

        If the masking condition is None, no mask is applied,
        otherwise the pack/unpack will be wrapped in When(mask, expr).
        This is used for the case where maps might have negative entries.
        """
        f, i, *j = multiindex
        n, layer = self.pick_loop_indices(*loop_indices)
        if self.view_index is not None:
            raise NotImplementedError
            j = tuple(j) + tuple(FixedIndex(i) for i in self.view_index)
        map_, (f, i) = self.map_.indexed((n, i, f), layer=layer)
        return Indexed(self.outer, MultiIndex(map_, *j)), self._mask(map_)


    def emit_pack_instruction(self):
        raise Exception("Need to chagne API")

    def gen_pack_instructions(self):
        """Return a temporary variable?"""
        if self.map_ is None:
            return None

        if hasattr(self, "_pack"):
            return self._pack,

        if self.interior_horizontal:
            shape = (2, )
        else:
            shape = (1, )

        if self.view_index:
            raise NotImplementedError
        if self.interior_horizontal:
            raise NotImplementedError

        shape = shape + self.map_.shape[1:]
        if self.view_index is None:
            shape = shape + self.outer.shape[1:]

        assignee = pym.Subscript(var(self.temp_var_name), (var(self.iname),))

        if self.access in {INC, WRITE}:
            expr = 0
        elif self.access in {READ, RW}:
            expr = pym.Subscript(var(self.outer.name), var(self.iname))
        elif self.access in {MIN, MAX}:
            raise NotImplementedError
            expr = 0 if self.init_with_zero else pym.Subscript(self.outer.name, var(self.iname))
        else:
            raise ValueError("Don't know how to initialise pack for '%s' access" % self.access)
        self._pack = lp.Assignment(assignee, expr, within_inames=self.within_inames,
                id=f"insn_{self.id}", within_inames_is_final=True)
        return self._pack,

    def kernel_arg(self, loop_indices=None):
        # This shouldn't be necessary...
        raise Exception
        if self.map_ is None:
            raise NotImplementedError
            if loop_indices is None:
                raise ValueError("Need iteration index")
            n, layer = self.pick_loop_indices(*loop_indices)
            shape = self.outer.shape
            if self.view_index is None:
                multiindex = (n, ) + tuple(Index(e) for e in shape[1:])
            else:
                multiindex = (n, ) + tuple(FixedIndex(i) for i in self.view_index)
            return Indexed(self.outer, multiindex)
        else:
            pack = self.pack(loop_indices)
            shape = pack.shape
            return Indexed(pack, (Index(e) for e in shape))

    def emit_unpack_instruction(self, *, depends_on=None):
        if self.access is READ:
            return ()
        elif self.access in {INC, MIN, MAX}:
            raise NotImplementedError
            op = {INC: Sum,
                  MIN: Min,
                  MAX: Max}[self.access]
            multiindex = tuple(Index(e) for e in pack.shape)
            rvalue, mask = self._rvalue(multiindex, loop_indices=loop_indices)
            acc = Accumulate(UnpackInst(), rvalue, op(rvalue, Indexed(pack, multiindex)))
            if mask is None:
                yield acc
            else:
                yield When(mask, acc)
        else:
            # TODO actually handle the map!!
            expr = pym.Subscript(pym.Variable(self.temp_var_name), (var(self.iname),))
            assignee = pym.Subscript(var(self.outer.name),
                                     (pym.Subscript(var(self.map_name), (var("outeriname"),var(self.iname))),))
            yield lp.Assignment(assignee, expr, depends_on=depends_on,
                                 within_inames=self.within_inames)


class MixedDatPack(Pack):
    def __init__(self, packs, access, dtype, interior_horizontal):
        self.packs = packs
        self.access = access
        self.dtype = dtype
        self.interior_horizontal = interior_horizontal

    def pack(self, loop_indices=None):
        if hasattr(self, "_pack"):
            return self._pack

        flat_shape = numpy.sum(tuple(numpy.prod(p.map_.shape[1:] + p.outer.shape[1:]) for p in self.packs))

        if self.interior_horizontal:
            _shape = (2,)
            flat_shape *= 2
        else:
            _shape = (1,)

        if self.access in {INC, WRITE}:
            val = Zero((), self.dtype)
            multiindex = MultiIndex(Index(flat_shape))
            self._pack = Materialise(PackInst(), val, multiindex)
        elif self.access in {READ, RW, MIN, MAX}:
            multiindex = MultiIndex(Index(flat_shape))
            val = Zero((), self.dtype)
            expressions = []
            offset = 0
            for p in self.packs:
                shape = _shape + p.map_.shape[1:] + p.outer.shape[1:]
                mi = MultiIndex(*(Index(e) for e in shape))
                expr, mask = p._rvalue(mi, loop_indices)
                extents = [numpy.prod(shape[i+1:], dtype=numpy.int32) for i in range(len(shape))]
                index = reduce(Sum, [Product(i, Literal(IntType.type(e), casting=False)) for i, e in zip(mi, extents)], Literal(IntType.type(0), casting=False))
                indices = MultiIndex(Sum(index, Literal(IntType.type(offset), casting=False)),)
                offset += numpy.prod(shape, dtype=numpy.int32)
                if mask is not None:
                    expr = When(mask, expr)
                expressions.append(expr)
                expressions.append(indices)

            self._pack = Materialise(PackInst(), val, multiindex, *expressions)
        else:
            raise ValueError("Don't know how to initialise pack for '%s' access" % self.access)

        return self._pack

    def kernel_arg(self, loop_indices=None):
        pack = self.pack(loop_indices)
        shape = pack.shape
        return Indexed(pack, (Index(e) for e in shape))

    def emit_pack_instruction(self, *, loop_indices=None):
        return ()

    def emit_unpack_instruction(self, *, loop_indices=None):
        pack = self.pack(loop_indices)
        if self.access is READ:
            return ()
        else:
            if self.interior_horizontal:
                _shape = (2,)
            else:
                _shape = (1,)
            offset = 0
            for p in self.packs:
                shape = _shape + p.map_.shape[1:] + p.outer.shape[1:]
                mi = MultiIndex(*(Index(e) for e in shape))
                rvalue, mask = p._rvalue(mi, loop_indices)
                extents = [numpy.prod(shape[i+1:], dtype=numpy.int32) for i in range(len(shape))]
                index = reduce(Sum, [Product(i, Literal(IntType.type(e), casting=False)) for i, e in zip(mi, extents)], Literal(IntType.type(0), casting=False))
                indices = MultiIndex(Sum(index, Literal(IntType.type(offset), casting=False)),)
                rhs = Indexed(pack, indices)
                offset += numpy.prod(shape, dtype=numpy.int32)

                if self.access in {INC, MIN, MAX}:
                    op = {INC: Sum,
                          MIN: Min,
                          MAX: Max}[self.access]
                    rhs = op(rvalue, rhs)

                acc = Accumulate(UnpackInst(), rvalue, rhs)
                if mask is None:
                    yield acc
                else:
                    yield When(mask, acc)


class MatPack(Pack):

    count = itertools.count()

    insertion_names = {False: "MatSetValuesBlockedLocal",
                       True: "MatSetValuesLocal"}
    """Function call name for inserting into the PETSc Mat. The keys
       are whether or not maps are "unrolled" (addressing dofs) or
       blocked (addressing nodes)."""

    def __init__(self, outer, access, maps, dims, dtype, interior_horizontal=False):
        self.outer = outer
        self.access = access
        self.maps = maps
        self.dims = dims
        self.dtype = dtype
        self.interior_horizontal = interior_horizontal

    @cached_property
    def shapes(self):
        ((rdim, cdim), ), = self.dims
        rmap, cmap = self.maps
        if self.interior_horizontal:
            shape = (2, )
        else:
            shape = (1, )
        rshape = shape + rmap.shape[1:] + (rdim, )
        cshape = shape + cmap.shape[1:] + (cdim, )
        return (rshape, cshape)

    def pack(self, loop_indices=None, only_declare=False):
        if hasattr(self, "_pack"):
            return self._pack
        shape = tuple(itertools.chain(*self.shapes))
        if only_declare:
            pack = Variable(f"matpack{next(self.count)}", shape, self.dtype)
            self._pack = pack
        if self.access in {WRITE, INC}:
            val = Zero((), self.dtype)
            multiindex = MultiIndex(*(Index(e) for e in shape))
            pack = Materialise(PackInst(), val, multiindex)
            self._pack = pack
        else:
            raise ValueError("Unexpected access type")
        return self._pack

    def kernel_arg(self, loop_indices=None):
        pack = self.pack(loop_indices=loop_indices)
        return Indexed(pack, tuple(Index(e) for e in pack.shape))

    def emit_pack_instruction(self, *, loop_indices=None):
        return ()

    def emit_unpack_instruction(self, *, loop_indices=None):
        from pyop2.codegen.rep2loopy import register_petsc_function
        ((rdim, cdim), ), = self.dims
        rmap, cmap = self.maps
        n, layer = self.pick_loop_indices(*loop_indices)
        unroll = any(m.unroll for m in self.maps)
        if unroll:
            maps = [map_.indexed_vector(n, (dim, ), layer=layer)
                    for map_, dim in zip(self.maps, (rdim, cdim))]
        else:
            maps = []
            for map_ in self.maps:
                i = Index()
                if self.interior_horizontal:
                    f = Index(2)
                else:
                    f = Index(1)
                maps.append(map_.indexed((n, i, f), layer=layer))
        (rmap, cmap), (rindices, cindices) = zip(*maps)

        pack = self.pack(loop_indices=loop_indices)
        name = self.insertion_names[unroll]
        if unroll:
            # The shape of MatPack is
            # (row, cols) if it has vector BC
            # (block_rows, row_cmpt, block_cols, col_cmpt) otherwise
            free_indices = rindices + cindices
            pack = Indexed(pack, free_indices)
        else:
            free_indices = rindices + (Index(), ) + cindices + (Index(), )
            pack = Indexed(pack, free_indices)

        access = Symbol({WRITE: "INSERT_VALUES",
                         INC: "ADD_VALUES"}[self.access])

        rextent = Extent(MultiIndex(*rindices))
        cextent = Extent(MultiIndex(*cindices))

        register_petsc_function(name)

        call = FunctionCall(name,
                            UnpackInst(),
                            (self.access, READ, READ, READ, READ, READ, READ),
                            free_indices,
                            self.outer,
                            rextent,
                            rmap,
                            cextent,
                            cmap,
                            pack,
                            access)

        yield call


class MixedMatPack(Pack):

    def __init__(self, packs, access, dtype, block_shape):
        self.access = access
        assert len(block_shape) == 2
        self.packs = numpy.asarray(packs).reshape(block_shape)
        self.dtype = dtype

    def pack(self, loop_indices=None):
        if hasattr(self, "_pack"):
            return self._pack
        rshape = 0
        cshape = 0
        # Need to compute row and col shape based on individual pack shapes
        for p in self.packs[:, 0]:
            shape, _ = p.shapes
            rshape += numpy.prod(shape, dtype=int)
        for p in self.packs[0, :]:
            _, shape = p.shapes
            cshape += numpy.prod(shape, dtype=int)
        shape = (rshape, cshape)
        if self.access in {WRITE, INC}:
            val = Zero((), self.dtype)
            multiindex = MultiIndex(*(Index(e) for e in shape))
            pack = Materialise(PackInst(), val, multiindex)
            self._pack = pack
            return pack
        else:
            raise ValueError("Unexpected access type")

    def kernel_arg(self, loop_indices=None):
        pack = self.pack(loop_indices=loop_indices)
        return Indexed(pack, tuple(Index(e) for e in pack.shape))

    def emit_pack_instruction(self, *, loop_indices=None):
        return ()

    def emit_unpack_instruction(self, *,
                                loop_indices=None):
        pack = self.pack(loop_indices=loop_indices)
        mixed_to_local = []
        local_to_global = []
        roffset = 0
        for row in self.packs:
            coffset = 0
            for p in row:
                rshape, cshape = p.shapes
                pack_ = p.pack(loop_indices=loop_indices, only_declare=True)
                rindices = tuple(Index(e) for e in rshape)
                cindices = tuple(Index(e) for e in cshape)
                indices = MultiIndex(*rindices, *cindices)
                lvalue = Indexed(pack_, indices)
                rextents = [numpy.prod(rshape[i+1:], dtype=numpy.int32) for i in range(len(rshape))]
                cextents = [numpy.prod(cshape[i+1:], dtype=numpy.int32) for i in range(len(cshape))]
                flat_row_index = reduce(Sum, [Product(i, Literal(IntType.type(e), casting=False))
                                              for i, e in zip(rindices, rextents)],
                                        Literal(IntType.type(0), casting=False))
                flat_col_index = reduce(Sum, [Product(i, Literal(IntType.type(e), casting=False))
                                              for i, e in zip(cindices, cextents)],
                                        Literal(IntType.type(0), casting=False))

                flat_index = MultiIndex(Sum(flat_row_index, Literal(IntType.type(roffset), casting=False)),
                                        Sum(flat_col_index, Literal(IntType.type(coffset), casting=False)))
                rvalue = Indexed(pack, flat_index)
                # Copy from local mixed element tensor into non-mixed
                mixed_to_local.append(Accumulate(PreUnpackInst(), lvalue, rvalue))
                # And into global matrix.
                local_to_global.extend(p.emit_unpack_instruction(loop_indices=loop_indices))
                coffset += numpy.prod(cshape, dtype=numpy.int32)
            roffset += numpy.prod(rshape, dtype=numpy.int32)
        yield from iter(mixed_to_local)
        yield from iter(local_to_global)


class WrapperBuilder(object):

    def __init__(self, *, kernel, subset, extruded, constant_layers, iteration_region=None, single_cell=False,
                 pass_layer_to_kernel=False, forward_arg_types=()):
        self.kernel = kernel
        self.local_knl_args = iter(kernel.arguments)
        self.arguments = []
        self.argument_accesses = []
        self.packed_args = []
        self.indices = []
        self.maps = OrderedDict()
        self.subset = subset
        self.extruded = extruded
        self.constant_layers = constant_layers
        if iteration_region is None:
            self.iteration_region = ALL
        else:
            self.iteration_region = iteration_region
        self.pass_layer_to_kernel = pass_layer_to_kernel
        self.single_cell = single_cell
        self.forward_arguments = tuple(Argument((), fa, pfx="farg") for fa in forward_arg_types)
        self.loop_iname = "outeriname"
        self.kernel_parameters = {}

    @property
    def requires_zeroed_output_arguments(self):
        return self.kernel.requires_zeroed_output_arguments

    @cached_property
    def loop_extents(self):
        return (Argument((), IntType, name="start"),
                Argument((), IntType, name="end"))

    @cached_property
    def _loop_index(self):
        start, end = self.loop_extents
        return RuntimeIndex(start, end,
                            LogicalAnd(
                                Comparison("<=", Zero((), numpy.int32), start),
                                Comparison("<=", start, end)),
                            name="n")

    @cached_property
    def _subset_indices(self):
        return Argument(("end", ), IntType, name="subset_indices")

    @cached_property
    def loop_index(self):
        n = self._loop_index
        if self.subset:
            n = Materialise(PackInst(), Indexed(self._subset_indices, MultiIndex(n)), MultiIndex())
        return n

    @cached_property
    def _layers_array(self):
        if self.constant_layers:
            return Argument((1, 2), IntType, name="layers")
        else:
            return Argument((None, 2), IntType, name="layers")

    @cached_property
    def bottom_layer(self):
        if self.iteration_region == ON_TOP:
            return Materialise(PackInst(),
                               Indexed(self._layers_array, (self._layer_index, FixedIndex(0))),
                               MultiIndex())
        else:
            start, _ = self.layer_extents
            return start

    @cached_property
    def top_layer(self):
        if self.iteration_region == ON_BOTTOM:
            return Materialise(PackInst(),
                               Sum(Indexed(self._layers_array, (self._layer_index, FixedIndex(1))),
                                   Literal(IntType.type(-1))),
                               MultiIndex())
        else:
            _, end = self.layer_extents
            return end

    @cached_property
    def layer_extents(self):
        if self.iteration_region == ON_BOTTOM:
            start = Indexed(self._layers_array, (self._layer_index, FixedIndex(0)))
            end = Sum(Indexed(self._layers_array, (self._layer_index, FixedIndex(0))),
                      Literal(IntType.type(1)))
        elif self.iteration_region == ON_TOP:
            start = Sum(Indexed(self._layers_array, (self._layer_index, FixedIndex(1))),
                        Literal(IntType.type(-2)))
            end = Sum(Indexed(self._layers_array, (self._layer_index, FixedIndex(1))),
                      Literal(IntType.type(-1)))
        elif self.iteration_region == ON_INTERIOR_FACETS:
            start = Indexed(self._layers_array, (self._layer_index, FixedIndex(0)))
            end = Sum(Indexed(self._layers_array, (self._layer_index, FixedIndex(1))),
                      Literal(IntType.type(-2)))
        elif self.iteration_region == ALL:
            start = Indexed(self._layers_array, (self._layer_index, FixedIndex(0)))
            end = Sum(Indexed(self._layers_array, (self._layer_index, FixedIndex(1))),
                      Literal(IntType.type(-1)))
        else:
            raise ValueError("Unknown iteration region")
        return (Materialise(PackInst(), start, MultiIndex()),
                Materialise(PackInst(), end, MultiIndex()))

    @cached_property
    def _layer_index(self):
        if self.constant_layers:
            return FixedIndex(0)
        else:
            return self.loop_index

    @cached_property
    def layer_index(self):
        if self.extruded:
            start, end = self.layer_extents
            return RuntimeIndex(start, end,
                                LogicalAnd(
                                    Comparison("<=", Zero((), numpy.int32), start),
                                    Comparison("<=", start, end)),
                                name="layer")
        else:
            return None

    @property
    def loop_indices(self):
        if self.extruded:
            return (self.loop_index, self.layer_index, self._loop_index)
        else:
            return (self.loop_index, None, self._loop_index)

    def add_argument(self, arg):
        local_arg = next(self.local_knl_args)
        access = local_arg.access
        dtype = local_arg.dtype
        interior_horizontal = self.iteration_region == ON_INTERIOR_FACETS

        if isinstance(arg, GlobalKernelArg):
            argument = Argument(arg.dim, dtype, pfx="glob")

            pack = GlobalPack(argument, access,
                              init_with_zero=self.requires_zeroed_output_arguments)
            self.arguments.append(argument)
        elif isinstance(arg, DatKernelArg):
            if arg.dim == ():
                shape = (None, 1)
            else:
                shape = (None, *arg.dim)
            argument = Argument(shape, dtype, pfx="dat")

            if arg.is_indirect:
                map_ = self._add_map(arg.map_)
            else:
                map_ = None
            pack = arg.pack(argument, access, map_=map_,
                            interior_horizontal=interior_horizontal,
                            view_index=arg.index,
                            init_with_zero=self.requires_zeroed_output_arguments,
                            within_inames=frozenset([self.loop_iname]))
            self.arguments.append(argument)
        elif isinstance(arg, MixedDatKernelArg):
            packs = []
            for a in arg:
                if a.dim == ():
                    shape = (None, 1)
                else:
                    shape = (None, *a.dim)
                argument = Argument(shape, dtype, pfx="mdat")

                if a.is_indirect:
                    map_ = self._add_map(a.map_)
                else:
                    map_ = None

                packs.append(arg.pack(argument, access, map_,
                                      interior_horizontal=interior_horizontal,
                                      init_with_zero=self.requires_zeroed_output_arguments))
                self.arguments.append(argument)
            pack = MixedDatPack(packs, access, dtype,
                                interior_horizontal=interior_horizontal)
        elif isinstance(arg, MatKernelArg):
            argument = Argument((), PetscMat(), pfx="mat")
            maps = tuple(self._add_map(m, arg.unroll)
                         for m in arg.maps)
            pack = arg.pack(argument, access, maps,
                            arg.dims, dtype,
                            interior_horizontal=interior_horizontal)
            self.arguments.append(argument)
        elif isinstance(arg, MixedMatKernelArg):
            packs = []
            for a in arg:
                argument = Argument((), PetscMat(), pfx="mat")
                maps = tuple(self._add_map(m, a.unroll)
                             for m in a.maps)

                packs.append(arg.pack(argument, access, maps,
                                      a.dims, dtype,
                                      interior_horizontal=interior_horizontal))
                self.arguments.append(argument)
            pack = MixedMatPack(packs, access, dtype,
                                arg.shape)
        else:
            raise ValueError("Unhandled argument type")

        self.packed_args.append(pack)
        self.argument_accesses.append(access)

    def _add_map(self, map_, unroll=False):
        if map_ is None:
            return None
        interior_horizontal = self.iteration_region == ON_INTERIOR_FACETS
        key = map_
        try:
            return self.maps[key]
        except KeyError:
            if isinstance(map_, PermutedMapKernelArg):
                imap = self._add_map(map_.base_map, unroll)
                map_ = PMap(imap, numpy.asarray(map_.permutation, dtype=IntType))
            else:
                map_ = Map(interior_horizontal,
                           (self.bottom_layer, self.top_layer),
                           arity=map_.arity, offset=map_.offset, dtype=IntType,
                           unroll=unroll,
                           extruded=self.extruded,
                           constant_layers=self.constant_layers)
            self.maps[key] = map_
            return map_

    @cached_property
    def loopy_argument_accesses(self):
        """Loopy wants the CallInstruction to have argument access
        descriptors aligned with how the callee treats the function.
        In the cases of TSFC kernels with WRITE access, this is not
        how we treats the function, so we have to keep track of the
        difference here."""
        if self.requires_zeroed_output_arguments:
            mapping = {WRITE: INC}
        else:
            mapping = {}
        return list(mapping.get(a, a) for a in self.argument_accesses)

    @property
    def kernel_args(self):
        return tuple(p.kernel_arg(self.loop_indices) for p in self.packed_args)

    @property
    def wrapper_args(self):
        # Loop extents come from here.
        args = list(self.forward_arguments)
        args.extend(self._loop_index.extents)
        if self.extruded:
            args.append(self._layers_array)
        if self.subset:
            args.append(self._subset_indices)
        # parloop args passed "as is"
        args.extend(self.arguments)
        # maps are refcounted
        for map_ in self.maps.values():
            # But we don't need to emit stuff for PMaps because they
            # are a Map (already seen + a permutation [encoded in the
            # indexing]).
            if not isinstance(map_, PMap):
                args.append(map_.values)
        return tuple(args)

    @property
    def domains(self):
        domains = f"{{[{self.loop_iname}]: start<={self.loop_iname}<end}}",
        domains += tuple(arg.domain for arg in self.packed_args)
        return domains

    @property
    def kernel_data(self):
        # ordering matters
        kernel_data = (lp.ValueArg("start", IntType, is_input=True),
                        lp.ValueArg("end", IntType, is_input=True))
        kernel_data += tuple(*(a.kernel_data for a in self.packed_args))
        return kernel_data

    def kernel_call(self):
        refs = []
        for arg in self.packed_args:
            insn, = arg.pack_instructions
            if isinstance(insn.assignee, pym.Subscript):
                swept_inames = var(arg.iname),  # this is wrong
                swept_inames = ()
                refs.append(lp.symbolic.SubArrayRef(swept_inames, insn.assignee))
            else:
                refs.append(insn.assignee)

        refs = tuple(refs)
        self.kernel_parameters[self.kernel.name] = refs

        depends_on = frozenset(insn.id for insn in self._pack_instructions)
        expr = pym.Call(pym.Variable(self.kernel.name), ())
        return lp.CallInstruction(refs, expr,
                                  within_inames=frozenset([self.loop_iname]),
                                  depends_on=depends_on, id="MYFUNCTIONID", within_inames_is_final=True)
        # args = self.kernel_args
        # access = tuple(self.loopy_argument_accesses)
        # # assuming every index is free index
        # free_indices = set(itertools.chain.from_iterable(arg.multiindex for arg in args))
        # # remove runtime index
        # free_indices = tuple(i for i in free_indices if isinstance(i, Index))
        # if self.pass_layer_to_kernel:
        #     args = args + (self.layer_index, )
        #     access = access + (READ,)
        # if self.forward_arguments:
        #     args = self.forward_arguments + args
        #     access = tuple([WRITE] * len(self.forward_arguments)) + access
        # return FunctionCall(self.kernel.name, KernelInst(), access, free_indices, *args)

    @property
    def _pack_instructions(self):
        return tuple(itertools.chain(*(pack.pack_instructions
                            for pack in self.packed_args)))

    def emit_instructions(self):
        yield from self._pack_instructions
        # Sometimes, actual instructions do not refer to all the loop
        # indices (e.g. all of them are globals). To ensure that loopy
        # knows about these indices, we emit a dummy instruction (that
        # doesn't generate any code) that does depend on them.
        # yield DummyInstruction(PackInst(), *(x for x in self.loop_indices if x is not None))
        kernel_call = self.kernel_call()
        yield kernel_call
        yield from itertools.chain(*(pack.emit_unpack_instruction(depends_on=frozenset({kernel_call.id}))
                                     for pack in self.packed_args))
