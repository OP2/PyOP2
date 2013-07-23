# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""
User API Unit Tests
"""

import pytest
import numpy as np
from mpi4py import MPI

from pyop2 import op2
from pyop2 import exceptions
from pyop2 import sequential
from pyop2 import base
from pyop2 import configuration as cfg


@pytest.fixture
def set():
    return op2.Set(5, 'foo')


@pytest.fixture
def iterset():
    return op2.Set(2, 'iterset')


@pytest.fixture
def toset():
    return op2.Set(3, 'toset')


@pytest.fixture(params=[1, 2, (2, 3)])
def dset(request, set):
    return op2.DataSet(set, request.param, 'dfoo')


@pytest.fixture
def diterset(iterset):
    return op2.DataSet(iterset, 1, 'diterset')


@pytest.fixture
def dtoset(toset):
    return op2.DataSet(toset, 1, 'dtoset')


@pytest.fixture
def m(iterset, toset):
    return op2.Map(iterset, toset, 2, [1] * 2 * iterset.size, 'm')


@pytest.fixture
def const(request):
    c = op2.Const(1, 1, 'test_const_nonunique_name')
    request.addfinalizer(c.remove_from_namespace)
    return c


@pytest.fixture
def sparsity(m, dtoset):
    return op2.Sparsity((dtoset, dtoset), (m, m))


class TestInitAPI:

    """
    Init API unit tests
    """

    def test_noninit(self):
        "RuntimeError should be raised when using op2 before calling init."
        with pytest.raises(RuntimeError):
            op2.Set(1)

    def test_invalid_init(self):
        "init should not accept an invalid backend."
        with pytest.raises(ImportError):
            op2.init(backend='invalid_backend')

    def test_init(self, backend):
        "init should correctly set the backend."
        assert op2.backends.get_backend() == 'pyop2.' + backend

    def test_double_init(self, backend):
        "Calling init again with the same backend should update the configuration."
        op2.init(backend=backend, foo='bar')
        assert op2.backends.get_backend() == 'pyop2.' + backend
        assert cfg.foo == 'bar'

    def test_change_backend_fails(self, backend):
        "Calling init again with a different backend should fail."
        with pytest.raises(RuntimeError):
            op2.init(backend='other')


class TestMPIAPI:

    """
    Init API unit tests
    """

    def test_running_sequentially(self, backend):
        "MPI.parallel should return false if running sequentially."
        assert not op2.MPI.parallel

    def test_set_mpi_comm_int(self, backend):
        "int should be converted to mpi4py MPI communicator."
        oldcomm = op2.MPI.comm
        op2.MPI.comm = 1
        assert isinstance(op2.MPI.comm, MPI.Comm)
        op2.MPI.comm = oldcomm

    def test_set_mpi_comm_mpi4py(self, backend):
        "Setting an mpi4py MPI communicator should be allowed."
        oldcomm = op2.MPI.comm
        op2.MPI.comm = MPI.COMM_SELF
        assert isinstance(op2.MPI.comm, MPI.Comm)
        op2.MPI.comm = oldcomm

    def test_set_mpi_comm_invalid_type(self, backend):
        "Invalid MPI communicator type should raise TypeError."
        with pytest.raises(TypeError):
            op2.MPI.comm = None


class TestAccessAPI:

    """
    Access API unit tests
    """

    @pytest.mark.parametrize("mode", base.Access._modes)
    def test_access_repr(self, backend, mode):
        "Access repr should produce an Access object when eval'd."
        from pyop2.base import Access
        assert isinstance(eval(repr(Access(mode))), Access)

    @pytest.mark.parametrize("mode", base.Access._modes)
    def test_access_str(self, backend, mode):
        "Access should have the expected string representation."
        assert str(base.Access(mode)) == "OP2 Access: %s" % mode

    def test_illegal_access(self, backend):
        "Illegal access modes should raise an exception."
        with pytest.raises(exceptions.ModeValueError):
            base.Access('ILLEGAL_ACCESS')


class TestSetAPI:

    """
    Set API unit tests
    """

    def test_set_illegal_size(self, backend):
        "Set size should be int."
        with pytest.raises(exceptions.SizeTypeError):
            op2.Set('illegalsize')

    def test_set_illegal_name(self, backend):
        "Set name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Set(1, 2)

    def test_set_repr(self, backend, set):
        "Set repr should produce a Set object when eval'd."
        from pyop2.op2 import Set  # noqa: needed by eval
        assert isinstance(eval(repr(set)), base.Set)

    def test_set_str(self, backend, set):
        "Set should have the expected string representation."
        assert str(set) == "OP2 Set: %s with size %s" % (set.name, set.size)

    def test_set_equality(self, backend, set):
        "The equality test for sets is identity, not attribute equality"
        setcopy = op2.Set(set.size, set.name)
        assert set == set and set != setcopy

    def test_dset_in_set(self, backend, set, dset):
        "The in operator should indicate compatibility of DataSet and Set"
        assert dset in set

    def test_dset_not_in_set(self, backend, dset):
        "The in operator should indicate incompatibility of DataSet and Set"
        assert dset not in op2.Set(5, 'bar')


class TestDataSetAPI:
    """
    DataSet API unit tests
    """

    def test_dset_illegal_set(self, backend):
        "Set should be Set."
        with pytest.raises(exceptions.SetTypeError):
            op2.DataSet('illegalsize', 1)

    def test_dset_illegal_dim(self, iterset, backend):
        "Set dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.DataSet(iterset, 'illegaldim')

    def test_dset_illegal_dim_tuple(self, iterset, backend):
        "Set dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.DataSet(iterset, (1, 'illegaldim'))

    def test_dset_illegal_name(self, iterset, backend):
        "Set name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.DataSet(iterset, 1, 2)

    def test_dset_default_dim(self, iterset, backend):
        "Set constructor should default dim to (1,)."
        assert op2.DataSet(iterset).dim == (1,)

    def test_dset_dim(self, iterset, backend):
        "Set constructor should create a dim tuple."
        s = op2.DataSet(iterset, 1)
        assert s.dim == (1,)

    def test_dset_dim_list(self, iterset, backend):
        "Set constructor should create a dim tuple from a list."
        s = op2.DataSet(iterset, [2, 3])
        assert s.dim == (2, 3)

    def test_dset_repr(self, backend, dset):
        "DataSet repr should produce a Set object when eval'd."
        from pyop2.op2 import Set, DataSet  # noqa: needed by eval
        assert isinstance(eval(repr(dset)), base.DataSet)

    def test_dset_str(self, backend, dset):
        "DataSet should have the expected string representation."
        assert str(dset) == "OP2 DataSet: %s on set %s, with dim %s" \
            % (dset.name, dset.set, dset.dim)

    def test_dset_equality(self, backend, dset):
        "The equality test for data sets is same dim and same set"
        setcopy = op2.DataSet(dset.set, dset.dim, dset.name)
        assert setcopy.set == dset.set and setcopy.dim == dset.dim

    def test_dat_in_dset(self, backend, dset):
        "The in operator should indicate compatibility of DataSet and Set"
        assert op2.Dat(dset) in dset

    def test_dat_not_in_dset(self, backend, dset):
        "The in operator should indicate incompatibility of DataSet and Set"
        assert op2.Dat(dset) not in op2.DataSet(op2.Set(5, 'bar'))


class TestDatAPI:

    """
    Dat API unit tests
    """

    def test_dat_illegal_set(self, backend):
        "Dat set should be DataSet."
        with pytest.raises(exceptions.DataSetTypeError):
            op2.Dat('illegalset', 1)

    def test_dat_illegal_name(self, backend, dset):
        "Dat name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Dat(dset, name=2)

    def test_dat_initialise_data(self, backend, dset):
        """Dat initilialised without the data should initialise data with the
        correct size and type."""
        d = op2.Dat(dset)
        assert d.data.size == dset.size * \
            np.prod(dset.dim) and d.data.dtype == np.float64

    def test_dat_initialise_data_type(self, backend, dset):
        """Dat intiialised without the data but with specified type should
        initialise its data with the correct type."""
        d = op2.Dat(dset, dtype=np.int32)
        assert d.data.dtype == np.int32

    def test_dat_illegal_map(self, backend, dset):
        """Dat __call__ should not allow a map with a toset other than this
        Dat's set."""
        d = op2.Dat(dset)
        set1 = op2.Set(3)
        set2 = op2.Set(2)
        to_set2 = op2.Map(set1, set2, 1, [0, 0, 0])
        with pytest.raises(exceptions.MapValueError):
            d(to_set2, op2.READ)

    def test_dat_dtype(self, backend, dset):
        "Default data type should be numpy.float64."
        d = op2.Dat(dset)
        assert d.dtype == np.double

    def test_dat_float(self, backend, dset):
        "Data type for float data should be numpy.float64."
        d = op2.Dat(dset, [1.0] * dset.size * np.prod(dset.dim))
        assert d.dtype == np.double

    def test_dat_int(self, backend, dset):
        "Data type for int data should be numpy.int."
        d = op2.Dat(dset, [1] * dset.size * np.prod(dset.dim))
        assert d.dtype == np.int

    def test_dat_convert_int_float(self, backend, dset):
        "Explicit float type should override NumPy's default choice of int."
        d = op2.Dat(dset, [1] * dset.size * np.prod(dset.dim), np.double)
        assert d.dtype == np.float64

    def test_dat_convert_float_int(self, backend, dset):
        "Explicit int type should override NumPy's default choice of float."
        d = op2.Dat(dset, [1.5] * dset.size * np.prod(dset.dim), np.int32)
        assert d.dtype == np.int32

    def test_dat_illegal_dtype(self, backend, dset):
        "Illegal data type should raise DataTypeError."
        with pytest.raises(exceptions.DataTypeError):
            op2.Dat(dset, dtype='illegal_type')

    def test_dat_illegal_length(self, backend, dset):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Dat(dset, [1] * (dset.size * np.prod(dset.dim) + 1))

    def test_dat_reshape(self, backend, dset):
        "Data should be reshaped according to the set's dim."
        d = op2.Dat(dset, [1.0] * dset.size * np.prod(dset.dim))
        assert d.data.shape == (dset.size,) + dset.dim

    def test_dat_properties(self, backend, dset):
        "Dat constructor should correctly set attributes."
        d = op2.Dat(dset, [1] * dset.size * np.prod(dset.dim), 'double', 'bar')
        assert d.dataset.set == dset.set and d.dtype == np.float64 and \
            d.name == 'bar' and d.data.sum() == dset.size * np.prod(dset.dim)

    def test_dat_repr(self, backend, dset):
        "Dat repr should produce a Dat object when eval'd."
        from pyop2.op2 import Dat, DataSet, Set  # noqa: needed by eval
        from numpy import dtype  # noqa: needed by eval
        d = op2.Dat(dset, dtype='double', name='bar')
        assert isinstance(eval(repr(d)), base.Dat)

    def test_dat_str(self, backend, dset):
        "Dat should have the expected string representation."
        d = op2.Dat(dset, dtype='double', name='bar')
        s = "OP2 Dat: %s on (%s) with datatype %s" \
            % (d.name, d.dataset, d.data.dtype.name)
        assert str(d) == s

    def test_dat_ro_accessor(self, backend, dset):
        "Attempting to set values through the RO accessor should raise an error."
        d = op2.Dat(dset, range(np.prod(dset.dim) * dset.size), dtype=np.int32)
        x = d.data_ro
        with pytest.raises((RuntimeError, ValueError)):
            x[0] = 1

    def test_dat_ro_write_accessor(self, backend, dset):
        "Re-accessing the data in writeable form should be allowed."
        d = op2.Dat(dset, range(np.prod(dset.dim) * dset.size), dtype=np.int32)
        x = d.data_ro
        with pytest.raises((RuntimeError, ValueError)):
            x[0] = 1
        x = d.data
        x[0] = -100
        assert (d.data_ro[0] == -100).all()


class TestSparsityAPI:

    """
    Sparsity API unit tests
    """

    @pytest.fixture
    def mi(cls, toset):
        iterset = op2.Set(3, 'iterset2')
        return op2.Map(iterset, toset, 1, [1] * iterset.size, 'mi')

    @pytest.fixture
    def dataset2(cls):
        return op2.Set(1, 'dataset2')

    @pytest.fixture
    def md(cls, iterset, dataset2):
        return op2.Map(iterset, dataset2, 1, [1] * iterset.size, 'md')

    @pytest.fixture
    def di(cls, toset):
        return op2.DataSet(toset, 1, 'di')

    @pytest.fixture
    def dd(cls, dataset2):
        return op2.DataSet(dataset2, 1, 'dd')

    def test_sparsity_illegal_rdset(self, backend, di, mi):
        "Sparsity rdset should be a DataSet"
        with pytest.raises(TypeError):
            op2.Sparsity(('illegalrmap', di), (mi, mi))

    def test_sparsity_illegal_cdset(self, backend, di, mi):
        "Sparsity cdset should be a DataSet"
        with pytest.raises(TypeError):
            op2.Sparsity((di, 'illegalrmap'), (mi, mi))

    def test_sparsity_illegal_rmap(self, backend, di, mi):
        "Sparsity rmap should be a Map"
        with pytest.raises(TypeError):
            op2.Sparsity((di, di), ('illegalrmap', mi))

    def test_sparsity_illegal_cmap(self, backend, di, mi):
        "Sparsity cmap should be a Map"
        with pytest.raises(TypeError):
            op2.Sparsity((di, di), (mi, 'illegalcmap'))

    def test_sparsity_single_dset(self, backend, di, mi):
        "Sparsity constructor should accept single Map and turn it into tuple"
        s = op2.Sparsity(di, mi, "foo")
        assert s.maps[0] == (mi, mi) and s.dims == (1, 1) and s.name == "foo" and s.dsets == (di, di)

    def test_sparsity_map_pair(self, backend, di, mi):
        "Sparsity constructor should accept a pair of maps"
        s = op2.Sparsity((di, di), (mi, mi), "foo")
        assert s.maps[0] == (mi, mi) and s.dims == (1, 1) and s.name == "foo" and s.dsets == (di, di)

    def test_sparsity_map_pair_different_dataset(self, backend, mi, md, di, dd, m):
        "Sparsity constructor should accept a pair of maps"
        s = op2.Sparsity((di, dd), (m, md), "foo")
        assert s.maps[0] == (m, md) and s.dims == (1, 1) and s.name == "foo" and s.dsets == (di, dd)

    def test_sparsity_multiple_map_pairs(self, backend, mi, di):
        "Sparsity constructor should accept tuple of pairs of maps"
        s = op2.Sparsity((di, di), ((mi, mi), (mi, mi)), "foo")
        assert s.maps == [(mi, mi), (mi, mi)] and s.dims == (1, 1)

    def test_sparsity_map_pairs_different_itset(self, backend, mi, di, dd, m):
        "Sparsity constructor should accept maps with different iteration sets"
        s = op2.Sparsity((di, di), ((m, m), (mi, mi)), "foo")
        # Note the order of the map pairs is not guaranteed
        assert len(s.maps) == 2 and s.dims == (1, 1)

    def test_sparsity_illegal_itersets(self, backend, mi, md, di, dd):
        "Both maps in a (rmap,cmap) tuple must have same iteration set"
        with pytest.raises(RuntimeError):
            op2.Sparsity((dd, di), (md, mi))

    def test_sparsity_illegal_row_datasets(self, backend, mi, md, di):
        "All row maps must share the same data set"
        with pytest.raises(RuntimeError):
            op2.Sparsity((di, di), ((mi, mi), (md, mi)))

    def test_sparsity_illegal_col_datasets(self, backend, mi, md, di, dd):
        "All column maps must share the same data set"
        with pytest.raises(RuntimeError):
            op2.Sparsity((di, di), ((mi, mi), (mi, md)))

    def test_sparsity_repr(self, backend, sparsity):
        "Sparsity should have the expected repr."

        # Note: We can't actually reproduce a Sparsity from its repr because
        # the Sparsity constructor checks that the maps are populated
        r = "Sparsity(%r, %r, %r)" % (sparsity.dsets, sparsity.maps, sparsity.name)
        assert repr(sparsity) == r

    def test_sparsity_str(self, backend, sparsity):
        "Sparsity should have the expected string representation."
        s = "OP2 Sparsity: dsets %s, rmaps %s, cmaps %s, name %s" % \
            (sparsity.dsets, sparsity.rmaps, sparsity.cmaps, sparsity.name)
        assert str(sparsity) == s


class TestMatAPI:

    """
    Mat API unit tests
    """

    def test_mat_illegal_sets(self, backend):
        "Mat sparsity should be a Sparsity."
        with pytest.raises(TypeError):
            op2.Mat('illegalsparsity')

    def test_mat_illegal_name(self, backend, sparsity):
        "Mat name should be string."
        with pytest.raises(sequential.NameTypeError):
            op2.Mat(sparsity, name=2)

    def test_mat_dtype(self, backend, sparsity):
        "Default data type should be numpy.float64."
        m = op2.Mat(sparsity)
        assert m.dtype == np.double

    def test_mat_properties(self, backend, sparsity):
        "Mat constructor should correctly set attributes."
        m = op2.Mat(sparsity, 'double', 'bar')
        assert m.sparsity == sparsity and  \
            m.dtype == np.float64 and m.name == 'bar'

    def test_mat_illegal_maps(self, backend, sparsity):
        m = op2.Mat(sparsity)
        set1 = op2.Set(2)
        set2 = op2.Set(3)
        wrongmap = op2.Map(set1, set2, 2, [0, 0, 0, 0])
        with pytest.raises(exceptions.MapValueError):
            m((wrongmap[0], wrongmap[1]), op2.INC)

    def test_mat_repr(self, backend, sparsity):
        "Mat should have the expected repr."

        # Note: We can't actually reproduce a Sparsity from its repr because
        # the Sparsity constructor checks that the maps are populated
        m = op2.Mat(sparsity)
        r = "Mat(%r, %r, %r)" % (m.sparsity, m.dtype, m.name)
        assert repr(m) == r

    def test_mat_str(self, backend, sparsity):
        "Mat should have the expected string representation."
        m = op2.Mat(sparsity)
        s = "OP2 Mat: %s, sparsity (%s), datatype %s" \
            % (m.name, m.sparsity, m.dtype.name)
        assert str(m) == s


class TestConstAPI:

    """
    Const API unit tests
    """

    def test_const_illegal_dim(self, backend):
        "Const dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Const('illegaldim', 1, 'test_const_illegal_dim')

    def test_const_illegal_dim_tuple(self, backend):
        "Const dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Const((1, 'illegaldim'), 1, 'test_const_illegal_dim_tuple')

    def test_const_nonunique_name(self, backend, const):
        "Const names should be unique."
        with pytest.raises(op2.Const.NonUniqueNameError):
            op2.Const(1, 1, 'test_const_nonunique_name')

    def test_const_remove_from_namespace(self, backend):
        "remove_from_namespace should free a global name."
        c = op2.Const(1, 1, 'test_const_remove_from_namespace')
        c.remove_from_namespace()
        c = op2.Const(1, 1, 'test_const_remove_from_namespace')
        c.remove_from_namespace()
        assert c.name == 'test_const_remove_from_namespace'

    def test_const_illegal_name(self, backend):
        "Const name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Const(1, 1, 2)

    def test_const_dim(self, backend):
        "Const constructor should create a dim tuple."
        c = op2.Const(1, 1, 'test_const_dim')
        c.remove_from_namespace()
        assert c.dim == (1,)

    def test_const_dim_list(self, backend):
        "Const constructor should create a dim tuple from a list."
        c = op2.Const([2, 3], [1] * 6, 'test_const_dim_list')
        c.remove_from_namespace()
        assert c.dim == (2, 3)

    def test_const_float(self, backend):
        "Data type for float data should be numpy.float64."
        c = op2.Const(1, 1.0, 'test_const_float')
        c.remove_from_namespace()
        assert c.dtype == np.double

    def test_const_int(self, backend):
        "Data type for int data should be numpy.int."
        c = op2.Const(1, 1, 'test_const_int')
        c.remove_from_namespace()
        assert c.dtype == np.int

    def test_const_convert_int_float(self, backend):
        "Explicit float type should override NumPy's default choice of int."
        c = op2.Const(1, 1, 'test_const_convert_int_float', 'double')
        c.remove_from_namespace()
        assert c.dtype == np.float64

    def test_const_convert_float_int(self, backend):
        "Explicit int type should override NumPy's default choice of float."
        c = op2.Const(1, 1.5, 'test_const_convert_float_int', 'int')
        c.remove_from_namespace()
        assert c.dtype == np.int

    def test_const_illegal_dtype(self, backend):
        "Illegal data type should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Const(1, 'illegal_type', 'test_const_illegal_dtype', 'double')

    @pytest.mark.parametrize("dim", [1, (2, 2)])
    def test_const_illegal_length(self, backend, dim):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Const(
                dim, [1] * (np.prod(dim) + 1), 'test_const_illegal_length_%r' % np.prod(dim))

    def test_const_reshape(self, backend):
        "Data should be reshaped according to dim."
        c = op2.Const((2, 2), [1.0] * 4, 'test_const_reshape')
        c.remove_from_namespace()
        assert c.dim == (2, 2) and c.data.shape == (2, 2)

    def test_const_properties(self, backend):
        "Data constructor should correctly set attributes."
        c = op2.Const((2, 2), [1] * 4, 'baz', 'double')
        c.remove_from_namespace()
        assert c.dim == (2, 2) and c.dtype == np.float64 and c.name == 'baz' \
            and c.data.sum() == 4

    def test_const_setter(self, backend):
        "Setter attribute on data should correct set data value."
        c = op2.Const(1, 1, 'c')
        c.remove_from_namespace()
        c.data = 2
        assert c.data.sum() == 2

    def test_const_setter_malformed_data(self, backend):
        "Setter attribute should reject malformed data."
        c = op2.Const(1, 1, 'c')
        c.remove_from_namespace()
        with pytest.raises(exceptions.DataValueError):
            c.data = [1, 2]

    def test_const_repr(self, backend, const):
        "Const repr should produce a Const object when eval'd."
        from pyop2.op2 import Const  # noqa: needed by eval
        from numpy import array  # noqa: needed by eval
        const.remove_from_namespace()
        c = eval(repr(const))
        assert isinstance(c, base.Const)
        c.remove_from_namespace()

    def test_const_str(self, backend, const):
        "Const should have the expected string representation."
        s = "OP2 Const: %s of dim %s and type %s with value %s" \
            % (const.name, const.dim, const.data.dtype.name, const.data)
        assert str(const) == s


class TestGlobalAPI:

    """
    Global API unit tests
    """

    def test_global_illegal_dim(self, backend):
        "Global dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Global('illegaldim')

    def test_global_illegal_dim_tuple(self, backend):
        "Global dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Global((1, 'illegaldim'))

    def test_global_illegal_name(self, backend):
        "Global name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Global(1, 1, name=2)

    def test_global_dim(self, backend):
        "Global constructor should create a dim tuple."
        g = op2.Global(1, 1)
        assert g.dim == (1,)

    def test_global_dim_list(self, backend):
        "Global constructor should create a dim tuple from a list."
        g = op2.Global([2, 3], [1] * 6)
        assert g.dim == (2, 3)

    def test_global_float(self, backend):
        "Data type for float data should be numpy.float64."
        g = op2.Global(1, 1.0)
        assert g.dtype == np.double

    def test_global_int(self, backend):
        "Data type for int data should be numpy.int."
        g = op2.Global(1, 1)
        assert g.dtype == np.int

    def test_global_convert_int_float(self, backend):
        "Explicit float type should override NumPy's default choice of int."
        g = op2.Global(1, 1, 'double')
        assert g.dtype == np.float64

    def test_global_convert_float_int(self, backend):
        "Explicit int type should override NumPy's default choice of float."
        g = op2.Global(1, 1.5, 'int')
        assert g.dtype == np.int

    def test_global_illegal_dtype(self, backend):
        "Illegal data type should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Global(1, 'illegal_type', 'double')

    @pytest.mark.parametrize("dim", [1, (2, 2)])
    def test_global_illegal_length(self, backend, dim):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Global(dim, [1] * (np.prod(dim) + 1))

    def test_global_reshape(self, backend):
        "Data should be reshaped according to dim."
        g = op2.Global((2, 2), [1.0] * 4)
        assert g.dim == (2, 2) and g.data.shape == (2, 2)

    def test_global_properties(self, backend):
        "Data globalructor should correctly set attributes."
        g = op2.Global((2, 2), [1] * 4, 'double', 'bar')
        assert g.dim == (2, 2) and g.dtype == np.float64 and g.name == 'bar' \
            and g.data.sum() == 4

    def test_global_setter(self, backend):
        "Setter attribute on data should correct set data value."
        c = op2.Global(1, 1)
        c.data = 2
        assert c.data.sum() == 2

    def test_global_setter_malformed_data(self, backend):
        "Setter attribute should reject malformed data."
        c = op2.Global(1, 1)
        with pytest.raises(exceptions.DataValueError):
            c.data = [1, 2]

    def test_global_repr(self, backend):
        "Global repr should produce a Global object when eval'd."
        from pyop2.op2 import Global  # noqa: needed by eval
        from numpy import array, dtype  # noqa: needed by eval
        g = op2.Global(1, 1, 'double')
        assert isinstance(eval(repr(g)), base.Global)

    def test_global_str(self, backend):
        "Global should have the expected string representation."
        g = op2.Global(1, 1, 'double')
        s = "OP2 Global Argument: %s with dim %s and value %s" \
            % (g.name, g.dim, g.data)
        assert str(g) == s


class TestMapAPI:

    """
    Map API unit tests
    """

    def test_map_illegal_iterset(self, backend, set):
        "Map iterset should be Set."
        with pytest.raises(exceptions.SetTypeError):
            op2.Map('illegalset', set, 1, [])

    def test_map_illegal_toset(self, backend, set):
        "Map toset should be Set."
        with pytest.raises(exceptions.SetTypeError):
            op2.Map(set, 'illegalset', 1, [])

    def test_map_illegal_arity(self, backend, set):
        "Map arity should be int."
        with pytest.raises(exceptions.ArityTypeError):
            op2.Map(set, set, 'illegalarity', [])

    def test_map_illegal_arity_tuple(self, backend, set):
        "Map arity should not be a tuple."
        with pytest.raises(exceptions.ArityTypeError):
            op2.Map(set, set, (2, 2), [])

    def test_map_illegal_name(self, backend, set):
        "Map name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Map(set, set, 1, [], name=2)

    def test_map_illegal_dtype(self, backend, set):
        "Illegal data type should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Map(set, set, 1, 'abcdefg')

    def test_map_illegal_length(self, backend, iterset, toset):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Map(iterset, toset, 1, [1] * (iterset.size + 1))

    def test_map_convert_float_int(self, backend, iterset, toset):
        "Float data should be implicitely converted to int."
        m = op2.Map(iterset, toset, 1, [1.5] * iterset.size)
        assert m.values.dtype == np.int32 and m.values.sum() == iterset.size

    def test_map_reshape(self, backend, iterset, toset):
        "Data should be reshaped according to arity."
        m = op2.Map(iterset, toset, 2, [1] * 2 * iterset.size)
        assert m.arity == 2 and m.values.shape == (iterset.size, 2)

    def test_map_properties(self, backend, iterset, toset):
        "Data constructor should correctly set attributes."
        m = op2.Map(iterset, toset, 2, [1] * 2 * iterset.size, 'bar')
        assert m.iterset == iterset and m.toset == toset and m.arity == 2 \
            and m.values.sum() == 2 * iterset.size and m.name == 'bar'

    def test_map_indexing(self, backend, iterset, toset):
        "Indexing a map should create an appropriate Arg"
        m = op2.Map(iterset, toset, 2, [1] * 2 * iterset.size, 'm')

        arg = m[0]
        assert arg.idx == 0

    def test_map_slicing(self, backend, iterset, toset):
        "Slicing a map is not allowed"
        m = op2.Map(iterset, toset, 2, [1] * 2 * iterset.size, 'm')

        with pytest.raises(NotImplementedError):
            m[:]

    def test_map_equality(self, backend, m):
        """A map is equal if all its attributes are equal, bearing in mind that
        equality is identity for sets."""
        m2 = op2.Map(m.iterset, m.toset, m.arity, m.values, m.name)
        assert m == m2

    def test_map_copied_set_inequality(self, backend, m):
        """Maps that have copied but not equal iteration sets are not equal"""
        itercopy = op2.Set(m.iterset.size, m.iterset.name)
        m2 = op2.Map(itercopy, m.toset, m.arity, m.values, m.name)
        assert m != m2

    def test_map_arity_inequality(self, backend, m):
        """Maps that have different arities are not equal"""
        m2 = op2.Map(m.iterset, m.toset,
                     m.arity * 2, list(m.values) * 2, m.name)
        assert m != m2

    def test_map_name_inequality(self, backend, m):
        """Maps with different names are not equal"""
        n = op2.Map(m.iterset, m.toset, m.arity, m.values, 'n')
        assert m != n

    def test_map_repr(self, backend, m):
        "Map should have the expected repr."
        r = "Map(%r, %r, %r, None, %r)" % (m.iterset, m.toset, m.arity, m.name)
        assert repr(m) == r

    def test_map_str(self, backend, m):
        "Map should have the expected string representation."
        s = "OP2 Map: %s from (%s) to (%s) with arity %s" \
            % (m.name, m.iterset, m.toset, m.arity)
        assert str(m) == s


class TestIterationSpaceAPI:

    """
    IterationSpace API unit tests
    """

    def test_iteration_space_illegal_iterset(self, backend, set):
        "IterationSpace iterset should be Set."
        with pytest.raises(exceptions.SetTypeError):
            op2.IterationSpace('illegalset', 1)

    def test_iteration_space_illegal_extents(self, backend, set):
        "IterationSpace extents should be int or int tuple."
        with pytest.raises(TypeError):
            op2.IterationSpace(set, 'illegalextents')

    def test_iteration_space_illegal_extents_tuple(self, backend, set):
        "IterationSpace extents should be int or int tuple."
        with pytest.raises(TypeError):
            op2.IterationSpace(set, (1, 'illegalextents'))

    def test_iteration_space_extents(self, backend, set):
        "IterationSpace constructor should create a extents tuple."
        m = op2.IterationSpace(set, 1)
        assert m.extents == (1,)

    def test_iteration_space_extents_list(self, backend, set):
        "IterationSpace constructor should create a extents tuple from a list."
        m = op2.IterationSpace(set, [2, 3])
        assert m.extents == (2, 3)

    def test_iteration_space_properties(self, backend, set):
        "IterationSpace constructor should correctly set attributes."
        i = op2.IterationSpace(set, (2, 3))
        assert i.iterset == set and i.extents == (2, 3)

    def test_iteration_space_repr(self, backend, set):
        """IterationSpace repr should produce a IterationSpace object when
        eval'd."""
        from pyop2.op2 import Set, IterationSpace  # noqa: needed by eval
        m = op2.IterationSpace(set, 1)
        assert isinstance(eval(repr(m)), base.IterationSpace)

    def test_iteration_space_str(self, backend, set):
        "IterationSpace should have the expected string representation."
        m = op2.IterationSpace(set, 1)
        s = "OP2 Iteration Space: %s with extents %s" % (m.iterset, m.extents)
        assert str(m) == s


class TestKernelAPI:

    """
    Kernel API unit tests
    """

    def test_kernel_illegal_name(self, backend):
        "Kernel name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Kernel("", name=2)

    def test_kernel_properties(self, backend):
        "Kernel constructor should correctly set attributes."
        k = op2.Kernel("", 'foo')
        assert k.name == 'foo'

    def test_kernel_repr(self, backend, set):
        "Kernel should have the expected repr."
        k = op2.Kernel("int foo() { return 0; }", 'foo')
        assert repr(k) == 'Kernel("""%s""", %r)' % (k.code, k.name)

    def test_kernel_str(self, backend, set):
        "Kernel should have the expected string representation."
        k = op2.Kernel("int foo() { return 0; }", 'foo')
        assert str(k) == "OP2 Kernel: %s" % k.name


class TestIllegalItersetMaps:

    """
    Pass args with the wrong iterset maps to ParLoops, and check that they are trapped.
    """

    def test_illegal_dat_iterset(self, backend):
        set1 = op2.Set(2)
        set2 = op2.Set(3)
        dset1 = op2.DataSet(set1, 1)
        dat = op2.Dat(dset1)
        map = op2.Map(set2, set1, 1, [0, 0, 0])
        kernel = op2.Kernel("void k() { }", "k")
        with pytest.raises(exceptions.MapValueError):
            base.ParLoop(kernel, set1, dat(map, op2.READ))

    def test_illegal_mat_iterset(self, backend, sparsity):
        set1 = op2.Set(2)
        m = op2.Mat(sparsity)
        rmap, cmap = sparsity.maps[0]
        kernel = op2.Kernel("void k() { }", "k")
        with pytest.raises(exceptions.MapValueError):
            base.ParLoop(kernel, set1(3, 3),
                         m((rmap[op2.i[0]], cmap[op2.i[1]]), op2.INC))


class TestSolverAPI:

    """
    Test the Solver API.
    """

    def test_solver_defaults(self, backend):
        s = op2.Solver()
        assert s.parameters == base.DEFAULT_SOLVER_PARAMETERS

    def test_set_options_with_params(self, backend):
        params = {'linear_solver': 'gmres',
                  'maximum_iterations': 25}
        s = op2.Solver(params)
        assert s.parameters['linear_solver'] == 'gmres' \
            and s.parameters['maximum_iterations'] == 25

    def test_set_options_with_kwargs(self, backend):
        s = op2.Solver(linear_solver='gmres', maximum_iterations=25)
        assert s.parameters['linear_solver'] == 'gmres' \
            and s.parameters['maximum_iterations'] == 25

    def test_update_parameters(self, backend):
        s = op2.Solver()
        params = {'linear_solver': 'gmres',
                  'maximum_iterations': 25}
        s.update_parameters(params)
        assert s.parameters['linear_solver'] == 'gmres' \
            and s.parameters['maximum_iterations'] == 25

    def test_set_params_and_kwargs_illegal(self, backend):
        params = {'linear_solver': 'gmres',
                  'maximum_iterations': 25}
        with pytest.raises(RuntimeError):
            op2.Solver(params, linear_solver='cgs')

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
