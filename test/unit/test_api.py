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

@pytest.fixture(params=[1, 2, (2, 3)])
def set(request):
    return op2.Set(5, request.param, 'foo')

@pytest.fixture
def iterset():
    return op2.Set(2, 1, 'iterset')

@pytest.fixture
def dataset():
    return op2.Set(3, 1, 'dataset')

@pytest.fixture
def m(iterset, dataset):
    return op2.Map(iterset,  dataset, 2, [1] * 2 * iterset.size, 'm')

@pytest.fixture
def const(request):
    c = op2.Const(1, 1, 'test_const_nonunique_name')
    request.addfinalizer(c.remove_from_namespace)
    return c

@pytest.fixture
def sparsity(m):
    return op2.Sparsity((m, m))

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
        assert op2.backends.get_backend() == 'pyop2.'+backend

    def test_double_init(self, backend):
        "Calling init again with the same backend should update the configuration."
        op2.init(backend=backend, foo='bar')
        assert op2.backends.get_backend() == 'pyop2.'+backend
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

    def test_set_illegal_dim(self, backend):
        "Set dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Set(1, 'illegaldim')

    def test_set_illegal_dim_tuple(self, backend):
        "Set dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Set(1, (1,'illegaldim'))

    def test_set_illegal_name(self, backend):
        "Set name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Set(1, 1, 2)

    def test_set_dim(self, backend):
        "Set constructor should create a dim tuple."
        s = op2.Set(1, 1)
        assert s.dim == (1,)

    def test_set_dim_list(self, backend):
        "Set constructor should create a dim tuple from a list."
        s = op2.Set(1, [2,3])
        assert s.dim == (2,3)

    def test_set_repr(self, backend, set):
        "Set repr should produce a Set object when eval'd."
        from pyop2.op2 import Set
        assert isinstance(eval(repr(set)), base.Set)

    def test_set_str(self, backend, set):
        "Set should have the expected string representation."
        assert str(set) == "OP2 Set: %s with size %s, dim %s" \
                % (set.name, set.size, set.dim)

    def test_set_equality(self, backend, set):
        "The equality test for sets is identity, not attribute equality"
        setcopy = op2.Set(set.size, set.dim, set.name)
        assert set == set and set != setcopy

    # FIXME: test Set._lib_handle

class TestDatAPI:
    """
    Dat API unit tests
    """

    def test_dat_illegal_set(self, backend):
        "Dat set should be Set."
        with pytest.raises(exceptions.SetTypeError):
            op2.Dat('illegalset', 1)

    def test_dat_illegal_name(self, backend, set):
        "Dat name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Dat(set, name=2)

    def test_dat_initialise_data(self, backend, set):
        """Dat initilialised without the data should initialise data with the
        correct size and type."""
        d = op2.Dat(set)
        assert d.data.size == set.size * np.prod(set.dim) and d.data.dtype == np.float64

    def test_dat_initialise_data_type(self, backend, set):
        """Dat intiialised without the data but with specified type should
        initialise its data with the correct type."""
        d = op2.Dat(set, dtype=np.int32)
        assert d.data.dtype == np.int32

    def test_dat_illegal_map(self, backend, set):
        """Dat __call__ should not allow a map with a dataset other than this
        Dat's set."""
        d = op2.Dat(set)
        set1 = op2.Set(3)
        set2 = op2.Set(2)
        to_set2 = op2.Map(set1, set2, 1, [0, 0, 0])
        with pytest.raises(exceptions.MapValueError):
            d(to_set2, op2.READ)

    def test_dat_dtype(self, backend, set):
        "Default data type should be numpy.float64."
        d = op2.Dat(set)
        assert d.dtype == np.double

    def test_dat_float(self, backend, set):
        "Data type for float data should be numpy.float64."
        d = op2.Dat(set, [1.0] * set.size * np.prod(set.dim))
        assert d.dtype == np.double

    def test_dat_int(self, backend, set):
        "Data type for int data should be numpy.int."
        d = op2.Dat(set, [1]*set.size * np.prod(set.dim))
        assert d.dtype == np.int

    def test_dat_convert_int_float(self, backend, set):
        "Explicit float type should override NumPy's default choice of int."
        d = op2.Dat(set, [1]*set.size * np.prod(set.dim), np.double)
        assert d.dtype == np.float64

    def test_dat_convert_float_int(self, backend, set):
        "Explicit int type should override NumPy's default choice of float."
        d = op2.Dat(set, [1.5]*set.size * np.prod(set.dim), np.int32)
        assert d.dtype == np.int32

    def test_dat_illegal_dtype(self, backend, set):
        "Illegal data type should raise DataTypeError."
        with pytest.raises(exceptions.DataTypeError):
            op2.Dat(set, dtype='illegal_type')

    def test_dat_illegal_length(self, backend, set):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Dat(set, [1]*(set.size*np.prod(set.dim)+1))

    def test_dat_reshape(self, backend, set):
        "Data should be reshaped according to the set's dim."
        d = op2.Dat(set, [1.0]*set.size*np.prod(set.dim))
        assert d.data.shape == (set.size,) + set.dim

    def test_dat_properties(self, backend, set):
        "Dat constructor should correctly set attributes."
        d = op2.Dat(set, [1]*set.size*np.prod(set.dim), 'double', 'bar')
        assert d.dataset == set and d.dtype == np.float64 and \
                d.name == 'bar' and d.data.sum() == set.size*np.prod(set.dim)

    def test_dat_repr(self, backend, set):
        "Dat repr should produce a Dat object when eval'd."
        from pyop2.op2 import Dat, Set
        from numpy import dtype
        d = op2.Dat(set, dtype='double', name='bar')
        assert isinstance(eval(repr(d)), base.Dat)

    def test_dat_str(self, backend, set):
        "Dat should have the expected string representation."
        d = op2.Dat(set, dtype='double', name='bar')
        s = "OP2 Dat: %s on (%s) with datatype %s" \
               % (d.name, d.dataset, d.data.dtype.name)
        assert str(d) == s

    def test_dat_ro_accessor(self, backend, set):
        "Attempting to set values through the RO accessor should raise an error."
        d = op2.Dat(set, range(np.prod(set.dim) * set.size), dtype=np.int32)
        x = d.data_ro
        with pytest.raises((RuntimeError, ValueError)):
            x[0] = 1

    def test_dat_ro_write_accessor(self, backend, set):
        "Re-accessing the data in writeable form should be allowed."
        d = op2.Dat(set, range(np.prod(set.dim) * set.size), dtype=np.int32)
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
    def mi(cls, dataset):
        iterset = op2.Set(3, 1, 'iterset2')
        return op2.Map(iterset,  dataset, 1, [1] * iterset.size, 'mi')

    @pytest.fixture
    def md(cls, iterset):
        dataset = op2.Set(1, 1, 'dataset2')
        return op2.Map(iterset,  dataset, 1, [1] * iterset.size, 'md')

    def test_sparsity_illegal_rmap(self, backend, m):
        "Sparsity rmap should be a Map"
        with pytest.raises(TypeError):
            op2.Sparsity(('illegalrmap', m))

    def test_sparsity_illegal_cmap(self, backend, m):
        "Sparsity cmap should be a Map"
        with pytest.raises(TypeError):
            op2.Sparsity((m, 'illegalcmap'))

    def test_sparsity_single_map(self, backend, m):
        "Sparsity constructor should accept single Map and turn it into tuple"
        s = op2.Sparsity(m, "foo")
        assert s.maps[0] == (m, m) and s.dims == (1, 1) and s.name == "foo"

    def test_sparsity_map_pair(self, backend, m):
        "Sparsity constructor should accept a pair of maps"
        s = op2.Sparsity((m, m), "foo")
        assert s.maps[0] == (m, m) and s.dims == (1, 1) and s.name == "foo"

    def test_sparsity_map_pair_different_dataset(self, backend, m, md):
        "Sparsity constructor should accept a pair of maps"
        s = op2.Sparsity((m, md), "foo")
        assert s.maps[0] == (m, md) and s.dims == (1, 1) and s.name == "foo"

    def test_sparsity_multiple_map_pairs(self, backend, m):
        "Sparsity constructor should accept tuple of pairs of maps"
        s = op2.Sparsity(((m, m), (m, m)), "foo")
        assert s.maps == [(m, m), (m, m)] and s.dims == (1,1)

    def test_sparsity_map_pairs_different_itset(self, backend, m, mi):
        "Sparsity constructor should accept maps with different iteration sets"
        s = op2.Sparsity(((m, m), (mi, mi)), "foo")
        # Note the order of the map pairs is not guaranteed
        assert len(s.maps) == 2 and s.dims == (1,1)

    def test_sparsity_illegal_itersets(self, backend, m, mi):
        "Both maps in a (rmap,cmap) tuple must have same iteration set"
        with pytest.raises(RuntimeError):
            op2.Sparsity((m, mi))

    def test_sparsity_illegal_row_datasets(self, backend, m, md):
        "All row maps must share the same data set"
        with pytest.raises(RuntimeError):
            op2.Sparsity(((m, m), (md, m)))

    def test_sparsity_illegal_col_datasets(self, backend, m, md):
        "All column maps must share the same data set"
        with pytest.raises(RuntimeError):
            op2.Sparsity(((m, m), (m, md)))

    def test_sparsity_repr(self, backend, sparsity):
        "Sparsity should have the expected repr."

        # Note: We can't actually reproduce a Sparsity from its repr because
        # the Sparsity constructor checks that the maps are populated
        r = "Sparsity(%r, %r)" % (tuple(sparsity.maps), sparsity.name)
        assert repr(sparsity) == r

    def test_sparsity_str(self, backend, sparsity):
        "Sparsity should have the expected string representation."
        s = "OP2 Sparsity: rmaps %s, cmaps %s, name %s" % \
               (sparsity.rmaps, sparsity.cmaps, sparsity.name)
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
            op2.Const((1,'illegaldim'), 1, 'test_const_illegal_dim_tuple')

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
        c = op2.Const([2,3], [1]*6, 'test_const_dim_list')
        c.remove_from_namespace()
        assert c.dim == (2,3)

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

    @pytest.mark.parametrize("dim", [1, (2,2)])
    def test_const_illegal_length(self, backend, dim):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Const(dim, [1]*(np.prod(dim)+1), 'test_const_illegal_length_%r' % np.prod(dim))

    def test_const_reshape(self, backend):
        "Data should be reshaped according to dim."
        c = op2.Const((2,2), [1.0]*4, 'test_const_reshape')
        c.remove_from_namespace()
        assert c.dim == (2,2) and c.data.shape == (2,2)

    def test_const_properties(self, backend):
        "Data constructor should correctly set attributes."
        c = op2.Const((2,2), [1]*4, 'baz', 'double')
        c.remove_from_namespace()
        assert c.dim == (2,2) and c.dtype == np.float64 and c.name == 'baz' \
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
        from pyop2.op2 import Const
        from numpy import array
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
            op2.Global((1,'illegaldim'))

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
        g = op2.Global([2,3], [1]*6)
        assert g.dim == (2,3)

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

    @pytest.mark.parametrize("dim", [1, (2,2)])
    def test_global_illegal_length(self, backend, dim):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Global(dim, [1]*(np.prod(dim)+1))

    def test_global_reshape(self, backend):
        "Data should be reshaped according to dim."
        g = op2.Global((2,2), [1.0]*4)
        assert g.dim == (2,2) and g.data.shape == (2,2)

    def test_global_properties(self, backend):
        "Data globalructor should correctly set attributes."
        g = op2.Global((2,2), [1]*4, 'double', 'bar')
        assert g.dim == (2,2) and g.dtype == np.float64 and g.name == 'bar' \
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
        from pyop2.op2 import Global
        from numpy import array, dtype
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

    def test_map_illegal_dataset(self, backend, set):
        "Map dataset should be Set."
        with pytest.raises(exceptions.SetTypeError):
            op2.Map(set, 'illegalset', 1, [])

    def test_map_illegal_dim(self, backend, set):
        "Map dim should be int."
        with pytest.raises(exceptions.DimTypeError):
            op2.Map(set, set, 'illegaldim', [])

    def test_map_illegal_dim_tuple(self, backend, set):
        "Map dim should not be a tuple."
        with pytest.raises(exceptions.DimTypeError):
            op2.Map(set, set, (2,2), [])

    def test_map_illegal_name(self, backend, set):
        "Map name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Map(set, set, 1, [], name=2)

    def test_map_illegal_dtype(self, backend, set):
        "Illegal data type should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Map(set, set, 1, 'abcdefg')

    def test_map_illegal_length(self, backend, iterset, dataset):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Map(iterset, dataset, 1, [1]*(iterset.size+1))

    def test_map_convert_float_int(self, backend, iterset, dataset):
        "Float data should be implicitely converted to int."
        m = op2.Map(iterset, dataset, 1, [1.5]*iterset.size)
        assert m.values.dtype == np.int32 and m.values.sum() == iterset.size

    def test_map_reshape(self, backend, iterset, dataset):
        "Data should be reshaped according to dim."
        m = op2.Map(iterset, dataset, 2, [1]*2*iterset.size)
        assert m.dim == 2 and m.values.shape == (iterset.size,2)

    def test_map_properties(self, backend, iterset, dataset):
        "Data constructor should correctly set attributes."
        m = op2.Map(iterset, dataset, 2, [1]*2*iterset.size, 'bar')
        assert m.iterset == iterset and m.dataset == dataset and m.dim == 2 \
                and m.values.sum() == 2*iterset.size and m.name == 'bar'

    def test_map_indexing(self, backend, iterset, dataset):
        "Indexing a map should create an appropriate Arg"
        m = op2.Map(iterset, dataset, 2, [1] * 2 * iterset.size, 'm')

        arg = m[0]
        assert arg.idx == 0

    def test_map_slicing(self, backend, iterset, dataset):
        "Slicing a map is not allowed"
        m = op2.Map(iterset, dataset, 2, [1] * 2 * iterset.size, 'm')

        with pytest.raises(NotImplementedError):
            arg = m[:]

    def test_map_equality(self, backend, m):
        """A map is equal if all its attributes are equal, bearing in mind that
        equality is identity for sets."""
        m2 = op2.Map(m.iterset, m.dataset, m.dim, m.values, m.name)
        assert m == m2

    def test_map_copied_set_inequality(self, backend, m):
        """Maps that have copied but not equal iteration sets are not equal"""
        itercopy = op2.Set(m.iterset.size, m.iterset.dim, m.iterset.name)
        m2 = op2.Map(itercopy, m.dataset, m.dim, m.values, m.name)
        assert m != m2

    def test_map_dimension_inequality(self, backend, m):
        """Maps that have different dimensions are not equal"""
        m2 = op2.Map(m.iterset, m.dataset, m.dim*2, list(m.values)*2, m.name)
        assert m != m2

    def test_map_name_inequality(self, backend, m):
        """Maps with different names are not equal"""
        n = op2.Map(m.iterset, m.dataset, m.dim, m.values, 'n')
        assert m != n

    def test_map_repr(self, backend, m):
        "Map should have the expected repr."
        r = "Map(%r, %r, %r, None, %r)" % (m.iterset, m.dataset, m.dim, m.name)
        assert repr(m) == r

    def test_map_str(self, backend, m):
        "Map should have the expected string representation."
        s = "OP2 Map: %s from (%s) to (%s) with dim %s" \
               % (m.name, m.iterset, m.dataset, m.dim)
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
            op2.IterationSpace(set, (1,'illegalextents'))

    def test_iteration_space_extents(self, backend, set):
        "IterationSpace constructor should create a extents tuple."
        m = op2.IterationSpace(set, 1)
        assert m.extents == (1,)

    def test_iteration_space_extents_list(self, backend, set):
        "IterationSpace constructor should create a extents tuple from a list."
        m = op2.IterationSpace(set, [2,3])
        assert m.extents == (2,3)

    def test_iteration_space_properties(self, backend, set):
        "IterationSpace constructor should correctly set attributes."
        i = op2.IterationSpace(set, (2,3))
        assert i.iterset == set and i.extents == (2,3)

    def test_iteration_space_repr(self, backend, set):
        """IterationSpace repr should produce a IterationSpace object when
        eval'd."""
        from pyop2.op2 import Set, IterationSpace
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
        dat = op2.Dat(set1)
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
            base.ParLoop(kernel, set1(3,3), m((rmap[op2.i[0]], cmap[op2.i[1]]), op2.INC))

class TestSolverAPI:
    """
    Test the Solver API.
    """

    def test_solver_defaults(self, backend):
        s = op2.Solver()
        assert s.parameters == base.DEFAULT_SOLVER_PARAMETERS

    def test_set_options_with_params(self, backend):
        params = {'linear_solver': 'gmres',
                  'maximum_iterations': 25 }
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
                  'maximum_iterations': 25 }
        s.update_parameters(params)
        assert s.parameters['linear_solver'] == 'gmres' \
           and s.parameters['maximum_iterations'] == 25

    def test_set_params_and_kwargs_illegal(self, backend):
        params = {'linear_solver': 'gmres',
                  'maximum_iterations': 25 }
        with pytest.raises(RuntimeError):
            op2.Solver(params, linear_solver='cgs')

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
