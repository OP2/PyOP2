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

import pytest
import numpy as np

from pyop2 import op2

nelems = 5


@pytest.fixture(scope='module')
def s():
    return op2.Set(nelems)


@pytest.fixture
def d1(s):
    return op2.Dat(s, range(nelems), dtype=np.float64)


@pytest.fixture
def mdat(d1):
    return op2.MixedDat([d1, d1])


class TestDat:

    """
    Test some properties of Dats
    """

    def test_copy_constructor(self, backend, d1):
        """Dat copy constructor should copy values"""
        d2 = op2.Dat(d1)
        assert d1.dataset.set == d2.dataset.set
        assert (d1.data_ro == d2.data_ro).all()
        d1.data[:] = -1
        assert (d1.data_ro != d2.data_ro).all()

    def test_copy_constructor_mixed(self, backend, mdat):
        """MixedDat copy constructor should copy values"""
        mdat2 = op2.MixedDat(mdat)
        assert mdat.dataset.set == mdat2.dataset.set
        assert all(all(d.data_ro == d_.data_ro) for d, d_ in zip(mdat, mdat2))
        for dat in mdat.data:
            dat[:] = -1
        assert all(all(d.data_ro != d_.data_ro) for d, d_ in zip(mdat, mdat2))

    def test_copy(self, backend, d1, s):
        """Copy method on a Dat should copy values into given target"""
        d2 = op2.Dat(s)
        d1.copy(d2)
        assert d1.dataset.set == d2.dataset.set
        assert (d1.data_ro == d2.data_ro).all()
        d1.data[:] = -1
        assert (d1.data_ro != d2.data_ro).all()

    def test_copy_mixed(self, backend, s, mdat):
        """Copy method on a MixedDat should copy values into given target"""
        mdat2 = op2.MixedDat([s, s])
        mdat.copy(mdat2)
        assert all(all(d.data_ro == d_.data_ro) for d, d_ in zip(mdat, mdat2))
        for dat in mdat.data:
            dat[:] = -1
        assert all(all(d.data_ro != d_.data_ro) for d, d_ in zip(mdat, mdat2))

    def test_copy_subset(self, backend, s, d1):
        """Copy method should copy values on a subset"""
        d2 = op2.Dat(s)
        ss = op2.Subset(s, range(1, nelems, 2))
        d1.copy(d2, subset=ss)
        assert (d1.data_ro[ss.indices] == d2.data_ro[ss.indices]).all()
        assert (d2.data_ro[::2] == 0).all()

    def test_copy_mixed_subset_fails(self, backend, s, mdat):
        """Copy method on a MixedDat does not support subsets"""
        with pytest.raises(TypeError):
            mdat.copy(op2.MixedDat([s, s]), subset=None)

    @pytest.mark.skipif('config.getvalue("backend")[0] not in ["cuda", "opencl"]')
    def test_copy_works_device_to_device(self, backend, d1):
        d2 = op2.Dat(d1)

        # Check we didn't do a copy on the host
        assert not d2._is_allocated
        assert not (d2._data == d1.data).all()
        from pyop2 import device
        assert d2.state is device.DeviceDataMixin.DEVICE

    @pytest.mark.parametrize('dim', [1, 2])
    def test_dat_nbytes(self, backend, dim):
        """Nbytes computes the number of bytes occupied by a Dat."""
        s = op2.Set(10)
        assert op2.Dat(s**dim).nbytes == 10*8*dim

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
