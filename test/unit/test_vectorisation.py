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
import numpy as np
from pyop2 import op2
from pyop2.configuration import configuration
import os
import pytest
from pyop2.parloop import LegacyParloop
from pyop2.types.glob import Global
from pyop2.types import Access


some_vectorisation_keys = ["__attribute__", "vector_size", "aligned", "#pragma omp simd"]


class TestVectorisation:

    @pytest.fixture
    def s(self):
        return op2.Set(1)

    @pytest.fixture
    def d1(self, s):
        return op2.Dat(s, [3], np.float64)

    @pytest.fixture
    def d(self, s):
        return op2.Dat(s, [4], np.float64)

    def inner(self, s, o):
        s._check_shape(o)
        ret = Global(1, data=0, dtype=s.dtype)
        inner_parloop = LegacyParloop(s._inner_kernel(o.dtype), s.dataset.set,
                                      s(Access.READ), o(Access.READ), ret(Access.INC))
        inner_parloop.compute()
        return (inner_parloop.global_kernel, ret.data_ro[0])

    def test_vectorisation(self, d1, d):
        # Test that vectorised code produced the correct result
        kernel1, ret = self.inner(d, d1)
        assert abs(ret - 12) < 1e-12
        kernel2, ret = self.inner(d1, d)
        assert abs(ret - 12) < 1e-12

        # Test that we actually vectorised
        assert all(key in kernel1.code_to_compile for key in some_vectorisation_keys), "The kernel for an inner(d, d) has not been succesfully vectorised."
        assert all(key in kernel2.code_to_compile for key in some_vectorisation_keys), "The kernel for an inner(d1, d) has not been succesfully vectorised."

    def test_no_vectorisation(self, d1, d):
        # turn vectorisation off
        configuration.reconfigure(vectorization_strategy="")

        # Test that unvectorised code produced the correct result
        kernel1, ret = self.inner(d, d1)
        assert abs(ret - 12) < 1e-12
        kernel2, ret = self.inner(d1, d)
        assert abs(ret - 12) < 1e-12

        # Test that we did not vectorise
        assert not any(key in kernel1.code_to_compile for key in some_vectorisation_keys), "The kernel for an inner(d, d) has been vectorised even though we turned it off."
        assert not any(key in kernel2.code_to_compile for key in some_vectorisation_keys), "The kernel for an inner(d1, d) has been vectorised even though we turned it off."

        # change vect config back to be turned on by default
        configuration.reconfigure(vectorization_strategy="cross-element")


if __name__ == '__main__':
    pytest.main(os.path.abspath(__file__))
