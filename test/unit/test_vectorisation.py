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
import glob
import os
import pytest


some_vectorisation_keys = ["__attribute__", "vector_size", "aligned", "#pragma omp simd"]


class TestVectorisation:

    @pytest.fixture
    def s(self):
        return op2.Set(1)

    @pytest.fixture
    def md1(self, s):
        n = op2.Dat(s, [3], np.float64)
        o = op2.Dat(s, [4], np.float64)
        md = op2.MixedDat([n, o])
        return md

    @pytest.fixture
    def md(self, s):
        n = op2.Dat(s, [4], np.float64)
        o = op2.Dat(s, [5], np.float64)
        md = op2.MixedDat([n, o])
        return md

    def test_vectorisation(self, md1, md):
        # Test that vectorised code produced the correct result
        ret = md.inner(md1)
        assert abs(ret - 32) < 1e-12
        ret = md1.inner(md)
        assert abs(ret - 32) < 1e-12

        # Test that we actually vectorised
        list_of_files = glob.glob(configuration["cache_dir"]+"/*/*.c")
        latest_file = max(list_of_files, key=os.path.getctime)
        with open(latest_file, 'r') as file:
            generated_code = file.read()
        assert (all(key in generated_code for key in some_vectorisation_keys)), "The kernel for an inner product has not been succesfully vectorised."

    def test_no_vectorisation(self, md1, md):
        # turn vectorisation off
        op2.init(**{"vectorization_strategy": ""})

        # Test that unvectorised code produced the correct result
        ret = md.inner(md1)
        assert abs(ret - 32) < 1e-12
        ret = md1.inner(md)
        assert abs(ret - 32) < 1e-12

        # Test that we did not vectorise
        list_of_files = glob.glob(configuration["cache_dir"]+"/*/*.c")
        latest_file = max(list_of_files, key=os.path.getctime)
        with open(latest_file, 'r') as file:
            generated_code = file.read()
        assert not (any(key in generated_code for key in some_vectorisation_keys)), "The kernel for an inner product has not been succesfully vectorised."

        # change vect config back to be turned on by default
        op2.init(**{"vectorization_strategy": "cross-element"})


if __name__ == '__main__':
    pytest.main(os.path.abspath(__file__))
