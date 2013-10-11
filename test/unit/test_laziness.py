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
Lazy evaluation unit tests.
"""

import pytest
import numpy

from pyop2 import op2
from pyop2 import configuration as cfg

nelems = 42


def _is_greedy():
    return not cfg['lazy_evaluation']


class TestLaziness:

    @pytest.fixture
    def iterset(cls):
        return op2.Set(nelems, name="iterset")

    @pytest.mark.skipif("_is_greedy()")
    def test_stable(self, backend, iterset):
        a = op2.Global(1, 0, numpy.uint32, "a")

        kernel = """
void
count(unsigned int* x)
{
  (*x) += 1;
}
"""
        op2.par_loop(op2.Kernel(kernel, "count"), iterset, a(op2.INC))

        assert a._data[0] == 0
        assert a.data[0] == nelems
        assert a.data[0] == nelems

    @pytest.mark.skipif("_is_greedy()")
    def test_reorder(self, backend, iterset):
        a = op2.Global(1, 0, numpy.uint32, "a")
        b = op2.Global(1, 0, numpy.uint32, "b")

        kernel = """
void
count(unsigned int* x)
{
  (*x) += 1;
}
"""
        op2.par_loop(op2.Kernel(kernel, "count"), iterset, a(op2.INC))
        op2.par_loop(op2.Kernel(kernel, "count"), iterset, b(op2.INC))

        assert a._data[0] == 0
        assert b._data[0] == 0
        assert b.data[0] == nelems
        assert a._data[0] == 0
        assert a.data[0] == nelems

    @pytest.mark.skipif("_is_greedy()")
    def test_chain(self, backend, iterset):
        a = op2.Global(1, 0, numpy.uint32, "a")
        x = op2.Dat(iterset, numpy.zeros(nelems), numpy.uint32, "x")
        y = op2.Dat(iterset, numpy.zeros(nelems), numpy.uint32, "y")

        kernel_add_one = """
void
add_one(unsigned int* x)
{
  (*x) += 1;
}
"""
        kernel_copy = """
void
copy(unsigned int* dst, unsigned int* src)
{
  (*dst) = (*src);
}
"""
        kernel_sum = """
void
sum(unsigned int* sum, unsigned int* x)
{
  (*sum) += (*x);
}
"""

        pl_add = op2.par_loop(op2.Kernel(kernel_add_one, "add_one"), iterset, x(op2.RW))
        pl_copy = op2.par_loop(op2.Kernel(kernel_copy, "copy"), iterset, y(op2.WRITE), x(op2.READ))
        pl_sum = op2.par_loop(op2.Kernel(kernel_sum, "sum"), iterset, a(op2.INC), x(op2.READ))

        # check everything is zero at first
        assert sum(x._data) == 0
        assert sum(y._data) == 0
        assert a._data[0] == 0
        assert op2.base._trace.in_queue(pl_add)
        assert op2.base._trace.in_queue(pl_copy)
        assert op2.base._trace.in_queue(pl_sum)

        # force computation affecting 'a' (1st and 3rd par_loop)
        assert a.data[0] == nelems
        assert not op2.base._trace.in_queue(pl_add)
        assert op2.base._trace.in_queue(pl_copy)
        assert not op2.base._trace.in_queue(pl_sum)
        assert sum(x.data) == nelems

        # force the last par_loop remaining (2nd)
        assert sum(y.data) == nelems
        assert not op2.base._trace.in_queue(pl_copy)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
