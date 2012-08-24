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

backends = ['sequential']

#max...
nelems = 92681

class TestLazyness:
    """
    Lazyness Tests.
    """

    def test_stable(self, backend):
        """Test accessing a dependency is stable."""
        iterset = op2.Set(nelems, "iterset")

        a = op2.Global(1, 0, numpy.uint32, "a")

        kernel_count = """
void
count(unsigned int* x)
{
  (*x) += 1;
}
"""
        op2.par_loop(op2.Kernel(kernel_count, "count"), iterset, a(op2.INC))

        assert(a._data[0] == 0)
        assert(a.data[0] == nelems)
        assert(a.data[0] == nelems)

    def test_const(self, backend):
        """Test constant dependencies."""
        iterset = op2.Set(nelems, "iterset")

        a = op2.Global(1, 0, numpy.uint32, "a")
        b = op2.Global(1, 0, numpy.uint32, "b")

        c = op2.Const(1, 0, "c", dtype=numpy.uint32)

        kernel_count = """
void
count(unsigned int* x)
{
  (*x) += 1;
}
"""
        op2.par_loop(op2.Kernel(kernel_count, "count"), iterset, a(op2.INC))

        assert(a._data[0] == 0)
        assert(b._data[0] == 0)

        d = op2.Const(1, 0, "d", dtype=numpy.uint32)

        op2.par_loop(op2.Kernel(kernel_count, "count"), iterset, b(op2.INC))

        d.data = [1]
        c.data = [1]
        assert(a._data[0] == 0)
        assert(b._data[0] == 0)

        #force second par_loop
        assert(d.data[0] == 1)
        assert(a._data[0] == 0)
        assert(b._data[0] == nelems)

        #force first par_loop
        assert(c.data[0] == 1)
        assert(a._data[0] == nelems)
        assert(b._data[0] == nelems)

        # ??? FIX ???
        c.remove_from_namespace()
        d.remove_from_namespace()

    def test_reorder(self, backend):
        """Test two independant computations."""
        iterset = op2.Set(nelems, "iterset")

        a = op2.Global(1, 0, numpy.uint32, "a")
        b = op2.Global(1, 0, numpy.uint32, "b")

        kernel_count = """
void
count(unsigned int* x)
{
  (*x) += 1;
}
"""
        op2.par_loop(op2.Kernel(kernel_count, "count"), iterset, a(op2.INC))
        op2.par_loop(op2.Kernel(kernel_count, "count"), iterset, b(op2.INC))

        assert(a._data[0] == 0)
        assert(b._data[0] == 0)
        assert(b.data[0] == nelems)
        assert(a._data[0] == 0)
        assert(a.data[0] == nelems)

    def test_cascade(self, backend):
        """Test cascade of dependencies."""
        iterset = op2.Set(nelems, "iterset")

        a = op2.Dat(iterset, 1, numpy.zeros(nelems), numpy.uint32, "a")
        b = op2.Dat(iterset, 1, numpy.zeros(nelems), numpy.uint32, "b")
        c = op2.Global(1, 0, numpy.uint32, "c")

        kernel_ppone = """
void
ppone(unsigned int* x)
{
(*x) += 1;
}
"""
        kernel_transfer = """
void
transfer(unsigned int* x, unsigned int* y)
{
(*x) = (*y);
}
"""
        kernel_sum = """
void
sum(unsigned int* x, unsigned int* y)
{
(*x) += (*y);
}
"""

        op2.par_loop(op2.Kernel(kernel_ppone, "ppone"), iterset, a(op2.IdentityMap, op2.RW))
        op2.par_loop(op2.Kernel(kernel_transfer, "transfer"), iterset, b(op2.IdentityMap, op2.WRITE), a(op2.IdentityMap, op2.READ))
        op2.par_loop(op2.Kernel(kernel_sum, "sum"), iterset, c(op2.INC), a(op2.IdentityMap, op2.READ))

        assert(sum(a._data) == 0)
        assert(sum(b._data) == 0)
        assert(c._data[0] == 0)

        assert(c.data[0] == nelems)
        assert(sum(a._data) == nelems)
        assert(sum(a.data) == nelems)
        assert(sum(b._data) == 0)
        assert(sum(b.data) == nelems)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
