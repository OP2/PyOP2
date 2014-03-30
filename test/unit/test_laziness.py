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
from pyop2 import lazy


nelems = 42


class TestExecutionTrace:

    def test_read_write_dependency(self, backend, skip_greedy):
        a = lazy.LazyPass(set([1]), set())
        b = lazy.LazyPass(set(), set([1]))

        assert b.depends_on(a)

    def test_write_read_dependency(self, backend, skip_greedy):
        a = lazy.LazyPass(set(), set([1]))
        b = lazy.LazyPass(set([1]), set())

        assert b.depends_on(a)

    def test_maintain_write_order(self, backend, skip_greedy):
        a = lazy.LazyPass(set(), set([1]))
        b = lazy.LazyPass(set(), set([1]))

        assert b.depends_on(a)

    def test_empty_trace(self, backend, skip_greedy):
        """Check trace initial state."""
        trace = lazy.ExecutionTrace()
        assert trace.in_queue(trace.top)
        assert trace.in_queue(trace.bot)
        assert trace.bot in trace.children(trace.top)

    def test_enqueue(self, backend, skip_greedy):
        trace = lazy.ExecutionTrace()

        c = lazy.LazyPass(set(), set())
        trace._append(c)

        assert trace.in_queue(c)
        assert c in trace.children(trace.top)
        assert c in trace.parents(trace.bot)

    def test_siblings(self, backend, skip_greedy):
        trace = lazy.ExecutionTrace()

        a = lazy.LazyPass(set([1]), set([1]))
        b = lazy.LazyPass(set([2]), set([2]))

        trace._append(a)
        trace._append(b)

        assert a in trace.children(trace.top)
        assert b in trace.children(trace.top)
        assert len(trace.children(trace.top)) == 2

    def test_children(self, backend, skip_greedy):
        trace = lazy.ExecutionTrace()

        a = lazy.LazyPass(set([1]), set([1]))
        b = lazy.LazyPass(set([1]), set([2]))
        c = lazy.LazyPass(set([1]), set([3]))

        trace._append(a)
        trace._append(b)
        trace._append(c)

        assert a in trace.children(trace.top)
        assert len(trace.children(trace.top)) == 1
        assert b in trace.children(a)
        assert len(trace.parents(b)) == 1
        assert c in trace.children(a)
        assert len(trace.parents(c)) == 1
        assert len(trace.children(a)) == 2


class TestLaziness:

    @pytest.fixture
    def iterset(cls):
        return op2.Set(nelems, name="iterset")

    def test_stable(self, backend, skip_greedy, iterset):
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

    def test_ro_accessor(self, backend, skip_greedy, iterset):
        """Read-only access to a Dat should force computation that writes to it."""
        lazy._trace.clear()
        d = op2.Dat(iterset, numpy.zeros(iterset.total_size), dtype=numpy.float64)
        k = op2.Kernel('void k(double *x) { *x = 1.0; }', 'k')
        op2.par_loop(k, iterset, d(op2.WRITE))
        assert all(d.data_ro == 1.0)

    def test_rw_accessor(self, backend, skip_greedy, iterset):
        """Read-write access to a Dat should force computation that writes to it,
        and any pending computations that read from it."""
        lazy._trace.clear()
        d = op2.Dat(iterset, numpy.zeros(iterset.total_size), dtype=numpy.float64)
        d2 = op2.Dat(iterset, numpy.empty(iterset.total_size), dtype=numpy.float64)
        k = op2.Kernel('void k(double *x) { *x = 1.0; }', 'k')
        k2 = op2.Kernel('void k2(double *x, double *y) { *x = *y; }', 'k2')
        op2.par_loop(k, iterset, d(op2.WRITE))
        op2.par_loop(k2, iterset, d2(op2.WRITE), d(op2.READ))
        assert all(d.data == 1.0)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
