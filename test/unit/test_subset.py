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

from coffee.base import *

backends = ['sequential', 'openmp', 'opencl', 'cuda']

nelems = 32


@pytest.fixture(params=[(nelems, nelems, nelems, nelems),
                        (0, nelems, nelems, nelems),
                        (nelems / 2, nelems, nelems, nelems)])
def iterset(request):
    return op2.Set(request.param, "iterset")


class TestSubSet:

    """
    SubSet tests
    """

    def test_direct_loop(self, backend, iterset):
        """Test a direct ParLoop on a subset"""
        indices = np.array([i for i in range(nelems) if not i % 2], dtype=np.int)
        ss = op2.Subset(iterset, indices)

        d = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        k = op2.Kernel("void inc(unsigned int* v) { *v += 1; }", "inc")
        op2.par_loop(k, ss, d(op2.RW))
        inds, = np.where(d.data)
        assert (inds == indices).all()

    def test_direct_loop_empty(self, backend, iterset):
        """Test a direct loop with an empty subset"""
        ss = op2.Subset(iterset, [])
        d = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        k = op2.Kernel("void inc(unsigned int* v) { *v += 1; }", "inc")
        op2.par_loop(k, ss, d(op2.RW))
        inds, = np.where(d.data)
        assert (inds == []).all()

    def test_direct_complementary_subsets(self, backend, iterset):
        """Test direct par_loop over two complementary subsets"""
        even = np.array([i for i in range(nelems) if not i % 2], dtype=np.int)
        odd = np.array([i for i in range(nelems) if i % 2], dtype=np.int)

        sseven = op2.Subset(iterset, even)
        ssodd = op2.Subset(iterset, odd)

        d = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        k = op2.Kernel("void inc(unsigned int* v) { *v += 1; }", "inc")
        op2.par_loop(k, sseven, d(op2.RW))
        op2.par_loop(k, ssodd, d(op2.RW))
        assert (d.data == 1).all()

    def test_direct_complementary_subsets_with_indexing(self, backend, iterset):
        """Test direct par_loop over two complementary subsets"""
        even = np.arange(0, nelems, 2, dtype=np.int)
        odd = np.arange(1, nelems, 2, dtype=np.int)

        sseven = iterset(even)
        ssodd = iterset(odd)

        d = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        k = op2.Kernel("void inc(unsigned int* v) { *v += 1; }", "inc")
        op2.par_loop(k, sseven, d(op2.RW))
        op2.par_loop(k, ssodd, d(op2.RW))
        assert (d.data == 1).all()

    def test_direct_loop_sub_subset(self, backend, iterset):
        indices = np.arange(0, nelems, 2, dtype=np.int)
        ss = op2.Subset(iterset, indices)
        indices = np.arange(0, nelems/2, 2, dtype=np.int)
        sss = op2.Subset(ss, indices)

        d = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        k = op2.Kernel("void inc(unsigned int* v) { *v += 1; }", "inc")
        op2.par_loop(k, sss, d(op2.RW))

        indices = np.arange(0, nelems, 4, dtype=np.int)
        ss2 = op2.Subset(iterset, indices)
        d2 = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        op2.par_loop(k, ss2, d2(op2.RW))

        assert (d.data == d2.data).all()

    def test_direct_loop_sub_subset_with_indexing(self, backend, iterset):
        indices = np.arange(0, nelems, 2, dtype=np.int)
        ss = iterset(indices)
        indices = np.arange(0, nelems/2, 2, dtype=np.int)
        sss = ss(indices)

        d = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        k = op2.Kernel("void inc(unsigned int* v) { *v += 1; }", "inc")
        op2.par_loop(k, sss, d(op2.RW))

        indices = np.arange(0, nelems, 4, dtype=np.int)
        ss2 = iterset(indices)
        d2 = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        op2.par_loop(k, ss2, d2(op2.RW))

        assert (d.data == d2.data).all()

    def test_indirect_loop(self, backend, iterset):
        """Test a indirect ParLoop on a subset"""
        indices = np.array([i for i in range(nelems) if not i % 2], dtype=np.int)
        ss = op2.Subset(iterset, indices)

        indset = op2.Set(2, "indset")
        map = op2.Map(iterset, indset, 1, [(1 if i % 2 else 0) for i in range(nelems)])
        d = op2.Dat(indset ** 1, data=None, dtype=np.uint32)

        k = op2.Kernel("void inc(unsigned int* v) { *v += 1;}", "inc")
        op2.par_loop(k, ss, d(op2.INC, map[0]))

        assert d.data[0] == nelems / 2

    def test_indirect_loop_empty(self, backend, iterset):
        """Test a indirect ParLoop on an empty"""
        ss = op2.Subset(iterset, [])

        indset = op2.Set(2, "indset")
        map = op2.Map(iterset, indset, 1, [(1 if i % 2 else 0) for i in range(nelems)])
        d = op2.Dat(indset ** 1, data=None, dtype=np.uint32)

        k = op2.Kernel("void inc(unsigned int* v) { *v += 1;}", "inc")
        d.data[:] = 0
        op2.par_loop(k, ss, d(op2.INC, map[0]))

        assert (d.data == 0).all()

    def test_indirect_loop_with_direct_dat(self, backend, iterset):
        """Test a indirect ParLoop on a subset"""
        indices = np.array([i for i in range(nelems) if not i % 2], dtype=np.int)
        ss = op2.Subset(iterset, indices)

        indset = op2.Set(2, "indset")
        map = op2.Map(iterset, indset, 1, [(1 if i % 2 else 0) for i in range(nelems)])

        values = [2976579765] * nelems
        values[::2] = [i/2 for i in range(nelems)][::2]
        dat1 = op2.Dat(iterset ** 1, data=values, dtype=np.uint32)
        dat2 = op2.Dat(indset ** 1, data=None, dtype=np.uint32)

        k = op2.Kernel("void inc(unsigned* s, unsigned int* d) { *d += *s;}", "inc")
        op2.par_loop(k, ss, dat1(op2.READ), dat2(op2.INC, map[0]))

        assert dat2.data[0] == sum(values[::2])

    def test_complementary_subsets(self, backend, iterset):
        """Test par_loop on two complementary subsets"""
        even = np.array([i for i in range(nelems) if not i % 2], dtype=np.int)
        odd = np.array([i for i in range(nelems) if i % 2], dtype=np.int)

        sseven = op2.Subset(iterset, even)
        ssodd = op2.Subset(iterset, odd)

        indset = op2.Set(nelems, "indset")
        map = op2.Map(iterset, indset, 1, [i for i in range(nelems)])
        dat1 = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        dat2 = op2.Dat(indset ** 1, data=None, dtype=np.uint32)

        k = op2.Kernel("""\
void
inc(unsigned int* v1, unsigned int* v2) {
  *v1 += 1;
  *v2 += 1;
}
""", "inc")
        op2.par_loop(k, sseven, dat1(op2.RW), dat2(op2.INC, map[0]))
        op2.par_loop(k, ssodd, dat1(op2.RW), dat2(op2.INC, map[0]))

        assert np.sum(dat1.data) == nelems
        assert np.sum(dat2.data) == nelems

    def test_matrix(self, backend, skip_opencl):
        """Test a indirect par_loop with a matrix argument"""
        iterset = op2.Set(2)
        idset = op2.Set(2)
        ss01 = op2.Subset(iterset, [0, 1])
        ss10 = op2.Subset(iterset, [1, 0])
        indset = op2.Set(4)

        dat = op2.Dat(idset ** 1, data=[0, 1], dtype=np.float)
        map = op2.Map(iterset, indset, 4, [0, 1, 2, 3, 0, 1, 2, 3])
        idmap = op2.Map(iterset, idset, 1, [0, 1])
        sparsity = op2.Sparsity((indset, indset), (map, map))
        mat = op2.Mat(sparsity, np.float64)
        mat01 = op2.Mat(sparsity, np.float64)
        mat10 = op2.Mat(sparsity, np.float64)

        assembly = c_for("i", 4,
                         c_for("j", 4,
                               Incr(Symbol("mat", ("i", "j")), FlatBlock("(*dat)*16+i*4+j"))))
        kernel_code = FunDecl("void", "unique_id",
                              [Decl("double*", c_sym("dat")),
                               Decl("double", Symbol("mat", (4, 4)))],
                              Block([assembly], open_scope=False))
        k = op2.Kernel(kernel_code, "unique_id")

        mat.zero()
        mat01.zero()
        mat10.zero()

        op2.par_loop(k, iterset,
                     dat(op2.READ, idmap[0]),
                     mat(op2.INC, (map[op2.i[0]], map[op2.i[1]])))
        mat.assemble()
        op2.par_loop(k, ss01,
                     dat(op2.READ, idmap[0]),
                     mat01(op2.INC, (map[op2.i[0]], map[op2.i[1]])))
        mat01.assemble()
        op2.par_loop(k, ss10,
                     dat(op2.READ, idmap[0]),
                     mat10(op2.INC, (map[op2.i[0]], map[op2.i[1]])))
        mat10.assemble()

        assert (mat01.values == mat.values).all()
        assert (mat10.values == mat.values).all()
