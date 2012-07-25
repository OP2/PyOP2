import pytest
import numpy

from pyop2 import op2

def setup_module(module):
    # Initialise OP2
    op2.init(backend='sequential', diags=0)

def teardown_module(module):
    op2.exit()

def _seed():
    return 0.02041724

#max...
nelems = 92681

class TestLazyness:
    """
    Lazyness
    """

    def test_stable(self):
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

    def test_const(self):
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

        assert(b._data[0] == nelems)
        assert(a._data[0] == 0)

        c.data = [1]

        assert(a._data[0] == nelems)

    def test_reorder(self):
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

    def test_cascade(self):
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
