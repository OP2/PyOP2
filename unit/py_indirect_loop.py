import unittest
import numpy
import random

from pyop2 import op2, py2c
# Initialise OP2
op2.init(backend='opencl', diags=0)

def _seed():
    return 0.02041724

#max...
nelems = 92681

@py2c.kernel_types({"x" : "__local uint"})
def kernel_wo(x):
    x = 42

@py2c.kernel_types({"x" : "__local uint"})
def kernel_rw(x):
    x += 1

@py2c.kernel_types({"x" : "__private uint"})
def kernel_inc(x):
    x = x + 1

@py2c.kernel_types({"x" : "__local uint", "inc" : "__private uint"})
def kernel_global_inc(x, inc):
    x += 1
    inc += x

class IndirectLoopTest(unittest.TestCase):
    """

    Indirect Loop Tests

    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_onecolor_wo(self):
        iterset = op2.Set(nelems, "iterset")
        indset = op2.Set(nelems, "indset")

        x = op2.Dat(indset, 1, numpy.array(range(nelems), dtype=numpy.uint32), numpy.uint32, "x")

        u_map = numpy.array(range(nelems), dtype=numpy.uint32)
        random.shuffle(u_map, _seed)
        iterset2indset = op2.Map(iterset, indset, 1, u_map, "iterset2indset")

        op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"), iterset, x(iterset2indset(0), op2.WRITE))
        self.assertTrue(all(map(lambda x: x==42, x.data)))

    def test_onecolor_rw(self):
        iterset = op2.Set(nelems, "iterset")
        indset = op2.Set(nelems, "indset")

        x = op2.Dat(indset, 1, numpy.array(range(nelems), dtype=numpy.uint32), numpy.uint32, "x")

        u_map = numpy.array(range(nelems), dtype=numpy.uint32)
        random.shuffle(u_map, _seed)
        iterset2indset = op2.Map(iterset, indset, 1, u_map, "iterset2indset")

        op2.par_loop(op2.Kernel(kernel_rw, "kernel_rw"), iterset, x(iterset2indset(0), op2.RW))
        self.assertEqual(sum(x.data), nelems * (nelems + 1) / 2);

    def test_indirect_inc(self):
        iterset = op2.Set(nelems, "iterset")
        unitset = op2.Set(1, "unitset")

        u = op2.Dat(unitset, 1, numpy.array([0], dtype=numpy.uint32), numpy.uint32, "u")

        u_map = numpy.zeros(nelems, dtype=numpy.uint32)
        iterset2unit = op2.Map(iterset, unitset, 1, u_map, "iterset2unitset")

        op2.par_loop(op2.Kernel(kernel_inc, "kernel_inc"), iterset, u(iterset2unit(0), op2.INC))
        self.assertEqual(u.data[0], nelems)

    def test_global_inc(self):
        iterset = op2.Set(nelems, "iterset")
        indset = op2.Set(nelems, "indset")

        x = op2.Dat(indset, 1, numpy.array(range(nelems), dtype=numpy.uint32), numpy.uint32, "x")
        g = op2.Global(1, 0, numpy.uint32, "g")

        u_map = numpy.array(range(nelems), dtype=numpy.uint32)
        random.shuffle(u_map, _seed)
        iterset2indset = op2.Map(iterset, indset, 1, u_map, "iterset2indset")

        op2.par_loop(op2.Kernel(kernel_global_inc, "kernel_global_inc"), iterset,
                     x(iterset2indset(0), op2.RW),
                     g(op2.INC))
        self.assertEqual(sum(x.data), nelems * (nelems + 1) / 2)
        self.assertEqual(g.data[0], nelems * (nelems + 1) / 2)

suite = unittest.TestLoader().loadTestsFromTestCase(IndirectLoopTest)
unittest.TextTestRunner(verbosity=0, failfast=False).run(suite)

# refactor to avoid recreating input data for each test cases
