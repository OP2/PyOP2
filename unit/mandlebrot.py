import unittest
import numpy

from pyop2 import op2, py2c
# Initialise OP2
op2.init(backend='opencl', diags=0)

@py2c.kernel_types({"idx" : "uint"})
def mandlebrot(idx):
    bailout = 16
    max_iterations = 8
    x = idx % 120 - 60
    y = idx / 120 - 60
    cr = float(y)/40.0 - 0.5
    ci = float(x)/40.0
    zi = 0.0
    zr = 0.0
    i = 1
    while True:
        temp = zr * zi
        zr2 = zr * zr
        zi2 = zi * zi
        zr = zr2 - zi2 + cr
        zi = temp + temp + ci
        if zi2 + zr2 > bailout:
            idx = i
            return
        elif i > max_iterations:
            idx = i
            return
        i += 1

class MandlebrotTest(unittest.TestCase):
    "Mandlebrot Test"

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_mandlebrot(self):
        """Test write only argument."""
        nelems = 100 * 120
        iterset = op2.Set(nelems, "elems")
        x = op2.Dat(iterset, 1, numpy.array(range(nelems), dtype=numpy.uint32), numpy.uint32, "x")

        op2.par_loop(op2.Kernel(mandlebrot, "mandlebrot"), iterset, x(op2.IdentityMap, op2.RW))

        s = ""
        i = 0

        f = mandlebrot()[2]
        for i in range(len(x.data)):
            val = f(i)[0]
            s += ("\033[%d;%dm%s\033[0m" % (val // 8, 30 + (val % 8), str(val)))
            if i > 0 and i % 120 == 0:
                print s
                s = ""
            assert(int(x.data[i][0]) == val)

suite = unittest.TestLoader().loadTestsFromTestCase(MandlebrotTest)
unittest.TextTestRunner(verbosity=0).run(suite)
