from cPickle import dump, load, HIGHEST_PROTOCOL
from os.path import join, dirname, abspath

from pyop2 import op2

_kerneldir = join(dirname(abspath(__file__)), 'kernels')


def dump_kernel(kernel):
    with open(join(_kerneldir, '%s.pickle' % kernel.name), 'w') as f:
        dump(kernel._ast, f, HIGHEST_PROTOCOL)


def load_kernel(name):
    with open(join(_kerneldir, '%s.pickle' % name)) as f:
        return op2.Kernel(load(f), name, include_dirs=[_kerneldir])
