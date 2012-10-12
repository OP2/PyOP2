from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import os, sys

try:
    OP2_DIR = os.environ['OP2_DIR']
except KeyError:
    sys.exit("""Error: Could not find OP2 library.

Set the environment variable OP2_DIR to point to the op2 subdirectory
of your OP2 source tree""")

OP2_INC = OP2_DIR + '/c/include'
OP2_LIB = OP2_DIR + '/c/lib'

os.environ['CC'] = 'mpicc'
os.environ['CXX'] = 'mpicxx'

setup(name='pyop2',
      version='0,1',
      packages=['pyop2','pyop2_utils'],
      package_dir={'pyop2':'pyop2','pyop2_utils':'pyop2_utils'},
      package_data={'pyop2': ['assets/*']},
      cmdclass = {'build_ext' : build_ext},
      ext_modules=[Extension('pyop2.op_lib_core', ['pyop2/op_lib_core.pyx'],
                   pyrex_include_dirs=['pyop2'],
                   include_dirs=[OP2_INC] + [np.get_include()],
                   library_dirs=[OP2_LIB],
                   runtime_library_dirs=[OP2_LIB],
                   libraries=["op2_seq"])])
