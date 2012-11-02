from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import os, sys

# Get OP2 include and library paths
execfile('pyop2/paths.py')
print """Using the OP2 library installed in %s

If this is incorrect or you want to use a different OP2 installation,
set the environment variable OP2_DIR to point to the op2 subdirectory
of your OP2 source tree or OP2_PREFIX to point to the location of an
OP2 install.
""" % OP2_PREFIX

os.environ['CC'] = 'mpicc'
os.environ['CXX'] = 'mpicxx'

setup(name='pyop2',
      version='0,1',
      description = 'OP2 runtime library and python bindings',
      author = 'Imperial College London and others',
      author_email = 'mapdes@imperial.ac.uk',
      url = 'https://github.com/OP2/PyOP2/',
      classifiers = [
            'Development Status :: 2 - Pre-Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
            'Programming Language :: C',
            'Programming Language :: Cython',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
         ],
      packages=['pyop2','pyop2_utils'],
      package_dir={'pyop2':'pyop2','pyop2_utils':'pyop2_utils'},
      package_data={'pyop2': ['assets/*', 'mat_utils.*']},
      ext_modules=[Extension('pyop2.op_lib_core', ['pyop2/op_lib_core.c'],
                   pyrex_include_dirs=['pyop2'],
                   include_dirs=[OP2_INC] + [np.get_include()],
                   library_dirs=[OP2_LIB],
                   runtime_library_dirs=[OP2_LIB],
                   libraries=["op2_seq"])])
