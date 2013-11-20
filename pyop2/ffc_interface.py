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

"""Provides the interface to FFC for compiling a form, and transforms the FFC-
generated code in order to make it suitable for passing to the backends."""

from hashlib import md5
import os
import tempfile

from ufl import Form
from ufl.algorithms import as_form
from ffc import default_parameters, compile_form as ffc_compile_form
from ffc import constants
from ffc.log import set_level, ERROR

from caching import DiskCached
from op2 import Kernel
from mpi import MPI

from ir.ast_base import PreprocessNode, Root

_form_cache = {}

# Silence FFC
set_level(ERROR)

ffc_parameters = default_parameters()
ffc_parameters['write_file'] = False
ffc_parameters['format'] = 'pyop2'

AST = False
if AST:
    ffc_parameters['pyop2-ir'] = True
else:
    ffc_parameters['pyop2-ir'] = False


def _check_version():
    from version import __compatible_ffc_version_info__ as compatible_version, \
        __compatible_ffc_version__ as version
    try:
        if constants.PYOP2_VERSION_INFO[:2] == compatible_version[:2]:
            return
    except AttributeError:
        pass
    raise RuntimeError("Incompatible PyOP2 version %s and FFC PyOP2 version %s."
                       % (version, getattr(constants, 'PYOP2_VERSION', 'unknown')))


class FFCKernel(DiskCached):

    _cache = {}
    _cachedir = os.path.join(tempfile.gettempdir(),
                             'pyop2-ffc-kernel-cache-uid%d' % os.getuid())

    @classmethod
    def _cache_key(cls, form, name):
        form_data = form.compute_form_data()
        return md5(form_data.signature + name + Kernel._backend.__name__ +
                   constants.FFC_VERSION + constants.PYOP2_VERSION).hexdigest()

    def __init__(self, form, name):
        # If in cache, return
        # if self._initialized:
        #    return

        if AST:
            incl = PreprocessNode('#include "pyop2_geometry.h"\n')
            ffc_tree = ffc_compile_form(
                form, prefix=name, parameters=ffc_parameters)
            ast = Root([incl] + [subtree for subtree in ffc_tree])
            code = ast.gencode()
            # print code
        else:
            code = '#include "pyop2_geometry.h"\n'
            code += ffc_compile_form(form, prefix=name,
                                     parameters=ffc_parameters)
            # print code

        form_data = form.form_data()

        self.kernels = tuple([Kernel(code, '%s_%s_integral_0_%s' %
                            (name, ida.domain_type, ida.domain_id))
            for ida in form_data.integral_data])
        self._initialized = True


def compile_form(form, name):
    """Compile a form using FFC and return a :class:`pyop2.op2.Kernel`."""

    # Check that we get a Form
    if not isinstance(form, Form):
        form = as_form(form)

    return FFCKernel(form, name).kernels

_check_version()
if not os.path.exists(FFCKernel._cachedir) and MPI.comm.rank == 0:
    os.makedirs(FFCKernel._cachedir)
