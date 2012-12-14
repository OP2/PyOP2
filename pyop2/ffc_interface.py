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

from ufl import Form
from ufl.algorithms import as_form
from ufl.algorithms.signature import compute_form_signature
from ufl.algorithms.formtransformations import PartExtracter
from ffc import default_parameters, compile_form as ffc_compile_form
from ffc import constants
from ffc.log import set_level, ERROR
from ffc.jitobject import JITObject
import re

from op2 import Kernel

_form_cache = {}

class ElementPartExtracter(PartExtracter):

    def indexed(self, x):
        expression, index = x.operands()
        part, provides = self.visit(expression)

        if isinstance(part,Zero):
            # The expression doesn't directly provide anything we want
            part, provides = (zero(x), set())
            if isinstance(expression, Argument):
                # But we may be able to break the Argument to provide something
                # that we want
                e = expression.element()
                if isinstance(e, MixedElement) and index.free_indices == ():
                    i = index.evaluate(None, None, None, None) - 1
                    c = expression.count()
                    part = Argument(e.sub_elements[i], c)
                    provides = set((part,))

        return (part, provides)

def split_mixed_form(form):
    elements = extract_unique_sub_elements( extract_unique_elements(form) )
    args = []
    for e in aelements:
        args.append(Argument(e,
    e = ElementPartExtracter()


def compile_form(form, name):
    """Compile a form using FFC and return an OP2 kernel"""

    # Check that we get a Form
    if not isinstance(form, Form):
        form = as_form(form)

    from pudb import set_trace; set_trace()

    ffc_parameters = default_parameters()
    ffc_parameters['write_file'] = False
    ffc_parameters['format'] = 'pyop2'

    # Silence FFC
    set_level(ERROR)

    # As of UFL 1.0.0-2 a form signature is stable w.r.t. to Coefficient/Index
    # counts
    key = compute_form_signature(form)
    # Check the cache first: this saves recompiling the form for every time
    # step in time-varying problems
    kernels, form_data = _form_cache.get(key, (None, None))
    if form_data is None:
        code = ffc_compile_form(form, prefix=name, parameters=ffc_parameters)
        form_data = form.form_data()

        kernels = [ Kernel(code, '%s_%s_integral_0_%s' % \
                    (name, ida.domain_type, ida.domain_id)) \
                    for ida in form_data.integral_data ]
        kernels = tuple(kernels)
        _form_cache[key] = kernels, form_data

    # Attach the form data FFC has computed for our form (saves preprocessing
    # the form later on)
    form._form_data = form_data
    return kernels
