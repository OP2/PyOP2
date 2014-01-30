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
import numpy as np

from ufl import split
from ufl import Argument, Form, FiniteElement, VectorElement
from ufl.algorithms import as_form, traverse_terminals, ReuseTransformer
from ufl.indexing import FixedIndex, MultiIndex
from ffc import default_parameters, compile_form as ffc_compile_form
from ffc import constants
from ffc.log import set_level, ERROR

from caching import DiskCached, KernelCached
from op2 import Kernel
from mpi import MPI

from ir.ast_base import PreprocessNode, Root
from utils import as_tuple

_form_cache = {}

# Silence FFC
set_level(ERROR)

ffc_parameters = default_parameters()
ffc_parameters['write_file'] = False
ffc_parameters['format'] = 'pyop2'
ffc_parameters['pyop2-ir'] = True

# Include an md5 hash of pyop2_geometry.h in the cache key
with open(os.path.join(os.path.dirname(__file__), 'pyop2_geometry.h')) as f:
    _pyop2_geometry_md5 = md5(f.read()).hexdigest()


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


as_list = lambda o: o if isinstance(o, list) else [o]


class FormSplitter(ReuseTransformer):
    """Split a form into a subtree for each component of the mixed space it is
    built on. This is a no-op on forms over non-mixed spaces."""

    def split(self, form):
        """Split the given form."""
        fd = form.form_data()
        # If there is no mixed element involved, return the unmodified form
        if all(isinstance(e, (FiniteElement, VectorElement)) for e in fd.unique_sub_elements):
            return [((0, 0), form)]
        return [[f * i.measure() for f in as_tuple(self.visit(i.integrand()))]
                for i in form.integrals()]

    def sum(self, o, l, r):

        def pop_index(l, idx):
            "Pop item with index idx from list l or return (None, None)."
            for e in l:
                if e[0] == idx:
                    l.remove(e)
                    return e
            return None, None
        from IPython import embed
        embed()
        res = []
        # For each (index, argument) tuple in the left operand list, look for
        # a tuple with corresponding index in the right operand list. If
        # there is one, append the sum of the arguments with that index to the
        # results list, otherwise just the tuple from the left operand list
        for idx, s1 in l:
            _, s2 = pop_index(r, idx)
            if s2:
                res.append((idx, o.reconstruct(s1, s2)))
            else:
                res.append((idx, s1))
        # All remaining tuples in the right operand list had no matches, so we
        # append them to the results list
        return res + r

    def _binop(self, o, l, r):
        # Combine operands where one lives on a test and one on a trial
        # function. The combined index has the test function first.
        if isinstance(l, tuple) and isinstance(r, tuple):
            def find_count(e):
                "Find the count of an argument: -2 test function, -1 trial function."
                for t in traverse_terminals(e):
                    if isinstance(t, Argument):
                        return t.count()
            i, op1 = l
            j, op2 = r
            if find_count(op1) < find_count(op2):
                return (i, j), o.reconstruct(op1, op2)
            else:
                return (j, i), o.reconstruct(op1, op2)
        # Only the left operand contains an Argument. Use its block index.
        elif isinstance(l, tuple):
            i, p1 = l
            return i, o.reconstruct(p1, r)
        # Only the right operand contains an Argument. Use its block index.
        elif isinstance(r, tuple):
            j, p2 = r
            return j, o.reconstruct(l, p2)
        # None of the operands contains an Argument. Don't use a block index.
        else:
            return o.reconstruct(l, r)

    def inner(self, o, l, r):
        """Reconstruct an inner product on each of the component spaces."""
        if isinstance(l, tuple) and isinstance(r, tuple):
            return tuple(o.reconstruct(op1, op2) for op1, op2 in zip(l, r))
        else:
            return o.reconstruct(l, r)

    def product(self, o, l, r):
        """Reconstruct a product on each of the component spaces."""
        return self._binop(o, l, r)

    def dot(self, o, l, r):
        """Reconstruct a product on each of the component spaces."""
        return self._binop(o, l, r)

    def indexed(self, o, arg, idx):
        """Reconstruct the :class:`ufl.indexed.Indexed` only if the coefficient
        is defined on a :class:`core_types.VectorFunctionSpace`."""
        if isinstance(idx._indices[0], FixedIndex):
            # Find the element to which the FixedIndex points. We might deal
            # with coefficients on vector elements, in which case we need to
            # reconstruct the indexed with an adjusted index space. Otherwise
            # we can just return the coefficient.
            i = idx._indices[0]._value
            pos = 0
            for op in arg:
                # If the FixedIndex points at a scalar (shapeless) operand,
                # return it
                if not op.shape() and i == pos:
                    return op
                size = np.prod(op.shape() or 1)
                # If the FixedIndex points at a component of the current
                # operand, reconstruct an Indexed with an adjusted index space
                if i < pos + size:
                    return o.reconstruct(op, MultiIndex(FixedIndex(i - pos), {}))
                # Otherwise update the position in the index space
                pos += size
            raise NotImplementedError("No idea what to in %r with %r" % (o, arg))
        else:
            return o.reconstruct(arg, idx)

    def argument(self, o):
        """Split an argument into its constituent spaces."""
        return split(o)

    def coefficient(self, o):
        """Split an argument into its constituent spaces."""
        return split(o)


class FFCKernel(DiskCached, KernelCached):

    _cache = {}
    _cachedir = os.path.join(tempfile.gettempdir(),
                             'pyop2-ffc-kernel-cache-uid%d' % os.getuid())

    @classmethod
    def _cache_key(cls, form, name):
        form_data = form.compute_form_data()
        return md5(form_data.signature + name + Kernel._backend.__name__ +
                   _pyop2_geometry_md5 + constants.FFC_VERSION +
                   constants.PYOP2_VERSION).hexdigest()

    def __init__(self, original_form, name):
        if self._initialized:
            return

        incl = PreprocessNode('#include "pyop2_geometry.h"\n')
        kernels = []
        # Note that split forms are batched by integral i.e. they will only
        # ever contain a single integral. We therefore always return the first
        # element of any lists that contain different integrals.
        for forms in FormSplitter().split(original_form):
            trees = [ffc_compile_form(form, prefix=name + str(i),
                                      parameters=ffc_parameters)[0]
                     for i, form in enumerate(forms)]
            from IPython import embed
            embed()

            ida = forms[0].form_data().integral_data[0]
            # Set optimization options
            opts = {} if ida.domain_type not in ['cell'] else \
                   {'licm': False,
                    'tile': None,
                    'vect': None,
                    'ap': False}

            def fname(name, ida, blkid=''):
                return '%s_%s_integral_0_%s' % (name + blkid, ida.domain_type,
                                                ida.domain_id)
            kernels.append(Kernel(Root([incl] + trees), fname(name, ida), opts,
                                  [fname(name, ida, str(i))
                                   for i, form in enumerate(forms)]))
        self.kernels = tuple(kernels)
        self._initialized = True


def compile_form(form, name):
    """Compile a form using FFC and return a :class:`pyop2.op2.Kernel`."""

    # Check that we get a Form
    if not isinstance(form, Form):
        form = as_form(form)

    return FFCKernel(form, name).kernels


def clear_cache():
    """Clear the PyOP2 FFC kernel cache."""
    if MPI.comm.rank != 0:
        return
    if os.path.exists(FFCKernel._cachedir):
        import shutil
        shutil.rmtree(FFCKernel._cachedir, ignore_errors=True)
        _ensure_cachedir()


def _ensure_cachedir():
    """Ensure that the FFC kernel cache directory exists."""
    if not os.path.exists(FFCKernel._cachedir) and MPI.comm.rank == 0:
        os.makedirs(FFCKernel._cachedir)

_check_version()
_ensure_cachedir()
