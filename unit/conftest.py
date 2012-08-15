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

"""Global test configuration."""

import pytest

from pyop2 import op2
from pyop2.backends import backends

def pytest_addoption(parser):
    parser.addoption("--backend", action="append",
        help="Selection the backend: one of %s" % backends.keys())

def pytest_collection_modifyitems(items):
    """Group test collection by backend instead of iterating through backends
    per test."""
    def get_backend_param(item):
        try:
            return item.callspec.getparam("backend")
        # AttributeError if no callspec, ValueError if no backend parameter
        except (AttributeError, ValueError):
            # If a test does not take the backend parameter, make sure it
            # is run before tests that take a backend
            return '_nobackend'
    items.sort(key=get_backend_param)

def pytest_funcarg__skip_cuda(request):
    return None

def pytest_funcarg__skip_opencl(request):
    return None

def pytest_funcarg__skip_sequential(request):
    return None

def pytest_generate_tests(metafunc):
    """Parametrize tests to run on all backends."""

    if 'backend' in metafunc.funcargnames:

        # Allow skipping individual backends by passing skip_<backend> as a parameter
        skip_backends = set()
        for b in backends.keys():
            if 'skip_'+b in metafunc.funcargnames:
                skip_backends.add(b)
        # Skip backends specified on the module level
        if hasattr(metafunc.module, 'skip_backends'):
            skip_backends = skip_backends.union(set(metafunc.module.skip_backends))
        # Skip backends specified on the class level
        if hasattr(metafunc.cls, 'skip_backends'):
            skip_backends = skip_backends.union(set(metafunc.cls.skip_backends))

        # Use only backends specified on the command line if any
        if metafunc.config.option.backend:
            backend = set(map(lambda x: x.lower(), metafunc.config.option.backend))
        # Otherwise use all available backends
        else:
            backend = set(backends.keys())
        # Restrict to set of backends specified on the module level
        if hasattr(metafunc.module, 'backends'):
            backend = backend.intersection(set(metafunc.module.backends))
        # Restrict to set of backends specified on the class level
        if hasattr(metafunc.cls, 'backends'):
            backend = backend.intersection(set(metafunc.cls.backends))

        selected_backends = backend.difference(skip_backends)
        # If there are no selected backends left, skip the test
        if not selected_backends:
            pytest.skip()
        # Otherwise, parametrize the backend
        metafunc.parametrize("backend", selected_backends, indirect=True)

        # If we only run OpenCL, run for all possible devices
        # Requires tests methods to take the ctx_factory, device or platform
        # arguments, see http://documen.tician.de/pyopencl/tools.html#testing
        if selected_backends == set(['opencl']):
            from pyopencl.tools import pytest_generate_tests_for_pyopencl
            pytest_generate_tests_for_pyopencl(metafunc)

def op2_init(backend):
    # We need to clean up the previous backend first, because the teardown
    # hook is only run at the end of the session
    op2.exit()
    op2.init(backend=backend)

def pytest_funcarg__backend(request):
    # If a testcase has the backend parameter but the parametrization leaves
    # i with no backends the request won't have a param, so return None
    if not hasattr(request, 'param'):
        return None
    # Call init/exit only once per session
    request.cached_setup(scope='session', setup=lambda: op2_init(request.param),
                         teardown=lambda backend: op2.exit(),
                         extrakey=request.param)
    return request.param
