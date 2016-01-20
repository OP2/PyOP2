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

"""PyOP2 global configuration."""

import os
from tempfile import gettempdir

from exceptions import ConfigurationError
from contextlib import contextmanager


class Configuration(dict):
    """PyOP2 configuration parameters

    :param backend: Select the PyOP2 backend (one of `cuda`,
        `opencl`, `openmp` or `sequential`).
    :param debug: Turn on debugging for generated code (turns off
        compiler optimisations).
    :param type_check: Should PyOP2 type-check API-calls?  (Default,
        yes)
    :param check_src_hashes: Should PyOP2 check that generated code is
        the same on all processes?  (Default, no).
    :param log_level: How chatty should PyOP2 be?  Valid values
        are "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
    :param lazy_evaluation: Should lazy evaluation be on or off?
    :param lazy_max_trace_length: How many :func:`par_loop`\s
        should be queued lazily before forcing evaluation?  Pass
        `0` for an unbounded length.
    :param loop_fusion: Should loop fusion be on or off?
    :param dump_gencode: Should PyOP2 write the generated code
        somewhere for inspection?
    :param dump_gencode_path: Where should the generated code be
        written to?
    :param print_cache_size: Should PyOP2 print the size of caches at
        program exit?
    :param print_summary: Should PyOP2 print a summary of timings at
        program exit?
    :param profiling: Profiling mode (CUDA kernels are launched synchronously)
    :param matnest: Should matrices on mixed maps be built as nests? (Default yes)
    """
    # name, env variable, type, default, write once
    DEFAULTS = {
        # Enable profiling of the wrapper functions.
        "hpc_profiling": ("PYOP2_HPC_PROFILING", bool, False),
        # Enable debugging of the wrapper functions.
        "hpc_debug": ("PYOP2_HPC_DEBUG", bool, False),
        # Enable optimization
        "hpc_optimize": ("PYOP2_HPC_OPTIMIZE", bool, False),
        # Enable a specific type of code gen
        "hpc_code_gen": ("PYOP2_HPC_CODE_GEN", int, 1),
        # Enable saving the result
        "hpc_save_result": ("PYOP2_HPC_SAVE_RESULT", bool, False),
        # Enable checking the result, compare it with the saved result
        "hpc_check_result": ("PYOP2_HPC_CHECK_RESULT", bool, False),
        # Enable checking the result, compare it with the saved result
        "hpc_save_order": ("PYOP2_HPC_ORDER", int, 0),
        "hpc_check_order": ("PYOP2_HPC_ORDER", int, 0),
        # Teams and threads for the OpenMP 4.0 backend
        "teams": ("PYOP2_OMP4_TEAMS", int, 256),
        "threads": ("PYOP2_OMP4_THREADS", int, 512),
        "output_llvm": ("PYOP2_OUTPUT_LLVM", bool, False),
        # File basename
        "basename": ("PYOP2_CODE_BASENAME", str, "default_name"),
        "nvlink_info": ("PYOP2_OMP4_NVLINK", str, "default"),
        # Turn on likwid. TODO: make it true when
        # either inner or outer likwid flags are true
        "likwid": ("PYOP2_LIKWID", bool, False),
        # add likwid instrumentation for the kernel
        # can be enbaled or disabled at any point in the code
        "likwid_inner": ("PYOP2_LIKWID_INNER", bool, False),
        # add likwid instrumentation for the wrapper (includes kernel too)
        # can be enbaled or disabled at any point in the code
        "likwid_outer": ("PYOP2_LIKWID_OUTER", bool, False),
        # Give a suitable name to the region we want to measure
        "region_name": ("PYOP2_REGION_NAME", str, "default"),
        # Measure the time around the kernel only
        "only_kernel": ("PYOP2_ONLY_KERNEL", bool, False),
        # For a given code region only report the indirect loops
        "only_indirect_loops": ("PYOP2_ONLY_INDIRECT_LOOPS", bool, True),
        # For a given code region only report the indirect loops
        "papi_flops": ("PYOP2_PAPI_FLOPS", bool, False),
        # For extruded meshes: horizontally DG dofs per column
        "dg_dpc": ("PYOP2_DG_DPC", int, 0),
        # For extruded meshes: DG coords correction term for MBW
        "dg_coords": ("PYOP2_DG_COORDS", int, 0),
        # Randomize the mesh by mixing the maps
        "randomize": ("PYOP2_RANDOMIZE", bool, False),
        # Number of times the wrapper code is being run. This is for testing only.
        "times": ("PYOP2_TIMES", int, 1),
        # Enable intel IACA instrumentation
        "iaca": ("PYOP2_IACA", bool, False),

        "backend": ("PYOP2_BACKEND", str, "sequential"),
        "compiler": ("PYOP2_BACKEND_COMPILER", str, "gnu"),
        "simd_isa": ("PYOP2_SIMD_ISA", str, "sse"),
        "blas": ("PYOP2_BLAS", str, ""),
        "debug": ("PYOP2_DEBUG", int, 0),
        "type_check": ("PYOP2_TYPE_CHECK", bool, True),
        "check_src_hashes": ("PYOP2_CHECK_SRC_HASHES", bool, False),
        "log_level": ("PYOP2_LOG_LEVEL", (str, int), "WARNING"),
        "lazy_evaluation": ("PYOP2_LAZY", bool, True),
        "lazy_max_trace_length": ("PYOP2_MAX_TRACE_LENGTH", int, 100),
        "loop_fusion": ("PYOP2_LOOP_FUSION", bool, False),
        "dump_gencode": ("PYOP2_DUMP_GENCODE", bool, False),
        "cache_dir": ("PYOP2_CACHE_DIR", str,
                      os.path.join(gettempdir(),
                                   "pyop2-cache-uid%s" % os.getuid())),
        "no_fork_available": ("PYOP2_NO_FORK_AVAILABLE", bool, False),
        "print_cache_size": ("PYOP2_PRINT_CACHE_SIZE", bool, False),
        "print_summary": ("PYOP2_PRINT_SUMMARY", bool, False),
        "profiling": ("PYOP2_PROFILING", bool, False),
        "dump_gencode_path": ("PYOP2_DUMP_GENCODE_PATH", str,
                              os.path.join(gettempdir(), "pyop2-gencode")),
        "matnest": ("PYOP2_MATNEST", bool, True),
    }
    """Default values for PyOP2 configuration parameters"""
    READONLY = ['backend']
    """List of read-only configuration keys."""

    def __init__(self):
        def convert(env, typ, v):
            if not isinstance(typ, type):
                typ = typ[0]
            try:
                if typ is bool:
                    return bool(int(os.environ.get(env, v)))
                return typ(os.environ.get(env, v))
            except ValueError:
                raise ValueError("Cannot convert value of environment variable %s to %r" % (env, typ))
        defaults = dict((k, convert(env, typ, v))
                        for k, (env, typ, v) in Configuration.DEFAULTS.items())
        super(Configuration, self).__init__(**defaults)
        self._set = set()
        self._defaults = defaults

    def reset(self):
        """Reset the configuration parameters to the default values."""
        self.update(self._defaults)
        self._set = set()

    def reconfigure(self, **kwargs):
        """Update the configuration parameters with new values."""
        for k, v in kwargs.items():
            self[k] = v

    def unsafe_reconfigure(self, **kwargs):
        """"Unsafely reconfigure (just replacing the values)"""
        self.update(kwargs)

    def __setitem__(self, key, value):
        """Set the value of a configuration parameter.

        :arg key: The parameter to set
        :arg value: The value to set it to.

        .. note::
           Some configuration parameters are read-only in which case
           attempting to set them raises an error, see
           :attr:`Configuration.READONLY` for details of which.
        """
        if key in Configuration.READONLY and key in self._set and value != self[key]:
            raise ConfigurationError("%s is read only" % key)
        if key in Configuration.DEFAULTS:
            valid_type = Configuration.DEFAULTS[key][1]
            if not isinstance(value, valid_type):
                raise ConfigurationError("Values for configuration key %s must be of type %r, not %r"
                                         % (key, valid_type, type(value)))
        self._set.add(key)
        super(Configuration, self).__setitem__(key, value)

configuration = Configuration()


@contextmanager
def configure(flag, value):
    old_value = configuration[flag]
    configuration[flag] = value
    if configuration["likwid"]:
        import likwid
        likwid.initialise()
    yield
    if configuration["likwid"]:
        import likwid
        likwid.finalise()
    configuration[flag] = old_value
