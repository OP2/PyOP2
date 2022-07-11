import abc
from dataclasses import dataclass
import hashlib
from typing import Union

import coffee
from coffee.visitors import EstimateFlops
import loopy as lp
from loopy.tools import LoopyKeyBuilder
import numpy as np

from pyop2 import version
from pyop2.datatypes import ScalarType
from pyop2.exceptions import NameTypeError
from pyop2.types import Access
from pyop2.utils import cached_property, validate_type
from pyop2.mpi import COMM_WORLD
from pyop2.configuration import configuration


@dataclass(frozen=True)
class LocalKernelArg:
    """Class representing a kernel argument.

    :param access: Access descriptor for the argument.
    :param dtype: The argument's datatype.
    """

    access: Access
    dtype: Union[np.dtype, str]


@validate_type(("name", str, NameTypeError))
def Kernel(code, name, **kwargs):
    """Construct a local kernel.

    For a description of the arguments to this function please see :class:`LocalKernel`.
    """
    if isinstance(code, str):
        return CStringLocalKernel(code, name, **kwargs)
    elif isinstance(code, coffee.base.Node):
        return CoffeeLocalKernel(code, name, **kwargs)
    elif isinstance(code, (lp.LoopKernel, lp.TranslationUnit)):
        return LoopyLocalKernel(code, name, **kwargs)
    else:
        raise TypeError("code argument is the wrong type")


class LocalKernel(abc.ABC):
    """Class representing the kernel executed per member of the iterset.

    :arg code: Function definition (including signature).
    :arg name: The kernel name. This must match the name of the kernel
        function given in `code`.
    :arg accesses: Optional iterable of :class:`Access` instances describing
        how each argument in the function definition is accessed.

    :kwarg cpp: Is the kernel actually C++ rather than C?  If yes,
        then compile with the C++ compiler (kernel is wrapped in
        extern C for linkage reasons).
    :kwarg flop_count: The number of FLOPs performed by the kernel.
    :kwarg headers: list of system headers to include when compiling the kernel
        in the form ``#include <header.h>`` (optional, defaults to empty)
    :kwarg include_dirs: list of additional include directories to be searched
        when compiling the kernel (optional, defaults to empty)
    :kwarg ldargs: A list of arguments to pass to the linker when
        compiling this Kernel.
    :kwarg opts: An options dictionary for declaring optimisations to apply.
    :kwarg requires_zeroed_output_arguments: Does this kernel require the
        output arguments to be zeroed on entry when called? (default no)
    :kwarg user_code: code snippet to be executed once at the very start of
        the generated kernel wrapper code (optional, defaults to
        empty)
    :kwarg events: Tuple of log event names which are called in the C code of the local kernels

    Consider the case of initialising a :class:`~pyop2.Dat` with seeded random
    values in the interval 0 to 1. The corresponding :class:`~pyop2.Kernel` is
    constructed as follows: ::

      op2.CStringKernel("void setrand(double *x) { x[0] = (double)random()/RAND_MAX); }",
                        name="setrand",
                        headers=["#include <stdlib.h>"], user_code="srandom(10001);")

    .. note::
        When running in parallel with MPI the generated code must be the same
        on all ranks.
    """

    @validate_type(("name", str, NameTypeError))
    def __init__(self, code, name, accesses=None, *,
                 cpp=False,
                 flop_count=None,
                 headers=(),
                 include_dirs=(),
                 ldargs=(),
                 opts=None,
                 requires_zeroed_output_arguments=False,
                 user_code="",
                 events=()):
        self.code = code
        self.name = name
        self.accesses = accesses
        self.cpp = cpp
        self.flop_count = flop_count
        self.headers = headers
        self.include_dirs = include_dirs
        self.ldargs = ldargs
        self.opts = opts or {}
        self.requires_zeroed_output_arguments = requires_zeroed_output_arguments
        self.user_code = user_code
        self.events = events

    @property
    @abc.abstractmethod
    def dtypes(self):
        """Return the dtypes of the arguments to the kernel."""

    @property
    def cache_key(self):
        return self._immutable_cache_key, self.accesses, self.dtypes

    @cached_property
    def _immutable_cache_key(self):
        # We need this function because self.accesses is mutable due to legacy support
        if isinstance(self.code, coffee.base.Node):
            code = self.code.gencode()
        elif isinstance(self.code, lp.TranslationUnit):
            key_hash = hashlib.sha256()
            self.code.update_persistent_hash(key_hash, LoopyKeyBuilder())
            code = key_hash.hexdigest()
        else:
            code = self.code

        key = (code, self.name, self.cpp, self.flop_count,
               self.headers, self.include_dirs, self.ldargs, sorted(self.opts.items()),
               self.requires_zeroed_output_arguments, self.user_code, version.__version__)
        return hashlib.md5(str(key).encode()).hexdigest()

    @property
    def _wrapper_cache_key_(self):
        import warnings
        warnings.warn("_wrapper_cache_key is deprecated, use cache_key instead", DeprecationWarning)

        return self.cache_key

    @property
    def arguments(self):
        """Return an iterable of :class:`LocalKernelArg` instances representing
        the arguments expected by the kernel.
        """
        assert len(self.accesses) == len(self.dtypes)

        return tuple(LocalKernelArg(acc, dtype)
                     for acc, dtype in zip(self.accesses, self.dtypes))

    @cached_property
    @abc.abstractmethod
    def num_flops(self):
        """Compute the numbers of FLOPs if not already known."""

    def __eq__(self, other):
        if not isinstance(other, LocalKernel):
            return NotImplemented
        else:
            return self.cache_key == other.cache_key

    def __hash__(self):
        return hash(self.cache_key)

    def __str__(self):
        return f"OP2 Kernel: {self.name}"

    def __repr__(self):
        return 'Kernel("""%s""", %r)' % (self.code, self.name)


class CStringLocalKernel(LocalKernel):
    """:class:`LocalKernel` class where `code` is a string of C code.

    :kwarg dtypes: Iterable of datatypes (either `np.dtype` or `str`) for
        each kernel argument. This is not required for :class:`CoffeeLocalKernel`
        or :class:`LoopyLocalKernel` because it can be inferred.

    All other `__init__` parameters are the same.
    """

    @validate_type(("code", str, TypeError))
    def __init__(self, code, name, accesses=None, dtypes=None, **kwargs):
        super().__init__(code, name, accesses, **kwargs)
        self._dtypes = dtypes

    @property
    def dtypes(self):
        return self._dtypes

    @dtypes.setter
    def dtypes(self, dtypes):
        self._dtypes = dtypes

    @cached_property
    def num_flops(self):
        if self.flop_count is not None:
            return self.flop_count
        else:
            return 0


class CoffeeLocalKernel(LocalKernel):
    """:class:`LocalKernel` class where `code` has type :class:`coffee.base.Node`."""

    @validate_type(("code", coffee.base.Node, TypeError))
    def __init__(self, code, name, accesses=None, dtypes=None, **kwargs):
        super().__init__(code, name, accesses, **kwargs)
        self._dtypes = dtypes

    @property
    def dtypes(self):
        return self._dtypes

    @dtypes.setter
    def dtypes(self, dtypes):
        self._dtypes = dtypes

    @cached_property
    def num_flops(self):
        if self.flop_count is not None:
            return self.flop_count
        else:
            v = EstimateFlops()
            return v.visit(self.code)


class LoopyLocalKernel(LocalKernel):
    """:class:`LocalKernel` class where `code` has type :class:`loopy.LoopKernel`
        or :class:`loopy.TranslationUnit`.
    """

    @validate_type(("code", (lp.LoopKernel, lp.TranslationUnit), TypeError))
    def __init__(self, code, *args, **kwargs):
        super().__init__(code, *args, **kwargs)

    @property
    def dtypes(self):
        return tuple(a.dtype for a in self._loopy_arguments)

    @cached_property
    def _loopy_arguments(self):
        """Return the loopy arguments associated with the kernel."""
        return tuple(a for a in self.code.callables_table[self.name].subkernel.args
                     if isinstance(a, lp.ArrayArg))

    @cached_property
    def num_flops(self):
        if self.flop_count is not None:
            return self.flop_count
        else:
            if isinstance(self.code, lp.TranslationUnit):
                prog = self.code.with_entrypoints(self.name)
                knl = prog.default_entrypoint
                warnings = list(knl.silenced_warnings)
                warnings.extend(['insn_count_subgroups_upper_bound',
                                 'get_x_map_guessing_subgroup_size',
                                 'summing_if_branches_ops'])
                knl = knl.copy(silenced_warnings=warnings,
                               options=lp.Options(ignore_boostable_into=True))
                knl = lp.fix_parameters(knl, layer=1)
                prog = prog.with_kernel(knl)
            else:
                prog = self.code
            op_map = lp.get_op_map(prog, subgroup_size=1)

            flops =  op_map.filter_by(name=['add', 'sub', 'mul', 'div'],
                                      dtype=[ScalarType]).eval_and_sum({})
            if (configuration["dump_slate_flops"] and
                COMM_WORLD.rank == 0 and "slate" in self.name):
                with open(configuration["dump_slate_flops"] , 'a+') as txt_file:
                    txt_file.write(f'Flops of {self.name}: {flops}\n')

            return flops
