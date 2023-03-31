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


from abc import ABC
import os
import platform
import shutil
import subprocess
import sys
import ctypes
import shlex
from hashlib import md5
from packaging.version import Version, InvalidVersion


from pyop2 import mpi
from pyop2.configuration import configuration
from pyop2.logger import warning, debug, progress, INFO
from pyop2.exceptions import CompilationError
from petsc4py import PETSc


def _check_hashes(x, y, datatype):
    """MPI reduction op to check if code hashes differ across ranks."""
    if x == y:
        return x
    return False


_check_op = mpi.MPI.Op.Create(_check_hashes, commute=True)
_compiler = None


def set_default_compiler(compiler):
    """Set the PyOP2 default compiler, globally.

    :arg compiler: String with name or path to compiler executable
        OR a subclass of the Compiler class
    """
    global _compiler
    if _compiler:
        warning(
            "`set_default_compiler` should only ever be called once, calling"
            " multiple times is untested and may produce unexpected results"
        )
    if isinstance(compiler, str):
        _compiler = sniff_compiler(compiler)
    elif isinstance(compiler, type) and issubclass(compiler, Compiler):
        _compiler = compiler
    else:
        raise TypeError(
            "compiler must be a path to a compiler (a string) or a subclass"
            " of the pyop2.compilation.Compiler class"
        )


def sniff_compiler(exe):
    """Obtain the correct compiler class by calling the compiler executable.

    :arg exe: String with name or path to compiler executable
    :returns: A compiler class
    """
    try:
        output = subprocess.run(
            [exe, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            encoding="utf-8"
        ).stdout
    except (subprocess.CalledProcessError, UnicodeDecodeError):
        output = ""

    # Find the name of the compiler family
    if output.startswith("gcc") or output.startswith("g++"):
        name = "GNU"
    elif output.startswith("clang"):
        name = "clang"
    elif output.startswith("Apple LLVM") or output.startswith("Apple clang"):
        name = "clang"
    elif output.startswith("icc"):
        name = "Intel"
    elif "Cray" in output.split("\n")[0]:
        # Cray is more awkward eg:
        # Cray clang version 11.0.4  (<some_hash>)
        # gcc (GCC) 9.3.0 20200312 (Cray Inc.)
        name = "Cray"
    else:
        name = "unknown"

    # Set the compiler instance based on the platform (and architecture)
    if sys.platform.find("linux") == 0:
        if name == "Intel":
            compiler = LinuxIntelCompiler
        elif name == "GNU":
            compiler = LinuxGnuCompiler
        elif name == "clang":
            compiler = LinuxClangCompiler
        elif name == "Cray":
            compiler = LinuxCrayCompiler
        else:
            compiler = AnonymousCompiler
    elif sys.platform.find("darwin") == 0:
        if name == "clang":
            machine = platform.uname().machine
            if machine == "arm64":
                compiler = MacClangARMCompiler
            elif machine == "x86_64":
                compiler = MacClangCompiler
        elif name == "GNU":
            compiler = MacGNUCompiler
        else:
            compiler = AnonymousCompiler
    else:
        compiler = AnonymousCompiler
    return compiler


def _check_src_hashes(comm, global_kernel):
    hsh = md5(str(global_kernel.cache_key[1:]).encode())
    basename = hsh.hexdigest()
    dirpart, basename = basename[:2], basename[2:]
    cachedir = configuration["cache_dir"]
    cachedir = os.path.join(cachedir, dirpart)

    if configuration["check_src_hashes"] or configuration["debug"]:
        matching = comm.allreduce(basename, op=_check_op)
        if matching != basename:
            # Dump all src code to disk for debugging
            output = os.path.join(cachedir, "mismatching-kernels")
            srcfile = os.path.join(output, "src-rank%d.c" % comm.rank)
            if comm.rank == 0:
                os.makedirs(output, exist_ok=True)
            comm.barrier()
            with open(srcfile, "w") as f:
                f.write(global_kernel.code_to_compile)
            comm.barrier()
            raise CompilationError("Generated code differs across ranks"
                                   f" (see output in {output})")


class Compiler(ABC):
    """A compiler for shared libraries.

    :arg extra_compiler_flags: A list of arguments to the C compiler (CFLAGS)
        or the C++ compiler (CXXFLAGS)
        (optional, prepended to any flags specified as the cflags configuration option).
        The environment variables ``PYOP2_CFLAGS`` and ``PYOP2_CXXFLAGS``
        can also be used to extend these options.
    :arg extra_linker_flags: A list of arguments to the linker (LDFLAGS)
    (optional, prepended to any flags specified as the ldflags configuration option).
        The environment variable ``PYOP2_LDFLAGS`` can also be used to
        extend these options.
    :arg cpp: Should we try and use the C++ compiler instead of the C
        compiler?.
    :kwarg comm: Optional communicator to compile the code on
        (defaults to pyop2.mpi.COMM_WORLD).
    """
    _name = "unknown"

    _cc = "mpicc"
    _cxx = "mpicxx"
    _ld = None

    _cflags = ()
    _cxxflags = ()
    _ldflags = ()

    _optflags = ()
    _debugflags = ()

    def __init__(self, extra_compiler_flags=(), extra_linker_flags=(), cpp=False, comm=None):
        # Get compiler version ASAP since it is used in __repr__
        self.sniff_compiler_version()

        self._extra_compiler_flags = tuple(extra_compiler_flags)
        self._extra_linker_flags = tuple(extra_linker_flags)

        self._cpp = cpp
        self._debug = configuration["debug"]

        # Compilation communicators are reference counted on the PyOP2 comm
        self.pcomm = mpi.internal_comm(comm)
        self.comm = mpi.compilation_comm(self.pcomm)

    def __del__(self):
        if hasattr(self, "comm"):
            mpi.decref(self.comm)
        if hasattr(self, "pcomm"):
            mpi.decref(self.pcomm)

    def __repr__(self):
        return f"<{self._name} compiler, version {self.version or 'unknown'}>"

    @property
    def cc(self):
        return configuration["cc"] or self._cc

    @property
    def cxx(self):
        return configuration["cxx"] or self._cxx

    @property
    def ld(self):
        return configuration["ld"] or self._ld

    @property
    def cflags(self):
        cflags = self._cflags + self._extra_compiler_flags + self.bugfix_cflags
        if self._debug:
            cflags += self._debugflags
        else:
            cflags += self._optflags
        cflags += tuple(shlex.split(configuration["cflags"]))
        return cflags

    @property
    def cxxflags(self):
        cxxflags = self._cxxflags + self._extra_compiler_flags + self.bugfix_cflags
        if self._debug:
            cxxflags += self._debugflags
        else:
            cxxflags += self._optflags
        cxxflags += tuple(shlex.split(configuration["cxxflags"]))
        return cxxflags

    @property
    def ldflags(self):
        ldflags = self._ldflags + self._extra_linker_flags
        ldflags += tuple(shlex.split(configuration["ldflags"]))
        return ldflags

    def sniff_compiler_version(self, cpp=False):
        """Attempt to determine the compiler version number.

        :arg cpp: If set to True will use the C++ compiler rather than
            the C compiler to determine the version number.
        """
        exe = self.cxx if cpp else self.cc
        self.version = None
        # `-dumpversion` is not sufficient to get the whole version string (for some compilers),
        # but other compilers do not implement `-dumpfullversion`!
        for dumpstring in ["-dumpfullversion", "-dumpversion"]:
            try:
                output = subprocess.run(
                    [exe, dumpstring],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    encoding="utf-8"
                ).stdout
                self.version = Version(output)
                break
            except (subprocess.CalledProcessError, UnicodeDecodeError, InvalidVersion):
                continue

    @property
    def bugfix_cflags(self):
        return ()

    @staticmethod
    def expandWl(ldflags):
        """Generator to expand the `-Wl` compiler flags for use as linker flags
        :arg ldflags: linker flags for a compiler command
        """
        for flag in ldflags:
            if flag.startswith('-Wl'):
                for f in flag.lstrip('-Wl')[1:].split(','):
                    yield f
            else:
                yield flag

    @mpi.collective
    def get_so(self, jitmodule, extension):
        """Build a shared library and load it

        :arg jitmodule: The JIT Module which can generate the code to compile.
        :arg extension: extension of the source file (c, cpp).
        Returns a :class:`ctypes.CDLL` object of the resulting shared
        library."""

        # C or C++
        if self._cpp:
            compiler = self.cxx
            compiler_flags = self.cxxflags
        else:
            compiler = self.cc
            compiler_flags = self.cflags

        # Determine cache key
        hsh = md5(str(jitmodule.cache_key).encode())
        hsh.update(compiler.encode())
        if self.ld:
            hsh.update(self.ld.encode())
        hsh.update("".join(compiler_flags).encode())
        hsh.update("".join(self.ldflags).encode())

        basename = hsh.hexdigest()

        cachedir = configuration['cache_dir']

        dirpart, basename = basename[:2], basename[2:]
        cachedir = os.path.join(cachedir, dirpart)
        pid = os.getpid()
        cname = os.path.join(cachedir, "%s_p%d.%s" % (basename, pid, extension))
        oname = os.path.join(cachedir, "%s_p%d.o" % (basename, pid))
        soname = os.path.join(cachedir, "%s.so" % basename)
        # Link into temporary file, then rename to shared library
        # atomically (avoiding races).
        tmpname = os.path.join(cachedir, "%s_p%d.so.tmp" % (basename, pid))

        _check_src_hashes(self.comm, jitmodule)

        try:
            # Are we in the cache?
            return ctypes.CDLL(soname)
        except OSError:
            # No, let's go ahead and build
            if self.comm.rank == 0:
                # No need to do this on all ranks
                os.makedirs(cachedir, exist_ok=True)
                logfile = os.path.join(cachedir, "%s_p%d.log" % (basename, pid))
                errfile = os.path.join(cachedir, "%s_p%d.err" % (basename, pid))
                with progress(INFO, 'Compiling wrapper'):
                    with open(cname, "w") as f:
                        f.write(jitmodule.code_to_compile)
                    # Compiler also links
                    if not self.ld:
                        cc = (compiler,) \
                            + compiler_flags \
                            + ('-o', tmpname, cname) \
                            + self.ldflags
                        debug('Compilation command: %s', ' '.join(cc))
                        with open(logfile, "w") as log, open(errfile, "w") as err:
                            log.write("Compilation command:\n")
                            log.write(" ".join(cc))
                            log.write("\n\n")
                            try:
                                if configuration['no_fork_available']:
                                    cc += ["2>", errfile, ">", logfile]
                                    cmd = " ".join(cc)
                                    status = os.system(cmd)
                                    if status != 0:
                                        raise subprocess.CalledProcessError(status, cmd)
                                else:
                                    subprocess.check_call(cc, stderr=err, stdout=log)
                            except subprocess.CalledProcessError as e:
                                raise CompilationError(
                                    """Command "%s" return error status %d.
Unable to compile code
Compile log in %s
Compile errors in %s""" % (e.cmd, e.returncode, logfile, errfile))
                    else:
                        cc = (compiler,) \
                            + compiler_flags \
                            + ('-c', '-o', oname, cname)
                        # Extract linker specific "cflags" from ldflags
                        ld = tuple(shlex.split(self.ld)) \
                            + ('-o', tmpname, oname) \
                            + tuple(self.expandWl(self.ldflags))
                        debug('Compilation command: %s', ' '.join(cc))
                        debug('Link command: %s', ' '.join(ld))
                        with open(logfile, "a") as log, open(errfile, "a") as err:
                            log.write("Compilation command:\n")
                            log.write(" ".join(cc))
                            log.write("\n\n")
                            log.write("Link command:\n")
                            log.write(" ".join(ld))
                            log.write("\n\n")
                            try:
                                if configuration['no_fork_available']:
                                    cc += ["2>", errfile, ">", logfile]
                                    ld += ["2>>", errfile, ">>", logfile]
                                    cccmd = " ".join(cc)
                                    ldcmd = " ".join(ld)
                                    status = os.system(cccmd)
                                    if status != 0:
                                        raise subprocess.CalledProcessError(status, cccmd)
                                    status = os.system(ldcmd)
                                    if status != 0:
                                        raise subprocess.CalledProcessError(status, ldcmd)
                                else:
                                    subprocess.check_call(cc, stderr=err, stdout=log)
                                    subprocess.check_call(ld, stderr=err, stdout=log)
                            except subprocess.CalledProcessError as e:
                                raise CompilationError(
                                    """Command "%s" return error status %d.
Unable to compile code
Compile log in %s
Compile errors in %s""" % (e.cmd, e.returncode, logfile, errfile))
                    # Atomically ensure soname exists
                    os.rename(tmpname, soname)
            # Wait for compilation to complete
            self.comm.barrier()
            # Load resulting library
            return ctypes.CDLL(soname)


class MacClangCompiler(Compiler):
    """A compiler for building a shared library on Mac systems."""
    _name = "Mac Clang"

    _cflags = ("-fPIC", "-Wall", "-framework", "Accelerate", "-std=gnu11")
    _cxxflags = ("-fPIC", "-Wall", "-framework", "Accelerate")
    _ldflags = ("-dynamiclib",)

    _optflags = ("-O3", "-ffast-math", "-march=native")
    _debugflags = ("-O0", "-g")


class MacClangARMCompiler(MacClangCompiler):
    """A compiler for building a shared library on ARM based Mac systems."""
    # See https://stackoverflow.com/q/65966969
    _optflags = ("-O3", "-ffast-math", "-mcpu=apple-a14")
    # Need to pass -L/opt/homebrew/opt/gcc/lib/gcc/11 to prevent linker error:
    # ld: file not found: @rpath/libgcc_s.1.1.dylib for architecture arm64 This
    # seems to be a homebrew configuration issue somewhere. Hopefully this
    # requirement will go away at some point.
    _ldflags = ("-dynamiclib", "-L/opt/homebrew/opt/gcc/lib/gcc/11")


class MacGNUCompiler(MacClangCompiler):
    """A compiler for building a shared library on Mac systems with a GNU compiler."""
    _name = "Mac GNU"


class LinuxGnuCompiler(Compiler):
    """The GNU compiler for building a shared library on Linux systems."""
    _name = "GNU"

    _cflags = ("-fPIC", "-Wall", "-std=gnu11")
    _cxxflags = ("-fPIC", "-Wall")
    _ldflags = ("-shared",)

    _optflags = ("-march=native", "-O3", "-ffast-math")
    _debugflags = ("-O0", "-g")

    def sniff_compiler_version(self, cpp=False):
        super(LinuxGnuCompiler, self).sniff_compiler_version()
        if self.version >= Version("7.0"):
            try:
                # gcc-7 series only spits out patch level on dumpfullversion.
                exe = self.cxx if cpp else self.cc
                output = subprocess.run(
                    [exe, "-dumpfullversion"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    encoding="utf-8"
                ).stdout
                self.version = Version(output)
            except (subprocess.CalledProcessError, UnicodeDecodeError, InvalidVersion):
                pass

    @property
    def bugfix_cflags(self):
        """Flags to work around bugs in compilers."""
        ver = self.version
        cflags = ()
        if Version("4.8.0") <= ver < Version("4.9.0"):
            # GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61068
            cflags = ("-fno-ivopts",)
        if Version("5.0") <= ver <= Version("5.4.0"):
            cflags = ("-fno-tree-loop-vectorize",)
        if Version("6.0.0") <= ver < Version("6.5.0"):
            # GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=79920
            cflags = ("-fno-tree-loop-vectorize",)
        if Version("7.1.0") <= ver < Version("7.1.2"):
            # GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81633
            cflags = ("-fno-tree-loop-vectorize",)
        if Version("7.3") <= ver <= Version("7.5"):
            # GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=90055
            # See also https://github.com/firedrakeproject/firedrake/issues/1442
            # And https://github.com/firedrakeproject/firedrake/issues/1717
            # Bug also on skylake with the vectoriser in this
            # combination (disappears without
            # -fno-tree-loop-vectorize!)
            cflags = ("-fno-tree-loop-vectorize", "-mno-avx512f")
        return cflags


class LinuxClangCompiler(Compiler):
    """The clang for building a shared library on Linux systems."""
    _name = "Clang"

    _ld = "ld.lld"

    _cflags = ("-fPIC", "-Wall", "-std=gnu11")
    _cxxflags = ("-fPIC", "-Wall")
    _ldflags = ("-shared", "-L/usr/lib")

    _optflags = ("-march=native", "-O3", "-ffast-math")
    _debugflags = ("-O0", "-g")


class LinuxIntelCompiler(Compiler):
    """The Intel compiler for building a shared library on Linux systems."""
    _name = "Intel"

    _cc = "mpiicc"
    _cxx = "mpiicpc"

    _cflags = ("-fPIC", "-no-multibyte-chars", "-std=gnu11")
    _cxxflags = ("-fPIC", "-no-multibyte-chars")
    _ldflags = ("-shared",)

    _optflags = ("-Ofast", "-xHost")
    _debugflags = ("-O0", "-g")


class LinuxCrayCompiler(Compiler):
    """The Cray compiler for building a shared library on Linux systems."""
    _name = "Cray"

    _cc = "cc"
    _cxx = "CC"

    _cflags = ("-fPIC", "-Wall", "-std=gnu11")
    _cxxflags = ("-fPIC", "-Wall")
    _ldflags = ("-shared",)

    _optflags = ("-march=native", "-O3", "-ffast-math")
    _debugflags = ("-O0", "-g")

    @property
    def ldflags(self):
        ldflags = super(LinuxCrayCompiler).ldflags
        if '-llapack' in ldflags:
            ldflags = tuple(flag for flag in ldflags if flag != '-llapack')
        return ldflags


class AnonymousCompiler(Compiler):
    """Compiler for building a shared library on systems with unknown compiler.
    The properties of this compiler are entirely controlled through environment
    variables"""
    _name = "Unknown"


@mpi.collective
def load(jitmodule, extension, fn_name, cppargs=(), ldargs=(),
         argtypes=None, restype=None, comm=None):
    """Build a shared library and return a function pointer from it.

    :arg jitmodule: The JIT Module which can generate the code to compile, or
        the string representing the source code.
    :arg extension: extension of the source file (c, cpp)
    :arg fn_name: The name of the function to return from the resulting library
    :arg cppargs: A tuple of arguments to the C compiler (optional)
    :arg ldargs: A tuple of arguments to the linker (optional)
    :arg argtypes: A list of ctypes argument types matching the arguments of
         the returned function (optional, pass ``None`` for ``void``). This is
         only used when string is passed in instead of JITModule.
    :arg restype: The return type of the function (optional, pass
         ``None`` for ``void``).
    :kwarg comm: Optional communicator to compile the code on (only
        rank 0 compiles code) (defaults to pyop2.mpi.COMM_WORLD).
    """
    from pyop2.global_kernel import GlobalKernel

    if isinstance(jitmodule, str):
        class StrCode(object):
            def __init__(self, code, argtypes):
                self.code_to_compile = code
                self.cache_key = (None, code)  # We peel off the first
                # entry, since for a jitmodule, it's a process-local
                # cache key
                self.argtypes = argtypes
        code = StrCode(jitmodule, argtypes)
    elif isinstance(jitmodule, GlobalKernel):
        code = jitmodule
    else:
        raise ValueError("Don't know how to compile code of type %r" % type(jitmodule))

    cpp = (extension == "cpp")
    global _compiler
    if _compiler:
        # Use the global compiler if it has been set
        compiler = _compiler
    else:
        # Sniff compiler from executable
        if cpp:
            exe = configuration["cxx"] or "mpicxx"
        else:
            exe = configuration["cc"] or "mpicc"
        compiler = sniff_compiler(exe)
    dll = compiler(cppargs, ldargs, cpp=cpp, comm=comm).get_so(code, extension)

    if isinstance(jitmodule, GlobalKernel):
        _add_profiling_events(dll, code.local_kernel.events)

    fn = getattr(dll, fn_name)
    fn.argtypes = code.argtypes
    fn.restype = restype
    return fn


def _add_profiling_events(dll, events):
    """
        If PyOP2 is in profiling mode, events are attached to dll to profile the local linear algebra calls.
        The event is generated here in python and then set in the shared library,
        so that memory is not allocated over and over again in the C kernel. The naming
        convention is that the event ids are named by the event name prefixed by "ID_".
    """
    if PETSc.Log.isActive():
        # also link the events from the linear algebra callables
        if hasattr(dll, "solve"):
            events += ('solve_memcpy', 'solve_getrf', 'solve_getrs')
        if hasattr(dll, "inverse"):
            events += ('inv_memcpy', 'inv_getrf', 'inv_getri')
        # link all ids in DLL to the events generated here in python
        for e in list(filter(lambda e: e is not None, events)):
            ctypes.c_int.in_dll(dll, 'ID_'+e).value = PETSc.Log.Event(e).id


def clear_cache(prompt=False):
    """Clear the PyOP2 compiler cache.

    :arg prompt: if ``True`` prompt before removing any files
    """
    cachedir = configuration['cache_dir']

    if not os.path.exists(cachedir):
        print("Cache directory could not be found")
        return
    if len(os.listdir(cachedir)) == 0:
        print("No cached libraries to remove")
        return

    remove = True
    if prompt:
        user = input(f"Remove cached libraries from {cachedir}? [Y/n]: ")

        while user.lower() not in ['', 'y', 'n']:
            print("Please answer y or n.")
            user = input(f"Remove cached libraries from {cachedir}? [Y/n]: ")

        if user.lower() == 'n':
            remove = False

    if remove:
        print(f"Removing cached libraries from {cachedir}")
        shutil.rmtree(cachedir)
    else:
        print("Not removing cached libraries")


def _get_code_to_compile(comm, global_kernel):
    # Determine cache key
    hsh = md5(str(global_kernel.cache_key[1:]).encode())
    basename = hsh.hexdigest()
    cachedir = configuration["cache_dir"]
    dirpart, basename = basename[:2], basename[2:]
    cachedir = os.path.join(cachedir, dirpart)
    cname = os.path.join(cachedir, f"{basename}_code.cu")

    _check_src_hashes(comm, global_kernel)

    if os.path.isfile(cname):
        # Are we in the cache?
        with open(cname, "r") as f:
            code_to_compile = f.read()
    else:
        # No, let"s go ahead and build
        if comm.rank == 0:
            # No need to do this on all ranks
            os.makedirs(cachedir, exist_ok=True)
            with progress(INFO, "Compiling wrapper"):
                # make sure that compiles successfully before writing to file
                code_to_compile = global_kernel.code_to_compile
                with open(cname, "w") as f:
                    f.write(code_to_compile)
        comm.barrier()

    return code_to_compile


@mpi.collective
def get_prepared_cuda_function(comm, global_kernel):
    from pycuda.compiler import SourceModule

    # Determine cache key
    hsh = md5(str(global_kernel.cache_key[1:]).encode())
    basename = hsh.hexdigest()
    cachedir = configuration["cache_dir"]
    dirpart, basename = basename[:2], basename[2:]
    cachedir = os.path.join(cachedir, dirpart)

    nvcc_opts = ["-use_fast_math", "-w"]

    code_to_compile = _get_code_to_compile(comm, global_kernel)
    source_module = SourceModule(code_to_compile, options=nvcc_opts,
                                 cache_dir=cachedir)

    cu_func = source_module.get_function(global_kernel.name)

    type_map = {ctypes.c_void_p: "P", ctypes.c_int: "i"}
    argtypes = "".join(type_map[t] for t in global_kernel.argtypes)
    cu_func.prepare(argtypes)

    return cu_func


@mpi.collective
def get_opencl_kernel(comm, global_kernel):
    import pyopencl as cl
    from pyop2.backends.opencl import opencl_backend
    cl_ctx = opencl_backend.context

    # Determine cache key
    hsh = md5(str(global_kernel.cache_key[1:]).encode())
    basename = hsh.hexdigest()
    cachedir = configuration["cache_dir"]
    dirpart, basename = basename[:2], basename[2:]
    cachedir = os.path.join(cachedir, dirpart)

    code_to_compile = _get_code_to_compile(comm, global_kernel)

    prg = cl.Program(cl_ctx, code_to_compile).build(options=[],
                                                    cache_dir=cachedir)

    cl_knl = cl.Kernel(prg, global_kernel.name)
    return cl_knl
