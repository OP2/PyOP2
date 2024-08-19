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
from textwrap import dedent
from functools import partial


from pyop2 import mpi
from pyop2.caching import memory_cache, default_parallel_hashkey
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
    """Set the PyOP2 default compiler, globally over COMM_WORLD.

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


def sniff_compiler_version(compiler, cpp=False):
    """Attempt to determine the compiler version number.

    :arg compiler: Instance of compiler to sniff the version of
    :arg cpp: If set to True will use the C++ compiler rather than
        the C compiler to determine the version number.
    """
    # Note:
    # Sniffing the compiler version for very large numbers of
    # MPI ranks is expensive, ensure this is only run on rank 0
    exe = compiler.cxx if cpp else compiler.cc
    version = None
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
            version = Version(output)
            break
        except (subprocess.CalledProcessError, UnicodeDecodeError, InvalidVersion):
            continue
    return version


def sniff_compiler(exe, comm=mpi.COMM_WORLD):
    """Obtain the correct compiler class by calling the compiler executable.

    :arg exe: String with name or path to compiler executable
    :arg comm: Comm over which we want to determine the compiler type
    :returns: A compiler class
    """
    compiler = None
    if comm.rank == 0:
        # Note:
        # Sniffing compiler for very large numbers of MPI ranks is
        # expensive so we do this on one rank and broadcast
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

        # Now try and get a version number
        temp = Compiler()
        version = sniff_compiler_version(temp)
        compiler = partial(compiler, version=version)

    return comm.bcast(compiler, 0)


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

    def __init__(self, extra_compiler_flags=(), extra_linker_flags=(), cpp=False, version=None):
        self.version = version

        self._extra_compiler_flags = tuple(extra_compiler_flags)
        self._extra_linker_flags = tuple(extra_linker_flags)

        self._cpp = cpp
        self._debug = configuration["debug"]

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

    @property
    def bugfix_cflags(self):
        return ()


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


def load_hashkey(*args, **kwargs):
    from pyop2.global_kernel import GlobalKernel
    if isinstance(args[0], str):
        code_hash = md5(args[0].encode()).hexdigest()
    elif isinstance(args[0], GlobalKernel):
        code_hash = md5(str(args[0].cache_key).encode()).hexdigest()
    else:
        pass  # This will raise an error in load
    return default_parallel_hashkey(code_hash, *args[1:], **kwargs)


@mpi.collective
@memory_cache(hashkey=load_hashkey, broadcast=False)
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
        compiler = sniff_compiler(exe, comm)

    compiler_instance = compiler(cppargs, ldargs, cpp=cpp)
    dll = make_so(compiler_instance, code, extension, comm)

    if isinstance(jitmodule, GlobalKernel):
        _add_profiling_events(dll, code.local_kernel.events)

    fn = getattr(dll, fn_name)
    fn.argtypes = code.argtypes
    fn.restype = restype
    return fn


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
def make_so(compiler, jitmodule, extension, comm):
    """Build a shared library and load it

    :arg compiler: The compiler to use to create the shared library.
    :arg jitmodule: The JIT Module which can generate the code to compile.
    :arg extension: extension of the source file (c, cpp).
    :arg comm: Communicator over which to perform compilation.
    Returns a :class:`ctypes.CDLL` object of the resulting shared
    library."""
    # Compilation communicators are reference counted on the PyOP2 comm
    pcomm = mpi.internal_comm(comm, compiler)
    comm = mpi.compilation_comm(pcomm, compiler)

    # C or C++
    if compiler._cpp:
        exe = compiler.cxx
        compiler_flags = compiler.cxxflags
    else:
        exe = compiler.cc
        compiler_flags = compiler.cflags

    # Determine cache key
    hsh = md5(str(jitmodule.cache_key).encode())
    hsh.update(exe.encode())
    if compiler.ld:
        hsh.update(compiler.ld.encode())
    hsh.update("".join(compiler_flags).encode())
    hsh.update("".join(compiler.ldflags).encode())

    basename = hsh.hexdigest()  # This is hash key

    cachedir = configuration['cache_dir']  # This is cachedir

    dirpart, basename = basename[:2], basename[2:]
    cachedir = os.path.join(cachedir, dirpart)
    pid = os.getpid()
    cname = os.path.join(cachedir, f"{basename}_p{pid}.{extension}")
    oname = os.path.join(cachedir, f"{basename}_p{pid}.o")
    soname = os.path.join(cachedir, f"{basename}.so")
    # Link into temporary file, then rename to shared library
    # atomically (avoiding races).
    tmpname = os.path.join(cachedir, f"{basename}_p{pid}.so.tmp")

    if configuration['check_src_hashes'] or configuration['debug']:
        matching = comm.allreduce(basename, op=_check_op)
        if matching != basename:
            # Dump all src code to disk for debugging
            output = os.path.join(configuration["cache_dir"], "mismatching-kernels")
            srcfile = os.path.join(output, f"src-rank{comm.rank}.{extension}")
            if comm.rank == 0:
                os.makedirs(output, exist_ok=True)
            comm.barrier()
            with open(srcfile, "w") as f:
                f.write(jitmodule.code_to_compile)
            comm.barrier()
            raise CompilationError(f"Generated code differs across ranks (see output in {output})")

    # Check whether this shared object already written to disk
    try:
        dll = ctypes.CDLL(soname)
    except OSError:
        dll = None
    got_dll = bool(dll)
    all_dll = comm.allgather(got_dll)

    # If the library is not loaded _on all ranks_ build it
    if not min(all_dll):
        if comm.rank == 0:
            # No need to do this on all ranks
            os.makedirs(cachedir, exist_ok=True)
            logfile = os.path.join(cachedir, f"{basename}_p{pid}.log")
            errfile = os.path.join(cachedir, f"{basename}_p{pid}.err")
            with progress(INFO, 'Compiling wrapper'):
                with open(cname, "w") as f:
                    f.write(jitmodule.code_to_compile)
                # Compiler also links
                if not compiler.ld:
                    cc = (exe,) \
                        + compiler_flags \
                        + ('-o', tmpname, cname) \
                        + compiler.ldflags
                    debug(f"Compilation command: {' '.join(cc)}")
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
                            raise CompilationError(dedent(f"""
                                Command "{e.cmd}" return error status {e.returncode}.
                                Unable to compile code
                                Compile log in {logfile}
                                Compile errors in {errfile}
                                """))
                else:
                    cc = (exe,) \
                        + compiler_flags \
                        + ('-c', '-o', oname, cname)
                    # Extract linker specific "cflags" from ldflags
                    ld = tuple(shlex.split(compiler.ld)) \
                        + ('-o', tmpname, oname) \
                        + tuple(expandWl(compiler.ldflags))
                    debug(f"Compilation command: {' '.join(cc)}", )
                    debug(f"Link command: {' '.join(ld)}")
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
                            raise CompilationError(dedent(f"""
                                    Command "{e.cmd}" return error status {e.returncode}.
                                    Unable to compile code
                                    Compile log in {logfile}
                                    Compile errors in {errfile}
                                    """))
                # Atomically ensure soname exists
                os.rename(tmpname, soname)
        # Wait for compilation to complete
        comm.barrier()
        # Load resulting library
        dll = ctypes.CDLL(soname)
    return dll


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
        shutil.rmtree(cachedir, ignore_errors=True)
    else:
        print("Not removing cached libraries")
