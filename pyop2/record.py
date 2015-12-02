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

# These descriptors return the values scaled to one process or run
AVG = lambda x: sum(x) * 1.0 / len(x) if len(x) > 0 else 0.0
MIN = lambda x: min(x) if len(x) > 0 else 0.0
MAX = lambda x: max(x) if len(x) > 0 else 0.0

# These descritpros return the values scaled to the number of processors or runs
SUM = lambda x: sum(x)
MIN_SUM = lambda x: min(x) if len(x) > 0 else 0.0
MAX_SUM = lambda x: max(x) if len(x) > 0 else 0.0


class ReductionRecord(object):

    """
    The current reduction record implementation supports profiling of extruded meshes
    where a loop over the cells in column exists.

    NOTE: The current implenentation has not been tested on non-extruded meshes.

    This keeps track of the values obtained by profiling the same code over several
    MPI processes. Each MPI process will generate a contribution to one of the parameters
    which are recorded in this data structure.

    For each parameter there exists a reduction descriptor: min, max, sum or avg depending on which
    reduction operation we want to perform. Each paramter will have a default reduction desriptor.
    """
    folds = {}

    def __init__(self,
                 threads=None,
                 layers=None,
                 space_name=None,
                 fold_bw=SUM,
                 fold_time=MIN,
                 fold_stats=SUM,
                 is_proc=False):
        # Name of the parallel loop
        self._name = None
        # Important data for the plots
        self._threads = threads
        self._layers = layers
        self._space_name = space_name
        # Reduction descriptors
        self._fold_bw = fold_bw
        self._fold_time = fold_time
        self._fold_stats = fold_stats
        # If the reduction is within one process then is_proc is True
        self._is_proc = is_proc

        # Python measures runtime
        self._python_time = []
        self.folds["python_time"] = fold_time
        # Valuable Data Volume
        self._v_volume = []
        self.folds["v_volume"] = fold_bw
        # Valuable Data Volume with increments counted twice
        self._mv_volume = []
        self.folds["mv_volume"] = fold_bw
        # Valuable Data Volume with no re-use
        self._m_volume = []
        self.folds["m_volume"] = fold_bw
        # Start of the first and end of the last process runtime calculation
        self._start_time = []
        self.folds["start_time"] = MIN
        self._end_time = []
        self.folds["end_time"] = MAX
        self._runtime = None
        # C runtimes per process
        self._c_runtime = []
        self.folds["c_runtime"] = fold_time
        # C runtimes of the un-ordered (reordered) mesh
        # Start of the first and end of the last process runtime calculation
        self._rv_start_time = []
        self.folds["rv_start_time"] = MIN
        self._rv_end_time = []
        self.folds["rv_end_time"] = MAX
        self._rv_runtime = None
        # Un-ordered runtime per process
        self._rv_c_runtime = []
        self.folds["rv_c_runtime"] = fold_time
        # Wrapper flops: PAPI instrumentation is at the beginning and end of the wrapper
        self._papi_flops = []
        self.folds["papi_flops"] = fold_stats
        # Floating point operations of the loop over columns (extruded mesh)
        # Measured using the IACA tool
        self._iaca_flops = []
        self.folds["iaca_flops"] = fold_stats
        # IACA reported cycles for the loop over the columns (extruded mesh)
        self._cycles = []
        self.folds["cycles"] = SUM if self._is_proc else AVG
        # Teams and thread_limit
        self._teams = []
        self._thread_limit = []
        # NVLINK info
        self._nvlink_info = []

    def _ncalls(self):
        """Number of calls per process"""
        return len(self._c_runtime)

    def _reduce(self, attr, values):
        """Internal function for performing different types of reductions."""
        if self.folds[attr] in [MIN_SUM, MAX_SUM]:
            return self._ncalls() * self.folds[attr](values)
        else:
            return self.folds[attr](values)

    def reset(self):
        """Reset timers."""
        self._python_time = []
        self._c_runtime = []
        self._rv_runtime = []
        self._v_volume = []
        self._mv_volume = []
        self._m_volume = []
        self._start_time = []
        self._end_time = []
        self._rv_start_time = []
        self._rv_end_time = []
        self._iaca_flops = []
        self._papi_flops = []
        self._cycles = []
        self._teams = []
        self._thread_limit = []
        self._nvlink_info = []

    def isEmpty(self):
        return len(self._c_runtime) == 0

    def output_line(self):
        """ Old style output line for reference.
        All the output will now be saved using the
        compact_line format.
        """
        str_line = " | ".join(["%%%ds" % (15)] + ["%%%dg" % (15) for i in range(13)] +
                              ["%%%d.%dg" % (15, 15) for i in range(3)] +
                              ["%d" for i in range(2)] +
                              ["%s"]) + "\n"
        values = (self._name,
                  self.python_time,
                  len(self._python_time),
                  sum(self._python_time) / len(self._python_time),
                  -1,
                  sum(self._c_runtime),
                  sum(self._c_runtime) / len(self._c_runtime),
                  self.mv_volume,
                  -2, -1,
                  self.vbw, self.mbw, self.mvbw, self.rvbw,
                  self.start_time, self.end_time, self.iaca_flops,
                  self.teams, self.thread_limit,
                  self.nvlink_info)
        return str_line % values

    def compact_line(self):
        values = (self._name,
                  self._ncalls(),
                  sum(self._c_runtime),
                  self.c_runtime,
                  self.v_volume, self.m_volume, self.mv_volume,
                  self.vbw, self.mbw, self.mvbw, self.rvbw,
                  self.start_time, self.end_time,
                  self.iaca_flops, self.cycles,
                  self.teams, self.thread_limit,
                  self.nvlink_info)
        str_line = " | ".join(["%%%ds" % (15)] +
                              ["%%%d.%dg" % (15, 15) for i in range(len(values) - 4)] +
                              ["%d", "%d", "%s"]) + "\n"
        return str_line % values

    def full_line(self):
        values = (self._name,
                  self._ncalls(),
                  self.python_time,
                  self.c_runtime,
                  self.rv_c_runtime,
                  self.v_volume, self.m_volume, self.mv_volume,
                  self.vbw, self.mbw, self.mvbw, self.rvbw,
                  self.start_time, self.end_time,
                  self.rv_start_time, self.rv_end_time,
                  self.iaca_flops, self.papi_flops, self.cycles,
                  self.teams, self.thread_limit,
                  self.nvlink_info)
        vals = []
        for i in range(1, len(values) - 3):
            if values[i] == 0 or (isinstance(values[i], int) and abs(values[i]) < 10):
                vals += ["%%%d.%dg" % (1, 1)]
            else:
                vals += ["%%%d.%dg" % (15, 15)]
        vals += ["%d", "%d", "%s"]
        str_line = " | ".join(["%%%ds" % (15)] + vals) + "\n"
        # str_line = " | ".join(["%%%ds" % (15)] + ["%%%d.%dg" % (15, 15) if values[i] != 0 else "%%%d.%dg" % (1, 1) for i in range(len(values) - 1)]) + "\n"
        return str_line % values

    def add_values(self, words):
        self.python_time = float(words[4])
        self.c_runtime = float(words[6])
        self.rv_c_runtime = float(words[8])
        self.v_volume = float(words[10])
        self.m_volume = float(words[12])
        self.mv_volume = float(words[14])
        self.start_time = float(words[24])
        self.end_time = float(words[26])
        self.rv_start_time = float(words[28])
        self.rv_end_time = float(words[30])
        self.iaca_flops = float(words[32])
        self.papi_flops = float(words[34])
        self.cycles = float(words[36])
        self.teams = int(words[38])
        self.thread_limit = int(words[40])
        self.nvlink_info = " ".join(words[42:])

    def plot_list(self, frequency):
        return [self.runtime,
                self.rv_runtime,
                self.v_volume, self.m_volume, self.mv_volume,
                self.vbw, self.mbw, self.mvbw, self.rvbw,
                self.iaca_flops, self.papi_flops,
                self.iaca_mflops, self.papi_mflops, self.cycles * 1.0 / frequency, self.c_runtime]

    @property
    def name(self):
        return self._name

    ###############################################
    # The following values are reducible measures #
    ###############################################

    @property
    def python_time(self):
        return self._reduce("python_time", self._python_time)

    @property
    def v_volume(self):
        return self._reduce("v_volume", self._v_volume)

    @property
    def mv_volume(self):
        return self._reduce("mv_volume", self._mv_volume)

    @property
    def m_volume(self):
        return self._reduce("m_volume", self._m_volume)

    @property
    def start_time(self):
        return self._reduce("start_time", self._start_time)

    @property
    def end_time(self):
        return self._reduce("end_time", self._end_time)

    @property
    def c_runtime(self):
        return self._reduce("c_runtime", self._c_runtime)

    @property
    def rv_start_time(self):
        return self._reduce("rv_start_time", self._rv_start_time)

    @property
    def rv_end_time(self):
        return self._reduce("rv_end_time", self._rv_end_time)

    @property
    def rv_c_runtime(self):
        return self._reduce("rv_c_runtime", self._rv_c_runtime)

    @property
    def papi_flops(self):
        return self._reduce("papi_flops", self._papi_flops)

    @property
    def iaca_flops(self):
        return self._reduce("iaca_flops", self._iaca_flops)

    @property
    def cycles(self):
        if not self._is_proc:
            return self._reduce("cycles", self._cycles)
        return self._reduce("cycles", self._cycles) / 1e9

    @property
    def teams(self):
        if self._teams:
            return self._teams[0]
        return 0

    @property
    def thread_limit(self):
        if self._thread_limit:
            return self._thread_limit[0]
        return 0

    @property
    def nvlink_info(self):
        if self._nvlink_info:
            return self._nvlink_info[0]
        return "N/A"

    #################################################
    # The following methods add the values to lists #
    #################################################

    @name.setter
    def name(self, value):
        self._name = value

    @v_volume.setter
    def v_volume(self, value):
        self._v_volume += [value]

    @mv_volume.setter
    def mv_volume(self, value):
        self._mv_volume += [value]

    @m_volume.setter
    def m_volume(self, value):
        self._m_volume += [value]

    @start_time.setter
    def start_time(self, value):
        if not self._is_proc:
            self._start_time += [value]
        elif len(self._start_time) == 0:
            self._start_time = [value]

    @end_time.setter
    def end_time(self, value):
        if not self._is_proc:
            self._end_time += [value]
        elif len(self._end_time) == 0:
            self._end_time = [value]

    @rv_start_time.setter
    def rv_start_time(self, value):
        if not self._is_proc:
            self._rv_start_time += [value]
        elif len(self._rv_start_time) == 0:
            self._rv_start_time = [value]

    @rv_end_time.setter
    def rv_end_time(self, value):
        if not self._is_proc:
            self._rv_end_time += [value]
        elif len(self._rv_end_time) == 0:
            self._rv_end_time = [value]

    @python_time.setter
    def python_time(self, value):
        self._python_time += [value]

    @c_runtime.setter
    def c_runtime(self, value):
        self._c_runtime += [value]

    @rv_c_runtime.setter
    def rv_c_runtime(self, value):
        self._rv_c_runtime += [value]

    @papi_flops.setter
    def papi_flops(self, value):
        self._papi_flops += [value]

    @iaca_flops.setter
    def iaca_flops(self, value):
        self._iaca_flops += [value]

    @cycles.setter
    def cycles(self, value):
        self._cycles += [value]

    @teams.setter
    def teams(self, value):
        self._teams += [value]

    @thread_limit.setter
    def thread_limit(self, value):
        self._thread_limit += [value]

    @nvlink_info.setter
    def nvlink_info(self, value):
        self._nvlink_info += [value]

    ###########################################################
    # The following values will be the ones used in the plots #
    ###########################################################

    @property
    def runtime(self):
        if not self._is_proc:
            return self.end_time - self.start_time
        return self._ncalls() * (self.end_time - self.start_time)

    @property
    def rv_runtime(self):
        if not self._is_proc:
            return self.rv_end_time - self.rv_start_time
        return self._ncalls() * (self.rv_end_time - self.rv_start_time)

    @property
    def vbw(self):
        return self.v_volume / self.runtime

    @property
    def mvbw(self):
        return self.mv_volume / self.runtime

    @property
    def mbw(self):
        return self.m_volume / self.runtime

    @property
    def rvbw(self):
        if len(self._rv_c_runtime) > 0 and self.rv_runtime > 0:
            return self.mv_volume / self.rv_runtime
        return -1

    @property
    def iaca_mflops(self):
        return self.iaca_flops / self.runtime

    @property
    def papi_mflops(self):
        return self.papi_flops / self.runtime


class GPUReductionRecord(ReductionRecord):
    """
    For OpenMP 4.0 on GPU, the number of teams and threads
    used within the calculation differs so a reduction to find the
    optimal number of teams and thread is required.

    Find the minimum rentime and the teams and threads required.
    """

    def _min_index(self):
        min_runtime = min(self._c_runtime)
        min_index = self._c_runtime.index(min_runtime)
        return min_index

    def _reduce(self, attr, values):
        """Return the measure corresponding to the fastest run."""
        return values[self._min_index()]

    @property
    def runtime(self):
        return self._reduce("runtime", self._c_runtime)

    @property
    def teams(self):
        return self._reduce("teams", self._teams)

    @property
    def thread_limit(self):
        return self._reduce("thread_limit", self._thread_limit)

    @property
    def nvlink_info(self):
        return self._reduce("nvlink_info", self._nvlink_info)

    @teams.setter
    def teams(self, value):
        self._teams += [value]

    @thread_limit.setter
    def thread_limit(self, value):
        self._thread_limit += [value]

    @nvlink_info.setter
    def nvlink_info(self, value):
        self._nvlink_info += [value]

    def plot_list(self, frequency):
        return [self.runtime,
                self.rv_runtime,
                self.v_volume, self.m_volume, self.mv_volume,
                self.vbw, self.mbw, self.mvbw, self.rvbw,
                self.iaca_flops, self.papi_flops,
                self.iaca_mflops, self.papi_mflops, self.cycles * 1.0 / frequency,
                self.c_runtime,
                self.teams, self.thread_limit, self.nvlink_info]
