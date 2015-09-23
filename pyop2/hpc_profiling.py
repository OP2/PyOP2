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

"""Profiling classes/functions."""

from time import time
from contextlib import contextmanager
from configuration import configuration
from record import *
import numpy as np


class Timer(object):

    """Generic timer class.

    :param name: The name of the timer, used as unique identifier.
    :param timer: The timer function to use. Takes no parameters and returns
        the current time. Defaults to time.time.
    """

    _timers = {}
    _output_file = None
    _flp_ops = 0
    _gflops = 0.0
    _flp_ops_2 = 0.0

    def __new__(cls, name=None, timer=time):
        n = name or 'timer' + str(len(cls._timers))
        if n in cls._timers:
            return cls._timers[n]
        return super(Timer, cls).__new__(cls, name, timer)

    def __init__(self, name=None, timer=time):
        n = name or 'timer' + str(len(self._timers))
        if n in self._timers:
            return
        self._name = n
        self._timer = timer
        self._timers[n] = self
        # Use the reduction record to reduce values on the same process
        self._record = ReductionRecord(fold_bw=SUM, fold_time=SUM, fold_stats=SUM, is_proc=True)
        self._record.name = n
        self._start = None

    def data_volume(self, vol, vol_mvbw, vol_mbw, other_measures):
        self._record.v_volume = vol / (1024.0 * 1024.0)
        self._record.mv_volume = vol_mvbw / (1024.0 * 1024.0)
        self._record.m_volume = vol_mbw / (1024.0 * 1024.0)
        # Only take the first and last time stamp of the first run
        self._record.start_time = other_measures[4]
        self._record.end_time = other_measures[5]
        self._record.iaca_flops = other_measures[6] / 1e6
        self._record.cycles = other_measures[7]

    def rand_start_end(self, other_measures):
        self._record.rv_start_time = other_measures[4]
        self._record.rv_end_time = other_measures[5]

    def c_time(self, c_time):
        """Time value from the kernel wrapper."""
        self._record.c_runtime = c_time

    def c_rand_time(self, c_rand_time):
        """Time value from the kernel wrapper in the randomized case."""
        self._record.rv_c_runtime = c_rand_time

    def papi_gflops(self, papi_measures):
        """Time value from the kernel wrapper."""
        self._record.papi_flops = papi_measures[0] / 1e6

    def start(self):
        """Start the timer."""
        if self._name not in Timer._timers:
            self.reset()
            Timer._timers[self._name] = self
        self._start = self._timer()

    def stop(self):
        """Stop the timer."""
        assert self._start, "Timer %s has not been started yet." % self._name
        t = self._timer() - self._start
        self._record.python_time = t
        self._start = None
        return t

    def reset(self):
        """Reset the timer."""
        self._record.reset()

    @property
    def name(self):
        """Name of the timer."""
        return self._name

    @property
    def record(self):
        """The profiling data record."""
        return self._record

    @classmethod
    def summary(cls, filename=None):
        """Print a summary table for all timers or write CSV to filename."""
        if not cls._timers:
            return
        keys = sorted(cls._timers.keys(), key=lambda k: k[-6:])
        for k in keys:
            t = cls._timers[k]
            if cls._output_file is not None:
                with open(cls._output_file, "a") as f:
                    f.write(t.record.full_line())
            else:
                print t.record.full_line()

    @classmethod
    def get_timers(cls):
        """Return a dict containing all Timers."""
        return cls._timers

    @classmethod
    def reset_all(cls):
        """Clear all timer information previously recorded."""
        cls._output_file = None
        cls._extra_param = None
        cls._only_kernel = False
        if not cls._timers:
            return
        cls._timers = {}
        cls._gflops = 0.0
        cls._flp_ops = 0

    @classmethod
    def output_file(cls, value):
        """Set the output file name."""
        cls._output_file = value

    @classmethod
    def extra_param(cls, value):
        """Used for printing extra information about the run."""
        cls._extra_param = value

    @classmethod
    def only_kernel(cls, value):
        """Only time the kernel execution and return the value otherwise
        time the whole wrapper."""
        cls._only_kernel = value

    @classmethod
    def set_max_bw(cls, value):
        """Include a distribution of values per element."""
        cls._max_bw = value


@contextmanager
def hpc_profiling(t, name):
    timer = Timer("%s-%s" % (t, name))
    timer.start()
    yield
    timer.stop()


def add_data_volume(t, name, vol, vol_mvbw, vol_mbw, other_measures=np.zeros(8)):
    if not configuration['randomize']:
        Timer("%s-%s" % (t, name)).data_volume(vol, vol_mvbw, vol_mbw, other_measures)
    else:
        Timer("%s-%s" % (t, name)).rand_start_end(other_measures)


def add_c_time(t, name, time):
    if not configuration['randomize']:
        Timer("%s-%s" % (t, name)).c_time(time)
    else:
        Timer("%s-%s" % (t, name)).c_rand_time(time)


def add_papi_gflops(t, name, papi_measures):
    if not configuration['randomize']:
        Timer("%s-%s" % (t, name)).papi_gflops(papi_measures)


def summary(filename=None):
    """Print a summary table for all timers or write CSV to filename."""
    Timer.summary(filename)


def get_timers(reset=False):
    """Return a dict containing all Timers."""
    ret = Timer.get_timers()
    if reset:
        Timer.reset_all()
    return ret


def reset():
    """Clear all timer information previously recorded."""
    Timer.reset_all()


def output_file(value):
    """Set an output file for the profiling summary."""
    Timer.output_file(value)


def extra_param(value):
    """Pass in information about the run to distinguish it from other runs if necessary."""
    if not configuration['randomize']:
        Timer.extra_param(value)


def only_kernel(value):
    """Only insert instrumentation code and timers around the kernel. By Default
    the instrumentation and timers are set at the beginning."""
    Timer.only_kernel(value)
