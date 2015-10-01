import numpy as np

class PerformanceData(object):
    """Class holding performance data.

    Stores the FLOP and memory reference count of a particular
    ParLoop and an array of timing of all invocations of that loop.
    Contains methods for printing out this information.

    :arg label: Unique, user defined label
    :arg flops: Number of FLOPs for one invocation of the ParLoop
    :arg perfect_bytes: Number of moved bytes (read + stored)
    for one invocation, assuming perfect caching
    :arg pessimal_bytes: Number of moved bytes (read + stored)
    for one invocation, assuming perfect caching
    """
    def __init__(self,label,flops,perfect_bytes,pessimal_bytes):
        self._label = label
        self._flops = flops
        self._perfect_bytes = perfect_bytes
        self._pessimal_bytes = pessimal_bytes
        self._timings = []

    def add_timing(self,t):
        """Add another timing data point.

        :arg t: New timing to add
        """
        self._timings.append(t)

    def timing_str(self):
        """Return string with timing information"""
        return self._stat_str(self._timings)

    def flops_str(self):
        """Return string with FLOPs information"""
        return self._stat_str([1.09E-9*self._flops/t for t in self._timings])

    def perfect_bandwidth_str(self):
        """Return string with perfect caching memory bandwidth"""
        mem_traffic = self._perfect_bytes.loads + \
            self._perfect_bytes.stores
        return self._stat_str([1.0E-9*mem_traffic/t for t in self._timings])

    def pessimal_bandwidth_str(self):
        """Return string with pessimal caching memory bandwidth"""
        mem_traffic = self._pessimal_bytes.loads + \
            self._pessimal_bytes.stores
        return self._stat_str([1.0E-9*mem_traffic/t for t in self._timings])

    @staticmethod
    def header():
        '''string with column header'''
        s = ''
        s += ('%24s' % '')+' '
        s += ('%10s' % 'calls')+' '
        s += ('%10s' % 'min')+' '
        s += ('%10s' % 'avg')+' '
        s += ('%10s' % 'max')+' '
        s += ('%10s' % 'stddev')+' '
        s += ' raw data'
        return s

    def _stat_str(self,data):
        '''Data summary string
        
        Convert data to a string which contains the following information:
        * label
        * Number of calls
        * min value
        * avg value
        * max value
        * standard deviation
        * Raw data in the form (x_1,x_2,...,x_n)

        :arg data: numpy array with raw data
        '''
        ndata = np.array(data)
        s = ('%24s' % self._label)+' '
        s += ('%10d' % len(ndata))+' '
        s += ('%10.3e' % np.amin(ndata))+' '
        s += ('%10.3e' % np.mean(ndata))+' '
        s += ('%10.3e' % np.amax(ndata))+' '
        s += ('%10.3e' % np.std(ndata))+' ('
        for i,x in enumerate(ndata):
            s += ('%10.3e' % x)
            if (i == len(ndata)-1):
                s += ' )'
            else:
                s += ', '
        return s
