import numpy as np
from mpi4py import MPI

class PerformanceData(object):
    label={'timing':'Time [s]',
           'flops':'Floating point performance [GFLOPs]',
           'bandwidth_perfect':'memory BW (perfect caching) [GB/s]',
           'bandwidth_pessimal':'memory BW (pessimal caching) [GB/s]'}
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
        quantities = ('timing','flops','bandwidth_perfect','bandwidth_pessimal')
        self._data = {quantity:np.empty(0) for quantity in quantities}
        self._data_global = {quantity:None for quantity in quantities}
        self._props = {'flops':1.E-9*flops,
                       'bandwidth_perfect_loads':1.E-9*perfect_bytes.loads,
                       'bandwidth_perfect_stores':1.E-9*perfect_bytes.stores,
                       'bandwidth_perfect':1.E-9*(perfect_bytes.loads + \
                                                  perfect_bytes.stores),
                       'bandwidth_pessimal_loads':1.E-9*pessimal_bytes.loads,
                       'bandwidth_pessimal_stores':1.E-9*pessimal_bytes.stores,
                       'bandwidth_pessimal':1.E-9*(pessimal_bytes.loads + \
                                                       pessimal_bytes.stores)}
        self._props_global = {quantity:None for quantity in self._props.keys()}
        # Flag to check whether data has been gathered in paralled
        self._allgathered=False

    def add_timing(self,t):
        """Add another timing data point.

        :arg t: New timing to add
        """
        self._data["timing"] = np.append(self._data["timing"],t)
        for quantity in ('flops','bandwidth_perfect','bandwidth_pessimal'):
            self._data[quantity] = np.append(self._data[quantity],
                                             self._props[quantity]/t)
        self._allgathered = False

    def all_gather(self,comm):
        """Gather all data

        :arg comm: MPI communicator
        """
        # Gather measured data
        for quantity in self._data.keys():
            data_global = np.zeros((comm.Get_size(),len(self._data[quantity])))
            comm.Allgather([self._data[quantity],MPI.DOUBLE],
                           [data_global,MPI.DOUBLE])
            self._data_global[quantity] = data_global
        # Gather loop properties
        for quantity in self._props.keys():
            props_global = np.zeros(comm.Get_size())
            tmp = np.array([self._props[quantity]])
            comm.Allgather([tmp,MPI.DOUBLE],
                           [props_global,MPI.DOUBLE])
            self._props_global[quantity] = props_global
        self._allgathered = True

    def data_str(self,quantity,p=None):
        """Return string with information on a particular quantity

        :arg quantity: Quantity to print (timing, flops, bandwidth_perfect
                       or bandwidth_pessimal)
        :arg p: processor rank. If none, takes max/min over all processors
        """
        assert(self._allgathered)
        if p==None:
            if (quantity == 'timing'):
                # Calculate the maximal time...
                return self._stat_str(np.amax(self._data_global[quantity],0))
            else:
                # ... but minimal FLOPs and BW to get a conservative estimate 
                return self._stat_str(np.sum(self._data_global[quantity],0))
        else:
            # Return quantity on processor p
            return self._stat_str(self._data_global[quantity][p],p)


    @staticmethod
    def header():
        """string with column header"""
        s = ''
        s += ('%64s' % '')+' '
        s += (' %8s ' % 'proc')+' '
        s += ('%10s' % 'calls')+' '
        s += ('%10s' % 'min')+' '
        s += ('%10s' % 'avg')+' '
        s += ('%10s' % 'max')+' '
        s += ('%10s' % 'stddev')+' '
        s += ' raw data'
        return s

    def _stat_str(self,data,p=None):
        """Data summary string
        
        Convert data to a string which contains the following information:
        * label
        * Number of calls
        * min value
        * avg value
        * max value
        * standard deviation
        * Raw data in the form (x_1,x_2,...,x_n)

        :arg data: numpy array with raw data
        :arg p: processor (None for parallel min/max)
        """
        ndata = np.array(data)
        s = ('%64s' % self._label)+' '
        if (p==None):
            s += ('[%8s]' % 'total')+' '
        else:
            s += ('[%8d]' % p)+' '
        if (len(data) == 0):
            s += ' not called'
            return s
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
    
    @staticmethod
    def quantities_header():
        """Print out header for loop quantities"""
        s = ('%64s' % '')+' '
        s += (' %8s ' % 'proc')+' '
        s += ('%10s' % 'GFLOPs')+' '
        s += ('%32s' % 'perfect caching [GB]')+' '
        s += ('%32s' % 'pessimal caching [GB]')+' '
        s += ('%21s' % 'arithmetic intensity')+'\n'
        s += ('%64s' % '')+' '
        s += (' %8s ' % '')+' '
        s += ('%10s' % '')+' '
        for i in range(2):
            s += ('%10s' % 'loads')+' '
            s += ('%10s' % 'stores')+' '
            s += ('%10s' % 'total')+' '
        s += ('%10s' % 'perfect')+' '
        s += ('%10s' % 'pessimal')
        return s

    def quantities_str(self,p=None):
        """Print out quantities of loop

        * Label
        * Loads/stores/loads+stores [perfect caching]
        * Loads/stores/loads+stores [pessimal caching]
        * arithmetic intensity [perfect and pessimal caching]

        :arg p: Processor rank (None to print out min/max)
        """
        assert(self._allgathered)
        s = ('%64s' % self._label)+' '
        if (p==None):
            s += ('[%8s]' % 'total')+' '
        else:
            s += ('[%8d]' % p)+' '

        # Floating point performance
        for quantity in ('flops',
                         'bandwidth_perfect_loads',
                         'bandwidth_perfect_stores',
                         'bandwidth_perfect',
                         'bandwidth_pessimal_loads',
                         'bandwidth_pessimal_stores',
                         'bandwidth_pessimal'):
            if (p==None):
                # If no processor is specified, sum over all
                tmp = np.sum(self._props_global[quantity])
            else:
                tmp = self._props_global[quantity][p]
            s += ('%10.3e' % tmp)+' '
        if (p==None):
            intensity_perfect = np.sum(self._props_global['flops']) \
                / (np.sum(self._props_global['bandwidth_perfect']))
            intensity_pessimal = np.sum(self._props_global['flops']) \
                / (np.sum(self._props_global['bandwidth_pessimal']))
        else:
            intensity_perfect = self._props_global['flops'][p] \
                / self._props_global['bandwidth_perfect'][p]
            intensity_pessimal = self._props_global['flops'][p] \
                / self._props_global['bandwidth_pessimal'][p]
        s += ('%10.3e' % intensity_perfect)+' '
        s += ('%10.3e' % intensity_pessimal)+' '
        return s
