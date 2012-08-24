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

"""OP2 abstract lazy backend."""

import numpy as np

from exceptions import *
from utils import *
import runtime_base
from runtime_base import READ, WRITE, RW, INC, MIN, MAX, IterationSpace, Set, Map,\
                         DataCarrier, IterationIndex, i, IdentityMap, Kernel, Sparsity

class LazyComputation(object):
    """Lazy Continuation type."""

    def reads(self):
        """Return read dependencies."""
        assert False

    def writes(self):
        """Return write dependencies."""
        assert False

    def _compute(self):
        """Execute delayed computation."""
        assert False

    @property
    def evaluate(self):
        """Set by trace management."""
        return not not (self._evaluate)

    @property
    def dotname(self):
        assert False


class Dummy(LazyComputation):
    def __init__(self, cst, value, dotname):
        self._cst = cst
        self._value = value
        self._reads = set()
        self._writes = set([self._cst])
        self._dotname = dotname

    def reads(self):
        return self._reads

    def writes(self):
        return self._writes

    def _compute(self):
        self._cst._data_setter(self._value)

    @property
    def dotname(self):
        return self._dotname


class Dat(runtime_base.Dat):

    @property
    def data(self):
        """Data array."""
        _force(set([self]), set())

        if len(self._data) is 0:
            raise RuntimeError("Illegal access: No data associated with this Dat!")
        return self._data


class Const(runtime_base.Const):

    @property
    def data(self):
        """Data array."""
        _force(set([self]), set())
        return self._data

    @data.setter
    def data(self, value):
        # call reshape to ensure type and shape error are returned immedialty
        _trace.append(Dummy(self,
                            verify_reshape(value, self.dtype, self.dim),
                            self._cst._name + '_write_' + str(hash(self))))

    def _data_setter(self, value):
        self._data = value


class Global(runtime_base.Global):

    @property
    def data(self):
        """Data array."""
        _force(set([self]), set())
        return self._data

    @data.setter
    def data(self, value):
        # call reshape to ensure type and shape error are returned immedialty
        _trace.append(Dummy(self,
                            verify_reshape(value, self.dtype, self.dim),
                            self._gbl._name + '_write_' + str(hash(self))))

    def _data_setter(self, value):
        self._data = value


class Mat(runtime_base.Mat):

    @property
    def values(self):
        _force(set([self]), set())
        return self._c_handle.values


class ParLoop(LazyComputation):
    """OP2 parallel loop."""

    def __init__(self, kernel, it_space, *args):
        self._kernel = kernel
        self._it_space = it_space
        self._args = args
        self._consts = Const._defs.copy()

        self._reads = set(arg.data for arg in self._args if arg.access in [READ, RW, INC, MIN, MAX]).union(self._consts)
        self._writes = set(arg.data for arg in self._args if arg.access in [WRITE, RW, INC, MIN, MAX])

        global _trace
        _trace.append(self)

    def reads(self):
        return self._reads

    def writes(self):
        return self._writes

    @property
    def dotname(self):
        return self._kernel._name + str(hash(self))


# delayed trace managment
def _depends_on(rreads, rwrites, creads, cwrites):
    return not not ([d for d in rreads if d in cwrites] or \
                    [d for d in rwrites if d in creads] or \
                    [d for d in rwrites if d in cwrites])

def _force(reads, writes):
    global _trace

    for cont in reversed(_trace):
        if _depends_on(reads, writes, cont.reads(), cont.writes()):
            cont._evaluate = True
            reads = set([r for r in reads | cont.reads() if r not in cont.writes()])
            writes = writes | cont.writes()
        else:
            cont._evaluate = False

    nt = list()
    #_trace2dag(filter(c for c in _trace if c.evaluate))
    for cont in _trace:
        if cont.evaluate:
            cont._compute()
        else:
            nt.append(cont)
    _trace = nt

# DAG construction
class _Node(object):
    def __init__(self, cont=None):
        self._cont = cont
        self.parents = set()
        self.children = set()
        self._descendants = None

    @property
    def ancestors(self):
        anc = set()
        anc.update(self.parents)
        for p in self.parents:
            anc.update(p.ancestors)
        return anc

    @property
    def descendants(self):
        if not self._descendants:
            self._descendants = set()
            self._descendants.update(self.children)
            for p in self.children:
                self._descendants.update(p.descendants)
        return self._descendants

    @property
    def dotname(self):
        return self._cont.dotname

class _FakeNode(_Node):
    def __init__(self, name):
        super(_FakeNode, self).__init__()
        self._name = name

    @property
    def dotname(self):
        return self._name

    @property
    def descendants(self):
        return set()

def _trace2dag(tr):
    bottom = _FakeNode("bottom")
    top = _FakeNode("top")

    read_info = dict()
    write_info = dict()

    def add_read(dat, pl):
        if not read_info.has_key(dat):
            read_info[dat] = list()
        read_info[dat].append(pl)

    def get_reads(dat):
        if not read_info.has_key(dat):
            read_info[dat] = list()
        r = read_info[dat]
        del read_info[dat]
        return r

    def set_write(dat, pl):
        write_info[dat] = pl

    def get_write(dat):
        if not write_info.has_key(dat):
            write_info[dat] = bottom
        return write_info[dat]

    nodes = [None] * len(tr)
    for i, c in enumerate(reversed(tr)):
        nodes[i] = _Node(c)
        newdescendants = set()

        for d in c.reads().difference(c.writes()):
            newdescendants.add(get_write(d))
            add_read(d, nodes[i])

        for d in c.writes().difference(c.reads()):
            newdescendants.update(get_reads(d))
            newdescendants.add(get_write(d))
            set_write(d, nodes[i])

        for d in c.reads().intersection(c.writes()):
            newdescendants.update(get_reads(d))
            newdescendants.add(get_write(d))
            set_write(d, nodes[i])
            add_read(d, nodes[i])

        ndes = newdescendants.copy()
        for c in newdescendants:
            ndes.difference_update(c.descendants)

        for c in ndes:
            #print nodes[i].dotname + ' -> ' + c.dotname
            c.parents.add(nodes[i])
            nodes[i].children.add(c)

    for c in nodes:
        if not c.parents:
            top.children.add(c)
            c.parents.add(top)

    #outputing
    _f = open("tod.dot", "w")
    _f.write('digraph {\n')
    _f.write('  ' + bottom.dotname + ';\n')
    _f.write('  ' + top.dotname + ';\n')
    for n in nodes:
        _f.write('  ' + n.dotname + ';\n')
    for n in nodes:
        for c in n.children:
            _f.write('  ' + n.dotname + ' -> ' + c.dotname + ';\n')
    for c in top.children:
        _f.write('  ' + top.dotname + ' -> ' + c.dotname + ';\n')
    _f.write('}\n')
    _f.close()

_trace = list()
