# This file is part of PyOP2.
#
# PyOP2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# PyOP2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# PyOP2.  If not, see <http://www.gnu.org/licenses>
#
# Copyright (c) 2011, Graham Markall <grm08@doc.ic.ac.uk> and others. Please see
# the AUTHORS file in the main source directory for a full list of copyright
# holders.

"""OP2 sequential backend."""

import numpy as np

from exceptions import *
from utils import *
import runtime_base
from runtime_base import READ, WRITE, RW, INC, MIN, MAX, IterationSpace,\
                         DataCarrier, IterationIndex, i, IdentityMap, Kernel

class LazyComputation(object):
    """Interface for lazy computation."""

    def reads(self):
        """Return computation's read dependencies."""
        assert False

    def writes(self):
        """Return computation's write dependencies."""
        assert False

    def _compute(self):
        """Called when the computation must proceed."""
        assert False

    @property
    def evaluate(self):
        return not not (self._evaluate)

    @property
    def dotname(self):
        return "dummy"


class Dat(runtime_base.Dat):

    @property
    def data(self):
        """Data array."""
        #force computation
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
        _force(set(), set([self]))
        self._data = verify_reshape(value, self.dtype, self.dim)



class Global(runtime_base.Global):

    @property
    def data(self):
        """Data array."""
        _force(set([self]), set())
        return self._data

    class Dummy(LazyComputation):
        def __init__(self, gbl, value):
            self._gbl = gbl
            self._value = value
            self._reads = set()
            self._writes = set([self._gbl])

        def reads(self):
            return self._reads

        def writes(self):
            return self._writes

        def _compute(self):
            self._gbl._data_setter(self._value)

        @property
        def dotname(self):
            return self._gbl._name + '_write_' + str(hash(self))

    @data.setter
    def data(self, value):
        _trace.append(Global.Dummy(self, value))

    def _data_setter(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)


class ParLoop(LazyComputation):
    """OP2 parallel loop."""

    def __init__(self, kernel, it_space, *args):
        self._kernel = kernel
        self._it_space = it_space
        self._args = args

        self._reads = set(arg.data for arg in self._args if arg.access in [READ, RW, INC, MIN, MAX]).union(Const._defs.copy())
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
    _trace2dag(filter(lambda c: c.evaluate, _trace))
    for cont in _trace:
        if cont.evaluate:
            cont._compute()
        else:
            nt.append(cont)
    _trace = nt

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

    #def reduce_reads():
    #    for d, conts in read_info.iteritems():
    #        read_info[d] = [c for c in conts if c.ancestors.isdisjoint(conts)]

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
