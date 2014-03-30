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

"""Module supporting PyOP2's lazy evaluation scheme.
"""

from configuration import configuration
from utils import to_set


class LazyComputation(object):

    """Helper class holding computation to be carried later on."""

    def __init__(self, reads, writes):
        self.reads = to_set(reads)
        self.writes = to_set(writes)

    def depends_on(self, c):
        return self.reads & c.writes or self.writes & c.reads or self.writes & c.writes

    def needed_for(self, reads, writes):
        return self.reads & writes or self.writes & reads

    def enqueue(self):
        _trace.append(self)
        return self

    def _run(self):
        assert False, "Not implemented"


class LazyPass(LazyComputation):

    """Lazy no op class."""

    def _run(self):
        pass

    def __str__(self):
        return "pass"


class LazyMethodCall(LazyComputation):

    """Helper class for the lazy evaluation of a method call."""

    def __init__(self, reads, writes, bound_method, *args, **kwargs):
        LazyComputation.__init__(self, reads, writes)
        self._method = bound_method
        self._args = args
        self._kwargs = kwargs

    def _run(self):
        self._method(*self._args, **self._kwargs)

    def __str__(self):
        return "LazyMethodCall(%r(%r))" % \
               (str(self._method), ", ".join([str(a) for a in self._args]))


class LazyHaloSend(LazyMethodCall):

    def __init__(self, arg, parloop):
        LazyMethodCall.__init__(self,
                                set([CORE(arg.data), OWNED(arg.data)]),
                                set([NET(arg.data, parloop)]),
                                arg.halo_exchange_begin)

        self._dat = arg.data

    def __str__(self):
        return "LazyHaloSend(%r)" % self._dat


class LazyHaloRecv(LazyMethodCall):

    def __init__(self, arg, parloop):
        LazyMethodCall.__init__(self,
                                set([NET(arg.data, parloop)]),
                                set([HALOEXEC(arg.data), HALOIMPORT(arg.data)]),
                                arg.halo_exchange_end)
        self._dat = arg.data

    def __str__(self):
        return "LazyHaloRecv(%r)" % self._dat


class LazyReductionBegin(LazyMethodCall):

    def __init__(self, arg):
        LazyMethodCall.__init__(self, set([arg.data]), set(), arg.reduction_begin)
        self._dat = arg.data

    def __str__(self):
        return "LazyReductionBegin(%r)" % self._dat


class LazyReductionEnd(LazyMethodCall):

    def __init__(self, arg):
        LazyMethodCall.__init__(self, set(), set([arg.data]), arg.reduction_end)
        self._dat = arg.data

    def __str__(self):
        return "LazyReductionEnd(%r)" % self._dat


class LazyCompute(LazyMethodCall):

    def __init__(self, reads, writes, parloop, section):
        LazyMethodCall.__init__(self, reads, writes, parloop.compute, section)
        self._parloop = parloop
        self._section = section

    def __str__(self):
        return "compute(%r)[%r]" % (self._parloop, self._section)


class Dependency(object):

    """Contract for lazy evaluation dependencies."""

    def __init__(self, dat):
        self._dat = dat

    def __eq__(self, other):
        return type(self) == type(other) and self._dat is other._dat

    def __hash__(self):
        return self._HASH ^ hash(self._dat)

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self._dat._name)


class CORE(Dependency):
    _HASH = 0b1001


class OWNED(Dependency):
    _HASH = 0b0110


class HALOEXEC(Dependency):
    _HASH = 0b1100


class HALOIMPORT(Dependency):
    _HASH = 0b0011


class NET(Dependency):

    """An abstract 'dependencies' for network operations."""

    _HASH = 0b1111

    def __init__(self, dat, parloop):
        self._dat = dat
        self._parloop = parloop

    def __eq__(self, other):
        return type(self) == type(other) and self._dat is other._dat and self._parloop is other._parloop

    def __hash__(self):
        return self._HASH ^ hash(self._dat) ^ hash(self._parloop)

    def __str__(self):
        return "NET(%r, %r)" % (self._dat, self._parloop)


def ALL(dat):
    return [CORE(dat), OWNED(dat), HALOEXEC(dat), HALOIMPORT(dat)]


class ExecutionTrace(object):

    """Container maintaining :class:`LazyComputation` until they are executed.

    Objects of this class maintain lazy computation inside a directed acyclic
    graph representing a partial order of the execution based on the reads and
    writes specifications of each :class:LazyComputation.
    """

    def __init__(self):
        self._init()

    def _init(self):
        self.top = LazyPass(None, None)
        self.bot = LazyPass(None, None)
        self._edges = set()
        self._edges.add((self.top, self.bot))

    def children(self, node):
        return set([e[1] for e in self._edges if e[0] is node])

    def parents(self, node):
        return set([e[0] for e in self._edges if e[1] is node])

    def descendants(self, node):
        d = self.children(node)
        children = set(d)
        for c in children:
            d.update(self.descendants(c))
        return d

    def ancestors(self, node):
        a = self.parents(node)
        parents = set(a)
        for p in parents:
            a.update(self.ancestors(p))
        return a

    def clear(self):
        """Forcefully drops delayed computations. Only use this if you know
        what you are doing."""
        self._init()

    def append(self, computation):
        if not configuration['lazy_evaluation']:
            computation._run()
        elif configuration['lazy_max_trace_length'] > 0 and \
                configuration['lazy_max_trace_length'] == len(self._trace):
            self.evaluate(computation.reads, computation.writes)
            computation._run()
        else:
            self._append(computation)

    def _append(self, node):
        """Inserts `node` in the DAG."""
        # find where to attach the new closure:
        # fallback in case we find no other closure to attach to
        anchors = set([self.top])
        # find all closure the new on depends_on
        # NOTE: this search could be cut back
        anchors.update([n for n in self.descendants(self.top) if node.depends_on(n)])
        # only attach to anchors which have no descendants in anchors
        anchors.difference_update([n for a in anchors for n in self.ancestors(a)])
        # attach the new node to its anchors and bot
        self._edges.update([(a, node) for a in anchors])
        self._edges.add((node, self.bot))
        self._edges.difference_update([(a, self.bot) for a in anchors])
        self._edges.discard((self.top, self.bot))

    def in_queue(self, computation):
        return computation in [e[0] for e in self._edges] or\
            computation in [e[1] for e in self._edges]

    def remove(self, node):
        parents = self.parents(node)
        children = self.children(node)

        self._edges.difference_update([(p, node) for p in parents])
        self._edges.difference_update([(node, c) for c in children])
        self._edges.update([(p, c) for p in parents for c in children])

    def evaluate_all(self):
        """Forces the evaluation of all delayed computations."""
        self._schedule(self.top)
        self._run(self.top)
        self._init()

    def _run(self, node):
        assert len(self.children(node)) < 2
        node._run()
        if self.children(node):
            self._run(list(self.children(node))[0])

    def _schedule(self, node):
        if node is self.bot:
            return
        if len(self.children(node)) == 1:
            self._schedule(list(self.children(node))[0])
            return

        # if multiple computations can be scheduled
        runnables = [n for n in self.children(node) if len(self.parents(n)) == 1]
        sends = [n for n in runnables if isinstance(n, LazyHaloSend)]
        # prioritize halo sends
        if sends:
            self._move_up(sends[0])
        else:
            self._move_up(runnables[0])

        # continue until the dag is a total order
        self._schedule(self.top)

    def _move_up(self, node):
        """Make `node` a parent of all its siblings."""
        assert len(self.parents(node)) == 1
        p = list(self.parents(node))[0]

        siblings = self.children(p)
        siblings.discard(node)
        self._edges.difference_update([(p, n) for n in siblings])
        self._edges.update([(node, s) for s in siblings])

        dups = self.children(node) & set([d for n in self.children(node) for d in self.descendants(n)])
        for d in dups:
            self._edges.discard((node, d))

    def evaluate(self, reads=None, writes=None):
        """Force the evaluation of delayed computation on which reads and writes
        depend.

        :arg reads: the :class:`DataCarrier`\s which you wish to read from.
                    This forces evaluation of all :func:`par_loop`\s that write to
                    the :class:`DataCarrier` (and any other dependent computation).
        :arg writes: the :class:`DataCarrier`\s which you will write to (i.e. modify values).
                     This forces evaluation of all :func:`par_loop`\s that read from the
                     :class:`DataCarrier` (and any other dependent computation).
        """
        self.evaluate_all()

    def _graphviz(self):
        """DEBUG, dumps the DAG in the `dot` format."""
        def name(node):
            if node is self.top:
                return "top"
            if node is self.bot:
                return "bot"
            return id(node)

        def indent(str):
            return "\n".join(map(lambda s: "  %s" % s, str.splitlines()))

        a = set(self.top.descendants)
        a.add(self.top)
        nodes = "\n".join(["%d [label=\"%s\"];" % (id(n), name(n)) for n in a])
        cedges = "\n".join(["%d -> %d [color=red];" % (id(s), id(d)) for s in a for d in s.parents])
        edges = "\n".join(["%d -> %d;" % (id(s), id(d)) for s in a for d in s.children])
        extras = "  {rank = min; %s }\n  { rank = sink; %s }\n" % (id(self.top), id(self.bot))

        def sibling(n):
            return " ".join(["%d;" % id(s) for s in n.children if len(n.children) > 1])
        siblings = "\n".join(["  { rank = same; %s }" % sibling(n) for n in a])

        return """
        digraph d {
          %(nodes)s
          %(siblings)s
          %(extras)s
          %(cedges)s
          %(edges)s
        }""" % {'nodes': indent(nodes),
                'cedges': indent(cedges),
                'edges': indent(edges),
                'extras': extras,
                'siblings': siblings}


_trace = ExecutionTrace()
