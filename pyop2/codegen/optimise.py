from gem.node import traversal, reuse_if_untouched, Memoizer
from functools import singledispatch
from pyop2.codegen.representation import (Index, RuntimeIndex, FixedIndex,
                                          Node, FunctionCall)


def collect_indices(expressions):
    """Collect indices in expressions.

    :arg expressions: an iterable of expressions to collect indices
        from.
    :returns: iterable of nodes of type :class:`Index` or
        :class:`RuntimeIndex`.
    """
    for node in traversal(expressions):
        if isinstance(node, (Index, RuntimeIndex)):
            yield node


@singledispatch
def replace_indices(node, self):
    raise AssertionError("Unhandled node type %r" % type(node))


replace_indices.register(Node)(reuse_if_untouched)


@replace_indices.register(Index)
def replace_indices_index(node, self):
    if node.extent == 1:
        return FixedIndex(0)
    return self.subst.get(node, node)


def index_merger(instructions, cache=None):
    """Merge indices across an instruction stream.

    Indices are candidates for merging if they have the same extent as
    an already seen index in the instruction stream, and appear at the
    same level of the loop nest.

    :arg instructions:  Iterable of nodes to merge indices across.
    :returns: iterable of instructions, possibly with indices replaced.
    """
    if cache is None:
        cache = {}

    appeared = {}
    subst = []

    index_replacer = Memoizer(replace_indices)

    for insn in instructions:
        if isinstance(insn, FunctionCall):
            # yield insn.reconstruct(*merge_indices(insn.children))
            continue

        indices = tuple(i for i in collect_indices([insn]))
        runtime = tuple(i for i in indices if not isinstance(i, Index))
        free = tuple(i for i in indices if isinstance(i, Index))

        indices = runtime + free

        key = runtime + tuple(i.extent for i in free)
        full_key = key
        # Look for matching key prefix
        while key not in cache and len(key):
            key = key[:-1]

        if key in cache:
            new_indices = cache[key] + indices[len(key):]
        else:
            new_indices = indices

        for i in range(len(key), len(full_key) + 1):
            cache[full_key[:i]] = new_indices[:i]

        for i, ni in zip(indices, new_indices):
            if i in appeared:
                subst.append((i, appeared[i]))
            if i != ni:
                if i in appeared:
                    assert appeared[i] == ni
                appeared[i] = ni
                subst.append((i, ni))

    index_replacer.subst = dict(subst)
    return index_replacer
