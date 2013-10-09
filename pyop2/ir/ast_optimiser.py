# Loops optimiser: licm, register tiling, unroll-and-jam, peeling
#   licm is usually about moving stuff independent of the inner-most loop
#       here, a slightly different algorithm is employed: only const values are
#       searched in a statement (i.e. read-only values), but their motion
#       takes into account the whole loop nest. Therefore, this is licm
#       tailored to assembly routines
#   register tiling
#   unroll-and-jam
#   peeling
# Memory optimiser: padding, data alignment, trip count/bound adjustment
#   padding and data alignment are for aligned unit-stride load
#   trip count/bound adjustment is for auto-vectorisation

from collections import defaultdict
from copy import deepcopy as dcopy

from pyop2.ir.ast_base import *


class LoopOptimiser(object):

    """Loops optimiser: licm, register tiling, unroll-and-jam, peeling."""

    def __init__(self, loop_nest):
        self.loop_nest = loop_nest
        self.fors = self._explore_perfect_nest(loop_nest)

    def _explore_perfect_nest(self, node, fors=[]):
        """Explore perfect loop nests."""

        if isinstance(node, Block):
            self.block = node
            return self._explore_perfect_nest(node.children[0])
        elif isinstance(node, For):
            fors.append(node)
            return self._explore_perfect_nest(node.children[0])
        else:
            return fors

    def licm(self):
        """Loop-invariant code motion."""

        def extract_const(node, expr_dep):
            # Return the iteration variable dependence if it's just a symbol
            if isinstance(node, Symbol):
                return (node.loop_dep, node.symbol not in written_vars)

            # Keep traversing the tree if a parentheses object
            if isinstance(node, Par):
                return (extract_const(node.children[0], expr_dep))

            # Traverse the expression tree
            left = node.children[0]
            right = node.children[1]
            dep_left, invariant_l = extract_const(left, expr_dep)
            dep_right, invariant_r = extract_const(right, expr_dep)

            if dep_left == dep_right:
                # Children match up, keep traversing the tree in order to see
                # if this sub-expression is actually a child of a larger
                # loop-invariant sub-expression
                return (dep_left, True)
            elif len(dep_left) == 0:
                # The left child does not depend on any iteration variable,
                # so it's loop invariant
                return (dep_right, True)
            elif len(dep_right) == 0:
                # The right child does not depend on any iteration variable,
                # so it's loop invariant
                return (dep_left, True)
            else:
                # Iteration variables of the two children do not match, add
                # the children to the dict of invariant expressions iff
                # they were invariant w.r.t. some loops and not just symbols
                if invariant_l and not isinstance(left, Symbol):
                    expr_dep[dep_left].append(left)
                if invariant_r and not isinstance(right, Symbol):
                    expr_dep[dep_right].append(right)
                return ((), False)

        # Find out all variables which are written to in this loop nest
        written_vars = []
        for s in self.block.children:
            if type(s) in [Assign, Incr]:
                written_vars.append(s.children[0].symbol)

        # Extract read-only sub-expressions that do not depend on at least
        # one loop in the loop nest
        for s in self.block.children:
            expr_dep = defaultdict(list)
            if type(s) in [Assign, Incr]:
                typ = decl[s.children[0].symbol][0]
                extract_const(s.children[1], expr_dep)

            # Create a new sub-tree for each invariant sub-expression
            # The logic is: the invariant expression goes after the outermost
            # depending loop (e.g if exp depends on i,j and the nest is i-j-k,
            # the exp goes after i). The expression is then wrapped with all
            # the inner loops it depends on (in order to be autovectorized).
            for dep, expr in expr_dep.items():
                # Create the new loop
                il = dep[-1]
                loop = [l for l in self.fors if l.init.sym.symbol == il][0]
                var_decl = [Decl(typ, Symbol("LI%s" % i, (loop.size(),)))
                            for i, e in zip(range(len(expr)), expr)]
                var_sym = [Symbol(s.sym.symbol, (loop.init.sym.symbol))
                           for s in var_decl]
                var_ass = [Assign(s, e) for s, e in zip(var_sym, expr)]
                inv_for = For(dcopy(loop.init), dcopy(loop.cond),
                              dcopy(loop.incr), Block(var_ass, open_scope=True))
                inv_block = Block(var_decl + [inv_for])
                print inv_block

                # Append the node at the right level in the loop nest

                # Replace invariant sub-trees with the proper temp variable

    def interchange(self):
        pass
