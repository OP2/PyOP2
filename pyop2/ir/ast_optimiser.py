from collections import defaultdict
from copy import deepcopy as dcopy

from pyop2.ir.ast_base import *


class LoopOptimiser(object):

    """ Loops optimiser:
        * LICM:
        is usually about moving stuff independent of the inner-most loop
        here, a slightly different algorithm is employed: only const values are
        searched in a statement (i.e. read-only values), but their motion
        takes into account the whole loop nest. Therefore, this is licm
        tailored to assembly routines
        * register tiling:
        -
        * interchange:
        permute the loops in the nest
        * unroll-and-jam:
        unroll outer loops and jam unrolled bodies

        * peeling:
        - """

    def __init__(self, loop_nest):
        self.loop_nest = loop_nest
        self.out_prod_loops = None
        self.fors, self.decls, self.sym = self._explore_perfect_nest(loop_nest)

    def _explore_perfect_nest(self, node):
        """Explore the loop nest and collect various info like:
            - which loops are in the nest
            - declarations
            - optimisations suggested by the higher layers via pragmas
            - ...
        ."""

        def check_opts(node, loops):
            """Check if node is associated some pragma. If that is the case,
            it saves this info so as to enable pyop2 optimising such node. """
            if node.pragma:
                opts = node.pragma.split(" ", 2)
                if len(opts) < 3:
                    return
                if opts[1] == "pyop2":
                    delim = opts[2].find('(')
                    opt_name = opts[2][:delim].replace(" ", "")
                    opt_par = opts[2][delim:].replace(" ", "")
                    # Found high-level optimisation
                    if opt_name == "outerproduct":
                        # Find outer product loops
                        for l in loops:
                            if l.it_var() in [opt_par[1], opt_par[3]]:
                                self.out_prod_loops = l
                    else:
                        # TODO: return a proper error
                        print "Unrecognised opt %s - skipping it", opt_name
                else:
                    # TODO: return a proper error
                    print "Unrecognised pragma - skipping it"

        def traverse_tree(node, fors, decls, symbols):
            if isinstance(node, Block):
                self.block = node
                return traverse_tree(node.children[0], fors, decls, symbols)
            elif isinstance(node, For):
                fors.append(node)
                return traverse_tree(node.children[0], fors, decls, symbols)
            elif isinstance(node, Par):
                return traverse_tree(node.children[0], fors, decls, symbols)
            elif isinstance(node, Decl):
                decls[node.sym.symbol] = node
                return (fors, decls, symbols)
            elif isinstance(node, Symbol):
                if node.symbol not in symbols and node.rank:
                    symbols.append(node.symbol)
                return (fors, decls, symbols)
            elif isinstance(node, BinExpr):
                traverse_tree(node.children[0], fors, decls, symbols)
                traverse_tree(node.children[1], fors, decls, symbols)
                return (fors, decls, symbols)
            elif perf_stmt(node):
                check_opts(node, fors)
                traverse_tree(node.children[0], fors, decls, symbols)
                traverse_tree(node.children[1], fors, decls, symbols)
                return (fors, decls, symbols)
            else:
                return (fors, decls, symbols)

        return traverse_tree(node, [], {}, [])

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

        def replace_const(node, syms_dict):
            # Reached a leaf, go back
            if isinstance(node, Symbol):
                return False
            # Reached a parentheses, found or go deeper
            if isinstance(node, Par):
                if node in syms_dict:
                    return True
                else:
                    return replace_const(node.children[0], syms_dict)
            # Found invariant sub-expression
            if node in syms_dict:
                return True

            # Traverse the expression tree and replace
            left = node.children[0]
            right = node.children[1]
            if replace_const(left, syms_dict):
                node.children[0] = syms_dict[left]
            if replace_const(right, syms_dict):
                node.children[1] = syms_dict[right]

            return False

        # Find out all variables which are written to in this loop nest
        written_vars = []
        for s in self.block.children:
            if type(s) in [Assign, Incr]:
                written_vars.append(s.children[0].symbol)

        # Extract read-only sub-expressions that do not depend on at least
        # one loop in the loop nest
        for s in self.block.children:
            expr_dep = defaultdict(list)
            if isinstance(s, (Assign, Incr)):
                typ = decl[s.children[0].symbol].typ
                extract_const(s.children[1], expr_dep)

            # Create a new sub-tree for each invariant sub-expression
            # The logic is: the invariant expression goes after the outermost
            # non-depending loop and after the faster varying dimension loop
            # (e.g if exp depends on i,j and the nest is i-j-k, the exp goes
            # after i). The expression is then wrapped with all the inner
            # loops it depends on (in order to be autovectorized).
            for dep, expr in expr_dep.items():
                # 1) Find the loops that should wrap invariant statement
                # and where the new for block should be placed in the original
                # loop nest (in a pre-header block if out of the outermost).

                # Invariant code must be out of the faster varying dimension
                fast_for = [l for l in self.fors if l.it_var() == dep[-1]][0]
                # Invariant code must be out of the outermost non-depending dim
                n_dep_for = [l for l in self.fors if l.it_var() not in dep][0]

                # Find where to put the new invariant for
                pre_loop = None
                for l in self.fors:
                    if l.it_var() not in [fast_for.it_var(), n_dep_for.it_var()]:
                        pre_loop = l
                    else:
                        break
                if pre_loop:
                    place, wl = (pre_loop, [fast_for])
                else:
                    place, wl = (None,
                                 [l for l in self.fors if l.it_var() in dep])

                # 2) Create the new loop
                sym_rank = tuple([l.size() for l in wl],)
                syms = [Symbol("LI_%s_%s" % (wl[-1].it_var(), i), sym_rank)
                        for i in range(len(expr))]
                var_decl = [Decl(typ, _s) for _s in syms]
                for_rank = tuple([l.it_var() for l in wl])
                for_sym = [Symbol(_s.sym.symbol, for_rank) for _s in var_decl]
                for_ass = [Assign(_s, e) for _s, e in zip(for_sym, expr)]
                block = Block(for_ass, open_scope=True)
                for l in reversed(wl):
                    inv_for = For(dcopy(l.init), dcopy(l.cond),
                                  dcopy(l.incr), block)
                    block = inv_for
                inv_block = Block(var_decl + [inv_for])
                print inv_block

                # Update the lists of symbols accessed and of decls
                self.sym += [d.sym.symbol for d in var_decl]
                self.decls.update(dict(zip([d.sym.symbol for d in var_decl],
                                           var_decl)))

                # 3) Append the node at the right level in the loop nest
                new_block = var_decl + [inv_for] + place.children[0].children
                place.children[0].children = new_block

                # 4) Replace invariant sub-trees with the proper tmp variable
                replace_const(s.children[1], dict(zip(expr, for_sym)))

    def interchange(self):
        pass
