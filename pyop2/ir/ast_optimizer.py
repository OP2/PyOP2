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

from collections import defaultdict
from copy import deepcopy as dcopy

from ast_base import *
import ast_plan


class LoopOptimiser(object):

    """Loops optimiser:

    * Loop Invariant Code Motion (LICM)
      Backend compilers apply LICM to innermost loops only. In addition,
      hoisted expressions are usually not vectorized. Here, we apply LICM to
      terms which are known to be constant (i.e. they are declared const)
      and all of the loops in the nest are searched for sub-expressions
      involving such const terms only. Also, hoisted terms are wrapped
      within loops to exploit compiler autovectorization. This has proved to
      be beneficial for loop nests in which the bounds of all loops are
      relatively small (let's say less than 50-60).

    * Register tiling:
      Given a rectangular iteration space, register tiling slices it into
      square tiles of user-provided size, with the aim of improving register
      pressure and register re-use."""

    def __init__(self, loop_nest, pre_header, kernel_decls):
        self.loop_nest = loop_nest
        self.pre_header = pre_header
        self.kernel_decls = kernel_decls
        self.out_prods = {}
        self.itspace = []
        fors_loc, self.decls, self.sym = self._visit_nest(loop_nest)
        self.fors, self.for_parents = zip(*fors_loc)

    def _visit_nest(self, node):
        """Explore the loop nest and collect various info like:

        * Loops
        * Declarations and Symbols
        * Optimisations requested by the higher layers via pragmas"""

        def check_opts(node, parent, fors):
            """Check if node is associated some pragma. If that is the case,
            it saves this info so as to enable pyop2 optimising such node. """
            if node.pragma:
                opts = node.pragma.split(" ", 2)
                if len(opts) < 3:
                    return
                if opts[1] == "pyop2":
                    if opts[2] == "itspace":
                        # Found high-level optimisation
                        self.itspace.append((node, parent))
                        return
                    delim = opts[2].find('(')
                    opt_name = opts[2][:delim].replace(" ", "")
                    opt_par = opts[2][delim:].replace(" ", "")
                    if opt_name == "outerproduct":
                        # Found high-level optimisation
                        # Store outer product iteration variables, parent, loops
                        it_vars = [opt_par[1], opt_par[3]]
                        fors, fors_parents = zip(*fors)
                        loops = [l for l in fors if l.it_var() in it_vars]
                        self.out_prods[node] = (it_vars, parent, loops)
                    else:
                        raise RuntimeError("Unrecognised opt %s - skipping it", opt_name)
                else:
                    raise RuntimeError("Unrecognised pragma found '%s'", node.pragma)

        def inspect(node, parent, fors, decls, symbols):
            if isinstance(node, Block):
                self.block = node
                for n in node.children:
                    inspect(n, node, fors, decls, symbols)
                return (fors, decls, symbols)
            elif isinstance(node, For):
                check_opts(node, parent, fors)
                fors.append((node, parent))
                return inspect(node.children[0], node, fors, decls, symbols)
            elif isinstance(node, Par):
                return inspect(node.children[0], node, fors, decls, symbols)
            elif isinstance(node, Decl):
                decls[node.sym.symbol] = (node, ast_plan.LOCAL_VAR)
                return (fors, decls, symbols)
            elif isinstance(node, Symbol):
                symbols.add(node)
                return (fors, decls, symbols)
            elif isinstance(node, Expr):
                for child in node.children:
                    inspect(child, node, fors, decls, symbols)
                return (fors, decls, symbols)
            elif isinstance(node, Perfect):
                check_opts(node, parent, fors)
                for child in node.children:
                    inspect(child, node, fors, decls, symbols)
                return (fors, decls, symbols)
            else:
                return (fors, decls, symbols)

        return inspect(node, self.pre_header, [], {}, set())

    def extract_itspace(self):
        """Remove fully-parallel loop from the iteration space. These are
        the loops that were marked by the user/higher layer with a ``pragma
        pyop2 itspace``."""

        itspace_vrs = []
        for node, parent in reversed(self.itspace):
            parent.children.extend(node.children[0].children)
            parent.children.remove(node)
            itspace_vrs.append(node.it_var())

        any_in = lambda a, b: any(i in b for i in a)
        accessed_vrs = [s for s in self.sym if any_in(s.rank, itspace_vrs)]

        return (itspace_vrs, accessed_vrs)

    def op_licm(self):
        """Perform loop-invariant code motion.

        Invariant expressions found in the loop nest are moved "after" the
        outermost independent loop and "after" the fastest varying dimension
        loop. Here, "after" means that if the loop nest has two loops i and j,
        and j is in the body of i, then i comes after j (i.e. the loop nest
        has to be read from right to left)

        For example, if a sub-expression E depends on [i, j] and the loop nest
        has three loops [i, j, k], then E is hoisted out from the body of k to
        the body of i). All hoisted expressions are then wrapped within a
        suitable loop in order to exploit compiler autovectorization.
        """

        def extract_const(node, expr_dep):
            if isinstance(node, Symbol):
                return (node.loop_dep, node.symbol not in written_vars)
            if isinstance(node, Par):
                return (extract_const(node.children[0], expr_dep))

            # Traverse the expression tree
            left, right = node.children
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
                if invariant_l:
                    left = Par(left) if isinstance(left, Symbol) else left
                    expr_dep[dep_left].append(left)
                if invariant_r:
                    right = Par(right) if isinstance(right, Symbol) else right
                    expr_dep[dep_right].append(right)
                return ((), False)

        def replace_const(node, syms_dict):
            if isinstance(node, Symbol):
                if str(Par(node)) in syms_dict:
                    return True
                else:
                    return False
            if isinstance(node, Par):
                if str(node) in syms_dict:
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
                left = Par(left) if isinstance(left, Symbol) else left
                node.children[0] = syms_dict[str(left)]
            if replace_const(right, syms_dict):
                right = Par(right) if isinstance(right, Symbol) else right
                node.children[1] = syms_dict[str(right)]
            return False

        # Find out all variables which are written to in this loop nest
        written_vars = []
        for s in self.out_prods.keys():
            if type(s) in [Assign, Incr]:
                written_vars.append(s.children[0].symbol)

        # Extract read-only sub-expressions that do not depend on at least
        # one loop in the loop nest
        ext_loops = []
        self.hoisted_syms = {}
        for s, op in self.out_prods.items():
            expr_dep = defaultdict(list)
            if isinstance(s, (Assign, Incr)):
                typ = self.kernel_decls[s.children[0].symbol][0].typ
                extract_const(s.children[1], expr_dep)

            for dep, expr in expr_dep.items():
                # 1) Determine the loops that should wrap invariant statements
                # and where such for blocks should be placed in the loop nest
                n_dep_for = None
                fast_for = None
                # Collect some info about the loops
                for l in self.fors:
                    if l.it_var() == dep[-1]:
                        fast_for = fast_for or l
                    if l.it_var() not in dep:
                        n_dep_for = n_dep_for or l
                    if l.it_var() == op[0][0]:
                        op_loop = l
                if not fast_for or not n_dep_for:
                    continue

                # Find where to put the new invariant for
                pre_loop = None
                for l in self.fors:
                    if l.it_var() not in [fast_for.it_var(), n_dep_for.it_var()]:
                        pre_loop = l
                    else:
                        break
                if pre_loop:
                    place = pre_loop.children[0]
                    ofs = place.children.index(op_loop)
                    wl = [fast_for]
                else:
                    place = self.pre_header
                    ofs = place.children.index(self.loop_nest)
                    wl = [l for l in self.fors if l.it_var() in dep]

                # 2) Create the new loop
                sym_rank = tuple([l.size() for l in wl],)
                syms = [Symbol("LI_%s_%s" % (wl[0].it_var(), i), sym_rank)
                        for i in range(len(expr))]
                var_decl = [Decl(typ, _s) for _s in syms]
                for_rank = tuple([l.it_var() for l in reversed(wl)])
                for_sym = [Symbol(_s.sym.symbol, for_rank) for _s in var_decl]
                for_ass = [Assign(_s, e) for _s, e in zip(for_sym, expr)]
                block = Block(for_ass, open_scope=True)
                for l in wl:
                    inv_for = For(dcopy(l.init), dcopy(l.cond), dcopy(l.incr), block)
                    block = Block([inv_for], open_scope=True)

                # Update the lists of symbols accessed and of decls
                self.sym.update([d.sym for d in var_decl])
                lv = ast_plan.LOCAL_VAR
                self.decls.update(dict(zip([d.sym.symbol for d in var_decl],
                                           [(v, lv) for v in var_decl])))

                # 3) Append the new node at the right level in the loop nest
                new_block = var_decl + [inv_for] + place.children[ofs:]
                place.children = place.children[:ofs] + new_block

                # Track hoisted symbols
                self.hoisted_syms.update(zip(for_sym, [(i, (place, ofs, inv_for)) for i in expr]))

                # 4) Replace invariant sub-trees with the proper tmp variable
                replace_const(s.children[1], dict(zip([str(i) for i in expr], for_sym)))

                # 5) Record invariant loops which have been hoisted out of
                # the present loop nest
                if not pre_loop:
                    ext_loops.append(inv_for)

        return ext_loops

    def op_tiling(self, tile_sz=None):
        """Perform tiling at the register level for this nest.
        This function slices the iteration space, and relies on the backend
        compiler for unrolling and vector-promoting the tiled loops.
        By default, it slices the inner outer-product loop."""

        if tile_sz == -1:
            tile_sz = 20  # Actually, should be determined for each form

        for stmt, stmt_info in self.out_prods.items():
            # First, find outer product loops in the nest
            loops = self.op_loops[stmt]

            # Build tiled loops
            tiled_loops = []
            n_loops = loops[1].cond.children[1].symbol / tile_sz
            rem_loop_sz = loops[1].cond.children[1].symbol
            init = 0
            for i in range(n_loops):
                loop = dcopy(loops[1])
                loop.init.init = Symbol(init, ())
                loop.cond.children[1] = Symbol(tile_sz * (i + 1), ())
                init += tile_sz
                tiled_loops.append(loop)

            # Build remainder loop
            if rem_loop_sz > 0:
                init = tile_sz * n_loops
                loop = dcopy(loops[1])
                loop.init.init = Symbol(init, ())
                loop.cond.children[1] = Symbol(rem_loop_sz, ())
                tiled_loops.append(loop)

            # Append tiled loops at the right point in the nest
            par_block = loops[0].children[0]
            pb = par_block.children
            idx = pb.index(loops[1])
            par_block.children = pb[:idx] + tiled_loops + pb[idx + 1:]

    def op_split(self, cut=1, length=0):
        """Split assembly to improve resources utilization (e.g. vector
        registers). The splitting ``cuts'' the expressions into length
        blocks of #cut outer products.

        For example:
        for i
          for j
            A[i][j] += X[i]*Y[j] + Z[i]*K[j] + B[i]*X[j]
        with cut=1, length=1 this would be transformed into:
        for i
          for j
            A[i][j] += X[i]*Y[j]
        for i
          for j
            A[i][j] += Z[i]*K[j] + B[i]*X[j]

        If length is 0, then cut is ignored, and the expression is fully cut
        into chunks containing a single outer product.
        """

        def check_sum(par_node):
            """Return true if there are no sums in the sub-tree rooted in
            par_node, false otherwise."""
            if isinstance(par_node, Symbol):
                return False
            elif isinstance(par_node, Sum):
                return True
            elif isinstance(par_node, Par):
                return check_sum(par_node.children[0])
            elif isinstance(par_node, Prod):
                left = check_sum(par_node.children[0])
                right = check_sum(par_node.children[1])
                return left or right
            else:
                raise RuntimeError("Checking whether a node contains sums, but \
                        found an unknown node %s:" % str(par_node))

        def split_sum(node, parent, is_left, found, sum_count):
            """Exploit sum's associativity to cut node when a sum is found."""
            if isinstance(node, Symbol):
                return False
            elif isinstance(node, Par):
                return split_sum(node.children[0], (node, 0), is_left, found, sum_count)
            elif isinstance(node, Prod) and found:
                return False
            elif isinstance(node, Prod) and not found:
                if not split_sum(node.children[0], (node, 0), is_left, found, sum_count):
                    return split_sum(node.children[1], (node, 1), is_left, found, sum_count)
                return True
            elif isinstance(node, Sum):
                sum_count += 1
                if not found:
                    # Track the first Sum we found while cutting
                    found = parent
                if sum_count == cut:
                    # Perform the cut
                    if is_left:
                        parent, parent_leaf = parent
                        parent.children[parent_leaf] = node.children[0]
                    else:
                        found, found_leaf = found
                        found.children[found_leaf] = node.children[1]
                    return True
                else:
                    if not split_sum(node.children[0], (node, 0), is_left, found, sum_count):
                        return split_sum(node.children[1], (node, 1), is_left, found, sum_count)
                    return True
            else:
                raise RuntimeError("Splitting expression, but actually found an unknown \
                                    node: %s" % node.gencode())

        def split_and_update(out_prods):
            op_split, op_splittable = ({}, {})
            for stmt, stmt_info in out_prods.items():
                it_vars, parent, loops = stmt_info
                stmt_left = dcopy(stmt)
                stmt_right = dcopy(stmt)
                expr_left = Par(stmt_left.children[1])
                expr_right = Par(stmt_right.children[1])
                sleft = split_sum(expr_left.children[0], (expr_left, 0), True, None, 0)
                sright = split_sum(expr_right.children[0], (expr_right, 0), False, None, 0)

                if sleft and sright:
                    # Append the left-split expression. Re-use loop nest
                    parent.children[parent.children.index(stmt)] = stmt_left
                    # Append the right-split (reminder) expression. Create new loop nest
                    split_loop = dcopy([f for f in self.fors if f.it_var() == it_vars[0]][0])
                    split_inner_loop = split_loop.children[0].children[0].children[0]
                    split_inner_loop.children[0] = stmt_right
                    self.loop_nest.children[0].children.append(split_loop)
                    stmt_right_loops = [split_loop, split_loop.children[0].children[0]]
                    # Update outer product dictionaries
                    op_splittable[stmt_right] = (it_vars, split_inner_loop, stmt_right_loops)
                    if check_sum(stmt_left.children[1]):
                        op_splittable[stmt_left] = (it_vars, parent, loops)
                    else:
                        op_split[stmt_left] = (it_vars, parent, loops)
            return op_split, op_splittable

        if not self.out_prods:
            return
        new_out_prods = {}
        splittable = self.out_prods

        if length:
            # Split into at most length blocks
            for i in range(length-1):
                split, splittable = split_and_update(splittable)
                new_out_prods.update(split)
                if not splittable:
                    break
            if splittable:
                new_out_prods.update(splittable)
        else:
            # Split everything into blocks of length 1
            while splittable:
                split, splittable = split_and_update(splittable)
                new_out_prods.update(split)
        self.out_prods = new_out_prods

    def op_expand(self):
        """Expand outer product expressions such that:

        Y[j] = f(...)
        (X[i]*Y[j])*F + ...

        becomes:

        Y[j] = f(...)*F
        (X[i]*Y[j]) + ...

        This might be useful for various purposes:
        - Relieve register pressure; when, for example, (X[i]*Y[j]) is computed
        in a loop L' different than the loop L'' in which Y[j] is evaluated,
        and the cost(L') > cost(L'')
        - To "clean" outer products: this might better expose well-known linear
        algebra operations, like matrix-matrix multiplications. """

        CONST = -1
        ITVAR = -2
        updated_syms = defaultdict(list)

        def find_expandable(node, parent, it_vars):
            if isinstance(node, Symbol):
                if not node.rank:
                    return ([node], CONST)
                elif node.rank[-1] not in it_vars:
                    return ([node], CONST)
                else:
                    return ([node], ITVAR)
            elif isinstance(node, Par):
                return find_expandable(node.children[0], node, it_vars)
            elif isinstance(node, Prod):
                left_node, left_type = find_expandable(node.children[0], node, it_vars)
                right_node, right_type = find_expandable(node.children[1], node, it_vars)
                if left_type == ITVAR and right_type == ITVAR:
                    # Found an expandable product
                    return (left_node, ITVAR)
                elif left_type == CONST and right_type == CONST:
                    # Product of constants; they are both used for expansion (if any)
                    return ([node], CONST)
                else:
                    # Do the expansion
                    const = left_node[0] if left_type == CONST else right_node[0]
                    expandable, exp_node = (left_node, node.children[0]) if left_type == ITVAR \
                        else (right_node, node.children[1])
                    for sym in expandable:
                        if not (updated_syms.get(sym) and const in updated_syms[sym]):
                            # Perform the expansion; we do the expansion for a symbol only
                            # if we hadn't done it before (that's the purpose of updated_syms)
                            if sym not in self.hoisted_syms:
                                raise RuntimeError("Expanding expression, but found one outer\
                                    product symbol which was not hoisted: %s" % sym.gencode())
                            old_expr, for_loop = self.hoisted_syms[sym]
                            new_node = Prod(Par(old_expr.children[0]), const)
                            old_expr.children[0] = new_node
                            updated_syms[sym].append(const)
                    # Update the parent node, since an expression has been expanded
                    if parent.children[0] == node:
                        parent.children[0] = exp_node
                    elif parent.children[1] == node:
                        parent.children[1] = exp_node
                    else:
                        raise RuntimeError("Expanding expression, but can't find the\
                                relationship between parent and child node")
                    return (expandable, ITVAR)
            elif isinstance(node, Sum):
                left_node, left_type = find_expandable(node.children[0], node, it_vars)
                right_node, right_type = find_expandable(node.children[1], node, it_vars)
                if left_type == ITVAR and right_type == ITVAR:
                    return (left_node + right_node, ITVAR)
                elif left_type == CONST and right_type == CONST:
                    return ([node], CONST)
                else:
                    return (None, CONST)
            else:
                raise RuntimeError("Finding expandable expression, but actually \
                                    found an unknown node: %s" % node.gencode())

        for stmt, stmt_info in self.out_prods.items():
            it_vars, parent, loops = stmt_info
            find_expandable(stmt.children[1], stmt, it_vars)

    def assembly_precompute(self):
        """In an assembly loop nest, only the local tensor is updated every
        iteration. All other written to variables, are actually const, since
        they are written only once.

        This method exploits this property to precompute all written variables
        (but the local tensor) out of the loop nest. This has di effect of
        making the loop nest perfect.

        For example:
        for i
          double A[...] = ...
          for r
            A[r] += f(...)
          for j
            // A is only read here
            ...

        becomes
        for i
          for r
            A[i][r] += f(...)
        for i
          for j
            ...
        """

        def precomputing_error(node):
            raise RuntimeError("Precomputing assembly expressions; found \
                    an unexpteced AST node while precomputing: %s" % str(node))

        # There must be at least three loops for precomputation to be meaningful
        if self.loop_nest in [l for l, b in self.itspace]:
            return

        outer_loop = self.fors[0].children[0]
        outer_var = self.fors[0].it_var()
        outer_bound = self.fors[0].size()
        op_outer_loop = self.itspace[0][0]

        # Precomputation
        new_outer_block = []
        expanded_syms = {}
        for i in outer_loop.children:
            if i == op_outer_loop:
                break
            elif isinstance(i, Decl):
                # Vector expand the declaration of the precomputed symbol
                i.sym.rank = (outer_bound,) + i.sym.rank
                if isinstance(i.init, Symbol):
                    i.init.symbol = "{%s}" % i.init.symbol
                new_outer_block.append(i)
            elif isinstance(i, For):
                # Vector expand every assignment
                nested_stmts = i.children[0].children
                precompute_for = dcopy(self.fors[0])
                precompute_for.children[0].children = [i]
                new_outer_block.append(precompute_for)
                for j in nested_stmts:
                    if isinstance(j, (Assign, Incr)):
                        symbol = j.children[0]
                        new_rank = (outer_var,) + symbol.rank
                        expanded_syms[str(symbol)] = new_rank
                        symbol.rank = new_rank
                    else:
                        precomputing_error(j)
            else:
                precomputing_error(i)

        # Delete precomputed stmts from original loop nest
        outer_loop.children = [op_outer_loop]

        # Update the AST adding the newly precomputed blocks
        root = self.pre_header.children
        ofs = root.index(self.fors[0])
        self.pre_header.children = root[:ofs] + new_outer_block + root[ofs:]

        # Update the AST by vector-expanding the accessed variables that have
        # been pre-computed
        for s in self.sym:
            if str(s) in expanded_syms:
                s.rank = expanded_syms[str(s)]

    def turn_into_blas_dgemm(self, blas, kernel_decls):
        """Find any perfect loop nest whose computation is actually a matrix-matrix
        multiply (MMM), and transform it into a call to BLAS dgemm. Involved matrix'
        layout is modified accordingly.

        ``blas'' indicates which BLAS implementation should be used. Supported values
        are, currently: {mkl}.
        """

        def update_symbols(node, parent, syms_to_change, ofs_sizes, to_transpose):
            """Change the storage layout of symbols involved in MMMs."""
            if isinstance(node, Symbol) and node.symbol in syms_to_change:
                if isinstance(parent, Decl):
                    node.rank = (int(node.rank[0])*int(node.rank[1]),)
                else:
                    if node.symbol in to_transpose:
                        node.offset = ((ofs_sizes[0], node.rank[0]),)
                        node.rank = (node.rank[-1],)
                    else:
                        node.offset = ((ofs_sizes[2], node.rank[-1]),)
                        node.rank = (node.rank[0],)
            elif isinstance(node, (Par, For)):
                update_symbols(node.children[0], node, syms_to_change, ofs_sizes, to_transpose)
            elif isinstance(node, Decl):
                update_symbols(node.sym, node, syms_to_change, ofs_sizes, to_transpose)
            elif isinstance(node, (Assign, Incr)):
                update_symbols(node.children[0], node, syms_to_change, ofs_sizes, to_transpose)
                update_symbols(node.children[1], node, syms_to_change, ofs_sizes, to_transpose)
            elif isinstance(node, (Root, Block)):
                for n in node.children:
                    update_symbols(n, node, syms_to_change, ofs_sizes, to_transpose)
            else:
                pass

        def check_prod(node):
            """Return (e1, e2) if the node is a product between two symbols s1
            and s2, () otherwise.
            For example:
            - Par(Par(Prod(s1, s2))) -> (s1, s2)
            - Prod(s1, s2) -> (s1, s2)
            - Sum -> ()
            - Prod(Sum, s1) -> ()"""
            if isinstance(node, Par):
                return check_prod(node.children[0])
            elif isinstance(node, Prod):
                left, right = (node.children[0], node.children[1])
                if isinstance(left, Expr) and isinstance(right, Expr):
                    return (left, right)
                return ()
            return ()

        # There must be at least three loops to extract a MMM
        if self.loop_nest in [l for l, b in self.itspace]:
            return

        outer_loop = self.loop_nest
        ofs = self.pre_header.children.index(outer_loop)
        found_mmm = False

        # 1) Split potential MMM into different perfect loop nests
        to_remove, to_transpose = ([], [])
        to_transform = {}
        for middle_loop in outer_loop.children[0].children:
            if not isinstance(middle_loop, For):
                continue
            found = False
            inner_loop = middle_loop.children[0].children
            if not (len(inner_loop) == 1 and isinstance(inner_loop[0], For)):
                continue
            # Found a perfect loop nest, now check body operation
            body = inner_loop[0].children[0].children
            if not (len(body) == 1 and isinstance(body[0], Incr)):
                continue
            # The body is actually as a single statement, as expected
            lhs = body[0].children[0].rank
            rhs = check_prod(body[0].children[1])
            if not rhs:
                continue
            # Check memory access pattern
            it_var = inner_loop[0].it_var()
            rhs_l, rhs_r = (rhs[0].rank, rhs[1].rank)
            if lhs[0] == rhs_l[0] and lhs[1] == rhs_r[1] and rhs_l[1] == rhs_r[0] or \
                    lhs[0] == rhs_r[1] and lhs[1] == rhs_r[0] and rhs_l[1] == rhs_r[0]:
                found = True
            elif lhs[0] == rhs_l[1] and lhs[1] == rhs_r[1] and rhs_l[0] == rhs_r[0] or \
                    lhs[0] == rhs_r[1] and lhs[1] == rhs_l[1] and rhs_l[0] == rhs_r[0] or \
                    lhs[0] == rhs_l[0] and lhs[1] == rhs_r[0] and rhs_l[1] == rhs_r[1] or \
                    lhs[0] == rhs_r[0] and lhs[1] == rhs_l[0] and rhs_l[1] == rhs_r[1]:
                found = True
                to_transpose.append(rhs[0].symbol if rhs_l[1] == it_var else rhs[1].symbol)
            if found:
                new_outer = dcopy(outer_loop)
                new_outer.children[0].children = [middle_loop]
                to_remove.append(middle_loop)
                self.pre_header.children.insert(ofs, new_outer)
                loop_sizes = (outer_loop.size(), middle_loop.size(), inner_loop[0].size())
                to_transform[new_outer] = (body[0].children[0], rhs, loop_sizes)
                found_mmm = True
        # Clean up
        for l in to_remove:
            outer_loop.children[0].children.remove(l)
        if not outer_loop.children[0].children:
            self.pre_header.children.remove(outer_loop)

        # 2) Delegate to BLAS
        change_layout = []
        for l, mmm in to_transform.items():
            lhs, rhs, loop_sizes = mmm
            blas_interface = ast_plan.blas_interface
            dgemm = blas_interface['dgemm'] % \
                {'ver': blas_interface['prefix'],
                 'notrans': blas_interface['no_trans'],
                 'order': blas_interface['row_major'],
                 'm1size': loop_sizes[2],
                 'm2size': loop_sizes[1],
                 'm3size': loop_sizes[0],
                 'm1': rhs[0].symbol,
                 'm2': rhs[1].symbol,
                 'm3': lhs.symbol}
            self.pre_header.children[self.pre_header.children.index(l)] = FlatBlock(dgemm)
            to_change = [rhs[0].symbol, rhs[1].symbol, lhs.symbol]
            change_layout.extend([s for s in to_change if s not in change_layout])
        # Change the storage layout of involved matrices
        if change_layout:
            update_symbols(self.pre_header, None, change_layout, loop_sizes, to_transpose)
            update_symbols(kernel_decls[lhs.symbol][0], None, change_layout,
                           loop_sizes, to_transpose)

        return found_mmm
