from math import ceil

from pyop2.ir.ast_base import *


class LoopVectoriser(object):

    """ Loop vectorizer
        * Vectorization:
          Outer-product vectorisation.
        * Memory:
          padding, data alignment, trip count/bound adjustment
          padding and data alignment are for aligned unit-stride load
          trip count/bound adjustment is for auto-vectorisation. """

    def __init__(self, kernel_ast, loop_optimiser, isa, compiler):
        self.lo = loop_optimiser
        self.ast = kernel_ast
        self.intr = self._set_isa(isa)
        self.comp = self._set_compiler(compiler)
        self.i_loops = []
        self._inner_loops(self.lo.loop_nest, self.i_loops)

    # Memory optimisations #

    def pad_and_align(self, decl):
        """Pad each data structure accessed in the loop nest to a multiple
        of the vector length. """

        # Get the declarations of the symbols accessed in the loop nest
        acc_decl = [d for s, d in decl.items() if s in self.lo.sym]

        # Do the rounding of the inner dimension (padding)
        for d in acc_decl:
            rounded = self._roundup(d.sym.rank[-1])
            d.sym.rank = d.sym.rank[:-1] + (rounded,)

    def adjust_loop(self, only_bound):
        """Adjust trip count and bound of each innermost loop in the nest."""

        for l in self.i_loops:
            # Also adjust the loop increment factor
            if not only_bound:
                l.incr.children[1] = c_sym(self.intr["dp_reg"])

            # Adjust the loops bound
            bound = l.cond.children[1]
            l.cond.children[1] = c_sym(self._roundup(bound.symbol))

    def set_alignment(self, decl, autovectorisation=False):
        """Align arrays in the kernel to the size of the vector length in
        order to issue aligned loads and stores. Also tell this information to
        the back-end compiler by adding suitable pragmas over loops in case
        we rely on autovectorisation. """

        for d in decl.values():
            d.attr.append(self.comp["align"](self.intr["alignment"]))

        if autovectorisation:
            for l in self.i_loops:
                l.pragma = self.comp["decl_aligned_for"]

    # Vectorisation
    def outer_product(self):

        class Alloc(object):

            """Handle allocation of register variables. """

            def __init__(self, intr, factor):
                # TODO Use factor when unrolling...
                n_res = intr["dp_reg"]
                self.n_tot = intr["avail_reg"]
                self.res = [intr["reg"](v) for v in range(n_res)]
                self.var = [intr["reg"](v) for v in range(n_res, self.n_tot)]

            def get_reg(self):
                if len(self.var) == 0:
                    l = self.n_tot * 2
                    self.var += [intr["reg"](v) for v in range(self.n_tot, l)]
                    self.n_tot = l
                return self.var.pop(0)

            def free_reg(self, reg):
                self.var.insert(0, reg)

            def get_tensor(self):
                return self.res

        def swap_reg(step, reg):
            """Swap values in a vector register. """

            if step == 0:
                return []
            elif step in [1, 3]:
                return [self.intr["l_perm"](r, "5") for r in reg.values()]
            elif step == 2:
                return [self.intr["g_perm"](r, r, "1") for r in reg.values()]

        def vect_mem(node, regs, intr, loops, decls=[], in_vrs={}, out_vrs={}):
            """Return a list of vector variables declarations representing
            loads, sets, broadcasts. Also return dicts of allocated inner
            and outer variables. """

            if isinstance(node, Symbol):
                # Check if this symbol's values are iterated over with the
                # fastest varying dimension of the loop nest
                reg = regs.get_reg()
                if node.rank and loops[1] == node.rank[-1]:
                    in_vrs[node] = Symbol(reg, ())
                else:
                    out_vrs[node] = Symbol(reg, ())
                expr = intr["symbol"](node.symbol, node.rank)
                decls.append(Decl(intr["decl_var"], Symbol(reg, ()), expr))
            elif isinstance(node, Par):
                child = node.children[0]
                vect_mem(child, regs, intr, loops, decls, in_vrs, out_vrs)
            else:
                left = node.children[0]
                right = node.children[1]
                vect_mem(left, regs, intr, loops, decls, in_vrs, out_vrs)
                vect_mem(right, regs, intr, loops, decls, in_vrs, out_vrs)

            return (decls, in_vrs, out_vrs)

        def vect_expr(node, in_vrs, out_vrs):
            """Turn a scalar expression into its intrinsics equivalent. """

            if isinstance(node, Symbol):
                return in_vrs.get(node) or out_vrs.get(node)
            elif isinstance(node, Par):
                return vect_expr(node.children[0], in_vrs, out_vrs)
            else:
                left = vect_expr(node.children[0], in_vrs, out_vrs)
                right = vect_expr(node.children[1], in_vrs, out_vrs)
                if isinstance(node, Sum):
                    return self.intr["add"](left, right)
                elif isinstance(node, Sub):
                    return self.intr["sub"](left, right)
                elif isinstance(node, Prod):
                    return self.intr["mul"](left, right)
                elif isinstance(node, Div):
                    return self.intr["div"](left, right)

        def incr_tensor(tensor, ofs, out_reg, mode):
            """Add the right hand side contained in out_reg to tensor."""
            if mode == 0:
                # Store in memory
                loc = (tensor.rank[0] + "+" + str(ofs), tensor.rank[1])
                return self.intr["store"](Symbol(tensor.symbol, loc), out_reg)

        def restore_layout(regs, tensor, mode):
            print "Restore layout"
            return []

        # TODO: need to determine order of loops w.r.t. the local tensor
        # entries. E.g. if j-k inner loops and A[j][k], then increments of
        # A are performed within the k loop. On the other hand, if ip is
        # the innermost loop, stores in memory are done outside of ip
        mode = 0  # 0 == Stores, 1 == Local incrs
        loops = (self.lo.fors[-2].it_var(), self.lo.fors[-1].it_var())

        for stmt in self.lo.block.children:  # FIXME: need find outer prods
            tensor = stmt.children[0]
            expr = stmt.children[1]

            # Get source-level variables
            regs = Alloc(self.intr, 1)  # TODO: set appropriate factor

            # Find required loads
            decls, in_vrs, out_vrs = vect_mem(expr, regs, self.intr, loops)
            stmt = []
            for i in range(self.intr["dp_reg"]):
                # Register shuffles, vectorisation of a row, update tensor
                stmt.extend(swap_reg(i, in_vrs))
                intr_expr = vect_expr(expr, in_vrs, out_vrs)
                stmt.append(incr_tensor(tensor, i, intr_expr, mode))
            # Restore the tensor layout
            stmt.extend(restore_layout(regs, tensor, mode))

        # Create the vectorized for body
        new_block = []
        for d in decls + stmt:
            new_block.append(d)
        self.lo.block.children = new_block

    # Utilities
    def _inner_loops(self, node, loops):
        """Find the inner loops in the tree rooted in node."""
        if perf_stmt(node):
            return False
        elif isinstance(node, Block):
            return any([self._inner_loops(s, loops) for s in node.children])
        elif isinstance(node, For):
            found = self._inner_loops(node.children[0], loops)
            if not found:
                loops.append(node)
            return True

    def _set_isa(self, isa):
        """Set the proper intrinsics instruction set. """

        if isa == "AVX":
            return {
                "inst_set": "AVX",
                "avail_reg": 16,
                "alignment": 32,
                "dp_reg": 4,  # Number of double values per register
                "reg": lambda n: "ymm%s" % n,
                "zeroall": "_mm256_zeroall ()",
                "setzero": "_mm256_setzero_pd ()",
                "decl_var": "__m256d",
                "align_array": lambda p: "__attribute__((aligned(%s)))" % p,
                "symbol": lambda s, r: AVXLoad(s, r),
                "store": lambda m, r: AVXStore(m, r),
                "mul": lambda r1, r2: AVXProd(r1, r2),
                "div": lambda r1, r2: AVXDiv(r1, r2),
                "add": lambda r1, r2: AVXSum(r1, r2),
                "sub": lambda r1, r2: AVXSub(r1, r2),
                "l_perm": lambda r, f: AVXLocalPermute(r, f),
                "g_perm": lambda r1, r2, f: AVXGlobalPermute(r1, r2, f),
                "unpck_hi": lambda r1, r2: AVXUnpackHi(r1, r2),
                "unpck_lo": lambda r1, r2: AVXUnpackLo(r1, r2),
            }

    def _set_compiler(self, compiler):
        """Set compiler-specific keywords. """

        if compiler == "INTEL":
            return {
                "align": lambda o: "__attribute__((aligned(%s)))" % o,
                "decl_aligned_for": "#pragma vector aligned"
            }

    def _roundup(self, x):
        """Return x rounded up to the vector length. """
        word_len = self.intr["dp_reg"]
        return int(ceil(x / float(word_len))) * word_len
