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
                # TODO Use factor
                nreg = (intr["avail_reg"] - intr["dp_reg"])
                self.res = [intr["reg"](v) for v in range(intr["dp_reg"])]
                self.var = [intr["reg"](v)
                            for v in range(intr["dp_reg"], nreg)]

            def get_reg(self):
                return self.var.pop(0)

            def free_reg(self, reg):
                self.var.insert(0, reg)

            def get_tensor(self):
                return self.res

        def swap_reg(reg, intr):
            # Swap values in a vector register
            print "Swap"

        def scan_tree(regs, node):
            pass

        def incr_tensor(tensor, out_regs, mode):
            print "Incr tensor"

        def restore_layout(regs, tensor, mode):
            print "Restore layout"

        # TODO: need to determine order of loops w.r.t. the local tensor
        # entries. E.g. if j-k inner loops and A[j][k], then increments of
        # A are performed within the k loop. On the other hand, if ip is
        # the innermost loop, stores in memory are done outside of ip
        # mode = 0  # 0 == Stores, 1 == Local incrs
        """
        for stmt in self.lo.block.children: #FIXME: need find outer prods
            tensor = stmt.children[0]
            expr = stmt.children[1]

            # Get source-level variables
            regs = Alloc(self.intr, 1)  # TODO: set appropriate factor

            inner_var = []
            for i in range(intr["dp_reg"]):
                # Register shuffles, vectorisation of a row, update tensor
                swap_reg(inner_var, self.intr)
                out_reg = scan_tree(regs, expr)
                incr_tensor(tensor, out_reg, mode)
            # Restore the tensor layout
            restore_layout(regs, tensor, mode)"""

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
                "decl_var": lambda n: "__m256d %s" % n,
                "align_array": lambda pad: "__attribute__((aligned(%s)))" % pad,
                "load": lambda m: "_mm256_load_pd (&%s)" % m,
                "store": lambda m, r: "_mm256_store_pd (%s, %s);" % (m, r),
                "mul": lambda r1, r2: "_mm256_mul_pd (%s, %s)" % (r1, r2),
                "div": lambda r1, r2: "_mm256_div_pd (%s, %s)" % (r1, r2),
                "broadcast": lambda m: "_mm256_broadcast_sd (&%s)" % m,
                "add": lambda r1, r2: "_mm256_add_pd (%s, %s)" % (r1, r2),
                "sub": lambda r1, r2: "_mm256_sub_pd (%s, %s)" % (r1, r2),
                "set": lambda i: "_mm256_set1_pd (%s)" % i,
                "l_perm": lambda r, f: "_mm256_permute_pd (%s, %s)" % (r, f),
                "g_perm": lambda r1, r2, f: "_mm256_permute2f128_pd (%s, %s, %s)" % (r1, r2, f),
                "unpck_hi": lambda r1, r2: "_mm256_unpackhi_pd (%s, %s)" % (r1, r2),
                "unpck_lo": lambda r1, r2: "_mm256_unpacklo_pd (%s, %s)" % (r1, r2),
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
