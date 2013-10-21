from math import ceil

from pyop2.ir.ast_base import *


class LoopVectoriser(object):

    """ Loop vectorizer
        * Vectorization:
          Outer-product vectorisation.
        * Memory:
          padding, data alignment, trip count/bound adjustment
          padding and data alignment are for aligned unit-stride load
          trip count/bound adjustment is for auto-vectorisation.
        * Unroll_and_jam:
          unroll and jam outer loops if beneficial for vectorisation. """

    def __init__(self, kernel_ast, loop_optimiser, isa):
        self.lo = loop_optimiser
        self.ast = kernel_ast
        self.intr = self._set_isa(isa)

    # Memory optimisations #

    def pad_and_align(self):
        """Pad each data structure accessed in the loop nest to a multiple
        of the vector length. """

        # Get the declarations of the symbols accessed in the loop nest
        acc_decl = [d for s, d in decl.items() if s in self.lo.symbols]

        # Do the rounding of the inner dimension (padding)
        for d in acc_decl:
            rounded = self._roundup(d.sym.rank[-1])
            d.sym.rank = d.sym.rank[:-1] + (rounded,)

    def adjust_loop(self, only_bound=True):
        """Adjust trip count and bound of each innermost loop in the nest."""

        def inner_loops(node, loops):
            """Find the inner loops in the tree rooted in node."""
            if perf_stmt(node):
                return False
            elif isinstance(node, For):
                found = inner_loops(node.children[0], loops)
                if not found:
                    loops.append(node)
                return True
            elif isinstance(node, Block):
                return any([inner_loops(s, loops) for s in node.children])

        i_loops = []
        inner_loops(self.lo.loop_nest, i_loops)

        if not only_bound:
            # Also adjust the loop increment factor
            print ""

        # Adjust the loops bound

    # Vectorisation
    def outer_product(self):
        pass

    def _set_isa(self, isa):
        """Set the proper intrinsics instruction set for the vectorizer """

        if isa == "AVX":
            return {
                "inst_set": "AVX",
                "avail_reg": 16,
                "double_words": 4,
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

    # Utilities

    def _roundup(self, x):
        """Return x rounded up to the vector length. """

        word_len = self.intr["double_words"]
        return int(ceil(x / float(word_len))) * word_len
